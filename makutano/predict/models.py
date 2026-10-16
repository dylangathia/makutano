"""Arrival forecasters, all with the same interface: fit(table[, sequences]) and
predict(table[, sequences]) -> DataFrame with one column per target.

Persistence and HistoricalAverage are the baselines the learned models must
beat. GradientBoosting needs scikit-learn; GRUForecaster needs PyTorch.
"""
import pickle

import numpy as np
import pandas as pd

from .features import FEATURES, HORIZONS_MIN, TARGETS


class Persistence:
    """The next h minutes will bring as many arrivals as the last h minutes did."""
    name = "persistence"

    def fit(self, table, sequences=None):
        return self

    def predict(self, table, sequences=None):
        return pd.DataFrame({f"y_next_{h}m": table[f"arr_last_{h}m"].to_numpy() for h in HORIZONS_MIN},
                            index=table.index)


class HistoricalAverage:
    """The average seen at this approach and time of day on the training days."""
    name = "historical_average"

    def __init__(self, bin_min=15):
        self.bin_min = bin_min

    def _keys(self, table):
        return table["approach_code"].astype(int), (table["minute_of_day"] // self.bin_min).astype(int)

    def fit(self, table, sequences=None):
        a, b = self._keys(table)
        frame = table[TARGETS].assign(a=a.to_numpy(), b=b.to_numpy())
        self.by_bin = frame.groupby(["a", "b"])[TARGETS].mean()
        self.by_approach = frame.groupby("a")[TARGETS].mean()
        self.overall = frame[TARGETS].mean()
        return self

    def predict(self, table, sequences=None):
        a, b = self._keys(table)
        idx = pd.MultiIndex.from_arrays([a, b])
        out = self.by_bin.reindex(idx).set_axis(table.index)
        # Times or approaches not seen in training fall back to coarser averages.
        out = out.fillna(self.by_approach.reindex(a).set_axis(table.index)).fillna(self.overall)
        return out[TARGETS]


class GradientBoosting:
    """One gradient-boosted tree model per horizon, on the tabular features."""
    name = "gradient_boosting"

    def __init__(self, **params):
        self.params = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31,
                       "early_stopping": True, "random_state": 0, **params}

    def fit(self, table, sequences=None):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = table[FEATURES].to_numpy()
        self.models = {}
        for target in TARGETS:
            m = HistGradientBoostingRegressor(loss="poisson", categorical_features=[0], **self.params)
            self.models[target] = m.fit(X, table[target].to_numpy())
        return self

    def predict(self, table, sequences=None):
        X = table[FEATURES].to_numpy()
        return pd.DataFrame({t: m.predict(X) for t, m in self.models.items()}, index=table.index)


class GRUForecaster:
    """A small GRU over the last few minutes of arrivals and queue, joined with
    the time-of-day features, predicting all horizons at once."""
    name = "gru"
    STATIC = ["tod_sin", "tod_cos", "approach_code", "weekday"]

    def __init__(self, hidden=32, epochs=60, lr=1e-3, batch=256, patience=8, seed=0):
        self.hidden, self.epochs, self.lr, self.batch, self.patience, self.seed = hidden, epochs, lr, batch, patience, seed

    def _tensors(self, table, sequences):
        import torch
        seq = (sequences - self.seq_mean) / self.seq_std
        static = (table[self.STATIC].to_numpy(float) - self.static_mean) / self.static_std
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(static, dtype=torch.float32)

    def _build(self):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self, hidden, n_static, n_out):
                super().__init__()
                self.gru = nn.GRU(2, hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

            def forward(self, seq, static):
                _, h = self.gru(seq)
                # Softplus keeps predicted counts non-negative.
                return nn.functional.softplus(self.head(torch.cat([h[-1], static], dim=1)))

        torch.manual_seed(self.seed)
        return Net(self.hidden, len(self.STATIC), len(TARGETS))

    def fit(self, table, sequences, val_table=None, val_sequences=None):
        import torch
        if sequences is None:
            raise ValueError("GRUForecaster needs sequences: build_examples(..., with_sequences=True)")
        self.seq_mean = sequences.reshape(-1, 2).mean(0)
        self.seq_std = sequences.reshape(-1, 2).std(0) + 1e-6
        static = table[self.STATIC].to_numpy(float)
        self.static_mean, self.static_std = static.mean(0), static.std(0) + 1e-6
        self.net = self._build()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        seq, st = self._tensors(table, sequences)
        y = torch.tensor(table[TARGETS].to_numpy(float), dtype=torch.float32)
        val = None
        if val_table is not None and len(val_table):
            vs, vst = self._tensors(val_table, val_sequences)
            val = (vs, vst, torch.tensor(val_table[TARGETS].to_numpy(float), dtype=torch.float32))
        best, best_state, stale = np.inf, None, 0
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            self.net.train()
            for b in torch.randperm(len(y), generator=gen).split(self.batch):
                opt.zero_grad()
                loss = torch.nn.functional.poisson_nll_loss(self.net(seq[b], st[b]), y[b], log_input=False)
                loss.backward()
                opt.step()
            if val is None:
                continue
            self.net.eval()
            with torch.no_grad():
                v = torch.nn.functional.poisson_nll_loss(self.net(val[0], val[1]), val[2], log_input=False).item()
            if v < best - 1e-4:
                best, stale = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # The network class is built inside _build so torch stays an optional
    # dependency; pickle the weights and rebuild the network on load.
    def __getstate__(self):
        state = self.__dict__.copy()
        net = state.pop("net", None)
        if net is not None:
            state["net_state"] = net.state_dict()
        return state

    def __setstate__(self, state):
        net_state = state.pop("net_state", None)
        self.__dict__.update(state)
        if net_state is not None:
            self.net = self._build()
            self.net.load_state_dict(net_state)

    def predict(self, table, sequences):
        import torch
        self.net.eval()
        with torch.no_grad():
            out = self.net(*self._tensors(table, sequences)).numpy()
        return pd.DataFrame(out, columns=TARGETS, index=table.index)


def save(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
