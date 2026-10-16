"""Forecasting examples built from the series written by makutano.see.measure.

Each example is one approach at one moment. Features describe the recent past
(arrivals over the last 1, 5 and 15 minutes, the queue now and over the last
minute) and the time of day; targets are the arrivals over the next 1, 5 and 15
minutes. The same function builds the single live example the controller needs
inside the simulation, so training and use can never drift apart.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS_MIN = (1, 5, 15)
SEQUENCE_MIN = 5          # history the recurrent model sees, in minutes
FEATURES = ["approach_code", "minute_of_day", "tod_sin", "tod_cos", "weekday",
            "arr_last_1m", "arr_last_5m", "arr_last_15m", "queue_now", "queue_mean_1m"]
TARGETS = [f"y_next_{h}m" for h in HORIZONS_MIN]


def load_series(paths):
    """Concatenate series.csv files; each file is one recording session."""
    frames = []
    for p in map(Path, paths):
        s = pd.read_csv(p)
        if "timestamp" not in s:
            raise ValueError(f"{p} has no timestamp column: rerun makutano.see.measure with --start")
        s["timestamp"] = pd.to_datetime(s["timestamp"])
        # series.csv usually sits in a folder named after the session.
        s["session"] = p.parent.name if p.stem == "series" else p.stem
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def _time_features(ts):
    ts = pd.DatetimeIndex(ts)
    minute = ts.hour * 60 + ts.minute + ts.second / 60
    angle = 2 * np.pi * minute / 1440
    return {"minute_of_day": np.asarray(minute, float), "tod_sin": np.sin(angle),
            "tod_cos": np.cos(angle), "weekday": np.asarray(ts.weekday, float)}


def _window_features(arrivals, queue, idx, per_min):
    """Features for forecasts made just before step idx (uses steps < idx only)."""
    csum_a = np.concatenate([[0.0], np.cumsum(arrivals)])
    csum_q = np.concatenate([[0.0], np.cumsum(queue)])
    out = {f"arr_last_{h}m": csum_a[idx] - csum_a[idx - h * per_min] for h in HORIZONS_MIN}
    out["queue_now"] = queue[idx - 1]
    out["queue_mean_1m"] = (csum_q[idx] - csum_q[idx - per_min]) / per_min
    return out


def build_examples(series, approaches, step_s=5, with_sequences=False):
    """All complete examples in the series.

    Returns (table, sequences): the table holds features, targets and metadata;
    sequences, when requested, has shape (n, steps, 2) with recent arrivals and
    queue for the recurrent model, row-aligned with the table.
    """
    per_min = int(round(60 / step_s))
    history = max(HORIZONS_MIN) * per_min
    ahead = max(HORIZONS_MIN) * per_min
    seq_len = SEQUENCE_MIN * per_min
    codes = {a: i for i, a in enumerate(approaches)}
    tables, seqs = [], []
    for (session, approach), g in series.groupby(["session", "approach"], sort=False):
        g = g.sort_values("t_s")
        arrivals = g["arrivals"].to_numpy(float)
        queue = g["queue_m"].to_numpy(float)
        idx = np.arange(history, len(g) - ahead + 1)
        if len(idx) == 0:
            continue
        cols = _window_features(arrivals, queue, idx, per_min)
        csum = np.concatenate([[0.0], np.cumsum(arrivals)])
        for h in HORIZONS_MIN:
            cols[f"y_next_{h}m"] = csum[idx + h * per_min] - csum[idx]
        ts = g["timestamp"].to_numpy()[idx]
        cols.update(_time_features(ts))
        cols["approach_code"] = np.full(len(idx), codes[approach], float)
        t = pd.DataFrame(cols)
        t["approach"], t["session"], t["timestamp"] = approach, session, ts
        t["day"] = pd.DatetimeIndex(ts).date
        tables.append(t)
        if with_sequences:
            window = np.stack([np.stack([arrivals[i - seq_len:i], queue[i - seq_len:i]], axis=-1) for i in idx])
            seqs.append(window)
    if not tables:
        raise ValueError(f"no complete examples: each session needs at least {2 * max(HORIZONS_MIN)} minutes")
    table = pd.concat(tables, ignore_index=True)
    return table, (np.concatenate(seqs) if with_sequences else None)


def live_example(arrivals_hist, queue_hist, timestamp, approach, approaches, step_s=5):
    """One feature row from the most recent history (oldest first), for online use."""
    per_min = int(round(60 / step_s))
    need = max(HORIZONS_MIN) * per_min
    a = np.zeros(need)
    q = np.zeros(need)
    a_hist = np.asarray(arrivals_hist, float)[-need:]
    q_hist = np.asarray(queue_hist, float)[-need:]
    if len(a_hist):
        a[-len(a_hist):] = a_hist
        q[-len(q_hist):] = q_hist
    row = {k: v for k, v in _window_features(a, q, np.array([need]), per_min).items()}
    row.update(_time_features([timestamp]))
    row["approach_code"] = np.array([approaches.index(approach)], float)
    seq_len = SEQUENCE_MIN * per_min
    sequence = np.stack([a[-seq_len:], q[-seq_len:]], axis=-1)[None]
    return pd.DataFrame({k: np.atleast_1d(v) for k, v in row.items()}), sequence


def split_by_day(table, val_days=1, test_days=1):
    """Train/validation/test masks so that every model is tested on unseen days."""
    days = sorted(table["day"].unique())
    if len(days) < val_days + test_days + 1:
        raise ValueError(f"need at least {val_days + test_days + 1} recording days, found {len(days)}")
    test = set(days[-test_days:])
    val = set(days[-(test_days + val_days):-test_days])
    day = table["day"]
    return ~day.isin(test | val), day.isin(val), day.isin(test)


def time_of_day_profile(series, bin_min=15):
    """Mean and spread of arrivals per approach per time-of-day bin, over days."""
    s = series.copy()
    s["day"] = s["timestamp"].dt.date
    s["tod_bin"] = (s["timestamp"].dt.hour * 60 + s["timestamp"].dt.minute) // bin_min * bin_min
    per_day = s.groupby(["approach", "day", "tod_bin"])["arrivals"].sum().reset_index()
    return (per_day.groupby(["approach", "tod_bin"])["arrivals"]
            .agg(mean="mean", std="std", days="size").reset_index().fillna({"std": 0.0}))
