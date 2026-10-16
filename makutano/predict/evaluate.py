"""Train and compare the arrival forecasters on held-out days.

    python -m makutano.predict.evaluate --config config/site.yaml \
        --series outputs/*/series.csv --out outputs/predict

Models are trained on the earlier recording days, chosen on the validation
day(s) and scored once on the test day(s), which no model has seen. Writes:
  metrics.csv            MAE and RMSE per model, horizon and split
  metrics_by_approach.csv  the same on the test days, per approach
  profile.csv            time-of-day arrival profile from the training days
  profile_metrics.csv    error of that profile on the test days
  forecaster.pkl         the model with the lowest validation error, for the controller
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from . import models
from .features import HORIZONS_MIN, TARGETS, build_examples, load_series, split_by_day, time_of_day_profile


def errors(truth, pred):
    rows = []
    for h, target in zip(HORIZONS_MIN, TARGETS):
        e = pred[target].to_numpy() - truth[target].to_numpy()
        rows.append({"horizon_min": h, "mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean())})
    return pd.DataFrame(rows)


def available_models(use_gru):
    candidates = [models.Persistence(), models.HistoricalAverage()]
    try:
        import sklearn  # noqa: F401
        candidates.append(models.GradientBoosting())
    except ImportError:
        print("scikit-learn not installed: skipping gradient boosting")
    if use_gru:
        try:
            import torch  # noqa: F401
            candidates.append(models.GRUForecaster())
        except ImportError:
            print("PyTorch not installed: skipping the GRU")
    return candidates


def profile_errors(series, profile, test_days, bin_min=15):
    s = series[series["timestamp"].dt.date.isin(test_days)].copy()
    s["day"] = s["timestamp"].dt.date
    s["tod_bin"] = (s["timestamp"].dt.hour * 60 + s["timestamp"].dt.minute) // bin_min * bin_min
    observed = s.groupby(["approach", "day", "tod_bin"])["arrivals"].sum().rename("observed").reset_index()
    m = observed.merge(profile, on=["approach", "tod_bin"], how="left")
    m["error"] = m["mean"] - m["observed"]
    return (m.groupby("approach")["error"]
            .agg(mae=lambda e: e.abs().mean(), rmse=lambda e: np.sqrt((e ** 2).mean()), bins="size")
            .reset_index())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--series", nargs="+", required=True, help="series.csv files (globs allowed)")
    p.add_argument("--out", required=True)
    p.add_argument("--gru", action="store_true", help="also train the GRU (needs PyTorch)")
    args = p.parse_args()

    cfg = load_config(args.config)
    pc = cfg.get("predict", {})
    paths = sorted({f for pattern in args.series for f in glob.glob(pattern)})
    series = load_series(paths)
    step_s = cfg["indicators"].get("series_step_s", 5)
    approaches = list(cfg["approaches"])
    table, sequences = build_examples(series, approaches, step_s, with_sequences=args.gru)
    train, val, test = split_by_day(table, pc.get("val_days", 1), pc.get("test_days", 1))
    print(f"{len(paths)} sessions, {table['day'].nunique()} days, "
          f"{int(train.sum())}/{int(val.sum())}/{int(test.sum())} train/val/test examples")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq = (lambda mask: sequences[mask.to_numpy()]) if sequences is not None else (lambda mask: None)
    results, fitted = [], {}
    for model in available_models(args.gru):
        if isinstance(model, models.GRUForecaster):
            model.fit(table[train], seq(train), table[val], seq(val))
        else:
            model.fit(table[train], seq(train))
        fitted[model.name] = model
        for split, mask in (("val", val), ("test", test)):
            e = errors(table[mask], model.predict(table[mask], seq(mask)))
            results.append(e.assign(model=model.name, split=split))
    metrics = pd.concat(results, ignore_index=True)[["model", "split", "horizon_min", "mae", "rmse"]]
    metrics.to_csv(out / "metrics.csv", index=False)

    by_approach = []
    for name, model in fitted.items():
        t = table[test]
        pred = model.predict(t, seq(test))
        for approach in approaches:
            mask = (t["approach"] == approach).to_numpy()
            if mask.any():
                by_approach.append(errors(t[mask], pred[mask]).assign(model=name, approach=approach))
    pd.concat(by_approach).to_csv(out / "metrics_by_approach.csv", index=False)

    # Choose on validation only; the test numbers are reported, never used to choose.
    best = metrics[metrics["split"] == "val"].groupby("model")["mae"].mean().idxmin()
    models.save(fitted[best], out / "forecaster.pkl")

    train_days = set(table.loc[train, "day"])
    test_days = set(table.loc[test, "day"])
    profile = time_of_day_profile(series[series["timestamp"].dt.date.isin(train_days)])
    profile.to_csv(out / "profile.csv", index=False)
    profile_errors(series, profile, test_days).to_csv(out / "profile_metrics.csv", index=False)

    print(metrics[metrics["split"] == "test"].pivot(index="model", columns="horizon_min", values="mae")
          .round(2).to_string())
    print(f"chosen on validation: {best} (saved to {out / 'forecaster.pkl'})")


if __name__ == "__main__":
    main()
