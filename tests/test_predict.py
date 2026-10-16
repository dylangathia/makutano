"""Module 2 (Predict) on synthetic series: no data or GPU needed."""
import numpy as np
import pandas as pd
import pytest

from makutano.predict import models
from makutano.predict.features import (
    FEATURES, TARGETS, build_examples, live_example, split_by_day, time_of_day_profile)

STEP = 5
APPROACHES = ["north", "south"]


def synthetic_series(days=4, minutes=60, seed=0, pattern=True):
    """Each day 07:00-08:00; arrival rate rises then falls the same way every day."""
    rng = np.random.default_rng(seed)
    steps = minutes * 60 // STEP
    frames = []
    for d in range(days):
        start = pd.Timestamp("2026-10-19 07:00") + pd.Timedelta(days=d)
        for a, base in zip(APPROACHES, (0.4, 0.2)):
            t = np.arange(steps) * STEP
            rate = base * (1 + np.sin(np.pi * t / (minutes * 60))) if pattern else np.full(steps, base)
            arrivals = rng.poisson(rate)
            frames.append(pd.DataFrame({
                "t_s": t, "approach": a, "arrivals": arrivals, "queue_m": arrivals * 6.0,
                "timestamp": start + pd.to_timedelta(t, unit="s"), "session": f"day{d}"}))
    return pd.concat(frames, ignore_index=True)


def test_examples_have_correct_windows_and_targets():
    steps = 40 * 60 // STEP
    s = pd.DataFrame({"t_s": np.arange(steps) * STEP, "approach": "north", "arrivals": 1, "queue_m": 3.0,
                      "timestamp": pd.Timestamp("2026-10-19 07:00") + pd.to_timedelta(np.arange(steps) * STEP, unit="s"),
                      "session": "a"})
    table, seq = build_examples(s, APPROACHES, STEP, with_sequences=True)
    # 15 min of history and 15 min ahead are needed, so 40 min of data leaves 10 min + 1 step.
    assert len(table) == steps - 2 * 15 * 12 + 1
    row = table.iloc[0]
    assert row["arr_last_1m"] == 12 and row["arr_last_5m"] == 60 and row["arr_last_15m"] == 180
    assert row["y_next_1m"] == 12 and row["y_next_15m"] == 180
    assert row["queue_now"] == 3.0 and row["queue_mean_1m"] == 3.0
    assert row["minute_of_day"] == pytest.approx(7 * 60 + 15)   # first forecast is made at 07:15
    assert seq.shape == (len(table), 5 * 12, 2)


def test_live_example_matches_training_example():
    s = synthetic_series(days=1)
    g = s[s["approach"] == "south"].sort_values("t_s").reset_index(drop=True)
    table, seq = build_examples(s, APPROACHES, STEP, with_sequences=True)
    train_rows = table[table["approach"] == "south"].reset_index(drop=True)
    i = 15 * 12 + 37                       # some step with a full history behind it
    k = i - 15 * 12                        # its row in the training table
    live_row, live_seq = live_example(g["arrivals"][:i], g["queue_m"][:i], g["timestamp"][i],
                                      "south", APPROACHES, STEP)
    for f in FEATURES:
        assert live_row[f].iat[0] == pytest.approx(train_rows[f].iat[k]), f
    south_seq = seq[(table["approach"] == "south").to_numpy()]
    np.testing.assert_allclose(live_seq[0], south_seq[k])


def test_split_by_day_keeps_test_days_unseen():
    table, _ = build_examples(synthetic_series(days=4), APPROACHES, STEP)
    train, val, test = split_by_day(table)
    days = sorted(table["day"].unique())
    assert set(table.loc[test, "day"]) == {days[-1]}
    assert set(table.loc[val, "day"]) == {days[-2]}
    assert not set(table.loc[train, "day"]) & set(table.loc[val | test, "day"])
    with pytest.raises(ValueError):
        split_by_day(table[table["day"] == days[0]])


def test_historical_average_beats_persistence_on_a_daily_pattern():
    table, _ = build_examples(synthetic_series(days=6, seed=1), APPROACHES, STEP)
    train, _, test = split_by_day(table)
    truth = table[test]
    def mae(model):
        pred = model.fit(table[train]).predict(truth)
        return np.mean([np.abs(pred[t] - truth[t]).mean() for t in TARGETS])
    # A repeating rhythm is what the time-of-day average captures and persistence misses.
    assert mae(models.HistoricalAverage()) < mae(models.Persistence())


def test_persistence_repeats_the_recent_past():
    table, _ = build_examples(synthetic_series(days=1), APPROACHES, STEP)
    pred = models.Persistence().predict(table)
    np.testing.assert_array_equal(pred["y_next_5m"], table["arr_last_5m"])


def test_gradient_boosting_fits_and_predicts(tmp_path):
    pytest.importorskip("sklearn")
    table, _ = build_examples(synthetic_series(days=4), APPROACHES, STEP)
    train, _, test = split_by_day(table)
    m = models.GradientBoosting(max_iter=30).fit(table[train])
    models.save(m, tmp_path / "f.pkl")
    pred = models.load(tmp_path / "f.pkl").predict(table[test])
    assert list(pred.columns) == TARGETS and (pred.to_numpy() >= 0).all()


def test_time_of_day_profile():
    s = synthetic_series(days=3, pattern=False)
    p = time_of_day_profile(s)
    assert set(p["tod_bin"]) == {420, 435, 450, 465}       # 07:00-08:00 in 15-minute bins
    north = p[p["approach"] == "north"]
    # 0.4 arrivals per 5 s is about 72 per 15 minutes.
    assert north["mean"].mean() == pytest.approx(72, rel=0.15)
    assert (p["days"] == 3).all()
