"""Compare simulated indicators with measured ones.

    python -m makutano.compare --config config/site.yaml \
        --measured outputs/pilot --sim outputs/pilot/sim

Volumes are checked with the GEH statistic on hourly-equivalent flows per
movement per interval; the usual acceptance target is GEH < 5 for at least 85%
of movements. Delay and queue length are checked as percentage error.
Simulated values are means over seeds with a 95% confidence interval.

Because demand is generated from the measured counts, volume GEH mainly shows
whether SUMO could insert and serve that demand. Delay and queue length are the
real calibration targets, and they should be checked on a period that was not
used for calibration.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

# Two-sided 95% Student t critical values by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def t95(dof):
    if dof < 1:
        return np.nan
    eligible = [k for k in T95 if k <= dof]
    return T95[max(eligible)] if dof <= 30 else 1.96


def geh(model, observed):
    model, observed = np.asarray(model, float), np.asarray(observed, float)
    total = model + observed
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.sqrt(2 * (model - observed) ** 2 / total)
    return np.where(total > 0, g, 0.0)


def seed_summary(df, keys, value, n_seeds, missing_as_zero=True):
    """Mean and 95% CI over seeds.

    For counts, a seed with no row for a key had zero vehicles. For averages
    such as delay, a missing seed has no value and is left out instead.
    """
    wide = df.pivot_table(index=keys, columns="seed", values=value, aggfunc="sum")
    wide = wide.reindex(columns=range(n_seeds))
    if missing_as_zero:
        wide = wide.fillna(0.0)
    n = wide.notna().sum(axis=1)
    mean = wide.mean(axis=1)
    half = wide.std(axis=1, ddof=1) / np.sqrt(n) * n.map(lambda k: t95(k - 1))
    return pd.DataFrame({f"sim_{value}": mean, f"sim_{value}_ci95": half}).reset_index()


def compare_volumes(measured, sim, interval_s, n_seeds):
    keys = ["interval_start_s", "entry", "exit"]
    obs = measured.groupby(keys)["count"].sum().rename("obs_count").reset_index()
    simulated = seed_summary(sim.groupby(keys + ["seed"])["count"].sum().reset_index(), keys, "count", n_seeds)
    v = obs.merge(simulated, on=keys, how="outer").fillna({"obs_count": 0, "sim_count": 0})
    per_hour = 3600 / interval_s
    v["obs_veh_h"] = v["obs_count"] * per_hour
    v["sim_veh_h"] = v["sim_count"] * per_hour
    v["geh"] = geh(v["sim_veh_h"], v["obs_veh_h"])
    return v


def compare_metric(measured, sim, keys, value, n_seeds):
    s = seed_summary(sim, keys, value, n_seeds, missing_as_zero=False)
    m = measured[keys + [value]].rename(columns={value: f"obs_{value}"})
    c = m.merge(s, on=keys, how="outer")
    c["pct_error"] = 100 * (c[f"sim_{value}"] - c[f"obs_{value}"]) / c[f"obs_{value}"]
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--measured", required=True, help="directory written by makutano.indicators")
    p.add_argument("--sim", required=True, help="directory written by makutano.run_sim")
    p.add_argument("--geh-threshold", type=float, default=5.0)
    args = p.parse_args()

    cfg = load_config(args.config)
    interval_s = cfg["indicators"]["interval_s"]
    m, s = Path(args.measured), Path(args.sim)
    sim_volumes = pd.read_csv(s / "sim_volumes.csv")
    n_seeds = sim_volumes["seed"].nunique()

    volumes = compare_volumes(pd.read_csv(m / "movement_volumes.csv"), sim_volumes, interval_s, n_seeds)
    delay = compare_metric(pd.read_csv(m / "delay.csv").query("cls == 'all'"),
                           pd.read_csv(s / "sim_delay.csv").query("cls == 'all'"),
                           ["interval_start_s", "cls"], "mean_delay_s", n_seeds)
    queues = compare_metric(pd.read_csv(m / "queues.csv"), pd.read_csv(s / "sim_queues.csv"),
                            ["interval_start_s", "approach"], "mean_queue_m", n_seeds)

    volumes.to_csv(s / "compare_volumes.csv", index=False)
    delay.to_csv(s / "compare_delay.csv", index=False)
    queues.to_csv(s / "compare_queues.csv", index=False)

    passing = (volumes["geh"] < args.geh_threshold).mean() * 100
    print(f"movements with GEH < {args.geh_threshold:g}: {passing:.0f}% of {len(volumes)} (target 85%)")
    print(f"mean absolute delay error:  {delay['pct_error'].abs().mean():.1f}%")
    print(f"mean absolute queue error:  {queues['pct_error'].abs().replace(np.inf, np.nan).mean():.1f}%")
    print(f"details in {s}/compare_*.csv")


if __name__ == "__main__":
    main()
