"""Arrival and queue series recorded from SUMO, as the camera would see them.

    python -m makutano.sim.sim_series --config config/site.synthetic.yaml \
        --routes "sumo/synthetic/train/*.rou.xml" --out sumo/synthetic/series

Runs each route file under the network's fixed signal plan and records, every
`indicators.series_step_s` seconds, the arrivals and the longest queue on each
approach within the camera's range: the same series.csv format that
makutano.see.measure writes from video. Each route file becomes one "recording
day" with its own date, so makutano.predict.evaluate can hold days out.

Used only with synthetic demand, to train and test the forecaster before real
series exist. Results from it must be labelled as synthetic.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..control.env import JunctionSim


def record(cfg, route, date, seed=0):
    step_s = cfg["indicators"].get("series_step_s", 5)
    end_s = cfg["control"].get("episode_minutes", 120) * 60
    start = pd.Timestamp(f"{date} {pd.Timestamp(cfg['control']['sim_start']).time()}")
    sim = JunctionSim(cfg, np.random.default_rng(seed))
    sim.start(route, seed)
    approaches = list(cfg["approaches"])
    rows, arrivals, queue = [], {a: 0 for a in approaches}, {a: 0.0 for a in approaches}
    try:
        while sim.running(end_s):
            sim.step()
            q, new = sim.measure()
            for a in approaches:
                arrivals[a] += new[a]
                queue[a] = max(queue[a], q[a])
            if int(sim.time) % step_s == 0:
                t0 = sim.time - step_s
                for a in approaches:
                    rows.append((t0, a, arrivals[a], queue[a]))
                arrivals, queue = {a: 0 for a in approaches}, {a: 0.0 for a in approaches}
    finally:
        sim.close()
    s = pd.DataFrame(rows, columns=["t_s", "approach", "arrivals", "queue_m"])
    s["timestamp"] = start + pd.to_timedelta(s["t_s"], unit="s")
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--routes", required=True, help="glob of route files, one per day")
    p.add_argument("--out", required=True)
    p.add_argument("--first-date", default="2026-10-05", help="date of the first day; days are weekdays")
    args = p.parse_args()

    cfg = load_config(args.config)
    routes = sorted(glob.glob(args.routes))
    dates = pd.bdate_range(args.first_date, periods=len(routes))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, (route, date) in enumerate(zip(routes, dates)):
        s = record(cfg, route, date.date(), seed=i)
        path = out / f"{Path(route).name.split('.')[0]}.csv"
        s.to_csv(path, index=False)
        print(f"{path}: {date.date()}, {int(s['arrivals'].sum())} arrivals counted")


if __name__ == "__main__":
    main()
