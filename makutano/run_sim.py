"""Run the SUMO model over several seeds and tabulate simulated indicators.

    python -m makutano.run_sim --config config/site.yaml \
        --routes sumo/pilot.rou.xml --out outputs/pilot/sim --seeds 10

Writes sim_volumes.csv, sim_delay.csv and sim_queues.csv with a `seed`
column, in the same shape as the measured tables so makutano.compare can line
them up. Simulated delay is SUMO's timeLoss, which matches the measured delay
definition only if the network is clipped to roughly the camera's study zone.
"""
import argparse
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .indicators import interval_start
from .sumo_demand import parse_flow_id


def sumo_binary():
    home = os.environ.get("SUMO_HOME")
    if home:
        for name in ("sumo", "sumo.exe"):
            candidate = Path(home) / "bin" / name
            if candidate.exists():
                return str(candidate)
    found = shutil.which("sumo")
    if not found:
        raise SystemExit("SUMO not found: install it and set SUMO_HOME, or put sumo on PATH")
    return found


def run_seed(cfg, routes, seed, out_dir, additional=()):
    s = cfg["sumo"]
    tripinfo = out_dir / f"tripinfo_seed{seed}.xml"
    queues = out_dir / f"queues_seed{seed}.xml"
    cmd = [
        sumo_binary(), "-n", s["net_file"], "-r", str(routes),
        "--seed", str(seed),
        "--lateral-resolution", str(s.get("lateral_resolution", 0.8)),
        "--tripinfo-output", str(tripinfo),
        "--queue-output", str(queues),
        "--time-to-teleport", str(s.get("time_to_teleport", 300)),
        "--no-step-log", "true",
    ]
    extra = list(s.get("additional_files", [])) + list(additional)
    if extra:
        cmd += ["-a", ",".join(extra)]
    subprocess.run(cmd, check=True)
    return tripinfo, queues


def parse_tripinfo(path):
    rows = []
    for _, el in ET.iterparse(path):
        if el.tag == "tripinfo":
            entry, exit_, cls, _ = parse_flow_id(el.get("id"))
            rows.append((entry, exit_, cls, float(el.get("depart")), float(el.get("timeLoss"))))
            el.clear()
    return pd.DataFrame(rows, columns=["entry", "exit", "cls", "depart_s", "delay_s"])


def parse_queues(path, cfg, sample_s=1.0):
    """Queue length per approach per time step: the longest lane queue on its edges."""
    lane_to_approach = {}
    for name, a in cfg["approaches"].items():
        for edge in a.get("sumo_queue_edges", [a["sumo_from_edge"]]):
            lane_to_approach[edge] = name
    rows = []
    for _, el in ET.iterparse(path):
        if el.tag == "data":
            t = float(el.get("timestep"))
            for lane in el.iter("lane"):
                edge = lane.get("id").rsplit("_", 1)[0]
                if edge in lane_to_approach:
                    rows.append((lane_to_approach[edge], t, float(lane.get("queueing_length"))))
            el.clear()
    df = pd.DataFrame(rows, columns=["approach", "t_s", "queue_m"])
    # SUMO only reports lanes that have a queue, so fill the gaps with zero.
    t_max = df["t_s"].max() if len(df) else 0.0
    grid = np.arange(0.0, t_max + sample_s, sample_s)
    full = []
    for name in cfg["approaches"]:
        q = df[df["approach"] == name]
        per_t = q.groupby(np.floor(q["t_s"] / sample_s) * sample_s)["queue_m"].max()
        full.append(pd.DataFrame({"approach": name, "t_s": grid,
                                  "queue_m": per_t.reindex(grid, fill_value=0.0).to_numpy()}))
    return pd.concat(full, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--routes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--additional", nargs="*", default=[],
                   help="extra additional files, e.g. a signal plan for a scenario")
    args = p.parse_args()

    cfg = load_config(args.config)
    interval_s = cfg["indicators"]["interval_s"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    volumes, delays, queues = [], [], []
    for seed in range(args.seeds):
        print(f"seed {seed}", flush=True)
        tripinfo, queue_file = run_seed(cfg, args.routes, seed, out, args.additional)
        trips = parse_tripinfo(tripinfo)
        trips["interval_start_s"] = interval_start(trips["depart_s"], interval_s)
        volumes.append(trips.groupby(["interval_start_s", "entry", "exit", "cls"]).size()
                       .rename("count").reset_index().assign(seed=seed))
        d = pd.concat([trips, trips.assign(cls="all")])
        delays.append(d.groupby(["interval_start_s", "cls"])["delay_s"].agg(["mean", "size"])
                      .rename(columns={"mean": "mean_delay_s", "size": "n"}).reset_index().assign(seed=seed))
        q = parse_queues(queue_file, cfg, cfg["indicators"].get("queue_sample_s", 1.0))
        q["interval_start_s"] = interval_start(q["t_s"], interval_s)
        queues.append(q.groupby(["interval_start_s", "approach"])["queue_m"]
                      .agg(mean_queue_m="mean", max_queue_m="max").reset_index().assign(seed=seed))

    pd.concat(volumes).to_csv(out / "sim_volumes.csv", index=False)
    pd.concat(delays).to_csv(out / "sim_delay.csv", index=False)
    pd.concat(queues).to_csv(out / "sim_queues.csv", index=False)
    print(f"wrote simulated indicators for {args.seeds} seeds to {out}")


if __name__ == "__main__":
    main()
