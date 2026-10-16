"""Demand scenarios: varied training days and the high-pressure stress tests.

    python -m makutano.control.scenarios --config config/site.yaml \
        --profile outputs/predict/profile.csv --volumes outputs/pilot/movement_volumes.csv \
        --out sumo/train --n 30

Training routes are sampled from the time-of-day profile learned by the
prediction module, so the agent trains on many plausible peaks instead of one
recorded day. Scenarios from the config's `scenarios` section modify a route
file (demand scale, surges, pedestrian scale) or act while the simulation runs
(a blocked lane, a camera outage).
"""
import argparse
import copy
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..sim.sumo_demand import build_routes


@dataclass
class Scenario:
    name: str
    demand_scale: float = 1.0
    ped_scale: float = 1.0
    surge: dict = None           # {approach, start_s, duration_s, factor}
    lane_block: dict = None      # {lane, start_s, duration_s}
    camera_outage: dict = None   # {start_s, duration_s}
    stress: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def disruption(self):
        """(start_s, end_s) of the scenario's disruption, if it has one."""
        for d in (self.surge, self.lane_block, self.camera_outage):
            if d:
                return d["start_s"], d["start_s"] + d["duration_s"]
        return None

    def camera_out(self, t):
        c = self.camera_outage
        return bool(c) and c["start_s"] <= t < c["start_s"] + c["duration_s"]


def load_scenarios(cfg):
    raw = cfg.get("scenarios") or {"baseline": {}}
    known = set(Scenario.__dataclass_fields__) - {"name", "extra"}
    return [Scenario(name=n, **{k: v for k, v in (s or {}).items() if k in known}) for n, s in raw.items()]


def _scale_flow(el, factor):
    if el.get("number") is not None:
        el.set("number", str(max(0, round(float(el.get("number")) * factor))))
    elif el.get("probability") is not None:
        el.set("probability", f"{min(1.0, float(el.get('probability')) * factor):.6f}")
    elif el.get("period") is not None:
        period = el.get("period")
        m = re.fullmatch(r"exp\(([\d.eE+-]+)\)", period)
        if m:
            el.set("period", f"exp({float(m.group(1)) * factor:.6f})")
        else:
            el.set("period", f"{float(period) / factor:.6f}")
    elif el.get("vehsPerHour") is not None or el.get("personsPerHour") is not None:
        key = "vehsPerHour" if el.get("vehsPerHour") is not None else "personsPerHour"
        el.set(key, f"{float(el.get(key)) * factor:.3f}")


def apply_to_routes(route_file, scenario, cfg, out_file):
    """Write a copy of route_file with the scenario's demand changes applied."""
    tree = ET.parse(route_file)
    root = tree.getroot()
    if scenario.demand_scale != 1.0:
        for el in root.iter("flow"):
            _scale_flow(el, scenario.demand_scale)
    if scenario.ped_scale != 1.0:
        for el in root.iter("personFlow"):
            _scale_flow(el, scenario.ped_scale)
    if scenario.surge:
        s = scenario.surge
        edge = cfg["approaches"][s["approach"]]["sumo_from_edge"]
        start, end = s["start_s"], s["start_s"] + s["duration_s"]
        for el in list(root.iter("flow")):
            begin, finish = float(el.get("begin", 0)), float(el.get("end", 0))
            overlap = max(0.0, min(end, finish) - max(start, begin))
            if el.get("from") != edge or overlap <= 0 or el.get("number") is None:
                continue
            # Extra vehicles in the window, in the same movement and class mix.
            extra = round(float(el.get("number")) * overlap / (finish - begin) * (s["factor"] - 1))
            if extra > 0:
                surge = copy.deepcopy(el)
                surge.set("id", el.get("id") + "_surge")
                surge.set("begin", str(max(start, begin)))
                surge.set("end", str(min(end, finish)))
                surge.set("number", str(extra))
                root.append(surge)
        # SUMO needs flows sorted by departure time.
        flows = sorted(root.findall("flow"), key=lambda e: float(e.get("begin", 0)))
        for el in flows:
            root.remove(el)
        root.extend(flows)
    ET.indent(tree)
    tree.write(out_file, encoding="utf-8", xml_declaration=True)
    return out_file


class RuntimeEvents:
    """Scenario events applied during a run through a TraCI connection."""

    def __init__(self, scenario, cfg):
        self.scenario = scenario
        self.blocked = False
        self.saved = None
        self.classes = sorted({v["vClass"] for v in cfg["sumo"]["vtypes"].values()})

    def on_step(self, conn, t):
        b = self.scenario.lane_block
        if not b:
            return
        active = b["start_s"] <= t < b["start_s"] + b["duration_s"]
        if active and not self.blocked:
            self.saved = list(conn.lane.getDisallowed(b["lane"]))
            conn.lane.setDisallowed(b["lane"], self.classes)
            self.blocked = True
        elif not active and self.blocked:
            conn.lane.setDisallowed(b["lane"], self.saved)
            self.blocked = False


def movement_shares(volumes):
    """Share of each (exit, class) among vehicles entering from each approach."""
    totals = volumes.groupby(["entry", "exit", "cls"])["count"].sum()
    return totals / totals.groupby(level="entry").transform("sum")


def sample_training_volumes(profile, shares, start_min, end_min, rng, bin_min=15, day_scale_sd=0.1):
    """One synthetic peak: arrivals per approach per bin drawn around the profile.

    Each sampled day gets an overall level (day-to-day variation) and each bin
    its own noise from the spread seen across recorded days.
    """
    level = max(0.2, rng.normal(1.0, day_scale_sd))
    rows = []
    window = profile[(profile["tod_bin"] >= start_min) & (profile["tod_bin"] < end_min)]
    for r in window.itertuples():
        mean = max(0.0, rng.normal(r.mean, r.std)) * level
        n = rng.poisson(mean)
        if r.approach not in shares.index.get_level_values("entry"):
            continue
        split = shares.loc[r.approach]
        counts = rng.multinomial(n, split.to_numpy() / split.sum())
        for (exit_, cls), c in zip(split.index, counts):
            if c:
                rows.append({"interval_start_s": int((r.tod_bin - start_min) * 60), "entry": r.approach,
                             "exit": exit_, "cls": cls, "count": int(c)})
    return pd.DataFrame(rows, columns=["interval_start_s", "entry", "exit", "cls", "count"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--profile", required=True, help="profile.csv from makutano.predict.evaluate")
    p.add_argument("--volumes", required=True, help="movement_volumes.csv, for turning and class shares")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=30, help="number of training route files")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    cfg = load_config(args.config)
    c = cfg["control"]
    start = pd.Timestamp(c["sim_start"])
    start_min = start.hour * 60 + start.minute
    end_min = start_min + c.get("episode_minutes", 120)
    profile = pd.read_csv(args.profile)
    shares = movement_shares(pd.read_csv(args.volumes))
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        vols = sample_training_volumes(profile, shares, start_min, end_min, rng)
        tree = build_routes(vols, cfg, 15 * 60, arrivals=cfg["sumo"].get("arrivals", "poisson"))
        ET.indent(tree)
        tree.write(out / f"train_{i:03d}.rou.xml", encoding="utf-8", xml_declaration=True)
    print(f"wrote {args.n} training route files to {out}")


if __name__ == "__main__":
    main()
