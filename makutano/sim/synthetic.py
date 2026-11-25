"""Synthetic demand: stand-in movement counts and route files until footage exists.

    python -m makutano.sim.synthetic --config config/site.synthetic.yaml \
        --out sumo/synthetic --days 12 --seed 0

Writes, for each synthetic "day", a movement_volumes CSV in exactly the format
makutano.see.measure produces, and the SUMO route file built from it by
makutano.sim.sumo_demand, with pedestrian flows added. Everything downstream
(training, experiments) then runs unchanged; when real counts exist, they replace
these files and nothing else changes.

The morning peak shape, approach volumes, turning shares and class mix are
assumptions, not measurements: every result produced from them must be labelled
as coming from synthetic demand.
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from ..classes import BODA_BODA, HANDCART, MATATU, PRIVATE_CAR, TRUCK_BUS
from ..config import load_config
from .sumo_demand import build_routes

# Assumed, not measured. Approach volumes are veh/h near the peak.
BASE_VEH_H = {"north": 340, "south": 390, "east": 540, "west": 490}
TURN_SHARE = {"straight": 0.72, "left": 0.20, "right": 0.08}
CLASS_SHARE = {PRIVATE_CAR: 0.455, MATATU: 0.20, BODA_BODA: 0.21, TRUCK_BUS: 0.13, HANDCART: 0.005}
PEDS_PER_H = 480          # all pedestrian movements together
PEAK_H, PEAK_WIDTH_H = 7.75, 0.65
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
LEFT_OF = {"north": "east", "east": "south", "south": "west", "west": "north"}  # traffic keeps left


def peak_shape(hour):
    return 0.62 + 0.45 * math.exp(-(((hour - PEAK_H) / PEAK_WIDTH_H) ** 2))


def destination(origin, movement):
    if movement == "straight":
        return OPPOSITE[origin]
    return LEFT_OF[origin] if movement == "left" else OPPOSITE[LEFT_OF[origin]]


def day_volumes(start_hour, minutes, interval_s, rng, day_scale=1.0, approaches=BASE_VEH_H):
    """Poisson counts per (interval, entry, exit, class) for one synthetic morning."""
    rows = []
    for k in range(int(minutes * 60 // interval_s)):
        t0 = k * interval_s
        hour = start_hour + (t0 + interval_s / 2) / 3600
        for origin, veh_h in approaches.items():
            mean_total = veh_h * day_scale * peak_shape(hour) * interval_s / 3600
            for movement, m_share in TURN_SHARE.items():
                for cls, c_share in CLASS_SHARE.items():
                    n = rng.poisson(mean_total * m_share * c_share)
                    if n:
                        rows.append((t0, origin, destination(origin, movement), cls, int(n)))
    return pd.DataFrame(rows, columns=["interval_start_s", "entry", "exit", "cls", "count"])


def add_pedestrians(tree, cfg, end_s, per_hour=PEDS_PER_H):
    """Pedestrian flows between every pair of arms; SUMO routes them over the crossings."""
    root = tree.getroot()
    arms = list(cfg["approaches"])
    pairs = [(a, b) for a in arms for b in arms if a != b]
    # SUMO reads a route file in departure order, so flows that start at 0 go before the
    # vehicle flows of later intervals: straight after the vehicle types.
    at = sum(1 for el in root if el.tag == "vType")
    for a, b in pairs:
        flow = ET.Element("personFlow", id=f"ped__{a}__{b}", begin="0", end=str(end_s),
                          personsPerHour=f"{per_hour / len(pairs):.2f}")
        ET.SubElement(flow, "walk", **{"from": cfg["approaches"][a]["sumo_from_edge"],
                                       "to": cfg["approaches"][b]["sumo_to_edge"]})
        root.insert(at, flow)
        at += 1
    return tree


def write_days(cfg, out, days, seed=0, day_scale_sd=0.1):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    interval_s = cfg["indicators"]["interval_s"]
    minutes = cfg["control"].get("episode_minutes", 120)
    start = pd.Timestamp(cfg["control"]["sim_start"])
    start_hour = start.hour + start.minute / 60
    files = []
    for d in range(days):
        scale = max(0.5, rng.normal(1.0, day_scale_sd))
        vols = day_volumes(start_hour, minutes, interval_s, rng, scale)
        vols.to_csv(out / f"day{d:02d}_movement_volumes.csv", index=False)
        tree = add_pedestrians(build_routes(vols, cfg, interval_s), cfg, minutes * 60)
        ET.indent(tree)
        path = out / f"day{d:02d}.rou.xml"
        tree.write(path, encoding="utf-8", xml_declaration=True)
        files.append((path, scale, int(vols["count"].sum())))
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--days", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for path, scale, n in write_days(load_config(args.config), args.out, args.days, args.seed):
        print(f"{path}: day scale {scale:.2f}, {n} vehicles")


if __name__ == "__main__":
    main()
