"""Re-timed fixed plan (Webster, 1958) and the matching SUMO signal programs.

    python -m makutano.control.webster --config config/site.yaml \
        --volumes outputs/pilot/movement_volumes.csv --out sumo/programs.add.xml

Writes two programs for the junction's traffic light, built from the phases
already in the SUMO network: "webster" (fixed, re-timed from measured flows)
and "actuated" (SUMO's queue-actuated control with the same phases). The
network's own program stays the baseline, "today's plan".
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from ..config import load_config


def webster_plan(critical_flows, saturation_flows, lost_time_per_phase_s=5.0, min_green_s=10.0,
                 max_cycle_s=150.0):
    """Cycle length and green times from critical flows (veh/h) per phase.

    Optimum cycle C0 = (1.5 L + 5) / (1 - Y), where L is total lost time and Y
    the sum of flow ratios; greens are shared in proportion to flow ratios. An
    oversaturated junction (Y >= 0.95) gets the longest allowed cycle.
    """
    y = [q / s for q, s in zip(critical_flows, saturation_flows)]
    Y = sum(y)
    lost = lost_time_per_phase_s * len(y)
    cycle = max_cycle_s if Y >= 0.95 else (1.5 * lost + 5) / (1 - Y)
    cycle = min(max(cycle, lost + min_green_s * len(y)), max_cycle_s)
    effective = cycle - lost
    greens = [max(min_green_s, effective * yi / Y) if Y > 0 else min_green_s for yi in y]
    return sum(greens) + lost, greens


def critical_flows(volumes, cfg, interval_s):
    """Peak-hour flow per lane of the busiest approach served by each phase."""
    c = cfg["control"]
    hourly = (volumes.groupby(["interval_start_s", "entry"])["count"].sum()
              .groupby("entry").max() * 3600 / interval_s)
    lanes = {a: cfg["approaches"][a].get("lanes", 1) for a in cfg["approaches"]}
    flows = []
    for served in c["phase_approaches"]:
        per_lane = [hourly.get(a, 0.0) / lanes[a] for a in served]
        flows.append(max(per_lane) if per_lane else 0.0)
    return flows


def read_program(net_file, tls_id):
    for tl in ET.parse(net_file).getroot().iter("tlLogic"):
        if tl.get("id") == tls_id:
            return [dict(p.attrib) for p in tl.iter("phase")]
    raise ValueError(f"traffic light {tls_id!r} not found in {net_file}")


def write_programs(net_file, tls_id, green_phases, greens_s, out_path, min_green_s, max_green_s):
    base = read_program(net_file, tls_id)
    root = ET.Element("additional")
    for program, kind in (("webster", "static"), ("actuated", "actuated")):
        tl = ET.SubElement(root, "tlLogic", id=tls_id, type=kind, programID=program, offset="0")
        for i, phase in enumerate(base):
            attrs = {"state": phase["state"], "duration": phase["duration"]}
            if i in green_phases:
                g = greens_s[green_phases.index(i)]
                attrs["duration"] = f"{g:.0f}"
                if kind == "actuated":
                    attrs.update(minDur=f"{min_green_s:.0f}", maxDur=f"{max_green_s:.0f}")
            ET.SubElement(tl, "phase", **attrs)
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--volumes", required=True, help="movement_volumes.csv from makutano.see.measure")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    c = cfg["control"]
    shield = c.get("shield", {})
    flows = critical_flows(pd.read_csv(args.volumes), cfg, cfg["indicators"]["interval_s"])
    sat = [c.get("saturation_flow_veh_h_lane", 1800)] * len(flows)
    lost = shield.get("yellow_s", 3) + shield.get("all_red_s", 2)
    cycle, greens = webster_plan(flows, sat, lost, shield.get("min_green_s", 10))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_programs(cfg["sumo"]["net_file"], c["tls_id"], c["green_phases"], greens, args.out,
                   shield.get("min_green_s", 10), shield.get("max_green_s", 60))
    print(f"Webster cycle {cycle:.0f} s, greens {[round(g) for g in greens]} s; programs written to {args.out}")


if __name__ == "__main__":
    main()
