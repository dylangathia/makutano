"""Write a SUMO route file from measured movement volumes.

    python -m makutano.sim.sumo_demand --config config/site.yaml \
        --volumes outputs/pilot/movement_volumes.csv --out sumo/pilot.rou.xml

Each (interval, entry, exit, class) count becomes one <flow> from the entry
approach's incoming edge to the exit approach's outgoing edge. Vehicle types
come from the `sumo.vtypes` section of the config, which is where calibration
changes them.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from ..config import load_config


def flow_id(entry, exit_, cls, begin):
    return f"{entry}__{exit_}__{cls}__{int(begin)}"


def parse_flow_id(vehicle_id):
    """SUMO names flow vehicles <flow id>.<n>; recover the movement from that."""
    entry, exit_, cls, begin = vehicle_id.rsplit(".", 1)[0].split("__")
    return entry, exit_, cls, int(begin)


def build_routes(volumes, cfg, interval_s, scale=1.0, arrivals="uniform"):
    approaches = cfg["approaches"]
    vtypes = cfg["sumo"]["vtypes"]
    root = ET.Element("routes")
    for vid, attrs in vtypes.items():
        ET.SubElement(root, "vType", id=vid, **{k: str(v) for k, v in attrs.items()})

    skipped = set()
    for r in volumes.sort_values(["interval_start_s", "entry", "exit", "cls"]).itertuples():
        if r.cls not in vtypes:
            skipped.add(r.cls)
            continue
        n = round(r.count * scale)
        if n <= 0:
            continue
        begin = int(r.interval_start_s)
        attrs = {
            "id": flow_id(r.entry, r.exit, r.cls, begin),
            "type": r.cls,
            "begin": str(begin),
            "end": str(begin + interval_s),
            "from": approaches[r.entry]["sumo_from_edge"],
            "to": approaches[r.exit]["sumo_to_edge"],
            "departLane": "best",
            "departSpeed": "max",
        }
        if arrivals == "poisson":
            # Random arrivals at the measured rate; the count then varies by seed.
            attrs["period"] = f"exp({n / interval_s:.6f})"
        else:
            attrs["number"] = str(n)
        ET.SubElement(root, "flow", **attrs)
    if skipped:
        print(f"warning: no vType for classes {sorted(skipped)}; those counts were left out")
    return ET.ElementTree(root)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--volumes", required=True, help="movement_volumes.csv from makutano.see.measure")
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=1.0, help="multiply all counts, e.g. for demand scenarios")
    p.add_argument("--arrivals", choices=["uniform", "poisson"], default="uniform")
    args = p.parse_args()

    cfg = load_config(args.config)
    tree = build_routes(pd.read_csv(args.volumes), cfg, cfg["indicators"]["interval_s"],
                        args.scale, args.arrivals)
    ET.indent(tree)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {len(tree.getroot().findall('flow'))} flows to {out}")


if __name__ == "__main__":
    main()
