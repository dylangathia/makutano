"""Build a SUMO network of a four-arm signalised junction, with its signal program.

    python -m makutano.sim.build_network --config config/site.yaml --out sumo/junction.net.xml

This is the placeholder geometry used until the real junction is drawn from the
video and satellite imagery: four straight arms, the same number of lanes each
way, traffic keeping left, sidewalks and zebra crossings on every arm. Edge
names follow the config (`<approach>_in`, `<approach>_out`) and the traffic light
gets the id in `control.tls_id`.

The signal program has the phases the controller expects (`control.green_phases`
and `control.phase_approaches`), each green followed by its amber:

    0 north-south through   (right turns may go when the gap allows)
    2 north-south right turns, protected
    4 east-west through     (right turns may go when the gap allows)
    6 pedestrians only      (every crossing; all vehicles stop)

Durations in the written program are a fixed plan; the RL environment and the
experiments set the lights themselves through TraCI.
"""
import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from ..config import load_config
from .run_sim import sumo_binary

ARMS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}  # SUMO's y points north
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
FIXED_PLAN_S = [40, 3, 8, 3, 25, 3, 10, 3]   # green / amber durations for the written program


def netconvert_binary():
    return str(Path(sumo_binary()).with_name(Path(sumo_binary()).name.replace("sumo", "netconvert")))


def write_plain(folder, arm_m, lanes, speed_mps):
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="J1", x="0", y="0", type="traffic_light", tl="J1")
    edges = ET.Element("edges")
    for arm, (dx, dy) in ARMS.items():
        ET.SubElement(nodes, "node", id=arm, x=str(dx * arm_m), y=str(dy * arm_m), type="priority")
        common = {"numLanes": str(lanes), "speed": f"{speed_mps:.2f}", "sidewalkWidth": "2.0"}
        ET.SubElement(edges, "edge", id=f"{arm}_in", **{"from": arm, "to": "J1"}, **common)
        ET.SubElement(edges, "edge", id=f"{arm}_out", **{"from": "J1", "to": arm}, **common)
    for tree, name in ((nodes, "junction.nod.xml"), (edges, "junction.edg.xml")):
        ET.indent(tree)
        ET.ElementTree(tree).write(folder / name, encoding="utf-8", xml_declaration=True)


def turn(origin, dest):
    """Movement for traffic keeping left: coming from `origin`, leaving towards `dest`."""
    if dest == OPPOSITE[origin]:
        return "straight"
    ox, oy = ARMS[origin]
    tx, ty = -ox, -oy                    # direction of travel
    left = (-ty, tx)                     # 90 degrees anticlockwise (y points north)
    return "left" if ARMS[dest] == left else "right"


def link_table(net_file, tls_id):
    """One entry per signal link index: ('vehicle', origin, dest, movement) or ('crossing', edge)."""
    import sumolib
    net = sumolib.net.readNet(str(net_file), withPedestrianConnections=True)
    links = {}
    for in_lane, out_lane, index in net.getTLS(tls_id).getConnections():
        src, dst = in_lane.getEdge(), out_lane.getEdge()
        if dst.getFunction() == "crossing" or src.getFunction() == "crossing":
            crossing = dst if dst.getFunction() == "crossing" else src
            links[index] = ("crossing", crossing.getID())
        else:
            origin, dest = src.getID().rsplit("_", 1)[0], dst.getID().rsplit("_", 1)[0]
            links[index] = ("vehicle", origin, dest, turn(origin, dest))
    return [links[i] for i in range(max(links) + 1)]


def phase_states(links):
    """The eight states of the program, in the order described in the module docstring."""
    def green(ns_or_ew):
        axis = {"north", "south"} if ns_or_ew == "ns" else {"east", "west"}
        s = []
        for link in links:
            if link[0] == "vehicle" and link[1] in axis:
                s.append("g" if link[3] == "right" else "G")   # right turns give way to oncoming
            else:
                s.append("r")
        return "".join(s)

    def amber(state):
        return "".join("y" if c in "Gg" else c for c in state)

    ns, ew = green("ns"), green("ew")
    rights = "".join("G" if l[0] == "vehicle" and l[1] in {"north", "south"} and l[3] == "right" else "r"
                     for l in links)
    peds = "".join("G" if l[0] == "crossing" else "r" for l in links)
    return [ns, amber(ns), rights, amber(rights), ew, amber(ew), peds, "r" * len(links)]


def build(cfg, out, arm_m=200.0, lanes=2, speed_mps=13.89):
    out = Path(out)
    folder = out.parent
    folder.mkdir(parents=True, exist_ok=True)
    tls_id = cfg["control"]["tls_id"]
    write_plain(folder, arm_m, lanes, speed_mps)
    common = ["--lefthand", "true", "--sidewalks.guess", "true", "--crossings.guess", "true",
              "--no-turnarounds", "true", "--junctions.corner-detail", "5", "--no-warnings", "true"]
    subprocess.run([netconvert_binary(), "-n", str(folder / "junction.nod.xml"),
                    "-e", str(folder / "junction.edg.xml"), "-o", str(out)] + common, check=True)

    states = phase_states(link_table(out, tls_id))
    tll = ET.Element("tlLogics")
    logic = ET.SubElement(tll, "tlLogic", id=tls_id, type="static", programID="0", offset="0")
    for state, dur in zip(states, FIXED_PLAN_S):
        ET.SubElement(logic, "phase", duration=str(dur), state=state)
    ET.indent(tll)
    ET.ElementTree(tll).write(folder / "junction.tll.xml", encoding="utf-8", xml_declaration=True)
    subprocess.run([netconvert_binary(), "-s", str(out), "--tllogic-files", str(folder / "junction.tll.xml"),
                    "-o", str(out)] + common, check=True)
    return states


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arm-m", type=float, default=200.0, help="length of each arm (longer than the camera range)")
    p.add_argument("--lanes", type=int, default=2, help="lanes each way on every arm")
    args = p.parse_args()
    states = build(load_config(args.config), args.out, args.arm_m, args.lanes)
    print(f"wrote {args.out}; signal program with {len(states[0])} links:")
    for i, s in enumerate(states):
        print(f"  phase {i}: {s}")


if __name__ == "__main__":
    main()
