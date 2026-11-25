"""Site configuration loading."""
from pathlib import Path

import yaml

from .classes import CLASSES, VEHICLE_CLASSES


def load_config(path):
    with open(Path(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for name in cfg.get("approaches", {}):
        # Approach names are encoded into SUMO flow ids as entry__exit__class__begin.
        if "__" in name or "." in name:
            raise ValueError(f"approach name {name!r} must not contain '__' or '.'")
    # Class names have to match the detector's, or counts and simulated flows won't line up.
    for name in (cfg.get("sumo") or {}).get("vtypes", {}):
        if name not in VEHICLE_CLASSES:
            raise ValueError(f"sumo vtype {name!r} is not a vehicle class: use one of {VEHICLE_CLASSES}")
    speeds = (cfg.get("indicators") or {}).get("free_flow_speed_mps")
    for name in speeds if isinstance(speeds, dict) else ():
        if name != "default" and name not in CLASSES:
            raise ValueError(f"free-flow speed given for unknown class {name!r}: use one of {CLASSES}")
    return cfg
