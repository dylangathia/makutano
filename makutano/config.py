"""Site configuration loading."""
from pathlib import Path

import yaml


def load_config(path):
    with open(Path(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for name in cfg.get("approaches", {}):
        # Approach names are encoded into SUMO flow ids as entry__exit__class__begin.
        if "__" in name or "." in name:
            raise ValueError(f"approach name {name!r} must not contain '__' or '.'")
    return cfg
