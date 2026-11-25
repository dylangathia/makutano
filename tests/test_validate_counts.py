"""Checks for the shared class list and the manual-count validation."""
import numpy as np
import pandas as pd
import pytest

from makutano.classes import CLASSES, COCO_TO_LOCAL, VEHICLE_CLASSES
from makutano.config import load_config
from makutano.see.validate_counts import read_manual, validate

START = "2026-10-20T07:00:00"


def test_six_classes_and_coco_map_stays_inside_them():
    assert len(CLASSES) == 6 and "pedestrian" in CLASSES and "pedestrian" not in VEHICLE_CLASSES
    assert set(COCO_TO_LOCAL.values()) <= set(CLASSES)


def test_example_config_uses_project_classes():
    cfg = load_config("config/site.example.yaml")
    assert set(cfg["sumo"]["vtypes"]) == set(VEHICLE_CLASSES)


def test_config_rejects_unknown_vehicle_class(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text("approaches: {}\nsumo:\n  vtypes:\n    tuk_tuk: {vClass: passenger}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tuk_tuk"):
        load_config(p)


def _passes():
    # Three matatus south->north in the first five minutes, one at 07:05:10, one boda south->east,
    # and one matatu that entered but was never seen leaving.
    return pd.DataFrame({
        "track_id": range(6),
        "entry": ["south", "south", "south", "south", "south", "south"],
        "exit": ["north", "north", "north", "north", "east", np.nan],
        "cls": ["matatu", "matatu", "matatu", "matatu", "boda_boda", "matatu"],
        "t_entry": [10.0, 120.0, 299.0, 310.0, 50.0, 200.0],
    })


def test_counts_match_cells_by_movement_class_and_time():
    manual = read_manual(pd.DataFrame({
        "start": ["07:00", "07:05", "07:00"], "end": ["07:05", "07:10", "07:05"],
        "entry": ["south", "south", "south"], "exit": ["north", "north", "east"],
        "cls": ["matatu", "matatu", "boda_boda"], "count": [4, 1, 1],
    }), START)
    detail, summary = validate(_passes(), manual)
    assert detail["auto"].tolist() == [3, 1, 1]          # interval ends are exclusive: 299 s in, 310 s out
    assert detail["pct_error"].iloc[0] == pytest.approx(-25.0)
    overall = summary[summary["level"] == "overall"].iloc[0]
    assert (overall["manual"], overall["auto"]) == (6, 5)
    matatu = summary[(summary["level"] == "class") & (summary["group"] == "matatu")].iloc[0]
    assert (matatu["manual"], matatu["auto"]) == (5, 4)


def test_approach_level_count_includes_vehicles_without_an_exit():
    manual = read_manual(pd.DataFrame({"start_s": [0], "end_s": [300], "entry": ["south"], "count": [5]}))
    detail, _ = validate(_passes(), manual)
    assert detail["auto"].iat[0] == 5                    # all classes, any or no exit
    assert detail["geh"].iat[0] == pytest.approx(0.0)


def test_manual_count_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown classes"):
        read_manual(pd.DataFrame({"start_s": [0], "end_s": [60], "entry": ["south"], "cls": ["tuk_tuk"], "count": [1]}))
    with pytest.raises(ValueError, match="before the video starts"):
        read_manual(pd.DataFrame({"start": ["06:55"], "end": ["07:00"], "entry": ["south"], "count": [1]}), START)
    with pytest.raises(ValueError, match="end after it starts"):
        read_manual(pd.DataFrame({"start_s": [60], "end_s": [60], "entry": ["south"], "count": [1]}))
