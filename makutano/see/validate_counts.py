"""Check the pipeline's vehicle counts against a manual count of the same video.

    python -m makutano.see.validate_counts --config config/site.yaml \
        --passes outputs/day1_am/passes.csv --manual data/manual/day1_am_counts.csv \
        --out outputs/day1_am/count_validation

The manual count is a CSV with one row per counted cell:

    start,end,entry,exit,cls,count
    07:00,07:05,south,north,matatu,23
    07:00,07:05,south,north,boda_boda,41

- start, end: clock times (HH:MM or HH:MM:SS) on the video's clock, or
  start_s, end_s in seconds from the first frame. Intervals can be any length,
  so a few sampled five-minute periods are enough.
- entry: the approach vehicles came from. exit (optional): where they left; leave
  the column out, or a cell empty, to count the whole approach.
- cls (optional): one of the project's vehicle classes; leave it out, or empty,
  to count all classes together.

The same cells are counted from passes.csv (written by makutano.see.measure) and
compared: the difference, percentage error and GEH per cell, and totals by
class and by approach or movement. Writes count_validation.csv (per cell) and
count_validation_summary.csv.
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..classes import VEHICLE_CLASSES
from ..config import load_config
from ..sim.compare import geh


def _seconds(values, video_start):
    """Clock times or timestamps -> seconds from the first frame."""
    t0 = datetime.fromisoformat(str(video_start))
    out = []
    for v in values:
        v = str(v).strip()
        if "T" in v or "-" in v:
            t = datetime.fromisoformat(v)
        else:
            parts = [int(p) for p in v.split(":")]
            h, m, s = (parts + [0, 0])[:3]
            t = t0.replace(hour=h, minute=m, second=s, microsecond=0)
        sec = (t - t0).total_seconds()
        if sec < 0:
            raise ValueError(f"manual count time {v} is before the video starts ({t0:%H:%M:%S})")
        out.append(sec)
    return np.array(out)


def read_manual(path_or_df, video_start=None):
    m = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df, dtype={"start": str, "end": str})
    m = m.copy()
    if {"start_s", "end_s"} <= set(m.columns):
        m["start_s"], m["end_s"] = m["start_s"].astype(float), m["end_s"].astype(float)
    elif {"start", "end"} <= set(m.columns):
        if video_start is None:
            raise ValueError("clock times need the video start time: set site.video_start or pass --start")
        m["start_s"], m["end_s"] = _seconds(m["start"], video_start), _seconds(m["end"], video_start)
    else:
        raise ValueError("the manual count needs start/end clock times or start_s/end_s columns")
    if (m["end_s"] <= m["start_s"]).any():
        raise ValueError("every manual interval must end after it starts")
    for col in ("exit", "cls"):
        m[col] = m[col].fillna("").astype(str).str.strip() if col in m.columns else ""
    m["entry"] = m["entry"].astype(str).str.strip()
    unknown = sorted(set(m["cls"]) - set(VEHICLE_CLASSES) - {""})
    if unknown:
        raise ValueError(f"unknown classes in the manual count: {unknown}; use {VEHICLE_CLASSES} "
                         "(pedestrians are not counted as passes)")
    return m


def count_cells(passes, manual):
    """For each manual cell, count matching vehicles in passes."""
    p = passes
    autos = []
    for r in manual.itertuples():
        sel = (p["entry"] == r.entry) & (p["t_entry"] >= r.start_s) & (p["t_entry"] < r.end_s)
        if r.exit:
            sel &= p["exit"] == r.exit
        if r.cls:
            sel &= p["cls"] == r.cls
        autos.append(int(sel.sum()))
    return np.array(autos)


def _error_table(df, keys):
    g = df.groupby(keys, dropna=False)[["manual", "auto"]].sum().reset_index()
    g["diff"] = g["auto"] - g["manual"]
    g["pct_error"] = np.where(g["manual"] > 0, 100 * g["diff"] / g["manual"].where(g["manual"] > 0), np.nan)
    return g


def validate(passes, manual):
    """Per-cell comparison and a summary table (overall, by class, by approach or movement)."""
    cells = manual.copy()
    cells["manual"] = cells["count"].astype(int)
    cells["auto"] = count_cells(passes, cells)
    cells["diff"] = cells["auto"] - cells["manual"]
    cells["pct_error"] = np.where(cells["manual"] > 0,
                                  100 * cells["diff"] / cells["manual"].where(cells["manual"] > 0), np.nan)
    per_hour = 3600 / (cells["end_s"] - cells["start_s"])
    cells["geh"] = geh(cells["auto"] * per_hour, cells["manual"] * per_hour)
    detail = cells[["start_s", "end_s", "entry", "exit", "cls", "manual", "auto", "diff", "pct_error", "geh"]]

    parts = [_error_table(cells.assign(group="all"), ["group"]).assign(level="overall")]
    if (cells["cls"] != "").any():
        parts.append(_error_table(cells[cells["cls"] != ""], ["cls"]).rename(columns={"cls": "group"})
                     .assign(level="class"))
    movement = cells["entry"] + np.where(cells["exit"] != "", " -> " + cells["exit"], "")
    parts.append(_error_table(cells.assign(group=movement), ["group"]).assign(level="approach_or_movement"))
    summary = pd.concat(parts, ignore_index=True)[["level", "group", "manual", "auto", "diff", "pct_error"]]
    summary.attrs["geh_under_5"] = float((cells["geh"] < 5).mean())
    summary.attrs["mean_abs_error"] = float(cells["diff"].abs().mean())
    return detail, summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--passes", required=True, help="passes.csv written by makutano.see.measure")
    p.add_argument("--manual", required=True, help="the manual count CSV")
    p.add_argument("--out", required=True, help="directory for the validation tables")
    p.add_argument("--start", default=None, help="clock time of the first frame, if not site.video_start")
    args = p.parse_args()

    cfg = load_config(args.config)
    video_start = args.start or (cfg.get("site") or {}).get("video_start")
    manual = read_manual(args.manual, video_start)
    detail, summary = validate(pd.read_csv(args.passes), manual)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "count_validation.csv", index=False)
    summary.to_csv(out / "count_validation_summary.csv", index=False)

    overall = summary[summary["level"] == "overall"].iloc[0]
    print(f"{len(detail)} counted cells: manual {overall['manual']}, pipeline {overall['auto']} "
          f"({overall['pct_error']:+.1f}%); mean absolute error {summary.attrs['mean_abs_error']:.1f} "
          f"vehicles per cell; {100 * summary.attrs['geh_under_5']:.0f}% of cells have GEH < 5")
    for _, r in summary[summary["level"] != "overall"].iterrows():
        err = "n/a" if np.isnan(r["pct_error"]) else f"{r['pct_error']:+.1f}%"
        print(f"  {r['level']:<20} {r['group']:<24} manual {r['manual']:>5}  pipeline {r['auto']:>5}  {err}")
    print(f"Tables written to {out}")


if __name__ == "__main__":
    main()
