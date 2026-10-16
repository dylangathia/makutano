"""Turn tracks into intersection indicators.

    python -m makutano.see.measure --config config/site.yaml \
        --tracks outputs/pilot/tracks.csv --out outputs/pilot

Writes, per interval:
  movement_volumes.csv  entry -> exit counts by class (complete passes only)
  approach_volumes.csv  entries by approach and class (includes passes whose exit was not seen)
  delay.csv             mean delay in the study zone by class, plus an "all" row
  speeds.csv            space-mean speed in the study zone by class
  queues.csv            mean and max queue length per approach
and passes.csv, one row per tracked vehicle, for checking the numbers by eye.

It also writes series.csv: arrivals and queue length per approach in short
steps (series_step_s, default 5 s). This is the input to the prediction
module. Pass --start with the wall-clock time of the first frame so the
series carries real timestamps; the predictor needs time of day and date.

Definitions (state these in the report):
  delay        = time inside the study zone - path length / free-flow speed
  speed        = total distance / total time over all passes (space-mean speed)
  queue length = distance from the stop line to the furthest vehicle inside the
                 approach's queue zone moving slower than queue_speed_mps,
                 sampled every queue_sample_s and averaged over the interval
Zones and stop lines are drawn in image pixels; distances use the homography.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from .homography import homography_from_config, to_ground

PEDESTRIAN = "pedestrian"
TWO_WHEELERS = {"boda_boda", "bicycle"}


def points_in_polygon(pts, poly):
    """Vectorised even-odd ray casting. pts (N, 2), poly (M, 2) -> bool (N,)."""
    pts = np.asarray(pts, dtype=float)
    poly = np.asarray(poly, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_cross)
        j = i
    return inside


def point_to_segment_distance(pts, a, b):
    pts, a, b = np.asarray(pts, float), np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip((pts - a) @ ab / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def drop_riders(df, min_overlap=0.3, min_share=0.5):
    """Remove pedestrian tracks that are really boda boda or bicycle riders.

    A person counts as riding in a frame when much of their box overlaps a
    two-wheeler box and their feet are not below its wheels. A track riding in
    at least `min_share` of its frames is dropped.
    """
    persons = df[df["cls"] == PEDESTRIAN]
    bikes = df[df["cls"].isin(TWO_WHEELERS)]
    if persons.empty or bikes.empty:
        return df
    box = ["frame", "x1", "y1", "x2", "y2"]
    m = persons[["track_id"] + box].merge(bikes[box], on="frame", suffixes=("", "_b"))
    ix = (np.minimum(m["x2"], m["x2_b"]) - np.maximum(m["x1"], m["x1_b"])).clip(lower=0)
    iy = (np.minimum(m["y2"], m["y2_b"]) - np.maximum(m["y1"], m["y1_b"])).clip(lower=0)
    area = ((m["x2"] - m["x1"]) * (m["y2"] - m["y1"])).clip(lower=1e-6)
    feet_ok = m["y2"] <= m["y2_b"] + 0.1 * (m["y2_b"] - m["y1_b"])
    on_bike = (ix * iy / area > min_overlap) & feet_ok
    riding_frames = m[on_bike].groupby("track_id")["frame"].nunique()
    total_frames = persons.groupby("track_id")["frame"].nunique()
    share = riding_frames.reindex(total_frames.index, fill_value=0) / total_frames
    riders = share[share >= min_share].index
    return df[~((df["cls"] == PEDESTRIAN) & df["track_id"].isin(riders))]


def prepare_tracks(df, H, min_frames=10, smooth_frames=7):
    """Clean tracks and add foot points, ground positions and speeds."""
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    # Trackers keep the id but the class can flicker; use the class seen most often.
    df["cls"] = df.groupby("track_id")["cls"].transform(lambda s: s.mode().iat[0])
    df = drop_riders(df)
    df = df[df.groupby("track_id")["frame"].transform("size") >= min_frames].reset_index(drop=True)

    # The bottom-centre of the box is the best single estimate of where the
    # object touches the road, which is the plane the homography maps.
    df["u"] = (df["x1"] + df["x2"]) / 2
    df["v"] = df["y2"]
    ground = to_ground(H, df[["u", "v"]].to_numpy())
    df["gx"], df["gy"] = ground[:, 0], ground[:, 1]

    g = df.groupby("track_id")
    for c in ("gx", "gy"):
        df[c + "_s"] = g[c].transform(
            lambda s: s.rolling(smooth_frames, center=True, min_periods=1).median())

    # Speed as a central difference over `half` rows each way: per-frame
    # differences are dominated by box jitter.
    half = max(1, smooth_frames // 2)
    cols = ["gx_s", "gy_s", "t_s"]
    g = df.groupby("track_id")
    fwd = g[cols].shift(-half).fillna(df[cols])
    back = g[cols].shift(half).fillna(df[cols])
    dt = fwd["t_s"] - back["t_s"]
    dist = np.hypot(fwd["gx_s"] - back["gx_s"], fwd["gy_s"] - back["gy_s"])
    df["speed_mps"] = (dist / dt.where(dt > 0)).fillna(0.0)

    step = np.hypot(g["gx_s"].diff(), g["gy_s"].diff())
    df["step_m"] = step.fillna(0.0)
    return df


def _first_or_last_in_zone(df, zones, first):
    uv = df[["u", "v"]].to_numpy()
    parts = []
    for name, poly in zones.items():
        hit = df.loc[points_in_polygon(uv, poly), ["track_id", "t_s"]]
        t = hit.groupby("track_id")["t_s"].min() if first else hit.groupby("track_id")["t_s"].max()
        parts.append(pd.DataFrame({"track_id": t.index, "approach": name, "t": t.to_numpy()}))
    if not parts:
        return pd.DataFrame(columns=["track_id", "approach", "t"])
    allhits = pd.concat(parts, ignore_index=True).sort_values("t", ascending=first)
    return allhits.drop_duplicates("track_id")


def vehicle_passes(df, cfg, free_flow):
    """One row per vehicle track: movement, and delay/speed inside the study zone."""
    veh = df[df["cls"] != PEDESTRIAN]
    approaches = cfg["approaches"]
    entries = _first_or_last_in_zone(veh, {n: a["entry_zone"] for n, a in approaches.items()}, first=True)
    exits = _first_or_last_in_zone(veh, {n: a["exit_zone"] for n, a in approaches.items()}, first=False)
    passes = (entries.rename(columns={"approach": "entry", "t": "t_entry"})
              .merge(exits.rename(columns={"approach": "exit", "t": "t_exit"}), on="track_id", how="left"))
    backwards = passes["t_exit"] <= passes["t_entry"]
    passes.loc[backwards, ["exit", "t_exit"]] = np.nan
    passes = passes.merge(veh.groupby("track_id")["cls"].first().reset_index(), on="track_id")

    in_zone = veh[points_in_polygon(veh[["u", "v"]].to_numpy(), cfg["study_zone"])]
    zone = in_zone.groupby("track_id").agg(t_in=("t_s", "min"), t_out=("t_s", "max"),
                                           length_m=("step_m", "sum")).reset_index()
    passes = passes.merge(zone, on="track_id", how="left")
    passes["zone_time_s"] = passes["t_out"] - passes["t_in"]
    ffs = passes["cls"].map(lambda c: free_flow.get(c, free_flow["default"]))
    passes["delay_s"] = (passes["zone_time_s"] - passes["length_m"] / ffs).clip(lower=0)
    # Delay and speed only mean something for vehicles seen all the way through.
    incomplete = passes["exit"].isna() | ~(passes["zone_time_s"] > 0)
    passes.loc[incomplete, ["delay_s", "zone_time_s", "length_m"]] = np.nan
    return passes


def queue_samples(df, cfg, H, sample_s, queue_speed):
    """Queue length per approach, one value per `sample_s` bin."""
    veh = df[df["cls"] != PEDESTRIAN]
    bins = np.arange(int(np.floor(df["t_s"].max() / sample_s)) + 1)
    out = []
    for name, a in cfg["approaches"].items():
        if "queue_zone" not in a or "stop_line" not in a:
            continue
        stop = to_ground(H, a["stop_line"])
        queued = veh[points_in_polygon(veh[["u", "v"]].to_numpy(), a["queue_zone"])
                     & (veh["speed_mps"] < queue_speed)]
        dist = point_to_segment_distance(queued[["gx_s", "gy_s"]].to_numpy(), stop[0], stop[1])
        per_bin = (pd.Series(dist, index=np.floor(queued["t_s"].to_numpy() / sample_s).astype(int))
                   .groupby(level=0).max().reindex(bins, fill_value=0.0))
        out.append(pd.DataFrame({"approach": name, "t_s": bins * sample_s, "queue_m": per_bin.to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["approach", "t_s", "queue_m"])


def interval_start(t, interval_s):
    return (np.floor(t / interval_s) * interval_s).astype("Int64")


def summarise(passes, queues, interval_s):
    p = passes.copy()
    p["interval_start_s"] = interval_start(p["t_entry"], interval_s)
    approach_volumes = p.groupby(["interval_start_s", "entry", "cls"]).size().rename("count").reset_index()
    done = p[p["exit"].notna()]
    movement_volumes = (done.groupby(["interval_start_s", "entry", "exit", "cls"]).size()
                        .rename("count").reset_index())

    timed = done[done["delay_s"].notna()]
    delay = pd.concat([
        timed.groupby(["interval_start_s", "cls"])["delay_s"].agg(["mean", "size"]).reset_index(),
        timed.assign(cls="all").groupby(["interval_start_s", "cls"])["delay_s"].agg(["mean", "size"]).reset_index(),
    ]).rename(columns={"mean": "mean_delay_s", "size": "n"})

    sums = timed.groupby(["interval_start_s", "cls"])[["length_m", "zone_time_s"]].sum()
    speeds = (sums["length_m"] / sums["zone_time_s"]).rename("space_mean_speed_mps").reset_index()

    q = queues.copy()
    q["interval_start_s"] = interval_start(q["t_s"], interval_s)
    queue = (q.groupby(["interval_start_s", "approach"])["queue_m"]
             .agg(mean_queue_m="mean", max_queue_m="max").reset_index())
    return {"approach_volumes": approach_volumes, "movement_volumes": movement_volumes,
            "delay": delay, "speeds": speeds, "queues": queue}


def arrival_times(df, passes, cfg):
    """When each vehicle first appears on its approach.

    That is its first point inside the approach's queue zone, which reaches as
    far upstream as the camera sees. This is demand, unlike the stop-line time,
    which the signal controls. Falls back to the stop-line time without a zone.
    """
    out = passes["t_entry"].copy()
    uv = df[["u", "v"]].to_numpy()
    for name, a in cfg["approaches"].items():
        if "queue_zone" not in a:
            continue
        first = df.loc[points_in_polygon(uv, a["queue_zone"]), ["track_id", "t_s"]].groupby("track_id")["t_s"].min()
        mask = passes["entry"] == name
        t = passes.loc[mask, "track_id"].map(first)
        out.loc[mask] = np.minimum(t.fillna(passes.loc[mask, "t_entry"]), passes.loc[mask, "t_entry"])
    return out


def approach_series(passes, queues, approaches, t_end, step_s, start=None):
    """Arrivals and queue length per approach in fixed time steps.

    An arrival is a vehicle first appearing on the approach (t_arrival); queue
    length is the largest queue seen in the step. Steps with nothing seen
    count as zero.
    """
    steps = np.arange(int(np.floor(t_end / step_s)) + 1)
    rows = []
    for name in approaches:
        t_in = passes.loc[passes["entry"] == name, "t_arrival"].to_numpy()
        arrivals = np.bincount(np.floor(t_in / step_s).astype(int), minlength=len(steps))[:len(steps)]
        q = queues[queues["approach"] == name]
        queue = (q.groupby(np.floor(q["t_s"].to_numpy() / step_s).astype(int))["queue_m"].max()
                 .reindex(steps, fill_value=0.0).to_numpy())
        rows.append(pd.DataFrame({"t_s": steps * step_s, "approach": name,
                                  "arrivals": arrivals, "queue_m": queue}))
    series = pd.concat(rows, ignore_index=True)
    if start is not None:
        series["timestamp"] = pd.Timestamp(start) + pd.to_timedelta(series["t_s"], unit="s")
    return series


def compute(tracks, cfg, start=None):
    ind = cfg["indicators"]
    H = homography_from_config(cfg)
    free_flow = ind["free_flow_speed_mps"]
    if not isinstance(free_flow, dict):
        free_flow = {"default": free_flow}
    df = prepare_tracks(tracks, H, ind.get("min_track_frames", 10), ind.get("smooth_frames", 7))
    passes = vehicle_passes(df, cfg, free_flow)
    passes["t_arrival"] = arrival_times(df[df["cls"] != PEDESTRIAN], passes, cfg)
    queues = queue_samples(df, cfg, H, ind.get("queue_sample_s", 1.0), ind.get("queue_speed_mps", 1.4))
    tables = summarise(passes, queues, ind["interval_s"])
    tables["passes"] = passes
    tables["series"] = approach_series(passes, queues, cfg["approaches"], df["t_s"].max(),
                                       ind.get("series_step_s", 5), start)
    return tables


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--out", required=True, help="directory for the indicator CSVs")
    p.add_argument("--start", default=None,
                   help="wall-clock time of the first frame, e.g. 2026-10-20T07:00:00")
    args = p.parse_args()

    tables = compute(pd.read_csv(args.tracks), load_config(args.config), args.start)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    passes = tables["passes"]
    print(f"{len(passes)} vehicle tracks entered an approach; "
          f"{passes['exit'].notna().sum()} were seen leaving. Tables written to {out}")


if __name__ == "__main__":
    main()
