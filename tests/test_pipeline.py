"""Checks on synthetic data, runnable without a GPU, video or SUMO."""
import numpy as np
import pandas as pd
import pytest

from makutano.compare import geh, seed_summary
from makutano.homography import fit_homography, latlon_to_local, leave_one_out_errors, to_ground
from makutano.indicators import compute, drop_riders, points_in_polygon
from makutano.run_sim import parse_queues, parse_tripinfo
from makutano.sumo_demand import build_routes, parse_flow_id

FPS = 10.0


def box_rows(track_id, cls, frames, u, v, w=40, h=30):
    """Boxes whose bottom-centre (foot point) is at (u[i], v[i])."""
    return [dict(frame=f, t_s=f / FPS, track_id=track_id, cls=cls, conf=0.9,
                 x1=uu - w / 2, y1=vv - h, x2=uu + w / 2, y2=vv)
            for f, uu, vv in zip(frames, u, v)]


@pytest.fixture
def cfg():
    # 10 px = 1 m: the image is a scaled ground plan, so expected values are easy.
    corners_px = [[0, 0], [1000, 0], [1000, 1000], [0, 1000], [500, 300]]
    return {
        "homography": {"image_points": corners_px,
                       "ground_points": [[x / 10, y / 10] for x, y in corners_px]},
        "indicators": {"interval_s": 900, "min_track_frames": 5, "smooth_frames": 5,
                       "free_flow_speed_mps": {"default": 10.0},
                       "queue_speed_mps": 1.4, "queue_sample_s": 1.0},
        "study_zone": [[0, 100], [1000, 100], [1000, 900], [0, 900]],
        "approaches": {
            "south": {"entry_zone": [[450, 850], [550, 850], [550, 900], [450, 900]],
                      "exit_zone": [[550, 850], [650, 850], [650, 900], [550, 900]],
                      "queue_zone": [[450, 900], [550, 900], [550, 1000], [450, 1000]],
                      "stop_line": [[450, 900], [550, 900]],
                      "sumo_from_edge": "s_in", "sumo_to_edge": "s_out"},
            "north": {"entry_zone": [[550, 100], [650, 100], [650, 150], [550, 150]],
                      "exit_zone": [[450, 100], [550, 100], [550, 150], [450, 150]],
                      "sumo_from_edge": "n_in", "sumo_to_edge": "n_out"},
        },
        "sumo": {"vtypes": {"private_car": {"vClass": "passenger"}}},
    }


def test_homography_recovers_known_mapping():
    img = np.array([[100, 100], [900, 120], [880, 700], [120, 690], [500, 400], [300, 600]], float)
    H_true = np.array([[0.05, 0.01, -3.0], [0.002, 0.08, -5.0], [0.0001, 0.0008, 1.0]])
    gnd = to_ground(H_true, img)
    H = fit_homography(img, gnd)
    assert np.allclose(to_ground(H, [[640, 360]]), to_ground(H_true, [[640, 360]]), atol=1e-6)
    assert leave_one_out_errors(img, gnd).max() < 1e-3  # under a millimetre


def test_latlon_to_local_scale():
    local = latlon_to_local([[-1.283, 36.82], [-1.282, 36.82]], origin=(-1.283, 36.82))
    assert local[1, 1] == pytest.approx(111.19, abs=0.1)  # 0.001 degrees of latitude
    assert local[1, 0] == pytest.approx(0.0)


def test_points_in_polygon():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    inside = points_in_polygon([[5, 5], [15, 5], [-1, -1], [9.9, 0.1]], square)
    assert inside.tolist() == [True, False, False, True]


def test_drop_riders_keeps_walkers():
    frames = range(10)
    bike = box_rows(1, "boda_boda", frames, [500] * 10, [500] * 10, w=40, h=40)
    rider = box_rows(2, "pedestrian", frames, [500] * 10, [490] * 10, w=30, h=60)
    walker = box_rows(3, "pedestrian", frames, [800] * 10, [500] * 10, w=30, h=60)
    kept = drop_riders(pd.DataFrame(bike + rider + walker))
    assert set(kept["track_id"]) == {1, 3}


def test_indicators_on_synthetic_tracks(cfg):
    # Car 1 drives south -> north at a steady 10 m/s (100 px/s): no delay.
    n = 91
    car = box_rows(1, "private_car", range(n), [500] * n, np.linspace(950, 50, n))
    # Car 2 waits on the south approach, 8 m behind the stop line, for 6 s.
    waiting = box_rows(2, "private_car", range(60), [500] * 60, [980] * 60)
    out = compute(pd.DataFrame(car + waiting), cfg)

    passes = out["passes"].set_index("track_id")
    assert passes.loc[1, "entry"] == "south" and passes.loc[1, "exit"] == "north"
    assert passes.loc[1, "delay_s"] == pytest.approx(0.0, abs=0.3)
    assert 2 not in passes.index  # never reached the entry gate

    speed = out["speeds"].query("cls == 'private_car'")["space_mean_speed_mps"].iat[0]
    assert speed == pytest.approx(10.0, rel=0.05)

    mv = out["movement_volumes"]
    assert mv[["entry", "exit", "count"]].values.tolist() == [["south", "north", 1]]

    q = out["queues"].query("approach == 'south'")
    assert q["max_queue_m"].iat[0] == pytest.approx(8.0, abs=0.1)


def test_routes_round_trip(cfg):
    volumes = pd.DataFrame({"interval_start_s": [0, 900], "entry": ["south", "south"],
                            "exit": ["north", "north"], "cls": ["private_car", "boda_boda"],
                            "count": [12, 5]})
    root = build_routes(volumes, cfg, 900).getroot()
    flows = root.findall("flow")
    assert len(flows) == 1  # boda_boda has no vType in this config
    f = flows[0]
    assert (f.get("from"), f.get("to"), f.get("number")) == ("s_in", "n_out", "12")
    assert parse_flow_id(f.get("id") + ".7") == ("south", "north", "private_car", 0)


def test_parse_sumo_outputs(tmp_path, cfg):
    trip = tmp_path / "trip.xml"
    trip.write_text('<tripinfos><tripinfo id="south__north__private_car__0.3" depart="12.00" '
                    'timeLoss="4.50"/></tripinfos>')
    t = parse_tripinfo(trip)
    assert t.iloc[0].tolist() == ["south", "north", "private_car", 12.0, 4.5]

    queue = tmp_path / "queue.xml"
    queue.write_text('<queue-export><data timestep="2.00"><lanes>'
                     '<lane id="s_in_0" queueing_time="3" queueing_length="7.5"/>'
                     '<lane id="s_in_1" queueing_time="3" queueing_length="12.0"/>'
                     '</lanes></data></queue-export>')
    q = parse_queues(queue, cfg)
    south = q[q["approach"] == "south"].set_index("t_s")["queue_m"]
    assert south.loc[2.0] == 12.0 and south.loc[0.0] == 0.0


def test_geh_and_seed_summary():
    assert geh([100, 150, 0], [100, 100, 0]).tolist() == pytest.approx([0.0, np.sqrt(20), 0.0])
    sim = pd.DataFrame({"k": ["a", "a"], "seed": [0, 1], "count": [10, 14]})
    s = seed_summary(sim, ["k"], "count", n_seeds=3)
    assert s["sim_count"].iat[0] == pytest.approx(8.0)  # seed 2 had no vehicles
