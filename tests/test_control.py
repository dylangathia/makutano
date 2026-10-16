"""Module 3 (Control): the parts that run without SUMO, Gymnasium or a GPU."""
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from makutano.control.experiments import clear_time, paired_differences
from makutano.control.rule_based import choose_phase
from makutano.control.scenarios import (
    Scenario, apply_to_routes, movement_shares, sample_training_volumes)
from makutano.control.shield import SafetyShield, ShieldConfig
from makutano.control.webster import webster_plan
from makutano.sim.sumo_demand import build_routes

PHASES = [["north", "south"], ["east", "west"], []]   # the last is pedestrians only
ALL = {"north": True, "south": True, "east": True, "west": True}
NONE = {a: False for a in ALL}


def make_shield(**kw):
    return SafetyShield(PHASES, ShieldConfig(**kw), ped_phase=2)


def run(shield, seconds, demand, ped=False):
    for _ in range(seconds):
        shield.advance(1.0, demand, ped)


def test_minimum_green_is_enforced():
    s = make_shield(min_green_s=10)
    run(s, 4, ALL)
    assert s.request(1, ALL) == 0          # too early: stays on phase 0
    run(s, 6, ALL)
    assert s.request(1, ALL) == 1          # minimum served: switch allowed


def test_switch_runs_amber_then_all_red():
    s = make_shield(min_green_s=0, yellow_s=3, all_red_s=2)
    s.request(1, ALL)
    states = []
    for _ in range(6):
        states.append(s.signal_state())
        s.advance(1.0, ALL)
    assert states == [("yellow", 0)] * 3 + [("all_red", None)] * 2 + [("green", 1)]
    assert s.green_time == pytest.approx(1.0)


def test_no_change_of_mind_during_clearance():
    s = make_shield(min_green_s=0)
    s.request(1, ALL)
    s.advance(1.0, ALL)
    assert s.request(0, ALL) == 1


def test_maximum_green_forces_a_switch_only_if_someone_waits():
    s = make_shield(min_green_s=0, max_green_s=20)
    run(s, 20, NONE)
    assert s.request(0, NONE) == 0         # nobody else waiting: keep green
    demand = {**NONE, "east": True}
    run(s, 1, demand)
    assert s.request(0, demand) == 1       # east is waiting: green must end


def test_starved_approach_is_served():
    s = make_shield(min_green_s=0, max_green_s=1000, max_wait_s=30)
    run(s, 31, ALL)
    assert s.request(0, ALL) == 1          # the agent asks to stay; east/west has waited 31 s


def test_waiting_pedestrians_get_their_phase():
    s = make_shield(min_green_s=0, max_green_s=1000, max_wait_s=1000, ped_max_wait_s=20)
    run(s, 21, ALL, ped=True)
    assert s.request(0, ALL, ped_waiting=True) == 2


def test_rule_based_serves_the_greatest_need():
    queues = {"north": 40.0, "south": 10.0, "east": 20.0, "west": 0.0}
    waits = {a: 0.0 for a in queues}
    assert choose_phase(queues, waits, PHASES, ped_phase=2) == 0
    waits["east"] = 200.0                  # 200 s waiting outweighs 30 m of queue
    assert choose_phase(queues, waits, PHASES, ped_phase=2) == 1
    assert choose_phase(queues, {a: 0.0 for a in queues}, PHASES, ped_phase=2,
                        ped_waiting=15, ped_wait_s=60) == 2


def test_webster_plan():
    # Two phases with flow ratio 0.3 each and 5 s lost per phase:
    # C0 = (1.5 * 10 + 5) / (1 - 0.6) = 50 s, and 40 s of green split evenly.
    cycle, greens = webster_plan([540, 540], [1800, 1800], 5.0, min_green_s=5)
    assert cycle == pytest.approx(50.0)
    assert greens == pytest.approx([20.0, 20.0])
    cycle, _ = webster_plan([1000, 1000], [1800, 1800], 5.0, min_green_s=5, max_cycle_s=120)
    assert cycle == pytest.approx(120.0)   # oversaturated: longest allowed cycle


@pytest.fixture
def cfg():
    return {
        "approaches": {"north": {"sumo_from_edge": "n_in", "sumo_to_edge": "n_out"},
                       "south": {"sumo_from_edge": "s_in", "sumo_to_edge": "s_out"}},
        "sumo": {"vtypes": {"private_car": {"vClass": "passenger"}, "boda_boda": {"vClass": "motorcycle"}}},
    }


def test_scenarios_scale_and_surge_routes(tmp_path, cfg):
    vols = pd.DataFrame({"interval_start_s": [0, 900], "entry": ["north", "south"], "exit": ["south", "north"],
                         "cls": ["private_car", "private_car"], "count": [100, 50]})
    base = tmp_path / "base.rou.xml"
    build_routes(vols, cfg, 900).write(base)

    out = apply_to_routes(base, Scenario("heavy", demand_scale=1.5), cfg, tmp_path / "heavy.rou.xml")
    numbers = sorted(int(f.get("number")) for f in ET.parse(out).getroot().iter("flow"))
    assert numbers == [75, 150]

    surge = Scenario("surge", surge={"approach": "north", "start_s": 300, "duration_s": 300, "factor": 2.0})
    flows = list(ET.parse(apply_to_routes(base, surge, cfg, tmp_path / "s.rou.xml")).getroot().iter("flow"))
    extra = [f for f in flows if f.get("id").endswith("_surge")]
    assert len(extra) == 1 and extra[0].get("number") == "33"   # 100 over 900 s -> 33 extra in 300 s
    begins = [float(f.get("begin")) for f in flows]
    assert begins == sorted(begins)


def test_camera_outage_window():
    s = Scenario("cam", camera_outage={"start_s": 100, "duration_s": 50})
    assert not s.camera_out(99) and s.camera_out(100) and s.camera_out(149) and not s.camera_out(150)
    assert s.disruption == (100, 150)


def test_training_volumes_follow_the_profile():
    profile = pd.DataFrame({"approach": ["north"] * 4, "tod_bin": [420, 435, 450, 465],
                            "mean": [100.0] * 4, "std": [0.0] * 4, "days": [3] * 4})
    vols = pd.DataFrame({"interval_start_s": [0, 0], "entry": ["north", "north"], "exit": ["south", "east"],
                         "cls": ["private_car", "private_car"], "count": [75, 25]})
    shares = movement_shares(vols)
    rng = np.random.default_rng(0)
    totals = [sample_training_volumes(profile, shares, 420, 480, rng, day_scale_sd=0)["count"].sum()
              for _ in range(50)]
    assert np.mean(totals) == pytest.approx(400, rel=0.05)   # 4 bins of ~100 arrivals


def test_clear_time():
    t = np.arange(0, 400)
    queue = np.where((t >= 100) & (t < 200), 50.0, 5.0)
    queue[200:250] = 30.0                  # queue lingers 50 s after the disruption ends
    assert clear_time(t, queue, (100, 200), before_s=100, window_s=1) == pytest.approx(50.0)
    assert np.isnan(clear_time(t, queue, None))


def test_paired_differences():
    rows = []
    for seed in range(5):
        rows += [{"scenario": "base", "strategy": "today", "seed": seed, "mean_delay_s": 60.0 + seed},
                 {"scenario": "base", "strategy": "rl", "seed": seed, "mean_delay_s": 50.0 + seed}]
    df = pd.DataFrame(rows).assign(throughput_veh_h=1.0, mean_queue_m=1.0, ped_wait_s_per_person=1.0,
                                   clear_time_s=np.nan)
    d = paired_differences(df)
    r = d[(d["metric"] == "mean_delay_s") & (d["strategy"] == "rl") & (d["minus"] == "today")].iloc[0]
    # The same 10 s saving on every seed: exact difference, zero spread.
    assert r["mean_difference"] == pytest.approx(-10.0) and r["ci95"] == pytest.approx(0.0)
    assert r["n_seeds"] == 5
