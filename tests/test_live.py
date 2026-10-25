"""Checks for the live console's simulation (no browser or server needed)."""
import pytest

from makutano.live.layout import ROUTES, dest_of
from makutano.live.session import Session


def test_left_hand_traffic_turns():
    # Coming from the north (heading south), a left turn goes east and a right turn west.
    assert dest_of("north", "left") == "east"
    assert dest_of("north", "right") == "west"
    assert dest_of("east", "left") == "south"   # heading west, the left is south
    r = ROUTES[("north", "south")]
    assert r.points[0][0] > 0           # southbound traffic keeps to the east side
    assert r.s_stop < r.box_in < r.box_out < r.length


def _run(controller, scenario="normal", minutes=12, seed=3):
    s = Session(seed=seed)
    s.command({"cmd": "controller", "value": controller})
    s.command({"cmd": "scenario", "value": scenario})
    for _ in range(int(minutes * 600)):
        s.step(0.1)
        heads = s.heads()
        ns = {heads["north"], heads["south"]}
        ew = {heads["east"], heads["west"]}
        assert not ("G" in ns and "G" in ew), "conflicting greens"
        for lane in s.engine.inbound.values():
            for lead, follow in zip(lane, lane[1:]):
                assert lead.s - lead.length - follow.s > -0.05, "vehicles overlap"
    return s


@pytest.mark.parametrize("controller", ["fixed", "webster", "rule", "rl"])
def test_traffic_flows_under_every_controller(controller):
    s = _run(controller)
    f = s.frame(full=True)
    assert f["kpi"]["served"] > 200
    assert f["kpi"]["throughput"] > 1000


def test_adaptive_beats_todays_plan_in_normal_peak():
    fixed = _run("fixed", minutes=15).summary()["delay"]
    rl = _run("rl", minutes=15).summary()["delay"]
    assert rl < fixed


def test_camera_failure_hides_queues_from_controller():
    s = _run("rl", "camfail", minutes=3)
    assert not s.observe()["camera_ok"]
    assert s.frame()["camera_ok"] is False


def test_commands_and_runs_table():
    s = Session(seed=1)
    for _ in range(2400):
        s.step(0.1)
    s.command({"cmd": "controller", "value": "fixed"})
    assert s.runs and s.runs[0]["controller"] == "rl"
    s.command({"cmd": "speed", "value": 100})
    assert s.speed == 20
    s.command({"cmd": "controller", "value": "nonsense"})
    assert s.controller == "fixed"
