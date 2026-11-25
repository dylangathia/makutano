"""Checks for the live console's simulation (no browser or server needed)."""
import math

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


def _footprint(v, shrink=0.15):
    """Corners of a vehicle's body, slightly shrunk so touching bumpers don't count."""
    from makutano.live.engine import TYPES
    x, y, h = v.pose()
    t = TYPES[v.type]
    length, half = t.length - 2 * shrink, t.width / 2 - shrink
    c, s = math.cos(h), math.sin(h)
    return [(x + u * c - w * s, y + u * s + w * c)
            for u, w in ((-shrink, -half), (-shrink, half), (-shrink - length, half), (-shrink - length, -half))]


def _overlap(a, b):
    for poly in (a, b):
        for i in range(4):
            (px, py), (qx, qy) = poly[i], poly[(i + 1) % 4]
            nx, ny = qy - py, px - qx
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def _collisions(s):
    vs = [v for v in s.engine.vehicles() if 0 < v.s <= v.route.length]
    fps = [_footprint(v) for v in vs]
    hits = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            if abs(fps[i][0][0] - fps[j][0][0]) < 14 and abs(fps[i][0][1] - fps[j][0][1]) < 14                     and _overlap(fps[i], fps[j]):
                hits.append((vs[i].id, vs[j].id))
    return hits


@pytest.mark.parametrize("controller,scenario", [("rl", "normal"), ("fixed", "heavy"), ("rule", "peds")])
def test_vehicles_never_overlap(controller, scenario):
    s = Session(seed=11)
    s.command({"cmd": "controller", "value": controller})
    s.command({"cmd": "scenario", "value": scenario})
    for k in range(6000):
        s.step(0.1)
        if k % 5 == 0:
            assert not _collisions(s), f"vehicles overlap at {s.frame()['clock']}"


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
