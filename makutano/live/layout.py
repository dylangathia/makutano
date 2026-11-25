"""Geometry of the live junction: arms, lanes, stop lines, crossings and routes.

Coordinates are metres with the junction centre at (0, 0), x to the east and y
to the south (the same orientation as the camera image, so the frontend can
draw them directly). Traffic keeps left. Each approach has one inbound and one
outbound lane; every vehicle follows a route polyline from where it enters the
camera view, through the junction box, to where it leaves the view.
"""
import math
from dataclasses import dataclass

APPROACHES = ("north", "south", "east", "west")
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
OUT = {"north": (0.0, -1.0), "south": (0.0, 1.0), "east": (1.0, 0.0), "west": (-1.0, 0.0)}

VIEW_W, VIEW_H = 222.0, 144.0  # camera footprint after the homography, metres
HALF = 6.6                     # road half-width: one 3.3 m lane each way
LANE = 3.3                     # lane centre offset from the centre line
STOP = 10.5                    # stop line distance from the junction centre
CROSS = (HALF + 0.6, HALF + 3.4)  # zebra crossing band, distance from centre
OFFSCREEN = 15.0               # routes start and end this far outside the view


def _left(t):
    """Unit vector to the driver's left for travel direction t (y points south)."""
    return (t[1], -t[0])


def _add(p, q, k=1.0):
    return (p[0] + q[0] * k, p[1] + q[1] * k)


def _reach(arm):
    return (VIEW_H if OUT[arm][0] == 0 else VIEW_W) / 2 + OFFSCREEN


def inbound_line(arm):
    """(start, box entry) of the lane arriving from `arm`."""
    t = (-OUT[arm][0], -OUT[arm][1])
    off = _add((0.0, 0.0), _left(t), LANE)
    return _add(off, OUT[arm], _reach(arm)), _add(off, OUT[arm], HALF)


def outbound_line(arm):
    """(box exit, end) of the lane leaving towards `arm`."""
    off = _add((0.0, 0.0), _left(OUT[arm]), LANE)
    return _add(off, OUT[arm], HALF), _add(off, OUT[arm], _reach(arm))


def turn_of(origin, dest):
    if dest == OPPOSITE[origin]:
        return "straight"
    t_in = (-OUT[origin][0], -OUT[origin][1])
    return "left" if OUT[dest] == _left(t_in) else "right"


def dest_of(origin, turn):
    for d in APPROACHES:
        if d != origin and turn_of(origin, d) == turn:
            return d
    raise ValueError(turn)


@dataclass
class Route:
    origin: str
    dest: str
    turn: str
    points: list      # polyline vertices
    cum: list         # cumulative length at each vertex
    s_stop: float     # front bumper position at the stop line
    box_in: float     # where the route enters the junction box
    box_out: float    # where it leaves the box onto the outbound lane
    length: float

    def at(self, s):
        """(x, y, heading in radians) at distance s along the route."""
        s = min(max(s, 0.0), self.length)
        lo, hi = 0, len(self.cum) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.cum[mid] <= s:
                lo = mid
            else:
                hi = mid
        (x0, y0), (x1, y1) = self.points[lo], self.points[hi]
        seg = self.cum[hi] - self.cum[lo] or 1e-9
        u = (s - self.cum[lo]) / seg
        return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, math.atan2(y1 - y0, x1 - x0)


def _bezier(p0, p1, p2, p3, n=14):
    pts = []
    for i in range(1, n):
        u = i / n
        a, b, c, d = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return pts


def build_route(origin, dest):
    start, entry = inbound_line(origin)
    exit_, end = outbound_line(dest)
    turn = turn_of(origin, dest)
    pts = [start, entry]
    if turn != "straight":
        t_in = (-OUT[origin][0], -OUT[origin][1])
        r = math.dist(entry, exit_) / math.sqrt(2)
        k = 0.5523 * r
        pts += _bezier(entry, _add(entry, t_in, k), _add(exit_, OUT[dest], -k), exit_)
    pts += [exit_, end]
    cum = [0.0]
    for p, q in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.dist(p, q))
    box_in = math.dist(start, entry)
    box_out = cum[pts.index(exit_)]
    return Route(origin, dest, turn, pts, cum, box_in - (STOP - HALF), box_in, box_out, cum[-1])


ROUTES = {(o, d): build_route(o, d) for o in APPROACHES for d in APPROACHES if o != d}

CLEARANCE = 2.9      # centre lines closer than this mean two vehicles there would touch
ZONE_PAST_BOX = 8.0  # zones reach this far onto the exit lane, to cover merges


def _conflict_zone(r1, r2, step=0.4):
    """Where routes r1 and r2 come close enough to collide, as (a0, a1) along r1 and
    (b0, b1) along r2; None if they never do. Only the junction box and the start of
    the exit lane are searched: before the stop line every lane is separate."""
    def samples(r):
        n = int((r.box_out + ZONE_PAST_BOX - r.s_stop) / step) + 1
        return [(r.s_stop + i * step, r.at(r.s_stop + i * step)) for i in range(n)]
    p1, p2 = samples(r1), samples(r2)
    s1 = [s for s, (x, y, _) in p1 if any(math.hypot(x - u, y - w) < CLEARANCE for _, (u, w, _) in p2)]
    if not s1:
        return None
    s2 = [s for s, (x, y, _) in p2 if any(math.hypot(x - u, y - w) < CLEARANCE for _, (u, w, _) in p1)]
    return min(s1), max(s1), min(s2), max(s2)


# ZONES[(origin1, dest1, origin2, dest2)] = (a0, a1, b0, b1) for every pair of different
# routes that can collide.
ZONES = {}
for _k1, _r1 in ROUTES.items():
    for _k2, _r2 in ROUTES.items():
        if _k1 < _k2:
            _z = _conflict_zone(_r1, _r2)
            if _z:
                ZONES[_k1 + _k2] = _z
                ZONES[_k2 + _k1] = (_z[2], _z[3], _z[0], _z[1])


def zone(r1, r2):
    return ZONES.get((r1.origin, r1.dest, r2.origin, r2.dest))


def _merge(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


# BOX_SPANS[(origin, dest)]: the stretches of a route that cross or merge with traffic from
# other approaches. A vehicle must not come to rest in one ("don't block the box").
BOX_SPANS = {k: _merge([z[:2] for kk, z in ZONES.items() if kk[:2] == k and kk[2] != k[0]]) for k in ROUTES}


def crossing_point(arm, along):
    """A point on the zebra crossing of `arm`; `along` runs across the road."""
    mid = (CROSS[0] + CROSS[1]) / 2
    side = _left(OUT[arm])
    return _add(_add((0.0, 0.0), OUT[arm], mid), side, along)


def describe():
    """Everything the frontend needs to draw the junction."""
    return {
        "view": [VIEW_W, VIEW_H], "half": HALF, "lane": LANE, "stop": STOP, "cross": list(CROSS),
        "approaches": list(APPROACHES),
        "inbound": {a: [list(p) for p in inbound_line(a)] for a in APPROACHES},
        "outbound": {a: [list(p) for p in outbound_line(a)] for a in APPROACHES},
        "out": {a: list(OUT[a]) for a in APPROACHES},
    }
