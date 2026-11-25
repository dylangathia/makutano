"""A small microsimulation of the junction, for the live demo.

This is a stand-in for SUMO so the console runs on any laptop: vehicles follow
their leader with a simple safe-speed rule, stop for red and amber, give way to
pedestrians on the zebra crossings, and never enter a part of the junction box
another vehicle is using or has the right of way through (first in goes first;
right-turners wait for a gap in the oncoming flow). Signal state comes from outside (the safety shield), one of
'G', 'A' or 'R' per approach. A TraCI-backed engine with the same methods
(step, snapshot, queue, arrivals) can replace it later.
"""
import math
from dataclasses import dataclass, field

from ..classes import BODA_BODA, HANDCART, MATATU, PRIVATE_CAR, TRUCK_BUS
from .layout import (APPROACHES, BOX_SPANS, CROSS, HALF, OPPOSITE, OUT, ROUTES, ZONE_PAST_BOX,
                     crossing_point, dest_of, zone)

ACCEL = 2.2        # m/s²
BRAKE = 7.0        # m/s², used for the safe-speed rule
AMBER_BRAKE = 4.0  # a driver stops on amber if this deceleration is enough
JAM_GAP = 1.0      # bumper-to-bumper gap when stopped, m
START_GAP = 3.0    # a stopped vehicle moves off once its gap reaches this (sets saturation flow)
TURN_SPEED = 5.5   # m/s through the box when turning
QUEUE_SPEED = 1.4  # slower than this counts as queued (matches the video config)
PED_SPEED = (1.1, 1.5)
ZONE_GAP = 0.8     # where a vehicle waits short of a conflict zone, m
MOVING = 2.0       # a vehicle ahead going at least this fast can be followed into the box, m/s
TURN_GAP = 1.5     # extra gap behind a vehicle that is turning: its rear corner swings out, m
LOOK_BACK = 40.0   # vehicles this close to the stop line take part in conflict checks
STUCK_YIELD = 8.0  # stopped this long inside the box: stop giving way to claims (breaks deadlocks)
STUCK_GO = 40.0    # stopped this long inside the box: removed, as SUMO teleports gridlocked vehicles
GAP_S = 2.0        # a right-turner goes if oncoming traffic is further away than this, s


@dataclass(frozen=True)
class VehicleType:
    name: str       # how it is drawn
    cls: str        # detector class (makutano/classes.py)
    length: float
    width: float
    vmax: float     # m/s
    share: float


TYPES = (
    VehicleType("car", PRIVATE_CAR, 4.4, 1.8, 13.0, 0.455),
    VehicleType("matatu", MATATU, 5.4, 2.0, 12.5, 0.20),
    VehicleType("boda", BODA_BODA, 2.0, 0.8, 14.0, 0.21),
    VehicleType("bus", TRUCK_BUS, 10.5, 2.5, 11.0, 0.05),
    VehicleType("truck", TRUCK_BUS, 8.0, 2.5, 10.0, 0.08),
    VehicleType("handcart", HANDCART, 2.4, 1.2, 2.0, 0.005),
)
TURN_SHARE = {"straight": 0.72, "left": 0.20, "right": 0.08}


@dataclass(eq=False)
class Vehicle:
    id: int
    type: int
    route: object
    born: float          # when it joined the queue, including time beyond the view
    s: float = 0.0
    v: float = 0.0
    wait: float = 0.0    # total time queued, s
    stopped_for: float = 0.0
    paint: int = 0
    entered: float = None  # when the front crossed the stop line

    @property
    def length(self):
        return TYPES[self.type].length

    def gap_behind(self):
        """Gap a follower on the same path keeps behind this vehicle's rear."""
        r = self.route
        if r.turn != "straight" and self.s > r.box_in:
            return JAM_GAP + TURN_GAP
        return JAM_GAP

    @property
    def s_out(self):
        return self.s - self.route.box_out

    def pose(self):
        """(x, y, heading) of the front, with the body lying between the front and rear
        points on the path, as a vehicle's axles do, so long vehicles stay in lane on turns."""
        x, y, h = self.route.at(self.s)
        rx, ry, _ = self.route.at(self.s - self.length)
        if math.hypot(x - rx, y - ry) > 0.5:
            h = math.atan2(y - ry, x - rx)
        return x, y, h


@dataclass(eq=False)
class Pedestrian:
    t0: float
    side: int            # which kerb they start from
    off: float           # where on the kerb they stand, 0..1
    p: float = -1.0      # crossing progress 0..1, or -1 while waiting
    speed: float = 1.3


@dataclass
class Incident:
    """Part of an approach where vehicles must slow down (a stalled truck)."""
    approach: str
    s_from: float
    s_to: float
    vmax: float = 2.5


@dataclass
class Junction:
    rng: object
    t: float = 0.0
    inbound: dict = field(default_factory=lambda: {a: [] for a in APPROACHES})
    outbound: dict = field(default_factory=lambda: {a: [] for a in APPROACHES})
    backlog: dict = field(default_factory=lambda: {a: [] for a in APPROACHES})
    peds: dict = field(default_factory=lambda: {a: [] for a in APPROACHES})
    incidents: list = field(default_factory=list)
    next_id: int = 1

    def __post_init__(self):
        self.heads = {}
        self.arrived = {a: 0 for a in APPROACHES}   # vehicle arrivals since last read
        self.finished = []                          # (exit time, total wait) of vehicles that left
        self.ped_waits = []                         # (start time, wait) of pedestrians who started crossing
        self.removed = []                           # (time, vehicle id) taken out of a gridlock

    # ---------------- demand ----------------
    def add_arrival(self, approach):
        self.backlog[approach].append(self.t)
        self.arrived[approach] += 1

    def add_pedestrian(self, arm):
        r = self.rng
        self.peds[arm].append(Pedestrian(self.t, int(r.random() < 0.5), r.random(),
                                         speed=r.uniform(*PED_SPEED)))

    def take_arrivals(self):
        out, self.arrived = self.arrived, {a: 0 for a in APPROACHES}
        return out

    def _new_vehicle(self, approach, born):
        r = self.rng
        k = r.choices(range(len(TYPES)), weights=[t.share for t in TYPES])[0]
        turn = r.choices(list(TURN_SHARE), weights=list(TURN_SHARE.values()))[0]
        v = Vehicle(self.next_id, k, ROUTES[(approach, dest_of(approach, turn))], born,
                    paint=r.randrange(8))
        v.wait = self.t - born
        self.next_id += 1
        return v

    def _admit(self):
        for a in APPROACHES:
            lane = self.inbound[a]
            if not self.backlog[a]:
                continue
            last = lane[-1] if lane else None
            if last is None or last.s - last.length > 8.0:
                v = self._new_vehicle(a, self.backlog[a].pop(0))
                v.v = min(TYPES[v.type].vmax, last.v) if last else TYPES[v.type].vmax * 0.8
                lane.append(v)

    # ---------------- helpers ----------------
    def crossing_busy(self, arm):
        return any(p.p >= 0 for p in self.peds[arm])

    def crossing_clear(self, arm):
        """No vehicle that has passed its stop line still has to drive over this arm's zebra."""
        beyond = CROSS[1] - HALF + 0.5
        for v in self.vehicles():
            if v.entered is None:
                continue
            r = v.route
            if r.origin == arm and v.s - v.length < r.box_in:
                return False
            if r.dest == arm and v.s - v.length < r.box_out + beyond:
                return False
        return True

    def _has_way(self, u, v):
        """True if u may go through the zone it shares with v before v does."""
        ru, rv = u.route, v.route
        if u.stopped_for > 1.0:
            return False  # a vehicle that is itself held up claims nothing (no stand-offs)
        if u.entered is None:
            # Before its stop line only oncoming traffic about to enter on green counts, and
            # only against a right-turner, who must wait for a gap.
            return (rv.turn == "right" and ru.origin == OPPOSITE[rv.origin] and ru.turn != "right"
                    and self.heads.get(ru.origin) == "G" and u.v > 2.0 and ru.s_stop - u.s < u.v * GAP_S)
        if v.entered is None:
            return True
        if rv.turn == "right" and ru.origin == OPPOSITE[rv.origin] and ru.turn != "right":
            return True
        if ru.turn == "right" and rv.origin == OPPOSITE[ru.origin] and rv.turn != "right":
            return False
        return (u.entered, u.id) < (v.entered, v.id)

    def _conflict_limit(self, v, nearby, recent):
        """How far v may go before it would run into another vehicle's path in the box, as
        (limit, hard): hard counts only obstacles that are not moving away."""
        r = v.route
        limit = hard = math.inf
        if r.s_stop - LOOK_BACK > v.s or v.s - v.length > r.box_out + ZONE_PAST_BOX:
            return limit, hard
        stuck = v.stopped_for if v.entered is not None else 0.0
        # Vehicles from the same approach that have already left this lane: keep following
        # them until they are clear of the shared part of the path.
        for u in recent[r.origin]:
            z = None if u.route is r else zone(r, u.route)
            if u.route is r or (z and u.s - u.length < z[1] + 0.5):
                gap = u.s - u.length - u.gap_behind() - v.s
                limit = min(limit, gap)
                if u.v < MOVING:
                    hard = min(hard, gap)
        for u in nearby:
            if u.route.origin == r.origin:
                continue
            z = zone(r, u.route)
            if not z:
                continue
            a0, a1, b0, b1 = z
            if v.s >= a0 - 0.1 or v.s - v.length > a1:
                continue  # already in the zone (must clear it) or past it
            # In the zone, or moving too fast to stop before it: either way it has the space.
            committed = u.s > b0 or u.s + u.v * u.v / (2 * BRAKE) + 0.5 > b0
            occupied = committed and u.s - u.length < b1 + 0.3
            if occupied:
                limit = min(limit, a0 - ZONE_GAP - v.s)
                if u.v < MOVING:
                    hard = min(hard, a0 - ZONE_GAP - v.s)
            elif u.s <= b0 and stuck <= STUCK_YIELD and self._has_way(u, v):
                # Someone with the right of way is coming: wait at the edge of the shared
                # stretch (a right-turner waits in the middle of the junction for a gap).
                limit = min(limit, a0 - ZONE_GAP - v.s)
        return limit, hard

    @staticmethod
    def _keep_box_clear(v, hard):
        """If v would come to rest inside a stretch it shares with other approaches, stop it
        short of that stretch instead, so a waiting vehicle never blocks crossing traffic."""
        if hard == math.inf:
            return math.inf
        rest = v.s + hard
        for c0, c1 in BOX_SPANS[(v.route.origin, v.route.dest)]:
            if v.s >= c0 - 0.1:
                continue  # already in it: it has to clear it
            if rest > c0 and rest - v.length < c1:
                return c0 - ZONE_GAP - v.s
        return math.inf

    def _move(self, v, limit, dt, vmax):
        if v.v < 0.3 and limit < START_GAP - JAM_GAP:
            v.v = 0.0
        else:
            safe = math.sqrt(max(0.0, 2 * BRAKE * (limit - 0.3)))
            target = min(vmax, safe)
            v.v = min(target, v.v + ACCEL * dt) if target > v.v else target
            v.s += v.v * dt
        if v.v < QUEUE_SPEED:
            v.wait += dt
            v.stopped_for += dt
        else:
            v.stopped_for = 0.0

    # ---------------- step ----------------
    def step(self, dt, heads, ped_ok):
        """heads: approach -> 'G' | 'A' | 'R'; ped_ok: arm -> pedestrians may start crossing."""
        self.t += dt
        self.heads = heads
        self._admit()
        busy = {a: self.crossing_busy(a) for a in APPROACHES}
        nearby = [u for u in self.vehicles()
                  if u.route.s_stop - LOOK_BACK < u.s and u.s - u.length < u.route.box_out + ZONE_PAST_BOX]
        recent = {a: [] for a in APPROACHES}
        for lane in self.outbound.values():
            for u in lane:
                if u.s - u.length < u.route.box_out + ZONE_PAST_BOX:
                    recent[u.route.origin].append(u)

        for a in APPROACHES:
            lane = self.inbound[a]
            for i, v in enumerate(lane):
                r, vt = v.route, TYPES[v.type]
                limit = hard = math.inf     # hard: limits set by things that are not moving away
                if i > 0:
                    lead = lane[i - 1]
                    limit = lead.s - lead.length - lead.gap_behind() - v.s
                    if lead.v < MOVING:
                        hard = limit
                c_limit, c_hard = self._conflict_limit(v, nearby, recent)
                limit, hard = min(limit, c_limit), min(hard, c_hard)
                if v.s <= r.s_stop + 0.3:
                    d = r.s_stop - v.s
                    st = heads[a]
                    if st == "R" or busy[a] or (st == "A" and d > v.v * v.v / (2 * AMBER_BRAKE) + 1.0):
                        limit, hard = min(limit, d), min(hard, d)
                elif busy[r.dest]:
                    limit = hard = min(limit, r.box_out + 0.3 - v.s)
                exit_lane = self.outbound[r.dest]
                if exit_lane:
                    last = exit_lane[-1]
                    gap = r.box_out + last.s_out - last.length - JAM_GAP - v.s
                    limit = min(limit, gap)
                    if last.v < MOVING:
                        hard = min(hard, gap)
                limit = min(limit, self._keep_box_clear(v, hard))
                vmax = vt.vmax
                if r.turn != "straight" and r.s_stop < v.s < r.box_out:
                    vmax = min(vmax, TURN_SPEED)
                for inc in self.incidents:
                    if inc.approach == a and inc.s_from < v.s < inc.s_to:
                        vmax = min(vmax, inc.vmax)
                self._move(v, limit, dt, vmax)
                if v.entered is None and v.s > r.s_stop + 0.5:  # clearly past, not a small overshoot at red
                    v.entered = self.t
            for v in [v for v in lane if v.s > v.route.box_in and v.stopped_for > STUCK_GO]:
                lane.remove(v)
                self.finished.append((self.t, v.wait))
                self.removed.append((self.t, v.id))
            while lane and lane[0].s >= lane[0].route.box_out:
                v = lane.pop(0)
                self.outbound[v.route.dest].append(v)

        for b in APPROACHES:
            lane = self.outbound[b]
            lane.sort(key=lambda u: -u.s_out)  # merges can join ahead of a vehicle already on the lane
            for i, v in enumerate(lane):
                limit = math.inf
                if i > 0:
                    lead = lane[i - 1]
                    limit = lead.s_out - lead.length - JAM_GAP - v.s_out
                self._move(v, limit, dt, TYPES[v.type].vmax)
            while lane and lane[0].s >= lane[0].route.length + lane[0].length:
                v = lane.pop(0)
                self.finished.append((self.t, v.wait))

        span = 2 * HALF + 3.0
        clear = {arm: self.crossing_clear(arm) for arm in APPROACHES if self.peds[arm]}
        for arm in APPROACHES:
            keep = []
            for p in self.peds[arm]:
                if p.p < 0 and ped_ok[arm] and clear[arm]:
                    p.p = 0.0
                    self.ped_waits.append((self.t, self.t - p.t0))
                if p.p >= 0:
                    p.p += p.speed * dt / span
                if p.p < 1.0:
                    keep.append(p)
            self.peds[arm] = keep

    # ---------------- measurement ----------------
    def queue(self, approach):
        """Vehicles queued on the approach, including those beyond the camera view."""
        n = len(self.backlog[approach])
        metres = n * 6.0
        for v in self.inbound[approach]:
            if v.s <= v.route.s_stop + 0.5 and v.v < QUEUE_SPEED:
                n += 1
                metres += v.length + JAM_GAP
        return n, metres

    def longest_wait(self, approach):
        w = self.t - self.backlog[approach][0] if self.backlog[approach] else 0.0
        for v in self.inbound[approach]:
            if v.s <= v.route.s_stop + 0.5 and v.v < QUEUE_SPEED:
                w = max(w, v.stopped_for)
        return w

    def approaching(self, approach):
        """True if any vehicle is queued or on its way to the stop line."""
        return bool(self.backlog[approach]) or any(v.s < v.route.s_stop for v in self.inbound[approach])

    def ped_waiting(self, arm):
        w = [self.t - p.t0 for p in self.peds[arm] if p.p < 0]
        return len(w), max(w, default=0.0)

    def vehicles(self):
        for lanes in (self.inbound, self.outbound):
            for lane in lanes.values():
                yield from lane

    def snapshot(self):
        cars = []
        for v in self.vehicles():
            x, y, h = v.pose()
            cars.append([v.id, round(x, 2), round(y, 2), round(h, 3), v.type, v.paint,
                         1 if v.v < QUEUE_SPEED else 0])
        peds = []
        span = HALF + 1.5
        for arm in APPROACHES:
            for p in self.peds[arm]:
                sign = 1 if p.side else -1
                along = sign * (span + 0.6 + p.off * 1.6) if p.p < 0 else sign * span * (1 - 2 * p.p)
                x, y = crossing_point(arm, along)
                jitter = (p.off - 0.5) * 1.4
                x, y = x + OUT[arm][0] * jitter, y + OUT[arm][1] * jitter
                peds.append([round(x, 2), round(y, 2), 1 if p.p >= 0 else 0])
        return cars, peds


__all__ = ["Junction", "Incident", "TYPES", "CROSS"]
