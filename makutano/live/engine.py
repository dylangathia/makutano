"""A small microsimulation of the junction, for the live demo.

This is a stand-in for SUMO so the console runs on any laptop: vehicles follow
their leader with a simple safe-speed rule, stop for red and amber, give way to
pedestrians on the zebra crossings, and right-turners wait for a gap in the
opposing flow. Signal state comes from outside (the safety shield), one of
'G', 'A' or 'R' per approach. A TraCI-backed engine with the same methods
(step, snapshot, queue, arrivals) can replace it later.
"""
import math
from dataclasses import dataclass, field

from .layout import APPROACHES, CROSS, HALF, OPPOSITE, OUT, ROUTES, crossing_point, dest_of

ACCEL = 2.2        # m/s²
BRAKE = 7.0        # m/s², used for the safe-speed rule
AMBER_BRAKE = 4.0  # a driver stops on amber if this deceleration is enough
JAM_GAP = 1.0      # bumper-to-bumper gap when stopped, m
START_GAP = 3.0    # a stopped vehicle moves off once its gap reaches this (sets saturation flow)
TURN_SPEED = 5.5   # m/s through the box when turning
QUEUE_SPEED = 1.4  # slower than this counts as queued (matches the video config)
PED_SPEED = (1.1, 1.5)


@dataclass(frozen=True)
class VehicleType:
    name: str
    cls: str        # detector class
    length: float
    width: float
    vmax: float     # m/s
    share: float


TYPES = (
    VehicleType("car", "car", 4.4, 1.8, 13.0, 0.44),
    VehicleType("matatu", "matatu", 5.4, 2.0, 12.5, 0.20),
    VehicleType("boda", "boda_boda", 2.0, 0.8, 14.0, 0.21),
    VehicleType("bus", "bus", 10.5, 2.5, 11.0, 0.05),
    VehicleType("truck", "truck", 8.0, 2.5, 10.0, 0.07),
    VehicleType("tuktuk", "tuk_tuk", 2.7, 1.4, 9.0, 0.03),
)
TURN_SHARE = {"straight": 0.68, "left": 0.20, "right": 0.12}


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

    @property
    def length(self):
        return TYPES[self.type].length

    @property
    def s_out(self):
        return self.s - self.route.box_out


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
        self.arrived = {a: 0 for a in APPROACHES}   # vehicle arrivals since last read
        self.finished = []                          # (exit time, total wait) of vehicles that left
        self.ped_waits = []                         # (start time, wait) of pedestrians who started crossing

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

    def _opposing_conflict(self, approach):
        """True if oncoming straight or left-turning traffic is moving through, or towards, the box.

        Stopped oncoming vehicles don't count: otherwise two right-turners facing each
        other, each with traffic queued behind, would wait for one another for ever."""
        for o in self.inbound[OPPOSITE[approach]]:
            if o.route.turn == "right" or o.v < 1.0:
                continue
            if o.route.s_stop - 25 < o.s < o.route.box_out:
                return True
        return False

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
        self._admit()
        busy = {a: self.crossing_busy(a) for a in APPROACHES}

        for a in APPROACHES:
            lane = self.inbound[a]
            for i, v in enumerate(lane):
                r, vt = v.route, TYPES[v.type]
                limit = math.inf
                if i > 0:
                    lead = lane[i - 1]
                    limit = lead.s - lead.length - JAM_GAP - v.s
                if v.s <= r.s_stop + 0.3:
                    d = r.s_stop - v.s
                    st = heads[a]
                    if st == "R" or busy[a] or (st == "A" and d > v.v * v.v / (2 * AMBER_BRAKE) + 1.0):
                        limit = min(limit, d)
                else:
                    if v.route.turn == "right":
                        hold = r.box_in + 0.4 * (r.box_out - r.box_in)
                        if v.s <= hold + 0.2 and self._opposing_conflict(a):
                            limit = min(limit, hold - v.s)
                    if busy[r.dest]:
                        limit = min(limit, r.box_out + 0.3 - v.s)
                exit_lane = self.outbound[r.dest]
                if exit_lane:
                    last = exit_lane[-1]
                    limit = min(limit, r.box_out + last.s_out - last.length - JAM_GAP - v.s)
                vmax = vt.vmax
                if r.turn != "straight" and r.s_stop < v.s < r.box_out:
                    vmax = min(vmax, TURN_SPEED)
                for inc in self.incidents:
                    if inc.approach == a and inc.s_from < v.s < inc.s_to:
                        vmax = min(vmax, inc.vmax)
                self._move(v, limit, dt, vmax)
            while lane and lane[0].s >= lane[0].route.box_out:
                v = lane.pop(0)
                self.outbound[v.route.dest].append(v)

        for b in APPROACHES:
            lane = self.outbound[b]
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
        for arm in APPROACHES:
            keep = []
            for p in self.peds[arm]:
                if p.p < 0 and ped_ok[arm]:
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
            x, y, h = v.route.at(v.s)
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
