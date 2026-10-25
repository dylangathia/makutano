"""One live run: demand, forecasts, the controller, the safety shield and the engine.

The session owns the clock. Each 0.1 s step it draws arrivals from the
time-of-day profile (changed by the active scenario), lets the controller ask
for a phase every decision interval, passes that request through the
SafetyShield, and moves the traffic. frame() packs the state the console
draws; command() applies what the operator clicks.
"""
import math
import random
from collections import deque

from ..control.shield import SafetyShield, ShieldConfig
from .controllers import PHASES, build_controllers
from .engine import QUEUE_SPEED, TYPES, Incident, Junction
from .layout import APPROACHES, ROUTES

STEP = 0.1
START = 7 * 3600 + 20 * 60          # 07:20, early in the morning peak
WARMUP = 240.0                      # seconds simulated before the first frame
DECISION_S = 5.0                    # the agent decides every 5 s
BASE = {"north": 400, "south": 460, "east": 640, "west": 580}   # veh/h near the peak
PED_BASE = 120                      # pedestrians/h per crossing

SCENARIOS = {
    "normal": {"label": "Normal peak"},
    "heavy": {"label": "Heavy demand +40%", "demand_scale": 1.4},
    "surge": {"label": "Surge from west", "surge": ("west", 2.1)},
    "blocked": {"label": "Blocked lane (east)", "incident": ("east", 44.0, 60.0)},
    "peds": {"label": "Pedestrian surge", "ped_scale": 5.0},
    "camfail": {"label": "Camera failure", "camera": False},
}


def peak_shape(t):
    """Demand relative to BASE at clock time t (seconds of the day)."""
    h = t / 3600
    return 0.62 + 0.45 * math.exp(-(((h - 7.75) / 0.65) ** 2))


def clock(t):
    t = int(t)
    return f"{t // 3600 % 24:02d}:{t // 60 % 60:02d}:{t % 60:02d}"


class Forecaster:
    """Stand-in for the GRU: the time-of-day profile scaled by how the last
    five minutes compared with it. With the camera down it keeps its last scale."""

    def __init__(self, seed_t, rng):
        self.hist = {a: deque(maxlen=30) for a in APPROACHES}
        self.scale = {a: 1.0 for a in APPROACHES}
        for a in APPROACHES:
            for i in range(15, 0, -1):
                tm = seed_t - i * 60
                self.hist[a].append((tm, _poisson(profile_rate(a, tm) * 60, rng)))

    def record_minute(self, t, counts, camera_ok):
        for a in APPROACHES:
            self.hist[a].append((t, counts[a]))
            if camera_ok:
                recent = list(self.hist[a])[-5:]
                seen = sum(n for _, n in recent)
                expected = sum(profile_rate(a, tm) * 60 for tm, _ in recent) or 1.0
                self.scale[a] = min(2.5, max(0.5, seen / expected))

    def rate(self, a, t):
        return profile_rate(a, t) * self.scale[a]

    def count(self, a, t, minutes):
        return sum(self.rate(a, t + i * 60 + 30) * 60 for i in range(minutes))

    def series(self, a, t):
        past = [n for _, n in list(self.hist[a])[-15:]]
        nxt = [self.rate(a, t + i * 60 + 30) * 60 for i in range(15)]
        return past, nxt


def profile_rate(a, t):
    return BASE[a] * peak_shape(t) / 3600


def _poisson(lam, rng):
    k, p, limit = 0, 1.0, math.exp(-lam)
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


class Session:
    def __init__(self, seed=None):
        self.controllers = build_controllers()
        self.controller = "rl"
        self.scenario = "normal"
        self.speed = 3.0
        self.paused = False
        self.runs = []
        self.reset(seed)

    # ---------------- lifecycle ----------------
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        self.engine = Junction(self.rng, t=START - WARMUP)
        self.shield = SafetyShield([set(p) for p in PHASES], ShieldConfig(
            min_green_s=10, max_green_s=60, yellow_s=3, all_red_s=2, max_wait_s=120))
        self.forecaster = Forecaster(self.engine.t, self.rng)
        self.minute = int(self.engine.t // 60)
        self.next_decision = DECISION_S
        self.decision = {"wanted": 0, "target": 0, "p": 0.0, "why": "", "reason": None}
        self.log = deque(maxlen=60)
        self._apply_scenario()
        self._acc = 0.0
        while self.engine.t < START:
            self.step(STEP)
        self.run_t0 = self.engine.t
        self.log.clear()
        self.note(f"Run started: {self.ctrl.label}, {SCENARIOS[self.scenario]['label'].lower()}")

    @property
    def ctrl(self):
        return self.controllers[self.controller]

    @property
    def camera_ok(self):
        return SCENARIOS[self.scenario].get("camera", True)

    def _apply_scenario(self):
        sc = SCENARIOS[self.scenario]
        inc = sc.get("incident")
        self.engine.incidents = [Incident(*inc)] if inc else []

    def note(self, text, kind="ops"):
        self.log.appendleft((self.engine.t, kind, text))

    def new_run(self, summary, reason):
        """Keep the run that just ended (summary taken before the change) and start a new one."""
        if summary["minutes"] >= 3 and summary["delay"] is not None:
            self.runs = [r for r in self.runs if (r["controller"], r["scenario"]) != (summary["controller"], summary["scenario"])]
            self.runs.insert(0, summary)
            del self.runs[6:]
        self.run_t0 = self.engine.t
        self.note(reason)

    def command(self, msg):
        cmd, val = msg.get("cmd"), msg.get("value")
        ended = self.summary()
        if cmd == "controller" and val in self.controllers and val != self.controller:
            self.controller = val
            self.next_decision = max(self.shield.green_time + 1, DECISION_S)
            self.new_run(ended, f"Controller set to {self.ctrl.label}")
        elif cmd == "scenario" and val in SCENARIOS and val != self.scenario:
            self.scenario = val
            self._apply_scenario()
            self.new_run(ended, f"Scenario: {SCENARIOS[val]['label'].lower()}")
        elif cmd == "speed":
            self.speed = float(min(20, max(0.25, float(val))))
        elif cmd == "pause":
            self.paused = bool(val)
        elif cmd == "reset":
            self.reset()

    # ---------------- simulation ----------------
    def rate(self, a, t):
        sc = SCENARIOS[self.scenario]
        r = profile_rate(a, t) * sc.get("demand_scale", 1.0)
        surge = sc.get("surge")
        if surge and surge[0] == a:
            r *= surge[1]
        return r

    def heads(self):
        state, ph = self.shield.signal_state()
        out = {a: "R" for a in APPROACHES}
        if state in ("green", "yellow"):
            for a in PHASES[ph]:
                out[a] = "G" if state == "green" else "A"
        return out

    def observe(self):
        eng, sh = self.engine, self.shield
        obs = {"phase": sh.phase, "green_time": sh.green_time, "camera_ok": self.camera_ok,
               "waits": dict(sh.wait), "queue": {}, "queue_m": {}, "moving": {},
               "arrivals_20s": {a: self.forecaster.rate(a, eng.t) * 20 for a in APPROACHES}}
        for a in APPROACHES:
            if self.camera_ok:
                n, m = eng.queue(a)
                moving = sum(1 for v in eng.inbound[a] if v.v >= QUEUE_SPEED and v.route.s_stop - 60 < v.s < v.route.s_stop)
            else:
                # Camera down: estimate the queue from forecast arrivals since the approach was last served.
                n = round(self.forecaster.rate(a, eng.t) * sh.wait[a])
                m, moving = n * 6.0, 0
            obs["queue"][a], obs["queue_m"][a], obs["moving"][a] = n, m, moving
        return obs

    def _decide(self, demand):
        sh = self.shield
        before = sh.phase
        wanted, info = self.ctrl.decide(self.observe())
        target = sh.request(wanted, demand)
        reason = None
        if target != wanted:
            if sh.green_time < sh.cfg.min_green_s:
                reason = "min_green"
                if self.decision.get("reason") != "min_green" and self.ctrl.adaptive:
                    self.note(f"Shield held {_axis(before)} green: minimum {sh.cfg.min_green_s:.0f} s not reached", "shield")
            elif wanted == before and sh.green_time >= sh.cfg.max_green_s:
                reason = "max_green"
                self.note(f"Shield ended {_axis(before)} green at the {sh.cfg.max_green_s:.0f} s maximum", "shield")
            else:
                reason = "max_wait"
                self.note(f"Shield served {_axis(target)}: an approach waited {sh.cfg.max_wait_s:.0f} s", "shield")
        elif target != before and self.ctrl.adaptive:
            who = "Agent" if self.controller == "rl" else "Controller"
            self.note(f"{who} gave green to {_axis(target)} after {sh.green_time:.0f} s of {_axis(before)}", "agent")
        self.decision = {"wanted": wanted, "target": target, "p": info["p"], "why": info["why"], "reason": reason}

    def step(self, dt):
        eng, sh = self.engine, self.shield
        sc = SCENARIOS[self.scenario]
        for a in APPROACHES:
            if self.rng.random() < self.rate(a, eng.t) * dt:
                eng.add_arrival(a)
            if self.rng.random() < PED_BASE * sc.get("ped_scale", 1.0) * (0.6 + 0.4 * peak_shape(eng.t)) / 3600 * dt:
                eng.add_pedestrian(a)

        demand = {a: eng.approaching(a) for a in APPROACHES}
        if not sh.in_clearance and (not self.ctrl.adaptive or sh.green_time >= self.next_decision):
            self._decide(demand)
            if self.ctrl.adaptive:
                self.next_decision = (math.floor(sh.green_time / DECISION_S) + 1) * DECISION_S
        was_clearing = sh.in_clearance
        sh.advance(dt, demand)
        if was_clearing and not sh.in_clearance:
            self.next_decision = DECISION_S

        heads = self.heads()
        state, ph = sh.signal_state()
        ped_ok = {a: state == "green" and a not in PHASES[ph] for a in APPROACHES}
        eng.step(dt, heads, ped_ok)

        m = int(eng.t // 60)
        if m != self.minute:
            self.forecaster.record_minute(self.minute * 60, eng.take_arrivals(), self.camera_ok)
            self.minute = m
            cutoff = min(self.run_t0 if hasattr(self, "run_t0") else eng.t, eng.t - 600)
            eng.finished = [f for f in eng.finished if f[0] >= cutoff]
            eng.ped_waits = [f for f in eng.ped_waits if f[0] >= cutoff]

    def advance(self, real_dt):
        if self.paused:
            return
        self._acc += min(real_dt, 0.25) * self.speed
        n = 0
        while self._acc >= STEP and n < 400:
            self.step(STEP)
            self._acc -= STEP
            n += 1

    # ---------------- reporting ----------------
    def summary(self):
        eng = self.engine
        delays = [w for t, w in eng.finished if t >= self.run_t0]
        return {"controller": self.controller, "scenario": self.scenario,
                "minutes": (eng.t - self.run_t0) / 60,
                "delay": sum(delays) / len(delays) if delays else None}

    def frame(self, full=False):
        eng, sh = self.engine, self.shield
        cars, peds = eng.snapshot()
        state, ph = sh.signal_state()
        queues = {}
        for a in APPROACHES:
            n, m = eng.queue(a)
            queues[a] = [n, round(m, 1), round(eng.longest_wait(a), 1), len(eng.backlog[a])]
        out = {
            "type": "frame", "t": round(eng.t, 2), "clock": clock(eng.t),
            "paused": self.paused, "speed": self.speed,
            "controller": self.controller, "scenario": self.scenario, "camera_ok": self.camera_ok,
            "adaptive": self.ctrl.adaptive,
            "signal": {"state": state, "phase": sh.phase if ph is None else ph, "green_time": round(sh.green_time, 1),
                       "clearance_left": round(sh.clearance_left, 1), "heads": self.heads(),
                       "next_decision_in": round(max(0.0, self.next_decision - sh.green_time), 1)},
            "decision": {**self.decision, "p": round(self.decision["p"], 3)},
            "vehicles": cars, "peds": peds, "queues": queues,
            "incidents": [{"approach": i.approach, "s_from": i.s_from, "s_to": i.s_to} for i in eng.incidents],
        }
        if full:
            out["full"] = True
            delays = [w for t, w in eng.finished if t >= self.run_t0]
            recent = sum(1 for t, _ in eng.finished if t > eng.t - 300)
            span = min(300.0, eng.t - self.run_t0)
            pw = [w for t, w in eng.ped_waits if t >= self.run_t0]
            out["kpi"] = {
                "delay": round(sum(delays) / len(delays), 1) if delays else None,
                "throughput": round(recent / span * 3600) if span > 30 else None,
                "ped_wait": round(sum(pw) / len(pw), 1) if pw else None,
                "served": len(delays), "since": clock(self.run_t0),
            }
            fc = {}
            for a in APPROACHES:
                past, nxt = self.forecaster.series(a, eng.t)
                fc[a] = {"past": past, "next": [round(x, 2) for x in nxt],
                         "n1": round(self.forecaster.count(a, eng.t, 1), 1),
                         "n5": round(self.forecaster.count(a, eng.t, 5), 1),
                         "n15": round(self.forecaster.count(a, eng.t, 15), 1)}
            out["forecast"] = fc
            cur = self.summary()
            out["runs"] = [{**cur, "current": True}] + self.runs
            out["log"] = [[clock(t), k, m] for t, k, m in list(self.log)[:16]]
        return out

    def info(self):
        """Static description for the frontend."""
        return {
            "controllers": [{"key": c.key, "label": c.label, "adaptive": c.adaptive} for c in self.controllers.values()],
            "scenarios": [{"key": k, "label": v["label"]} for k, v in SCENARIOS.items()],
            "types": [{"name": t.name, "cls": t.cls, "length": t.length, "width": t.width} for t in TYPES],
            "shield": {"min_green": self.shield.cfg.min_green_s, "max_green": self.shield.cfg.max_green_s,
                       "yellow": self.shield.cfg.yellow_s, "all_red": self.shield.cfg.all_red_s,
                       "max_wait": self.shield.cfg.max_wait_s},
            "decision_interval": DECISION_S,
            "routes_stop": {a: ROUTES[(a, _straight(a))].s_stop for a in APPROACHES},
        }


def _axis(ph):
    return "N–S" if ph == 0 else "E–W"


def _straight(a):
    return {"north": "south", "south": "north", "east": "west", "west": "east"}[a]
