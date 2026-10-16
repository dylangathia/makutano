"""The calibrated SUMO junction as a reinforcement learning environment.

JunctionSim wraps one SUMO run through TraCI and measures the junction the way
the camera would: queues and arrivals only within camera range, with random
error matching the validated error of the video pipeline. make_env turns it
into a Gymnasium environment (built inside a function so gymnasium stays an
optional dependency): every 5 seconds the agent asks for a green phase,
the safety shield decides what is shown, and the reward is the reduction in
vehicle waiting time (minus a penalty for pedestrian waiting).

Needs SUMO (with SUMO_HOME set) and `pip install -r requirements-ml.txt`.
"""
import os
import sys
import uuid
from collections import deque

import numpy as np
import pandas as pd

from ..sim.run_sim import sumo_binary
from .scenarios import RuntimeEvents, Scenario
from .shield import SafetyShield, ShieldConfig


def import_traci():
    try:
        import traci
    except ImportError:
        home = os.environ.get("SUMO_HOME")
        if not home:
            raise SystemExit("TraCI not found: install SUMO and set SUMO_HOME, or pip install traci")
        sys.path.append(os.path.join(home, "tools"))
        import traci
    return traci


class JunctionSim:
    """One SUMO run of the junction, measured as the camera would see it."""

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        c = cfg["control"]
        self.tls = c["tls_id"]
        self.green_phases = c["green_phases"]
        self.camera_range = c.get("camera_range_m", 80.0)
        self.queue_speed = cfg["indicators"].get("queue_speed_mps", 1.4)
        noise = c.get("observation_noise", {})
        self.queue_noise = noise.get("queue_rel_sd", 0.1)
        self.arrival_noise = noise.get("arrival_rel_sd", 0.1)
        self.rng = rng or np.random.default_rng(0)
        self.approach_edges = {a: v.get("sumo_queue_edges", [v["sumo_from_edge"]])
                               for a, v in cfg["approaches"].items()}
        self.conn = None

    def start(self, route_file, seed, additional=(), tripinfo=None, program=None):
        traci = import_traci()
        s = self.cfg["sumo"]
        cmd = [sumo_binary(), "-n", s["net_file"], "-r", str(route_file), "--seed", str(seed),
               "--lateral-resolution", str(s.get("lateral_resolution", 0.8)),
               "--time-to-teleport", str(s.get("time_to_teleport", 300)),
               "--waiting-time-memory", "10000", "--no-step-log", "true", "--no-warnings", "true"]
        extra = list(s.get("additional_files", [])) + list(additional)
        if extra:
            cmd += ["-a", ",".join(map(str, extra))]
        if tripinfo:
            cmd += ["--tripinfo-output", str(tripinfo)]
        self.label = f"mk-{uuid.uuid4().hex[:8]}"
        traci.start(cmd, label=self.label)
        self.conn = traci.getConnection(self.label)
        if program:
            self.conn.trafficlight.setProgram(self.tls, program)
        logic = self.conn.trafficlight.getAllProgramLogics(self.tls)[0]
        phases = getattr(logic, "phases", None) or logic.getPhases()
        self.states = [p.state for p in phases]
        self.lanes = {a: [[f"{e}_{i}" for i in range(self.conn.edge.getLaneNumber(e))] for e in edges]
                      for a, edges in self.approach_edges.items()}
        self.offsets = {a: np.cumsum([0.0] + [self.conn.lane.getLength(lanes[0]) for lanes in per_edge[:-1]])
                        for a, per_edge in self.lanes.items()}
        self.counted = {a: set() for a in self.approach_edges}
        self.persons_seen = set()

    @property
    def time(self):
        return self.conn.simulation.getTime()

    def running(self, end_s=None):
        if end_s is not None and self.time >= end_s:
            return False
        return self.conn.simulation.getMinExpectedNumber() > 0

    def step(self):
        self.conn.simulationStep()

    def set_signal(self, kind, phase):
        """Show a green, the amber that ends it, or all-red, by state string."""
        if kind == "green":
            state = self.states[self.green_phases[phase]]
        elif kind == "yellow":
            state = self.states[self.green_phases[phase] + 1]
        else:
            state = "r" * len(self.states[0])
        self.conn.trafficlight.setRedYellowGreenState(self.tls, state)

    def measure(self):
        """True queue (m) and new arrivals per approach, within camera range."""
        queues, arrivals = {}, {}
        for a, per_edge in self.lanes.items():
            q, new = 0.0, 0
            for offset, lanes in zip(self.offsets[a], per_edge):
                for lane in lanes:
                    length = self.conn.lane.getLength(lane)
                    for v in self.conn.lane.getLastStepVehicleIDs(lane):
                        dist = offset + length - self.conn.vehicle.getLanePosition(v)
                        if dist > self.camera_range:
                            continue
                        if v not in self.counted[a]:
                            self.counted[a].add(v)
                            new += 1
                        if self.conn.vehicle.getSpeed(v) < self.queue_speed:
                            q = max(q, dist)
            queues[a], arrivals[a] = q, new
        self.last_queues = queues   # true queues, kept for the experiment's measurements
        return queues, arrivals

    def observe(self):
        """The camera's view: measured queues and arrivals with realistic error."""
        queues, arrivals = self.measure()
        noisy_q = {a: max(0.0, q * (1 + self.rng.normal(0, self.queue_noise))) for a, q in queues.items()}
        noisy_a = {a: max(0, round(n * (1 + self.rng.normal(0, self.arrival_noise)))) for a, n in arrivals.items()}
        return noisy_q, noisy_a

    def pedestrians(self):
        """(number waiting, total seconds waited by those waiting now)."""
        ids = self.conn.person.getIDList()
        self.persons_seen.update(ids)
        waits = [self.conn.person.getWaitingTime(p) for p in ids]
        waiting = [w for w in waits if w > 0]
        self.last_ped = (len(waiting), float(sum(waiting)))
        return self.last_ped

    def vehicle_wait(self):
        total = 0.0
        for per_edge in self.lanes.values():
            for lanes in per_edge:
                for lane in lanes:
                    for v in self.conn.lane.getLastStepVehicleIDs(lane):
                        total += self.conn.vehicle.getAccumulatedWaitingTime(v)
        return total

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def _gym():
    try:
        import gymnasium
    except ImportError:
        raise SystemExit("gymnasium not installed: pip install -r requirements-ml.txt")
    return gymnasium


def make_env(cfg, route_files, use_prediction=True, forecaster=None, scenario=None,
             seed=0, end_s=None, observer=None):
    gym = _gym()
    from gymnasium import spaces
    from ..predict.features import live_example

    c = cfg["control"]
    approaches = list(cfg["approaches"])
    phases = c["phase_approaches"]
    ped_phase = c.get("ped_phase")
    shield_cfg = ShieldConfig.from_config(c.get("shield"))
    step_s = c.get("decision_interval_s", 5)
    if use_prediction and forecaster is None:
        raise ValueError("use_prediction needs a forecaster (forecaster.pkl from makutano.predict.evaluate)")
    history_len = 15 * int(60 / step_s)
    sim_start = pd.Timestamp(c["sim_start"])
    fallback_greens = c.get("fallback_greens_s", [30.0] * len(phases))
    weights = c.get("reward", {})
    ped_weight = weights.get("pedestrian_weight", 1.0)

    class _Env(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            n_obs = 3 * len(approaches) + len(phases) + 4 + (2 * len(approaches) if use_prediction else 0)
            self.observation_space = spaces.Box(0.0, np.inf, shape=(n_obs,), dtype=np.float32)
            self.action_space = spaces.Discrete(len(phases))
            self.rng = np.random.default_rng(seed)
            self.sim = JunctionSim(cfg, self.rng)
            self.shield = SafetyShield(phases, shield_cfg, ped_phase)
            self.scenario = scenario or Scenario("baseline")
            self.episode = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.sim.close()
            options = options or {}
            route = options.get("route") or route_files[self.rng.integers(len(route_files))]
            run_seed = options.get("sumo_seed", int(self.rng.integers(1_000_000)))
            self.sim.start(route, run_seed, tripinfo=options.get("tripinfo"))
            self.events = RuntimeEvents(self.scenario, cfg)
            self.shield.reset(0)
            self.hist = {a: (deque(maxlen=history_len), deque(maxlen=history_len)) for a in approaches}
            self.queues = {a: 0.0 for a in approaches}
            self.arrivals = {a: 0 for a in approaches}
            self.prev_wait, self.prev_ped = 0.0, 0.0
            self.ped_waiting, self.ped_wait_total = 0, 0.0
            self.sim.set_signal("green", 0)
            self.episode += 1
            return self._obs(), {}

        def _demand(self):
            if self.scenario.camera_out(self.sim.time):
                return {a: True for a in approaches}   # no data: assume everyone is waiting
            return {a: self.queues[a] > 0 or self.arrivals[a] > 0 for a in approaches}

        def _predictions(self):
            stamp = sim_start + pd.Timedelta(seconds=self.sim.time)
            out = []
            for a in approaches:
                row, seq = live_example(self.hist[a][0], self.hist[a][1], stamp, a, approaches, step_s)
                p = forecaster.predict(row, seq)
                out += [float(p["y_next_1m"].iloc[0]), float(p["y_next_5m"].iloc[0])]
            return out

        def _obs(self):
            out_flag = self.scenario.camera_out(self.sim.time)
            q = [0.0 if out_flag else self.queues[a] / 100 for a in approaches]
            n = [0.0 if out_flag else float(self.arrivals[a]) for a in approaches]
            w = [self.shield.wait[a] / shield_cfg.max_wait_s for a in approaches]
            one_hot = [1.0 if i == self.shield.phase else 0.0 for i in range(len(phases))]
            extra = [self.shield.green_time / shield_cfg.max_green_s, float(self.shield.in_clearance),
                     self.ped_waiting / 10, float(out_flag)]
            pred = []
            if use_prediction:
                pred = [0.0] * (2 * len(approaches)) if out_flag else [x / 10 for x in self._predictions()]
            return np.array(q + n + w + one_hot + extra + pred, dtype=np.float32)

        def _fallback(self):
            """Fixed-time plan used while the camera is down."""
            if self.shield.green_time >= fallback_greens[self.shield.phase]:
                return (self.shield.phase + 1) % len(phases)
            return self.shield.phase

        def step(self, action):
            t = self.sim.time
            wanted = self._fallback() if self.scenario.camera_out(t) else int(action)
            demand = self._demand()
            self.shield.request(wanted, demand, self.ped_waiting > 0)
            arrivals = {a: 0 for a in approaches}
            for _ in range(int(step_s)):
                self.sim.set_signal(*self.shield.signal_state())
                self.sim.step()
                self.events.on_step(self.sim.conn, self.sim.time)
                self.shield.advance(1.0, demand, self.ped_waiting > 0)
                self.queues, new = self.sim.observe()
                for a in approaches:
                    arrivals[a] += new[a]
                self.ped_waiting, self.ped_wait_total = self.sim.pedestrians()
                if observer is not None:
                    observer(self.sim, self.shield)
                if not self.sim.running(end_s):
                    break
            self.arrivals = arrivals
            for a in approaches:
                self.hist[a][0].append(arrivals[a])
                self.hist[a][1].append(self.queues[a])
            wait = self.sim.vehicle_wait()
            reward = ((self.prev_wait - wait) - ped_weight * (self.ped_wait_total - self.prev_ped)) / 100
            self.prev_wait, self.prev_ped = wait, self.ped_wait_total
            done = not self.sim.running(end_s)
            return self._obs(), float(reward), done, False, {"time": self.sim.time}

        def close(self):
            self.sim.close()

    return _Env()
