"""Safety shield: fixed rules that every controller's choice must pass.

A controller (the RL agent or the rule-based one) asks for a green phase; the
shield decides what actually happens. It keeps each green between its minimum
and maximum, always runs amber and all-red between greens, and forces service
of any approach, or of pedestrians, that has waited too long. Because the rules
sit outside the agent, a learned policy can never produce an unsafe or starving
signal sequence, whatever it has learned.
"""
from dataclasses import dataclass


@dataclass
class ShieldConfig:
    min_green_s: float = 10.0
    max_green_s: float = 60.0
    yellow_s: float = 3.0
    all_red_s: float = 2.0
    max_wait_s: float = 120.0       # longest any approach with demand may wait for green
    ped_max_wait_s: float = 90.0    # longest waiting pedestrians may wait for their phase

    @classmethod
    def from_config(cls, c):
        return cls(**{k: float(v) for k, v in (c or {}).items() if k in cls.__dataclass_fields__})


class SafetyShield:
    """phases: for each green phase, the approaches it serves. ped_phase: index of
    the pedestrian-only phase, or None if pedestrians cross with vehicle phases."""

    def __init__(self, phases, cfg=None, ped_phase=None):
        self.phases = [frozenset(p) for p in phases]
        self.cfg = cfg or ShieldConfig()
        self.ped_phase = ped_phase
        self.approaches = sorted(set().union(*self.phases))
        self.reset()

    def reset(self, phase=0):
        self.phase = phase
        self.green_time = 0.0
        self.clearance_left = 0.0
        self.next_phase = None
        self.wait = {a: 0.0 for a in self.approaches}
        self.ped_wait = 0.0

    @property
    def in_clearance(self):
        return self.clearance_left > 0

    def signal_state(self):
        """('green', phase), ('yellow', phase being ended) or ('all_red', None)."""
        if not self.in_clearance:
            return "green", self.phase
        if self.clearance_left > self.cfg.all_red_s:
            return "yellow", self.phase
        return "all_red", None

    def _serves(self, phase, approach):
        return approach in self.phases[phase]

    def _forced(self, demand, ped_waiting):
        """A phase that must be served next because someone has waited too long."""
        if (self.ped_phase is not None and ped_waiting and self.phase != self.ped_phase
                and self.ped_wait >= self.cfg.ped_max_wait_s):
            return self.ped_phase
        overdue = [a for a in self.approaches
                   if demand.get(a) and not self._serves(self.phase, a) and self.wait[a] >= self.cfg.max_wait_s]
        if not overdue:
            return None
        worst = max(overdue, key=lambda a: self.wait[a])
        # Serve the overdue approach with the phase that clears the most waiting demand.
        candidates = [i for i, p in enumerate(self.phases) if worst in p]
        return max(candidates, key=lambda i: sum(self.wait[a] for a in self.phases[i] if demand.get(a)))

    def _best_other(self, demand, ped_waiting):
        others = [i for i in range(len(self.phases)) if i != self.phase]
        def need(i):
            if i == self.ped_phase:
                return self.ped_wait if ped_waiting else -1.0
            served = [self.wait[a] for a in self.phases[i] if demand.get(a)]
            return max(served) if served else -1.0
        best = max(others, key=need, default=None)
        return best if best is not None and need(best) >= 0 else None

    def request(self, wanted, demand, ped_waiting=False):
        """Ask for phase `wanted`; returns the phase that will be green next.

        demand: approach -> True if vehicles are queued or arriving there.
        """
        if not 0 <= wanted < len(self.phases):
            raise ValueError(f"phase {wanted} does not exist")
        if self.in_clearance:
            return self.next_phase
        if self.green_time < self.cfg.min_green_s:
            return self.phase
        target = self._forced(demand, ped_waiting)
        if target is None:
            target = wanted
            if target == self.phase and self.green_time >= self.cfg.max_green_s:
                other = self._best_other(demand, ped_waiting)
                target = other if other is not None else self.phase
        if target != self.phase:
            self.next_phase = target
            self.clearance_left = self.cfg.yellow_s + self.cfg.all_red_s
        return target

    def advance(self, dt, demand, ped_waiting=False):
        """Move the clock on by dt seconds and update green and waiting times."""
        if self.in_clearance:
            self.clearance_left = max(0.0, self.clearance_left - dt)
            if not self.in_clearance:
                self.phase, self.next_phase, self.green_time = self.next_phase, None, 0.0
        else:
            self.green_time += dt
        green = None if self.in_clearance else self.phase
        for a in self.approaches:
            if green is not None and self._serves(green, a):
                self.wait[a] = 0.0
            elif demand.get(a):
                self.wait[a] += dt
        if green is not None and green == self.ped_phase:
            self.ped_wait = 0.0
        elif ped_waiting:
            self.ped_wait += dt
        else:
            self.ped_wait = 0.0
