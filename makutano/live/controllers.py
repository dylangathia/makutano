"""Signal controllers for the live console.

Each controller looks at an observation (what the camera and forecaster
report) and asks for a green phase. It never sets the lights itself: the
session passes the request through the project's SafetyShield, the same one
the RL environment uses.

The "agent" here is a stand-in policy: a forecast-aware pressure rule with
the same inputs and outputs as the trained PPO agent. Once training has
produced models/agent.zip, register a controller that loads it and maps this
observation to the agent's vector (see makutano.control.env).
"""
import math

from ..control.rule_based import choose_phase

PHASES = (("north", "south"), ("east", "west"))


class Controller:
    key = ""
    label = ""
    adaptive = True        # decides every decision interval; fixed plans check every step

    def decide(self, obs):
        """Return (wanted phase, info) where info has 'p' (0..1, how strongly it wants a
        switch) and 'why' (one line for the operator)."""
        raise NotImplementedError


class FixedPlan(Controller):
    adaptive = False

    def __init__(self, key, label, greens):
        self.key, self.label, self.greens = key, label, greens

    def decide(self, obs):
        ph, g = obs["phase"], obs["green_time"]
        plan = self.greens[ph]
        wanted = ph if g < plan else 1 - ph
        return wanted, {"p": min(1.0, g / plan), "why": f"{_name(ph)} green {int(g)} of {plan} s in the plan"}


class RuleBased(Controller):
    key, label = "rule", "Rule-based"

    def __init__(self, fallback):
        self.fallback = fallback

    def decide(self, obs):
        if not obs["camera_ok"]:
            wanted, info = self.fallback.decide(obs)
            info["why"] += " · camera down, using the fallback plan"
            return wanted, info
        q, w = obs["queue_m"], obs["waits"]
        scores = [sum(q[a] for a in p) + 0.5 * max(w[a] for a in p) for p in PHASES]
        wanted = choose_phase(q, w, [set(p) for p in PHASES], w_queue=1.0, w_wait=0.5)
        cur, other = scores[obs["phase"]], scores[1 - obs["phase"]]
        return wanted, {"p": _sig((other - cur) / 15), "why": f"Score {other:.0f} waiting vs {cur:.0f} on green"}


class PressureAgent(Controller):
    """Stand-in for the PPO agent: weighs queues, waits and forecast arrivals."""
    key, label = "rl", "RL agent"

    def decide(self, obs):
        ph = obs["phase"]
        q, w, f = obs["queue"], obs["waits"], obs["arrivals_20s"]

        def pressure(p):
            return sum(q[a] + 0.5 * f[a] + 0.03 * w[a] for a in p)

        cur = 0.6 * pressure(PHASES[ph]) + 0.5 * sum(obs["moving"][a] for a in PHASES[ph])
        other = pressure(PHASES[1 - ph])
        p = _sig((other - cur - 2.5) / 2.2)
        src = "forecasts" if obs["camera_ok"] else "forecast only, camera down"
        return (1 - ph if p > 0.5 else ph), {"p": p, "why": f"Pressure {other:.1f} waiting vs {cur:.1f} moving ({src})"}


def _sig(x):
    return 1 / (1 + math.exp(-x))


def _name(ph):
    return "N–S" if ph == 0 else "E–W"


def build_controllers(today=(40, 25), webster=(29, 37), fallback=(29, 37)):
    fb = FixedPlan("fallback", "Fallback plan", fallback)
    ctrls = [
        FixedPlan("fixed", "Today's plan", today),
        FixedPlan("webster", "Webster re-timed", webster),
        RuleBased(fb),
        PressureAgent(),
    ]
    return {c.key: c for c in ctrls}
