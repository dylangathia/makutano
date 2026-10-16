"""Compare every controller in every scenario over the same evaluation seeds.

    python -m makutano.control.experiments --config config/site.yaml \
        --routes sumo/eval.rou.xml --programs sumo/programs.add.xml \
        --forecaster outputs/predict/forecaster.pkl \
        --rl models/rl_pred/agent.zip --rl-nopred models/rl_nopred/agent.zip \
        --out outputs/experiments

Strategies: today (the network's observed plan), webster and actuated (from
makutano.control.webster), rule (the rule-based controller), rl and rl_nopred
(the agents trained with and without predictions). Strategies whose inputs are
missing are skipped. Every strategy runs on the same seeds, so results can be
compared seed by seed. Writes:
  results.csv             one row per scenario, strategy and seed
  paired_differences.csv  mean difference between every pair of strategies,
                          per scenario and metric, with a 95% confidence interval
"""
import argparse
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..sim.compare import t95
from .rule_based import choose_phase
from .scenarios import RuntimeEvents, apply_to_routes, load_scenarios

METRICS = ["mean_delay_s", "throughput_veh_h", "mean_queue_m", "ped_wait_s_per_person", "clear_time_s"]
EVAL_SEED_BASE = 10_000_000   # far from the training seeds (< 1,000,000)


class Collector:
    """Per-second measurements taken during one run."""

    def __init__(self):
        self.t, self.queue, self.ped_person_seconds = [], [], 0.0

    def __call__(self, sim, shield=None):
        self.t.append(sim.time)
        self.queue.append(float(np.mean(list(sim.last_queues.values()))))
        self.ped_person_seconds += sim.last_ped[0]


def trip_summary(tripinfo_file, duration_s):
    loss = [float(el.get("timeLoss")) for _, el in ET.iterparse(tripinfo_file) if el.tag == "tripinfo"]
    return {"mean_delay_s": float(np.mean(loss)) if loss else np.nan,
            "throughput_veh_h": len(loss) * 3600 / duration_s if duration_s else np.nan}


def clear_time(t, queue, disruption, before_s=600, window_s=60, tolerance=1.1):
    """Seconds after a disruption ends until the queue is back to its earlier level."""
    if disruption is None:
        return np.nan
    t, queue = np.asarray(t, float), np.asarray(queue, float)
    start, end = disruption
    before = queue[(t >= start - before_s) & (t < start)]
    if len(before) == 0:
        return np.nan
    level = before.mean() * tolerance + 1e-9
    smooth = pd.Series(queue).rolling(int(window_s), min_periods=1).mean().to_numpy()
    after = np.where((t >= end) & (smooth <= level))[0]
    return float(t[after[0]] - end) if len(after) else np.nan


def paired_differences(results):
    rows = []
    strategies = results["strategy"].unique()
    for (scenario, metric), (a, b) in itertools.product(
            itertools.product(results["scenario"].unique(), METRICS), itertools.permutations(strategies, 2)):
        r = results[results["scenario"] == scenario]
        pa = r[r["strategy"] == a].set_index("seed")[metric]
        pb = r[r["strategy"] == b].set_index("seed")[metric]
        d = (pa - pb).dropna()
        if len(d) == 0:
            continue
        half = t95(len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        rows.append({"scenario": scenario, "metric": metric, "strategy": a, "minus": b,
                     "mean_difference": d.mean(), "ci95": half, "n_seeds": len(d)})
    return pd.DataFrame(rows)


def run_program(cfg, strategy, route, seed, scenario, programs, end_s, tripinfo):
    from .env import JunctionSim
    sim = JunctionSim(cfg)
    collector = Collector()
    additional = [programs] if strategy != "today" else []
    sim.start(route, seed, additional=additional, tripinfo=tripinfo,
              program=None if strategy == "today" else strategy)
    events = RuntimeEvents(scenario, cfg)
    try:
        while sim.running(end_s):
            sim.step()
            events.on_step(sim.conn, sim.time)
            sim.measure()
            sim.pedestrians()
            collector(sim)
        duration = sim.time
    finally:
        sim.close()
    return collector, duration, len(sim.persons_seen)


def run_controller(cfg, strategy, route, seed, scenario, end_s, tripinfo, forecaster, agents):
    from .env import make_env
    c = cfg["control"]
    collector = Collector()
    use_prediction = strategy == "rl"
    env = make_env(cfg, [route], use_prediction, forecaster if use_prediction else None,
                   scenario, seed=seed, end_s=end_s, observer=collector)
    obs, _ = env.reset(options={"route": route, "sumo_seed": seed, "tripinfo": tripinfo})
    weights = c.get("rule_weights", {})
    try:
        done = False
        while not done:
            if strategy == "rule":
                action = choose_phase(env.queues, env.shield.wait, c["phase_approaches"], c.get("ped_phase"),
                                      env.ped_waiting, env.shield.ped_wait, **weights)
            else:
                action = int(agents[strategy].predict(obs, deterministic=True)[0])
            obs, _, done, _, _ = env.step(action)
        duration = env.sim.time
        persons = len(env.sim.persons_seen)
    finally:
        env.close()
    return collector, duration, persons


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--routes", required=True, help="evaluation route file (measured demand)")
    p.add_argument("--programs", help="programs.add.xml from makutano.control.webster")
    p.add_argument("--forecaster", help="forecaster.pkl, needed for the rl strategy")
    p.add_argument("--rl", help="agent trained with predictions")
    p.add_argument("--rl-nopred", help="agent trained without predictions")
    p.add_argument("--out", required=True)
    p.add_argument("--scenarios", nargs="*", help="run only these scenarios")
    args = p.parse_args()

    cfg = load_config(args.config)
    c = cfg["control"]
    seeds = [EVAL_SEED_BASE + i for i in range(c.get("eval_seeds", 20))]
    end_s = c.get("episode_minutes", 120) * 60
    out = Path(args.out)
    (out / "routes").mkdir(parents=True, exist_ok=True)

    strategies = ["today"]
    if args.programs:
        strategies += ["webster", "actuated"]
    strategies.append("rule")
    agents, forecaster = {}, None
    if args.rl or args.rl_nopred:
        from stable_baselines3 import DQN, PPO
        for name, path in (("rl", args.rl), ("rl_nopred", args.rl_nopred)):
            if path:
                # train.py writes agent.json next to agent.zip, recording the algorithm.
                meta = Path(path).with_suffix(".json")
                algo = json.loads(meta.read_text())["algo"] if meta.exists() else "ppo"
                agents[name] = (DQN if algo == "dqn" else PPO).load(path)
                strategies.append(name)
    if "rl" in agents:
        if not args.forecaster:
            raise SystemExit("--rl needs --forecaster")
        from ..predict import models as forecasters
        forecaster = forecasters.load(args.forecaster)

    scenarios = [s for s in load_scenarios(cfg) if not args.scenarios or s.name in args.scenarios]
    rows = []
    for scenario in scenarios:
        route = apply_to_routes(args.routes, scenario, cfg, out / "routes" / f"{scenario.name}.rou.xml")
        for seed, strategy in itertools.product(seeds, strategies):
            tripinfo = out / f"tripinfo_{scenario.name}_{strategy}_{seed}.xml"
            if strategy in ("today", "webster", "actuated"):
                col, duration, persons = run_program(cfg, strategy, route, seed, scenario, args.programs,
                                                     end_s, tripinfo)
            else:
                col, duration, persons = run_controller(cfg, strategy, route, seed, scenario, end_s,
                                                        tripinfo, forecaster, agents)
            row = {"scenario": scenario.name, "stress": scenario.stress, "strategy": strategy, "seed": seed,
                   **trip_summary(tripinfo, duration),
                   "mean_queue_m": float(np.mean(col.queue)) if col.queue else np.nan,
                   "ped_wait_s_per_person": col.ped_person_seconds / persons if persons else np.nan,
                   "clear_time_s": clear_time(col.t, col.queue, scenario.disruption)}
            rows.append(row)
            tripinfo.unlink(missing_ok=True)   # tripinfo files are large; the summary is kept
            print(f"{scenario.name:>16} {strategy:>10} seed {seed}: delay {row['mean_delay_s']:.1f} s", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    paired_differences(results).to_csv(out / "paired_differences.csv", index=False)
    print(results.groupby(["scenario", "strategy"])["mean_delay_s"].mean().unstack().round(1).to_string())


if __name__ == "__main__":
    main()
