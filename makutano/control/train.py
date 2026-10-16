"""Train the reinforcement learning signal controller in the calibrated junction.

    python -m makutano.control.train --config config/site.yaml \
        --routes "sumo/train/*.rou.xml" --forecaster outputs/predict/forecaster.pkl \
        --out models/rl_pred

    # the comparison agent, identical except that it gets no predictions:
    python -m makutano.control.train --config config/site.yaml \
        --routes "sumo/train/*.rou.xml" --no-prediction --out models/rl_nopred

Training draws a random training route file and SUMO seed for every episode.
Evaluation in makutano.control.experiments uses a separate block of seeds, so
the agent is never scored on traffic it trained on.
"""
import argparse
import glob
import json
from pathlib import Path

from ..config import load_config
from ..predict import models as forecasters
from .env import make_env


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--routes", required=True, help="glob of training route files")
    p.add_argument("--forecaster", help="forecaster.pkl from makutano.predict.evaluate")
    p.add_argument("--no-prediction", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    p.add_argument("--timesteps", type=int, default=200_000, help="agent decisions (one per 5 s)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor

    cfg = load_config(args.config)
    routes = sorted(glob.glob(args.routes))
    if not routes:
        raise SystemExit(f"no route files match {args.routes}")
    use_prediction = not args.no_prediction
    forecaster = forecasters.load(args.forecaster) if use_prediction else None
    end_s = cfg["control"].get("episode_minutes", 120) * 60
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = Monitor(make_env(cfg, routes, use_prediction, forecaster, seed=args.seed, end_s=end_s),
                  str(out / "monitor"))
    if args.algo == "ppo":
        model = PPO("MlpPolicy", env, seed=args.seed, n_steps=1024, batch_size=256, gamma=0.99, verbose=1)
    else:
        model = DQN("MlpPolicy", env, seed=args.seed, learning_starts=2000, buffer_size=100_000,
                    exploration_fraction=0.2, target_update_interval=1000, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(out / "agent")
    env.close()
    (out / "agent.json").write_text(json.dumps({
        "algo": args.algo, "use_prediction": use_prediction, "timesteps": args.timesteps,
        "seed": args.seed, "training_routes": len(routes),
        "forecaster": args.forecaster if use_prediction else None}, indent=2))
    print(f"saved {out / 'agent.zip'}; episode rewards are in {out / 'monitor.monitor.csv'}")


if __name__ == "__main__":
    main()
