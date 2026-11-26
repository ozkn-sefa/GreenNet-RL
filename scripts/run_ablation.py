"""
Reward-function ablation: trains one PPO per reward variant (identical budget/seed/training traffic)
and evaluates each on the held-out test scenarios (raw policy, no shield, no safe filter).
Writes results/ablation.json and results/ABLATION_REPORT.md.

Usage:
    python scripts/run_ablation.py --timesteps 30000 --seeds 10 --train-seed 0
"""

import argparse
import json
import os
import sys
import time

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent
from src.agents.experiments import TEST_SEED_BASE, METRICS, load_scenarios, run_matrix, aggregate, markdown_table

VARIANTS = {
    "full": {},
    "no_waste": {"waste": 0.0},
    "no_switching": {"switching": 0.0, "wakeup": 0.0},
    "no_congestion": {"congestion": 0.0},
    "no_delay": {"delay": 0.0},
    "no_loss": {"loss": 0.0},
    "no_energy": {"energy": 0.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=30000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    seeds = [TEST_SEED_BASE + i for i in range(args.seeds)]
    scenarios = {k: v for k, v in load_scenarios().items() if not k.startswith("failure")}
    os.makedirs("models/ablation", exist_ok=True)
    results, t0 = {}, time.time()
    for name in args.variants:
        cfg = VARIANTS[name]
        print(f"[*] Training variant '{name}' {cfg} for {args.timesteps} steps", flush=True)
        train_env = GreenNetworkEnv(k=4, max_steps_per_episode=150, training_mode=True, reward_cfg=cfg, reward_scale=0.05)
        path = f"models/ablation/ppo_{name}.zip"
        agent = GreenRLAgent(env=train_env, algo="PPO", seed=args.train_seed, use_shield=False)
        agent.train(total_timesteps=args.timesteps, save_path=path)
        # Evaluate with the DEFAULT (full) reward env so QoS/energy metrics are comparable across variants
        env = GreenNetworkEnv(k=4, max_steps_per_episode=args.steps)
        raw = run_matrix(env, {
            "OSPF / ECMP": lambda e: __import__("src.agents.baselines", fromlist=["x"]).OSPFBaselineAgent(env=e),
            name: lambda e, p=path: GreenRLAgent(env=e, algo="PPO", model_path=p, use_shield=False),
        }, scenarios, seeds, args.steps, progress=False)
        results[name] = aggregate(raw)["ALL (mean over scenarios)"][name]

    metrics = [m for m, _ in METRICS if m not in ("total_energy_kwh",)] + ["total_energy_kwh"]
    with open("results/ablation.json", "w", encoding="utf-8") as fh:
        json.dump({"timesteps": args.timesteps, "train_seed": args.train_seed, "test_seeds": seeds,
                   "variants": {k: VARIANTS[k] for k in args.variants}, "results": results}, fh, indent=1, default=float)
    L = ["# Reward Ablation (auto-generated)", "",
         f"One PPO per variant, {args.timesteps} training timesteps, training seed {args.train_seed}, "
         f"evaluated on {len(seeds)} held-out seeds x {len(scenarios)} load scenarios (raw policy, full-reward env). "
         "Cells: mean ± std [95% CI] over test seeds. Only ONE training run per variant: differences smaller than "
         "training-seed variance are not significant.", "",
         markdown_table(results, metrics), "",
         "Variant definitions (reward_cfg multipliers set to 0): " + json.dumps({k: VARIANTS[k] for k in args.variants})]
    with open("results/ABLATION_REPORT.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"[+] done in {time.time() - t0:.0f}s -> results/ablation.json, results/ABLATION_REPORT.md")


if __name__ == "__main__":
    main()
