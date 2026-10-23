"""
Training Script for Green SDN Reinforcement Learning Agent (PPO / DQN).
Usage:
    python scripts/train.py --algo PPO --timesteps 25000
"""

import argparse
import os
import sys

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent


def main():
    parser = argparse.ArgumentParser(description="Train Green SDN RL Agent")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "DQN"], help="RL Algorithm")
    parser.add_argument("--timesteps", type=int, default=25000, help="Total training timesteps")
    parser.add_argument("--save-path", type=str, default="models/ppo_green_agent.zip", help="Path to save trained weights")
    parser.add_argument("--reward-scale", type=float, default=0.05, help="Uniform reward scaling for training stability (1.0 = raw reward)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device for neural network (auto, cuda, cpu)")
    parser.add_argument("--topology", type=str, default="fattree_k4", help="Topology type (fattree_k4, abilene, etc.)")
    parser.add_argument("--traffic-source", type=str, default="synthetic", choices=["synthetic", "real"], help="Traffic source (synthetic or real)")
    parser.add_argument("--real-split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split for real traffic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-steps-per-episode", type=int, default=150, help="Max steps per episode")
    args = parser.parse_args()

    print("=" * 60)
    print(" [*] GREEN SDN: RL AGENT TRAINING INITIATED")
    print(f" [*] Algorithm: {args.algo} | Timesteps: {args.timesteps} | Device: {args.device}")
    print(f" [*] Topology: {args.topology} | Traffic Source: {args.traffic_source} (Split: {args.real_split})")
    print(f" [*] Seed: {args.seed} | Save Path: {args.save_path}")
    print("=" * 60)

    if args.traffic_source == "real":
        env = GreenNetworkEnv(
            topology_type=args.topology,
            max_steps_per_episode=args.max_steps_per_episode,
            training_mode=True,
            reward_scale=args.reward_scale,
            traffic_source="real",
            real_traffic_split=args.real_split,
            real_traffic_loop=True,
        )
    else:
        env = GreenNetworkEnv(
            k=4,
            topology_type=args.topology,
            max_steps_per_episode=args.max_steps_per_episode,
            training_mode=True,
            reward_scale=args.reward_scale,
            traffic_source="synthetic",
        )

    agent = GreenRLAgent(
        env=env,
        algo=args.algo,
        device=args.device,
        seed=args.seed
    )

    agent.train(total_timesteps=args.timesteps, save_path=args.save_path)
    print("\n[+] Training complete! Model ready for live simulation & benchmarking.")


if __name__ == "__main__":
    main()
