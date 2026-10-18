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
    args = parser.parse_args()

    print("=" * 60)
    print(" [*] GREEN SDN: RL AGENT TRAINING INITIATED")
    print(f" [*] Algorithm: {args.algo} | Timesteps: {args.timesteps}")
    print("=" * 60)

    env = GreenNetworkEnv(k=4, max_steps_per_episode=150, training_mode=True)
    agent = GreenRLAgent(env=env, algo=args.algo)

    agent.train(total_timesteps=args.timesteps, save_path=args.save_path)
    print("\n[+] Training complete! Model ready for live simulation & benchmarking.")


if __name__ == "__main__":
    main()
