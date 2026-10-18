"""
Academic Benchmark & Comparative Evaluation Script.
Runs OSPF / ECMP Baseline vs Heuristic vs Trained Green RL Agent.
Generates performance metrics table for project report.
"""

import os
import sys
import json
import pandas as pd

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent
from src.agents.evaluate import run_full_benchmark


def main():
    print("=" * 60)
    print(" [*] GREEN SDN: RUNNING FULL BENCHMARK EVALUATION")
    print("=" * 60)

    model_path = os.path.join("models", "ppo_green_agent.zip")
    env = GreenNetworkEnv(k=4, max_steps_per_episode=120)

    if os.path.exists(model_path):
        print(f"[*] Loading trained RL model from {model_path}...")
        rl_agent = GreenRLAgent(env=env, algo="PPO", model_path=model_path)
    else:
        print("[!] Warning: Trained model not found, evaluating with initial policy...")
        rl_agent = GreenRLAgent(env=env, algo="PPO")

    results = run_full_benchmark(rl_agent, num_steps=120, seed=123)

    # Format into comparison table
    rows = []
    for agent_name, stats in results.items():
        rows.append({
            "Yontem / Algoritma": agent_name,
            "Ort. Guc (Watt)": stats["mean_power_w"],
            "Enerji Tasarrufu (%)": f"{stats['mean_savings_percent']}%",
            "Ort. Gecikme (ms)": stats["mean_delay_ms"],
            "Maks. Gecikme (ms)": stats["max_delay_ms"],
            "Paket Kaybi (%)": f"{stats['mean_packet_loss_percent']}%",
            "Darbogaz Olaylari": stats["total_congested_events"],
            "Kumulatif Odul": stats["cumulative_reward"]
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print(" [BENCHMARK] AKADEMIK KARSILASTIRMA SONUCLARI")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Save to JSON
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n[+] Ayrintili zaman serisi sonuclari results/benchmark_summary.json dosyasina kaydedildi.")


if __name__ == "__main__":
    main()
