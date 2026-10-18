"""
Evaluation and Benchmark Engine comparing OSPF, Heuristics, and RL Agents.
Generates comprehensive comparative performance metrics for academic reporting.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

from src.env.green_network_env import GreenNetworkEnv
from src.agents.baselines import OSPFBaselineAgent, StaticHeuristicAgent, ThresholdLoadAgent


def evaluate_agent(agent: Any, env: GreenNetworkEnv, num_steps: int = 120, seed: int = 42) -> Dict[str, Any]:
    """
    Runs an evaluation run over a fixed seed and traffic schedule.
    Returns timeseries metrics and aggregated summary stats.
    """
    obs, info = env.reset(seed=seed)
    
    records = []
    total_reward = 0.0

    for step in range(num_steps):
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        records.append({
            "step": step,
            "power_w": info["total_power_w"],
            "savings_percent": info["energy_savings_percent"],
            "avg_delay_ms": info["avg_delay_ms"],
            "packet_loss_percent": info["packet_loss_percent"],
            "congested_links": info["congested_links_count"],
            "active_cores_count": len(info["active_cores"]),
            "reward": reward
        })

        if terminated or truncated:
            break

    df = pd.DataFrame(records)

    summary = {
        "mean_power_w": round(float(df["power_w"].mean()), 2),
        "mean_savings_percent": round(float(df["savings_percent"].mean()), 2),
        "mean_delay_ms": round(float(df["avg_delay_ms"].mean()), 2),
        "max_delay_ms": round(float(df["avg_delay_ms"].max()), 2),
        "mean_packet_loss_percent": round(float(df["packet_loss_percent"].mean()), 4),
        "total_congested_events": int(df["congested_links"].sum()),
        "cumulative_reward": round(float(total_reward), 2),
        "timeseries": records
    }
    return summary


def run_full_benchmark(rl_agent: Any, num_steps: int = 120, seed: int = 100) -> Dict[str, Any]:
    """
    Runs side-by-side benchmark comparing:
    1. OSPF / ECMP (All Active)
    2. Static Diurnal Heuristic
    3. Load-Threshold Reactive
    4. Green RL Agent (PPO)
    """
    env = GreenNetworkEnv(max_steps_per_episode=num_steps)

    agents = {
        "OSPF / ECMP (Baseline)": OSPFBaselineAgent(),
        "Static Heuristic": StaticHeuristicAgent(),
        "Load-Threshold Reactive": ThresholdLoadAgent(),
        "Green RL Agent (Ours)": rl_agent
    }

    results = {}
    for name, agent in agents.items():
        print(f"[*] Running benchmark for: {name}...")
        results[name] = evaluate_agent(agent, env, num_steps=num_steps, seed=seed)

    return results
