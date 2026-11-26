"""
Evaluation and Benchmark Engine comparing OSPF, Heuristics, and RL Agents.
Generates comprehensive comparative performance metrics for academic reporting.
"""

from typing import Dict, List, Any, Optional, Sequence
import numpy as np
import pandas as pd
try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

from src.env.green_network_env import GreenNetworkEnv, weighted_percentile
from src.agents.baselines import OSPFBaselineAgent, StaticHeuristicAgent, ThresholdLoadAgent


def evaluate_agent(
    agent: Any,
    env: GreenNetworkEnv,
    num_steps: int = 120,
    seed: int = 42,
    options: Optional[Dict[str, Any]] = None,
    include_timeseries: bool = True,
) -> Dict[str, Any]:
    """
    Runs an evaluation run over a fixed seed and traffic schedule.
    Returns timeseries metrics and aggregated summary stats.
    `options` is forwarded to env.reset (intensity, profile, burst, failures).
    """
    obs, info = env.reset(seed=seed, options=options)

    records = []
    total_reward = 0.0
    pooled_delays: List[np.ndarray] = []
    pooled_weights: List[np.ndarray] = []

    for step in range(num_steps):
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        pooled_delays.append(env.last_flow_delays)
        pooled_weights.append(env.last_flow_weights)

        records.append({
            "step": step,
            "power_w": info["total_power_w"],
            "savings_percent": info["energy_savings_percent"],
            "avg_delay_ms": info["avg_delay_ms"],
            "p95_delay_ms": info["p95_delay_ms"],
            "p99_delay_ms": info["p99_delay_ms"],
            "packet_loss_percent": info["packet_loss_percent"],
            "congested_links": info["congested_links_count"],
            "active_cores_count": len(info["active_cores"]),
            "step_energy_j": info["step_energy_j"],
            "switch_toggles": info["switch_toggles"],
            "action_changed": info["action_changed"],
            "sla_violation": info["sla_violation"],
            "reward": reward
        })

        if terminated or truncated:
            break

    df = pd.DataFrame(records)
    delays = np.concatenate(pooled_delays) if pooled_delays else np.zeros(0)
    weights = np.concatenate(pooled_weights) if pooled_weights else np.zeros(0)

    summary = {
        "mean_power_w": round(float(df["power_w"].mean()), 2),
        "mean_savings_percent": round(float(df["savings_percent"].mean()), 2),
        "mean_delay_ms": round(float(df["avg_delay_ms"].mean()), 2),
        "max_delay_ms": round(float(df["avg_delay_ms"].max()), 2),
        "p95_delay_ms": round(weighted_percentile(delays, weights, 95), 3),
        "p99_delay_ms": round(weighted_percentile(delays, weights, 99), 3),
        "mean_packet_loss_percent": round(float(df["packet_loss_percent"].mean()), 4),
        "total_energy_kwh": float(info["episode_energy_kwh"]),
        "switch_toggles": int(info["episode_switch_toggles"]),
        "action_changes": int(info["episode_action_changes"]),
        "sla_violation_rate": float(df["sla_violation"].mean()),
        "total_congested_events": int(df["congested_links"].sum()),
        "cumulative_reward": round(float(total_reward), 2),
    }
    if include_timeseries:
        summary["timeseries"] = records
    return summary


def mean_std_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """Mean, sample std (ddof=1) and two-sided Student-t confidence interval over independent seeds."""
    a = np.asarray(values, dtype=float)
    n = a.size
    mean = float(a.mean()) if n else float("nan")
    if n < 2:
        return {"n": n, "mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(a.std(ddof=1))
    if sp_stats is not None:
        half = float(sp_stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * std / np.sqrt(n))
    else:
        # Two-sided 95% Student-t critical values for df = 1..30 (exact fallback when scipy is absent)
        t_table_95 = {
            1: 12.7062047, 2: 4.3026527, 3: 3.1824463, 4: 2.7764451, 5: 2.5705818,
            6: 2.4469119, 7: 2.3646243, 8: 2.3060041, 9: 2.2621572, 10: 2.2281389,
            11: 2.2009852, 12: 2.1788128, 13: 2.1603687, 14: 2.1447867, 15: 2.1314495,
            16: 2.1199053, 17: 2.1098156, 18: 2.1009220, 19: 2.0930241, 20: 2.0859634,
            25: 2.0595386, 30: 2.0422725,
        }
        t_crit = t_table_95.get(n - 1, 1.959964) if abs(confidence - 0.95) < 1e-3 else 1.959964
        half = float(t_crit * std / np.sqrt(n))
    return {"n": n, "mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


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
        "OSPF / ECMP (Baseline)": OSPFBaselineAgent(env=env),
        "Static Heuristic": StaticHeuristicAgent(env=env),
        "Load-Threshold Reactive": ThresholdLoadAgent(env=env),
        "Green RL Agent (Ours)": rl_agent
    }

    results = {}
    for name, agent in agents.items():
        print(f"[*] Running benchmark for: {name}...")
        results[name] = evaluate_agent(agent, env, num_steps=num_steps, seed=seed)

    return results
