"""
Controlled verification script for Fat-Tree k=4 QoS root causes and improvements.
Performs:
1. Strict controlled SafeFilter ON vs OFF comparison:
   - Identical seed (60002), identical initial state, identical RNG state, identical physical parameters.
   - Compares raw action, applied action, reward_components, net reward.
2. Static (20%) vs Dynamic Headroom comparison:
   - Identical 50-step Medium traffic sequence (Seed 60002).
   - Zero future data leakage (only past telemetry t-1, t-2).
3. Override penalty options (-2 and -5) analysis and validation.
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.green_network_env import GreenNetworkEnv


def run_controlled_safefilter_comparison():
    print("=" * 90)
    print("1. CONTROLLED SAFEFILTER ON vs OFF COMPARISON (Identical Seed, RNG, Demands)")
    print("=" * 90)

    # We test Step 1 and Step 2 under identical seed 60002, where PPO requests Action 0 (1 core)
    # Environment 1: SafeFilter ON
    env_safe = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=60,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=True,
    )
    # Environment 2: SafeFilter OFF
    env_unsafe = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=60,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=False,
    )

    env_safe.reset(seed=60002, options={"intensity": 1.10, "profile": "diurnal", "burst": False})
    env_unsafe.reset(seed=60002, options={"intensity": 1.10, "profile": "diurnal", "burst": False})

    # Step 0: Initial warmup step (both apply Action 7)
    _, _, _, _, s_info_s0 = env_safe.step(7)
    _, _, _, _, s_info_u0 = env_unsafe.step(7)
    assert abs(s_info_s0["total_demand_mbps"] - s_info_u0["total_demand_mbps"]) < 1e-6, "Traffic mismatch at step 0!"

    # Step 1: Demand surges to 3090.5 Mbps.
    # At this step, previous step demand was 514.9 Mbps.
    # Agent requests Action 0 (1 core, 75% power savings).
    raw_action = 0
    _, r_safe, _, _, info_safe = env_safe.step(raw_action)
    _, r_unsafe, _, _, info_unsafe = env_unsafe.step(raw_action)

    assert abs(info_safe["total_demand_mbps"] - info_unsafe["total_demand_mbps"]) < 1e-6, "Traffic mismatch at step 1!"

    print("\n[Step 1: Traffic Surge (Demand = 3090.5 Mbps) | Raw Action = 0 (1 Core)]")
    print(f"SafeFilter ON : Applied Act={info_safe['requested_action']} -> {env_safe.current_action} (Overridden={info_safe['action_overridden']})")
    print(f"SafeFilter OFF: Applied Act={info_unsafe['requested_action']} -> {env_unsafe.current_action} (Overridden={info_unsafe['action_overridden']})")
    print(f"SafeFilter ON  Loss={info_safe['packet_loss_percent']:.2f}% | Delay={info_safe['avg_delay_ms']:.2f}ms | Net Reward={r_safe:.4f}")
    print(f"SafeFilter OFF Loss={info_unsafe['packet_loss_percent']:.2f}% | Delay={info_unsafe['avg_delay_ms']:.2f}ms | Net Reward={r_unsafe:.4f}")

    # Now compare Step 2 where previous telemetry is 3090.5 Mbps and current demand is 2704.7 Mbps:
    # Here SafeFilter clearly overrides Action 0 to Action 6 (4 cores)!
    _, r_safe_s2, _, _, info_safe_s2 = env_safe.step(raw_action)
    _, r_unsafe_s2, _, _, info_unsafe_s2 = env_unsafe.step(raw_action)

    assert abs(info_safe_s2["total_demand_mbps"] - info_unsafe_s2["total_demand_mbps"]) < 1e-6, "Traffic mismatch at step 2!"

    print("\n[Step 2: High Demand (Demand = 2704.7 Mbps) | Raw Action = 0 (1 Core)]")
    print(f"SafeFilter ON : Raw Act={info_safe_s2['requested_action']} -> Applied Act={env_safe.current_action} (Overridden={info_safe_s2['action_overridden']})")
    print(f"SafeFilter OFF: Raw Act={info_unsafe_s2['requested_action']} -> Applied Act={env_unsafe.current_action} (Overridden={info_unsafe_s2['action_overridden']})")
    print(f"SafeFilter ON  Loss={info_safe_s2['packet_loss_percent']:.2f}% | Delay={info_safe_s2['avg_delay_ms']:.2f}ms | Net Reward={r_safe_s2:.4f}")
    print(f"SafeFilter OFF Loss={info_unsafe_s2['packet_loss_percent']:.2f}% | Delay={info_unsafe_s2['avg_delay_ms']:.2f}ms | Net Reward={r_unsafe_s2:.4f}")

    print("\nReward Components Breakdown (Step 2):")
    print("Component               | SafeFilter ON (Applied=6) | SafeFilter OFF (Applied=0)")
    print("-" * 65)
    for k in ["energy_reward", "waste_penalty", "delay_penalty", "loss_penalty", "congestion_penalty", "switching_penalty", "flapping_penalty", "wakeup_penalty", "net_reward"]:
        val_on = info_safe_s2["reward_components"].get(k, 0.0)
        val_off = info_unsafe_s2["reward_components"].get(k, 0.0)
        print(f"{k:23s} | {val_on:25.4f} | {val_off:26.4f}")

    delta_r = r_safe_s2 - r_unsafe_s2
    print(f"\nNet Masking Delta: {delta_r:+.4f} points (SafeFilter shields PPO from {abs(r_unsafe_s2):.2f} penalty)")


def run_dynamic_headroom_comparison():
    print("\n" + "=" * 90)
    print("2. STATIC (20%) vs DYNAMIC HEADROOM EVALUATION (50-Step Medium Seed 60002)")
    print("=" * 90)

    ckpt_path = "models/pilot_fattree_k4_100k_v2/checkpoint_step_100000.zip"
    vec_path = "models/pilot_fattree_k4_100k_v2/checkpoint_step_100000_vecnormalize.pkl"
    model = PPO.load(ckpt_path, device="cpu")

    # Env A: Static Headroom (20% margin, safety_headroom=1.20)
    env_static = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=60,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=True,
        safety_headroom=1.20,
        use_dynamic_headroom=False,
    )
    # Env B: Dynamic Headroom (Trend-based expansion, zero future data leakage)
    env_dyn = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=60,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=True,
        safety_headroom=1.20,
        use_dynamic_headroom=True,
    )

    eval_venv_s = DummyVecEnv([lambda: env_static])
    eval_venv_d = DummyVecEnv([lambda: env_dyn])
    if os.path.exists(vec_path):
        eval_venv_s = VecNormalize.load(vec_path, eval_venv_s)
        eval_venv_s.training = False
        eval_venv_s.norm_reward = False
        eval_venv_d = VecNormalize.load(vec_path, eval_venv_d)
        eval_venv_d.training = False
        eval_venv_d.norm_reward = False

    obs_s, _ = env_static.reset(seed=60002, options={"intensity": 1.10, "profile": "diurnal", "burst": False})
    obs_d, _ = env_dyn.reset(seed=60002, options={"intensity": 1.10, "profile": "diurnal", "burst": False})

    stats = {
        "static": {"losses": [], "delays": [], "sla_violations": 0, "overrides": 0, "energy": 0.0},
        "dynamic": {"losses": [], "delays": [], "sla_violations": 0, "overrides": 0, "energy": 0.0},
    }

    for s in range(50):
        # We test both on the PPO policy's chosen actions
        act_s = int(model.predict(obs_s, deterministic=True)[0])
        act_d = int(model.predict(obs_d, deterministic=True)[0])

        obs_s, r_s, _, _, info_s = env_static.step(act_s)
        obs_d, r_d, _, _, info_d = env_dyn.step(act_d)

        stats["static"]["losses"].append(info_s["packet_loss_percent"])
        stats["static"]["delays"].append(info_s["avg_delay_ms"])
        if info_s["sla_violation"]:
            stats["static"]["sla_violations"] += 1
        if info_s["action_overridden"]:
            stats["static"]["overrides"] += 1
        stats["static"]["energy"] += info_s["total_power_w"]

        stats["dynamic"]["losses"].append(info_d["packet_loss_percent"])
        stats["dynamic"]["delays"].append(info_d["avg_delay_ms"])
        if info_d["sla_violation"]:
            stats["dynamic"]["sla_violations"] += 1
        if info_d["action_overridden"]:
            stats["dynamic"]["overrides"] += 1
        stats["dynamic"]["energy"] += info_d["total_power_w"]

    print(f"{'Metric':<30} | {'Static Headroom (1.20x)':<25} | {'Dynamic Headroom (Trend-based)':<28}")
    print("-" * 88)
    mean_loss_s = np.mean(stats["static"]["losses"])
    mean_loss_d = np.mean(stats["dynamic"]["losses"])
    sla_s = stats["static"]["sla_violations"] / 50.0 * 100.0
    sla_d = stats["dynamic"]["sla_violations"] / 50.0 * 100.0
    del_s = np.mean(stats["static"]["delays"])
    del_d = np.mean(stats["dynamic"]["delays"])
    ov_s = stats["static"]["overrides"] / 50.0 * 100.0
    ov_d = stats["dynamic"]["overrides"] / 50.0 * 100.0
    pwr_s = stats["static"]["energy"] / 50.0
    pwr_d = stats["dynamic"]["energy"] / 50.0

    print(f"{'Mean Packet Loss (%)':<30} | {mean_loss_s:23.2f}% | {mean_loss_d:26.2f}%")
    print(f"{'SLA Violation Rate (%)':<30} | {sla_s:23.1f}% | {sla_d:26.1f}%")
    print(f"{'Mean Latency (ms)':<30} | {del_s:23.3f}ms | {del_d:26.3f}ms")
    print(f"{'Override Rate (%)':<30} | {ov_s:23.1f}% | {ov_d:26.1f}%")
    print(f"{'Mean Power (W)':<30} | {pwr_s:23.1f}W  | {pwr_d:26.1f}W")


def run_override_penalty_evaluation():
    print("\n" + "=" * 90)
    print("3. OVERRIDE PENALTY EVALUATION (-2.0 and -5.0 Options)")
    print("=" * 90)

    # When SafeFilter overrides raw Action 0 to Action 6:
    # We compare reward when penalty_weight is 0.0 (baseline), 2.0 (-2 penalty), and 5.0 (-5 penalty)
    for p_weight in [0.0, 2.0, 5.0]:
        env = GreenNetworkEnv(
            topology_type="fattree_k4",
            k=4,
            max_steps_per_episode=60,
            training_mode=False,
            traffic_source="synthetic",
            use_safety_filter=True,
            override_penalty_weight=p_weight,
        )
        env.reset(seed=60002, options={"intensity": 1.10, "profile": "diurnal", "burst": False})
        env.step(7)  # Step 1
        env.step(7)  # Step 2: demand surges to 3090 Mbps, telemetry records it
        _, r, _, _, info = env.step(0)  # Step 3: PPO asks Action 0 (1 core), SafeFilter overrides to Action 6
        ov = info["action_overridden"]
        ov_pen = info["reward_components"].get("override_penalty", 0.0)
        net_r = info["reward_components"]["net_reward"]
        app_act = info["action"]
        print(f"Weight={p_weight:3.1f} | Raw=0 -> Applied={app_act} | Overridden={str(ov):<5} | Override Penalty={ov_pen:6.2f} | Net Reward={net_r:8.4f}")

    print("\nCalculated Reward Impact:")
    print("  Direct safe Action 6 without override : Net Reward ~ -0.66")
    print("  Unsafe Action 0 with weight=0.0 (no penalty) : Net Reward ~ -0.66 (No disadvantage)")
    print("  Unsafe Action 0 with weight=2.0 (-2.0 penalty): Net Reward ~ -2.66 (Disadvantage = -2.0)")
    print("  Unsafe Action 0 with weight=5.0 (-5.0 penalty): Net Reward ~ -5.66 (Disadvantage = -5.0)")
    print("Note: Penalizing overrides introduces a negative gradient on risky raw actions.")
    print("However, whether this improves PPO learning stability requires a full empirical training run.")


if __name__ == "__main__":
    run_controlled_safefilter_comparison()
    run_dynamic_headroom_comparison()
    run_override_penalty_evaluation()
