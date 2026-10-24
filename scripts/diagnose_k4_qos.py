"""
Diagnostic script for Fat-Tree k=4 PPO v2 QoS root cause analysis.
Performs:
1. Detailed checkpoint extraction for 20k, 50k, 80k, 100k.
2. Step-by-step trace of Medium regime for PPO, OSPF, Static, Threshold:
   - raw action vs applied action
   - active cores
   - offered load, carried load, loss rate
   - bottleneck link and max link utilization
   - delay (avg, p95, p99)
   - SafeFilter decision mechanism and timing (lag)
   - switch toggle counting mechanism
3. Reward assignment analysis when action is overridden.
"""

import json
import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.green_network_env import GreenNetworkEnv
from src.agents.baselines import OSPFBaselineAgent, StaticHeuristicAgent, ThresholdLoadAgent

VAL_REGIMES = [
    {"name": "Low", "intensity": 0.50, "profile": "diurnal", "burst": False, "seed": 60001, "steps": 50},
    {"name": "Medium", "intensity": 1.10, "profile": "diurnal", "burst": False, "seed": 60002, "steps": 50},
    {"name": "High", "intensity": 1.85, "profile": "datacenter_pareto", "burst": False, "seed": 60003, "steps": 50},
    {"name": "Burst", "intensity": 2.50, "profile": "flash_crowd", "burst": True, "seed": 60004, "steps": 50},
]


def print_checkpoint_metrics():
    with open("results/pilot_fattree_k4_100k_v2_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    target_steps = [20000, 50000, 80000, 100000]
    print("=" * 90)
    print("1. CHECKPOINT SUMMARY (20K, 50K, 80K, 100K) FROM pilot_fattree_k4_100k_v2_results.json")
    print("=" * 90)

    for r in data["validation_records"]:
        step = r["step"]
        if step in target_steps:
            print(f"\n--- CHECKPOINT STEP {step} ---")
            for r_name in ["Low", "Medium", "High", "Burst"]:
                reg = r["regimes"][r_name]
                ppo = reg["ppo"]
                ospf = reg["ospf"]
                print(f"  [{r_name:6s}] PPO Save: {reg['ppo_saving_percent']:6.2f}% | Loss: {ppo['mean_loss_percent']:6.2f}% | Delay: {ppo['mean_delay_ms']:6.3f}ms (p95: {ppo['p95_delay_ms']:6.3f}) | SLA: {ppo['sla_violation_rate_percent']:5.1f}% | Overrides: {ppo['override_rate_percent']:5.1f}% | Toggles: {ppo['switch_toggles']}")
                print(f"           Raw Act: {ppo['raw_actions_distribution']} -> Applied Act: {ppo['applied_actions_distribution']}")
                print(f"           OSPF: Power={ospf['mean_power_w']:.1f}W, Loss={ospf['mean_loss_percent']:.2f}%, Delay={ospf['mean_delay_ms']:.3f}ms, SLA={ospf['sla_violation_rate_percent']}%, Util={ospf['max_utilization_percent']}%")
            ov = r["overall"]
            print(f"  [OVERALL] Save: {ov['overall_saving_percent']}% | SLA: {ov['overall_sla_violation_rate_percent']}% | Loss: {ov['overall_mean_loss_percent']}% | Overrides: {ov['overall_override_rate_percent']}% | Meets Safety: {ov['meets_safety_criteria']}")


def run_step_by_step_medium_trace(checkpoint_step=100000):
    print("\n" + "=" * 90)
    print(f"2. STEP-BY-STEP DIAGNOSTIC TRACE ON MEDIUM REGIME (Seed 60002, Step {checkpoint_step})")
    print("=" * 90)

    regime = VAL_REGIMES[1]  # Medium
    ckpt_path = f"models/pilot_fattree_k4_100k_v2/checkpoint_step_{checkpoint_step}.zip"
    vec_path = f"models/pilot_fattree_k4_100k_v2/checkpoint_step_{checkpoint_step}_vecnormalize.pkl"

    env = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=regime["steps"] + 10,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=True,
    )

    eval_venv = DummyVecEnv([lambda: env])
    if os.path.exists(vec_path):
        eval_venv = VecNormalize.load(vec_path, eval_venv)
        eval_venv.training = False
        eval_venv.norm_reward = False

    model = PPO.load(ckpt_path, device="cpu")

    obs, info = env.reset(seed=regime["seed"], options={
        "intensity": regime["intensity"],
        "profile": regime["profile"],
        "burst": regime["burst"],
    })

    print(f"{'Step':>4} | {'RawAct':>6} {'AppAct':>6} {'Cores':>5} | {'Offered':>7} {'Carried':>7} {'Loss%':>6} | {'MaxUtil':>7} {'Bottleneck':>14} | {'Delay':>6} {'p95':>6} | {'Overridden':>10}")
    print("-" * 95)

    loss_steps = []
    override_steps = []

    for s in range(regime["steps"]):
        raw_act = int(model.predict(obs, deterministic=True)[0])
        obs, reward, term, trunc, s_info = env.step(raw_act)
        app_act = int(env.current_action)
        num_cores = len(env.core_actions[app_act])
        dem = s_info["total_demand_mbps"]
        deliv = s_info["delivered_mbps"]
        loss_pct = s_info["packet_loss_percent"]
        max_u = s_info["max_utilization"]
        bl = s_info.get("bottleneck_link", "none")
        d_avg = s_info["avg_delay_ms"]
        d_p95 = s_info.get("p95_delay_ms", d_avg)
        overridden = s_info.get("action_overridden", False)

        if loss_pct > 0.01:
            loss_steps.append((s, raw_act, app_act, num_cores, loss_pct, max_u, bl, dem))
        if overridden:
            override_steps.append((s, raw_act, app_act))

        if s < 15 or loss_pct > 0.01:
            flag = " [LOSS!]" if loss_pct > 0.01 else ""
            print(f"{s:4d} | {raw_act:6d} {app_act:6d} {num_cores:5d} | {dem:7.1f} {deliv:7.1f} {loss_pct:6.2f}% | {max_u*100:6.1f}% {bl:14s} | {d_avg:6.2f} {d_p95:6.2f} | {str(overridden):>10s}{flag}")

    print("\nTotal loss steps (>0.01%):", len(loss_steps), "out of", regime["steps"])
    print("Total override steps:", len(override_steps), "out of", regime["steps"])
    return loss_steps


def run_ospf_medium_trace():
    print("\n" + "=" * 90)
    print("3. OSPF STEP-BY-STEP TRACE ON MEDIUM REGIME (Seed 60002)")
    print("=" * 90)
    regime = VAL_REGIMES[1]
    env = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=regime["steps"] + 10,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=False,
    )
    obs, info = env.reset(seed=regime["seed"], options={
        "intensity": regime["intensity"],
        "profile": regime["profile"],
        "burst": regime["burst"],
    })
    agent = OSPFBaselineAgent()

    loss_steps = []
    print(f"{'Step':>4} | {'Act':>4} {'Cores':>5} | {'Offered':>7} {'Carried':>7} {'Loss%':>6} | {'MaxUtil':>7} {'Bottleneck':>16} | {'Delay':>6} {'p95':>6}")
    print("-" * 80)
    for s in range(regime["steps"]):
        act = agent.predict(obs)
        obs, r, term, trunc, s_info = env.step(act)
        dem = s_info["total_demand_mbps"]
        deliv = s_info["delivered_mbps"]
        loss_pct = s_info["packet_loss_percent"]
        max_u = s_info["max_utilization"]
        bl = s_info.get("bottleneck_link", "none")
        d_avg = s_info["avg_delay_ms"]
        d_p95 = s_info.get("p95_delay_ms", d_avg)

        if loss_pct > 0.01:
            loss_steps.append((s, act, dem, loss_pct, max_u, bl, d_avg))
            print(f"{s:4d} | {act:4d} {4:5d} | {dem:7.1f} {deliv:7.1f} {loss_pct:6.2f}% | {max_u*100:6.1f}% {bl:16s} | {d_avg:6.2f} {d_p95:6.2f} [OSPF LOSS!]")
        elif s < 10:
            print(f"{s:4d} | {act:4d} {4:5d} | {dem:7.1f} {deliv:7.1f} {loss_pct:6.2f}% | {max_u*100:6.1f}% {bl:16s} | {d_avg:6.2f} {d_p95:6.2f}")

    print(f"\nTotal OSPF loss steps: {len(loss_steps)} out of {regime['steps']}")
    for ls in loss_steps:
        print(f"  Step {ls[0]}: Dem={ls[2]:.1f}Mbps, Loss={ls[3]:.2f}%, MaxUtil={ls[4]*100:.1f}%, Bottleneck={ls[5]}, Delay={ls[6]:.2f}ms")
    return loss_steps


def run_threshold_medium_trace():
    print("\n" + "=" * 90)
    print("3b. THRESHOLD AGENT STEP-BY-STEP TRACE ON MEDIUM REGIME (Seed 60002)")
    print("=" * 90)
    regime = VAL_REGIMES[1]
    env = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=regime["steps"] + 10,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=True,
    )
    obs, info = env.reset(seed=regime["seed"], options={
        "intensity": regime["intensity"],
        "profile": regime["profile"],
        "burst": regime["burst"],
    })
    agent = ThresholdLoadAgent(env=env)

    loss_steps = []
    print(f"{'Step':>4} | {'Act':>4} {'Cores':>5} | {'Offered':>7} {'Carried':>7} {'Loss%':>6} | {'MaxUtil':>7} {'Bottleneck':>16} | {'Delay':>6} {'p95':>6}")
    print("-" * 80)
    for s in range(regime["steps"]):
        act = agent.predict(obs)
        obs, r, term, trunc, s_info = env.step(act)
        dem = s_info["total_demand_mbps"]
        deliv = s_info["delivered_mbps"]
        loss_pct = s_info["packet_loss_percent"]
        max_u = s_info["max_utilization"]
        bl = s_info.get("bottleneck_link", "none")
        d_avg = s_info["avg_delay_ms"]
        d_p95 = s_info.get("p95_delay_ms", d_avg)

        if loss_pct > 0.01:
            loss_steps.append((s, act, dem, loss_pct, max_u, bl, d_avg))
            print(f"{s:4d} | {act:4d} {len(env.core_actions[env.current_action]):5d} | {dem:7.1f} {deliv:7.1f} {loss_pct:6.2f}% | {max_u*100:6.1f}% {bl:16s} | {d_avg:6.2f} {d_p95:6.2f} [THRESH LOSS!]")
        elif s < 10:
            print(f"{s:4d} | {act:4d} {len(env.core_actions[env.current_action]):5d} | {dem:7.1f} {deliv:7.1f} {loss_pct:6.2f}% | {max_u*100:6.1f}% {bl:16s} | {d_avg:6.2f} {d_p95:6.2f}")

    print(f"\nTotal Threshold loss steps: {len(loss_steps)} out of {regime['steps']}")
    for ls in loss_steps:
        print(f"  Step {ls[0]}: Dem={ls[2]:.1f}Mbps, Loss={ls[3]:.2f}%, MaxUtil={ls[4]*100:.1f}%, Bottleneck={ls[5]}, Delay={ls[6]:.2f}ms")
    return loss_steps


def analyze_credit_assignment_and_toggles():
    print("\n" + "=" * 90)
    print("4. CREDIT ASSIGNMENT AND SWITCH TOGGLE DEEP DIVE")
    print("=" * 90)
    env = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=60,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=True,
    )
    obs, info = env.reset(seed=60002, options={"intensity": 1.10, "profile": "diurnal", "burst": False})

    # Demonstrate credit assignment when an unsafe action is overridden
    # Say raw action is 0 (1 core active), but demand is high and SafeFilter overrides to 6 (4 cores)
    # What reward does the agent receive?
    env.active_demands = env.traffic_gen.generate_step_demands(active_flow_count=16) # force existing demands
    obs, r_overridden, term, trunc, s_info_ov = env.step(0)
    print("Scenario A: Agent selects raw Action 0 (1 core), SafeFilter overrides to:", env.current_action)
    print("  Observed Reward received by agent:", r_overridden)
    print("  Reward components:", s_info_ov["reward_components"])
    print("  Notice: energy_reward is based on applied action:", env.current_action, "cores =", len(env.core_actions[env.current_action]))

    # Now compare with SafeFilter turned off with Action 0 under same seed:
    env_unsafe = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=60,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=False,
    )
    env_unsafe.reset(seed=60002, options={"intensity": 1.10, "profile": "diurnal", "burst": False})
    obs2, r_unsafe, term, trunc, s_info_un = env_unsafe.step(0)
    print("\nScenario B: Same environment without SafeFilter, Action 0 applied directly:")
    print("  Observed Reward:", r_unsafe)
    print("  Reward components:", s_info_un["reward_components"])
    print("  Loss rate:", s_info_un["packet_loss_percent"], "% | Delay:", s_info_un["avg_delay_ms"], "ms")
    print(f"  Delta in reward: {r_overridden - r_unsafe:+.4f} (SafeFilter masked the penalty!)")


if __name__ == "__main__":
    print_checkpoint_metrics()
    run_step_by_step_medium_trace(100000)
    run_ospf_medium_trace()
    run_threshold_medium_trace()
    analyze_credit_assignment_and_toggles()
