"""
Audit and Re-Evaluation of Fat-Tree k=4 100K Pilot Checkpoints.
Re-evaluates Step 20k, 50k, 90k, and 100k checkpoints alongside baselines:
  - Raw PPO (no filter)
  - PPO + SafeFilter
  - OSPF (All Active)
  - Static Heuristic
  - Threshold Load
across:
  - Controllable Regimes: Low (0.5x), Medium (1.1x)
  - Overload Regimes: High (1.85x), Burst (2.5x stress test)
Records offered load, carried load, dropped load, absolute QoS, and degradation vs OSPF.
"""

import json
import os
import sys
import numpy as np
import torch
from typing import Dict, Any, List

from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.green_network_env import GreenNetworkEnv
from src.agents.baselines import OSPFBaselineAgent, StaticHeuristicAgent, ThresholdLoadAgent

REGIMES = [
    {"name": "Low", "intensity": 0.50, "profile": "diurnal", "burst": False, "seed": 60001, "steps": 50, "type": "Controllable"},
    {"name": "Medium", "intensity": 1.10, "profile": "diurnal", "burst": False, "seed": 60002, "steps": 50, "type": "Controllable"},
    {"name": "High", "intensity": 1.85, "profile": "datacenter_pareto", "burst": False, "seed": 60003, "steps": 50, "type": "Boundary Overload"},
    {"name": "Burst", "intensity": 2.50, "profile": "flash_crowd", "burst": True, "seed": 60004, "steps": 50, "type": "Extreme Stress Test"},
]

CHECKPOINTS = [20000, 50000, 90000, 100000]


def evaluate_agent_on_regime(
    agent: Any,
    regime: Dict[str, Any],
    use_filter: bool = True,
    is_raw_ppo: bool = False,
) -> Dict[str, Any]:
    env = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=regime["steps"] + 10,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=use_filter,
    )

    obs, info = env.reset(seed=regime["seed"], options={
        "intensity": regime["intensity"],
        "profile": regime["profile"],
        "burst": regime["burst"],
    })

    offered_loads = []
    carried_loads = []
    dropped_loads = []
    powers = []
    delays = []
    p95_delays = []
    p99_delays = []
    losses = []
    max_utils = []
    sla_violations = 0
    raw_actions = []
    applied_actions = []
    overrides = 0

    for _ in range(regime["steps"]):
        if hasattr(agent, "predict"):
            act = agent.predict(obs, deterministic=True)
            if isinstance(act, tuple):
                raw_act = int(act[0])
            elif isinstance(act, np.ndarray):
                raw_act = int(act.item())
            else:
                raw_act = int(act)
        else:
            raw_act = 7

        raw_actions.append(raw_act)
        
        # In raw mode, safe filter is off; in filtered mode, safe filter runs
        obs, reward, term, trunc, s_info = env.step(raw_act)

        applied_act = int(env.current_action)
        applied_actions.append(applied_act)
        if s_info.get("safe_action_overridden", False) or applied_act != raw_act:
            overrides += 1

        dem = s_info["total_demand_mbps"]
        deliv = s_info["delivered_mbps"]
        offered_loads.append(dem)
        carried_loads.append(deliv)
        dropped_loads.append(max(0.0, dem - deliv))

        powers.append(s_info["total_power_w"])
        delays.append(s_info["avg_delay_ms"])
        p95_delays.append(s_info.get("p95_delay_ms", s_info["avg_delay_ms"]))
        p99_delays.append(s_info.get("p99_delay_ms", s_info["avg_delay_ms"]))
        losses.append(s_info["packet_loss_rate"])
        max_utils.append(s_info["max_utilization"])

        if s_info["avg_delay_ms"] > env.sla_delay_ms or s_info["packet_loss_rate"] > 1e-4:
            sla_violations += 1

    total_kwh = env.power_model.total_energy_kwh

    return {
        "mean_offered_mbps": round(float(np.mean(offered_loads)), 1),
        "mean_carried_mbps": round(float(np.mean(carried_loads)), 1),
        "mean_dropped_mbps": round(float(np.mean(dropped_loads)), 1),
        "mean_power_w": round(float(np.mean(powers)), 2),
        "total_energy_kwh": round(float(total_kwh), 6),
        "mean_delay_ms": round(float(np.mean(delays)), 3),
        "p95_delay_ms": round(float(np.mean(p95_delays)), 3),
        "p99_delay_ms": round(float(np.mean(p99_delays)), 3),
        "mean_loss_percent": round(float(np.mean(losses) * 100.0), 4),
        "max_utilization_percent": round(float(np.max(max_utils) * 100.0), 2),
        "sla_violations": int(sla_violations),
        "sla_violation_rate_percent": round(float(sla_violations / regime["steps"] * 100.0), 2),
        "raw_actions_distribution": {int(a): raw_actions.count(a) for a in set(raw_actions)},
        "applied_actions_distribution": {int(a): applied_actions.count(a) for a in set(applied_actions)},
        "override_count": int(overrides),
        "override_rate_percent": round(float(overrides / regime["steps"] * 100.0), 2),
    }


def run_audit():
    print("=" * 100)
    print("AUDIT & RE-EVALUATION OF FAT-TREE k=4 100K PILOT CHECKPOINTS")
    print("=" * 100)

    # Baselines
    ref_env = GreenNetworkEnv(topology_type="fattree_k4", k=4)
    ospf_agent = OSPFBaselineAgent(ref_env)
    static_agent = StaticHeuristicAgent(ref_env)
    thresh_agent = ThresholdLoadAgent(env=ref_env)

    # 1. Evaluate Baselines First on all regimes
    baseline_results = {}
    print("\n--- Evaluating Baselines on All 4 Regimes ---")
    for reg in REGIMES:
        r_name = reg["name"]
        ospf_res = evaluate_agent_on_regime(ospf_agent, reg, use_filter=True)
        static_res = evaluate_agent_on_regime(static_agent, reg, use_filter=True)
        thresh_res = evaluate_agent_on_regime(thresh_agent, reg, use_filter=True)

        baseline_results[r_name] = {
            "ospf": ospf_res,
            "static": static_res,
            "threshold": thresh_res,
        }
        print(f"  [{r_name}] OSPF: {ospf_res['mean_power_w']} W, Loss: {ospf_res['mean_loss_percent']}%, Delay: {ospf_res['mean_delay_ms']} ms, Offered: {ospf_res['mean_offered_mbps']} Mbps, Carried: {ospf_res['mean_carried_mbps']} Mbps")

    # 2. Evaluate Checkpoints (Raw PPO & Guarded PPO)
    audit_results = {
        "metadata": {
            "topology": "fattree_k4",
            "k": 4,
            "bisection_capacity_mbps": 16000.0,
            "host_access_capacity_mbps": 16000.0,
            "checkpoints_evaluated": CHECKPOINTS,
        },
        "baselines": baseline_results,
        "checkpoints": {},
    }

    for ckpt_step in CHECKPOINTS:
        ckpt_file = f"models/pilot_fattree_k4_100k/checkpoint_step_{ckpt_step}.zip"
        if not os.path.exists(ckpt_file):
            print(f"[ERROR] Checkpoint file {ckpt_file} not found!")
            continue

        print(f"\n=======================================================")
        print(f"EVALUATING CHECKPOINT STEP {ckpt_step}")
        print(f"=======================================================")
        model = PPO.load(ckpt_file, device="cpu")

        ckpt_data = {"regimes": {}}
        for reg in REGIMES:
            r_name = reg["name"]
            ospf_ref = baseline_results[r_name]["ospf"]

            # A. Raw PPO (no filter)
            raw_ppo = evaluate_agent_on_regime(model, reg, use_filter=False, is_raw_ppo=True)
            # B. Guarded PPO (with SafeFilter)
            guarded_ppo = evaluate_agent_on_regime(model, reg, use_filter=True, is_raw_ppo=False)

            # Savings vs OSPF
            raw_saving = (1.0 - (raw_ppo["total_energy_kwh"] / max(1e-9, ospf_ref["total_energy_kwh"]))) * 100.0
            guarded_saving = (1.0 - (guarded_ppo["total_energy_kwh"] / max(1e-9, ospf_ref["total_energy_kwh"]))) * 100.0

            # Degradation vs OSPF
            delta_loss_guarded = guarded_ppo["mean_loss_percent"] - ospf_ref["mean_loss_percent"]
            delta_delay_guarded = guarded_ppo["mean_delay_ms"] - ospf_ref["mean_delay_ms"]
            delta_sla_guarded = guarded_ppo["sla_violation_rate_percent"] - ospf_ref["sla_violation_rate_percent"]

            reg_eval = {
                "regime_type": reg["type"],
                "raw_ppo": raw_ppo,
                "guarded_ppo": guarded_ppo,
                "raw_saving_percent": round(raw_saving, 2),
                "guarded_saving_percent": round(guarded_saving, 2),
                "delta_loss_vs_ospf": round(delta_loss_guarded, 4),
                "delta_delay_vs_ospf_ms": round(delta_delay_guarded, 3),
                "delta_sla_vs_ospf_percent": round(delta_sla_guarded, 2),
            }
            ckpt_data["regimes"][r_name] = reg_eval

            print(f"  Regime: {r_name} ({reg['type']})")
            print(f"    Raw PPO     : Act={raw_ppo['raw_actions_distribution']}, Power={raw_ppo['mean_power_w']} W, Sav={raw_saving:.2f}%, Loss={raw_ppo['mean_loss_percent']}%, Delay={raw_ppo['mean_delay_ms']} ms")
            print(f"    Guarded PPO : Act={guarded_ppo['applied_actions_distribution']}, Power={guarded_ppo['mean_power_w']} W, Sav={guarded_saving:.2f}%, Loss={guarded_ppo['mean_loss_percent']}%, Delay={guarded_ppo['mean_delay_ms']} ms, Override={guarded_ppo['override_rate_percent']}%")
            print(f"    vs OSPF     : DeltaLoss={delta_loss_guarded:+.2f}%, DeltaDelay={delta_delay_guarded:+.3f} ms, DeltaSLA={delta_sla_guarded:+.1f}%")

        audit_results["checkpoints"][str(ckpt_step)] = ckpt_data

    # Save to JSON
    os.makedirs("results", exist_ok=True)
    out_file = "results/audit_k4_checkpoints_reevaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\n[SUCCESS] Saved comprehensive audit re-evaluation to {out_file}")
    return audit_results


if __name__ == "__main__":
    run_audit()
