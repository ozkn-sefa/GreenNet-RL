"""
Standalone re-evaluation runner for past Fat-Tree k=4 PPO checkpoints under fixed audit rules.
Preserves existing original JSON files by outputting to a separate result file:
  results/pilot_fattree_k4_100k_v2_reevaluation.json

Evaluates Checkpoints (20k, 50k, 80k, 100k) and Baselines (OSPF, Static, Threshold)
across identical validation regimes (Seeds 60001-60004).
Applies:
- Active VecNormalize normalization for PPO policy inference
- GreenNetworkEnv SLA contract (s_info["sla_violation"])
- Standard action_overridden telemetry and requested/applied action tracking
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Any, List, Optional
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


def evaluate_agent(
    agent: Any,
    regime: Dict[str, Any],
    is_ppo: bool = False,
    vec_path: Optional[str] = None,
) -> Dict[str, Any]:
    env = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=regime["steps"] + 10,
        training_mode=False,
        traffic_source="synthetic",
        use_safety_filter=True,
    )

    vec_norm = None
    if vec_path and os.path.exists(vec_path):
        eval_venv = DummyVecEnv([lambda: env])
        vec_norm = VecNormalize.load(vec_path, eval_venv)
        vec_norm.training = False
        vec_norm.norm_reward = False

    obs, info = env.reset(seed=regime["seed"], options={
        "intensity": regime["intensity"],
        "profile": regime["profile"],
        "burst": regime["burst"],
    })

    powers = []
    delays = []
    p95_delays = []
    losses = []
    utils = []
    sla_violations = 0
    raw_actions = []
    applied_actions = []
    overrides = 0

    for _ in range(regime["steps"]):
        policy_obs = vec_norm.normalize_obs(obs) if (vec_norm is not None and is_ppo) else obs
        if hasattr(agent, "predict"):
            act = agent.predict(policy_obs, deterministic=True)
            if isinstance(act, tuple):
                raw_act = int(act[0])
            elif isinstance(act, np.ndarray):
                raw_act = int(act.item())
            else:
                raw_act = int(act)
        else:
            raw_act = 7

        raw_actions.append(raw_act)
        obs, reward, term, trunc, s_info = env.step(raw_act)
        applied_act = int(env.current_action)
        applied_actions.append(applied_act)

        if s_info.get("action_overridden", applied_act != raw_act):
            overrides += 1

        powers.append(s_info["total_power_w"])
        delays.append(s_info["avg_delay_ms"])
        p95_delays.append(s_info.get("p95_delay_ms", s_info["avg_delay_ms"]))
        losses.append(s_info["packet_loss_rate"])
        utils.append(s_info["max_utilization"])

        if s_info.get("sla_violation", False):
            sla_violations += 1

    total_kwh = env.power_model.total_energy_kwh
    toggles = env.episode_switch_toggles

    return {
        "mean_power_w": round(float(np.mean(powers)), 2),
        "total_energy_kwh": round(float(total_kwh), 6),
        "mean_delay_ms": round(float(np.mean(delays)), 3),
        "p95_delay_ms": round(float(np.mean(p95_delays)), 3),
        "mean_loss_percent": round(float(np.mean(losses) * 100.0), 4),
        "max_utilization_percent": round(float(np.max(utils) * 100.0), 2),
        "sla_violations": int(sla_violations),
        "sla_violation_rate_percent": round(float(sla_violations / regime["steps"] * 100.0), 2),
        "switch_toggles": int(toggles),
        "raw_actions_distribution": {int(a): raw_actions.count(a) for a in set(raw_actions)},
        "applied_actions_distribution": {int(a): applied_actions.count(a) for a in set(applied_actions)},
        "override_count": int(overrides),
        "override_rate_percent": round(float(overrides / regime["steps"] * 100.0), 2),
    }


def run_reevaluation():
    ckpt_dir = "models/pilot_fattree_k4_100k_v2"
    target_steps = [20000, 50000, 80000, 100000]
    output_path = "results/pilot_fattree_k4_100k_v2_reevaluation.json"

    ref_env = GreenNetworkEnv(topology_type="fattree_k4", k=4)
    ospf_agent = OSPFBaselineAgent(ref_env)
    static_agent = StaticHeuristicAgent(ref_env)
    thresh_agent = ThresholdLoadAgent(env=ref_env)

    # Baselines
    print("Evaluating Baselines across 4 validation regimes...")
    baselines = {}
    for reg in VAL_REGIMES:
        r_name = reg["name"]
        baselines[r_name] = {
            "ospf": evaluate_agent(ospf_agent, reg),
            "static": evaluate_agent(static_agent, reg),
            "threshold": evaluate_agent(thresh_agent, reg),
        }

    records = []
    print(f"Re-evaluating checkpoints {target_steps} from {ckpt_dir}...")
    for step in target_steps:
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}.zip")
        vec_path = os.path.join(ckpt_dir, f"checkpoint_step_{step}_vecnormalize.pkl")
        if not os.path.exists(ckpt_path):
            continue

        model = PPO.load(ckpt_path, device="cpu")
        regimes_data = {}
        ppo_kwh_total = 0.0
        ospf_kwh_total = 0.0

        for reg in VAL_REGIMES:
            r_name = reg["name"]
            ppo_res = evaluate_agent(model, reg, is_ppo=True, vec_path=vec_path)
            ospf_res = baselines[r_name]["ospf"]
            static_res = baselines[r_name]["static"]
            thresh_res = baselines[r_name]["threshold"]

            saving = (1.0 - (ppo_res["total_energy_kwh"] / max(1e-9, ospf_res["total_energy_kwh"]))) * 100.0
            ppo_kwh_total += ppo_res["total_energy_kwh"]
            ospf_kwh_total += ospf_res["total_energy_kwh"]

            regimes_data[r_name] = {
                "ppo_saving_percent": round(saving, 2),
                "ppo": ppo_res,
                "ospf": ospf_res,
                "static": static_res,
                "threshold": thresh_res,
            }

        overall_saving = (1.0 - (ppo_kwh_total / max(1e-9, ospf_kwh_total))) * 100.0
        records.append({
            "step": step,
            "overall_saving_percent": round(overall_saving, 2),
            "regimes": regimes_data,
        })

    payload = {
        "metadata": {
            "description": "Re-evaluation of pilot v2 checkpoints using fixed SLA contract and active VecNormalize",
            "source_checkpoint_dir": ckpt_dir,
            "validation_regimes": VAL_REGIMES,
            "evaluated_steps": target_steps,
        },
        "validation_records": records,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[+] Re-evaluation saved cleanly to {output_path}")


if __name__ == "__main__":
    run_reevaluation()
