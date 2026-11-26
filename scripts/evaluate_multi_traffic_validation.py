"""
GreenNet-RL: Multi-Traffic Regime Validation Analysis on Real SNDlib Abilene Dataset.
Evaluates Low, Medium, and Peak traffic windows exclusively within the validation split.
Test split is strictly protected and never touched.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.green_network_env import GreenNetworkEnv
from src.data.abilene_loader import AbileneDataLoader
from src.agents.baselines import OSPFBaselineAgent, StaticHeuristicAgent, ThresholdLoadAgent
from stable_baselines3 import PPO


def evaluate_window(env, agent, steps=200, start_step=0):
    obs, info = env.reset(options={"step": start_step})
    total_energy_kwh = 0.0
    sla_violations = 0
    delays = []
    losses = []
    powers = []
    toggles = 0
    step_count = 0

    for _ in range(steps):
        if hasattr(agent, "predict"):
            action = agent.predict(obs)
            if isinstance(action, tuple):
                action = int(action[0])
            else:
                action = int(action)
        elif callable(agent):
            action = int(agent(obs))
        else:
            action = 7

        obs, r, term, trunc, s_info = env.step(action)
        step_count += 1

        power_w = float(s_info.get("total_power_w", 0.0))
        powers.append(power_w)
        step_kwh = (power_w * env.step_duration_s) / (3600.0 * 1000.0)
        total_energy_kwh += step_kwh

        avg_delay = float(s_info.get("avg_delay_ms", 0.0))
        loss_rate = float(s_info.get("packet_loss_rate", 0.0))
        delays.append(avg_delay)
        losses.append(loss_rate)

        if avg_delay > env.sla_delay_ms or loss_rate > env.sla_loss_rate:
            sla_violations += 1

        if term or trunc:
            break

    delays_a = np.array(delays, dtype=float)
    return {
        "steps": step_count,
        "mean_power_w": round(float(np.mean(powers)), 1),
        "total_energy_kwh": round(total_energy_kwh, 4),
        "sla_violation_percent": round((sla_violations / step_count) * 100.0, 2),
        "avg_delay_ms": round(float(np.mean(delays_a)), 2),
        "p95_delay_ms": round(float(np.percentile(delays_a, 95)), 2),
        "mean_packet_loss_rate": round(float(np.mean(losses)), 5),
        "switch_toggles": int(env.episode_switch_toggles),
    }


def main():
    loader = AbileneDataLoader()
    checkpoint_path = "models/pilot_abilene_100k/checkpoint_step_100000.zip"
    ppo_model = PPO.load(checkpoint_path, device="cpu")

    class ModelWrapper:
        def __init__(self, m):
            self.m = m
        def predict(self, o):
            act, _ = self.m.predict(o, deterministic=True)
            return int(act)

    eval_ppo = ModelWrapper(ppo_model)

    windows = [
        {"name": "Low Traffic (Off-Peak/Night)", "start_step": 0, "steps": 200},
        {"name": "Medium Traffic (Normal Day)", "start_step": 2000, "steps": 200},
        {"name": "Peak Traffic (Burst/High Load)", "start_step": 4500, "steps": 200},
    ]

    capacities = [1000.0, 2500.0]
    full_report = {}

    for cap in capacities:
        cap_key = f"{int(cap)}Mbps"
        full_report[cap_key] = {}
        print(f"\n{'='*75}\n [*] EVALUATION AT LINK CAPACITY: {cap} Mbps\n{'='*75}")

        dummy_env = GreenNetworkEnv(topology_type="abilene", link_capacity_mbps=cap)
        agents = {
            "PPO (100k)": eval_ppo,
            "OSPF (All-Active)": OSPFBaselineAgent(env=dummy_env),
            "Static Heuristic": StaticHeuristicAgent(env=dummy_env),
            "Threshold Reactive": ThresholdLoadAgent(env=dummy_env),
        }

        for win in windows:
            win_name = win["name"]
            full_report[cap_key][win_name] = {}
            print(f"\n [>] Scenario: {win_name} (Steps {win['start_step']}..{win['start_step']+win['steps']}):")

            env = GreenNetworkEnv(
                topology_type="abilene",
                link_capacity_mbps=cap,
                traffic_source="real",
                real_traffic_loader=loader,
                real_traffic_split="val",
                real_traffic_loop=False,
            )

            # Measure OSPF baseline first for energy reference
            ospf_res = evaluate_window(env, agents["OSPF (All-Active)"], steps=win["steps"], start_step=win["start_step"])
            ref_energy = ospf_res["total_energy_kwh"]

            for agent_name, agent_obj in agents.items():
                if agent_name == "OSPF (All-Active)":
                    res = ospf_res
                else:
                    res = evaluate_window(env, agent_obj, steps=win["steps"], start_step=win["start_step"])

                savings = ((ref_energy - res["total_energy_kwh"]) / ref_energy * 100.0) if ref_energy > 0 else 0.0
                res["energy_savings_percent"] = round(savings, 2)
                full_report[cap_key][win_name][agent_name] = res

                print(f"     {agent_name:<20}: Power={res['mean_power_w']}W | Savings={savings:5.1f}% | SLA Viol={res['sla_violation_percent']:5.1f}% | Loss={res['mean_packet_loss_rate']*100:6.2f}% | p95 Delay={res['p95_delay_ms']}ms | Toggles={res['switch_toggles']}")

    output_path = "results/multi_traffic_validation_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[+] Full multi-regime report saved to: {output_path}")


if __name__ == "__main__":
    main()
