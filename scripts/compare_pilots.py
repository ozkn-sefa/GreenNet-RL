import json
import os

def load_res(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

v1 = load_res("results/audit_k4_checkpoints_reevaluation.json") or load_res("results/pilot_fattree_k4_100k_results.json")
v2 = load_res("results/pilot_fattree_k4_100k_v2_results.json")

print("=" * 80)
print("PILOT v1 vs PILOT v2 (VecNormalize + Monitor) COMPARISON")
print("=" * 80)

# Check telemetry in v2
print("\n--- v2 LEARNING TELEMETRY SAMPLES ---")
for entry in v2["training_telemetry"]:
    step = entry["num_timesteps"]
    if step in [1024, 5120, 10240, 20480, 50176, 51200, 90112, 100000]:
        print(f"Step {step:6d} | EpRewMean: {entry['mean_reward']:10.2f} | ValLoss: {entry['value_loss']:.5f} | PolLoss: {entry['policy_loss']:.5f} | Ent: {entry['entropy_loss']:.4f} | KL: {entry['approx_kl']:.5f}")

# Check regimes for steps 20k, 50k, 90k, 100k
target_steps = [10000, 20000, 50000, 90000, 100000]
print("\n--- v2 CHECKPOINT VALIDATION RECORDS ---")
for r in v2["validation_records"]:
    step = r["step"]
    if step in target_steps:
        print(f"\n>>> CHECKPOINT STEP {step} <<<")
        for reg_name, reg in r["regimes"].items():
            ppo = reg["ppo"]
            ospf = reg["ospf"]
            print(f"  {reg_name:6s} | PPO Power: {ppo['mean_power_w']} W (Save: {reg['ppo_saving_percent']}%) | Loss: {ppo['mean_loss_percent']:.2f}% | Delay: {ppo['mean_delay_ms']} ms (p95: {ppo['p95_delay_ms']}) | SLA Viol: {ppo['sla_violation_rate_percent']}% | Override: {ppo['override_rate_percent']}% | Toggles: {ppo['switch_toggles']}")
            print(f"         | Raw Actions: {ppo['raw_actions_distribution']} -> Applied Actions: {ppo['applied_actions_distribution']}")
            print(f"         | OSPF Power: {ospf['mean_power_w']} W | OSPF Loss: {ospf['mean_loss_percent']:.2f}% | OSPF Delay: {ospf['mean_delay_ms']} ms")
        ov = r["overall"]
        print(f"  OVERALL | PPO Energy: {ov['ppo_total_energy_kwh']:.4f} kWh vs OSPF: {ov['ospf_total_energy_kwh']:.4f} kWh | Save: {ov['overall_saving_percent']}% | Mean Loss: {ov['overall_mean_loss_percent']:.4f}% | SLA Viol: {ov['overall_sla_violation_rate_percent']}% | Override: {ov['overall_override_rate_percent']}% | Meets Safety: {ov['meets_safety_criteria']}")
