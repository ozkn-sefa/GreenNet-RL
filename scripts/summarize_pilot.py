import json

def summarize():
    with open("results/pilot_fattree_k4_100k_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 115)
    print("FAT-TREE k=4 100K PILOT CHECKPOINT PROGRESSION SUMMARY")
    print("=" * 115)
    print(f"{'Step':<8} | {'Overall Sav%':<13} | {'Low Sav%':<10} | {'Med Sav%':<10} | {'High Sav%':<10} | {'Burst Sav%':<11} | {'Override%':<10} | {'PPO Raw Actions (Low Regime)'}")
    print("-" * 115)

    for r in data["validation_records"]:
        step = r["step"]
        ov = r["overall"]
        low_s = r["regimes"]["Low"]["ppo_saving_percent"]
        med_s = r["regimes"]["Medium"]["ppo_saving_percent"]
        high_s = r["regimes"]["High"]["ppo_saving_percent"]
        burst_s = r["regimes"]["Burst"]["ppo_saving_percent"]
        raw_dist = r["regimes"]["Low"]["ppo"]["raw_actions_distribution"]
        applied_dist = r["regimes"]["Low"]["ppo"]["applied_actions_distribution"]
        print(f"{step:<8} | {ov['overall_saving_percent']:>11.2f}% | {low_s:>8.2f}% | {med_s:>8.2f}% | {high_s:>8.2f}% | {burst_s:>9.2f}% | {ov['overall_override_rate_percent']:>8.1f}% | Raw: {raw_dist} -> App: {applied_dist}")

    print("\n" + "=" * 115)
    print("COMPARISON WITH BASELINES AT STEP 50000 (STEADY STATE)")
    print("=" * 115)
    step50 = next(r for r in data["validation_records"] if r["step"] == 50000)
    for reg_name in ["Low", "Medium", "High", "Burst"]:
        reg = step50["regimes"][reg_name]
        print(f"\n--- Regime: {reg_name} ---")
        print(f"{'Algorithm':<20} | {'Power (W)':<12} | {'Energy (kWh)':<14} | {'Saving (%)':<12} | {'Loss (%)':<10} | {'Delay (ms)':<10} | {'SLA Breach':<10} | {'Actions'}")
        print("-" * 115)
        for alg_key, alg_name in [("ppo", "PPO (Step 50k)"), ("ospf", "OSPF (All Cores)"), ("static", "Static Heuristic"), ("threshold", "Threshold Load")]:
            a_data = reg[alg_key]
            sav = reg.get(f"{alg_key}_saving_percent", 0.0)
            print(f"{alg_name:<20} | {a_data['mean_power_w']:<12.2f} | {a_data['total_energy_kwh']:<14.4f} | {sav:>10.2f}% | {a_data['mean_loss_percent']:<10.2f} | {a_data['mean_delay_ms']:<10.3f} | {a_data['sla_violation_rate_percent']:<10.1f}% | {a_data.get('applied_actions_distribution', {})}")

    # Learning Telemetry Summary
    print("\n" + "=" * 115)
    print("PPO LEARNING TELEMETRY SUMMARY")
    print("=" * 115)
    telem = data["training_telemetry"]
    if telem:
        print(f"Total Telemetry Log Entries: {len(telem)}")
        first = telem[0]
        mid = telem[len(telem)//2]
        last = telem[-1]
        for name, entry in [("Start (Step ~1k)", first), ("Mid (Step ~50k)", mid), ("End (Step ~100k)", last)]:
            print(f"{name:<18} : Reward={entry['mean_reward']:.2f}, Loss={entry['loss']:.4f}, PolicyLoss={entry['policy_loss']:.4f}, ValueLoss={entry['value_loss']:.4f}, Entropy={entry['entropy_loss']:.4f}, ApproxKL={entry['approx_kl']:.6f}, ExplVar={entry['explained_variance']:.4f}")

if __name__ == "__main__":
    summarize()
