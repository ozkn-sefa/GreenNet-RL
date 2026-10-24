import json

with open('results/pilot_fattree_k4_100k_v2_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for rec in data['validation_records']:
    s = rec['step']
    if s in [20000, 50000, 80000, 100000]:
        print(f"=== STEP {s} ===")
        for reg_name in ['Low', 'Medium', 'High', 'Burst']:
            r = rec['regimes'][reg_name]
            p = r['ppo']
            o = r['ospf']
            print(f"[{reg_name:6s}] PPO_save={r['ppo_saving_percent']:6.2f}% | loss={p['mean_loss_percent']:6.2f}% | delay={p['mean_delay_ms']:6.3f}ms | p95={p['p95_delay_ms']:6.3f}ms | sla={p['sla_violation_rate_percent']:5.1f}% | override={p['override_rate_percent']:5.1f}% | toggles={p['switch_toggles']:3d} || OSPF: loss={o['mean_loss_percent']:6.2f}% | delay={o['mean_delay_ms']:6.3f}ms | sla={o['sla_violation_rate_percent']:5.1f}%")
