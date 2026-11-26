import json

old = json.load(open('results/pilot_fattree_k4_100k_v2_results.json', encoding='utf-8'))
new = json.load(open('results/pilot_fattree_k4_100k_v2_reevaluation.json', encoding='utf-8'))

old_recs = {r['step']: r for r in old['validation_records']}
new_recs = {r['step']: r for r in new['validation_records']}

print(f"{'Step':<6} | {'Regime':<8} | {'Old SLA':<8} | {'New SLA':<8} | {'Old Loss':<8} | {'New Loss':<8} | {'Old Sav':<8} | {'New Sav':<8} | {'Old Override':<12} | {'New Override':<12}")
print("-" * 105)

for s in [20000, 50000, 80000, 100000]:
    o_reg = old_recs[s]['regimes']
    n_reg = new_recs[s]['regimes']
    for r in ['Low', 'Medium', 'High', 'Burst']:
        o_p = o_reg[r]['ppo']
        n_p = n_reg[r]['ppo']
        o_sla = f"{o_p['sla_violation_rate_percent']:.1f}%"
        n_sla = f"{n_p['sla_violation_rate_percent']:.1f}%"
        o_loss = f"{o_p['mean_loss_percent']:.2f}%"
        n_loss = f"{n_p['mean_loss_percent']:.2f}%"
        o_sav = f"{o_reg[r]['ppo_saving_percent']:.2f}%"
        n_sav = f"{n_reg[r]['ppo_saving_percent']:.2f}%"
        o_ov = f"{o_p['override_rate_percent']:.1f}%"
        n_ov = f"{n_p['override_rate_percent']:.1f}%"
        print(f"{s:<6} | {r:<8} | {o_sla:<8} | {n_sla:<8} | {o_loss:<8} | {n_loss:<8} | {o_sav:<8} | {n_sav:<8} | {o_ov:<12} | {n_ov:<12}")
