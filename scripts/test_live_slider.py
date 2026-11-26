from src.web.server import sim_engine

print("=" * 70)
print("CANLI SIMULATOR SLIDER VE TRAFIK TEPKI TESTI")
print("=" * 70)

for intensity, label in [(0.35, "Gece %35"), (0.85, "Normal %85"), (1.45, "Yogun %145"), (2.50, "Pik %250")]:
    sim_engine.set_intensity(intensity)
    data = sim_engine.trigger_step()
    active_cores = data.get("active_cores", [])
    total_cores = len(sim_engine.env.core_switches)
    sleep_count = total_cores - len(active_cores)
    demand = data.get("total_demand_mbps", 0)
    delay = data.get("avg_delay_ms", 0)
    savings = data.get("energy_savings_percent", 0)
    print(f"[{label:<10}]: Trafik={demand:<7} Mbps | Aktif={len(active_cores)}/{total_cores} ({sleep_count} UYKUDA) | Tasarruf=%{savings:<5} | Gecikme={delay}ms")

print("=" * 70)

