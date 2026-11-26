from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent

env = GreenNetworkEnv(topology_type='fattree_k6', k=6)
agent = GreenRLAgent(env=env, algo='PPO')

tests = [
    (0.35, "Gece (%35)"),
    (0.85, "Normal (%85)"),
    (1.45, "Yogun (%145)"),
    (2.5, "Ani Patlama (Burst)")
]

print("=" * 75)
print("TEST SONUCLARI: TRAFIK DEGISIKLIGINE GORE AJANIN KARARLARI")
print("=" * 75)
for intensity, name in tests:
    env.traffic_gen.set_intensity(intensity)
    if "Burst" in name:
        env.traffic_gen.set_burst_mode(True)
    else:
        env.traffic_gen.set_burst_mode(False)
    
    obs, _ = env.reset()
    act = agent.predict(obs)
    obs, r, _, _, info = env.step(act)
    active_cnt = len(info["active_cores"])
    sleep_cnt = 9 - active_cnt
    savings = info["energy_savings_percent"]
    delay = info["avg_delay_ms"]
    print(f"{name:<20}: Action {act} -> {active_cnt}/9 Aktif, {sleep_cnt} UYKUDA! (Tasarruf: %{savings}, Gecikme: {delay}ms)")
print("=" * 75)

