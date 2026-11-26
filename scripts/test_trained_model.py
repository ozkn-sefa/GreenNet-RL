from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent

env = GreenNetworkEnv(topology_type='fattree_k6', k=6)
agent = GreenRLAgent(env=env, algo='PPO', model_path='models/ppo_green_agent_fattree_k6.zip', use_shield=False)

print("=" * 75)
print("EĞİTİLMİŞ SAF PPO SİNİR AĞI KARAR TESTİ (HİÇBİR KURAL / IF-ELSE YOK)")
print("Model Yüklendi mi:", agent.is_trained)
print("Model Tipi       :", type(agent.model))
print("=" * 75)

tests = [
    (0.30, "Gece / Düşük Yük (%30)"),
    (0.85, "Normal İş Günü (%85)"),
    (1.40, "Yoğun Veri Akışı (%140)"),
    (2.50, "Ani Patlama / Peak (%250)")
]

for intensity, name in tests:
    env.traffic_gen.set_intensity(intensity)
    if "Burst" in name:
        env.traffic_gen.set_burst_mode(True)
    else:
        env.traffic_gen.set_burst_mode(False)
    
    obs, _ = env.reset()
    # Doğrudan modelin sinir ağı tahmini:
    raw_act, _ = agent.model.predict(obs, deterministic=True)
    obs, r, _, _, info = env.step(int(raw_act))
    
    active_cnt = len(info["active_cores"])
    sleep_cnt = 9 - active_cnt
    savings = info["energy_savings_percent"]
    delay = info["avg_delay_ms"]
    loss = info["packet_loss_percent"]
    
    print(f"{name:<28}: PPO Çıktısı: Eylem {raw_act} -> {active_cnt}/9 Omurga Aktif, {sleep_cnt} UYKUDA! | Tasarruf: %{savings} | Gecikme: {delay}ms | Kayıp: %{loss}")
print("=" * 75)

