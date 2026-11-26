"""
Fat-Tree k=6 (9 Core, 54 Host) 8-Kademeli Performans, Karbon ve Proaktif Tahmin Benchmark Betigi.
Kullanim:
    python scripts/benchmark_k6.py
"""

import numpy as np
from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent


def main():
    print("=" * 112)
    print(" [GREEN DCN] FAT-TREE k=6 — SAF SINIR AGI vs. KALKANLI MODEL, KARBON & PROAKTIF TAHMIN RAPORU")
    print("=" * 112)

    env_raw = GreenNetworkEnv(topology_type="fattree_k6", max_steps_per_episode=15)
    agent_raw = GreenRLAgent(env=env_raw, device="cpu", use_shield=False)

    env_shield = GreenNetworkEnv(topology_type="fattree_k6", max_steps_per_episode=15)
    agent_shield = GreenRLAgent(env=env_shield, device="cpu", use_shield=True)

    intensities = [0.22, 0.35, 0.55, 0.85, 1.15, 1.50, 1.85, 2.40]
    print(
        f"{'Yük':<6} | {'Saf NN Core':<11} | {'Kalkanlı Core':<13} | {'Trafik(Mbps)':<12} | "
        f"{'Tahmin Doğruluğu':<16} | {'OSPF Tasarruf':<13} | {'CO2 Kazancı':<14} | {'RTT(ms)':<7}"
    )
    print("-" * 112)

    for inten in intensities:
        obs_r, _ = env_raw.reset(seed=100, options={"intensity": inten})
        obs_s, _ = env_shield.reset(seed=100, options={"intensity": inten})
        # 1 isinma adimi (baslangic telemetrisi olusmasi icin)
        obs_r, _, _, _, _ = env_raw.step(3)
        obs_s, _, _, _, info_0 = env_shield.step(3)

        raw_cores, shield_cores, demands, savings, co2_saved, rtts, pred_errs = [], [], [], [], [], [], []
        prev_pred = info_0.get("predicted_next_demand_mbps", info_0["total_demand_mbps"])

        for _ in range(8):
            a_r = agent_raw.predict(obs_r, deterministic=True)
            a_s = agent_shield.predict(obs_s, deterministic=True)
            obs_r, _, _, _, info_r = env_raw.step(a_r)
            obs_s, _, _, _, info_s = env_shield.step(a_s)

            actual_d = info_s["total_demand_mbps"]
            if prev_pred is not None and actual_d > 0:
                pred_errs.append(abs(prev_pred - actual_d) / actual_d * 100.0)
            prev_pred = info_s.get("predicted_next_demand_mbps", actual_d)

            raw_cores.append(len(info_r["active_cores"]))
            shield_cores.append(len(info_s["active_cores"]))
            demands.append(actual_d)
            savings.append(info_s["net_sleep_savings_percent"])
            co2_saved.append(info_s["carbon_metrics"]["saved_carbon_vs_ospf_g_hr"])
            rtts.append(info_s["avg_delay_ms"])

        acc = max(0.0, 100.0 - np.mean(pred_errs)) if pred_errs else 100.0
        print(
            f"%{int(inten*100):<5} | {np.mean(raw_cores):<11.1f} | {np.mean(shield_cores):<13.1f} | "
            f"{np.mean(demands):<12.0f} | %{acc:<15.1f} | %{np.mean(savings):<12.2f} | "
            f"{np.mean(co2_saved):<9.1f} g/sa | {np.mean(rtts):<7.2f}"
        )
    print("=" * 112)


if __name__ == "__main__":
    main()

