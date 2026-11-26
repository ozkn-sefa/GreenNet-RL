import numpy as np
from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent

env = GreenNetworkEnv(topology_type='fattree_k6', k=6)
agent = GreenRLAgent(env=env, algo='PPO')
env.traffic_gen.set_intensity(0.35)
obs, _ = env.reset()

base_env = agent._get_base_env()
core_switches = getattr(base_env, "core_switches", [])
num_cores = len(core_switches)
print("num_cores:", num_cores)

link_utils = obs[num_cores:-3]
print("len link_utils:", len(link_utils))
print("max_util:", float(np.max(link_utils)))
print("avg_util:", float(np.mean(link_utils)))
print("critical_count:", int(np.sum(link_utils >= 0.95)))
print("congested_count:", int(np.sum(link_utils >= 0.80)))

norm_demand = float(obs[-2])
demand_mbps = norm_demand * 10000.0
print("demand_mbps:", demand_mbps)

is_burst = bool(getattr(getattr(base_env, "traffic_gen", None), "burst_mode", False))
intensity = float(getattr(getattr(base_env, "traffic_gen", None), "traffic_intensity", 1.0))
print("is_burst:", is_burst, "intensity:", intensity)

cond1 = (is_burst or int(np.sum(link_utils >= 0.95)) > 0 or float(np.max(link_utils)) >= 0.88 or demand_mbps >= 11000 or intensity >= 2.2)
print("cond1 (emergency):", cond1)

