from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent

env = GreenNetworkEnv(topology_type='fattree_k6', k=6)
agent = GreenRLAgent(env=env, algo='PPO')
env.traffic_gen.set_intensity(0.35)
obs, _ = env.reset()

print("agent.model:", agent.model)
print("agent.is_trained:", agent.is_trained)
print("agent.use_shield:", agent.use_shield)

# Test raw elastic prediction:
elastic_act = agent.predict_elastic(obs)
print("Direct predict_elastic:", elastic_act)

# Test predict:
pred_act = agent.predict(obs)
print("predict:", pred_act)

