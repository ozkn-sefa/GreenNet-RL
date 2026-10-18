"""
Live Training and Real-Time Evaluation Manager for Green SDN.
Streams training telemetry (reward curves, losses, action frequencies)
and synchronous Head-to-Head (OSPF vs RL) live testing over WebSockets.
"""

import threading
import time
import queue
from typing import Dict, Any, Optional
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback

from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent
from src.agents.baselines import OSPFBaselineAgent


class LiveTrainingCallback(BaseCallback):
    """
    SB3 Callback that streams training progress and episode rewards in real time.
    """

    def __init__(self, update_queue: queue.Queue, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.update_queue = update_queue
        self.total_timesteps = total_timesteps
        self.last_emit_time = time.time()
        self.current_episode = 0
        self.episode_rewards = []
        self.current_ep_reward = 0.0

    def _on_step(self) -> bool:
        # Check if stop requested
        if hasattr(self.training_env, "stop_requested") and self.training_env.stop_requested:
            return False

        # Accumulate reward
        reward = self.locals.get("rewards", [0.0])[0]
        self.current_ep_reward += reward

        done = self.locals.get("dones", [False])[0]
        if done:
            self.current_episode += 1
            self.episode_rewards.append(self.current_ep_reward)
            recent_mean = float(np.mean(self.episode_rewards[-20:])) if self.episode_rewards else self.current_ep_reward

            info = self.locals.get("infos", [{}])[0]
            savings = info.get("energy_savings_percent", 0.0)
            delay = info.get("avg_delay_ms", 1.0)
            active_cores = info.get("active_cores", [])

            progress_pct = round((self.num_timesteps / max(1, self.total_timesteps)) * 100, 1)

            payload = {
                "type": "train_progress",
                "timestep": self.num_timesteps,
                "total_timesteps": self.total_timesteps,
                "progress_pct": progress_pct,
                "episode": self.current_episode,
                "episode_reward": round(float(self.current_ep_reward), 2),
                "mean_reward": round(recent_mean, 2),
                "savings_pct": savings,
                "delay_ms": delay,
                "active_cores_count": len(active_cores),
                "status": "TRAINING"
            }
            self.update_queue.put(payload)
            self.current_ep_reward = 0.0

        return True


class LiveTrainingManager:
    """
    Manages live interactive RL training sessions in a background thread.
    """

    def __init__(self):
        self.active_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.telemetry_queue: queue.Queue = queue.Queue()
        self.is_training = False
        self.model = None

    def start_training(self, algo: str = "PPO", timesteps: int = 5000, lr: float = 3e-4, topology_type: str = "fattree_k4"):
        if self.is_training:
            return {"status": "error", "message": "Eğitim zaten devam ediyor."}

        self.stop_flag.clear()
        self.is_training = True

        def _train_worker():
            try:
                env = GreenNetworkEnv(topology_type=topology_type, max_steps_per_episode=120, training_mode=True)
                callback = LiveTrainingCallback(self.telemetry_queue, total_timesteps=timesteps)

                policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
                if algo == "PPO":
                    self.model = PPO(
                        "MlpPolicy",
                        env,
                        learning_rate=lr,
                        n_steps=128,
                        batch_size=32,
                        n_epochs=5,
                        policy_kwargs=policy_kwargs,
                        verbose=0
                    )
                else:
                    self.model = DQN(
                        "MlpPolicy",
                        env,
                        learning_rate=lr,
                        buffer_size=5000,
                        learning_starts=200,
                        batch_size=32,
                        verbose=0
                    )

                self.telemetry_queue.put({
                    "type": "train_start",
                    "algo": algo,
                    "topology": topology_type,
                    "total_timesteps": timesteps,
                    "message": f"{algo} ({topology_type}) Ajan Eğitimi Başlatıldı..."
                })

                self.model.learn(total_timesteps=timesteps, callback=callback)

                # Save topology-specific and default model
                import os
                os.makedirs("models", exist_ok=True)
                self.model.save(f"models/{algo.lower()}_green_agent_{topology_type}.zip")
                if topology_type == "fattree_k4":
                    self.model.save("models/ppo_green_agent.zip")

                self.telemetry_queue.put({
                    "type": "train_complete",
                    "status": "COMPLETED",
                    "message": f"Model başarıyla eğitildi ve models/{algo.lower()}_green_agent_{topology_type}.zip dosyasına kaydedildi!"
                })
            except Exception as e:
                self.telemetry_queue.put({
                    "type": "train_error",
                    "status": "ERROR",
                    "error": str(e)
                })
            finally:
                self.is_training = False

        self.active_thread = threading.Thread(target=_train_worker, daemon=True)
        self.active_thread.start()
        return {"status": "ok", "message": f"{algo} eğitimi başlatıldı."}

    def stop_training(self):
        if self.is_training:
            self.stop_flag.set()
            self.is_training = False
            return {"status": "ok", "message": "Eğitim durduruldu."}
        return {"status": "error", "message": "Aktif bir eğitim bulunmuyor."}


class LiveHeadToHeadArena:
    """
    Executes a side-by-side battle between OSPF (Classic) and RL Agent (Ours)
    using identical random seeds and simultaneous traffic bursts.
    """

    def __init__(self, topology_type: str = "fattree_k4"):
        self.topology_type = topology_type
        self.env_ospf = GreenNetworkEnv(topology_type=topology_type, max_steps_per_episode=500)
        self.env_rl = GreenNetworkEnv(topology_type=topology_type, max_steps_per_episode=500)
        self.ospf_agent = OSPFBaselineAgent()
        self.rl_agent = None
        self.step_idx = 0
        self.total_joules_ospf = 0.0
        self.total_joules_rl = 0.0
        self.reset(seed=101)

    def load_rl_agent(self, model_path: Optional[str] = None):
        self.rl_agent = GreenRLAgent(env=self.env_rl, algo="PPO", model_path=model_path)

    def reset(self, seed: int = 101):
        self.step_idx = 0
        self.total_joules_ospf = 0.0
        self.total_joules_rl = 0.0
        self.obs_ospf, self.info_ospf = self.env_ospf.reset(seed=seed)
        self.obs_rl, self.info_rl = self.env_rl.reset(seed=seed)
        # Sync traffic generators to identical seeds
        self.env_ospf.traffic_gen = self.env_rl.traffic_gen

    def step(self) -> Dict[str, Any]:
        self.step_idx += 1

        # 1. OSPF Step
        act_ospf = self.ospf_agent.predict(self.obs_ospf)
        self.obs_ospf, r_ospf, _, _, self.info_ospf = self.env_ospf.step(act_ospf)

        # 2. RL Step
        if self.rl_agent:
            act_rl = self.rl_agent.predict(self.obs_rl, deterministic=True)
        else:
            act_rl = 4  # default balanced
        self.obs_rl, r_rl, _, _, self.info_rl = self.env_rl.step(act_rl)

        # Accumulate Joules (Watts * 1 sec)
        ospf_w = self.info_ospf["total_power_w"]
        rl_w = self.info_rl["total_power_w"]
        self.total_joules_ospf += ospf_w
        self.total_joules_rl += rl_w

        joules_saved = max(0.0, self.total_joules_ospf - self.total_joules_rl)
        percent_saved = max(0.0, ((ospf_w - rl_w) / max(1.0, ospf_w)) * 100.0)

        return {
            "type": "battle_step",
            "step": self.step_idx,
            "ospf": {
                "power_w": ospf_w,
                "delay_ms": self.info_ospf["avg_delay_ms"],
                "active_cores": self.info_ospf["active_cores"],
                "congested": self.info_ospf["congested_links_count"],
                "packet_loss": self.info_ospf["packet_loss_percent"],
                "total_joules": round(self.total_joules_ospf, 1)
            },
            "rl": {
                "power_w": rl_w,
                "delay_ms": self.info_rl["avg_delay_ms"],
                "active_cores": self.info_rl["active_cores"],
                "congested": self.info_rl["congested_links_count"],
                "packet_loss": self.info_rl["packet_loss_percent"],
                "total_joules": round(self.total_joules_rl, 1)
            },
            "comparison": {
                "watts_saved": round(ospf_w - rl_w, 1),
                "percent_saved": round(percent_saved, 2),
                "cumulative_joules_saved": round(joules_saved, 1),
                "carbon_saved_grams": round(joules_saved * 0.00011, 2)  # g CO2
            }
        }
