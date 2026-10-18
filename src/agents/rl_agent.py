"""
Reinforcement Learning Agent Wrapper using Stable-Baselines3 (PPO & DQN).
Provides automated training, checkpointing, topology-aware inference,
and dynamic elastic green policy fallback for SDN networks.
"""

import os
from typing import Optional
import gymnasium as gym
from stable_baselines3 import PPO, DQN
import numpy as np


class GreenRLAgent:
    """
    Wrapper for training and running Deep Reinforcement Learning agents on GreenNetworkEnv.
    Supports PPO (recommended for continuous state & discrete action policy learning) and DQN.
    Includes Dynamic Elastic Fallback Policy for seamless zero-shot generalization
    across any topology dimension (k=4, k=6, WAN/NSFNET).
    """

    def __init__(
        self,
        env: gym.Env,
        algo: str = "PPO",
        model_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        seed: int = 42
    ):
        self.env = env
        self.algo_name = algo.upper()
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None
        self.is_trained = False
        self.last_predicted_action: Optional[int] = None

        # Resolve topology-specific model path
        target_path = self._resolve_model_path(model_path)
        if target_path and os.path.exists(target_path):
            self.load(target_path)
        else:
            self._init_model(learning_rate, seed)

    def _resolve_model_path(self, explicit_path: Optional[str] = None) -> Optional[str]:
        """Finds best matching model file for current environment topology."""
        topo = getattr(self.env, "topology_type", "fattree_k4")
        if explicit_path and os.path.exists(explicit_path):
            return explicit_path

        topo_path = os.path.join("models", f"ppo_green_agent_{topo}.zip")
        if os.path.exists(topo_path):
            return topo_path

        # Only use generic ppo_green_agent.zip if topology is fattree_k4
        if topo in ["fattree_k4", "k4"]:
            default_path = os.path.join("models", "ppo_green_agent.zip")
            if os.path.exists(default_path):
                return default_path

        return None

    def _init_model(self, lr: float, seed: int):
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

        if self.algo_name == "PPO":
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=lr,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                gamma=0.98,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=policy_kwargs,
                verbose=0,
                seed=seed
            )
        elif self.algo_name == "DQN":
            self.model = DQN(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=lr,
                buffer_size=10000,
                learning_starts=500,
                batch_size=64,
                tau=1.0,
                gamma=0.98,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=250,
                exploration_fraction=0.2,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                verbose=0,
                seed=seed
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algo_name}. Choose PPO or DQN.")

    def train(self, total_timesteps: int = 30000, save_path: str = "models/ppo_green_agent.zip"):
        """Trains the agent on the environment and saves model weights."""
        if self.model is None:
            self._init_model(self.learning_rate, self.seed)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[*] Starting {self.algo_name} training for {total_timesteps} timesteps on {getattr(self.env, 'topology_type', 'default')}...")
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(save_path)
        self.is_trained = True
        print(f"[+] Model saved successfully to: {save_path}")

    def predict_elastic(self, observation: np.ndarray) -> int:
        """
        Dinamik Elastik Yeşil Ağ Politikası (Zero-Shot Adaptive Policy):
        Model ağırlıkları mevcut topolojiyle uyuşmadığında veya henüz eğitilmediğinde devreye girer.
        Ağdaki canlı darboğazı, anlık trafik talebini, link doluluklarını ve patlama (burst) durumunu
        izleyerek aktif omurga (core) sayısını dinamik olarak ayarlar.
        Darboğaz veya patlama oluştuğunda hemen ek omurga düğümlerini uyandırır (açılmayan node kalmaz!);
        trafik azaldığında ise kademeli olarak uyku moduna alıp enerjiden tasarruf sağlar.
        """
        num_cores = len(self.env.core_switches) if hasattr(self.env, "core_switches") else 4
        num_actions = len(self.env.core_actions) if hasattr(self.env, "core_actions") else 8
        max_action_idx = num_actions - 1

        # Link utilizations from observation vector
        if len(observation) >= num_cores + 3:
            link_utils = observation[num_cores:-3]
            max_util = float(np.max(link_utils)) if len(link_utils) > 0 else 0.0
            avg_util = float(np.mean(link_utils)) if len(link_utils) > 0 else 0.0
            congested_count = int(np.sum(link_utils >= 0.80))
            critical_count = int(np.sum(link_utils >= 0.95))
        else:
            max_util = 0.0
            avg_util = 0.0
            congested_count = 0
            critical_count = 0

        # Traffic demand from observation [-2] (normalized by 10,000)
        norm_demand = float(observation[-2]) if len(observation) >= 2 else 0.0
        demand_mbps = norm_demand * 10000.0

        # Check burst mode on traffic generator
        is_burst = False
        if hasattr(self.env, "traffic_gen") and getattr(self.env.traffic_gen, "burst_mode", False):
            is_burst = True

        # 1. ACİL DURUM / PATLAMA / AŞIRI DARBOĞAZ:
        # Ağ kırmızıya döndüğünde (congested linkler veya max_util >= 0.90 veya ani patlama veya >12000 Mbps talep)
        # tüm omurga switch'lerini anında AÇARAK darboğazı anında dağıt!
        if is_burst or critical_count > 0 or max_util >= 0.90 or demand_mbps >= 12000:
            return max_action_idx  # Action 7 (Tüm omurgalar aktif)

        # 2. YÜKSEK TRAFİK / TEHLİKELİ DOLULUK (utilization >= %75 veya hat dolulukları)
        if congested_count >= 2 or max_util >= 0.75 or demand_mbps >= 7000:
            return min(max_action_idx, max(5, max_action_idx - 1))

        # 3. ORTA-YÜKSEK TRAFİK (utilization >= %60)
        if max_util >= 0.60 or demand_mbps >= 4000:
            return min(max_action_idx, 4)

        # 4. NORMAL DENGELİ TRAFİK (utilization >= %45)
        if max_util >= 0.45 or demand_mbps >= 2000:
            return min(max_action_idx, 3)

        # 5. HAFİF-ORTA TRAFİK
        if max_util >= 0.30 or demand_mbps >= 1000:
            return min(max_action_idx, 2)

        # 6. DÜŞÜK TRAFİK
        if max_util >= 0.15 or demand_mbps >= 450:
            return min(max_action_idx, 1)

        # 7. SAKİN GECE / MAKSİMUM YEŞİL TASARRUF: 1 Omurga yeterli
        return 0

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Deep Reinforcement Learning Policy Inference with SLA Shielding:
        Model eğitilmiş ve topolojiye uyumluysa derin sinir ağı karar verir.
        Eğer model mevcut değilse veya kritik bir darboğaz durumunda model yetersiz
        aksiyon üretmişse, SLA Güvenlik Kalkanı (Safe RL Guard) devreye girerek
        ağdaki tıkanıklığı önlemek için dinamik elastik aksiyona geçer.
        """
        action = None
        if self.model is not None and self.is_trained:
            try:
                raw_act, _ = self.model.predict(observation, deterministic=deterministic)
                action = int(raw_act)
            except Exception:
                action = None

        if action is None:
            action = self.predict_elastic(observation)
        else:
            # SLA Safety Shield (Safe RL Guard):
            # Trafik patlaması veya aşırı darboğaz durumunda ağın kırmızı kalmasını önler
            num_cores = len(self.env.core_switches) if hasattr(self.env, "core_switches") else 4
            if len(observation) >= num_cores + 3:
                link_utils = observation[num_cores:-3]
                max_util = float(np.max(link_utils)) if len(link_utils) > 0 else 0.0
                critical_count = int(np.sum(link_utils >= 0.90))
                is_burst = hasattr(self.env, "traffic_gen") and getattr(self.env.traffic_gen, "burst_mode", False)
                if is_burst or critical_count >= 2 or max_util >= 0.95:
                    action = self.predict_elastic(observation)

        self.last_predicted_action = action
        return action

    def load(self, model_path: str):
        """Loads trained model weights (auto-detects PPO or DQN) or activates dynamic elastic policy if topology dimension changes."""
        print(f"[*] Loading model from {model_path}...")
        try:
            self.model = PPO.load(model_path, env=self.env)
            self.algo_name = "PPO"
            self.is_trained = True
            print("[+] Loaded as PPO model.")
        except Exception as e:
            try:
                self.model = DQN.load(model_path, env=self.env)
                self.algo_name = "DQN"
                self.is_trained = True
                print("[+] Loaded as DQN model.")
            except Exception as e2:
                print(f"[*] Bilgi: Model ağırlıkları mevcut topoloji boyutuyla eşleşmiyor ({e2}). Dinamik elastik politika aktif.")
                self.model = None
                self.is_trained = False
