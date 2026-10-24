"""
Reinforcement Learning Agent Wrapper using Stable-Baselines3 (PPO & DQN).
Provides automated training, checkpointing, topology-aware inference,
and dynamic elastic green policy fallback for SDN networks.
"""

import os
from typing import Optional
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
        seed: int = 42,
        use_shield: bool = True,
        device: str = "auto",
        use_monitor: bool = True,
        use_vec_normalize: bool = False,
        norm_reward: bool = True,
        clip_reward: Optional[float] = None,
    ):
        self.use_shield = use_shield  # False => raw policy output (used for unbiased benchmarking)
        self.algo_name = algo.upper()
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = device
        self.model = None
        self.is_trained = False
        self.last_predicted_action: Optional[int] = None
        self.use_monitor = use_monitor
        self.use_vec_normalize = use_vec_normalize
        self.norm_reward = norm_reward
        self.clip_reward = clip_reward

        # GreenNet-RL Proje Tercihi: Reward clipping ile dinamik varyans normalizasyonu
        # ceza skalasını ve QoS hassasiyetini korumak amacıyla bir arada uygulanmaz.
        # Bu kısıtlama SB3 kütüphane zorunluluğu değil, projenin ödül dinamiklerini koruma tasarım tercihidir.
        if use_vec_normalize and norm_reward and clip_reward is not None and clip_reward < float("inf"):
            raise ValueError(
                "GreenNet-RL proje tercihi: Reward clipping ve VecNormalize reward normalizasyonu "
                "birlikte uygulanmaz (QoS ceza skalasını korumak için unclipped VecNormalize tercih edilir). "
                "VecNormalize(norm_reward=True, clip_reward=inf) veya bağımsız clipping'den birini seçiniz."
            )

        # Wrap with Monitor if requested and not already a Monitor
        if use_monitor and not isinstance(env, (Monitor, VecNormalize, DummyVecEnv)):
            env = Monitor(env)

        # Wrap with VecNormalize if requested
        if use_vec_normalize and not isinstance(env, VecNormalize):
            if not isinstance(env, DummyVecEnv):
                env = DummyVecEnv([lambda: env])
            env = VecNormalize(
                env,
                training=True,
                norm_obs=False,
                norm_reward=norm_reward,
                clip_reward=float("inf"),
                gamma=0.99,
            )

        self.env = env

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
                seed=seed,
                device=self.device
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
                seed=seed,
                device=self.device
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
        vec_env = getattr(self.model, "get_vec_normalize_env", lambda: None)() or (self.env if isinstance(self.env, VecNormalize) else None)
        if vec_env is not None and hasattr(vec_env, "save"):
            vec_path = save_path.replace(".zip", "_vecnormalize.pkl")
            vec_env.save(vec_path)
            print(f"[+] VecNormalize statistics saved to: {vec_path}")
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
        elif self.use_shield:
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
        vec_path = model_path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vec_path):
            if not isinstance(self.env, VecNormalize):
                if not isinstance(self.env, DummyVecEnv):
                    self.env = DummyVecEnv([lambda: self.env])
                self.env = VecNormalize.load(vec_path, self.env)
            self.env.training = False
            self.env.norm_reward = False
            print(f"[+] Loaded VecNormalize statistics from {vec_path} (training=False, norm_reward=False)")
        try:
            self.model = PPO.load(model_path, env=self.env, device=self.device)
            self.algo_name = "PPO"
            self.is_trained = True
            print("[+] Loaded as PPO model.")
        except Exception as e:
            try:
                self.model = DQN.load(model_path, env=self.env, device=self.device)
                self.algo_name = "DQN"
                self.is_trained = True
                print("[+] Loaded as DQN model.")
            except Exception as e2:
                print(f"[*] Bilgi: Model ağırlıkları mevcut topoloji boyutuyla eşleşmiyor ({e2}). Dinamik elastik politika aktif.")
                self.model = None
                self.is_trained = False
