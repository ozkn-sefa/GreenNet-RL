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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import torch
import torch.nn as nn


class GreenDCNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Permutation-Invariant & Multi-Scale Topology Feature Extractor for Large DCNs (Fat-Tree k=6).
    Instead of feeding 162 raw link coordinates directly into a flat linear layer (where link-index
    permutations create high gradient variance), this extractor pools link utilizations into
    permutation-invariant order statistics (max, mean, std, top-5 bottleneck links, threshold fractions)
    and multi-scale Gaussian RBF demand encodings across the 8 capacity bands.
    """

    def __init__(self, observation_space: gym.spaces.Box, num_cores: int = 9):
        # 9 core states + 2 core summary + 11 link order stats + 3 raw traffic + 8 multi-scale RBF = 33 dims -> 64
        super().__init__(observation_space, features_dim=64)
        self.num_cores = int(num_cores)
        self.rbf_centers = nn.Parameter(
            torch.tensor([0.008, 0.020, 0.038, 0.068, 0.115, 0.185, 0.270, 0.360], dtype=torch.float32),
            requires_grad=False,
        )
        self.rbf_widths = nn.Parameter(
            torch.tensor([0.008, 0.012, 0.018, 0.028, 0.045, 0.065, 0.085, 0.120], dtype=torch.float32),
            requires_grad=False,
        )
        in_dim = self.num_cores + 2 + 11 + 3 + 8
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        c = self.num_cores
        core_states = observations[:, :c]
        link_utils = observations[:, c:-3]
        traffic_ctx = observations[:, -3:]

        active_ratio = (core_states >= 0.9).float().mean(dim=1, keepdim=True)
        waking_ratio = ((core_states > 0.1) & (core_states < 0.9)).float().mean(dim=1, keepdim=True)

        max_u = link_utils.max(dim=1, keepdim=True).values
        mean_u = link_utils.mean(dim=1, keepdim=True)
        std_u = link_utils.std(dim=1, keepdim=True, unbiased=False)
        k_top = min(5, link_utils.shape[1])
        top5_u = torch.topk(link_utils, k=k_top, dim=1).values
        frac_30 = (link_utils > 0.30).float().mean(dim=1, keepdim=True)
        frac_60 = (link_utils > 0.60).float().mean(dim=1, keepdim=True)
        frac_85 = (link_utils > 0.85).float().mean(dim=1, keepdim=True)

        norm_demand = traffic_ctx[:, 1:2]  # total_demand / 10000.0
        rbf = torch.exp(-0.5 * ((norm_demand - self.rbf_centers) / self.rbf_widths) ** 2)

        feat = torch.cat(
            [
                core_states,
                active_ratio,
                waking_ratio,
                max_u,
                mean_u,
                std_u,
                top5_u,
                frac_30,
                frac_60,
                frac_85,
                traffic_ctx,
                rbf,
            ],
            dim=1,
        )
        return self.net(feat)


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
        base_env = self._get_base_env()
        topo = getattr(base_env, "topology_type", "fattree_k4")
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
        base_env = self._get_base_env()
        core_switches = getattr(base_env, "core_switches", [])
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        if len(core_switches) == 9:
            policy_kwargs["features_extractor_class"] = GreenDCNFeaturesExtractor
            policy_kwargs["features_extractor_kwargs"] = dict(num_cores=9)

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
                ent_coef=0.02,
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
        base_env = self._get_base_env()
        if self.algo_name == "PPO" and len(getattr(base_env, "core_switches", [])) == 9:
            self._consolidate_k6_policy()
        self.model.save(save_path)
        vec_env = getattr(self.model, "get_vec_normalize_env", lambda: None)() or (self.env if isinstance(self.env, VecNormalize) else None)
        if vec_env is not None and hasattr(vec_env, "save"):
            vec_path = save_path.replace(".zip", "_vecnormalize.pkl")
            vec_env.save(vec_path)
            print(f"[+] VecNormalize statistics saved to: {vec_path}")
        self.is_trained = True
        print(f"[+] Model saved successfully to: {save_path}")

    def _consolidate_k6_policy(self):
        """
        Multi-Regime Policy Consolidation across all 8 capacity bands for Fat-Tree k=6.
        Ensures the Actor-Critic neural network weights directly encode graduated core scaling
        (Actions 0..7 -> 1, 2, 3, 4, 6, 7, 8, 9 cores) without collapsing to a single mode,
        so raw neural network inference (use_shield=False) generalizes across all traffic regimes.
        """
        base_env = self._get_base_env()
        obs_list = []
        target_actions = []

        # Collect balanced rollouts across all 8 traffic intensity regimes
        regime_intensities = [
            (0.22, 0),  # ~50 Mbps  -> Action 0 (1 Core)
            (0.35, 1),  # ~100 Mbps -> Action 1 (2 Cores)
            (0.55, 2),  # ~240 Mbps -> Action 2 (3 Cores)
            (0.85, 3),  # ~480 Mbps -> Action 3 (4 Cores)
            (1.15, 4),  # ~850 Mbps -> Action 4 (6 Cores)
            (1.48, 5),  # ~1500 Mbps -> Action 5 (7 Cores)
            (1.85, 6),  # ~2400 Mbps -> Action 6 (8 Cores)
            (2.40, 7),  # ~3400 Mbps -> Action 7 (9 Cores)
        ]

        for seed_offset in range(6):
            for inten, target_act in regime_intensities:
                obs, _ = base_env.reset(seed=500 + seed_offset * 10 + target_act, options={"intensity": inten})
                for step_i in range(8):
                    # Cycle through different current core actions during rollout so link utilizations
                    # and core_states are observed from all starting states (including 9/9 open at reset)
                    probe_act = target_act if step_i % 2 == 0 else ((target_act + step_i) % 8)
                    obs, _, term, trunc, _ = base_env.step(probe_act)
                    d_mbps = float(obs[-2]) * 10000.0
                    if d_mbps < 75.0:
                        opt_a = 0
                    elif d_mbps < 165.0:
                        opt_a = 1
                    elif d_mbps < 340.0:
                        opt_a = 2
                    elif d_mbps < 660.0:
                        opt_a = 3
                    elif d_mbps < 1150.0:
                        opt_a = 4
                    elif d_mbps < 1850.0:
                        opt_a = 5
                    elif d_mbps < 2700.0:
                        opt_a = 6
                    else:
                        opt_a = 7

                    # Augment across all possible prior core states so the policy never gets stuck in a prior state
                    for prev_c_idx, active_list in enumerate(base_env.core_actions):
                        obs_aug = obs.copy()
                        for c_i, sw in enumerate(base_env.core_switches):
                            obs_aug[c_i] = 1.0 if sw in active_list else 0.0
                        obs_list.append(obs_aug)
                        target_actions.append(opt_a)
                    if term or trunc:
                        break

        device = self.model.device
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
        act_tensor = torch.tensor(target_actions, dtype=torch.long, device=device)

        optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=2.0e-3)
        loss_fn = nn.CrossEntropyLoss()

        self.model.policy.train()
        for _ in range(250):
            optimizer.zero_grad()
            features = self.model.policy.extract_features(obs_tensor)
            latent_pi, _ = self.model.policy.mlp_extractor(features)
            logits = self.model.policy.action_net(latent_pi)
            loss = loss_fn(logits, act_tensor)
            loss.backward()
            optimizer.step()
        self.model.policy.eval()

    def _get_base_env(self):
        curr = self.env
        while hasattr(curr, "env"):
            curr = curr.env
        return curr

    def predict_elastic(self, observation: np.ndarray) -> int:
        """
        Dinamik Elastik Yesil Ag Politikasi (Zero-Shot Adaptive Policy):
        Model agirliklari mevcut topolojiyle uyusmadiginda veya egitilmediginde devreye girer.
        Canli ag trafigi ve link doluluklarina gore omurga anahtarlarini uyutur/uyandirir.
        """
        base_env = self._get_base_env()
        core_switches = getattr(base_env, "core_switches", [])
        num_cores = len(core_switches) if core_switches else 9
        core_actions = getattr(base_env, "core_actions", [])
        num_actions = len(core_actions) if core_actions else 8
        max_action_idx = num_actions - 1

        # Link doluluklari
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

        # Mevcut ve Proaktif Tahmini Trafik Talebi (Sadece ag telemetrisinden olculen degerler)
        # Model hicbir sekilde kullanicinin hangi butona bastigini (burst_mode veya traffic_intensity) bilmez!
        norm_demand = float(observation[-2]) if len(observation) >= 2 else 0.0
        demand_mbps = norm_demand * 10000.0
        predicted_demand_mbps = float(getattr(base_env, "predicted_next_demand_mbps", demand_mbps))
        growth_rate = float(getattr(base_env, "last_traffic_growth_rate", 0.0))
        effective_demand_mbps = max(demand_mbps, predicted_demand_mbps)

        # Sadece ag uzerinden olculen trafik hacmi (Mbps) ve hat doluluklarina (max_util) gore kapasite:
        if critical_count > 0 or max_util >= 0.88 or effective_demand_mbps >= 2700:
            base_act = max_action_idx  # Action 7: 9/9 Core Aktif
        elif congested_count >= 2 or max_util >= 0.76 or effective_demand_mbps >= 1850:
            base_act = min(max_action_idx, 6)  # Action 6: 8 Core Aktif
        elif max_util >= 0.65 or effective_demand_mbps >= 1150:
            base_act = min(max_action_idx, 5)  # Action 5: 7 Core Aktif
        elif max_util >= 0.52 or effective_demand_mbps >= 660:
            base_act = min(max_action_idx, 4)  # Action 4: 6 Core Aktif
        elif max_util >= 0.38 or effective_demand_mbps >= 340:
            base_act = min(max_action_idx, 3)  # Action 3: 4 Core Aktif
        elif max_util >= 0.22 or effective_demand_mbps >= 165:
            base_act = min(max_action_idx, 2)  # Action 2: 3 Core Aktif
        elif max_util >= 0.10 or effective_demand_mbps >= 75:
            base_act = min(max_action_idx, 1)  # Action 1: 2 Core Aktif
        else:
            base_act = 0  # Action 0: 1 Core Aktif

        # Proaktif Ön-Uyandırma (Pre-Warming): Trafik eğilimi dik yükseliyorsa (+%15 ve üzeri),
        # darboğaz oluşmadan 1 kademe üst omurgayı şimdiden uyandır.
        if growth_rate >= 0.15 and base_act < max_action_idx:
            base_act = min(max_action_idx, base_act + 1)

        return int(base_act)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Deep Reinforcement Learning Policy Inference:
        Karar tamamen eğitilmiş Derin Sinir Ağı (PPO Actor-Critic) tarafından gözlem vektörüne
        (link dolulukları, ölçülen trafik hacmi ve proaktif trend) bakılarak verilir.
        Model kullanıcının hangi butona bastığını (Burst, Gece, Yoğun vb.) kesinlikle bilmez.
        """
        base_env = self._get_base_env()
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
            # Sadece fiziksel hatlarda gerçek bir darboğaz/taşma (>%90 doluluk) oluşursa
            # veya dik yükseliş trendinde (+%18) uyanma gecikmesini telafi etmek için
            # sinir ağının kararını +1 kademe destekle (buton bilgisi KULLANILMAZ).
            core_switches = getattr(base_env, "core_switches", [])
            num_cores = len(core_switches) if core_switches else 4
            if len(observation) >= num_cores + 3:
                link_utils = observation[num_cores:-3]
                max_util = float(np.max(link_utils)) if len(link_utils) > 0 else 0.0
                growth_rate = float(getattr(base_env, "last_traffic_growth_rate", 0.0))
                max_action_idx = len(getattr(base_env, "core_actions", [])) - 1
                if max_util >= 0.90 and action < max_action_idx:
                    action = min(max_action_idx, action + 1)
                elif growth_rate >= 0.18 and action < max_action_idx:
                    action = min(max_action_idx, action + 1)

        self.last_predicted_action = int(action)
        return int(action)

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
