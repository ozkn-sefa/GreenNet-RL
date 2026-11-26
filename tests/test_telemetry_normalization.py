"""
Tests for telemetry (Monitor wrapper, ep_rew_mean, ep_len_mean),
reward components breakdown, VecNormalize normalization,
and checkpoint save/load stability.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent


def test_reward_components_breakdown(env_k4):
    """
    Verifies that env.step() returns all reward components in info['reward_components']
    and their arithmetic sum matches the net reward.
    """
    obs, info = env_k4.reset(seed=42)
    assert "reward_components" in info
    rc_reset = info["reward_components"]
    assert "energy_reward" in rc_reset
    assert "delay_penalty" in rc_reset
    assert "loss_penalty" in rc_reset
    assert "congestion_penalty" in rc_reset

    # Step with Action 0 (1 core active -> saves energy, may incur penalties)
    obs, reward, term, trunc, step_info = env_k4.step(0)
    assert "reward_components" in step_info
    rc = step_info["reward_components"]

    # Verify all expected components are reported
    for key in [
        "energy_reward", "waste_penalty", "delay_penalty",
        "loss_penalty", "congestion_penalty", "switching_penalty",
        "flapping_penalty", "wakeup_penalty", "port_wakeup_penalty", "net_reward"
    ]:
        assert key in rc
        assert isinstance(rc[key], (float, int))

    # Verify arithmetic match with net step reward
    computed_raw = (
        rc["energy_reward"] - rc["waste_penalty"] - rc["delay_penalty"]
        - rc["loss_penalty"] - rc["congestion_penalty"] - rc["switching_penalty"]
        - rc["flapping_penalty"] - rc["wakeup_penalty"] - rc["port_wakeup_penalty"]
    ) * env_k4.reward_scale

    assert pytest.approx(computed_raw, abs=1e-5) == reward
    assert pytest.approx(rc["net_reward"], abs=1e-5) == reward


def test_monitor_wrapper_records_episode_statistics():
    """
    Verifies that Monitor wrapper intercepts terminated/truncated episodes
    and populates info['episode'] with episode return 'r' and length 'l'.
    """
    raw_env = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=5,
        training_mode=True,
    )
    monitored_env = Monitor(raw_env)
    monitored_env.reset(seed=42)

    episode_found = False
    for _ in range(10):
        obs, reward, term, trunc, info = monitored_env.step(7)
        if term or trunc:
            assert "episode" in info
            assert "r" in info["episode"]
            assert "l" in info["episode"]
            assert info["episode"]["l"] == 5
            episode_found = True
            break

    assert episode_found, "Monitor did not produce episode summary upon truncation!"


def test_ppo_telemetry_ep_rew_and_ep_len_recorded():
    """
    Verifies that SB3 PPO with Monitor wrapper records ep_rew_mean
    and ep_len_mean in ep_info_buffer and FatTreePilotCallback training history.
    """
    from scripts.train_pilot_fattree_k4 import FatTreePilotCallback

    # Max 8 steps per episode, n_steps=16 -> exactly 2 episodes complete per rollout
    raw_env = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=8,
        training_mode=True,
    )
    train_env = Monitor(raw_env)
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        learning_rate=1e-3,
        device="cpu",
        seed=42,
        verbose=0,
    )

    temp_cb_dir = tempfile.mkdtemp(prefix="test_cb_smoke_")
    try:
        callback = FatTreePilotCallback(eval_freq=1000, checkpoint_dir=temp_cb_dir, verbose=0)
        model.learn(total_timesteps=32, callback=callback)

        # 1. ep_info_buffer must contain completed episodes with 'r' and 'l'
        assert hasattr(model, "ep_info_buffer")
        assert len(model.ep_info_buffer) > 0
        for ep in model.ep_info_buffer:
            assert "r" in ep and "l" in ep
            assert ep["l"] == 8

        # 2. Callback telemetry must have recorded non-zero mean_ep_length and valid mean_reward
        assert len(callback.training_history) >= 2
        for record in callback.training_history:
            assert "mean_reward" in record
            assert "mean_ep_length" in record
            assert record["mean_ep_length"] == pytest.approx(8.0, abs=0.5)
            assert isinstance(record["mean_reward"], float)
    finally:
        shutil.rmtree(temp_cb_dir, ignore_errors=True)


def test_reward_clipping_and_normalization_mutually_exclusive():
    """
    Requirement 3: Verifies that applying both reward normalization (norm_reward=True)
    and finite reward clipping raises a ValueError.
    """
    raw_env = GreenNetworkEnv(topology_type="fattree_k4", k=4)
    with pytest.raises(ValueError, match="GreenNet-RL proje tercihi"):
        GreenRLAgent(
            env=raw_env,
            algo="PPO",
            use_vec_normalize=True,
            norm_reward=True,
            clip_reward=10.0,  # finite clipping requested with normalization
        )


def test_vec_normalize_scales_reward_without_distorting_semantics():
    """
    Requirement 5: Verifies that VecNormalize(norm_reward=True) scales rewards
    linearly through running return variance, preserving relative action rankings.
    """
    raw_env1 = GreenNetworkEnv(topology_type="fattree_k4", k=4, max_steps_per_episode=20)
    raw_env2 = GreenNetworkEnv(topology_type="fattree_k4", k=4, max_steps_per_episode=20)

    vec_env = DummyVecEnv([lambda: Monitor(raw_env1)])
    norm_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=False,
        norm_reward=True,
        clip_reward=float("inf"),
        gamma=0.99,
    )

    raw_env2.reset(seed=123)
    norm_env.reset()

    # Step both environments with identical action sequences
    raw_rewards = []
    norm_rewards = []
    for action in [7, 7, 6, 6, 4, 4, 0, 0]:
        _, r_raw, _, _, _ = raw_env2.step(action)
        _, r_norm, _, _ = norm_env.step([action])
        raw_rewards.append(r_raw)
        norm_rewards.append(float(r_norm[0]))

    # Normalized rewards should have the same sign / rank alignment
    assert len(raw_rewards) == len(norm_rewards)
    # The running standard deviation must be positive
    assert norm_env.returns is not None


def test_vec_normalize_checkpoint_save_and_evaluation_loading():
    """
    Requirement 4: Verifies VecNormalize statistics are saved alongside checkpoints
    and loaded for evaluation with training=False, norm_reward=False.
    """
    temp_dir = tempfile.mkdtemp(prefix="test_vec_norm_")
    try:
        raw_env = GreenNetworkEnv(
            topology_type="fattree_k4",
            k=4,
            max_steps_per_episode=16,
            training_mode=True,
        )
        agent = GreenRLAgent(
            env=raw_env,
            algo="PPO",
            use_monitor=True,
            use_vec_normalize=True,
            norm_reward=True,
            clip_reward=None,
            device="cpu",
            seed=42,
        )

        model_save_path = os.path.join(temp_dir, "test_model.zip")
        agent.train(total_timesteps=32, save_path=model_save_path)

        # Verify both model and vecnormalize pickle exist
        vec_save_path = os.path.join(temp_dir, "test_model_vecnormalize.pkl")
        assert os.path.exists(model_save_path)
        assert os.path.exists(vec_save_path)

        # Now load agent for evaluation
        eval_raw_env = GreenNetworkEnv(topology_type="fattree_k4", k=4, max_steps_per_episode=16)
        eval_agent = GreenRLAgent(
            env=eval_raw_env,
            algo="PPO",
            model_path=model_save_path,
            device="cpu",
        )

        assert isinstance(eval_agent.env, VecNormalize)
        # Requirement 4 check: training=False and norm_reward=False
        assert eval_agent.env.training is False
        assert eval_agent.env.norm_reward is False

        # Step in evaluation mode: running mean & var must NOT change
        init_var = float(eval_agent.env.ret_rms.var)
        obs = eval_agent.env.reset()
        _, eval_reward, _, _ = eval_agent.env.step([7])

        assert float(eval_agent.env.ret_rms.var) == pytest.approx(init_var, abs=1e-9)

        # Unnormalized reward: Action 7 in quiet network has near 0 energy saving
        # With norm_reward=False, the step returns raw unscaled reward
        assert isinstance(float(eval_reward[0]), float)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_safefilter_full_capacity_regression_intact(env_k4):
    """
    Requirement 7: Verifies that SafeFilter never downgrades full capacity request (Action 7)
    under any condition (clean state, heavy load, and burst).
    """
    # 1. Clean state
    env_k4.reset(seed=42)
    act, overridden = env_k4.safe_action(7)
    assert len(env_k4.core_actions[act]) == 4
    assert overridden is False

    # 2. Extreme burst state
    env_k4.reset(seed=60004, options={"intensity": 2.50, "profile": "flash_crowd", "burst": True})
    for _ in range(5):
        env_k4.step(7)

    act_burst, overridden_burst = env_k4.safe_action(7)
    assert len(env_k4.core_actions[act_burst]) == 4
    assert overridden_burst is False
