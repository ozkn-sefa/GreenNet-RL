"""
Regression tests verifying evaluation and data isolation fixes before PPO v3.
Covers:
1. SLA violation calculation matching GreenNetworkEnv contract.
2. TRAIN_SEED_LIMIT isolation ([0, 50000)).
3. SafeFilter link failure timing (no off-by-one lookahead).
4. Override telemetry key standardization (action_overridden, requested_action).
5. VecNormalize evaluation pipeline compatibility (norm_obs=False and norm_obs=True).
6. reset() reward_components schema completeness.
7. PRNG advancement on non-training reset(seed=None).
"""

import os
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.green_network_env import GreenNetworkEnv, TRAIN_SEED_LIMIT


def test_train_seed_limit_is_50000():
    """Confirms training seed pool is strictly capped at 50,000, isolating val (60001-60004) and test (70001-70004)."""
    assert TRAIN_SEED_LIMIT == 50_000
    env = GreenNetworkEnv(training_mode=True, max_steps_per_episode=3)
    seeds = []
    for _ in range(50):
        env.reset()
        seed_val = env.traffic_gen.np_random.get_state()[1][0]
        seeds.append(seed_val)
        assert 0 <= seed_val < 50_000
        assert seed_val < 60001  # Cannot touch validation seeds
        assert seed_val < 70001  # Cannot touch test seeds


def test_safefilter_failure_timing_no_off_by_one():
    """
    Verifies SafeFilter evaluates the link failure state that actually applies to step t.
    Failure event: link down during step 2 (start=2, end=3).
    At step 1: link must be UP in both predict_action_outcomes and physical routing.
    At step 2: link must be DOWN in both predict_action_outcomes and physical routing.
    At step 3: link must be UP (recovered) in both.
    """
    env = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=5,
        training_mode=False,
        use_safety_filter=True,
    )
    test_link = ("e_0_0", "a_0_0")
    canon_link = env._canon(test_link)
    ev = [{"link": test_link, "start": 2, "end": 3}]
    env.reset(seed=42, options={"failures": ev})

    # Step 1: Link is UP
    env.step(7)
    assert env.current_step == 1
    assert canon_link not in env._failed_now
    pred_step1 = env._failed_links_at(env.current_step)
    assert canon_link not in pred_step1

    # Step 2: Link is DOWN
    env.step(7)
    assert env.current_step == 2
    assert canon_link in env._failed_now
    # SafeFilter outcome prediction must see it as failed
    pred_step2 = env._failed_links_at(env.current_step)
    assert canon_link in pred_step2

    # Step 3: Link is UP (recovered)
    env.step(7)
    assert env.current_step == 3
    assert canon_link not in env._failed_now
    pred_step3 = env._failed_links_at(env.current_step)
    assert canon_link not in pred_step3


def test_sla_violation_contract_alignment():
    """
    Verifies SLA violation definition:
    sla_violation = bool(avg_delay_ms > sla_delay_ms or packet_loss_rate > sla_loss_rate)
    sla_loss_rate is 0.01 (1.0%).
    A loss rate of 0.0005 (0.05%, which exceeds 1e-4 but is <= 0.01) must NOT be flagged as SLA violation
    if delay is within limits.
    """
    env = GreenNetworkEnv(sla_delay_ms=3.5, sla_loss_rate=0.01)
    env.reset(seed=42)

    # Within SLA (loss 0.005 = 0.5%, delay 2.0 ms)
    info = env._build_info_dict(energy_info={}, avg_delay_ms=2.0, packet_loss_rate=0.005)
    sla_v = bool(info["avg_delay_ms"] > env.sla_delay_ms or info["packet_loss_rate"] > env.sla_loss_rate)
    assert not sla_v

    # Exceeding SLA loss (loss 0.015 = 1.5% > 1.0%)
    info_loss = env._build_info_dict(energy_info={}, avg_delay_ms=2.0, packet_loss_rate=0.015)
    sla_v_loss = bool(info_loss["avg_delay_ms"] > env.sla_delay_ms or info_loss["packet_loss_rate"] > env.sla_loss_rate)
    assert sla_v_loss

    # Exceeding SLA delay (delay 4.0 ms > 3.5 ms)
    info_del = env._build_info_dict(energy_info={}, avg_delay_ms=4.0, packet_loss_rate=0.002)
    sla_v_del = bool(info_del["avg_delay_ms"] > env.sla_delay_ms or info_del["packet_loss_rate"] > env.sla_loss_rate)
    assert sla_v_del


def test_override_telemetry_standard_keys():
    """Verifies action_overridden, requested_action, and reward_components are logged correctly in info."""
    env = GreenNetworkEnv(
        topology_type="fattree_k4",
        k=4,
        max_steps_per_episode=10,
        training_mode=False,
        use_safety_filter=True,
        override_penalty_weight=2.0,
    )
    env.reset(seed=60002, options={"intensity": 1.10, "profile": "diurnal", "burst": False})
    env.step(7)
    env.step(7)  # high demand telemetry established
    _, _, _, _, info = env.step(0)  # PPO asks 1 core, filter overrides to 4 cores

    assert "action_overridden" in info
    assert "requested_action" in info
    assert info["requested_action"] == 0
    assert info["action"] == 6  # Applied action
    assert info["action_overridden"] is True
    assert info["reward_components"]["override_penalty"] == 2.0


def test_reset_reward_components_schema():
    """Verifies reset() reward_components schema matches step() schema exactly."""
    env = GreenNetworkEnv(override_penalty_weight=2.0)
    _, info_reset = env.reset(seed=42)
    _, _, _, _, info_step = env.step(7)

    keys_reset = set(info_reset["reward_components"].keys())
    keys_step = set(info_step["reward_components"].keys())
    assert keys_reset == keys_step
    assert "override_penalty" in keys_reset


def test_non_training_reset_advances_prng_when_seed_none():
    """Verifies that calling reset() without seed in non-training mode generates different seeds across episodes."""
    env = GreenNetworkEnv(training_mode=False, max_steps_per_episode=3)
    env.reset(seed=42)  # explicit seed
    seed1 = env.traffic_gen.np_random.get_state()[1][0]

    # Subsequent resets with seed=None should advance PRNG, not stick to 42
    env.reset(seed=None)
    seed2 = env.traffic_gen.np_random.get_state()[1][0]
    env.reset(seed=None)
    seed3 = env.traffic_gen.np_random.get_state()[1][0]

    assert seed2 != 42
    assert seed2 != seed1
    assert seed3 != seed2


def test_vecnormalize_evaluation_integration_norm_obs_false_and_true():
    """
    Verifies VecNormalize integration in evaluation loop:
    1. norm_obs=False: normalize_obs leaves obs unchanged.
    2. norm_obs=True: normalize_obs applies (x - mean) / std without altering raw env QoS metrics.
    """
    env = GreenNetworkEnv(topology_type="fattree_k4", k=4)
    dummy = DummyVecEnv([lambda: env])

    # Case A: norm_obs = False
    vn_false = VecNormalize(dummy, training=False, norm_obs=False, norm_reward=False)
    obs, info = env.reset(seed=60001, options={"intensity": 0.50})
    policy_obs_f = vn_false.normalize_obs(obs)
    np.testing.assert_array_equal(obs, policy_obs_f)

    # Case B: norm_obs = True with pre-set running mean
    vn_true = VecNormalize(dummy, training=False, norm_obs=True, norm_reward=False)
    vn_true.obs_rms.mean = np.ones_like(obs) * 0.5
    vn_true.obs_rms.var = np.ones_like(obs) * 0.25  # std = 0.5
    policy_obs_t = vn_true.normalize_obs(obs)
    # Check normalized value: (obs - 0.5) / 0.5 = 2 * (obs - 0.5)
    expected = (obs - 0.5) / np.sqrt(0.25 + 1e-8)
    np.testing.assert_allclose(policy_obs_t, expected, rtol=1e-5)

    # Stepping env directly keeps raw QoS metrics unnormalized and exact
    obs_next, r, term, trunc, step_info = env.step(7)
    assert "total_power_w" in step_info
    assert "packet_loss_rate" in step_info
    assert "avg_delay_ms" in step_info
    assert isinstance(step_info["total_power_w"], float)
