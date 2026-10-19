import numpy as np
import pytest

from src.env.green_network_env import GreenNetworkEnv, TRAIN_SEED_LIMIT
from src.agents.baselines import OSPFBaselineAgent
from src.agents.evaluate import evaluate_agent
from src.agents.safety import SafeAgent


def rollout(env, actions, seed=5, options=None):
    env.reset(seed=seed, options=options)
    return [env.step(a) for a in actions]


def test_spaces_and_obs_shape_unchanged(env_k4):
    obs, info = env_k4.reset(seed=0)
    assert env_k4.action_space.n == 8
    assert obs.shape == (len(env_k4.core_switches) + len(env_k4.links) + 3,) == (4 + 48 + 3,)
    o, r, term, trunc, info = env_k4.step(7)
    assert env_k4.observation_space.contains(o) and np.isfinite(r) and not term
    for key in ("avg_delay_ms", "packet_loss_percent", "total_power_w", "energy_savings_percent",
                "link_loads", "link_utilizations", "switch_states", "active_cores", "congested_links_count",
                "max_utilization", "total_demand_mbps", "p95_delay_ms", "p99_delay_ms", "sla_violation"):
        assert key in info


def test_truncation_at_max_steps():
    env = GreenNetworkEnv(max_steps_per_episode=5)
    env.reset(seed=0)
    for i in range(5):
        *_, trunc, _ = env.step(7)
    assert trunc


def test_same_seed_is_reproducible():
    e1, e2 = GreenNetworkEnv(max_steps_per_episode=10), GreenNetworkEnv(max_steps_per_episode=10)
    a = rollout(e1, [7, 3, 0, 5], seed=11)
    b = rollout(e2, [7, 3, 0, 5], seed=11)
    for (o1, r1, *_, i1), (o2, r2, *_, i2) in zip(a, b):
        assert np.array_equal(o1, o2) and r1 == r2 and i1["avg_delay_ms"] == i2["avg_delay_ms"]


def test_different_seeds_differ():
    e = GreenNetworkEnv(max_steps_per_episode=10)
    a = rollout(e, [7], seed=1)[0][0]
    b = rollout(e, [7], seed=2)[0][0]
    assert not np.array_equal(a, b)


def test_sleeping_cores_zero_their_links(env_k4):
    env_k4.reset(seed=3)
    _, _, _, _, info = env_k4.step(0)
    sleeping = [c for c in env_k4.core_switches if c not in info["active_cores"]]
    assert len(sleeping) == 3
    for l, load in env_k4.link_loads.items():
        if any(c in l for c in sleeping):
            assert load == 0.0
    assert info["sleeping_switches"] == 3


def test_training_mode_uses_train_seed_range_and_varies_traffic():
    env = GreenNetworkEnv(training_mode=True, max_steps_per_episode=3)
    seeds, intens = set(), set()
    for ep in range(6):
        env.reset(seed=ep if ep == 0 else None)
        seeds.add(env.traffic_gen.np_random.get_state()[1][0])
        intens.add(round(env.traffic_gen.traffic_intensity, 3))
    assert len(intens) > 1
    assert all(s < TRAIN_SEED_LIMIT for s in seeds)  # MT19937 state[0] == seed for RandomState(seed)
    assert len(seeds) > 1  # no longer the single hard-coded seed


def test_reset_options_override_scenario():
    env = GreenNetworkEnv(max_steps_per_episode=3)
    env.reset(seed=1, options={"intensity": 2.0, "profile": "flash_crowd", "burst": True})
    assert env.traffic_gen.traffic_intensity == 2.0
    assert env.traffic_gen.workload_profile == "flash_crowd" and env.traffic_gen.burst_mode


def test_scheduled_failure_and_recovery():
    env = GreenNetworkEnv(max_steps_per_episode=6)
    ev = [{"link": ("e_0_0", "a_0_0"), "start": 2, "end": 4}]
    env.reset(seed=1, options={"failures": ev})
    seen = []
    for _ in range(5):
        *_, info = env.step(7)
        seen.append(len(info["failed_links"]))
    assert seen == [0, 1, 1, 0, 0]
    env.reset(seed=1, options={"failures": []})
    assert env.step(7)[4]["failed_links"] == []


def test_failed_link_has_zero_load_and_port_off():
    env = GreenNetworkEnv(max_steps_per_episode=6)
    env.reset(seed=1)
    env.set_failed_links([("e_0_0", "a_0_0")])
    *_, info = env.step(7)
    u, v = env._canon(("e_0_0", "a_0_0"))
    assert info["link_loads"][f"{u}->{v}"] == 0.0
    assert info["ports_off"] >= 2


def test_reward_cfg_ablation_switches_term_off():
    base = GreenNetworkEnv(max_steps_per_episode=4)
    abl = GreenNetworkEnv(max_steps_per_episode=4, reward_cfg={"energy": 0.0})
    r0 = rollout(base, [0], seed=9)[0][1]
    r1 = rollout(abl, [0], seed=9)[0][1]
    assert r1 < r0  # removing the energy bonus lowers the reward for a sleeping-core action


def test_safety_mask_blocks_unsafe_actions_under_heavy_load():
    env = GreenNetworkEnv(max_steps_per_episode=10)
    env.reset(seed=4, options={"intensity": 2.5, "profile": "flash_crowd"})
    env.step(7)
    mask = env.action_mask()
    assert not mask[0] and mask.any()
    act, overridden = env.safe_action(0)
    assert overridden and mask[act] and len(env.core_actions[act]) > 1


def test_safety_mask_allows_green_action_under_light_load():
    env = GreenNetworkEnv(max_steps_per_episode=10)
    env.reset(seed=4, options={"intensity": 0.2, "profile": "diurnal"})
    env.step(7)
    assert env.action_mask()[0]
    assert env.safe_action(0) == (0, False)


def test_safety_mask_without_telemetry_is_conservative():
    env = GreenNetworkEnv(max_steps_per_episode=10)
    env.reset(seed=4)
    mask = env.action_mask()
    assert mask[env.full_action] and not mask[0]


def test_env_safety_filter_flag_overrides_in_step():
    env = GreenNetworkEnv(max_steps_per_episode=10, use_safety_filter=True)
    env.reset(seed=4, options={"intensity": 2.5, "profile": "flash_crowd"})
    env.step(7)
    *_, info = env.step(0)
    assert info["requested_action"] == 0 and info["action_overridden"] and info["action"] != 0


def test_safe_agent_reduces_sla_violations_vs_always_green():
    class AlwaysOne:
        name = "one-core"
        def predict(self, obs, deterministic=True):
            return 0

    opts = {"intensity": 2.0, "profile": "datacenter_pareto"}
    env = GreenNetworkEnv(max_steps_per_episode=40)
    raw = evaluate_agent(AlwaysOne(), env, num_steps=40, seed=TRAIN_SEED_LIMIT + 1, options=opts, include_timeseries=False)
    safe_agent = SafeAgent(AlwaysOne(), env)
    safe = evaluate_agent(safe_agent, env, num_steps=40, seed=TRAIN_SEED_LIMIT + 1, options=opts, include_timeseries=False)
    assert raw["sla_violation_rate"] > 0
    assert safe["sla_violation_rate"] < raw["sla_violation_rate"]
    assert safe_agent.overrides > 0


def test_evaluate_agent_reports_required_metrics():
    env = GreenNetworkEnv(max_steps_per_episode=20)
    s = evaluate_agent(OSPFBaselineAgent(env=env), env, num_steps=20, seed=TRAIN_SEED_LIMIT + 5, include_timeseries=False)
    for k in ("p95_delay_ms", "p99_delay_ms", "mean_packet_loss_percent", "total_energy_kwh",
              "switch_toggles", "action_changes", "sla_violation_rate"):
        assert k in s
    assert s["p99_delay_ms"] >= s["p95_delay_ms"] > 0
    assert s["total_energy_kwh"] > 0 and s["switch_toggles"] == 0 and s["action_changes"] == 0
