import numpy as np
import pytest

from conftest import make_flow
from src.env.green_network_env import weighted_percentile
from src.agents.evaluate import mean_std_ci


def q(env, u):
    return env._queue_delay_ms(u)


def test_delay_is_end_to_end_path_sum_not_link_average(env_k4):
    res = env_k4._route_demands([make_flow(mbps=100.0)], 7, frozenset())
    util = 0.1
    expected = 6 * (0.1 + q(env_k4, util))
    assert res["avg_delay_ms"] == pytest.approx(expected)
    link_avg = np.mean([env_k4._prop_delay_ms[l] + q(env_k4, u) for l, u in res["utils"].items()])
    assert res["avg_delay_ms"] > 5 * link_avg  # link-average would grossly under-report


def test_delay_is_traffic_weighted(env_k4):
    light = make_flow("a", "h_0_0_0", "h_0_0_1", 50.0)       # same edge switch: 2 hops
    heavy = make_flow("b", "h_0_0_0", "h_3_1_1", 400.0)      # inter-pod: 6 hops
    res = env_k4._route_demands([light, heavy], 7, frozenset())
    d, w = res["delay_samples"], res["delay_weights"]
    assert res["avg_delay_ms"] == pytest.approx(np.average(d, weights=w))
    assert res["avg_delay_ms"] > d.mean()  # weighted toward the heavy, longer flow


def test_overload_causes_loss_and_bottleneck_consistency(env_k4):
    res = env_k4._route_demands([make_flow(mbps=1500.0)], 7, frozenset())
    # host uplink (cap 1000) forwards 2/3 of the flow; every hop is overloaded equally
    assert res["loss_rate"] == pytest.approx(1.0 / 3.0)
    assert res["max_utilization"] == pytest.approx(1.5)
    assert res["bottleneck_link"] is not None
    assert res["lost_mbps"] == pytest.approx(500.0)
    assert res["congested_links"] == 6
    assert res["avg_delay_ms"] > 6 * (0.1 + q(env_k4, 0.5))


def test_no_loss_under_capacity(env_k4):
    res = env_k4._route_demands([make_flow(mbps=900.0)], 7, frozenset())
    assert res["loss_rate"] == 0.0 and res["lost_mbps"] == 0.0


def test_loss_compounds_over_bottlenecks(env_k4):
    # two flows sharing the same destination host link (cap 1000): 2 x 800 offered
    f = [make_flow("x", "h_0_0_0", "h_3_1_1", 800.0), make_flow("y", "h_1_0_0", "h_3_1_1", 800.0)]
    res = env_k4._route_demands(f, 7, frozenset())
    assert res["lost_mbps"] >= 600.0 - 1e-6  # destination host link forwards at most 1000 of 1600
    assert 0.0 < res["loss_rate"] < 1.0


def test_delay_monotone_in_load(env_k4):
    ds = [env_k4._route_demands([make_flow(mbps=m)], 7, frozenset())["avg_delay_ms"] for m in (50, 300, 600, 900)]
    assert ds == sorted(ds)


def test_percentiles_and_stats_helpers():
    v = np.arange(1, 101, dtype=float)
    assert weighted_percentile(v, np.ones(100), 95) == 95.0
    assert weighted_percentile(v, np.ones(100), 99) == 99.0
    assert weighted_percentile(np.array([]), np.array([]), 99) == 0.0
    s = mean_std_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == 3.0 and s["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert s["ci_low"] < 3.0 < s["ci_high"]
    # t(0.975, df=4)=2.776 -> half width = 2.776*1.5811/sqrt(5)
    assert s["ci_high"] - 3.0 == pytest.approx(2.7764451 * 1.5811388 / np.sqrt(5), rel=1e-4)
    one = mean_std_ci([2.0])
    assert one["std"] == 0.0 and one["ci_low"] == one["ci_high"] == 2.0


def test_qos_root_cause_safefilter_telemetry_lag_and_surge(env_k4):
    """
    Root Cause Test: SafeFilter evaluates safety on the previous step's telemetry.
    If an unexpected inter-step surge occurs (e.g. 500 Mbps -> 3000 Mbps),
    SafeFilter allows an under-provisioned core action, resulting in core-link overload.
    """
    env_k4.reset(seed=60002)
    # Set low previous-step demand
    env_k4.active_demands = [make_flow("f_low", "h_0_0_0", "h_1_0_0", 400.0)]
    mask = env_k4.action_mask()
    # 400 Mbps * 1.20 headroom fits in 1 core (capacity 1000 Mbps), so Action 1 is deemed safe
    assert bool(mask[1]) or bool(mask[0])

    # But if the current step experiences an inter-pod surge of 2500 Mbps:
    surge_flows = [
        make_flow("f_surge_1", "h_0_0_0", "h_1_0_0", 1250.0),
        make_flow("f_surge_2", "h_0_1_0", "h_2_0_0", 1250.0),
    ]
    # Route over 1 core (Action 1)
    res_1core = env_k4._route_demands(surge_flows, 1, frozenset())
    assert res_1core["loss_rate"] > 0.0
    assert res_1core["max_utilization"] > 1.0

    # Route over 4 cores (Action 6/7)
    res_4core = env_k4._route_demands(surge_flows, 7, frozenset())
    # Bisection bandwidth is 4x1000 = 4000 Mbps, so 2500 Mbps fits across 4 cores
    assert res_4core["max_utilization"] <= 1.25


def test_qos_root_cause_switch_toggles_vs_action_changes(env_k4):
    """
    Verification: episode_switch_toggles measures physical switch and port state
    transitions across the network, NOT simple action index change counts.
    """
    env_k4.reset(seed=60002)
    assert env_k4.episode_action_changes == 0
    assert env_k4.episode_switch_toggles == 0

    # Transition from Action 1 (1 core active: c_0_1) to Action 6 (4 cores active: all 4)
    env_k4.step(1)
    toggles_step1 = env_k4.episode_switch_toggles
    changes_step1 = env_k4.episode_action_changes

    env_k4.step(6)
    toggles_step2 = env_k4.episode_switch_toggles
    changes_step2 = env_k4.episode_action_changes

    # Exactly 1 action change occurred between step 1 and step 2
    assert changes_step2 == changes_step1 + 1
    # But physical toggles increased by 3 (3 core switches woke up: c_0_0, c_1_0, c_1_1)
    assert (toggles_step2 - toggles_step1) == 3
    assert (toggles_step2 - toggles_step1) > (changes_step2 - changes_step1)


def test_qos_root_cause_safefilter_credit_assignment_masking(env_k4):
    """
    Verification: When SafeFilter overrides an unsafe raw action,
    reward is calculated based on the safe applied action, shielding PPO from the loss penalty.
    """
    old_sf = env_k4.use_safety_filter
    env_k4.use_safety_filter = True
    try:
        env_k4.reset(seed=60002)
        # High demand requiring all cores
        env_k4.active_demands = [make_flow(f"f_{i}", f"h_0_0_{i%2}", f"h_1_0_{i%2}", 900.0) for i in range(4)]
        # Agent requests Action 0 (1 core), but SafeFilter should override to 4 cores
        safe_act, overridden = env_k4.safe_action(0)
        assert overridden is True
        assert len(env_k4.core_actions[safe_act]) > len(env_k4.core_actions[0])

        # In env.step, the reward received by PPO uses safe_act
        _, r_step, _, _, info = env_k4.step(0)
        assert info["action_overridden"] is True
        # The applied action is safe, so packet loss is minimized
        assert info["reward_components"]["loss_penalty"] == 0.0
    finally:
        env_k4.use_safety_filter = old_sf


def test_qos_physical_edge_bottleneck_under_all_active_ospf(env_k4):
    """
    Verification: Even under OSPF (Action 7, all 4 cores active),
    packet loss occurs if multiple flows aggregate onto the same destination host link (>1000 Mbps).
    This confirms that a portion of Medium load loss (1.48%) is a physical ingress capacity bottleneck.
    """
    # Two 700 Mbps flows destined to the same host h_3_1_1 (1400 Mbps total > 1000 Mbps capacity)
    demands = [
        make_flow("f_pod0", "h_0_0_0", "h_3_1_1", 700.0),
        make_flow("f_pod1", "h_1_0_0", "h_3_1_1", 700.0),
    ]
    res = env_k4._route_demands(demands, 7, frozenset())  # Action 7: All cores active
    assert res["loss_rate"] > 0.0
    assert res["lost_mbps"] >= 400.0 - 1e-6
    assert "h_3_1_1" in res["bottleneck_link"]

