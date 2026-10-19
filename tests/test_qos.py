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
