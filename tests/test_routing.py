import pytest

from conftest import make_flow


def test_every_action_keeps_all_hosts_reachable_k4(env_k4):
    for a in range(env_k4.action_space.n):
        paths = env_k4._paths_for(a, frozenset())
        assert all(len(p) > 0 for p in paths.values()), f"action {a} loses reachability"


def test_paths_use_only_active_cores(env_k4):
    for a in range(env_k4.action_space.n):
        active = set(env_k4.core_actions[a])
        for plist in env_k4._paths_for(a, frozenset()).values():
            for p in plist:
                assert {n for n in p if n.startswith("c_")} <= active


def test_inter_pod_path_is_six_hops_and_ecmp_multiplicity(env_k4):
    plist = env_k4._paths_for(7, frozenset())[("h_0_0_0", "h_3_1_1")]
    assert all(len(p) == 7 for p in plist)  # 6 links
    assert len(plist) == 4                    # 2 aggs x 2 cores
    one_core = env_k4._paths_for(0, frozenset())[("h_0_0_0", "h_3_1_1")]
    assert len(one_core) == 1


def test_loads_conserve_demand_over_path(env_k4):
    res = env_k4._route_demands([make_flow(mbps=100.0)], 7, frozenset())
    loaded = [l for l, v in res["loads"].items() if v > 0]
    assert len(loaded) == 6
    assert all(res["loads"][l] == pytest.approx(100.0) for l in loaded)


def test_ecmp_hash_is_deterministic(env_k4):
    f = [make_flow(fid=f"flow_{i}", mbps=10.0) for i in range(8)]
    a = env_k4._route_demands(f, 7, frozenset())["loads"]
    b = env_k4._route_demands(f, 7, frozenset())["loads"]
    assert a == b


def test_link_failure_reroutes_via_second_agg(env_k4):
    failed = frozenset({env_k4._canon(("e_0_0", "a_0_0"))})
    res = env_k4._route_demands([make_flow(mbps=100.0)], 7, failed)
    assert res["lost_mbps"] == pytest.approx(0.0)
    assert res["loads"][env_k4._canon(("e_0_0", "a_0_0"))] == 0.0


def test_failure_with_one_core_reroutes_over_longer_path(env_k4):
    # action 0 keeps only c_0_0 (attached to agg 0); with e_0_0-a_0_0 down the graph still has an
    # 8-hop detour (e_0_0 -> a_0_1 -> e_0_1 -> a_0_0 -> c_0_0 ...), so traffic is rerouted, not lost.
    ok = env_k4._route_demands([make_flow(mbps=100.0)], 0, frozenset())
    failed = frozenset({env_k4._canon(("e_0_0", "a_0_0"))})
    res = env_k4._route_demands([make_flow(mbps=100.0)], 0, failed)
    assert res["loss_rate"] == 0.0
    assert res["avg_delay_ms"] > ok["avg_delay_ms"]


def test_both_uplinks_failed_makes_flow_unroutable(env_k4):
    failed = frozenset({env_k4._canon(("e_0_0", "a_0_0")), env_k4._canon(("e_0_0", "a_0_1"))})
    for action in (0, 7):
        res = env_k4._route_demands([make_flow(mbps=100.0)], action, failed)
        assert res["loss_rate"] == pytest.approx(1.0)


def test_k6_and_wan_routable(env_k6, env_wan):
    assert all(len(p) > 0 for p in env_k6._paths_for(env_k6.action_space.n - 1, frozenset()).values())
    assert all(len(p) > 0 for p in env_wan._paths_for(env_wan.action_space.n - 1, frozenset()).values())
