import numpy as np
import pytest

from src.env.green_network_env import GreenNetworkEnv
from src.network.topology import FatTreeTopology
from src.sdn.policy_bridge import (
    PolicyBridge, TelemetryCollector, build_dpid_map, build_port_map, compute_forwarding,
    host_mac, simulate_forwarding,
)


class FixedAgent:
    def __init__(self, action):
        self.action = action

    def predict(self, obs, deterministic=True):
        return self.action


@pytest.fixture()
def setup():
    env = GreenNetworkEnv(max_steps_per_episode=5)
    col = TelemetryCollector(env.topology, link_capacity_mbps=100.0)
    return env, col


def feed(col, env, rate_mbps_by_link=None, t0=0.0, t1=10.0, default=0.0):
    """Feeds two counter snapshots so that each link carries the given (tx+rx) Mbps on its switch-side port."""
    dpid, ports = build_dpid_map(env.topology), build_port_map(env.topology)
    rate_mbps_by_link = rate_mbps_by_link or {}
    for t, frac in ((t0, 0.0), (t1, 1.0)):
        per_dp = {}
        for link in env.links:
            sw, nb = (link if link[0] in dpid else link[::-1])
            rate = rate_mbps_by_link.get(link, default)
            bytes_total = int(rate * 1e6 / 8.0 * (t1 - t0) * frac)
            per_dp.setdefault(dpid[sw], {})[ports[sw][nb]] = (bytes_total // 2, bytes_total - bytes_total // 2)
        for d, stats in per_dp.items():
            col.ingest_port_stats(d, stats, t)


def test_port_and_dpid_maps_are_deterministic_and_complete():
    topo = FatTreeTopology(k=4)
    d, p = build_dpid_map(topo), build_port_map(topo)
    assert sorted(d.values()) == list(range(1, 21))
    for sw in topo.switches:
        assert sorted(p[sw].values()) == list(range(1, len(p[sw]) + 1))
    assert build_port_map(topo) == p


@pytest.mark.parametrize("n_active", [1, 2, 4])
def test_forwarding_tables_deliver_all_pairs_without_sleeping_cores(n_active):
    topo = FatTreeTopology(k=4)
    active = set({1: ["c_0_0"], 2: ["c_0_0", "c_1_0"], 4: topo.core_switches}[n_active])
    tables, ok = compute_forwarding(topo, active)
    assert ok
    for s in topo.hosts:
        for t in topo.hosts:
            if s != t:
                path = simulate_forwarding(topo, tables, s, t)
                assert path is not None and path[-1] == t
                assert {n for n in path if n.startswith("c_")} <= active


def test_forwarding_reports_unreachable_when_agg_layer_uncovered():
    topo = FatTreeTopology(k=4)
    _, ok = compute_forwarding(topo, {"c_0_0"}, {("e_0_0", "a_0_0"), ("e_0_0", "a_0_1")})
    assert not ok


def test_telemetry_rates_and_observation_layout(setup):
    env, col = setup
    link = env.links[10]
    feed(col, env, {link: 50.0})
    assert col.link_rates_mbps()[link] == pytest.approx(50.0, rel=1e-3)
    obs = col.build_observation(env, set(env.core_switches), diurnal_factor=1.0)
    assert obs.shape == env.observation_space.shape
    idx = len(env.core_switches) + env.links.index(link)
    assert obs[idx] == pytest.approx(0.5, rel=1e-3)
    assert list(obs[:4]) == [1, 1, 1, 1]


def test_stale_telemetry_triggers_full_capacity_failsafe(setup):
    env, col = setup
    br = PolicyBridge(env, FixedAgent(0), col)
    d = br.decide(now=100.0)  # no telemetry at all
    assert d.reason == "stale-telemetry-failsafe" and d.action == env.full_action


def test_idle_network_lets_policy_choose_one_core(setup):
    env, col = setup
    feed(col, env, default=0.5)
    br = PolicyBridge(env, FixedAgent(0), col, min_dwell_s=0)
    d = br.decide(now=10.0)
    assert d.action == 0 and d.reason == "policy"


def test_heavy_load_overrides_unsafe_green_action(setup):
    env, col = setup
    core_links = {l: 70.0 for l in env.links if l[0] in env.core_switches or l[1] in env.core_switches}
    feed(col, env, core_links)
    br = PolicyBridge(env, FixedAgent(0), col, min_dwell_s=0)
    d = br.decide(now=10.0)
    assert d.overridden and d.reason == "safe-filter-override"
    assert len(env.core_actions[d.action]) > 1


def test_dwell_time_holds_scale_down_but_never_scale_up(setup):
    env, col = setup
    feed(col, env, default=0.5)
    br = PolicyBridge(env, FixedAgent(0), col, min_dwell_s=60)
    br.last_change_t = 5.0
    assert br.decide(now=10.0).reason == "dwell-hold"  # would scale down from full capacity
    br.current_action = 0
    br.agent = FixedAgent(7)
    assert br.decide(now=11.0).action == 7  # scale-up is immediate


def test_plan_is_make_before_break_and_verified(setup):
    env, col = setup
    br = PolicyBridge(env, FixedAgent(0), col)
    # initial: full capacity tables
    full = br.make_plan(env.full_action)
    assert br.verify_plan(full) and not full.ports_down
    br.commit(full, now=0.0)
    down = br.make_plan(0)
    assert down.ports_down and not down.ports_up
    assert br.verify_plan(down)
    br.commit(down, now=1.0)
    # sleeping cores' ports are all down (both ends)
    dpid, ports = build_dpid_map(env.topology), build_port_map(env.topology)
    for c in set(env.core_switches) - {"c_0_0"}:
        for nb, p in ports[c].items():
            assert (dpid[c], p) in br.down_ports and (dpid[nb], ports[nb][c]) in br.down_ports
    up = br.make_plan(env.full_action)
    assert up.ports_up and not up.ports_down and br.verify_plan(up)
    br.commit(up, now=2.0)
    assert not br.down_ports and br.current_action == env.full_action


def test_plan_that_would_blackhole_is_rejected(setup):
    env, col = setup
    br = PolicyBridge(env, FixedAgent(0), col)
    full = br.make_plan(env.full_action)
    br.commit(full)
    plan = br.make_plan(0)
    plan.flow_add = []  # corrupt: no new rules, old rules still point into the cores being put to sleep
    plan.flow_del = []
    assert not br.verify_plan(plan)
