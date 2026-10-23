"""
Regression tests for Abilene WAN energy modeling, link capacity resolution,
and graph connectivity under core switch operating actions.
"""

import pytest
from src.env.green_network_env import GreenNetworkEnv
from src.network.topology import AbileneTopology, TopologyFactory


def test_abilene_default_link_capacity():
    """Verify that GreenNetworkEnv preserves Abilene's natural 2500 Mbps capacity if unspecified."""
    env = GreenNetworkEnv(topology_type="abilene")
    assert env.link_capacity_mbps == 2500.0, f"Expected 2500.0 Mbps for Abilene, got {env.link_capacity_mbps}"
    assert env.topology.link_capacity_mbps == 2500.0


def test_fattree_default_link_capacity():
    """Verify that GreenNetworkEnv preserves FatTree's natural 1000 Mbps capacity."""
    env = GreenNetworkEnv(topology_type="fattree_k4")
    assert env.link_capacity_mbps == 1000.0, f"Expected 1000.0 Mbps for FatTree, got {env.link_capacity_mbps}"


def test_power_metrics_in_step_info():
    """Verify that total_power_w is present and positive in step_info."""
    env = GreenNetworkEnv(topology_type="abilene")
    obs, info = env.reset()
    obs, r, term, trunc, step_info = env.step(7)

    assert "total_power_w" in step_info, "total_power_w missing from step_info"
    assert isinstance(step_info["total_power_w"], (int, float))
    assert step_info["total_power_w"] > 0, "total_power_w must be positive"
    assert "energy_savings_percent" in step_info


def test_abilene_action_reachability():
    """
    Verify graph connectivity under Abilene core switch actions:
    Action 6 and 7 must provide 100% host reachability (0 unroutable pairs).
    Action 0 to 5 partition the WAN graph causing unreachable host pairs.
    """
    env = GreenNetworkEnv(topology_type="abilene")
    total_pairs = len(env.topology.hosts) * (len(env.topology.hosts) - 1)
    assert total_pairs == 132

    # Actions 6 and 7 have all 4 cores active -> 0 unroutable
    for act in [6, 7]:
        unroutable = sum(1 for (s, d), paths in env._action_paths[act].items() if len(paths) == 0)
        assert unroutable == 0, f"Action {act} must have 0 unroutable host pairs, got {unroutable}"

    # Actions 0 to 5 must have unroutable pairs due to WAN router deep-sleep
    for act in range(6):
        unroutable = sum(1 for (s, d), paths in env._action_paths[act].items() if len(paths) == 0)
        assert unroutable > 0, f"Action {act} was expected to partition Abilene graph, got {unroutable} unroutable pairs"
