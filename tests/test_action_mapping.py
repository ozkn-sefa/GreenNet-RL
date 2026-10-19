import numpy as np
import pytest

from src.agents.baselines import OSPFBaselineAgent, StaticHeuristicAgent, ThresholdLoadAgent


def obs_with(diurnal=1.0, demand=0.0, n=30):
    o = np.zeros(n, dtype=np.float32)
    o[-3], o[-2] = diurnal, demand
    return o


def cores(env, a):
    return len(env.core_actions[a])


# ---- k=4: env table is the reference -------------------------------------------------
def test_env_action_table_k4(env_k4):
    counts = [cores(env_k4, a) for a in range(8)]
    assert counts == [1, 1, 2, 2, 3, 3, 4, 4]


def test_ospf_selects_all_cores(env_k4):
    for ag in (OSPFBaselineAgent(), OSPFBaselineAgent(env=env_k4)):
        assert cores(env_k4, ag.predict(obs_with())) == 4


@pytest.mark.parametrize("diurnal,expected_cores", [(0.3, 1), (0.59, 1), (0.6, 2), (0.99, 2), (1.0, 4), (1.3, 4)])
def test_static_core_counts_k4(env_k4, diurnal, expected_cores):
    for ag in (StaticHeuristicAgent(), StaticHeuristicAgent(env=env_k4)):
        assert cores(env_k4, ag.predict(obs_with(diurnal=diurnal))) == expected_cores


@pytest.mark.parametrize("demand,expected_cores", [(0.0, 1), (0.14, 1), (0.15, 2), (0.34, 2), (0.35, 3), (0.54, 3), (0.55, 4), (3.0, 4)])
def test_threshold_core_counts_k4(env_k4, demand, expected_cores):
    for ag in (ThresholdLoadAgent(), ThresholdLoadAgent(env=env_k4)):
        assert cores(env_k4, ag.predict(obs_with(demand=demand))) == expected_cores


def test_baseline_two_core_action_covers_both_agg_layers(env_k4):
    """2-core baseline action must keep full inter-pod reachability (cores on both agg layers)."""
    a = StaticHeuristicAgent(env=env_k4).predict(obs_with(diurnal=0.8))
    active = set(env_k4.core_actions[a])
    assert {c.split("_")[1] for c in active} == {"0", "1"}


# ---- other topologies -----------------------------------------------------------------
@pytest.mark.parametrize("fixture", ["env_k6", "env_wan"])
def test_baselines_valid_on_other_topologies(request, fixture):
    env = request.getfixturevalue(fixture)
    n = env.action_space.n
    ospf = OSPFBaselineAgent(env=env)
    assert cores(env, ospf.predict(obs_with())) == len(env.core_switches)
    for ag in (StaticHeuristicAgent(env=env), ThresholdLoadAgent(env=env)):
        for d in (0.0, 0.2, 0.4, 0.9):
            a = ag.predict(obs_with(diurnal=d * 2, demand=d))
            assert 0 <= a < n
    # monotone: more load never selects fewer cores
    th = ThresholdLoadAgent(env=env)
    counts = [cores(env, th.predict(obs_with(demand=d))) for d in (0.0, 0.2, 0.4, 0.9)]
    assert counts == sorted(counts)
