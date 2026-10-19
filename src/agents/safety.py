"""
Safe action filter wrapper: makes any agent (PPO, heuristics) SLA-safe at inference time
without touching the underlying policy or its training.
"""

from typing import Any
import numpy as np


class SafeAgent:
    """
    Wraps an agent and passes its proposed action through `env.safe_action`
    (predictive action masking on the last observed demand matrix, see GreenNetworkEnv.action_mask).
    PPO weights/training are unchanged, so existing models stay compatible.
    """

    def __init__(self, agent: Any, env: Any):
        self.agent = agent
        self.env = env
        self.name = f"{getattr(agent, 'name', type(agent).__name__)} + SafeFilter"
        self.overrides = 0
        self.calls = 0

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        proposed = int(self.agent.predict(observation, deterministic=deterministic))
        action, overridden = self.env.safe_action(proposed)
        self.calls += 1
        self.overrides += int(overridden)
        return action


class TelemetrySafetyFilter:
    """
    Observation-only safe action filter (usable without simulator state, e.g. in the Ryu controller).

    Uses the observation layout [core states | link utils | diurnal, demand, inter-pod]:
      * total core-link load is conserved regardless of which cores are awake, so the predicted mean
        utilization of the links of candidate action `a` is
            headroom * balance_factor * sum(core_link_load) / (#links incident to active cores of a)
      * an action is allowed iff that prediction <= safe_util_limit and it keeps all hosts reachable.
    Approximation: ignores per-link ECMP imbalance beyond `balance_factor`; the env-based mask
    (GreenNetworkEnv.action_mask) is more precise because it re-routes the observed demand matrix.
    """

    def __init__(self, env: Any, safe_util_limit: float = 0.85, headroom: float = 1.25, balance_factor: float = 1.3):
        self.core_switches = list(env.core_switches)
        self.core_actions = [list(a) for a in env.core_actions]
        self.links = list(env.links)
        self.safe_util_limit = safe_util_limit
        self.headroom = headroom
        self.balance_factor = balance_factor
        core_set = set(self.core_switches)
        self._core_link_idx = [i for i, (u, v) in enumerate(self.links) if u in core_set or v in core_set]
        self._links_of_core = {c: sum(1 for (u, v) in self.links if c in (u, v)) for c in self.core_switches}
        self.full_action = int(env.full_action)
        import networkx as nx  # local import keeps module light
        self._reachable = []
        for a in range(len(self.core_actions)):
            paths = env._paths_for(a, frozenset())
            self._reachable.append(all(len(p) > 0 for p in paths.values()))

    def predicted_core_util(self, observation: np.ndarray, action: int) -> float:
        n_cores = len(self.core_switches)
        utils = np.asarray(observation[n_cores:-3], dtype=float)
        total_core_load = float(np.sum(utils[self._core_link_idx]))  # in units of link capacity
        n_links = sum(self._links_of_core[c] for c in self.core_actions[action])
        return self.headroom * self.balance_factor * total_core_load / max(1, n_links)

    def mask(self, observation: np.ndarray) -> np.ndarray:
        m = np.array([
            self._reachable[a] and self.predicted_core_util(observation, a) <= self.safe_util_limit
            for a in range(len(self.core_actions))
        ], dtype=bool)
        if not m.any():
            full_n = len(self.core_actions[self.full_action])
            m = np.array([len(self.core_actions[a]) == full_n for a in range(len(self.core_actions))], dtype=bool)
        return m

    def filter(self, observation: np.ndarray, action: int):
        """Returns (safe_action, overridden): lowest-power allowed action with >= requested cores."""
        m = self.mask(observation)
        if m[int(action)]:
            return int(action), False
        req = len(self.core_actions[int(action)])
        allowed = [a for a in range(len(m)) if m[a]]
        bigger = [a for a in allowed if len(self.core_actions[a]) >= req] or allowed
        return int(min(bigger, key=lambda a: (len(self.core_actions[a]), a))), True
