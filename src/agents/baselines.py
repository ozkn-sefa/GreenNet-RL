"""
Baseline Algorithms for Energy-Aware Data Center Routing Benchmark.
1. OSPF / All-Active (Always keeps every core active - 0% core power savings).
2. Static Time-of-Day Green Heuristic (Rule-based scheduler based on diurnal cycles).
3. Dynamic Threshold-Based Heuristic (Expands cores based on instantaneous traffic load).

Action indices MUST match `GreenNetworkEnv.core_actions` (k=4 table):
    0,1 -> 1 core | 2,3 -> 2 cores | 4,5 -> 3 cores | 6,7 -> all 4 cores
When an `env` is given, the baselines resolve actions from `env.core_actions` by core fraction,
so they stay consistent on k=6 / NSFNET tables too.
"""

from typing import Dict, Any, List, Optional
import numpy as np

# Core fraction (of all cores) targeted by each logical level
_LEVEL_FRACTION = {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
# k=4 table (identical to GreenNetworkEnv for num_cores == 4)
_K4_ACTIONS = {1: 0, 2: 2, 3: 4, 4: 7}


def resolve_action(env: Optional[Any], level: int) -> int:
    """Maps logical level (1=lowest .. 4=all cores) to an action index valid for `env`."""
    if env is None or not hasattr(env, "core_actions"):
        return _K4_ACTIONS[level]
    if level >= 4:
        return int(len(env.core_actions) - 1)  # last action always = all cores
    num_cores = len(env.core_switches)
    target = max(1, int(round(_LEVEL_FRACTION[level] * num_cores)))
    best = min(range(len(env.core_actions)), key=lambda a: (abs(len(env.core_actions[a]) - target), a))
    return int(best)


class OSPFBaselineAgent:
    """
    Standard legacy routing agent (Dijkstra / OSPF / ECMP oblivious).
    Always keeps all core switches powered on (Action 7).
    """

    def __init__(self, env: Optional[Any] = None):
        self.name = "OSPF / ECMP (All Active)"
        self.action = resolve_action(env, 4)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        return self.action


class StaticHeuristicAgent:
    """
    Heuristic rule-based agent mimicking standard cron/diurnal scheduling:
    - Off-peak (diurnal factor < 0.6): 1 Core
    - Normal (0.6 <= diurnal factor < 1.0): 2 Cores
    - Peak (diurnal factor >= 1.0): all Cores
    """

    def __init__(self, env: Optional[Any] = None):
        self.name = "Static Diurnal Heuristic"
        self.a_low = resolve_action(env, 1)
        self.a_mid = resolve_action(env, 2)
        self.a_high = resolve_action(env, 4)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        # Feature index: diurnal factor is at [-3]
        diurnal_factor = round(float(observation[-3]), 5)

        if diurnal_factor < 0.6:
            return self.a_low
        elif diurnal_factor < 1.0:
            return self.a_mid
        else:
            return self.a_high


class ThresholdLoadAgent:
    """
    Reactive threshold agent: adjusts core count based on observed total traffic demand.
    """

    def __init__(self, low_thresh: float = 0.3, high_thresh: float = 0.7, env: Optional[Any] = None):
        self.name = "Load-Threshold Reactive"
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.actions = [resolve_action(env, lv) for lv in (1, 2, 3, 4)]

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        # observation[-2] is normalized total demand (demand / 10000.0)
        norm_demand = round(float(observation[-2]), 5)

        if norm_demand < 0.15:
            return self.actions[0]  # 1 Core
        elif norm_demand < 0.35:
            return self.actions[1]  # 2 Cores
        elif norm_demand < 0.55:
            return self.actions[2]  # 3 Cores
        else:
            return self.actions[3]  # all Cores
