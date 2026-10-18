"""
Baseline Algorithms for Energy-Aware Data Center Routing Benchmark.
1. OSPF / All-Active (Always selects 4 active cores - 0% power savings).
2. Static Time-of-Day Green Heuristic (Rule-based scheduler based on diurnal cycles).
3. Dynamic Threshold-Based Heuristic (Expands cores based on instantaneous traffic load).
"""

from typing import Dict, Any
import numpy as np


class OSPFBaselineAgent:
    """
    Standard legacy routing agent (Dijkstra / OSPF / ECMP oblivious).
    Always keeps all core switches powered on (Action 7).
    """

    def __init__(self):
        self.name = "OSPF / ECMP (All Active)"

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        # Action 7 corresponds to all 4 cores active
        return 7


class StaticHeuristicAgent:
    """
    Heuristic rule-based agent mimicking standard cron/diurnal scheduling:
    - Off-peak (diurnal factor < 0.6): 1 Core
    - Normal (0.6 <= diurnal factor < 1.0): 2 Cores
    - Peak (diurnal factor >= 1.0): 4 Cores
    """

    def __init__(self):
        self.name = "Static Diurnal Heuristic"

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        # Feature index: diurnal factor is at [-3]
        diurnal_factor = observation[-3]
        
        if diurnal_factor < 0.6:
            return 0  # 1 Core (c00)
        elif diurnal_factor < 1.0:
            return 4  # 2 Cores (c00, c10)
        else:
            return 7  # 4 Cores (all)


class ThresholdLoadAgent:
    """
    Reactive threshold agent: adjusts core count based on observed total traffic demand.
    """

    def __init__(self, low_thresh: float = 0.3, high_thresh: float = 0.7):
        self.name = "Load-Threshold Reactive"
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        # observation[-2] is normalized total demand (demand / 10000.0)
        norm_demand = observation[-2]

        if norm_demand < 0.15:
            return 0  # 1 Core
        elif norm_demand < 0.35:
            return 4  # 2 Cores
        elif norm_demand < 0.55:
            return 6  # 3 Cores
        else:
            return 7  # 4 Cores
