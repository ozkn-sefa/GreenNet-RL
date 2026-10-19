"""
Dynamic Traffic Matrix & Flow Generator for Data Center Networks.
Simulates realistic workload distributions (e.g., MapReduce, Web Search, Data Mining, diurnal cycles).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import numpy as np
import random


@dataclass
class FlowDemand:
    flow_id: str
    src: str
    dst: str
    demand_mbps: float
    duration_steps: int
    is_burst: bool = False


class TrafficGenerator:
    """
    Generates dynamic traffic patterns across hosts in the data center.
    Supports baseline loads, inter-pod vs intra-pod probability, diurnal waves, and injected bursts.
    """

    def __init__(
        self,
        hosts: List[str],
        base_demand_mbps: float = 80.0,
        inter_pod_ratio: float = 0.65,
        seed: int = 42
    ):
        self.hosts = hosts
        self.base_demand_mbps = base_demand_mbps
        self.inter_pod_ratio = inter_pod_ratio
        self.np_random = np.random.RandomState(seed)
        self.current_step = 0
        self.burst_mode = False
        self.burst_multiplier = 3.5
        self.traffic_intensity = 1.0  # Dynamic traffic load factor (0.2x to 3.0x)
        self.workload_profile = "diurnal"  # "diurnal", "datacenter_pareto", "flash_crowd"

    def set_workload_profile(self, profile: str):
        """Sets active traffic profile: diurnal, datacenter_pareto, flash_crowd."""
        if profile in ["diurnal", "datacenter_pareto", "flash_crowd"]:
            self.workload_profile = profile

    def set_burst_mode(self, enabled: bool):
        self.burst_mode = enabled

    def set_intensity(self, intensity: float):
        """Sets traffic intensity multiplier (e.g. 0.35 for light, 0.85 for normal, 2.2 for peak)."""
        self.traffic_intensity = max(0.1, min(5.0, float(intensity)))

    def get_diurnal_factor(self, step: int) -> float:
        """
        Simulates diurnal (day/night) traffic waves over a 24-hour cycle.
        Returns a factor between 0.3 (night off-peak) and 1.2 (day peak).
        """
        if self.workload_profile == "flash_crowd":
            return 2.5  # Constant sustained high-intensity stress
        elif self.workload_profile == "datacenter_pareto":
            # Bimodal day/night with sharp spikes
            period = 120
            cycle = (step % period) / period
            factor = 0.60 + 0.50 * np.sin(2 * np.pi * cycle - (np.pi / 2))
            return float(np.clip(factor, 0.25, 1.4))
        else:
            period = 120  # 120 simulation steps = 1 full day cycle
            cycle = (step % period) / period
            # Shifted cosine wave: peaks at midday, troughs at midnight
            factor = 0.75 + 0.45 * np.sin(2 * np.pi * cycle - (np.pi / 2))
            return float(np.clip(factor, 0.3, 1.3))

    def generate_step_demands(self, active_flow_count: int = 12) -> List[FlowDemand]:
        """
        Samples a set of communication demands between hosts for the current step.
        Scales proportionally with diurnal factor and interactive traffic intensity.
        """
        self.current_step += 1
        diurnal_factor = self.get_diurnal_factor(self.current_step)
        
        # Intermittent random surge (rare 5% fluctuation) or user-triggered burst mode
        stochastic_burst = (self.np_random.rand() < 0.05) or (self.workload_profile == "flash_crowd")
        is_burst = self.burst_mode or stochastic_burst

        multiplier = diurnal_factor * self.traffic_intensity
        if is_burst:
            multiplier *= self.burst_multiplier

        # Dynamic flow count based on interactive traffic intensity
        target_flow_count = max(6, int(round(active_flow_count * np.sqrt(self.traffic_intensity))))
        if is_burst:
            target_flow_count = int(round(target_flow_count * 1.5))

        demands: List[FlowDemand] = []
        host_count = len(self.hosts)

        # Allow realistic concurrency (up to 3 concurrent flows per host)
        selected_pairs: Set[Tuple[str, str]] = set()
        src_counts: Dict[str, int] = {}
        dst_counts: Dict[str, int] = {}
        max_flows_per_host = max(2, int(np.ceil(target_flow_count / max(1, host_count / 2))))

        attempts = 0
        while len(demands) < target_flow_count and attempts < 250:
            attempts += 1
            src, dst = self.np_random.choice(self.hosts, size=2, replace=False)
            if (src, dst) in selected_pairs:
                continue
            if src_counts.get(src, 0) >= max_flows_per_host or dst_counts.get(dst, 0) >= max_flows_per_host:
                continue

            src_counts[src] = src_counts.get(src, 0) + 1
            dst_counts[dst] = dst_counts.get(dst, 0) + 1
            selected_pairs.add((src, dst))

            # Robust pod extraction (works for FatTree h_0_0_0 and WAN h_sea)
            src_pod = src.split("_")[1] if len(src.split("_")) > 1 else src
            dst_pod = dst.split("_")[1] if len(dst.split("_")) > 1 else dst
            is_inter_pod = (src_pod != dst_pod)

            # Modulate probability based on inter_pod_ratio
            if is_inter_pod and self.np_random.rand() > self.inter_pod_ratio:
                continue

            # Demand size: log-normal or Pareto distribution (mice vs elephant flows)
            if self.workload_profile == "datacenter_pareto":
                # 80/20 rule: 80% mice flows (10-80 Mbps), 20% elephant flows (400-900 Mbps)
                if self.np_random.rand() < 0.80:
                    raw_mbps = float(self.np_random.uniform(15.0, 75.0))
                else:
                    raw_mbps = float(self.np_random.uniform(400.0, 850.0))
            else:
                raw_mbps = float(self.np_random.lognormal(mean=np.log(self.base_demand_mbps), sigma=0.5))

            final_mbps = float(np.clip(raw_mbps * multiplier, 20.0, 950.0))

            flow = FlowDemand(
                flow_id=f"flow_{self.current_step}_{len(demands)}",
                src=src,
                dst=dst,
                demand_mbps=round(final_mbps, 1),
                duration_steps=int(self.np_random.randint(1, 4)),
                is_burst=is_burst
            )
            demands.append(flow)

        return demands

