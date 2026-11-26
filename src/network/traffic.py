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
        self._active_persistent_flows: List[FlowDemand] = []
        self._last_multiplier: float = 1.0

    def set_workload_profile(self, profile: str):
        """Sets active traffic profile: diurnal, datacenter_pareto, flash_crowd."""
        if profile in ["diurnal", "datacenter_pareto", "flash_crowd"]:
            self.workload_profile = profile

    def set_burst_mode(self, enabled: bool):
        if self.burst_mode != bool(enabled):
            # Clear persistent flows on burst state change so burst hits/stops immediately
            self._active_persistent_flows = []
        self.burst_mode = bool(enabled)

    def set_intensity(self, intensity: float):
        """Sets traffic intensity multiplier (e.g. 0.35 for light, 0.85 for normal, 2.2 for peak)."""
        new_val = max(0.1, min(5.0, float(intensity)))
        if abs(new_val - self.traffic_intensity) > 0.12:
            self._active_persistent_flows = []
        self.traffic_intensity = new_val

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
        host_count = len(self.hosts)

        # For k=6 DCN (>16 hosts), burst only triggers on explicit burst_mode or flash_crowd
        # so night/normal traffic evolves smoothly without unprompted 3.5x random noise spikes.
        if host_count > 16:
            stochastic_burst = (self.workload_profile == "flash_crowd")
        else:
            stochastic_burst = (self.np_random.rand() < 0.05) or (self.workload_profile == "flash_crowd")
        is_burst = self.burst_mode or stochastic_burst

        if host_count > 16:
            # Calibrated Fat-Tree k=6 Data Center Demand Curve (0 - 3800 Mbps physical capacity)
            # Gentle ±10% natural wave around the selected intensity so manual presets (35%, 85%, 145%)
            # stay true to their regime instead of being distorted 4x by the 120-step clock.
            natural_wave = 1.0 + 0.10 * float(np.sin(self.current_step * 0.45))
            if self.burst_mode:
                # Dynamic multi-step burst wave: surges across 7-core, 8-core, and 9-core regimes
                # (~1500 Mbps -> ~2300 Mbps -> ~3400 Mbps) so the AI dynamically scales 7 <-> 8 <-> 9 cores!
                burst_osc = 0.5 * (1.0 + float(np.sin(self.current_step * 0.85)))  # 0.0 to 1.0
                base_burst_mbps = 750.0 + float(self.traffic_intensity) * 950.0
                target_total_mbps = min(3550.0, base_burst_mbps + burst_osc * 1450.0)
            else:
                # Smooth mapping from traffic_intensity (0.20 .. 2.50) to DCN demand (45 .. 3400 Mbps)
                inten = float(self.traffic_intensity)
                if inten <= 0.26:
                    base_mbps = 48.0 * (inten / 0.22)          # ~48 Mbps  -> 1 Core
                elif inten <= 0.42:
                    base_mbps = 115.0 * (inten / 0.35)         # ~115 Mbps -> 2 Cores (Gece %35)
                elif inten <= 0.68:
                    base_mbps = 235.0 * (inten / 0.55)         # ~235 Mbps -> 3 Cores
                elif inten <= 0.98:
                    base_mbps = 490.0 * (inten / 0.85)         # ~490 Mbps -> 4 Cores (Normal %85)
                elif inten <= 1.28:
                    base_mbps = 880.0 * (inten / 1.15)         # ~880 Mbps -> 6 Cores
                elif inten <= 1.65:
                    base_mbps = 1480.0 * (inten / 1.45)        # ~1480 Mbps -> 7 Cores (Yoğun %145)
                elif inten <= 2.05:
                    base_mbps = 2250.0 * (inten / 1.85)        # ~2250 Mbps -> 8 Cores
                else:
                    base_mbps = min(3550.0, 3200.0 * (inten / 2.40))  # ~3200 Mbps -> 9 Cores
                target_total_mbps = base_mbps * natural_wave

            base_flows = max(12, int(round(host_count * 0.40)))
            eff_inten = 1.85 if self.burst_mode else float(self.traffic_intensity)
            target_flow_count = int(np.clip(round(base_flows * (eff_inten ** 0.70)), 10, 42))
            multiplier = target_total_mbps / max(1.0, target_flow_count * self.base_demand_mbps)
        else:
            multiplier = diurnal_factor * self.traffic_intensity
            if is_burst:
                multiplier *= self.burst_multiplier
            base_flows = max(10, int(round(host_count * 0.40)))
            target_flow_count = max(6, int(round(base_flows * (self.traffic_intensity ** 0.85))))
            if is_burst:
                target_flow_count = int(round(target_flow_count * 1.35))

        demands: List[FlowDemand] = []
        selected_pairs: Set[Tuple[str, str]] = set()
        src_counts: Dict[str, int] = {}
        dst_counts: Dict[str, int] = {}
        max_flows_per_host = max(2, int(np.ceil(target_flow_count / max(1, host_count / 2))))

        # Carry over multi-step persistent flows for realistic temporal autocorrelation (k=6 DCN)
        if host_count > 16 and self._active_persistent_flows:
            scale_ratio = multiplier / max(0.05, self._last_multiplier)
            max_carry = int(round(target_flow_count * 0.55))
            for pf in self._active_persistent_flows:
                if len(demands) >= max_carry:
                    break
                if pf.duration_steps > 1 and (pf.src, pf.dst) not in selected_pairs:
                    new_mbps = float(np.clip(pf.demand_mbps * scale_ratio, 5.0, 950.0))
                    carried = FlowDemand(
                        flow_id=pf.flow_id,
                        src=pf.src,
                        dst=pf.dst,
                        demand_mbps=round(new_mbps, 1),
                        duration_steps=pf.duration_steps - 1,
                        is_burst=is_burst,
                    )
                    demands.append(carried)
                    selected_pairs.add((pf.src, pf.dst))
                    src_counts[pf.src] = src_counts.get(pf.src, 0) + 1
                    dst_counts[pf.dst] = dst_counts.get(pf.dst, 0) + 1

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

            final_mbps = float(np.clip(raw_mbps * multiplier, 5.0, 950.0))

            flow = FlowDemand(
                flow_id=f"flow_{self.current_step}_{len(demands)}",
                src=src,
                dst=dst,
                demand_mbps=round(final_mbps, 1),
                duration_steps=int(self.np_random.randint(1, 4)),
                is_burst=is_burst
            )
            demands.append(flow)

        if host_count > 16:
            self._active_persistent_flows = list(demands)
            self._last_multiplier = float(multiplier)
        return demands

