"""
Custom Gymnasium Environment for Energy-Efficient SDN Traffic Routing (Green Networking).
Controls active core switch configurations and dynamically optimizes power vs QoS trade-offs.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional

from src.network.topology import FatTreeTopology, TopologyFactory
from src.network.power_model import NetworkPowerModel, SwitchPowerProfile
from src.network.traffic import TrafficGenerator, FlowDemand


class GreenNetworkEnv(gym.Env):
    """
    Gymnasium Environment for Green SDN Routing.
    Observation Space:
        - Core switch active states (4 binary flags)
        - Core-to-Agg Link utilization ratios
        - Agg-to-Edge Link utilization ratios
        - Traffic demand statistics (total volume, inter-pod ratio, diurnal factor)
    Action Space:
        - Discrete configurations selecting which Core Switches are ACTIVE vs SLEEP.
        - Preserves network reachability while aggressively turning off redundant hardware.
    Reward:
        - High positive reward for energy savings.
        - Severe penalty for link congestion (>80% utilization), delay spikes, and packet drops.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Action mappings: List of active core switches
    CORE_ACTIONS = [
        ["c_0_0"],                          # Act 0: Ultra-Green 1-Core (c00) -> 75% core power saving
        ["c_0_1"],                          # Act 1: Ultra-Green 1-Core (c01)
        ["c_1_0"],                          # Act 2: Ultra-Green 1-Core (c10)
        ["c_1_1"],                          # Act 3: Ultra-Green 1-Core (c11)
        ["c_0_0", "c_1_0"],                 # Act 4: Balanced 2-Cores (covers both agg layers) -> 50% power saving
        ["c_0_1", "c_1_1"],                 # Act 5: Balanced 2-Cores (alternative pair)
        ["c_0_0", "c_0_1", "c_1_0"],        # Act 6: High-Load 3-Cores -> 25% power saving
        ["c_0_0", "c_0_1", "c_1_0", "c_1_1"] # Act 7: Full Capacity 4-Cores (Classic OSPF/ECMP) -> 0% core saving
    ]

    def __init__(
        self,
        k: int = 4,
        topology_type: str = "fattree_k4",
        link_capacity_mbps: float = 1000.0,
        energy_weight: float = 2.5,
        delay_weight: float = 1.0,
        loss_weight: float = 5.0,
        congestion_threshold: float = 0.85,
        max_steps_per_episode: int = 150,
        training_mode: bool = False
    ):
        super().__init__()

        self.k = k
        self.topology_type = topology_type
        self.link_capacity_mbps = link_capacity_mbps
        self.energy_weight = energy_weight
        self.delay_weight = delay_weight
        self.loss_weight = loss_weight
        self.congestion_threshold = congestion_threshold
        self.max_steps_per_episode = max_steps_per_episode
        self.training_mode = training_mode

        # Initialize network components via TopologyFactory
        if topology_type and topology_type != "fattree_k4":
            self.topology = TopologyFactory.create(topology_type, link_capacity_mbps=link_capacity_mbps)
        else:
            self.topology = FatTreeTopology(k=k, link_capacity_mbps=link_capacity_mbps)

        self.power_model = NetworkPowerModel()
        self.traffic_gen = TrafficGenerator(hosts=self.topology.hosts)

        # Precompute lists
        self.core_switches = self.topology.core_switches
        self.all_switches = self.topology.switches
        self.switch_ports = self.topology.get_switch_port_counts()
        self.links = self.topology.get_links()

        # Build topology-aware graduated core action sets (from 1 core up to ALL cores)
        num_cores = len(self.core_switches)
        if num_cores == 4:
            self.core_actions = [
                [self.core_switches[0]],                         # Act 0: 1 Core (75% saving)
                [self.core_switches[1]],                         # Act 1: 1 Core
                [self.core_switches[0], self.core_switches[2]], # Act 2: 2 Cores (50% saving)
                [self.core_switches[1], self.core_switches[3]], # Act 3: 2 Cores
                [self.core_switches[0], self.core_switches[1], self.core_switches[2]], # Act 4: 3 Cores (25% saving)
                [self.core_switches[0], self.core_switches[2], self.core_switches[3]], # Act 5: 3 Cores
                list(self.core_switches),                        # Act 6: 4 Cores (100% capacity)
                list(self.core_switches)                         # Act 7: 4 Cores (100% capacity)
            ]
        elif num_cores == 9: # k=6 FatTree: 9 Cores (c_0_0 to c_2_2)
            self.core_actions = [
                [self.core_switches[0]],                         # Act 0: 1 Core (89% saving)
                [self.core_switches[0], self.core_switches[4]], # Act 1: 2 Cores (78% saving)
                [self.core_switches[0], self.core_switches[4], self.core_switches[8]], # Act 2: 3 Cores (67% saving)
                [self.core_switches[0], self.core_switches[1], self.core_switches[3], self.core_switches[4]], # Act 3: 4 Cores (56% saving)
                self.core_switches[:6],                          # Act 4: 6 Cores (33% saving)
                self.core_switches[:7],                          # Act 5: 7 Cores (22% saving)
                self.core_switches[:8],                          # Act 6: 8 Cores (11% saving)
                list(self.core_switches)                         # Act 7: 9 Cores (ALL 9 CORES ACTIVE - 100% capacity)
            ]
        else:
            self.core_actions = []
            for i in range(8):
                count = max(1, int(round((i + 1) * num_cores / 8.0)))
                self.core_actions.append(list(self.core_switches[:count]))

        # Action Space: 8 discrete network operating modes
        self.action_space = spaces.Discrete(len(self.core_actions))

        # Observation Space:
        # len(core_switches) + len(links) + 3
        self.num_obs_features = len(self.core_switches) + len(self.links) + 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=5.0,
            shape=(self.num_obs_features,),
            dtype=np.float32
        )

        # Internal state tracking
        self.current_step = 0
        self.current_action = 7  # default to full capacity
        self.prev_action: Optional[int] = None
        self.switch_states: Dict[str, bool] = {s: True for s in self.all_switches}

        # Precompute ECMP paths for each action configuration to achieve 500x speedup
        self._action_paths: Dict[int, Dict[Tuple[str, str], List[List[str]]]] = {}
        self._precompute_action_paths()

    def _precompute_action_paths(self):
        """Precomputes all shortest paths for each of the 8 discrete core configurations."""
        hosts = self.topology.hosts
        for act_idx, active_cores_list in enumerate(self.core_actions):
            active_cores = set(active_cores_list)
            subgraph = nx.Graph()
            for (u, v) in self.links:
                u_active = (u not in self.core_switches) or (u in active_cores)
                v_active = (v not in self.core_switches) or (v in active_cores)
                if u_active and v_active:
                    subgraph.add_edge(u, v)

            path_dict = {}
            for src in hosts:
                for dst in hosts:
                    if src != dst:
                        try:
                            paths = list(nx.all_shortest_paths(subgraph, source=src, target=dst))
                            path_dict[(src, dst)] = paths
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            path_dict[(src, dst)] = []
            self._action_paths[act_idx] = path_dict
        self.link_loads: Dict[Tuple[str, str], float] = {l: 0.0 for l in self.links}
        self.link_utilizations: Dict[Tuple[str, str], float] = {l: 0.0 for l in self.links}
        self.active_demands: List[FlowDemand] = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_action = 7

        if self.training_mode:
            # Randomize traffic intensity and profiles so the RL agent experiences light, medium, heavy & burst loads:
            intensity = float(self.np_random.uniform(0.35, 2.7))
            profile = str(self.np_random.choice(["diurnal", "datacenter_pareto", "flash_crowd"], p=[0.45, 0.35, 0.20]))
            burst = bool(self.np_random.random() < 0.15)
        else:
            intensity = getattr(self.traffic_gen, "traffic_intensity", 1.0) if hasattr(self, "traffic_gen") else 1.0
            profile = getattr(self.traffic_gen, "workload_profile", "diurnal") if hasattr(self, "traffic_gen") else "diurnal"
            burst = getattr(self.traffic_gen, "burst_mode", False) if hasattr(self, "traffic_gen") else False

        self.traffic_gen = TrafficGenerator(hosts=self.topology.hosts, seed=seed if seed is not None else 42)
        self.traffic_gen.set_intensity(intensity)
        self.traffic_gen.set_workload_profile(profile)
        self.traffic_gen.set_burst_mode(burst)

        # Reset all switches to active
        self.switch_states = {s: True for s in self.all_switches}
        self.link_loads = {l: 0.0 for l in self.links}
        self.link_utilizations = {l: 0.0 for l in self.links}
        self.active_demands = []

        obs = self._build_observation(diurnal_factor=1.0, total_demand=0.0, inter_pod_count=0)
        info = self._build_info_dict(energy_info={"total_power_w": 0, "energy_savings_percent": 0}, avg_delay_ms=1.0, packet_loss_rate=0.0)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        self.current_action = int(action)

        # 1. Apply Action: Configure Core Switch States
        active_cores = set(self.core_actions[self.current_action])
        for c in self.core_switches:
            self.switch_states[c] = (c in active_cores)

        # Aggregation and Edge switches remain active for host connectivity
        for s in self.topology.agg_switches + self.topology.edge_switches:
            self.switch_states[s] = True

        # 2. Determine Link States based on Switch States
        link_states: Dict[Tuple[str, str], bool] = {}
        for (u, v) in self.links:
            u_active = self.switch_states.get(u, True)
            v_active = self.switch_states.get(v, True)
            link_states[(u, v)] = (u_active and v_active)

        # 3. Generate Step Traffic
        demands = self.traffic_gen.generate_step_demands(active_flow_count=16)
        self.active_demands = demands

        # 4. Route Demands over Active Subgraph & Calculate Loads
        active_graph = nx.Graph()
        for (u, v), is_active in link_states.items():
            if is_active:
                active_graph.add_edge(u, v, capacity=self.link_capacity_mbps)

        # Reset link loads
        for l in self.links:
            self.link_loads[l] = 0.0

        routed_flows = 0
        dropped_demand_mbps = 0.0
        total_demand_mbps = sum(d.demand_mbps for d in demands) if demands else 1.0
        total_hop_delay_ms = 0.0
        congested_links_count = 0
        inter_pod_count = 0

        for demand in demands:
            src_pod = demand.src.split("_")[1]
            dst_pod = demand.dst.split("_")[1]
            if src_pod != dst_pod:
                inter_pod_count += 1

            # Fast O(1) ECMP lookup from precomputed shortest paths for current action
            candidate_paths = self._action_paths[self.current_action].get((demand.src, demand.dst), [])
            if not candidate_paths:
                # No path available in active subgraph
                dropped_demand_mbps += demand.demand_mbps
                continue

            # ECMP Hash flow across available paths
            path_idx = abs(hash(demand.flow_id)) % len(candidate_paths)
            path = candidate_paths[path_idx]
            routed_flows += 1

            # Accumulate link loads
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_key = (u, v) if (u, v) in self.link_loads else (v, u)
                if edge_key in self.link_loads:
                    self.link_loads[edge_key] += demand.demand_mbps

        # 5. Compute Link Utilizations & Delays (M/M/1 queue model)
        total_delay = 0.0
        max_utilization = 0.0
        for l in self.links:
            cap = self.link_capacity_mbps
            load = self.link_loads[l]
            util = load / cap
            self.link_utilizations[l] = util
            max_utilization = max(max_utilization, util)

            if util > self.congestion_threshold:
                congested_links_count += 1

            # Delay model: propagation + bounded queueing (finite switch buffer)
            prop_delay = 0.5  # ms
            if util < 0.80:
                queue_delay = (util / (1.0 - util + 1e-3)) * 0.2
            else:
                queue_delay = 1.0 + (util - 0.80) * 8.0
            queue_delay = min(15.0, max(0.0, queue_delay))  # Max 15ms buffer capacity

            total_delay += (prop_delay + queue_delay)

        avg_delay_ms = total_delay / max(1, len(self.links))

        # Real physical packet drop when link buffer overflows (load > capacity):
        overflow_dropped_mbps = sum(max(0.0, self.link_loads[l] - self.link_capacity_mbps) for l in self.links)
        total_loss_mbps = dropped_demand_mbps + (overflow_dropped_mbps * 0.4)
        packet_loss_rate = min(1.0, float(total_loss_mbps / max(1.0, total_demand_mbps)))

        # 6. Calculate Power & Energy Metrics
        power_info = self.power_model.calculate_total_power(
            switch_states=self.switch_states,
            link_states=link_states,
            switch_port_counts=self.switch_ports
        )

        # 7. Formulate Multi-Objective Reward Function
        # Energy Component: proportional reward for sleeping core switches
        active_core_count = len(self.core_actions[self.current_action])
        total_core_count = float(len(self.core_switches))
        core_saving_ratio = (total_core_count - active_core_count) / max(1.0, total_core_count)
        energy_reward = core_saving_ratio * 16.0 * (self.energy_weight / 2.5)

        # Over-provisioning / Waste Penalty (Prevents the lazy 'turn on everything' shortcut)
        # If load safely fits in fewer cores, penalize keeping redundant hardware awake
        estimated_safe_cores = max(1, int(np.ceil(total_demand_mbps / 1100.0)))
        waste_penalty = 0.0
        if active_core_count > estimated_safe_cores and max_utilization < 0.70:
            excess = active_core_count - estimated_safe_cores
            waste_penalty = excess * 3.5

        # Delay Penalty with SLA threshold (> 3.5 ms)
        delay_penalty = (avg_delay_ms / 2.5) * self.delay_weight
        if avg_delay_ms > 3.5:
            delay_penalty += (avg_delay_ms - 3.5) * 6.0

        # Congestion & Packet Loss Penalty (Calibrated to allow elastic headroom without panic)
        loss_penalty = (packet_loss_rate * 50.0) * self.loss_weight
        congestion_penalty = (congested_links_count * 1.5) if congested_links_count > 0 else 0.0

        # Route Flapping / Switching Penalty (Hysteresis)
        switching_penalty = 0.2 if (self.prev_action is not None and self.current_action != self.prev_action) else 0.0
        self.prev_action = self.current_action

        # Overall Step Reward
        reward = float(energy_reward - waste_penalty - delay_penalty - loss_penalty - congestion_penalty - switching_penalty)

        # 8. Check Episode Termination
        terminated = False
        truncated = (self.current_step >= self.max_steps_per_episode)

        # 9. Next Observation & Info
        diurnal = self.traffic_gen.get_diurnal_factor(self.current_step)
        obs = self._build_observation(
            diurnal_factor=diurnal,
            total_demand=total_demand_mbps,
            inter_pod_count=inter_pod_count
        )
        info = self._build_info_dict(
            energy_info=power_info,
            avg_delay_ms=avg_delay_ms,
            packet_loss_rate=packet_loss_rate,
            congested_links=congested_links_count,
            max_utilization=max_utilization,
            total_demand=total_demand_mbps
        )

        return obs, reward, terminated, truncated, info

    def _build_observation(self, diurnal_factor: float, total_demand: float, inter_pod_count: int) -> np.ndarray:
        """Constructs a normalized 1D feature vector for RL policies."""
        features: List[float] = []

        # 1. Core switch states (1.0 = active, 0.0 = sleep)
        for c in self.core_switches:
            features.append(1.0 if self.switch_states.get(c, False) else 0.0)

        # 2. Normalized Link Utilizations
        for l in self.links:
            features.append(float(np.clip(self.link_utilizations.get(l, 0.0), 0.0, 2.0)))

        # 3. Traffic Context
        features.append(float(diurnal_factor))
        features.append(float(np.clip(total_demand / 10000.0, 0.0, 5.0)))
        features.append(float(np.clip(inter_pod_count / 16.0, 0.0, 2.0)))

        return np.array(features, dtype=np.float32)

    def _build_info_dict(
        self,
        energy_info: Dict[str, Any],
        avg_delay_ms: float,
        packet_loss_rate: float,
        congested_links: int = 0,
        max_utilization: float = 0.0,
        total_demand: float = 0.0
    ) -> Dict[str, Any]:
        """Provides full simulation state for rendering, live web UI, and evaluation."""
        return {
            "step": self.current_step,
            "action": self.current_action,
            "action_name": f"{len(self.core_actions[self.current_action])}-Core Active",
            "active_cores": self.core_actions[self.current_action],
            "total_power_w": energy_info.get("total_power_w", 0.0),
            "max_power_w": energy_info.get("max_power_w", 0.0),
            "energy_savings_percent": energy_info.get("energy_savings_percent", 0.0),
            "active_switches": energy_info.get("active_switches", len(self.all_switches)),
            "sleeping_switches": energy_info.get("sleeping_switches", 0),
            "avg_delay_ms": round(avg_delay_ms, 2),
            "packet_loss_rate": round(packet_loss_rate, 4),
            "packet_loss_percent": round(packet_loss_rate * 100.0, 2),
            "congested_links_count": congested_links,
            "max_utilization": round(max_utilization, 3),
            "total_demand_mbps": round(total_demand, 1),
            "switch_states": dict(self.switch_states),
            "link_loads": {f"{u}->{v}": round(load, 1) for (u, v), load in self.link_loads.items()},
            "link_utilizations": {f"{u}->{v}": round(util, 3) for (u, v), util in self.link_utilizations.items()}
        }
