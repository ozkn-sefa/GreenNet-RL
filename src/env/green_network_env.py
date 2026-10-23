"""
Custom Gymnasium Environment for Energy-Efficient SDN Traffic Routing (Green Networking).
Controls active core switch configurations and dynamically optimizes power vs QoS trade-offs.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Iterable, FrozenSet

from src.network.topology import FatTreeTopology, TopologyFactory
from src.network.power_model import NetworkPowerModel, SwitchPowerProfile, joules_to_kwh
from src.network.traffic import TrafficGenerator, FlowDemand

# Training episodes draw traffic seeds from [0, TRAIN_SEED_LIMIT); evaluation scripts must use
# seeds >= TRAIN_SEED_LIMIT so train/test traffic realizations never overlap.
TRAIN_SEED_LIMIT = 1_000_000

DEFAULT_REWARD_CFG: Dict[str, float] = {
    "energy": 1.0, "waste": 1.0, "delay": 1.0, "loss": 1.0,
    "congestion": 1.0, "switching": 1.0, "wakeup": 1.0,
}


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Traffic-weighted percentile (q in [0, 100]) using the lower-cumulative definition."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0 or weights.sum() <= 0:
        return 0.0
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) / w.sum()
    idx = int(np.searchsorted(cum, q / 100.0, side="left"))
    return float(v[min(idx, len(v) - 1)])


class GreenNetworkEnv(gym.Env):
    """
    Gymnasium Environment for Green SDN Routing.
    Observation Space:
        - Core switch active states (binary flags)
        - Link utilization ratios (all links)
        - Traffic demand statistics (diurnal factor, total volume, inter-pod ratio)
    Action Space:
        - Discrete configurations selecting which Core Switches are ACTIVE vs SLEEP.
    Reward:
        - Positive reward for sleeping core switches; penalties for waste, delay (SLA), loss,
          congestion, action switching and switch wake-ups.

    QoS model (consistent per step):
        - Flows are ECMP-hashed on shortest paths of the active (and non-failed) subgraph.
        - Link load -> utilization; per-hop delay = propagation + bounded M/M/1-style queueing.
        - End-to-end flow delay = sum of per-hop delays along its path.
          `avg_delay_ms` is the TRAFFIC-WEIGHTED mean of end-to-end flow delays.
        - Loss: an overloaded link forwards capacity/load of its offered load; a flow's delivered
          fraction is the minimum forwarding fraction along its path (its bottleneck). Flows without
          a path are fully lost. The bottleneck link (max utilization) therefore drives both delay and loss.
    Energy model: see src/network/power_model.py (simulated; step duration `step_duration_s`).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Legacy k=4 action table (kept for reference; the effective table is built in __init__ as self.core_actions)
    CORE_ACTIONS = [
        ["c_0_0"],
        ["c_0_1"],
        ["c_1_0"],
        ["c_1_1"],
        ["c_0_0", "c_1_0"],
        ["c_0_1", "c_1_1"],
        ["c_0_0", "c_0_1", "c_1_0"],
        ["c_0_0", "c_0_1", "c_1_0", "c_1_1"]
    ]

    def __init__(
        self,
        k: int = 4,
        topology_type: str = "fattree_k4",
        link_capacity_mbps: Optional[float] = None,
        energy_weight: float = 2.5,
        delay_weight: float = 1.0,
        loss_weight: float = 5.0,
        congestion_threshold: float = 0.85,
        max_steps_per_episode: int = 150,
        training_mode: bool = False,
        step_duration_s: float = 720.0,
        sla_delay_ms: Optional[float] = None,
        sla_loss_rate: float = 0.01,
        reward_cfg: Optional[Dict[str, float]] = None,
        use_safety_filter: bool = False,
        safe_util_limit: float = 0.85,
        safety_headroom: float = 1.25,
        reward_scale: float = 1.0,
        traffic_source: str = "synthetic",
        real_traffic_loader: Optional[Any] = None,
        real_traffic_split: str = "train",
        real_traffic_loop: bool = False,
        green_mode: str = "switch_sleep",
        min_dwell_steps: int = 3,
    ):
        super().__init__()

        self.reward_scale = float(reward_scale)  # uniform scaling of the step reward (value-function conditioning)
        self.k = k
        self.topology_type = topology_type
        self.green_mode = str(green_mode).lower()
        self.min_dwell_steps = int(min_dwell_steps)
        self.steps_in_action = 0
        self.link_capacity_mbps = float(link_capacity_mbps) if link_capacity_mbps is not None else None
        self.energy_weight = energy_weight
        self.delay_weight = delay_weight
        self.loss_weight = loss_weight
        self.congestion_threshold = congestion_threshold
        self.max_steps_per_episode = max_steps_per_episode
        self.training_mode = training_mode
        self.step_duration_s = float(step_duration_s)  # 120 steps = 24 h diurnal cycle -> 720 s/step
        self.sla_loss_rate = float(sla_loss_rate)
        self.reward_cfg = dict(DEFAULT_REWARD_CFG)
        if reward_cfg:
            self.reward_cfg.update(reward_cfg)
        self.use_safety_filter = use_safety_filter
        self.safe_util_limit = safe_util_limit
        self.safety_headroom = safety_headroom

        # Real traffic integration
        self.traffic_source = str(traffic_source).lower()
        self.real_traffic_loader = real_traffic_loader
        self.real_traffic_split = str(real_traffic_split).lower()
        self.real_traffic_loop = bool(real_traffic_loop)
        self.real_source: Optional[Any] = None
        if self.traffic_source == "real" or self.real_traffic_loader is not None:
            self._init_real_traffic_source()

        # Initialize network components via TopologyFactory
        if topology_type and topology_type != "fattree_k4":
            self.topology = TopologyFactory.create(topology_type, link_capacity_mbps=self.link_capacity_mbps)
        else:
            cap = self.link_capacity_mbps if self.link_capacity_mbps is not None else 1000.0
            self.topology = FatTreeTopology(k=k, link_capacity_mbps=cap)

        if self.link_capacity_mbps is None:
            self.link_capacity_mbps = float(getattr(self.topology, "link_capacity_mbps", 1000.0))

        self.power_model = NetworkPowerModel()
        self.traffic_gen = TrafficGenerator(hosts=self.topology.hosts)

        # Precompute lists
        self.core_switches = self.topology.core_switches
        self.all_switches = self.topology.switches
        self.switch_ports = self.topology.get_switch_port_counts()
        self.links = self.topology.get_links()
        self._edge_lookup: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for (u, v) in self.links:
            self._edge_lookup[(u, v)] = (u, v)
            self._edge_lookup[(v, u)] = (u, v)

        # Per-link propagation delay (ms). Fat-tree: 0.1 ms/hop (intra-DC); WAN: topology link delay.
        is_fat_tree = isinstance(self.topology, FatTreeTopology)
        self._prop_delay_ms: Dict[Tuple[str, str], float] = {
            l: (0.1 if is_fat_tree else float(self.topology.graph.edges[l].get("delay", 1.0)))
            for l in self.links
        }
        if sla_delay_ms is None:
            if is_fat_tree:
                sla_delay_ms = 3.5
            else:  # WAN: 1.5x mean shortest-path propagation + 2 ms queueing budget
                hs = self.topology.hosts
                lens = []
                for s in hs:
                    for d in hs:
                        if s != d:
                            lens.append(nx.shortest_path_length(self.topology.graph, s, d, weight="delay"))
                sla_delay_ms = 1.5 * float(np.mean(lens)) + 2.0
        self.sla_delay_ms = float(sla_delay_ms)

        # Build topology-aware graduated core action sets (or link-level sleep sets for WAN)
        num_cores = len(self.core_switches)
        self.is_link_level = (self.green_mode == "link_level")
        self.sleeping_links: List[List[Tuple[str, str]]] = [[] for _ in range(8)]

        if self.is_link_level and "abilene" in self.topology_type.lower():
            # Dedicated Abilene Link/Port Energy Optimization
            # All 12 router chassis stay ON for 100% host reachability.
            # Actions 0..7 select strategic redundant link sleep combinations & EEE/LPI.
            self.sleeping_links = [
                [self._canon(('SNVAng', 'STTLng')), self._canon(('NYCMng', 'WASHng')), self._canon(('CHINng', 'IPLSng'))], # Act 0: 3 links sleep + EEE
                [self._canon(('DNVRng', 'SNVAng')), self._canon(('ATLAng', 'CHINng'))],                                     # Act 1: 2 links sleep
                [self._canon(('SNVAng', 'STTLng')), self._canon(('NYCMng', 'WASHng'))],                                     # Act 2: 2 links sleep
                [self._canon(('NYCMng', 'WASHng'))],                                                                         # Act 3: 1 link sleep
                [self._canon(('DNVRng', 'STTLng'))],                                                                         # Act 4: 1 link sleep
                [self._canon(('SNVAng', 'STTLng'))],                                                                         # Act 5: 1 link sleep
                [],                                                                                                           # Act 6: 0 links sleep (pure adaptive EEE/LPI)
                [],                                                                                                           # Act 7: 0 links sleep (All active / OSPF)
            ]
            self.core_actions = [list(self.core_switches) for _ in range(8)]
        elif num_cores == 4:
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
        self.full_action = int(max(range(len(self.core_actions)), key=lambda a: (len(self.core_actions[a]), a)))

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
        self._prev_switch_states: Dict[str, bool] = dict(self.switch_states)
        self._prev_link_states: Dict[Tuple[str, str], bool] = {l: True for l in self.links}

        # Failure state: scheduled events + manually injected failures (links stored in canonical form)
        self.failure_events: List[Dict[str, Any]] = []
        self.manual_failed_links: set = set()
        self._failed_now: FrozenSet[Tuple[str, str]] = frozenset()

        # Episode accumulators / metrics
        self.episode_energy_j = 0.0
        self.episode_switch_toggles = 0
        self.episode_action_changes = 0
        self.last_flow_delays: np.ndarray = np.zeros(0)
        self.last_flow_weights: np.ndarray = np.zeros(0)

        # Precompute ECMP paths for each action configuration (no failures) to achieve 500x speedup
        self._action_paths: Dict[int, Dict[Tuple[str, str], List[List[str]]]] = {}
        self._path_cache: Dict[Tuple[int, FrozenSet[Tuple[str, str]]], Dict[Tuple[str, str], List[List[str]]]] = {}
        self._precompute_action_paths()

        self.link_loads: Dict[Tuple[str, str], float] = {l: 0.0 for l in self.links}
        self.link_utilizations: Dict[Tuple[str, str], float] = {l: 0.0 for l in self.links}
        self.active_demands: List[FlowDemand] = []

    # ------------------------------------------------------------------ paths
    def _compute_paths(self, act_idx: int, failed: FrozenSet[Tuple[str, str]]) -> Dict[Tuple[str, str], List[List[str]]]:
        hosts = self.topology.hosts
        active_cores = set(self.core_actions[act_idx])
        sleep_links = set(self.sleeping_links[act_idx]) if getattr(self, "is_link_level", False) else set()
        subgraph = nx.Graph()
        for (u, v) in self.links:
            if (u, v) in failed or (u, v) in sleep_links:
                continue
            u_active = (u not in self.core_switches) or (u in active_cores)
            v_active = (v not in self.core_switches) or (v in active_cores)
            if u_active and v_active:
                subgraph.add_edge(u, v)

        path_dict = {}
        for src in hosts:
            for dst in hosts:
                if src != dst:
                    try:
                        path_dict[(src, dst)] = list(nx.all_shortest_paths(subgraph, source=src, target=dst))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        path_dict[(src, dst)] = []
        return path_dict

    def _precompute_action_paths(self):
        """Precomputes all shortest paths for each of the 8 discrete core configurations (no failures)."""
        empty: FrozenSet[Tuple[str, str]] = frozenset()
        for act_idx in range(len(self.core_actions)):
            self._action_paths[act_idx] = self._compute_paths(act_idx, empty)
            self._path_cache[(act_idx, empty)] = self._action_paths[act_idx]

    def _paths_for(self, action: int, failed: FrozenSet[Tuple[str, str]]) -> Dict[Tuple[str, str], List[List[str]]]:
        key = (action, failed)
        if key not in self._path_cache:
            self._path_cache[key] = self._compute_paths(action, failed)
        return self._path_cache[key]

    # --------------------------------------------------------------- failures
    def _canon(self, link: Iterable[str]) -> Tuple[str, str]:
        u, v = link
        if (u, v) not in self._edge_lookup:
            raise ValueError(f"Unknown link: {link}")
        return self._edge_lookup[(u, v)]

    def set_failed_links(self, links: Iterable[Iterable[str]]):
        """Manually fail links (persist until cleared). Pass [] to recover."""
        self.manual_failed_links = {self._canon(l) for l in links}

    def set_failure_events(self, events: Optional[List[Dict[str, Any]]]):
        """events: [{"link": (u, v), "start": step, "end": step_or_None}]; active for start <= step < end."""
        parsed = []
        for e in events or []:
            parsed.append({
                "link": self._canon(e["link"]),
                "start": int(e.get("start", 0)),
                "end": None if e.get("end") is None else int(e["end"]),
            })
        self.failure_events = parsed

    def _failed_links_at(self, step: int) -> FrozenSet[Tuple[str, str]]:
        failed = set(self.manual_failed_links)
        for e in self.failure_events:
            if e["start"] <= step and (e["end"] is None or step < e["end"]):
                failed.add(e["link"])
        return frozenset(failed)

    # ---------------------------------------------------------------- routing
    @staticmethod
    def _queue_delay_ms(util: float) -> float:
        """Bounded queueing delay (finite switch buffer, max 15 ms)."""
        if util < 0.80:
            q = (util / (1.0 - util + 1e-3)) * 0.2
        else:
            q = 1.0 + (util - 0.80) * 8.0
        return min(15.0, max(0.0, q))

    def _route_demands(
        self,
        demands: List[FlowDemand],
        action: int,
        failed: FrozenSet[Tuple[str, str]],
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Pure routing/QoS evaluation (no env state mutation).
        Returns link loads/utils, traffic-weighted end-to-end delay, loss, per-flow samples.
        """
        paths = self._paths_for(action, failed)
        loads = {l: 0.0 for l in self.links}
        routed: List[Tuple[FlowDemand, List[Tuple[str, str]], float]] = []  # (demand, edge path, mbps)
        dropped_unroutable = 0.0
        total_demand = 0.0

        for d in demands:
            mbps = d.demand_mbps * scale
            total_demand += mbps
            candidates = paths.get((d.src, d.dst), [])
            if not candidates:
                dropped_unroutable += mbps
                continue
            # Deterministic ECMP hash (stable across processes, unlike hash(str))
            h = sum(ord(c) * (i + 1) for i, c in enumerate(d.flow_id))
            path = candidates[h % len(candidates)]
            edges = [self._edge_lookup[(path[i], path[i + 1])] for i in range(len(path) - 1)]
            for e in edges:
                loads[e] += mbps
            routed.append((d, edges, mbps))

        cap = self.link_capacity_mbps
        utils = {l: loads[l] / cap for l in self.links}
        fwd = {l: (min(1.0, cap / loads[l]) if loads[l] > 0 else 1.0) for l in self.links}
        hop_delay = {l: self._prop_delay_ms[l] + self._queue_delay_ms(utils[l]) for l in self.links}

        delays, weights = [], []
        lost = dropped_unroutable
        for d, edges, mbps in routed:
            frac = 1.0
            delay = 0.0
            for e in edges:
                frac = min(frac, fwd[e])  # the path bottleneck limits delivery
                delay += hop_delay[e]
            lost += mbps * (1.0 - frac)
            delays.append(delay)
            weights.append(mbps * frac)

        delays_a = np.array(delays, dtype=float)
        weights_a = np.array(weights, dtype=float)
        if weights_a.sum() > 0:
            avg_delay = float(np.average(delays_a, weights=weights_a))
        elif delays_a.size:
            avg_delay = float(delays_a.mean())
        else:
            avg_delay = 0.0

        max_util = max(utils.values()) if utils else 0.0
        bottleneck = max(utils, key=utils.get) if utils else None
        return {
            "loads": loads,
            "utils": utils,
            "avg_delay_ms": avg_delay,
            "p95_delay_ms": weighted_percentile(delays_a, weights_a, 95),
            "p99_delay_ms": weighted_percentile(delays_a, weights_a, 99),
            "delay_samples": delays_a,
            "delay_weights": weights_a,
            "loss_rate": float(min(1.0, lost / total_demand)) if total_demand > 0 else 0.0,
            "lost_mbps": float(lost),
            "delivered_mbps": float(max(0.0, total_demand - lost)),
            "total_demand_mbps": float(total_demand),
            "max_utilization": float(max_util),
            "bottleneck_link": bottleneck,
            "congested_links": int(sum(1 for u in utils.values() if u > self.congestion_threshold)),
        }

    # -------------------------------------------------------- safe-action API
    def predict_action_outcomes(self, headroom: Optional[float] = None) -> List[Optional[Dict[str, Any]]]:
        """Predicted QoS of every action if the last observed demand matrix (x headroom) repeated."""
        if not self.active_demands:
            return [None] * len(self.core_actions)
        h = self.safety_headroom if headroom is None else headroom
        failed = self._failed_links_at(self.current_step + 1)
        return [self._route_demands(self.active_demands, a, failed, scale=h) for a in range(len(self.core_actions))]

    def action_mask(self) -> np.ndarray:
        """
        Boolean mask of SLA-safe actions (True = allowed), predicted on the last telemetry
        (observed demands scaled by `safety_headroom`). Safe means: no unroutable/lost traffic,
        max link utilization <= safe_util_limit and delay within SLA.
        With no telemetry yet or no safe action, only the full-capacity action(s) are allowed.
        """
        n = len(self.core_actions)
        mask = np.zeros(n, dtype=bool)
        outcomes = self.predict_action_outcomes()
        for a, o in enumerate(outcomes):
            if o is None:
                continue
            if (o["lost_mbps"] <= 1e-9 and o["max_utilization"] <= self.safe_util_limit
                    and o["avg_delay_ms"] <= self.sla_delay_ms):
                mask[a] = True
        if not mask.any():
            if getattr(self, "is_link_level", False):
                min_sleep = min(len(self.sleeping_links[a]) for a in range(n))
                for a in range(n):
                    if len(self.sleeping_links[a]) == min_sleep:
                        mask[a] = True
            else:
                full_n = len(self.core_actions[self.full_action])
                for a in range(n):
                    if len(self.core_actions[a]) == full_n:
                        mask[a] = True
        return mask

    def safe_action(self, action: int) -> Tuple[int, bool]:
        """Returns (safe_action, overridden). Unsafe requests become the lowest-power allowed action
        with at least as many active cores/links as requested (else the lowest-power allowed action)."""
        action = int(action)
        mask = self.action_mask()
        if mask[action]:
            return action, False

        if getattr(self, "is_link_level", False):
            allowed = [a for a in range(len(mask)) if mask[a]]
            higher = [a for a in allowed if a >= action]
            pool = higher if higher else allowed
            best = min(pool) if pool else self.full_action
            return int(best), True

        requested_cores = len(self.core_actions[action])
        max_cores = len(self.core_actions[self.full_action])

        # If requested action is already at full capacity, never downgrade it
        if requested_cores >= max_cores:
            return self.full_action, False

        allowed = [a for a in range(len(mask)) if mask[a]]
        if not allowed:
            return self.full_action, True

        bigger = [a for a in allowed if len(self.core_actions[a]) >= requested_cores]
        if bigger:
            best = min(bigger, key=lambda a: (len(self.core_actions[a]), a))
        else:
            best = max(allowed, key=lambda a: (len(self.core_actions[a]), a))
        return int(best), True

    def _init_real_traffic_source(self, split: Optional[str] = None, start_step: int = 0):
        """Initializes or rebinds RealTrafficSource for Abilene data."""
        if self.real_traffic_loader is None:
            from src.data.abilene_loader import AbileneDataLoader
            self.real_traffic_loader = AbileneDataLoader()
        from src.data.abilene_loader import RealTrafficSource
        target_split = split or self.real_traffic_split
        self.real_source = RealTrafficSource(
            loader=self.real_traffic_loader,
            split=target_split,
            topology_type=self.topology_type,
            start_step=start_step,
            loop=self.real_traffic_loop,
        )

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        options (all optional): intensity (float), profile (str), burst (bool),
        traffic_source ('synthetic'|'real'), split ('train'|'val'|'test'), step (int),
        failures (list of {"link": (u, v), "start": int, "end": int|None}).
        """
        super().reset(seed=seed)
        options = options or {}
        self.current_step = 0
        self.current_action = 7  # full capacity
        self.prev_action = None
        self.steps_in_action = 0

        if "traffic_source" in options:
            self.traffic_source = str(options["traffic_source"]).lower()
        if "split" in options:
            self.real_traffic_split = str(options["split"]).lower()

        if self.traffic_source == "real":
            # Strictly use real traffic data
            start_step = int(options.get("step", 0))
            self._init_real_traffic_source(split=self.real_traffic_split, start_step=start_step)
            self.real_source.reset(start_step=start_step)
        else:
            if self.training_mode:
                # Randomize traffic intensity and profiles so the RL agent experiences light, medium, heavy & burst loads:
                intensity = float(self.np_random.uniform(0.35, 2.7))
                profile = str(self.np_random.choice(["diurnal", "datacenter_pareto", "flash_crowd"], p=[0.45, 0.35, 0.20]))
                burst = bool(self.np_random.random() < 0.15)
                # Fresh traffic realization every episode, always inside the TRAIN seed range
                traffic_seed = int(self.np_random.integers(0, TRAIN_SEED_LIMIT))
            else:
                intensity = getattr(self.traffic_gen, "traffic_intensity", 1.0) if hasattr(self, "traffic_gen") else 1.0
                profile = getattr(self.traffic_gen, "workload_profile", "diurnal") if hasattr(self, "traffic_gen") else "diurnal"
                burst = getattr(self.traffic_gen, "burst_mode", False) if hasattr(self, "traffic_gen") else False
                traffic_seed = seed if seed is not None else 42

            intensity = float(options.get("intensity", intensity))
            profile = str(options.get("profile", profile))
            burst = bool(options.get("burst", burst))

            self.traffic_gen = TrafficGenerator(hosts=self.topology.hosts, seed=traffic_seed)
            self.traffic_gen.set_intensity(intensity)
            self.traffic_gen.set_workload_profile(profile)
            self.traffic_gen.set_burst_mode(burst)

        if "failures" in options:
            self.set_failure_events(options["failures"])
        self._failed_now = self._failed_links_at(0)

        # Reset all switches to active
        self.switch_states = {s: True for s in self.all_switches}
        self._prev_switch_states = dict(self.switch_states)
        self._prev_link_states = {l: (l not in self._failed_now) for l in self.links}
        self.link_loads = {l: 0.0 for l in self.links}
        self.link_utilizations = {l: 0.0 for l in self.links}
        self.active_demands = []
        self.power_model.reset_energy_counter()
        self.episode_energy_j = 0.0
        self.episode_switch_toggles = 0
        self.episode_action_changes = 0

        obs = self._build_observation(diurnal_factor=1.0, total_demand=0.0, inter_pod_count=0)
        info = self._build_info_dict(energy_info={"total_power_w": 0, "energy_savings_percent": 0}, avg_delay_ms=1.0, packet_loss_rate=0.0)
        info["traffic_source"] = self.traffic_source
        if self.traffic_source == "real":
            info["real_split"] = self.real_traffic_split
        return obs, info

    # ------------------------------------------------------------------- step
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        requested_action = int(action)
        overridden = False
        if self.use_safety_filter:
            action, overridden = self.safe_action(requested_action)
        self.current_action = int(action)

        # 1. Apply Action: Configure Switch and Link States
        if getattr(self, "is_link_level", False):
            for s in self.all_switches:
                self.switch_states[s] = True
            sleep_now = set(self.sleeping_links[self.current_action])
        else:
            active_cores = set(self.core_actions[self.current_action])
            for c in self.core_switches:
                self.switch_states[c] = (c in active_cores)
            for s in self.topology.agg_switches + self.topology.edge_switches:
                self.switch_states[s] = True
            sleep_now = set()

        # 2. Determine Link States based on Switch States, sleep links, and failures
        failed = self._failed_links_at(self.current_step)
        self._failed_now = failed
        link_states: Dict[Tuple[str, str], bool] = {}
        for (u, v) in self.links:
            u_active = self.switch_states.get(u, True)
            v_active = self.switch_states.get(v, True)
            link_states[(u, v)] = bool(u_active and v_active and (u, v) not in failed and (u, v) not in sleep_now)

        # 3. Generate Step Traffic
        if self.traffic_source == "real":
            if self.real_source is None:
                self._init_real_traffic_source()
            demands = self.real_source.get_step_demands()
            real_info = self.real_source.get_active_info()
        else:
            demands = self.traffic_gen.generate_step_demands(active_flow_count=16)
            real_info = {"traffic_source": "synthetic"}
        self.active_demands = demands

        inter_pod_count = 0
        for demand in demands:
            src_pod = demand.src.split("_")[1] if len(demand.src.split("_")) > 1 else demand.src
            dst_pod = demand.dst.split("_")[1] if len(demand.dst.split("_")) > 1 else demand.dst
            if src_pod != dst_pod:
                inter_pod_count += 1

        # 4. Route demands over the active subgraph; QoS from end-to-end paths
        res = self._route_demands(demands, self.current_action, failed)
        self.link_loads = res["loads"]
        self.link_utilizations = res["utils"]
        self.last_flow_delays = res["delay_samples"]
        self.last_flow_weights = res["delay_weights"]

        total_demand_mbps = res["total_demand_mbps"]
        avg_delay_ms = res["avg_delay_ms"]
        packet_loss_rate = res["loss_rate"]
        max_utilization = res["max_utilization"]
        congested_links_count = res["congested_links"]

        # 5. Power & Energy (with EEE/LPI, port-off, deep sleep and transition costs)
        # In link-level WAN mode: Action 7 is legacy baseline without EEE (constant 12W active ports).
        # Action 6 is full topology with adaptive IEEE 802.3az EEE/LPI enabled on all links.
        # Actions 0..5 combine physical link sleep (0W off) with EEE/LPI on remaining links.
        if getattr(self, "is_link_level", False) and self.current_action == self.full_action:
            utils_for_power = None
        else:
            utils_for_power = {l: min(1.0, u) for l, u in self.link_utilizations.items()}

        power_info = self.power_model.calculate_total_power(
            switch_states=self.switch_states,
            link_states=link_states,
            switch_port_counts=self.switch_ports,
            link_utilizations=utils_for_power,
        )
        trans = self.power_model.transition_energy_j(
            self._prev_switch_states, self.switch_states, self._prev_link_states, link_states
        )
        step_energy_j = self.power_model.step_energy_j(
            power_info["total_power_w"], self.step_duration_s, trans["transition_energy_j"]
        )
        self.episode_energy_j += step_energy_j
        toggles = trans["switch_wakeups"] + trans["switch_sleeps"] + trans.get("port_wakeups", 0) + trans.get("port_sleeps", 0)
        self.episode_switch_toggles += toggles
        self._prev_switch_states = dict(self.switch_states)
        self._prev_link_states = dict(link_states)

        # 6. Multi-Objective Reward
        cfg = self.reward_cfg
        if getattr(self, "is_link_level", False):
            # Link-level energy reward: saving proportional to sleeping ports
            total_backbone_links = max(1.0, float(len(self.sleeping_links[0]) if self.sleeping_links[0] else len(self.links)))
            link_saving_ratio = len(sleep_now) / total_backbone_links
            energy_reward = link_saving_ratio * 16.0 * (self.energy_weight / 2.5) * cfg["energy"]
            waste_penalty = 0.0
        else:
            active_core_count = len(self.core_actions[self.current_action])
            total_core_count = float(len(self.core_switches))
            core_saving_ratio = (total_core_count - active_core_count) / max(1.0, total_core_count)
            energy_reward = core_saving_ratio * 16.0 * (self.energy_weight / 2.5) * cfg["energy"]

            # Over-provisioning / Waste Penalty (Prevents the lazy 'turn on everything' shortcut)
            estimated_safe_cores = max(1, int(np.ceil(total_demand_mbps / 1100.0)))
            waste_penalty = 0.0
            if active_core_count > estimated_safe_cores and max_utilization < 0.70:
                waste_penalty = (active_core_count - estimated_safe_cores) * 3.5 * cfg["waste"]

        # Delay Penalty with SLA threshold (end-to-end, traffic-weighted)
        delay_penalty = (avg_delay_ms / 2.5) * self.delay_weight * cfg["delay"]
        if avg_delay_ms > self.sla_delay_ms:
            delay_penalty += (avg_delay_ms - self.sla_delay_ms) * 6.0 * cfg["delay"]

        loss_penalty = (packet_loss_rate * 50.0) * self.loss_weight * cfg["loss"]
        congestion_penalty = (congested_links_count * 1.5 * cfg["congestion"]) if congested_links_count > 0 else 0.0

        # Route Flapping / Switching Penalty (Hysteresis) + dwell time enforcement + wake-up cost
        action_changed = self.prev_action is not None and self.current_action != self.prev_action
        switching_penalty = 0.2 * cfg["switching"] if action_changed else 0.0

        flapping_penalty = 0.0
        if action_changed:
            if self.steps_in_action < self.min_dwell_steps and not overridden:
                flapping_penalty = float(self.min_dwell_steps - self.steps_in_action) * 1.5 * cfg.get("switching", 1.0)
            self.steps_in_action = 0
            self.episode_action_changes += 1
        else:
            self.steps_in_action += 1

        wakeup_penalty = 0.5 * trans["switch_wakeups"] * cfg["wakeup"]
        port_wakeup_penalty = 0.05 * trans.get("port_wakeups", 0) * cfg["wakeup"]
        self.prev_action = self.current_action

        reward = float(energy_reward - waste_penalty - delay_penalty - loss_penalty
                       - congestion_penalty - switching_penalty - flapping_penalty
                       - wakeup_penalty - port_wakeup_penalty) * self.reward_scale

        # 7. Check Episode Termination
        terminated = False
        truncated = (self.current_step >= self.max_steps_per_episode)

        # 8. Next Observation & Info
        if self.traffic_source == "real":
            diurnal = float(np.clip(total_demand_mbps / 2000.0, 0.2, 3.0))
        else:
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
        sla_violation = bool(avg_delay_ms > self.sla_delay_ms or packet_loss_rate > self.sla_loss_rate)
        bl = res["bottleneck_link"]
        info.update({
            "traffic_source": self.traffic_source,
            "p95_delay_ms": round(float(res["p95_delay_ms"]), 3),
            "p99_delay_ms": round(float(res["p99_delay_ms"]), 3),
            "delivered_mbps": round(res["delivered_mbps"], 1),
            "bottleneck_link": f"{bl[0]}->{bl[1]}" if bl else None,
            "step_energy_j": float(step_energy_j),
            "step_energy_kwh": joules_to_kwh(step_energy_j),
            "episode_energy_kwh": joules_to_kwh(self.episode_energy_j),
            "transition_energy_j": float(trans["transition_energy_j"]),
            "switch_wakeups": int(trans["switch_wakeups"]),
            "switch_toggles": int(toggles),
            "port_wakeups": int(trans["port_wakeups"]),
            "action_changed": bool(action_changed),
            "episode_switch_toggles": int(self.episode_switch_toggles),
            "episode_action_changes": int(self.episode_action_changes),
            "green_mode": self.green_mode,
            "is_link_level": getattr(self, "is_link_level", False),
            "steps_in_action": self.steps_in_action,
            "sleeping_links": [f"{u}->{v}" for (u, v) in (self.sleeping_links[self.current_action] if getattr(self, "is_link_level", False) else [])],
            "energy_model_assumptions": (
                "Analytical simulation model based on IEEE 802.3az EEE/LPI and port admin-down "
                "(12W active, 1W LPI, 0W off). Not measured or verified on physical hardware."
            ),
            "ports_active": power_info.get("ports_active", 0),
            "ports_lpi": power_info.get("ports_lpi", 0),
            "ports_off": power_info.get("ports_off", 0),
            "sla_violation": sla_violation,
            "sla_delay_ms": self.sla_delay_ms,
            "failed_links": sorted(f"{u}->{v}" for (u, v) in failed),
            "requested_action": requested_action,
            "action_overridden": bool(overridden),
        })
        if self.traffic_source == "real":
            info.update({
                "real_timestamp": real_info.get("timestamp", ""),
                "real_flow_count": real_info.get("flow_count", 0),
                "real_split": self.real_traffic_split,
            })

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
