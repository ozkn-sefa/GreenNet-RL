"""
Network Topology Generator for Data Center Networks (Fat-Tree) and Custom SDN Topologies.
Integrates with NetworkX and exports topological structures compatible with Mininet.
"""

from typing import Dict, List, Tuple, Set, Any
import networkx as nx
import itertools


class FatTreeTopology:
    """
    Standard Fat-Tree (k-ary) Data Center Network Topology.
    For k=4:
      - 4 Core switches (c00, c01, c10, c11)
      - 4 Pods, each with 2 Aggregation switches and 2 Edge switches (16 Pod switches)
      - 16 Hosts (2 per Edge switch)
      - Total nodes: 20 switches + 16 hosts = 36 nodes.
      - Rich multi-path redundancy between pods (ideal for Green Routing & traffic consolidation).
    """

    def __init__(self, k: int = 4, link_capacity_mbps: float = 1000.0, default_delay_ms: float = 1.0):
        self.k = k
        self.link_capacity_mbps = link_capacity_mbps
        self.default_delay_ms = default_delay_ms
        
        self.core_switches: List[str] = []
        self.agg_switches: List[str] = []
        self.edge_switches: List[str] = []
        self.switches: List[str] = []
        self.hosts: List[str] = []
        
        self.graph = nx.Graph()
        self._build_topology()
        self._precompute_k_paths()

    def _build_topology(self):
        k = self.k
        num_core = (k // 2) ** 2
        
        # 1. Create Core Switches
        for i in range(k // 2):
            for j in range(k // 2):
                c_name = f"c_{i}_{j}"
                self.core_switches.append(c_name)
                self.graph.add_node(c_name, type="core", layer=3, pos_y=300)

        # 2. Create Pods
        for pod in range(k):
            # Aggregation switches
            for agg in range(k // 2):
                a_name = f"a_{pod}_{agg}"
                self.agg_switches.append(a_name)
                self.graph.add_node(a_name, type="aggregation", pod=pod, layer=2, pos_y=200)

                # Connect Agg to Core
                # agg switch j in pod connects to core switches (j, *)
                for core_idx in range(k // 2):
                    c_name = f"c_{agg}_{core_idx}"
                    self.graph.add_edge(
                        a_name, c_name,
                        capacity=self.link_capacity_mbps,
                        delay=self.default_delay_ms,
                        active=True,
                        type="core-agg"
                    )

            # Edge switches
            for edge in range(k // 2):
                e_name = f"e_{pod}_{edge}"
                self.edge_switches.append(e_name)
                self.graph.add_node(e_name, type="edge", pod=pod, layer=1, pos_y=100)

                # Connect Edge to all Agg switches in the same pod
                for agg in range(k // 2):
                    a_name = f"a_{pod}_{agg}"
                    self.graph.add_edge(
                        e_name, a_name,
                        capacity=self.link_capacity_mbps,
                        delay=self.default_delay_ms,
                        active=True,
                        type="agg-edge"
                    )

                # Connect Edge to Hosts
                for h in range(k // 2):
                    h_name = f"h_{pod}_{edge}_{h}"
                    self.hosts.append(h_name)
                    self.graph.add_node(h_name, type="host", pod=pod, layer=0, pos_y=0)
                    self.graph.add_edge(
                        e_name, h_name,
                        capacity=self.link_capacity_mbps,
                        delay=self.default_delay_ms,
                        active=True,
                        type="edge-host"
                    )

        self.switches = self.core_switches + self.agg_switches + self.edge_switches

        # Assign layout coordinates (x, y) for UI visualizer
        self._calculate_node_positions()

    def _calculate_node_positions(self):
        """Precomputes 2D coordinates for visual layout."""
        # Core switches along top
        core_step = 1000.0 / (len(self.core_switches) + 1)
        for idx, c in enumerate(self.core_switches):
            self.graph.nodes[c]["x"] = (idx + 1) * core_step
            self.graph.nodes[c]["y"] = 40

        # Aggregation and Edge switches grouped by pod
        pod_width = 1000.0 / self.k
        for pod in range(self.k):
            pod_start = pod * pod_width
            # Agg
            for agg_idx in range(self.k // 2):
                a_name = f"a_{pod}_{agg_idx}"
                self.graph.nodes[a_name]["x"] = pod_start + (agg_idx + 1) * (pod_width / (self.k // 2 + 1))
                self.graph.nodes[a_name]["y"] = 140
            # Edge
            for edge_idx in range(self.k // 2):
                e_name = f"e_{pod}_{edge_idx}"
                self.graph.nodes[e_name]["x"] = pod_start + (edge_idx + 1) * (pod_width / (self.k // 2 + 1))
                self.graph.nodes[e_name]["y"] = 240

        # Hosts along bottom
        host_step = 1000.0 / (len(self.hosts) + 1)
        for idx, h in enumerate(self.hosts):
            self.graph.nodes[h]["x"] = (idx + 1) * host_step
            self.graph.nodes[h]["y"] = 340

    def _precompute_k_paths(self, k_paths: int = 4):
        """
        Precomputes k-shortest paths between all host pairs to allow instantaneous RL routing actions.
        """
        self.k_shortest_paths: Dict[Tuple[str, str], List[List[str]]] = {}
        for src in self.hosts:
            for dst in self.hosts:
                if src == dst:
                    continue
                try:
                    # Find simple paths sorted by length
                    paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, src, dst), k_paths))
                    self.k_shortest_paths[(src, dst)] = paths
                except nx.NetworkXNoPath:
                    self.k_shortest_paths[(src, dst)] = []

    def get_candidate_paths(self, src: str, dst: str) -> List[List[str]]:
        return self.k_shortest_paths.get((src, dst), [])

    def get_switch_port_counts(self) -> Dict[str, int]:
        """Returns the degree (number of connected ports) for each switch."""
        return {sw: self.graph.degree[sw] for sw in self.switches}

    def get_links(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges())


class NSFNETTopology:
    """
    Standard NSFNET (National Science Foundation Network) 14-Node WAN Backbone Topology.
    Used widely in optical & IP green routing literature to demonstrate wide-area energy management.
    Contains 14 backbone router switches, 14 local edge host nodes, and 21 inter-city backbone links.
    """

    # 14 NSFNET inter-city nodes with realistic relative layout positions
    NSFNET_NODES = [
        ("sea", "Seattle", 100, 60),
        ("scl", "Salt Lake City", 200, 140),
        ("sfo", "San Francisco", 80, 180),
        ("san", "San Diego", 120, 280),
        ("bld", "Boulder", 320, 160),
        ("linc", "Lincoln", 440, 140),
        ("hou", "Houston", 480, 310),
        ("ann", "Ann Arbor", 620, 100),
        ("urb", "Urbana", 560, 160),
        ("atl", "Atlanta", 680, 270),
        ("pit", "Pittsburgh", 740, 130),
        ("wdc", "Washington DC", 820, 180),
        ("prn", "Princeton", 860, 120),
        ("ith", "Ithaca", 800, 70),
    ]

    # 21 standard bidirectional backbone links
    NSFNET_EDGES = [
        ("sea", "scl"), ("sea", "sfo"),
        ("scl", "bld"), ("scl", "san"),
        ("sfo", "san"), ("sfo", "linc"),
        ("san", "hou"),
        ("bld", "ann"), ("bld", "urb"),
        ("linc", "hou"), ("linc", "urb"),
        ("hou", "atl"),
        ("ann", "urb"), ("ann", "pit"),
        ("urb", "wdc"),
        ("atl", "wdc"), ("atl", "prn"),
        ("pit", "prn"), ("pit", "ith"),
        ("wdc", "ith"),
        ("prn", "ith"),
    ]

    def __init__(self, link_capacity_mbps: float = 2500.0, default_delay_ms: float = 5.0):
        self.name = "NSFNET-WAN"
        self.link_capacity_mbps = link_capacity_mbps
        self.default_delay_ms = default_delay_ms

        self.switches: List[str] = [n[0] for n in self.NSFNET_NODES]
        self.core_switches: List[str] = ["bld", "linc", "urb", "ann"]  # Central transit hubs with sleepable bypasses
        self.agg_switches: List[str] = ["scl", "hou", "atl", "pit"]
        self.edge_switches: List[str] = ["sea", "sfo", "san", "wdc", "prn", "ith"]
        self.hosts: List[str] = [f"h_{sw}" for sw in self.switches]

        self.graph = nx.Graph()
        self._build_topology()
        self._precompute_k_paths()

    def _build_topology(self):
        # 1. Add backbone switches
        for sw_id, label, x, y in self.NSFNET_NODES:
            sw_type = "core" if sw_id in self.core_switches else ("aggregation" if sw_id in self.agg_switches else "edge")
            self.graph.add_node(sw_id, type=sw_type, label=label, x=x, y=y)

        # 2. Add backbone links
        for u, v in self.NSFNET_EDGES:
            self.graph.add_edge(
                u, v,
                capacity=self.link_capacity_mbps,
                delay=self.default_delay_ms,
                active=True,
                type="wan-backbone"
            )

        # 3. Add hosts connected to each switch
        for sw_id, label, x, y in self.NSFNET_NODES:
            h_id = f"h_{sw_id}"
            self.graph.add_node(h_id, type="host", label=f"Host-{label}", x=x, y=y + 40)
            self.graph.add_edge(
                sw_id, h_id,
                capacity=self.link_capacity_mbps,
                delay=0.5,
                active=True,
                type="host-edge"
            )

    def _precompute_k_paths(self, k_paths: int = 4):
        self.k_shortest_paths: Dict[Tuple[str, str], List[List[str]]] = {}
        for src in self.hosts:
            for dst in self.hosts:
                if src == dst:
                    continue
                try:
                    paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, src, dst), k_paths))
                    self.k_shortest_paths[(src, dst)] = paths
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    self.k_shortest_paths[(src, dst)] = []

    def get_candidate_paths(self, src: str, dst: str) -> List[List[str]]:
        return self.k_shortest_paths.get((src, dst), [])

    def get_switch_port_counts(self) -> Dict[str, int]:
        return {sw: self.graph.degree[sw] for sw in self.switches}

    def get_links(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges())


class AbileneTopology:
    """
    Standard Internet2 Abilene 12-Node WAN Backbone Topology from SNDlib.
    Contains 12 backbone router switches, 15 inter-city backbone links,
    and 12 attached edge hosts for traffic injection and validation.
    """

    # 12 SNDlib Abilene nodes with real geographic coordinates (lon, lat) mapped to 2D layout
    ABILENE_NODES = [
        ("ATLAM5", "Atlanta-M5", -84.3833, 33.75, 780, 270),
        ("ATLAng", "Atlanta-ng", -85.50, 34.50, 750, 250),
        ("CHINng", "Chicago-ng", -87.6167, 41.8333, 620, 130),
        ("DNVRng", "Denver-ng", -105.00, 40.75, 340, 160),
        ("HSTNng", "Houston-ng", -95.517364, 29.770031, 520, 310),
        ("IPLSng", "Indianapolis-ng", -86.159535, 39.780622, 650, 180),
        ("KSCYng", "Kansas City-ng", -96.596704, 38.961694, 480, 180),
        ("LOSAng", "Los Angeles-ng", -118.25, 34.05, 120, 260),
        ("NYCMng", "New York-ng", -73.9667, 40.7833, 860, 120),
        ("SNVAng", "Sunnyvale-ng", -122.02553, 37.38575, 80, 180),
        ("STTLng", "Seattle-ng", -122.30, 47.60, 100, 50),
        ("WASHng", "Washington-ng", -77.026842, 38.897303, 830, 190),
    ]

    # 15 standard bidirectional backbone links in SNDlib
    ABILENE_EDGES = [
        ("ATLAM5", "ATLAng"),
        ("ATLAng", "CHINng"),
        ("ATLAng", "HSTNng"),
        ("ATLAng", "WASHng"),
        ("CHINng", "IPLSng"),
        ("CHINng", "NYCMng"),
        ("DNVRng", "KSCYng"),
        ("DNVRng", "SNVAng"),
        ("DNVRng", "STTLng"),
        ("HSTNng", "KSCYng"),
        ("HSTNng", "LOSAng"),
        ("IPLSng", "KSCYng"),
        ("LOSAng", "SNVAng"),
        ("NYCMng", "WASHng"),
        ("SNVAng", "STTLng"),
    ]

    def __init__(self, link_capacity_mbps: float = 2500.0, default_delay_ms: float = 5.0):
        self.name = "Abilene-WAN"
        self.link_capacity_mbps = link_capacity_mbps
        self.default_delay_ms = default_delay_ms

        self.switches: List[str] = [n[0] for n in self.ABILENE_NODES]
        # 4 central transit hubs as core switches for 8-action RL policy compatibility
        self.core_switches: List[str] = ["KSCYng", "IPLSng", "DNVRng", "CHINng"]
        self.agg_switches: List[str] = ["ATLAng", "HSTNng", "SNVAng", "NYCMng"]
        self.edge_switches: List[str] = ["ATLAM5", "LOSAng", "STTLng", "WASHng"]
        self.hosts: List[str] = [f"h_{sw}" for sw in self.switches]

        self.graph = nx.Graph()
        self._build_topology()
        self._precompute_k_paths()

    def _build_topology(self):
        # 1. Add backbone router switches
        for sw_id, label, lon, lat, x, y in self.ABILENE_NODES:
            sw_type = "core" if sw_id in self.core_switches else ("aggregation" if sw_id in self.agg_switches else "edge")
            self.graph.add_node(sw_id, type=sw_type, label=label, lon=lon, lat=lat, x=x, y=y)

        # 2. Add backbone links
        for u, v in self.ABILENE_EDGES:
            self.graph.add_edge(
                u, v,
                capacity=self.link_capacity_mbps,
                delay=self.default_delay_ms,
                active=True,
                type="wan-backbone"
            )

        # 3. Add hosts connected to each switch
        for sw_id, label, lon, lat, x, y in self.ABILENE_NODES:
            h_id = f"h_{sw_id}"
            self.graph.add_node(h_id, type="host", label=f"Host-{label}", x=x, y=y + 35)
            self.graph.add_edge(
                sw_id, h_id,
                capacity=self.link_capacity_mbps,
                delay=0.5,
                active=True,
                type="host-edge"
            )

    def _precompute_k_paths(self, k_paths: int = 4):
        self.k_shortest_paths: Dict[Tuple[str, str], List[List[str]]] = {}
        for src in self.hosts:
            for dst in self.hosts:
                if src == dst:
                    continue
                try:
                    paths = list(itertools.islice(nx.shortest_simple_paths(self.graph, src, dst), k_paths))
                    self.k_shortest_paths[(src, dst)] = paths
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    self.k_shortest_paths[(src, dst)] = []

    def get_candidate_paths(self, src: str, dst: str) -> List[List[str]]:
        return self.k_shortest_paths.get((src, dst), [])

    def get_switch_port_counts(self) -> Dict[str, int]:
        return {sw: self.graph.degree[sw] for sw in self.switches}

    def get_links(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges())


class TopologyFactory:
    """Factory to instantiate parameterized Fat-Tree and WAN topologies."""

    @staticmethod
    def create(topology_type: str = "fattree_k4", link_capacity_mbps: float = 1000.0) -> Any:
        topo = topology_type.lower()
        if topo in ["fattree_k4", "k4", "fattree-k4"]:
            return FatTreeTopology(k=4, link_capacity_mbps=link_capacity_mbps)
        elif topo in ["fattree_k6", "k6", "fattree-k6"]:
            return FatTreeTopology(k=6, link_capacity_mbps=link_capacity_mbps)
        elif topo in ["nsfnet", "wan", "nsfnet_wan"]:
            return NSFNETTopology(link_capacity_mbps=link_capacity_mbps)
        elif topo in ["abilene", "abilenetopology", "abilene_wan"]:
            return AbileneTopology(link_capacity_mbps=link_capacity_mbps)
        else:
            return FatTreeTopology(k=4, link_capacity_mbps=link_capacity_mbps)


