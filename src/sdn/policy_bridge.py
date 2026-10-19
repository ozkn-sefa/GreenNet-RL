"""
SDN policy bridge: real telemetry -> observation -> (PPO) inference -> safe action -> flow/port plan.

Pure Python (no Ryu / Mininet imports) so every decision step is unit-testable offline.
The Ryu app (mininet_scripts/ryu_green_controller.py) only does I/O: it feeds OpenFlow port statistics
into `TelemetryCollector`, asks `PolicyBridge.decide()` and applies the returned `Plan` in safe order.

What is REAL when run on Mininet/OVS: port statistics, flow-mods, OFPPortMod PORT_DOWN on core ports.
What is SIMULATED: chassis deep-sleep, EEE/LPI and every Watt/kWh figure (analytical model in
src/network/power_model.py). A "sleeping" core is emulated by bringing all of its ports down.

Known approximations (documented, not hidden):
  * Link load follows the env semantics: (tx+rx) of the link's switch-side port, in env capacity units.
  * inter-pod feature of the observation is estimated from core-link share (no per-flow stats).
  * Telemetry has a different sampling interval than the training step (720 s simulated).
"""

import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.agents.safety import TelemetrySafetyFilter


# ------------------------------------------------------------------ deterministic addressing
def build_dpid_map(topology) -> Dict[str, int]:
    return {sw: i + 1 for i, sw in enumerate(topology.switches)}


def build_port_map(topology) -> Dict[str, Dict[str, int]]:
    """port_map[node][neighbor] = port number (1-based, neighbors sorted by name). Used by BOTH the
    Mininet script (explicit port numbers) and the controller, so no topology discovery is needed."""
    return {n: {nb: i + 1 for i, nb in enumerate(sorted(topology.graph.neighbors(n)))} for n in topology.graph.nodes}


def host_mac(topology) -> Dict[str, str]:
    return {h: "00:00:00:00:00:%02x" % (i + 1) for i, h in enumerate(topology.hosts)}


def host_ip(topology) -> Dict[str, str]:
    return {h: f"10.0.0.{i + 1}" for i, h in enumerate(topology.hosts)}


# ------------------------------------------------------------------------------------ telemetry
class TelemetryCollector:
    """Turns cumulative OpenFlow port counters into per-link utilization (env units)."""

    def __init__(self, topology, link_capacity_mbps: float = 100.0, env_capacity_mbps: float = 1000.0,
                 stale_after_s: float = 30.0):
        self.topology = topology
        self.dpid_map = build_dpid_map(topology)
        self.name_of = {d: n for n, d in self.dpid_map.items()}
        self.port_map = build_port_map(topology)
        self.link_capacity_mbps = float(link_capacity_mbps)
        self.env_capacity_mbps = float(env_capacity_mbps)
        self.stale_after_s = stale_after_s
        self.links: List[Tuple[str, str]] = topology.get_links()
        self._prev: Dict[Tuple[int, int], Tuple[float, int, int]] = {}   # (dpid, port) -> (t, rx, tx)
        self._rate: Dict[Tuple[int, int], float] = {}                    # (dpid, port) -> (rx+tx) Mbps
        self._rx_rate: Dict[Tuple[int, int], float] = {}
        self.last_update: Optional[float] = None
        self.port_up: Dict[Tuple[int, int], bool] = {}

    def ingest_port_stats(self, dpid: int, stats: Dict[int, Tuple[int, int]], now: float):
        """stats: {port_no: (rx_bytes, tx_bytes)} (cumulative OFPPortStats counters)."""
        for port, (rx, tx) in stats.items():
            key = (dpid, port)
            if key in self._prev:
                t0, rx0, tx0 = self._prev[key]
                dt = now - t0
                if dt > 0 and rx >= rx0 and tx >= tx0:  # counter reset => skip sample
                    self._rate[key] = ((rx - rx0) + (tx - tx0)) * 8.0 / dt / 1e6
                    self._rx_rate[key] = (rx - rx0) * 8.0 / dt / 1e6
            self._prev[key] = (now, rx, tx)
        self.last_update = now

    def set_port_state(self, dpid: int, port_no: int, up: bool):
        self.port_up[(dpid, port_no)] = up

    def is_stale(self, now: float) -> bool:
        return self.last_update is None or (now - self.last_update) > self.stale_after_s

    def _switch_end(self, link: Tuple[str, str]) -> Tuple[str, str]:
        u, v = link
        return (u, v) if u in self.dpid_map else (v, u)

    def link_rates_mbps(self) -> Dict[Tuple[str, str], float]:
        out = {}
        for link in self.links:
            sw, nb = self._switch_end(link)
            key = (self.dpid_map[sw], self.port_map[sw][nb])
            out[link] = self._rate.get(key, 0.0)
        return out

    def failed_links(self) -> Set[Tuple[str, str]]:
        failed = set()
        for link in self.links:
            for node, other in (link, link[::-1]):
                if node in self.dpid_map and not self.port_up.get((self.dpid_map[node], self.port_map[node][other]), True):
                    failed.add(link)
        return failed

    def demand_mbps(self) -> float:
        """Offered load = traffic entering the fabric from hosts (host->edge direction)."""
        total = 0.0
        for link in self.links:
            u, v = link
            if (u in self.dpid_map) != (v in self.dpid_map):  # exactly one host endpoint
                sw, h = self._switch_end(link)
                total += self._rx_rate.get((self.dpid_map[sw], self.port_map[sw][h]), 0.0)
        return total

    def build_observation(self, env: Any, active_cores: Set[str], diurnal_factor: float) -> np.ndarray:
        """Same layout/normalization as GreenNetworkEnv._build_observation."""
        rates = self.link_rates_mbps()
        utils = [float(np.clip(rates[l] / self.link_capacity_mbps, 0.0, 2.0)) for l in env.links]
        demand_env = self.demand_mbps() * (self.env_capacity_mbps / self.link_capacity_mbps)
        core_set = set(env.core_switches)
        core_load_mbps = sum(rates[l] for l in env.links if l[0] in core_set or l[1] in core_set)
        inter_share = (core_load_mbps / 2.0) / self.demand_mbps() if self.demand_mbps() > 1e-9 else 0.0
        feats = [1.0 if c in active_cores else 0.0 for c in env.core_switches]
        feats += utils
        feats += [float(diurnal_factor), float(np.clip(demand_env / 10000.0, 0.0, 5.0)),
                  float(np.clip(inter_share, 0.0, 2.0))]
        return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------- forwarding plans
def compute_forwarding(topology, active_cores: Set[str], failed_links: Set[Tuple[str, str]] = frozenset()
                       ) -> Tuple[Dict[int, Dict[str, int]], bool]:
    """
    Destination-based, loop-free ECMP-style tables: for each destination host, every switch forwards to
    one deterministic (crc32-hashed) next hop among its shortest-path neighbors over the ACTIVE subgraph.
    Returns ({dpid: {eth_dst: out_port}}, all_hosts_reachable).
    """
    import networkx as nx
    dpid = build_dpid_map(topology)
    ports = build_port_map(topology)
    macs = host_mac(topology)
    cores = set(topology.core_switches)
    canon_failed = {tuple(sorted(l)) for l in failed_links}
    g = nx.Graph()
    for (u, v) in topology.get_links():
        if tuple(sorted((u, v))) in canon_failed:
            continue
        if (u in cores and u not in active_cores) or (v in cores and v not in active_cores):
            continue
        g.add_edge(u, v)
    tables: Dict[int, Dict[str, int]] = {dpid[s]: {} for s in topology.switches}
    reachable = True
    for dst in topology.hosts:
        if dst not in g:
            reachable = False
            continue
        dist = nx.single_source_shortest_path_length(g, dst)
        for sw in topology.switches:
            if sw not in dist:
                continue
            nxt = sorted(nb for nb in g.neighbors(sw) if dist.get(nb, 1e9) == dist[sw] - 1)
            if not nxt:
                continue
            choice = nxt[zlib.crc32(f"{sw}->{dst}".encode()) % len(nxt)]
            tables[dpid[sw]][macs[dst]] = ports[sw][choice]
        for h in topology.hosts:
            if h != dst and h not in dist:
                reachable = False
    return tables, reachable


def simulate_forwarding(topology, tables: Dict[int, Dict[str, int]], src: str, dst: str,
                        down_ports: Set[Tuple[int, int]] = frozenset(), max_hops: int = 32) -> Optional[List[str]]:
    """Follows the flow tables hop by hop; returns the node path or None (blackhole / loop / down port)."""
    dpid = build_dpid_map(topology)
    name_of = {d: n for n, d in dpid.items()}
    ports = build_port_map(topology)
    inv = {n: {p: nb for nb, p in m.items()} for n, m in ports.items()}
    mac = host_mac(topology)[dst]
    path = [src]
    node = next(iter(topology.graph.neighbors(src)))  # host's attached edge switch
    path.append(node)
    for _ in range(max_hops):
        out = tables.get(dpid[node], {}).get(mac)
        if out is None or (dpid[node], out) in down_ports:
            return None
        nxt = inv[node][out]
        path.append(nxt)
        if nxt == dst:
            return path
        if nxt not in dpid:
            return None
        node = nxt
    return None


# --------------------------------------------------------------------------------------- bridge
@dataclass
class Plan:
    """Ordered, make-before-break application plan."""
    action: int
    active_cores: List[str]
    ports_up: List[Tuple[int, int]] = field(default_factory=list)       # phase 1
    flow_add: List[Tuple[int, str, int]] = field(default_factory=list)  # phase 2: (dpid, eth_dst, out_port)
    flow_del: List[Tuple[int, str]] = field(default_factory=list)       # phase 2 (after add)
    ports_down: List[Tuple[int, int]] = field(default_factory=list)     # phase 3 (after drain)
    reachable: bool = True


@dataclass
class Decision:
    action: int
    requested: int
    overridden: bool
    reason: str
    observation: Optional[np.ndarray] = None


class PolicyBridge:
    """
    Decision logic of the SDN integration. `agent.predict(obs, deterministic=True)` can be a trained PPO
    (GreenRLAgent with use_shield=False) or any baseline.
    Safety layers: telemetry-staleness failsafe -> action filter -> reachability check -> dwell time
    (scale-ups are never delayed).
    """

    def __init__(self, env: Any, agent: Any, collector: TelemetryCollector, min_dwell_s: float = 60.0,
                 dry_run: bool = False, safety: Optional[TelemetrySafetyFilter] = None):
        self.env = env
        self.agent = agent
        self.collector = collector
        self.topology = env.topology
        self.min_dwell_s = min_dwell_s
        self.dry_run = dry_run
        self.safety = safety or TelemetrySafetyFilter(env)
        self.current_action = int(env.full_action)
        self.last_change_t: Optional[float] = None
        self.virtual_step = 0
        self.applied_tables: Dict[int, Dict[str, int]] = {}
        self.down_ports: Set[Tuple[int, int]] = set()
        self.history: deque = deque(maxlen=1000)
        from src.network.traffic import TrafficGenerator
        self._diurnal = TrafficGenerator(hosts=self.topology.hosts)

    def active_cores(self, action: Optional[int] = None) -> Set[str]:
        return set(self.env.core_actions[self.current_action if action is None else action])

    def decide(self, now: Optional[float] = None) -> Decision:
        now = time.time() if now is None else now
        self.virtual_step += 1
        if self.collector.is_stale(now):
            d = Decision(self.env.full_action, self.env.full_action, self.env.full_action != self.current_action,
                         "stale-telemetry-failsafe")
            self.history.append(d)
            return d
        diurnal = self._diurnal.get_diurnal_factor(self.virtual_step)
        obs = self.collector.build_observation(self.env, self.active_cores(), diurnal)
        requested = int(self.agent.predict(obs, deterministic=True))
        safe, overridden = self.safety.filter(obs, requested)
        reason = "safe-filter-override" if overridden else "policy"

        failed = self.collector.failed_links()
        _, ok = compute_forwarding(self.topology, self.active_cores(safe), failed)
        if not ok:  # candidate would disconnect hosts -> fall back to all cores
            safe, reason = int(self.env.full_action), "reachability-fallback"

        more_capacity = len(self.env.core_actions[safe]) > len(self.env.core_actions[self.current_action])
        if (safe != self.current_action and not more_capacity and self.last_change_t is not None
                and now - self.last_change_t < self.min_dwell_s):
            safe, reason = self.current_action, "dwell-hold"
        d = Decision(safe, requested, overridden, reason, obs)
        self.history.append(d)
        return d

    def make_plan(self, action: int) -> Plan:
        dpid = build_dpid_map(self.topology)
        ports = build_port_map(self.topology)
        active = self.active_cores(action)
        failed = self.collector.failed_links()
        tables, ok = compute_forwarding(self.topology, active, failed)
        plan = Plan(action=action, active_cores=sorted(active), reachable=ok)

        sleeping = set(self.topology.core_switches) - active
        want_down: Set[Tuple[int, int]] = set()
        for c in sleeping:
            for nb, p in ports[c].items():
                want_down.add((dpid[c], p))
                want_down.add((dpid[nb], ports[nb][c]))
        plan.ports_up = sorted(self.down_ports - want_down)
        plan.ports_down = sorted(want_down - self.down_ports)

        for d, tbl in tables.items():
            old = self.applied_tables.get(d, {})
            for mac, out in tbl.items():
                if old.get(mac) != out:
                    plan.flow_add.append((d, mac, out))
            for mac in old:
                if mac not in tbl:
                    plan.flow_del.append((d, mac))
        return plan

    def verify_plan(self, plan: Plan) -> bool:
        """Dry check before touching the network: all host pairs deliver, no sleeping core / down port used."""
        tables = {d: dict(t) for d, t in self.applied_tables.items()}
        for d, mac, out in plan.flow_add:
            tables.setdefault(d, {})[mac] = out
        for d, mac in plan.flow_del:
            tables.get(d, {}).pop(mac, None)
        down = (self.down_ports - set(plan.ports_up)) | set(plan.ports_down)
        sleeping = set(self.topology.core_switches) - set(plan.active_cores)
        for s in self.topology.hosts:
            for t in self.topology.hosts:
                if s == t:
                    continue
                path = simulate_forwarding(self.topology, tables, s, t, down)
                if path is None or any(n in sleeping for n in path):
                    return False
        return plan.reachable

    def commit(self, plan: Plan, now: Optional[float] = None):
        """Record that `plan` has been applied."""
        now = time.time() if now is None else now
        for d, mac, out in plan.flow_add:
            self.applied_tables.setdefault(d, {})[mac] = out
        for d, mac in plan.flow_del:
            self.applied_tables.get(d, {}).pop(mac, None)
        self.down_ports = (self.down_ports - set(plan.ports_up)) | set(plan.ports_down)
        if plan.action != self.current_action:
            self.last_change_t = now
        self.current_action = plan.action
