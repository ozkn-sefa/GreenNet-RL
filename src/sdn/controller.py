"""
Software-Defined Networking (SDN) Controller Abstraction.
Simulates OpenFlow controller behavior (Ryu/OpenDaylight/ONOS paradigm):
- Flow-Mod installation
- Port statistics polling (OpenFlow PortStats)
- Link power state management (IEEE 802.3az EEE port power toggles)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Any
import time


@dataclass
class FlowEntry:
    table_id: int
    priority: int
    match: Dict[str, Any]
    actions: List[str]
    packet_count: int = 0
    byte_count: int = 0
    idle_timeout: int = 30


@dataclass
class PortStats:
    port_no: int
    rx_packets: int = 0
    tx_packets: int = 0
    rx_bytes: int = 0
    tx_bytes: int = 0
    is_down_or_sleep: bool = False
    link_utilization: float = 0.0


class SDNController:
    """
    OpenFlow SDN Controller abstraction managing the data plane switches.
    Communicates with the RL Optimizer to translate high-level routing/sleep policies into OpenFlow FlowMods.
    """

    def __init__(self, name: str = "GreenSDN-Controller"):
        self.name = name
        # Switch DPID -> List of FlowEntries
        self.flow_tables: Dict[str, List[FlowEntry]] = {}
        # Switch DPID -> Port number -> PortStats
        self.port_statistics: Dict[str, Dict[int, PortStats]] = {}
        # Topology active paths
        self.active_routes: Dict[Tuple[str, str], List[str]] = {}

    def register_switch(self, dpid: str, port_count: int):
        """Registers an OpenFlow switch with the controller."""
        self.flow_tables[dpid] = []
        self.port_statistics[dpid] = {
            p: PortStats(port_no=p) for p in range(1, port_count + 1)
        }

    def install_flow_rule(self, dpid: str, match: Dict[str, Any], actions: List[str], priority: int = 100):
        """Installs an OpenFlow Flow-Mod entry on the target switch."""
        entry = FlowEntry(
            table_id=0,
            priority=priority,
            match=match,
            actions=actions
        )
        if dpid not in self.flow_tables:
            self.flow_tables[dpid] = []
        self.flow_tables[dpid].append(entry)

    def set_port_power_state(self, dpid: str, port_no: int, sleep: bool):
        """Sends OpenFlow Port-Mod to enable/disable or low-power idle a port."""
        if dpid in self.port_statistics and port_no in self.port_statistics[dpid]:
            self.port_statistics[dpid][port_no].is_down_or_sleep = sleep

    def clear_flow_tables(self):
        """Flushes all flow entries (e.g. upon routing recalculation)."""
        for dpid in self.flow_tables:
            self.flow_tables[dpid].clear()

    def get_controller_telemetry(self) -> Dict[str, Any]:
        """Returns statistics for reporting and dashboard integration."""
        total_rules = sum(len(rules) for rules in self.flow_tables.values())
        sleeping_ports_count = sum(
            1 for ports in self.port_statistics.values() for p in ports.values() if p.is_down_or_sleep
        )
        return {
            "controller_name": self.name,
            "managed_switches": len(self.flow_tables),
            "active_flow_rules": total_rules,
            "sleeping_ports": sleeping_ports_count
        }
