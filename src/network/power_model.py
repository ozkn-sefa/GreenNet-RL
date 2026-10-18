"""
Power consumption model for data center switches and network links.
Based on IEEE 802.3az Energy-Efficient Ethernet (EEE) and data center green networking benchmarks.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SwitchPowerProfile:
    """
    Hardware energy profile for a network switch.
    """
    model_name: str = "Enterprise-Core-Switch"
    chassis_base_power_w: float = 140.0   # Active idle power (chassis, fans, CPU)
    switch_sleep_power_w: float = 8.0     # Deep sleep / standby power
    port_active_power_w: float = 12.0     # Power per active 10G link port
    port_sleep_power_w: float = 1.0       # Power per port in EEE Low Power Idle (LPI)
    wakeup_energy_penalty_j: float = 25.0 # Energy spike during sleep-to-active transition


class NetworkPowerModel:
    """
    Computes total real-time network power consumption in Watts and accumulated energy in Joules / kWh.
    """

    def __init__(self, profile: SwitchPowerProfile = None):
        self.profile = profile or SwitchPowerProfile()
        self.total_energy_consumed_joules = 0.0

    def calculate_switch_power(self, is_active: bool, active_ports_count: int, total_ports_count: int) -> float:
        """
        Calculates power for a single switch based on its operational state.
        """
        if not is_active:
            return self.profile.switch_sleep_power_w

        # Switch is active
        power = self.profile.chassis_base_power_w
        # Active ports vs sleeping ports
        power += active_ports_count * self.profile.port_active_power_w
        sleeping_ports = max(0, total_ports_count - active_ports_count)
        power += sleeping_ports * self.profile.port_sleep_power_w
        return power

    def calculate_total_power(self, switch_states: Dict[str, bool], link_states: Dict[tuple, bool], switch_port_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Calculates total power consumption across all switches and links.
        Returns a breakdown dictionary.
        """
        total_switch_power = 0.0
        active_switches = 0
        sleeping_switches = 0

        # Calculate active ports per switch from active links
        switch_active_ports: Dict[str, int] = {s: 0 for s in switch_states}
        for (u, v), is_link_active in link_states.items():
            if is_link_active:
                if u in switch_active_ports:
                    switch_active_ports[u] += 1
                if v in switch_active_ports:
                    switch_active_ports[v] += 1

        for sw_id, is_active in switch_states.items():
            total_ports = switch_port_counts.get(sw_id, 8)
            active_ports = switch_active_ports.get(sw_id, 0)
            
            p = self.calculate_switch_power(is_active, active_ports, total_ports)
            total_switch_power += p

            if is_active:
                active_switches += 1
            else:
                sleeping_switches += 1

        # Max theoretical power (when everything is active and all ports awake)
        max_possible_power = sum(
            self.profile.chassis_base_power_w + (switch_port_counts.get(s, 8) * self.profile.port_active_power_w)
            for s in switch_states
        )

        energy_savings_percent = 0.0
        if max_possible_power > 0:
            energy_savings_percent = max(0.0, (1.0 - (total_switch_power / max_possible_power)) * 100.0)

        return {
            "total_power_w": round(total_switch_power, 2),
            "max_power_w": round(max_possible_power, 2),
            "energy_savings_percent": round(energy_savings_percent, 2),
            "active_switches": active_switches,
            "sleeping_switches": sleeping_switches,
        }
