"""
Power consumption model for data center switches and network links.

IMPORTANT: This is an analytical SIMULATION model. None of the power states
below (switch deep-sleep, EEE LPI, wake-up energy) are measured on hardware or
actually enforced by Open vSwitch/Mininet. Parameters are assumptions typical of
the IEEE 802.3az / green-networking literature and are configurable.

Three distinct, mutually exclusive mechanisms are modelled:
  1. Switch deep sleep  : whole chassis asleep -> `switch_sleep_power_w`, all ports off.
  2. Port admin-down    : link/port disabled (peer asleep, failed, or turned off) -> `port_off_power_w`.
  3. EEE / LPI          : link is UP but idle; port power interpolates between
                          `port_lpi_power_w` and `port_active_power_w` with the active-time fraction.

Units: power in W, energy in J (= W * s), reporting in kWh (= J / 3.6e6).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

JOULES_PER_KWH = 3.6e6


def joules_to_kwh(joules: float) -> float:
    return float(joules) / JOULES_PER_KWH


def watts_to_joules(power_w: float, duration_s: float) -> float:
    return float(power_w) * float(duration_s)


class PortState(str, Enum):
    ACTIVE = "active"   # link up, carrying traffic (EEE active fraction applies)
    LPI = "lpi"         # link up, EEE Low Power Idle (no traffic)
    OFF = "off"         # port administratively down / peer asleep / failed link


@dataclass
class SwitchPowerProfile:
    """Hardware energy profile for a network switch (simulation assumptions)."""
    model_name: str = "Enterprise-Core-Switch"
    chassis_base_power_w: float = 140.0   # Active idle power (chassis, fans, CPU)
    switch_sleep_power_w: float = 8.0     # Deep sleep / standby power (whole chassis)
    port_active_power_w: float = 12.0     # Power per fully active link port
    port_lpi_power_w: float = 1.0         # Power per port in EEE Low Power Idle
    port_off_power_w: float = 0.0         # Power per administratively-down port
    eee_min_active_fraction: float = 0.05  # LPI refresh/wake overhead: min fraction of time in active
    lpi_util_threshold: float = 0.0       # utilization <= threshold => port is in pure LPI
    # Transition costs (energy spent while changing state)
    wakeup_energy_penalty_j: float = 3000.0  # switch sleep -> active (boot / link training), ~140 W * ~20 s
    port_wakeup_energy_j: float = 2.0        # admin-down -> up of a single port (link training)
    wakeup_latency_s: float = 20.0           # informational: time before a woken switch carries traffic


class NetworkPowerModel:
    """
    Computes real-time network power (W), transition energy (J) and accumulated energy (J / kWh).
    """

    def __init__(self, profile: SwitchPowerProfile = None):
        self.profile = profile or SwitchPowerProfile()
        self.total_energy_consumed_joules = 0.0

    # ------------------------------------------------------------------ ports
    def port_state(self, link_up: bool, utilization: float = 1.0) -> PortState:
        """EEE/port-off classification. If utilization is unknown (None) the port is treated as active."""
        if not link_up:
            return PortState.OFF
        if utilization is not None and utilization <= self.profile.lpi_util_threshold:
            return PortState.LPI
        return PortState.ACTIVE

    def calculate_port_power(self, link_up: bool, utilization: Optional[float] = None) -> float:
        """
        Port power in W.
          OFF    -> port_off_power_w
          LPI    -> port_lpi_power_w
          ACTIVE -> LPI + (active - LPI) * clip(util + eee_min_active_fraction, 0, 1)
        With utilization=None the port is assumed fully active (legacy behaviour).
        """
        p = self.profile
        if not link_up:
            return p.port_off_power_w
        if utilization is None:
            return p.port_active_power_w
        if utilization <= p.lpi_util_threshold:
            return p.port_lpi_power_w
        frac = min(1.0, max(0.0, utilization) + p.eee_min_active_fraction)
        return p.port_lpi_power_w + (p.port_active_power_w - p.port_lpi_power_w) * frac

    # --------------------------------------------------------------- switches
    def calculate_switch_power(self, is_active: bool, active_ports_count: int, total_ports_count: int) -> float:
        """
        Legacy per-switch API. Active ports are fully active; the remaining ports are
        administratively OFF (previously mis-modelled as LPI). Sleeping switch = deep sleep.
        """
        if not is_active:
            return self.profile.switch_sleep_power_w
        off_ports = max(0, total_ports_count - active_ports_count)
        return (
            self.profile.chassis_base_power_w
            + active_ports_count * self.profile.port_active_power_w
            + off_ports * self.profile.port_off_power_w
        )

    def calculate_total_power(
        self,
        switch_states: Dict[str, bool],
        link_states: Dict[tuple, bool],
        switch_port_counts: Dict[str, int],
        link_utilizations: Optional[Dict[tuple, float]] = None,
    ) -> Dict[str, float]:
        """
        Total power across switches. If `link_utilizations` is given, EEE/LPI scaling applies
        to active ports; otherwise active ports draw full port power.
        Returns a breakdown dictionary (power in W).
        """
        prof = self.profile
        total_switch_power = 0.0
        active_switches = 0
        sleeping_switches = 0
        ports_active = ports_lpi = ports_off = 0

        # Port power per awake switch; ports of sleeping switches are covered by deep-sleep power.
        port_power: Dict[str, float] = {s: 0.0 for s in switch_states}
        linked_ports: Dict[str, int] = {s: 0 for s in switch_states}
        for link, is_up in link_states.items():
            util = link_utilizations.get(link) if link_utilizations is not None else None
            for node in link:
                if node in port_power and switch_states.get(node, True):
                    linked_ports[node] += 1
                    port_power[node] += self.calculate_port_power(is_up, util)
                    if not is_up:
                        ports_off += 1
                    elif util is not None and util <= prof.lpi_util_threshold:
                        ports_lpi += 1
                    else:
                        ports_active += 1

        for sw_id, is_active in switch_states.items():
            total_ports = switch_port_counts.get(sw_id, 8)
            if is_active:
                total_switch_power += prof.chassis_base_power_w + port_power[sw_id]
                active_switches += 1
                # ports without any link entry are administratively off (power = port_off_power_w)
                unlinked = max(0, total_ports - linked_ports[sw_id])
                total_switch_power += unlinked * prof.port_off_power_w
                ports_off += unlinked
            else:
                total_switch_power += prof.switch_sleep_power_w
                sleeping_switches += 1
                ports_off += total_ports

        # Reference: everything awake with all ports fully active (no sleep, no EEE)
        max_possible_power = sum(
            prof.chassis_base_power_w + (switch_port_counts.get(s, 8) * prof.port_active_power_w)
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
            "ports_active": ports_active,
            "ports_lpi": ports_lpi,
            "ports_off": ports_off,
        }

    # ------------------------------------------------------------ transitions
    def transition_energy_j(
        self,
        prev_switch_states: Optional[Dict[str, bool]],
        new_switch_states: Dict[str, bool],
        prev_link_states: Optional[Dict[tuple, bool]] = None,
        new_link_states: Optional[Dict[tuple, bool]] = None,
    ) -> Dict[str, float]:
        """
        Energy (J) spent on state changes between two steps:
          - switch sleep -> active:  wakeup_energy_penalty_j each (its ports are included)
          - port off -> up (only for links whose endpoints did not just wake): port_wakeup_energy_j each
        Also returns counts of switch and port transitions (both directions) for switching metrics.
        """
        prof = self.profile
        switch_wakeups = 0
        switch_sleeps = 0
        woken: set = set()
        if prev_switch_states is not None:
            for sw, now_on in new_switch_states.items():
                was_on = prev_switch_states.get(sw, True)
                if now_on and not was_on:
                    switch_wakeups += 1
                    woken.add(sw)
                elif was_on and not now_on:
                    switch_sleeps += 1

        port_wakeups = 0
        port_sleeps = 0
        if prev_link_states is not None and new_link_states is not None:
            for link, up in new_link_states.items():
                was_up = prev_link_states.get(link, True)
                if up == was_up:
                    continue
                if up and not (link[0] in woken or link[1] in woken):
                    port_wakeups += 1
                elif not up and not (link[0] in self._slept(prev_switch_states, new_switch_states)
                                     or link[1] in self._slept(prev_switch_states, new_switch_states)):
                    port_sleeps += 1

        energy = switch_wakeups * prof.wakeup_energy_penalty_j + port_wakeups * prof.port_wakeup_energy_j
        return {
            "transition_energy_j": float(energy),
            "switch_wakeups": switch_wakeups,
            "switch_sleeps": switch_sleeps,
            "port_wakeups": port_wakeups,
            "port_sleeps": port_sleeps,
        }

    @staticmethod
    def _slept(prev: Optional[Dict[str, bool]], new: Dict[str, bool]) -> set:
        if prev is None:
            return set()
        return {s for s, on in new.items() if (not on) and prev.get(s, True)}

    def step_energy_j(self, power_w: float, duration_s: float, transition_j: float = 0.0) -> float:
        """Energy of one control interval in J: P * dt + transition energy. Accumulates the total."""
        e = watts_to_joules(power_w, duration_s) + float(transition_j)
        self.total_energy_consumed_joules += e
        return e

    @property
    def total_energy_kwh(self) -> float:
        return joules_to_kwh(self.total_energy_consumed_joules)

    def reset_energy_counter(self):
        self.total_energy_consumed_joules = 0.0
