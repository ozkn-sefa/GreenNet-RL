import pytest

from src.network.power_model import (
    NetworkPowerModel, SwitchPowerProfile, joules_to_kwh, watts_to_joules, JOULES_PER_KWH,
)
from src.env.green_network_env import GreenNetworkEnv


def test_unit_conversions():
    assert watts_to_joules(1000.0, 3600.0) == pytest.approx(JOULES_PER_KWH)
    assert joules_to_kwh(3.6e6) == pytest.approx(1.0)
    assert joules_to_kwh(watts_to_joules(500.0, 7200.0)) == pytest.approx(1.0)


def test_port_states_are_distinct():
    m = NetworkPowerModel()
    p = m.profile
    off = m.calculate_port_power(False, 0.0)
    lpi = m.calculate_port_power(True, 0.0)
    busy = m.calculate_port_power(True, 1.0)
    assert off == p.port_off_power_w < lpi == p.port_lpi_power_w < busy == p.port_active_power_w
    mid = m.calculate_port_power(True, 0.5)
    assert lpi < mid < busy
    assert mid == pytest.approx(p.port_lpi_power_w + (p.port_active_power_w - p.port_lpi_power_w) * (0.5 + p.eee_min_active_fraction))


def test_switch_sleep_differs_from_port_off():
    m = NetworkPowerModel()
    states = {"s": False}
    sleeping = m.calculate_total_power(states, {("s", "h"): False}, {"s": 4})["total_power_w"]
    assert sleeping == m.profile.switch_sleep_power_w
    awake_all_ports_off = m.calculate_total_power({"s": True}, {("s", "h"): False}, {"s": 1})["total_power_w"]
    assert awake_all_ports_off == pytest.approx(m.profile.chassis_base_power_w + m.profile.port_off_power_w)
    assert sleeping < awake_all_ports_off


def test_idle_links_use_lpi_active_links_use_full_power():
    m = NetworkPowerModel()
    link = ("s", "h")
    full = m.calculate_total_power({"s": True}, {link: True}, {"s": 1})["total_power_w"]
    idle = m.calculate_total_power({"s": True}, {link: True}, {"s": 1}, {link: 0.0})
    assert full == m.profile.chassis_base_power_w + m.profile.port_active_power_w
    assert idle["total_power_w"] == m.profile.chassis_base_power_w + m.profile.port_lpi_power_w
    assert idle["ports_lpi"] == 1


def test_wakeup_and_switching_energy():
    m = NetworkPowerModel()
    prev = {"a": True, "b": False}
    new = {"a": False, "b": True}
    t = m.transition_energy_j(prev, new)
    assert t["switch_wakeups"] == 1 and t["switch_sleeps"] == 1
    assert t["transition_energy_j"] == m.profile.wakeup_energy_penalty_j  # sleeping costs nothing, waking costs
    assert m.transition_energy_j(new, new)["transition_energy_j"] == 0.0


def test_port_wakeup_only_when_switch_did_not_wake():
    m = NetworkPowerModel()
    sw = {"a": True, "b": True}
    t = m.transition_energy_j(sw, sw, {("a", "b"): False}, {("a", "b"): True})
    assert t["port_wakeups"] == 1 and t["transition_energy_j"] == m.profile.port_wakeup_energy_j
    woke = m.transition_energy_j({"a": True, "b": False}, sw, {("a", "b"): False}, {("a", "b"): True})
    assert woke["port_wakeups"] == 0  # port cost is included in the switch wake-up


def test_energy_accumulates_in_joules_and_kwh():
    m = NetworkPowerModel()
    m.step_energy_j(1000.0, 3600.0)
    assert m.total_energy_kwh == pytest.approx(1.0)


def test_env_energy_accounting_with_transitions():
    env = GreenNetworkEnv(max_steps_per_episode=10)
    env.reset(seed=1)
    _, _, _, _, i1 = env.step(7)
    assert i1["transition_energy_j"] == 0.0
    assert i1["step_energy_j"] == pytest.approx(i1["total_power_w"] * env.step_duration_s)
    _, _, _, _, i2 = env.step(0)
    assert i2["switch_toggles"] == 3 and i2["switch_wakeups"] == 0
    assert i2["total_power_w"] < i1["total_power_w"]
    _, _, _, _, i3 = env.step(7)
    assert i3["switch_wakeups"] == 3
    assert i3["transition_energy_j"] == 3 * env.power_model.profile.wakeup_energy_penalty_j
    assert i3["step_energy_j"] == pytest.approx(i3["total_power_w"] * env.step_duration_s + i3["transition_energy_j"])
    total = i1["step_energy_j"] + i2["step_energy_j"] + i3["step_energy_j"]
    assert i3["episode_energy_kwh"] == pytest.approx(total / 3.6e6)
    assert i3["episode_switch_toggles"] == 6


def test_sleeping_cores_reduce_power():
    env = GreenNetworkEnv(max_steps_per_episode=10)
    env.reset(seed=1)
    p = {}
    for a in (7, 4, 2, 0):
        env.reset(seed=1)
        p[a] = env.step(a)[4]["total_power_w"]
    assert p[7] > p[4] > p[2] > p[0]
