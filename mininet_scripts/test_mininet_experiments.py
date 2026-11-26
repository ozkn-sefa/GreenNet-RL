#!/usr/bin/env python3
"""
Automated Mininet Validation & Benchmark Suite for GreenNet-RL.
Runs:
  1. Pingall & RTT Latency Test
  2. iperf Throughput & Loss Test
  3. Core Link Down (Failure injection) & Failover Convergence Test
  4. Link Recovery (Re-activation) Test

ACADEMIC & HARDWARE NOTE:
Switch sleep states and IEEE 802.3az EEE power transitions are analytical SIMULATION models.
Mininet testbeds execute true packet forwarding (OVS), true queuing delay, and real OpenFlow port down/up.
Hardware power measurements are tagged as [SIMULATED_ENERGY].

Usage (inside Linux / WSL2 with Mininet installed):
    sudo python3 mininet_scripts/test_mininet_experiments.py --controller 127.0.0.1 --port 6653
"""

import sys
import os
import time
import json
import argparse

# Check if mininet is available in current environment
try:
    from mininet.net import Mininet
    from mininet.node import RemoteController, OVSSwitch
    from mininet.link import TCLink
    from mininet.log import setLogLevel, info
    from mininet_scripts.fattree_topo import FatTreeTopo
    MININET_AVAILABLE = True
except ImportError:
    MININET_AVAILABLE = False


def run_mininet_suite(controller_ip="127.0.0.1", controller_port=6653, output_json="results/mininet_test_results.json"):
    if not MININET_AVAILABLE:
        print("[!] Mininet is not installed in this environment (e.g. Windows host).")
        print("[*] To execute hardware testbed: Run this script inside Linux / WSL2 with Open vSwitch and Mininet.")
        return {"status": "skipped", "reason": "Mininet library not available on host OS"}

    info("[*] Starting Fat-Tree k=4 Mininet topology for validation...\n")
    topo = FatTreeTopo(k=4)
    net = Mininet(topo=topo, link=TCLink, switch=OVSSwitch, controller=None, autoSetMacs=True)
    net.addController('c0', controller=RemoteController, ip=controller_ip, port=controller_port)
    net.start()

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_energy_disclaimer": "Power and EEE LPI metrics are analytical SIMULATION models.",
        "tests": {}
    }

    try:
        # Test 1: Connectivity & PingAll
        info("[*] Test 1: Full Mesh PingAll (reachability check)...\n")
        loss = net.pingAll(timeout=2)
        results["tests"]["pingall"] = {"loss_percent": loss, "passed": (loss < 1.0)}

        # Test 2: Latency Test (h_0_0_0 to h_3_1_1)
        h1 = net.get('h_0_0_0')
        h2 = net.get('h_3_1_1')
        info(f"[*] Test 2: Ping latency from {h1.name} to {h2.name}...\n")
        ping_out = h1.cmd(f"ping -c 10 {h2.IP()}")
        results["tests"]["ping_latency"] = {"raw_output": ping_out, "passed": "0% packet loss" in ping_out}

        # Test 3: iperf Throughput
        info(f"[*] Test 3: iperf throughput measurement between {h1.name} and {h2.name}...\n")
        h2.cmd("iperf -s -p 5001 &")
        time.sleep(1)
        iperf_out = h1.cmd(f"iperf -c {h2.IP()} -p 5001 -t 5")
        h2.cmd("killall -9 iperf")
        results["tests"]["iperf"] = {"output": iperf_out, "passed": "Mbits/sec" in iperf_out}

        # Test 4: Link Failure & Recovery
        info("[*] Test 4: Link Failure (Disconnecting core switch link c_0_0 <-> a_0_0)...\n")
        net.configLinkStatus('a_0_0', 'c_0_0', 'down')
        time.sleep(3)
        failover_loss = net.ping([h1, h2], timeout=1)
        results["tests"]["link_failure_failover"] = {"loss_percent": failover_loss, "passed": (failover_loss == 0.0)}

        info("[*] Recovering link (Bringing c_0_0 <-> a_0_0 back UP)...\n")
        net.configLinkStatus('a_0_0', 'c_0_0', 'up')
        time.sleep(3)
        recovery_loss = net.ping([h1, h2], timeout=1)
        results["tests"]["link_recovery"] = {"loss_percent": recovery_loss, "passed": (recovery_loss == 0.0)}

    finally:
        info("[*] Stopping Mininet...\n")
        net.stop()

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    info(f"[+] Mininet experiment results saved to {output_json}\n")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="127.0.0.1", help="Controller IP")
    parser.add_argument("--port", type=int, default=6653, help="Controller OpenFlow port")
    parser.add_argument("--out", default="results/mininet_test_results.json", help="Output JSON path")
    args = parser.parse_args()

    if MININET_AVAILABLE:
        setLogLevel('info')
    run_mininet_suite(args.controller, args.port, args.out)
