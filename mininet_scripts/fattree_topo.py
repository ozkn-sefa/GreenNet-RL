#!/usr/bin/env python3
"""
Custom Fat-Tree (k=4) Data Center Topology for Mininet.
Compatible with Open vSwitch (OVS) and Remote SDN Controller (Ryu).
Usage:
    sudo mn --custom fattree_topo.py --topo fattree --controller=remote,ip=127.0.0.1,port=6653 --switch ovsk
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info


class FatTreeTopo(Topo):
    """
    Fat-Tree Topo (k=4):
    - 4 Core Switches
    - 4 Pods (each with 2 Agg and 2 Edge switches)
    - 16 Hosts (2 per Edge switch)
    """

    def build(self, k=4):
        self.k = k
        core_switches = []
        agg_switches = []
        edge_switches = []
        hosts = []

        # 1. Create Core Switches
        for i in range(k // 2):
            for j in range(k // 2):
                c_name = f"c_{i}_{j}"
                core_switches.append(self.addSwitch(c_name, protocols='OpenFlow13'))

        # 2. Create Pods
        for pod in range(k):
            # Aggregation switches
            pod_agg = []
            for agg in range(k // 2):
                a_name = f"a_{pod}_{agg}"
                agg_sw = self.addSwitch(a_name, protocols='OpenFlow13')
                pod_agg.append(agg_sw)
                agg_switches.append(agg_sw)

                # Connect Agg to Core switches
                for core_idx in range(k // 2):
                    c_name = f"c_{agg}_{core_idx}"
                    self.addLink(agg_sw, c_name, bw=100, delay='1ms', loss=0)

            # Edge switches & Hosts
            for edge in range(k // 2):
                e_name = f"e_{pod}_{edge}"
                edge_sw = self.addSwitch(e_name, protocols='OpenFlow13')
                edge_switches.append(edge_sw)

                # Connect Edge to all Agg in this pod
                for agg_sw in pod_agg:
                    self.addLink(edge_sw, agg_sw, bw=100, delay='1ms', loss=0)

                # Connect Edge to 2 Hosts
                for h in range(k // 2):
                    h_name = f"h_{pod}_{edge}_{h}"
                    host = self.addHost(h_name, ip=f"10.{pod}.{edge}.{h+1}/24")
                    hosts.append(host)
                    self.addLink(host, edge_sw, bw=100, delay='1ms')


topos = {'fattree': (lambda: FatTreeTopo())}


def run_experiment():
    topo = FatTreeTopo(k=4)
    net = Mininet(topo=topo, link=TCLink, switch=OVSSwitch, controller=None)
    net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)
    net.start()
    info("[*] Fat-Tree Mininet Started. Testing pingAll...\n")
    net.pingAll()
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run_experiment()
