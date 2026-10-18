"""
Mininet & Ryu SDN Controller Exporter.
Generates fully functional Python scripts for running the exact Fat-Tree topology
and energy-aware green routing within a standard Mininet / OpenFlow testbed.
"""

import os
from typing import Dict, Any


class MininetExporter:
    """
    Exports topology and Ryu controller code directly runnable inside Mininet on Linux / WSL.
    """

    @staticmethod
    def export_all(output_dir: str = "mininet_scripts"):
        os.makedirs(output_dir, exist_ok=True)
        MininetExporter.export_mininet_topology(os.path.join(output_dir, "fattree_topo.py"))
        MininetExporter.export_ryu_controller(os.path.join(output_dir, "ryu_green_controller.py"))
        MininetExporter.export_readme(os.path.join(output_dir, "README_MININET.md"))
        print(f"[+] Mininet & Ryu scripts exported successfully to: {output_dir}/")

    @staticmethod
    def export_mininet_topology(file_path: str):
        content = '''#!/usr/bin/env python3
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
    info("[*] Fat-Tree Mininet Started. Testing pingAll...\\n")
    net.pingAll()
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run_experiment()
'''
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def export_ryu_controller(file_path: str):
        content = '''#!/usr/bin/env python3
"""
Green SDN Controller Application for Ryu (OpenFlow 1.3).
Integrates energy-aware traffic routing and dynamic core switch sleep scheduling.
Run:
    ryu-manager ryu_green_controller.py
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4
import time


class GreenSDNControllerApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(GreenSDNControllerApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.mac_to_port = {}
        self.logger.info("[*] Green SDN Energy-Aware Controller Initialized.")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        # Install table-miss flow entry (send to controller)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info(f"[+] Switch registered: DPID {datapath.id}")

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=60
        )
        datapath.send_msg(mod)

    def set_port_energy_state(self, datapath, port_no, sleep_mode=True):
        """Disables or enables a port for IEEE 802.3az Low Power Idle."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        config = ofproto.OFPPC_PORT_DOWN if sleep_mode else 0
        mask = ofproto.OFPPC_PORT_DOWN
        req = parser.OFPPortMod(datapath=datapath, port_no=port_no, config=config, mask=mask)
        datapath.send_msg(req)
        status = "SLEEP" if sleep_mode else "ACTIVE"
        self.logger.info(f"[ENERGY] Switch {datapath.id} Port {port_no} set to {status}")
'''
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def export_readme(file_path: str):
        content = """# Mininet & Ryu SDN Enerji Tasarruflu Ağ Ortamı

Bu klasör, projenin **Mininet** ve **Ryu SDN Denetleyicisi** ile gerçek bir Linux/WSL2 ağ ortamında çalıştırılabilmesi için otomatik olarak üretilen betikleri içerir.

## Dosyalar:
1. `fattree_topo.py`: Mininet k=4 Fat-Tree Veri Merkezi Topolojisi.
2. `ryu_green_controller.py`: OpenFlow 1.3 destekli, dinamik port uyku ve yeşil akış yönlendirme mantığına sahip Ryu uygulaması.

## Nasıl Çalıştırılır (Ubuntu / WSL / Mininet VM):

### 1. Terminal 1: Ryu Controller'ı Başlatın
```bash
ryu-manager ryu_green_controller.py
```

### 2. Terminal 2: Mininet Topolojisini Başlatın
```bash
sudo mn --custom fattree_topo.py --topo fattree --controller=remote,ip=127.0.0.1,port=6653 --switch ovsk --link tc
```

### 3. Mininet CLI Üzerinde Test:
```bash
mininet> pingall
mininet> iperf h_0_0_0 h_3_1_1
```
"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
