#!/usr/bin/env python3
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
