#!/usr/bin/env python3
"""
Green SDN Controller Application for Ryu (OpenFlow 1.3).
Integrates energy-aware traffic routing and dynamic core switch sleep scheduling
via PolicyBridge (real telemetry -> observation -> PPO inference -> safe flow/port plan).

NOTE ON HARDWARE ENERGY (ACADEMIC VALIDITY):
Real Open vSwitch / Mininet virtual switches do not support physical chassis deep sleep
or IEEE 802.3az EEE PHY-level power state transitions in software.
In this controller:
  - Port blocking and flow rerouting are REAL OpenFlow 1.3 operations (OFPPortMod, OFPFlowMod).
  - Switch deep sleep, EEE LPI, transition energy (Joules), and power consumption (Watts/kWh)
    are calculated via an analytical SIMULATION model (src.network.power_model).
Every energy metric is explicitly logged and tagged as [SIMULATED_ENERGY].

Run:
    PYTHONPATH=. ryu-manager mininet_scripts/ryu_green_controller.py
"""

import sys
import os
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet

from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent
from src.agents.safety import TelemetrySafetyFilter
from src.sdn.policy_bridge import PolicyBridge, TelemetryCollector, build_dpid_map, build_port_map


class GreenSDNRyuApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(GreenSDNRyuApp, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.topology_type = "fattree_k4"
        self.env = GreenNetworkEnv(topology_type=self.topology_type, max_steps_per_episode=1000)
        
        # Load trained PPO model if available, else raw policy with elastic shield
        model_path = os.path.join("models", "ppo_green_agent.zip")
        self.agent = GreenRLAgent(env=self.env, algo="PPO", model_path=model_path if os.path.exists(model_path) else None, use_shield=False)
        
        self.collector = TelemetryCollector(self.env.topology, link_capacity_mbps=100.0)
        self.safety = TelemetrySafetyFilter(self.env, safe_util_limit=0.85, headroom=1.25)
        self.bridge = PolicyBridge(self.env, self.agent, self.collector, min_dwell_s=30.0, safety=self.safety)
        
        self.monitor_thread = hub.spawn(self._telemetry_loop)
        self.logger.info("[*] Green SDN Ryu Controller initialized with PPO PolicyBridge.")
        self.logger.info("[!] [SIMULATED_ENERGY] Power & EEE metrics are analytical simulation models.")

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
                self.logger.info(f"[+] Switch connected: DPID {datapath.id}")
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]
                self.logger.info(f"[-] Switch disconnected: DPID {datapath.id}")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        # Install table-miss flow entry (send to controller for discovery/unhandled packets)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions, idle_timeout=0)
        self.logger.info(f"[+] Switch features configured: DPID {datapath.id}")

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout
        )
        datapath.send_msg(mod)

    def delete_flow(self, datapath, match):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=match
        )
        datapath.send_msg(mod)

    def set_port_state(self, datapath, port_no, port_down=True):
        """Sends OFPPortMod to change administrative state."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        config = ofproto.OFPPC_PORT_DOWN if port_down else 0
        mask = ofproto.OFPPC_PORT_DOWN
        req = parser.OFPPortMod(datapath=datapath, port_no=port_no, config=config, mask=mask)
        datapath.send_msg(req)

    def _telemetry_loop(self):
        """Periodic loop: poll port stats -> decide policy -> safe plan execution."""
        while True:
            hub.sleep(5)
            now = time.time()
            for dp in list(self.datapaths.values()):
                self._request_stats(dp)

            if len(self.datapaths) < len(self.env.topology.switches):
                continue  # Wait until all switches join

            decision = self.bridge.decide(now=now)
            plan = self.bridge.make_plan(decision.action)

            if not self.bridge.verify_plan(plan):
                self.logger.warning(f"[!] Plan rejected by safety verifier for action {plan.action}")
                continue

            # Safe Make-Before-Break Application:
            # 1. Bring required ports UP first
            for dpid, port_no in plan.ports_up:
                if dpid in self.datapaths:
                    self.set_port_state(self.datapaths[dpid], port_no, port_down=False)
                    self.collector.set_port_state(dpid, port_no, up=True)

            # 2. Add / Update Flow rules (reroute traffic)
            for dpid, eth_dst, out_port in plan.flow_add:
                if dpid in self.datapaths:
                    dp = self.datapaths[dpid]
                    parser = dp.ofproto_parser
                    match = parser.OFPMatch(eth_dst=eth_dst)
                    actions = [parser.OFPActionOutput(out_port)]
                    self.add_flow(dp, priority=100, match=match, actions=actions)

            # 3. Remove deprecated Flow rules
            for dpid, eth_dst in plan.flow_del:
                if dpid in self.datapaths:
                    dp = self.datapaths[dpid]
                    parser = dp.ofproto_parser
                    match = parser.OFPMatch(eth_dst=eth_dst)
                    self.delete_flow(dp, match)

            # 4. Turn ports DOWN on sleeping switches after drain
            for dpid, port_no in plan.ports_down:
                if dpid in self.datapaths:
                    self.set_port_state(self.datapaths[dpid], port_no, port_down=True)
                    self.collector.set_port_state(dpid, port_no, up=False)

            self.bridge.commit(plan, now=now)
            
            # Simulated Energy Telemetry Log
            sleeping = set(self.env.topology.core_switches) - set(plan.active_cores)
            self.logger.info(
                f"[CONTROL] Act: {plan.action} | Active Cores: {plan.active_cores} | "
                f"Sleeping Cores: {list(sleeping)} | Reason: {decision.reason} | "
                f"[SIMULATED_ENERGY] Power state emulated via OpenFlow port masking."
            )

    def _request_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto_v1_3.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        stats_dict = {}
        for stat in body:
            if stat.port_no <= ofproto_v1_3.OFPP_MAX:
                stats_dict[stat.port_no] = (stat.rx_bytes, stat.tx_bytes)
        self.collector.ingest_port_stats(dpid, stats_dict, time.time())
