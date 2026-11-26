import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from src.env.green_network_env import GreenNetworkEnv
from src.network.traffic import FlowDemand


@pytest.fixture(scope="module")
def env_k4():
    return GreenNetworkEnv(k=4, max_steps_per_episode=30)


@pytest.fixture(scope="module")
def env_k6():
    return GreenNetworkEnv(topology_type="fattree_k6", max_steps_per_episode=10)


@pytest.fixture(scope="module")
def env_wan():
    return GreenNetworkEnv(topology_type="nsfnet", max_steps_per_episode=10)


def make_flow(fid="f1", src="h_0_0_0", dst="h_3_1_1", mbps=100.0):
    return FlowDemand(flow_id=fid, src=src, dst=dst, demand_mbps=mbps, duration_steps=1)
