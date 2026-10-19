"""
Unit and integration tests for SNDlib Abilene traffic matrix loader,
chronological splitting, lazy loading, error validation, Abilene topology,
and GreenNetworkEnv real traffic integration.
"""

from datetime import datetime
import os
import tempfile
import pytest
import numpy as np

from src.data.abilene_loader import (
    AbileneDataLoader,
    AbileneDataError,
    AbileneCorruptedFileError,
    AbileneNegativeTrafficError,
    AbileneInvalidNodeError,
    AbileneMissingDataError,
    AbileneValidationError,
    RealTrafficSource,
    STANDARD_ABILENE_NODES,
)
from src.network.topology import AbileneTopology, TopologyFactory
from src.env.green_network_env import GreenNetworkEnv


RAW_ABILENE_DIR = os.path.abspath("data/raw/abilene/directed-abilene-zhang-5min-over-6months-ALL-native")


@pytest.fixture(scope="session")
def abilene_loader():
    """Shared AbileneDataLoader instance with manifest initialized."""
    loader = AbileneDataLoader(raw_dir=RAW_ABILENE_DIR)
    return loader


# =============================================================================
# 1. Parsing & Metadata Tests
# =============================================================================
def test_parse_valid_sndlib_file(abilene_loader):
    """Verifies that a valid SNDlib Native 1.0 file is parsed accurately."""
    sample_path = os.path.join(RAW_ABILENE_DIR, "demandMatrix-abilene-zhang-5min-20040301-0000.txt")
    matrix = abilene_loader.parse_file(sample_path)

    # Validate META
    assert matrix.meta.granularity == "5min"
    assert matrix.meta.unit == "MBITPERSEC"
    assert matrix.meta.timestamp_str == "20040301-0000"
    assert matrix.meta.timestamp == datetime(2004, 3, 1, 0, 0)

    # Validate NODES
    assert len(matrix.nodes) == 12
    for node_id in STANDARD_ABILENE_NODES:
        assert node_id in matrix.nodes
        lon, lat = matrix.nodes[node_id]
        assert isinstance(lon, float) and isinstance(lat, float)

    # Validate DEMANDS
    assert len(matrix.demands) == 132  # 12 * 11
    assert matrix.total_demand_mbps > 0.0
    for d in matrix.demands:
        assert d.src in matrix.nodes
        assert d.dst in matrix.nodes
        assert d.src != d.dst
        assert d.demand_mbps >= 0.0


def test_metadata_validation_rejects_non_mbps():
    """Verifies that SNDlib files with non-Mbps units raise AbileneValidationError."""
    content = """?SNDlib native format; type: network; version: 1.0
META (
  granularity = 5min
  time = 20040301-0000
  unit = KBITPERSEC
)
NODES (
  ATLAng ( -85.50 34.50 )
  CHINng ( -87.61 41.83 )
)
DEMANDS (
  ATLAng_CHINng ( ATLAng CHINng ) 1 10.0 UNLIMITED
)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        with pytest.raises(AbileneValidationError, match="Expected 'MBITPERSEC'"):
            AbileneDataLoader.parse_file(tmp_path)
    finally:
        os.remove(tmp_path)


def test_metadata_validation_rejects_invalid_timestamp():
    """Verifies that malformed timestamp format raises AbileneCorruptedFileError."""
    content = """?SNDlib native format; type: network; version: 1.0
META (
  granularity = 5min
  time = INVALID-TIMESTAMP
  unit = MBITPERSEC
)
NODES (
  ATLAng ( -85.50 34.50 )
)
DEMANDS ()
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        with pytest.raises(AbileneCorruptedFileError, match="Invalid timestamp format"):
            AbileneDataLoader.parse_file(tmp_path)
    finally:
        os.remove(tmp_path)


# =============================================================================
# 2. Corrupted File & Integrity Validation Tests
# =============================================================================
def test_corrupted_file_handling():
    """Verifies that empty files or missing sections raise AbileneCorruptedFileError."""
    # 1. Empty file
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("   \n")
        empty_path = f.name

    try:
        with pytest.raises(AbileneCorruptedFileError, match="Empty SNDlib file"):
            AbileneDataLoader.parse_file(empty_path)
    finally:
        os.remove(empty_path)

    # 2. Missing DEMANDS section
    content_no_demands = """
META (
  granularity = 5min
  time = 20040301-0000
  unit = MBITPERSEC
)
NODES (
  ATLAng ( -85.50 34.50 )
)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content_no_demands)
        no_dem_path = f.name

    try:
        with pytest.raises(AbileneCorruptedFileError, match="Missing DEMANDS section"):
            AbileneDataLoader.parse_file(no_dem_path)
    finally:
        os.remove(no_dem_path)


def test_negative_traffic_rejection():
    """Verifies that negative traffic demands raise AbileneNegativeTrafficError."""
    content_negative = """META (
  granularity = 5min
  time = 20040301-0000
  unit = MBITPERSEC
)
NODES (
  ATLAng ( -85.50 34.50 )
  CHINng ( -87.61 41.83 )
)
DEMANDS (
  ATLAng_CHINng ( ATLAng CHINng ) 1 -5.250 UNLIMITED
)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content_negative)
        neg_path = f.name

    try:
        with pytest.raises(AbileneNegativeTrafficError, match="Negative traffic demand"):
            AbileneDataLoader.parse_file(neg_path)
    finally:
        os.remove(neg_path)


def test_invalid_node_reference_rejection():
    """Verifies that demand references to unknown nodes raise AbileneInvalidNodeError."""
    content_invalid_node = """META (
  granularity = 5min
  time = 20040301-0000
  unit = MBITPERSEC
)
NODES (
  ATLAng ( -85.50 34.50 )
  CHINng ( -87.61 41.83 )
)
DEMANDS (
  ATLAng_GHOST ( ATLAng GHOSTng ) 1 12.0 UNLIMITED
)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content_invalid_node)
        inv_path = f.name

    try:
        with pytest.raises(AbileneInvalidNodeError, match="Unknown destination node"):
            AbileneDataLoader.parse_file(inv_path)
    finally:
        os.remove(inv_path)


# =============================================================================
# 3. Chronological Splits & Data Leakage Tests
# =============================================================================
def test_chronological_splits_and_no_leakage(abilene_loader):
    """
    Verifies:
      - Total matrix count equals 48,096.
      - Exact 70% train (33,667), 15% val (7,214), 15% test (7,215).
      - Strict chronological ordering without temporal leakage.
    """
    total = abilene_loader.total_count
    assert total == 48096

    n_train = abilene_loader.get_split_count("train")
    n_val = abilene_loader.get_split_count("val")
    n_test = abilene_loader.get_split_count("test")

    assert n_train == 33667
    assert n_val == 7214
    assert n_test == 7215
    assert n_train + n_val + n_test == 48096

    ranges = abilene_loader.get_split_ranges()
    train_end = ranges["train"]["end_time"]
    val_start = ranges["val"]["start_time"]
    val_end = ranges["val"]["end_time"]
    test_start = ranges["test"]["start_time"]

    # Strict chronological sequence: train < val < test
    assert train_end < val_start, f"Data leakage: train_end ({train_end}) >= val_start ({val_start})"
    assert val_end < test_start, f"Data leakage: val_end ({val_end}) >= test_start ({test_start})"

    # Verify split continuity indices
    assert ranges["train"]["start_index"] == 0
    assert ranges["train"]["end_index"] == 33666
    assert ranges["val"]["start_index"] == 33667
    assert ranges["val"]["end_index"] == 40880
    assert ranges["test"]["start_index"] == 40881
    assert ranges["test"]["end_index"] == 48095


# =============================================================================
# 4. Missing Intervals Detection Tests
# =============================================================================
def test_missing_intervals_detection(abilene_loader):
    """
    Verifies that check_missing_intervals correctly flags historical measurement gaps (> 5 min).
    """
    gaps = abilene_loader.check_missing_intervals(expected_delta_min=5)
    assert len(gaps) == 4, f"Expected 4 historical gaps in Yin Zhang dataset, got {len(gaps)}"

    # First gap in mid March 2004
    assert gaps[0]["prev_time"] == "20040314-2355"
    assert gaps[0]["next_time"] == "20040402-0000"

    # Validate continuity helper
    assert abilene_loader.validate_continuity(0, 10) is True
    # Range spanning the first gap (index around 4031) must return False
    gap_idx = gaps[0]["prev_index"]
    assert abilene_loader.validate_continuity(gap_idx, gap_idx + 1) is False


# =============================================================================
# 5. Lazy Loading & Memory Caching Tests
# =============================================================================
def test_lazy_loading_and_lru_cache(abilene_loader):
    """Verifies on-demand file loading and LRU cache operation."""
    abilene_loader.clear_cache()
    assert len(abilene_loader._lru_cache) == 0

    # Load matrix 0
    m0 = abilene_loader.get_matrix_by_index(0)
    assert len(abilene_loader._lru_cache) == 1
    assert 0 in abilene_loader._lru_cache

    # Retrieve again (should hit cache)
    m0_cached = abilene_loader.get_matrix_by_index(0)
    assert m0 is m0_cached

    # Load a few more
    for i in range(1, 5):
        abilene_loader.get_matrix_by_index(i)
    assert len(abilene_loader._lru_cache) == 5


# =============================================================================
# 6. Abilene Topology Tests
# =============================================================================
def test_abilene_topology_structure():
    """Verifies that AbileneTopology complies with NetworkX and GreenNet-RL standards."""
    topo = TopologyFactory.create("abilene", link_capacity_mbps=2500.0)
    assert isinstance(topo, AbileneTopology)
    assert topo.name == "Abilene-WAN"
    assert len(topo.switches) == 12
    assert len(topo.hosts) == 12
    assert len(topo.core_switches) == 4
    assert len(topo.agg_switches) == 4
    assert len(topo.edge_switches) == 4

    # 15 backbone links + 12 host-switch links = 27 edges
    assert len(topo.get_links()) == 27

    # Check candidate paths precomputed between hosts
    paths = topo.get_candidate_paths("h_ATLAM5", "h_SNVAng")
    assert len(paths) > 0
    for path in paths:
        assert path[0] == "h_ATLAM5"
        assert path[-1] == "h_SNVAng"
        # Path must transit through connected switches
        assert path[1] == "ATLAM5"


def test_topology_node_mapping():
    """Verifies mapping of Abilene nodes to FatTree, NSFNET, and Abilene topologies."""
    # Abilene mapping
    assert AbileneDataLoader.map_node_to_topology("ATLAng", "abilene") == "h_ATLAng"

    # FatTree mapping (maps to deterministic host in FatTree)
    ft_host = AbileneDataLoader.map_node_to_topology("ATLAng", "fattree_k4")
    assert ft_host.startswith("h_")

    # NSFNET mapping
    assert AbileneDataLoader.map_node_to_topology("STTLng", "nsfnet") == "h_sea"
    assert AbileneDataLoader.map_node_to_topology("CHINng", "nsfnet") == "h_urb"


# =============================================================================
# 7. RealTrafficSource & No Silent Fallback Tests
# =============================================================================
def test_real_traffic_source_yields_demands(abilene_loader):
    """Verifies RealTrafficSource returns FlowDemands in Mbps."""
    source = RealTrafficSource(abilene_loader, split="test", topology_type="abilene")
    assert source.total_steps == 7215

    demands = source.get_step_demands()
    assert len(demands) > 0
    info = source.get_active_info()
    assert info["traffic_source"] == "real"
    assert info["split"] == "test"
    assert info["unit"] == "MBITPERSEC"
    assert info["total_demand_mbps"] > 0.0

    # Demand values must not be modified from Mbps
    for flow in demands:
        assert flow.demand_mbps > 0.0
        assert flow.src.startswith("h_")
        assert flow.dst.startswith("h_")


def test_real_traffic_source_no_silent_fallback(abilene_loader):
    """
    Critical Rule Test: Verifies that when real traffic data is exhausted,
    the loader raises AbileneMissingDataError instead of silently falling back to synthetic.
    """
    # Create small source near end of test split
    source = RealTrafficSource(abilene_loader, split="test", start_step=7214, loop=False)
    # First step should succeed (step 7214)
    demands = source.get_step_demands()
    assert len(demands) > 0

    # Next step exceeds available test split data
    with pytest.raises(AbileneMissingDataError, match="Silent synthetic fallback is forbidden"):
        source.get_step_demands()


# =============================================================================
# 8. GreenNetworkEnv Real Traffic Integration Tests
# =============================================================================
def test_green_network_env_real_traffic_mode(abilene_loader):
    """
    Verifies GreenNetworkEnv works seamlessly with real SNDlib Abilene traffic:
      - Reset and step return valid observations and rewards.
      - Info dictionary clearly labels 'real' traffic and telemetry.
      - Observation space shape and dtype remain completely unchanged.
    """
    env = GreenNetworkEnv(
        topology_type="abilene",
        traffic_source="real",
        real_traffic_loader=abilene_loader,
        real_traffic_split="test",
        max_steps_per_episode=5,
    )

    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert info["traffic_source"] == "real"
    assert info["real_split"] == "test"

    for step_num in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert info["traffic_source"] == "real"
        assert "real_timestamp" in info
        assert info["total_demand_mbps"] > 0.0
        assert info["avg_delay_ms"] >= 0.0


def test_green_network_env_split_isolation(abilene_loader):
    """
    Rule Test: Verifies that training mode uses the train split,
    and evaluation uses val/test split without contamination.
    """
    # Train mode
    env_train = GreenNetworkEnv(
        topology_type="abilene",
        traffic_source="real",
        real_traffic_loader=abilene_loader,
        real_traffic_split="train",
    )
    _, info_train = env_train.reset()
    obs, _, _, _, step_info_train = env_train.step(6)
    train_ts = step_info_train["real_timestamp"]

    # Test mode
    env_test = GreenNetworkEnv(
        topology_type="abilene",
        traffic_source="real",
        real_traffic_loader=abilene_loader,
        real_traffic_split="test",
    )
    _, info_test = env_test.reset()
    obs, _, _, _, step_info_test = env_test.step(6)
    test_ts = step_info_test["real_timestamp"]

    assert step_info_train["real_split"] == "train"
    assert step_info_test["real_split"] == "test"
    assert train_ts < test_ts
    assert train_ts.startswith("20040301")
    assert test_ts.startswith("20040815")
