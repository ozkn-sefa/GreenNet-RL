"""
SNDlib Abilene Real Traffic Matrix Loader & Preprocessor.
Safely parses SNDlib Native 1.0 format, verifies nodes, timestamps, demands,
and provides chronological train/val/test splitting with lazy loading and validation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import OrderedDict

from src.network.traffic import FlowDemand

logger = logging.getLogger(__name__)

# Standard 12 Abilene node identifiers in SNDlib
STANDARD_ABILENE_NODES = [
    "ATLAM5",
    "ATLAng",
    "CHINng",
    "DNVRng",
    "HSTNng",
    "IPLSng",
    "KSCYng",
    "LOSAng",
    "NYCMng",
    "SNVAng",
    "STTLng",
    "WASHng",
]


class AbileneDataError(Exception):
    """Base exception for Abilene data processing errors."""
    pass


class AbileneCorruptedFileError(AbileneDataError):
    """Raised when an SNDlib file is malformed, truncated, or has missing sections."""
    pass


class AbileneNegativeTrafficError(AbileneDataError):
    """Raised when a traffic demand value is negative."""
    pass


class AbileneInvalidNodeError(AbileneDataError):
    """Raised when a demand references an invalid or unknown node."""
    pass


class AbileneMissingDataError(AbileneDataError):
    """Raised when real traffic data is missing, exhausted, or unavailable."""
    pass


class AbileneValidationError(AbileneDataError):
    """Raised when metadata validation fails (e.g., incorrect unit)."""
    pass


@dataclass
class AbileneMeta:
    """Parsed metadata from an SNDlib matrix file."""
    granularity: str
    timestamp: datetime
    timestamp_str: str
    unit: str
    origin: str


@dataclass
class AbileneDemand:
    """Individual source-destination traffic demand in Mbps."""
    demand_id: str
    src: str
    dst: str
    routing_unit: float
    demand_mbps: float
    max_path_length: str


@dataclass
class AbileneMatrix:
    """Complete traffic matrix parsed from an SNDlib file."""
    meta: AbileneMeta
    nodes: Dict[str, Tuple[float, float]]  # node_id -> (longitude, latitude)
    demands: List[AbileneDemand]
    source_file: str

    @property
    def total_demand_mbps(self) -> float:
        return sum(d.demand_mbps for d in self.demands)

    @property
    def demand_matrix_dict(self) -> Dict[Tuple[str, str], float]:
        return {(d.src, d.dst): d.demand_mbps for d in self.demands}


class AbileneDataLoader:
    """
    Production-grade loader for SNDlib Abilene dynamic traffic matrices.
    Features:
      - Native SNDlib 1.0 format parsing with strict section validation.
      - Chronological %70 train, %15 validation, %15 test splitting without data leakage.
      - Fast manifest indexing with persistent caching in data/processed/.
      - Lazy loading with in-memory LRU cache to prevent RAM exhaustion.
      - Data integrity checks (negative traffic, missing intervals, invalid nodes).
    """

    FILE_PATTERN = re.compile(r"demandMatrix-abilene-zhang-5min-(\d{8}-\d{4})\.txt$")

    def __init__(
        self,
        raw_dir: str = "data/raw/abilene/directed-abilene-zhang-5min-over-6months-ALL-native",
        processed_dir: str = "data/processed",
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        cache_size: int = 1024,
        auto_index: bool = True,
    ):
        self.raw_dir = os.path.abspath(raw_dir)
        self.processed_dir = os.path.abspath(processed_dir)
        self.split_ratios = split_ratios
        self.cache_size = cache_size

        if not (abs(sum(split_ratios) - 1.0) < 1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {split_ratios}")

        os.makedirs(self.processed_dir, exist_ok=True)
        self.manifest_file = os.path.join(self.processed_dir, "abilene_manifest.json")

        self.manifest: List[Dict[str, Union[int, str]]] = []
        self._split_indices: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        self._lru_cache: OrderedDict[int, AbileneMatrix] = OrderedDict()

        if auto_index and os.path.exists(self.raw_dir):
            self.load_or_build_manifest()

    def load_or_build_manifest(self, force_rebuild: bool = False) -> List[Dict[str, Union[int, str]]]:
        """
        Loads cached manifest if available, or scans raw_dir using fast os.scandir and builds it.
        """
        if not force_rebuild and os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0 and "split" in data[0]:
                        self.manifest = data
                        self._populate_split_indices()
                        logger.info("Loaded Abilene manifest from cache: %d files", len(self.manifest))
                        return self.manifest
            except Exception as e:
                logger.warning("Failed to load existing manifest (%s), rebuilding...", e)

        # Build fresh manifest
        self.manifest = self._build_manifest()
        self._populate_split_indices()
        self._save_manifest()
        return self.manifest

    def _build_manifest(self) -> List[Dict[str, Union[int, str]]]:
        if not os.path.exists(self.raw_dir):
            raise FileNotFoundError(f"Raw Abilene directory not found: {self.raw_dir}")

        entries: List[Tuple[str, str]] = []
        with os.scandir(self.raw_dir) as it:
            for entry in it:
                if entry.name.endswith(".txt"):
                    m = self.FILE_PATTERN.match(entry.name)
                    if m:
                        ts_str = m.group(1)
                        entries.append((ts_str, entry.name))

        # Sort strictly chronologically by timestamp
        entries.sort(key=lambda x: x[0])
        total = len(entries)
        if total == 0:
            raise AbileneMissingDataError(f"No matching SNDlib traffic matrices found in {self.raw_dir}")

        # Compute split boundaries
        n_train = int(total * self.split_ratios[0])
        n_val = int(total * self.split_ratios[1])

        manifest: List[Dict[str, Union[int, str]]] = []
        for i, (ts_str, filename) in enumerate(entries):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            manifest.append({
                "index": i,
                "filename": filename,
                "timestamp_str": ts_str,
                "split": split,
            })

        return manifest

    def _populate_split_indices(self):
        self._split_indices = {"train": [], "val": [], "test": []}
        for entry in self.manifest:
            split_name = str(entry["split"])
            if split_name in self._split_indices:
                self._split_indices[split_name].append(int(entry["index"]))

    def _save_manifest(self):
        try:
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2)
            logger.info("Saved Abilene manifest to %s (%d files)", self.manifest_file, len(self.manifest))
        except Exception as e:
            logger.error("Failed to save manifest to %s: %s", self.manifest_file, e)

    @property
    def total_count(self) -> int:
        return len(self.manifest)

    def get_split_count(self, split: str) -> int:
        return len(self._split_indices.get(split, []))

    def get_split_ranges(self) -> Dict[str, Dict[str, Union[int, str]]]:
        """Returns metadata and timestamp ranges for train, validation, and test splits."""
        ranges: Dict[str, Dict[str, Union[int, str]]] = {}
        for split, indices in self._split_indices.items():
            if not indices:
                ranges[split] = {"count": 0, "start_time": "", "end_time": ""}
                continue
            first_entry = self.manifest[indices[0]]
            last_entry = self.manifest[indices[-1]]
            ranges[split] = {
                "count": len(indices),
                "start_index": indices[0],
                "end_index": indices[-1],
                "start_time": str(first_entry["timestamp_str"]),
                "end_time": str(last_entry["timestamp_str"]),
            }
        return ranges

    # -------------------------------------------------------------------------
    # SNDlib Native 1.0 File Parser & Validator
    # -------------------------------------------------------------------------
    @staticmethod
    def parse_file(filepath: str) -> AbileneMatrix:
        """
        Parses an SNDlib Native 1.0 traffic matrix file with strict verification.
        Validates:
          - META section (granularity, timestamp, unit=MBITPERSEC)
          - NODES section (node names, coordinates)
          - DEMANDS section (src/dst in NODES, demand_value >= 0)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Traffic matrix file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f]

        if not any(lines):
            raise AbileneCorruptedFileError(f"Empty SNDlib file: {filepath}")

        sections: Dict[str, List[str]] = {}
        current_sec: Optional[str] = None
        sec_lines: List[str] = []

        for line in lines:
            if not line or line.startswith("#"):
                continue
            if line.endswith("(") and not line.startswith("("):
                parts = line.split()
                if len(parts) >= 2 and parts[-1] == "(":
                    current_sec = parts[-2]
                    sec_lines = []
                    continue
            elif line == ")" and current_sec:
                sections[current_sec] = sec_lines
                current_sec = None
                continue

            if current_sec:
                sec_lines.append(line)

        # 1. Parse META SECTION
        if "META" not in sections:
            raise AbileneCorruptedFileError(f"Missing META section in {filepath}")

        meta_dict: Dict[str, str] = {}
        for line in sections["META"]:
            if "=" in line:
                k, v = line.split("=", 1)
                meta_dict[k.strip()] = v.strip()

        required_meta = ["granularity", "time", "unit"]
        for rm in required_meta:
            if rm not in meta_dict:
                raise AbileneCorruptedFileError(f"Missing required META attribute '{rm}' in {filepath}")

        # Validate Unit
        unit = meta_dict["unit"].upper()
        if unit != "MBITPERSEC":
            raise AbileneValidationError(
                f"Unexpected traffic unit '{unit}' in {filepath}. Expected 'MBITPERSEC'."
            )

        # Validate Timestamp
        time_str = meta_dict["time"]
        try:
            timestamp = datetime.strptime(time_str, "%Y%m%d-%H%M")
        except ValueError as ve:
            raise AbileneCorruptedFileError(f"Invalid timestamp format '{time_str}' in {filepath}: {ve}")

        meta = AbileneMeta(
            granularity=meta_dict.get("granularity", "5min"),
            timestamp=timestamp,
            timestamp_str=time_str,
            unit=unit,
            origin=meta_dict.get("origin", ""),
        )

        # 2. Parse NODE SECTION
        if "NODES" not in sections:
            raise AbileneCorruptedFileError(f"Missing NODES section in {filepath}")

        nodes: Dict[str, Tuple[float, float]] = {}
        for line in sections["NODES"]:
            # Format: <node_id> [(<long>, <lat>)]
            node_m = re.match(r"^(\w+)(?:\s*\(\s*([\d\.\-]+)\s+([\d\.\-]+)\s*\))?", line)
            if node_m:
                node_id = node_m.group(1)
                lon = float(node_m.group(2)) if node_m.group(2) else 0.0
                lat = float(node_m.group(3)) if node_m.group(3) else 0.0
                nodes[node_id] = (lon, lat)

        if not nodes:
            raise AbileneCorruptedFileError(f"NODES section contains no nodes in {filepath}")

        node_set = set(nodes.keys())

        # 3. Parse DEMAND SECTION
        if "DEMANDS" not in sections:
            raise AbileneCorruptedFileError(f"Missing DEMANDS section in {filepath}")

        demands: List[AbileneDemand] = []
        for line in sections["DEMANDS"]:
            p_open = line.find("(")
            p_close = line.find(")")
            if p_open == -1 or p_close == -1 or p_close <= p_open:
                raise AbileneCorruptedFileError(f"Malformed demand entry in {filepath}: {line}")

            demand_id = line[:p_open].strip()
            src_dst = line[p_open + 1 : p_close].strip().split()
            if len(src_dst) < 2:
                raise AbileneCorruptedFileError(f"Malformed demand endpoints in {filepath}: {line}")

            src, dst = src_dst[0], src_dst[1]
            rest = line[p_close + 1 :].strip().split()
            if len(rest) < 2:
                raise AbileneCorruptedFileError(f"Malformed demand values in {filepath}: {line}")

            try:
                routing_unit = float(rest[0])
                demand_val = float(rest[1])
            except ValueError:
                raise AbileneCorruptedFileError(f"Non-numeric demand value in {filepath}: {line}")

            max_path = rest[2] if len(rest) > 2 else "UNLIMITED"

            # Check invalid node references
            if src not in node_set:
                raise AbileneInvalidNodeError(
                    f"Unknown source node '{src}' in demand '{demand_id}' in {filepath}"
                )
            if dst not in node_set:
                raise AbileneInvalidNodeError(
                    f"Unknown destination node '{dst}' in demand '{demand_id}' in {filepath}"
                )

            # Check negative traffic
            if demand_val < 0.0:
                raise AbileneNegativeTrafficError(
                    f"Negative traffic demand {demand_val} for {src}->{dst} in {filepath}"
                )

            demands.append(AbileneDemand(
                demand_id=demand_id,
                src=src,
                dst=dst,
                routing_unit=routing_unit,
                demand_mbps=demand_val,
                max_path_length=max_path,
            ))

        return AbileneMatrix(
            meta=meta,
            nodes=nodes,
            demands=demands,
            source_file=filepath,
        )

    # -------------------------------------------------------------------------
    # Lazy Loading & Cache Access
    # -------------------------------------------------------------------------
    def get_matrix_by_index(self, index: int) -> AbileneMatrix:
        """
        Lazily loads and returns the AbileneMatrix at global chronological index.
        Utilizes an LRU memory cache to maintain low RAM usage.
        """
        if index < 0 or index >= len(self.manifest):
            raise IndexError(f"Matrix index {index} out of bounds (0 <= idx < {len(self.manifest)})")

        if index in self._lru_cache:
            self._lru_cache.move_to_end(index)
            return self._lru_cache[index]

        entry = self.manifest[index]
        filepath = os.path.join(self.raw_dir, str(entry["filename"]))
        matrix = self.parse_file(filepath)

        # Cache management
        self._lru_cache[index] = matrix
        if len(self._lru_cache) > self.cache_size:
            self._lru_cache.popitem(last=False)

        return matrix

    def get_matrix_by_split(self, split: str, local_index: int) -> AbileneMatrix:
        """Loads matrix at local index within specific split ('train', 'val', or 'test')."""
        if split not in self._split_indices:
            raise ValueError(f"Unknown split '{split}'. Available: {list(self._split_indices.keys())}")
        indices = self._split_indices[split]
        if local_index < 0 or local_index >= len(indices):
            raise IndexError(f"Local index {local_index} out of bounds for split '{split}' ({len(indices)} items)")
        global_index = indices[local_index]
        return self.get_matrix_by_index(global_index)

    def clear_cache(self):
        """Clears in-memory matrix cache."""
        self._lru_cache.clear()

    # -------------------------------------------------------------------------
    # Missing Intervals Detection
    # -------------------------------------------------------------------------
    def check_missing_intervals(self, expected_delta_min: int = 5) -> List[Dict[str, Union[str, int]]]:
        """
        Scans all manifest timestamps and identifies measurement gaps exceeding expected_delta_min.
        Returns a list of detected gap events.
        """
        if len(self.manifest) < 2:
            return []

        expected = timedelta(minutes=expected_delta_min)
        gaps: List[Dict[str, Union[str, int]]] = []

        prev_entry = self.manifest[0]
        prev_dt = datetime.strptime(str(prev_entry["timestamp_str"]), "%Y%m%d-%H%M")

        for idx in range(1, len(self.manifest)):
            curr_entry = self.manifest[idx]
            curr_dt = datetime.strptime(str(curr_entry["timestamp_str"]), "%Y%m%d-%H%M")
            diff = curr_dt - prev_dt
            if diff > expected:
                gaps.append({
                    "prev_index": int(prev_entry["index"]),
                    "next_index": int(curr_entry["index"]),
                    "prev_time": str(prev_entry["timestamp_str"]),
                    "next_time": str(curr_entry["timestamp_str"]),
                    "gap_duration_minutes": int(diff.total_seconds() // 60),
                })
            prev_entry = curr_entry
            prev_dt = curr_dt

        return gaps

    def validate_continuity(self, start_idx: int, end_idx: int, expected_delta_min: int = 5) -> bool:
        """Verifies if an index range has no missing 5-minute intervals."""
        if start_idx < 0 or end_idx >= len(self.manifest) or start_idx >= end_idx:
            raise ValueError(f"Invalid range [{start_idx}, {end_idx}]")

        expected = timedelta(minutes=expected_delta_min)
        prev_dt = datetime.strptime(str(self.manifest[start_idx]["timestamp_str"]), "%Y%m%d-%H%M")
        for i in range(start_idx + 1, end_idx + 1):
            curr_dt = datetime.strptime(str(self.manifest[i]["timestamp_str"]), "%Y%m%d-%H%M")
            if curr_dt - prev_dt != expected:
                return False
            prev_dt = curr_dt
        return True

    # -------------------------------------------------------------------------
    # Topology Node Mapping & FlowDemand Converter
    # -------------------------------------------------------------------------
    @staticmethod
    def map_node_to_topology(node_name: str, topology_type: str = "abilene") -> str:
        """
        Maps an Abilene node name (e.g. 'ATLAng') to the corresponding host/node identifier
        in the target topology.
        """
        topo = topology_type.lower()
        if topo in ["abilene", "abilenetopology"]:
            return f"h_{node_name}"
        elif "fattree" in topo or topo in ["k4", "k6"]:
            # Deterministic mapping of 12 Abilene nodes into FatTree k=4 hosts (16 hosts available)
            fattree_hosts = [
                "h_0_0_0", "h_0_0_1", "h_0_1_0", "h_0_1_1",
                "h_1_0_0", "h_1_0_1", "h_1_1_0", "h_1_1_1",
                "h_2_0_0", "h_2_0_1", "h_2_1_0", "h_2_1_1",
            ]
            node_idx = STANDARD_ABILENE_NODES.index(node_name) if node_name in STANDARD_ABILENE_NODES else 0
            return fattree_hosts[node_idx % len(fattree_hosts)]
        elif "nsfnet" in topo or "wan" in topo:
            # Map Abilene nodes to NSFNET-WAN 14 hosts
            nsfnet_map = {
                "STTLng": "h_sea",
                "SNVAng": "h_sfo",
                "LOSAng": "h_san",
                "DNVRng": "h_bld",
                "KSCYng": "h_linc",
                "HSTNng": "h_hou",
                "CHINng": "h_urb",
                "IPLSng": "h_ann",
                "ATLAng": "h_atl",
                "ATLAM5": "h_scl",
                "WASHng": "h_wdc",
                "NYCMng": "h_prn",
            }
            return nsfnet_map.get(node_name, f"h_{node_name}")
        else:
            return f"h_{node_name}"

    def to_flow_demands(
        self,
        matrix: AbileneMatrix,
        topology_type: str = "abilene",
        min_demand_threshold_mbps: float = 0.01,
    ) -> List[FlowDemand]:
        """
        Converts an AbileneMatrix into a list of FlowDemand objects suitable for GreenNetworkEnv.
        Preserves original demand values in Mbps without artificial transformation.
        """
        flow_demands: List[FlowDemand] = []
        for idx, d in enumerate(matrix.demands):
            # Omit microscopic or zero flows below threshold
            if d.demand_mbps < min_demand_threshold_mbps:
                continue
            src_host = self.map_node_to_topology(d.src, topology_type)
            dst_host = self.map_node_to_topology(d.dst, topology_type)
            if src_host == dst_host:
                continue

            flow = FlowDemand(
                flow_id=f"real_{matrix.meta.timestamp_str}_{idx}",
                src=src_host,
                dst=dst_host,
                demand_mbps=round(float(d.demand_mbps), 4),
                duration_steps=1,
                is_burst=False,
            )
            flow_demands.append(flow)

        return flow_demands


class RealTrafficSource:
    """
    Stateful real traffic provider for GreenNetworkEnv.
    Yields real traffic matrices step-by-step from a specified chronological split.
    Prevents silent fallback to synthetic data if requested matrix is unavailable.
    """

    def __init__(
        self,
        loader: AbileneDataLoader,
        split: str = "train",
        topology_type: str = "abilene",
        start_step: int = 0,
        loop: bool = False,
    ):
        self.loader = loader
        self.split = split
        self.topology_type = topology_type
        self.split_indices = loader._split_indices.get(split, [])
        if not self.split_indices:
            raise AbileneMissingDataError(f"No traffic data available for split '{split}'")

        self.current_local_idx = start_step
        self.loop = loop
        self.active_matrix: Optional[AbileneMatrix] = None

    @property
    def total_steps(self) -> int:
        return len(self.split_indices)

    def reset(self, start_step: Optional[int] = None):
        """Resets stream pointer to specified local step."""
        if start_step is not None:
            if start_step < 0 or start_step >= self.total_steps:
                raise IndexError(f"Start step {start_step} out of bounds for split '{self.split}'")
            self.current_local_idx = start_step
        else:
            self.current_local_idx = 0
        self.active_matrix = None

    def get_step_demands(self) -> List[FlowDemand]:
        """
        Returns FlowDemands for the current step and advances step pointer.
        Raises AbileneMissingDataError when sequence is exhausted and loop=False.
        """
        if self.current_local_idx >= self.total_steps:
            if self.loop:
                self.current_local_idx = 0
            else:
                raise AbileneMissingDataError(
                    f"Real traffic split '{self.split}' exhausted at step {self.current_local_idx} "
                    f"of {self.total_steps}. Silent synthetic fallback is forbidden."
                )

        self.active_matrix = self.loader.get_matrix_by_split(self.split, self.current_local_idx)
        self.current_local_idx += 1
        return self.loader.to_flow_demands(self.active_matrix, topology_type=self.topology_type)

    def get_active_info(self) -> Dict[str, Union[str, int, float]]:
        """Returns metadata about the active step matrix for telemetry & observability."""
        if self.active_matrix is None:
            return {"traffic_source": "real", "split": self.split, "step_loaded": False}
        return {
            "traffic_source": "real",
            "split": self.split,
            "timestamp": self.active_matrix.meta.timestamp_str,
            "unit": self.active_matrix.meta.unit,
            "total_demand_mbps": round(self.active_matrix.total_demand_mbps, 2),
            "flow_count": len(self.active_matrix.demands),
            "step_index": self.current_local_idx - 1,
        }
