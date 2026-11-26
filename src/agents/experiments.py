"""
Experiment protocol shared by scripts/run_experiments.py and scripts/run_ablation.py.

* Train/test separation: training draws traffic seeds from [0, TRAIN_SEED_LIMIT); test episodes use
  seeds TEST_SEED_BASE + i (>= TRAIN_SEED_LIMIT). Test scenarios are explicit (profile x intensity x failures),
  while training uses randomized intensity/profile (training_mode=True).
* All numbers are produced by running the simulator; nothing here is hard-coded.
"""

import hashlib
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.env.green_network_env import GreenNetworkEnv, TRAIN_SEED_LIMIT
from src.agents.evaluate import evaluate_agent, mean_std_ci

TEST_SEED_BASE = TRAIN_SEED_LIMIT

METRICS = [
    ("total_energy_kwh", "Energy (kWh)"),
    ("saving_vs_ospf_percent", "Saving vs OSPF (%)"),
    ("mean_delay_ms", "Mean delay (ms)"),
    ("p95_delay_ms", "p95 delay (ms)"),
    ("p99_delay_ms", "p99 delay (ms)"),
    ("mean_packet_loss_percent", "Packet loss (%)"),
    ("sla_violation_percent", "SLA violation (% steps)"),
    ("switch_toggles", "Switch toggles"),
    ("action_changes", "Action changes"),
]

PROFILES = ["diurnal", "datacenter_pareto", "flash_crowd"]
INTENSITIES = [0.5, 1.0, 2.0]


def scenario_kind(name: str) -> str:
    """stress = permanent overload (flash_crowd saturates even all-active OSPF); reported separately."""
    return "stress" if name.startswith("flash_crowd") else "normal"


def load_scenarios() -> Dict[str, Dict[str, Any]]:
    """Held-out load profiles (3 profiles x 3 intensities) + link-failure scenarios (k=4 fat-tree)."""
    sc: Dict[str, Dict[str, Any]] = {}
    for p in PROFILES:
        for i in INTENSITIES:
            sc[f"{p}@{i}"] = {"profile": p, "intensity": i, "failures": []}
    sc["failure:core-agg@1.0"] = {
        "profile": "diurnal", "intensity": 1.0,
        "failures": [{"link": ("a_0_0", "c_0_0"), "start": 40, "end": 80}],
    }
    sc["failure:edge-agg@1.0"] = {
        "profile": "diurnal", "intensity": 1.0,
        "failures": [{"link": ("e_0_0", "a_0_0"), "start": 40, "end": 80}],
    }
    return sc


def file_fingerprint(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()[:16]
    return {"path": path, "sha256_16": h, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))}


def run_matrix(
    env: GreenNetworkEnv,
    agent_factories: Dict[str, Callable[[GreenNetworkEnv], Any]],
    scenarios: Dict[str, Dict[str, Any]],
    seeds: Sequence[int],
    num_steps: int,
    failure_filter: bool = True,
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Runs every agent on identical (scenario, seed) traffic. Returns raw per-episode metrics:
    raw[scenario][agent] = [metrics_per_seed...]. Adds saving_vs_ospf_percent (paired per seed)
    relative to the agent named 'OSPF / ECMP'.
    """
    raw: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    agents = {name: f(env) for name, f in agent_factories.items()}
    for sc_name, sc in scenarios.items():
        raw[sc_name] = {name: [] for name in agents}
        opts = {"profile": sc["profile"], "intensity": sc["intensity"], "burst": False, "failures": sc.get("failures", [])}
        for seed in seeds:
            for name, ag in agents.items():
                s = evaluate_agent(ag, env, num_steps=num_steps, seed=seed, options=opts, include_timeseries=False)
                s["sla_violation_percent"] = 100.0 * s["sla_violation_rate"]
                raw[sc_name][name].append(s)
        if progress:
            print(f"    [done] {sc_name}", flush=True)
        # paired savings vs OSPF
        if "OSPF / ECMP" in raw[sc_name]:
            base = [r["total_energy_kwh"] for r in raw[sc_name]["OSPF / ECMP"]]
            for name in raw[sc_name]:
                for r, b in zip(raw[sc_name][name], base):
                    r["saving_vs_ospf_percent"] = 100.0 * (1.0 - r["total_energy_kwh"] / b) if b > 0 else 0.0
    return raw


def aggregate(raw: Dict[str, Dict[str, List[Dict[str, float]]]]) -> Dict[str, Any]:
    """Mean/std/95% CI across seeds per scenario, plus an 'ALL' row (per-seed mean over non-stress scenarios)."""
    out: Dict[str, Any] = {}
    all_by_agent: Dict[str, List[List[Dict[str, float]]]] = {}
    for sc, by_agent in raw.items():
        out[sc] = {}
        for name, runs in by_agent.items():
            out[sc][name] = {m: mean_std_ci([r[m] for r in runs]) for m, _ in METRICS if m in runs[0]}
            if scenario_kind(sc) != "stress":
                all_by_agent.setdefault(name, []).append(runs)
    out["ALL (mean over scenarios)"] = {}
    for name, per_sc in all_by_agent.items():
        n_seeds = len(per_sc[0])
        out["ALL (mean over scenarios)"][name] = {
            m: mean_std_ci([float(np.mean([runs[i][m] for runs in per_sc])) for i in range(n_seeds)])
            for m, _ in METRICS if m in per_sc[0][0]
        }
    return out


def fmt(stat: Dict[str, float], digits: int = 3) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f} [{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"


def markdown_table(agg_block: Dict[str, Any], metrics: Sequence[str]) -> str:
    labels = dict(METRICS)
    lines = ["| Agent | " + " | ".join(labels[m] for m in metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for name, stats in agg_block.items():
        lines.append("| " + name + " | " + " | ".join(fmt(stats[m]) if m in stats else "n/a" for m in metrics) + " |")
    return "\n".join(lines)
