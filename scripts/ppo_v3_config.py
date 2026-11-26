"""
PPO v3 Controlled Ablation Experiment Configuration Module.
GreenNet-RL Project - Fat-Tree k=4 PPO Ablation Infrastructure.

This module defines the 4 independent experimental groups for PPO v3 ablation:
- Group A: Fixed headroom (1.20), no override penalty (weight=0.0)
- Group B: Dynamic headroom (active), no override penalty (weight=0.0)
- Group C: Fixed headroom (1.20), override penalty (weight=2.0)
- Group D: Dynamic headroom (active), override penalty (weight=2.0)

Safety headroom is strictly held at 1.20 across all groups.
PPO v2 hyperparameters are strictly conserved.
Seed isolation guarantees zero train/val/test data leakage.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Seed boundary constants (strictly isolated ranges)
TRAIN_SEED_MAX_LIMIT: int = 50_000
FIXED_VALIDATION_SEEDS: List[int] = [60001, 60002, 60003, 60004]
FIXED_TEST_SEEDS: List[int] = [70001, 70002, 70003, 70004]

# Fixed Validation Regimes definition (matching train_pilot_fattree_k4.py)
VAL_REGIMES_SPEC: List[Dict[str, Any]] = [
    {"name": "Low", "intensity": 0.50, "profile": "diurnal", "burst": False, "seed": 60001, "steps": 50},
    {"name": "Medium", "intensity": 1.10, "profile": "diurnal", "burst": False, "seed": 60002, "steps": 50},
    {"name": "High", "intensity": 1.85, "profile": "datacenter_pareto", "burst": False, "seed": 60003, "steps": 50},
    {"name": "Burst", "intensity": 2.50, "profile": "flash_crowd", "burst": True, "seed": 60004, "steps": 50},
]


def get_git_commit_hash() -> str:
    """Returns the current git HEAD commit hash, or 'unknown' if git fails."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class PPOV3Hyperparameters:
    """Fixed PPO v2 hyperparameters strictly preserved for PPO v3 ablation."""

    policy: str = "MlpPolicy"
    net_arch: tuple[int, int] = (128, 128)
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    n_epochs: int = 10
    total_timesteps: int = 100_000
    eval_freq: int = 10_000
    norm_reward: bool = True
    clip_reward: Optional[float] = None
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "net_arch": list(self.net_arch),
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "n_epochs": self.n_epochs,
            "total_timesteps": self.total_timesteps,
            "eval_freq": self.eval_freq,
            "norm_reward": self.norm_reward,
            "clip_reward": self.clip_reward,
            "device": self.device,
        }


@dataclass(frozen=True)
class PPOV3GroupConfig:
    """Configuration for an independent ablation experiment group."""

    group_id: str
    description: str
    use_dynamic_headroom: bool
    override_penalty_weight: float
    safety_headroom: float = 1.20
    training_seeds: tuple[int, ...] = (42,)
    base_checkpoint_dir: str = "models/ppo_v3_ablation"
    base_results_dir: str = "results/ppo_v3_ablation"
    hyperparameters: PPOV3Hyperparameters = field(default_factory=PPOV3Hyperparameters)

    @property
    def experiment_id(self) -> str:
        return f"PPO_V3_GROUP_{self.group_id.upper()}"

    def get_checkpoint_dir(self, seed: int = 42) -> str:
        """Returns the isolated checkpoint directory for this group and seed."""
        return os.path.normpath(os.path.join(self.base_checkpoint_dir, f"group_{self.group_id.upper()}_seed{seed}"))

    def get_results_path(self, seed: int = 42) -> str:
        """Returns the isolated JSON results filepath for this group and seed."""
        return os.path.normpath(os.path.join(self.base_results_dir, f"group_{self.group_id.upper()}_seed{seed}_results.json"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "group_id": self.group_id,
            "description": self.description,
            "use_dynamic_headroom": self.use_dynamic_headroom,
            "override_penalty_weight": self.override_penalty_weight,
            "safety_headroom": self.safety_headroom,
            "training_seeds": list(self.training_seeds),
            "hyperparameters": self.hyperparameters.to_dict(),
        }


# The 4 Ablation Groups defined according to specifications:
# A: use_dynamic_headroom=False, override_penalty_weight=0.0
# B: use_dynamic_headroom=True, override_penalty_weight=0.0
# C: use_dynamic_headroom=False, override_penalty_weight=2.0
# D: use_dynamic_headroom=True, override_penalty_weight=2.0
ABLATION_GROUPS: Dict[str, PPOV3GroupConfig] = {
    "A": PPOV3GroupConfig(
        group_id="A",
        description="Baseline SafeFilter: static headroom 1.20, no override penalty (w=0.0)",
        use_dynamic_headroom=False,
        override_penalty_weight=0.0,
        safety_headroom=1.20,
    ),
    "B": PPOV3GroupConfig(
        group_id="B",
        description="Dynamic Headroom Only: trend-aware headroom, no override penalty (w=0.0)",
        use_dynamic_headroom=True,
        override_penalty_weight=0.0,
        safety_headroom=1.20,
    ),
    "C": PPOV3GroupConfig(
        group_id="C",
        description="Override Penalty Only: static headroom 1.20, with override penalty (w=2.0)",
        use_dynamic_headroom=False,
        override_penalty_weight=2.0,
        safety_headroom=1.20,
    ),
    "D": PPOV3GroupConfig(
        group_id="D",
        description="Full PPO v3: dynamic headroom + override penalty (w=2.0)",
        use_dynamic_headroom=True,
        override_penalty_weight=2.0,
        safety_headroom=1.20,
    ),
}


def verify_seed_isolation(group_configs: Optional[List[PPOV3GroupConfig]] = None) -> Dict[str, Any]:
    """
    Verifies that all training seeds are strictly isolated:
    1. training_seeds < TRAIN_SEED_MAX_LIMIT (50,000)
    2. Zero overlap with FIXED_VALIDATION_SEEDS (60001-60004)
    3. Zero overlap with FIXED_TEST_SEEDS (70001-70004)
    """
    configs = group_configs or list(ABLATION_GROUPS.values())
    all_train_seeds = set()
    violations = []

    for cfg in configs:
        for s in cfg.training_seeds:
            all_train_seeds.add(s)
            if s >= TRAIN_SEED_MAX_LIMIT:
                violations.append(f"Training seed {s} exceeds limit {TRAIN_SEED_MAX_LIMIT}")
            if s in FIXED_VALIDATION_SEEDS:
                violations.append(f"Training seed {s} leaks into validation seeds {FIXED_VALIDATION_SEEDS}")
            if s in FIXED_TEST_SEEDS:
                violations.append(f"Training seed {s} leaks into test seeds {FIXED_TEST_SEEDS}")

    val_test_overlap = set(FIXED_VALIDATION_SEEDS).intersection(set(FIXED_TEST_SEEDS))
    if val_test_overlap:
        violations.append(f"Validation and test seeds overlap: {val_test_overlap}")

    is_isolated = len(violations) == 0
    return {
        "isolated": is_isolated,
        "violations": violations,
        "train_seeds": sorted(list(all_train_seeds)),
        "validation_seeds": FIXED_VALIDATION_SEEDS,
        "test_seeds": FIXED_TEST_SEEDS,
    }


def validate_group_configs() -> Dict[str, Any]:
    """
    Validates that:
    1. Exactly 4 groups exist (A, B, C, D).
    2. All groups have safety_headroom == 1.20.
    3. Hyperparameters are identical across all groups.
    4. Independent variables strictly match the 2x2 ablation design.
    """
    expected_keys = {"A", "B", "C", "D"}
    actual_keys = set(ABLATION_GROUPS.keys())
    assert actual_keys == expected_keys, f"Group keys mismatch: {actual_keys} vs {expected_keys}"

    ref_hp = ABLATION_GROUPS["A"].hyperparameters.to_dict()
    errors = []

    expected_specs = {
        "A": {"dyn": False, "pen": 0.0},
        "B": {"dyn": True, "pen": 0.0},
        "C": {"dyn": False, "pen": 2.0},
        "D": {"dyn": True, "pen": 2.0},
    }

    for gid, cfg in ABLATION_GROUPS.items():
        if cfg.safety_headroom != 1.20:
            errors.append(f"Group {gid} safety_headroom is {cfg.safety_headroom}, expected 1.20")
        if cfg.hyperparameters.to_dict() != ref_hp:
            errors.append(f"Group {gid} hyperparameters do not match reference PPO v2")
        exp = expected_specs[gid]
        if cfg.use_dynamic_headroom != exp["dyn"]:
            errors.append(f"Group {gid} use_dynamic_headroom is {cfg.use_dynamic_headroom}, expected {exp['dyn']}")
        if cfg.override_penalty_weight != exp["pen"]:
            errors.append(f"Group {gid} override_penalty_weight is {cfg.override_penalty_weight}, expected {exp['pen']}")

    seed_check = verify_seed_isolation()
    if not seed_check["isolated"]:
        errors.extend(seed_check["violations"])

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "groups": {k: v.to_dict() for k, v in ABLATION_GROUPS.items()},
    }


def check_existing_outputs(group: PPOV3GroupConfig, seed: int = 42) -> List[str]:
    """Returns list of already existing output paths to prevent accidental overwrite."""
    existing = []
    res_path = group.get_results_path(seed=seed)
    if os.path.exists(res_path):
        existing.append(res_path)

    ckpt_dir = group.get_checkpoint_dir(seed=seed)
    if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
        existing.append(ckpt_dir)

    return existing
