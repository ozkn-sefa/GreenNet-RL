#!/usr/bin/env python3
"""
PPO v3 Controlled Ablation Experiment Runner.
GreenNet-RL Project - Fat-Tree k=4 Controlled Ablation Infrastructure.

SAFETY LOCK:
By default, this script runs in DRY-RUN mode only.
Actual training can ONLY be triggered with the explicit `--execute` flag.
Never runs training automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.ppo_v3_config import (
    ABLATION_GROUPS,
    PPOV3GroupConfig,
    check_existing_outputs,
    get_git_commit_hash,
    validate_group_configs,
    verify_seed_isolation,
)
from scripts.train_pilot_fattree_k4 import train_pilot


def format_table_row(cols: List[str], widths: List[int]) -> str:
    return " | ".join(col.ljust(width) for col, width in zip(cols, widths))


def run_dry_run(groups: List[PPOV3GroupConfig], seed: int = 42) -> bool:
    """Performs a comprehensive dry-run inspection without training."""
    git_hash = get_git_commit_hash()
    print("=" * 110)
    print("GREENNET-RL: PPO v3 CONTROLLED ABLATION EXPERIMENT RUNNER (DRY-RUN)")
    print(f"Git HEAD Commit : {git_hash}")
    print(f"Base Seed       : {seed}")
    print("=" * 110)

    # 1. Validation of Group Configurations
    validation = validate_group_configs()
    print("\n[1] CONFIGURATION VALIDATION")
    if validation["valid"]:
        print("  ✓ All 4 ablation groups strictly adhere to the 2x2 factorial matrix.")
        print("  ✓ Hyperparameters strictly match PPO v2 reference baseline.")
        print("  ✓ Safety headroom held constant at 1.20 across all groups.")
    else:
        print("  ✗ Validation ERRORS encountered:")
        for err in validation["errors"]:
            print(f"    - {err}")
        return False

    # 2. Seed Isolation Verification
    seed_check = verify_seed_isolation(groups)
    print("\n[2] SEED ISOLATION & ZERO-LEAKAGE VERIFICATION")
    if seed_check["isolated"]:
        print(f"  ✓ Training Seed(s)     : {seed_check['train_seeds']} (< 50,000)")
        print(f"  ✓ Fixed Validation Seeds: {seed_check['validation_seeds']} (Range: 60001-60004)")
        print(f"  ✓ Fixed Test Seeds      : {seed_check['test_seeds']} (Range: 70001-70004)")
        print("  ✓ Zero data leakage confirmed across training, validation, and test sets.")
    else:
        print("  ✗ Seed isolation violations:")
        for v in seed_check["violations"]:
            print(f"    - {v}")
        return False

    # 3. Experiment Groups Table
    print("\n[3] EXPERIMENTAL GROUPS & HYPERPARAMETERS")
    headers = ["Group", "Dynamic Headroom", "Override Penalty W", "Safety Headroom", "Timesteps", "Eval Freq"]
    widths = [8, 18, 20, 16, 12, 10]
    sep = "-+-".join("-" * w for w in widths)
    print(format_table_row(headers, widths))
    print(sep)

    for g in groups:
        row = [
            g.group_id,
            "Enabled (True)" if g.use_dynamic_headroom else "Disabled (False)",
            f"{g.override_penalty_weight:.1f}",
            f"{g.safety_headroom:.2f}",
            str(g.hyperparameters.total_timesteps),
            str(g.hyperparameters.eval_freq),
        ]
        print(format_table_row(row, widths))

    # 4. Storage & Output Paths Isolation
    print("\n[4] STORAGE DIRECTORIES & RESULTS PATHS")
    has_existing = False
    for g in groups:
        ckpt_dir = g.get_checkpoint_dir(seed=seed)
        res_path = g.get_results_path(seed=seed)
        existing = check_existing_outputs(g, seed=seed)
        status = "EXISTS (Overwrite Protection active)" if existing else "Clean (Ready for write)"
        if existing:
            has_existing = True
        print(f"  Group {g.group_id} ({g.experiment_id}):")
        print(f"    Checkpoint Dir : {ckpt_dir}")
        print(f"    Results JSON   : {res_path}")
        print(f"    Status         : {status}")

    # 5. Dry-Run Summary Confirmation
    print("\n" + "=" * 110)
    print("DRY-RUN COMPLETE: NO MODEL TRAINING INITIATED. ZERO TRAINING STEPS EXECUTED.")
    print("To execute an experiment, you must explicitly supply the '--execute' flag.")
    print("=" * 110)
    return True


def execute_experiments(
    groups: List[PPOV3GroupConfig],
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    """Executes the designated ablation experiments. Strictly protected."""
    # Ensure validation passes
    validation = validate_group_configs()
    if not validation["valid"]:
        raise ValueError(f"Cannot execute experiments due to configuration errors: {validation['errors']}")

    seed_check = verify_seed_isolation(groups)
    if not seed_check["isolated"]:
        raise ValueError(f"Cannot execute experiments due to seed violations: {seed_check['violations']}")

    for g in groups:
        existing = check_existing_outputs(g, seed=seed)
        if existing and not overwrite:
            raise FileExistsError(
                f"Output destination already exists for Group {g.group_id} (Seed {seed}): {existing}. "
                "Use --overwrite to allow replacing."
            )

    for g in groups:
        print("\n" + "#" * 90)
        print(f"EXECUTING EXPERIMENT: {g.experiment_id} (Seed {seed})")
        print(f"Dynamic Headroom: {g.use_dynamic_headroom} | Override Penalty W: {g.override_penalty_weight}")
        print("#" * 90)

        train_pilot(
            total_timesteps=g.hyperparameters.total_timesteps,
            eval_freq=g.hyperparameters.eval_freq,
            checkpoint_dir=g.get_checkpoint_dir(seed=seed),
            results_path=g.get_results_path(seed=seed),
            norm_reward=g.hyperparameters.norm_reward,
            clip_reward=g.hyperparameters.clip_reward,
            seed=seed,
            safety_headroom=g.safety_headroom,
            use_dynamic_headroom=g.use_dynamic_headroom,
            override_penalty_weight=g.override_penalty_weight,
            experiment_id=g.experiment_id,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="GreenNet-RL PPO v3 Controlled Ablation Experiment Runner")
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        choices=["A", "B", "C", "D"],
        help="Target ablation group (default: all groups A, B, C, D)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (default: True)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Explicitly enable actual training execution",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting existing checkpoint/result files",
    )
    args = parser.parse_args()

    if args.group:
        target_groups = [ABLATION_GROUPS[args.group.upper()]]
    else:
        target_groups = [ABLATION_GROUPS[k] for k in ["A", "B", "C", "D"]]

    # Strict Safety Lock: Execution only if --execute is explicitly passed
    if args.execute:
        execute_experiments(target_groups, seed=args.seed, overwrite=args.overwrite)
    else:
        success = run_dry_run(target_groups, seed=args.seed)
        if not success:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
