"""
Unit and integration tests for PPO v3 Controlled Ablation Configuration and Runner.
Verifies:
1. Four groups have correct 2x2 factorial parameters.
2. Only planned variables differ across groups.
3. Strict seed isolation (train < 50,000, validation 60001-60004, test 70001-70004).
4. Dry-run never invokes training.
5. PPO.learn cannot be invoked without explicit --execute.
6. Overwrite protection prevents overwriting existing outputs.
7. Training and validation environments share identical safety configurations (anti-drift).
8. Legacy default values and backward compatibility of train_pilot CLI.
"""

import inspect
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from scripts.ppo_v3_config import (
    ABLATION_GROUPS,
    FIXED_TEST_SEEDS,
    FIXED_VALIDATION_SEEDS,
    TRAIN_SEED_MAX_LIMIT,
    PPOV3GroupConfig,
    PPOV3Hyperparameters,
    check_existing_outputs,
    validate_group_configs,
    verify_seed_isolation,
)
from scripts.run_ppo_v3_experiments import execute_experiments, main as runner_main, run_dry_run
from scripts.train_pilot_fattree_k4 import FatTreePilotCallback, train_pilot


def test_four_groups_exist_and_have_correct_parameters():
    """Confirms groups A, B, C, D match the 2x2 factorial ablation matrix."""
    assert set(ABLATION_GROUPS.keys()) == {"A", "B", "C", "D"}

    expected = {
        "A": {"dyn": False, "pen": 0.0, "headroom": 1.20},
        "B": {"dyn": True, "pen": 0.0, "headroom": 1.20},
        "C": {"dyn": False, "pen": 2.0, "headroom": 1.20},
        "D": {"dyn": True, "pen": 2.0, "headroom": 1.20},
    }

    for gid, exp in expected.items():
        grp = ABLATION_GROUPS[gid]
        assert grp.use_dynamic_headroom is exp["dyn"], f"Group {gid} dyn headroom mismatch"
        assert grp.override_penalty_weight == pytest.approx(exp["pen"]), f"Group {gid} penalty weight mismatch"
        assert grp.safety_headroom == pytest.approx(exp["headroom"]), f"Group {gid} safety headroom mismatch"


def test_only_intended_variables_differ_between_groups():
    """Ensures identical hyperparameters, architectures, and headroom across all groups."""
    val_report = validate_group_configs()
    assert val_report["valid"] is True, f"Config validation failed: {val_report['errors']}"

    ref_hp = ABLATION_GROUPS["A"].hyperparameters.to_dict()
    for gid in ["B", "C", "D"]:
        grp = ABLATION_GROUPS[gid]
        assert grp.hyperparameters.to_dict() == ref_hp, f"Group {gid} hyperparameters diverge from Group A"
        assert grp.safety_headroom == ABLATION_GROUPS["A"].safety_headroom


def test_seed_isolation_and_zero_leakage():
    """Verifies train seeds < 50,000 and zero overlap with validation (60001-60004) or test (70001-70004)."""
    isolation = verify_seed_isolation()
    assert isolation["isolated"] is True
    assert len(isolation["violations"]) == 0

    assert FIXED_VALIDATION_SEEDS == [60001, 60002, 60003, 60004]
    assert FIXED_TEST_SEEDS == [70001, 70002, 70003, 70004]

    for s in isolation["train_seeds"]:
        assert s < TRAIN_SEED_MAX_LIMIT
        assert s not in FIXED_VALIDATION_SEEDS
        assert s not in FIXED_TEST_SEEDS


def test_dry_run_executes_without_starting_training():
    """Confirms dry-run inspection produces zero training calls and returns success."""
    groups = [ABLATION_GROUPS["A"], ABLATION_GROUPS["B"]]
    with patch("scripts.train_pilot_fattree_k4.train_pilot") as mock_train:
        success = run_dry_run(groups, seed=42)
        assert success is True
        mock_train.assert_not_called()


def test_runner_cli_defaults_to_dry_run_without_execute():
    """Confirms that running run_ppo_v3_experiments without --execute performs dry run and never trains."""
    with patch("scripts.train_pilot_fattree_k4.train_pilot") as mock_train:
        with patch.object(sys, "argv", ["run_ppo_v3_experiments.py", "--dry-run"]):
            ret = runner_main()
            assert ret == 0
            mock_train.assert_not_called()

        with patch.object(sys, "argv", ["run_ppo_v3_experiments.py", "--group", "A"]):
            ret = runner_main()
            assert ret == 0
            mock_train.assert_not_called()


def test_overwrite_protection_prevents_accidental_clobbering():
    """Confirms that existing output files cause execute_experiments to raise FileExistsError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_res = os.path.join(tmpdir, "group_A_seed42_results.json")
        with open(dummy_res, "w") as f:
            f.write("{}")

        test_group = PPOV3GroupConfig(
            group_id="A",
            description="Test group",
            use_dynamic_headroom=False,
            override_penalty_weight=0.0,
            safety_headroom=1.20,
            base_checkpoint_dir=os.path.join(tmpdir, "checkpoints"),
            base_results_dir=tmpdir,
        )

        with pytest.raises(FileExistsError, match="already exists"):
            execute_experiments([test_group], seed=42, overwrite=False)


def test_training_and_validation_environments_share_safety_config():
    """
    Confirms anti-drift mechanism: FatTreePilotCallback stores and passes
    identical safety_headroom, use_dynamic_headroom, and override_penalty_weight
    to its validation environments.
    """
    cb = FatTreePilotCallback(
        eval_freq=5000,
        checkpoint_dir="models/test_cb",
        safety_headroom=1.20,
        use_dynamic_headroom=True,
        override_penalty_weight=2.0,
    )

    assert cb.safety_headroom == 1.20
    assert cb.use_dynamic_headroom is True
    assert cb.override_penalty_weight == 2.0

    # Inspect _eval_agent_on_regime source or behavior to ensure it instantiates GreenNetworkEnv with cb attributes
    source = inspect.getsource(cb._eval_agent_on_regime)
    assert "safety_headroom=self.safety_headroom" in source
    assert "use_dynamic_headroom=self.use_dynamic_headroom" in source
    assert "override_penalty_weight=self.override_penalty_weight" in source


def test_legacy_default_values_preserved_in_train_pilot():
    """Confirms backward compatibility: default arguments to train_pilot match legacy PPO v2 exactly."""
    sig = inspect.signature(train_pilot)
    params = sig.parameters

    assert params["safety_headroom"].default == 1.25
    assert params["use_dynamic_headroom"].default is False
    assert params["override_penalty_weight"].default == 0.0
    assert params["seed"].default == 42
    assert params["total_timesteps"].default == 100000
    assert params["eval_freq"].default == 10000
