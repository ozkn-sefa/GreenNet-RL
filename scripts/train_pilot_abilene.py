"""
GreenNet-RL: 100,000-Step Controlled PPO Pilot Training on Real SNDlib Abilene Traffic.
Configuration:
  - Device: CPU
  - Policy: MlpPolicy [128, 128]
  - Seed: 42
  - Hyperparameters: lr=3e-4, n_steps=1024, batch_size=256, gamma=0.99, gae_lambda=0.95, clip_range=0.2
Rules & Invariants:
  - Strict Train Split (33,667 matrices, %70) for training with chronological sequence.
  - Validation Split (7,214 matrices, %15) strictly for checkpoint evaluation.
  - Test Split (7,215 matrices, %15) NEVER touched.
  - Real traffic source verified; NO synthetic fallback allowed.
  - 10,000-step checkpointing with resumable state saving.
  - Fixed-scenario validation comparison against OSPF, Static, and Threshold baselines.
  - Safety-first best checkpoint selection (QoS/SLA compliance + energy optimization).
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.green_network_env import GreenNetworkEnv
from src.data.abilene_loader import AbileneDataLoader, RealTrafficSource, AbileneMissingDataError
from src.agents.baselines import OSPFBaselineAgent, StaticHeuristicAgent, ThresholdLoadAgent


class AbilenePilotValidationCallback(BaseCallback):
    """
    Evaluates model and baselines every eval_freq steps on the strict validation split.
    Saves resumable checkpoints and records comprehensive training/validation metrics.
    """

    def __init__(
        self,
        eval_freq: int = 10000,
        val_steps: int = 300,
        checkpoint_dir: str = "models/pilot_abilene_100k",
        loader: Optional[AbileneDataLoader] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.val_steps = val_steps
        self.checkpoint_dir = checkpoint_dir
        self.loader = loader or AbileneDataLoader()
        self.last_eval_step = 0
        self.training_history: List[Dict[str, Any]] = []
        self.validation_records: List[Dict[str, Any]] = []
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if evaluation is due
        current_step = int(self.num_timesteps)
        if current_step - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = current_step
            self._run_evaluation_and_checkpoint(current_step)
        return True

    def _on_rollout_end(self) -> None:
        """Capture training loss and rollout telemetry."""
        logger = self.model.logger
        if logger is not None and hasattr(logger, "name_to_value"):
            data = dict(logger.name_to_value)
            entry = {
                "num_timesteps": int(self.num_timesteps),
                "fps": float(data.get("time/fps", 0.0)),
                "loss": float(data.get("train/loss", 0.0)),
                "policy_loss": float(data.get("train/policy_gradient_loss", 0.0)),
                "value_loss": float(data.get("train/value_loss", 0.0)),
                "entropy_loss": float(data.get("train/entropy_loss", 0.0)),
                "approx_kl": float(data.get("train/approx_kl", 0.0)),
                "explained_variance": float(data.get("train/explained_variance", 0.0)),
                "clip_fraction": float(data.get("train/clip_fraction", 0.0)),
                "mean_reward": float(data.get("rollout/ep_rew_mean", 0.0)),
                "mean_ep_length": float(data.get("rollout/ep_len_mean", 0.0)),
            }
            self.training_history.append(entry)

    def _evaluate_agent_on_val(self, agent: Any, val_steps: int) -> Dict[str, float]:
        """Evaluates an agent on identical validation sequence without state contamination."""
        val_env = GreenNetworkEnv(
            topology_type="abilene",
            max_steps_per_episode=val_steps + 10,
            training_mode=False,
            reward_scale=1.0,
            traffic_source="real",
            real_traffic_loader=self.loader,
            real_traffic_split="val",
            real_traffic_loop=False,
        )

        obs, info = val_env.reset(options={"step": 0})
        total_energy_kwh = 0.0
        sla_violations = 0
        delays = []
        losses = []
        toggles = 0
        steps = 0

        for _ in range(val_steps):
            if hasattr(agent, "predict"):
                action = agent.predict(obs)
                if isinstance(action, tuple):
                    action = int(action[0])
                else:
                    action = int(action)
            elif callable(agent):
                action = int(agent(obs))
            else:
                action = 7

            obs, reward, terminated, truncated, step_info = val_env.step(action)
            steps += 1

            # Energy: read total_power_w directly from step_info
            power_w = float(step_info.get("total_power_w", 0.0))
            step_kwh = (power_w * val_env.step_duration_s) / (3600.0 * 1000.0)
            total_energy_kwh += step_kwh

            # SLA and QoS
            avg_delay = float(step_info.get("avg_delay_ms", 0.0))
            loss_rate = float(step_info.get("packet_loss_rate", 0.0))
            delays.append(avg_delay)
            losses.append(loss_rate)

            # SLA violation: delay exceeding SLA threshold or packet loss > sla_loss_rate
            if avg_delay > val_env.sla_delay_ms or loss_rate > val_env.sla_loss_rate:
                sla_violations += 1

            if terminated or truncated:
                break

        val_env.close()

        delays_arr = np.array(delays, dtype=float)
        p95_delay = float(np.percentile(delays_arr, 95)) if len(delays_arr) > 0 else 0.0
        p99_delay = float(np.percentile(delays_arr, 99)) if len(delays_arr) > 0 else 0.0
        avg_delay = float(np.mean(delays_arr)) if len(delays_arr) > 0 else 0.0
        mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        sla_rate = (sla_violations / steps) * 100.0 if steps > 0 else 0.0

        return {
            "steps": steps,
            "total_energy_kwh": round(total_energy_kwh, 4),
            "sla_violation_percent": round(sla_rate, 2),
            "avg_delay_ms": round(avg_delay, 3),
            "p95_delay_ms": round(p95_delay, 3),
            "p99_delay_ms": round(p99_delay, 3),
            "mean_packet_loss_rate": round(mean_loss, 5),
            "switch_toggles": int(val_env.episode_switch_toggles),
        }

    def _run_evaluation_and_checkpoint(self, timestep: int):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{timestep}.zip")
        self.model.save(checkpoint_path)
        if self.verbose > 0:
            print(f"\n[*] [Checkpoint @ {timestep:,} steps] Saved model to {checkpoint_path}")

        # Evaluate PPO checkpoint (raw unshielded policy to evaluate true learned behavior)
        class ModelWrapper:
            def __init__(self, m):
                self.m = m
            def predict(self, o):
                act, _ = self.m.predict(o, deterministic=True)
                return int(act)

        eval_model = ModelWrapper(self.model)

        # Baseline agents on identical validation sequence
        dummy_env = GreenNetworkEnv(topology_type="abilene")
        ospf_agent = OSPFBaselineAgent(env=dummy_env)
        static_agent = StaticHeuristicAgent(env=dummy_env)
        thresh_agent = ThresholdLoadAgent(env=dummy_env)

        ppo_metrics = self._evaluate_agent_on_val(eval_model, self.val_steps)
        ospf_metrics = self._evaluate_agent_on_val(ospf_agent, self.val_steps)
        static_metrics = self._evaluate_agent_on_val(static_agent, self.val_steps)
        thresh_metrics = self._evaluate_agent_on_val(thresh_agent, self.val_steps)

        # Calculate energy savings relative to OSPF
        ref_energy = ospf_metrics["total_energy_kwh"]
        ppo_savings = ((ref_energy - ppo_metrics["total_energy_kwh"]) / ref_energy * 100.0) if ref_energy > 0 else 0.0
        static_savings = ((ref_energy - static_metrics["total_energy_kwh"]) / ref_energy * 100.0) if ref_energy > 0 else 0.0
        thresh_savings = ((ref_energy - thresh_metrics["total_energy_kwh"]) / ref_energy * 100.0) if ref_energy > 0 else 0.0

        ppo_metrics["energy_savings_percent"] = round(ppo_savings, 2)
        static_metrics["energy_savings_percent"] = round(static_savings, 2)
        thresh_metrics["energy_savings_percent"] = round(thresh_savings, 2)
        ospf_metrics["energy_savings_percent"] = 0.0

        # Safety Criteria: SLA Violation <= 5.0% AND Mean Loss <= 0.015
        is_safe = bool(ppo_metrics["sla_violation_percent"] <= 5.0 and ppo_metrics["mean_packet_loss_rate"] <= 0.015)
        ppo_metrics["is_qos_safe"] = is_safe

        eval_entry = {
            "timestep": timestep,
            "checkpoint_path": checkpoint_path,
            "is_qos_safe": is_safe,
            "ppo": ppo_metrics,
            "ospf": ospf_metrics,
            "static": static_metrics,
            "threshold": thresh_metrics,
        }
        self.validation_records.append(eval_entry)

        if self.verbose > 0:
            print(f"    Validation Comparison ({self.val_steps} steps):")
            print(f"    - PPO:       Energy={ppo_metrics['total_energy_kwh']} kWh (Savings: {ppo_savings:.1f}%) | SLA Viol={ppo_metrics['sla_violation_percent']}% | Loss={ppo_metrics['mean_packet_loss_rate']:.4f} | p95 Delay={ppo_metrics['p95_delay_ms']} ms | Safe={is_safe}")
            print(f"    - OSPF:      Energy={ospf_metrics['total_energy_kwh']} kWh | SLA Viol={ospf_metrics['sla_violation_percent']}% | Loss={ospf_metrics['mean_packet_loss_rate']:.4f} | p95 Delay={ospf_metrics['p95_delay_ms']} ms")
            print(f"    - Static:    Energy={static_metrics['total_energy_kwh']} kWh (Savings: {static_savings:.1f}%) | SLA Viol={static_metrics['sla_violation_percent']}% | Loss={static_metrics['mean_packet_loss_rate']:.4f}")
            print(f"    - Threshold: Energy={thresh_metrics['total_energy_kwh']} kWh (Savings: {thresh_savings:.1f}%) | SLA Viol={thresh_metrics['sla_violation_percent']}% | Loss={thresh_metrics['mean_packet_loss_rate']:.4f}")


def run_pilot_training(
    total_timesteps: int = 100000,
    eval_freq: int = 10000,
    val_steps: int = 300,
    seed: int = 42,
    output_dir: str = "models/pilot_abilene_100k",
    results_file: str = "results/pilot_abilene_100k_results.json",
) -> Dict[str, Any]:
    print("=" * 75)
    print(" [*] GREENNET-RL: 100,000-STEP CONTROLLED PPO PILOT TRAINING INITIATED")
    print(f" [*] Target Device: CPU | Timesteps: {total_timesteps:,} | Eval Freq: {eval_freq:,}")
    print(f" [*] Topology: Abilene (12 Nodes, 15 Backbone Links, 4 Core Hubs)")
    print(f" [*] Hyperparameters: n_steps=1024, batch_size=256, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # 1. Dataset Verification & Strict Isolation Checks
    loader = AbileneDataLoader()
    split_ranges = loader.get_split_ranges()
    train_count = loader.get_split_count("train")
    val_count = loader.get_split_count("val")
    test_count = loader.get_split_count("test")

    print(f" [*] Dataset Manifest Verified:")
    print(f"     Total Matrices: {loader.total_count}")
    print(f"     Train Split:    {train_count:,} matrices (70%) | Indeks: 0 .. {train_count - 1}")
    print(f"     Val Split:      {val_count:,} matrices (15%) | Indeks: {train_count} .. {train_count + val_count - 1}")
    print(f"     Test Split:     {test_count:,} matrices (15%) | Indeks: {train_count + val_count} .. {loader.total_count - 1} [PROTECTED]")

    # Check missing intervals in train split
    all_gaps = loader.check_missing_intervals(expected_delta_min=5)
    train_gaps = [g for g in all_gaps if g["prev_index"] < split_ranges["train"]["end_index"]]
    print(f" [*] Measurement Gap Audit: {len(train_gaps)} measurement gaps (>5 min) identified in train split.")
    print(f"     Documented Train Recycling: 100,000 steps on 33,667 train matrices = {100000 / train_count:.2f} epochs of train sequence.")

    # 2. Setup Environment with Strict Train Split
    train_env = GreenNetworkEnv(
        topology_type="abilene",
        max_steps_per_episode=150,
        training_mode=True,
        reward_scale=0.05,
        traffic_source="real",
        real_traffic_loader=loader,
        real_traffic_split="train",
        real_traffic_loop=True,
    )

    obs, info = train_env.reset(seed=seed)
    assert train_env.traffic_source == "real", "FATAL: traffic_source must be 'real'"
    assert train_env.real_traffic_split == "train", "FATAL: real_traffic_split must be 'train'"
    assert train_env.real_source is not None, "FATAL: real_source is None"
    assert isinstance(train_env.real_source, RealTrafficSource), "FATAL: real_source must be RealTrafficSource"
    assert train_env.real_source.split == "train", "FATAL: RealTrafficSource must be bound to 'train'"

    # 3. Initialize PPO Model with Specified Hyperparameters on CPU
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="cpu",
    )

    # Save training configuration
    config = {
        "algo": "PPO",
        "device": "cpu",
        "policy": "MlpPolicy",
        "net_arch": [128, 128],
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "seed": seed,
        "train_split_count": train_count,
        "val_split_count": val_count,
        "test_split_count": test_count,
        "train_gaps_count": len(train_gaps),
    }
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 4. Execute Pilot Training with Validation Callback
    callback = AbilenePilotValidationCallback(
        eval_freq=eval_freq,
        val_steps=val_steps,
        checkpoint_dir=output_dir,
        loader=loader,
        verbose=1,
    )

    start_time = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_time_sec = time.perf_counter() - start_time

    # 5. Save Final Model Checkpoint
    final_model_path = os.path.join(output_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"\n[+] Final pilot model checkpoint saved to: {final_model_path}")

    # 6. Safety-First Best Checkpoint Selection
    # Rule: Filter all checkpoints that satisfy QoS/SLA constraints; among safe ones, pick highest energy savings.
    val_records = callback.validation_records
    safe_checkpoints = [r for r in val_records if r.get("is_qos_safe", False)]

    best_checkpoint = None
    best_dest_path = os.path.join(output_dir, "best_model.zip")
    if safe_checkpoints:
        # Pick checkpoint with highest energy savings among safe candidates
        best_checkpoint = max(safe_checkpoints, key=lambda r: r["ppo"]["energy_savings_percent"])
        print(f"\n[+] BEST CHECKPOINT SELECTED: Step {best_checkpoint['timestep']:,}")
        print(f"    - Safety: SLA Violations={best_checkpoint['ppo']['sla_violation_percent']}% | Loss={best_checkpoint['ppo']['mean_packet_loss_rate']:.4f}")
        print(f"    - Efficiency: Energy Savings={best_checkpoint['ppo']['energy_savings_percent']}% vs OSPF Baseline")
        # Save as best_model.zip
        best_source_path = best_checkpoint["checkpoint_path"]
        import shutil
        shutil.copyfile(best_source_path, best_dest_path)
        print(f"    - Copied best model to: {best_dest_path}")
    else:
        print("\n[!] VERIFIED SAFETY AUDIT: Zero checkpoints met strict QoS/SLA criteria (SLA <= 5%, Loss <= 1.5%).")
        print("    Following strict safety invariants: No model will be labeled as 'best_model.zip'.")
        if os.path.exists(best_dest_path):
            os.remove(best_dest_path)

    # 7. Compile Final Pilot Report Data
    report_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "training_time_sec": round(training_time_sec, 2),
        "overall_fps": round(total_timesteps / training_time_sec, 2),
        "training_history": callback.training_history,
        "validation_records": val_records,
        "best_checkpoint": best_checkpoint,
        "safe_checkpoint_count": len(safe_checkpoints),
        "total_checkpoints": len(val_records),
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Full pilot results exported to: {results_file}")

    train_env.close()
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="100k PPO Pilot on Real Abilene Traffic")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps (default 100k)")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation and checkpoint frequency")
    parser.add_argument("--val-steps", type=int, default=300, help="Steps evaluated on validation split")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    run_pilot_training(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        val_steps=args.val_steps,
        seed=args.seed,
    )
