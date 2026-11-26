"""
GreenNet-RL: Full-Scale Controlled PPO Training on Real SNDlib Abilene Traffic.
Supports:
  - Real SNDlib Abilene traffic dataset auto-discovery (48,096 total matrices)
  - Strict 70% Train split (33,667 matrices) with cyclic stream playback
  - Strict 15% Validation split (7,214 matrices) for checkpoint QoS evaluation
  - Strict 15% Test split (7,215 matrices) isolated and preserved
  - Link-level WAN energy optimization (green_mode="link_level") preserving 100% host reachability
  - CUDA GPU acceleration (RTX 3050 Laptop / CUDA 12.x / auto fallback)
  - Resumable training from latest checkpoint (--resume)
  - Periodic checkpointing to models/abilene_full_training/
  - Multi-baseline validation (OSPF all-active, Safe Link-Level Heuristic)
  - Safety-first best model selection based on SLA/QoS compliance & energy savings
  - Primary model deployment to models/ppo_green_agent_abilene.zip
"""

import argparse
import datetime
import gc
import json
import os
import re
import shutil
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
from src.agents.baselines import OSPFBaselineAgent


class FullAbileneValidationCallback(BaseCallback):
    """
    Evaluates model and baselines every eval_freq steps on the strict validation split.
    Saves resumable checkpoints and records comprehensive training/validation metrics.
    """

    def __init__(
        self,
        eval_freq: int = 10000,
        val_steps: int = 300,
        checkpoint_dir: str = "models/abilene_full_training",
        loader: Optional[AbileneDataLoader] = None,
        green_mode: str = "link_level",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.val_steps = val_steps
        self.checkpoint_dir = checkpoint_dir
        self.loader = loader or AbileneDataLoader()
        self.green_mode = green_mode
        self.last_eval_step = 0
        self.training_history: List[Dict[str, Any]] = []
        self.validation_records: List[Dict[str, Any]] = []
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
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
            green_mode=self.green_mode,
        )

        obs, info = val_env.reset(options={"step": 0})
        total_energy_kwh = 0.0
        sla_violations = 0
        delays = []
        losses = []
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

            power_w = float(step_info.get("total_power_w", 0.0))
            step_kwh = (power_w * val_env.step_duration_s) / (3600.0 * 1000.0)
            total_energy_kwh += step_kwh

            avg_delay = float(step_info.get("avg_delay_ms", 0.0))
            loss_rate = float(step_info.get("packet_loss_rate", 0.0))
            delays.append(avg_delay)
            losses.append(loss_rate)

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

        class ModelWrapper:
            def __init__(self, m):
                self.m = m
            def predict(self, o):
                act, _ = self.m.predict(o, deterministic=True)
                return int(act)

        eval_model = ModelWrapper(self.model)

        dummy_env = GreenNetworkEnv(topology_type="abilene", green_mode=self.green_mode)
        ospf_agent = OSPFBaselineAgent(env=dummy_env)

        ppo_metrics = self._evaluate_agent_on_val(eval_model, self.val_steps)
        ospf_metrics = self._evaluate_agent_on_val(ospf_agent, self.val_steps)

        # Baseline comparison
        ref_energy = ospf_metrics["total_energy_kwh"]
        ppo_savings = ((ref_energy - ppo_metrics["total_energy_kwh"]) / ref_energy * 100.0) if ref_energy > 0 else 0.0
        ppo_metrics["energy_savings_percent"] = round(ppo_savings, 2)
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
        }
        self.validation_records.append(eval_entry)

        if self.verbose > 0:
            print(f"    Validation Comparison ({self.val_steps} steps):")
            print(f"    - PPO:       Energy={ppo_metrics['total_energy_kwh']} kWh (Savings: {ppo_savings:.2f}%) | SLA Viol={ppo_metrics['sla_violation_percent']}% | Loss={ppo_metrics['mean_packet_loss_rate']:.4f} | p95 Delay={ppo_metrics['p95_delay_ms']} ms | Safe={is_safe}")
            print(f"    - OSPF:      Energy={ospf_metrics['total_energy_kwh']} kWh | SLA Viol={ospf_metrics['sla_violation_percent']}% | Loss={ospf_metrics['mean_packet_loss_rate']:.4f} | p95 Delay={ospf_metrics['p95_delay_ms']} ms")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[Tuple[str, int]]:
    """Scans checkpoint_dir for checkpoint_step_<N>.zip and returns path and step of highest step."""
    if not os.path.isdir(checkpoint_dir):
        return None
    pattern = re.compile(r"checkpoint_step_(\d+)\.zip$")
    highest_step = -1
    highest_path = None
    for fname in os.listdir(checkpoint_dir):
        m = pattern.match(fname)
        if m:
            step = int(m.group(1))
            if step > highest_step:
                highest_step = step
                highest_path = os.path.join(checkpoint_dir, fname)
    if highest_path is not None:
        return (highest_path, highest_step)
    return None


def run_full_training(
    total_timesteps: int = 100000,
    eval_freq: int = 10000,
    val_steps: int = 300,
    seed: int = 42,
    device: str = "auto",
    green_mode: str = "link_level",
    output_dir: str = "models/abilene_full_training",
    results_file: str = "results/abilene_full_training_results.json",
    primary_save_path: str = "models/ppo_green_agent_abilene.zip",
    resume: bool = False,
    batch_size: int = 256,
    n_steps: int = 1024,
    learning_rate: float = 3e-4,
) -> Dict[str, Any]:
    # Resolve compute device
    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("[!] WARNING: CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
            target_device = "cpu"
        else:
            target_device = "cuda"
    else:
        target_device = "cpu"

    device_name = torch.cuda.get_device_name(0) if target_device == "cuda" else "CPU Host"

    print("=" * 80)
    print(" [*] GREENNET-RL: FULL-SCALE RL TRAINING ON REAL SNDLIB ABILENE DATASET")
    print(f" [*] Compute Device : {target_device.upper()} ({device_name})")
    print(f" [*] Timesteps      : {total_timesteps:,} steps | Eval Freq: {eval_freq:,} steps")
    print(f" [*] Topology       : SNDlib Abilene (12 Router PoPs, 15 Backbone Links, 2500 Mbps)")
    print(f" [*] Green Mode     : {green_mode} (Preserves 100% Host Reachability)")
    print(f" [*] Hyperparameters: n_steps={n_steps}, batch_size={batch_size}, lr={learning_rate}")
    print(f" [*] Checkpoint Dir : {output_dir}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # 1. Dataset Discovery & Split Verification
    loader = AbileneDataLoader()
    split_ranges = loader.get_split_ranges()
    train_count = loader.get_split_count("train")
    val_count = loader.get_split_count("val")
    test_count = loader.get_split_count("test")

    print(f" [*] Real Traffic Dataset:")
    print(f"     Source Directory: {loader.raw_dir}")
    print(f"     Total Matrices  : {loader.total_count:,} files (6 months at 5-min intervals)")
    print(f"     Train Split     : {train_count:,} matrices (70%) | Chronological [Index 0 .. {train_count - 1}]")
    print(f"     Val Split       : {val_count:,} matrices (15%) | Chronological [Index {train_count} .. {train_count + val_count - 1}]")
    print(f"     Test Split      : {test_count:,} matrices (15%) | PROTECTED [Index {train_count + val_count} .. {loader.total_count - 1}]")

    epochs = total_timesteps / max(1, train_count)
    print(f"     Training Epochs : {epochs:.2f}x passes over the 33,667 real traffic sequence")

    # 2. Setup Training Environment
    train_env = GreenNetworkEnv(
        topology_type="abilene",
        max_steps_per_episode=300,
        training_mode=True,
        reward_scale=0.05,
        traffic_source="real",
        real_traffic_loader=loader,
        real_traffic_split="train",
        real_traffic_loop=True,
        green_mode=green_mode,
    )

    obs, info = train_env.reset(seed=seed)
    assert train_env.traffic_source == "real", "FATAL: traffic_source must be 'real'"
    assert train_env.real_traffic_split == "train", "FATAL: real_traffic_split must be 'train'"

    # 3. Model Initialization / Resume from Checkpoint
    latest_ckpt = find_latest_checkpoint(output_dir) if resume else None
    remaining_steps = total_timesteps

    if resume and latest_ckpt is not None:
        ckpt_path, ckpt_step = latest_ckpt
        print(f"\n[+] RESUMING FROM CHECKPOINT: {ckpt_path} (Completed {ckpt_step:,} steps)")
        model = PPO.load(ckpt_path, env=train_env, device=target_device)
        remaining_steps = max(0, total_timesteps - ckpt_step)
        print(f"[*] Remaining Timesteps to Train: {remaining_steps:,}")
    else:
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            device=target_device,
        )

    # Save training configuration
    config = {
        "algo": "PPO",
        "device": target_device,
        "device_name": device_name,
        "policy": "MlpPolicy",
        "net_arch": [128, 128],
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "val_steps": val_steps,
        "seed": seed,
        "topology": "abilene",
        "green_mode": green_mode,
        "train_matrices": train_count,
        "val_matrices": val_count,
        "test_matrices": test_count,
        "resumed_from": latest_ckpt[0] if latest_ckpt else None,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 4. Execute Full Training with Validation Callback
    callback = FullAbileneValidationCallback(
        eval_freq=eval_freq,
        val_steps=val_steps,
        checkpoint_dir=output_dir,
        loader=loader,
        green_mode=green_mode,
        verbose=1,
    )

    start_time = time.perf_counter()
    if remaining_steps > 0:
        model.learn(total_timesteps=remaining_steps, callback=callback, reset_num_timesteps=(not resume))
    training_time_sec = time.perf_counter() - start_time

    # 5. Save Final Model
    final_model_path = os.path.join(output_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"\n[+] Final trained model saved to: {final_model_path}")

    # 6. Safety-First Best Checkpoint Selection
    val_records = callback.validation_records
    safe_checkpoints = [r for r in val_records if r.get("is_qos_safe", False)]

    best_checkpoint = None
    best_dest_path = os.path.join(output_dir, "best_model.zip")
    if safe_checkpoints:
        best_checkpoint = max(safe_checkpoints, key=lambda r: r["ppo"]["energy_savings_percent"])
        print(f"\n[+] BEST CHECKPOINT SELECTED: Step {best_checkpoint['timestep']:,}")
        print(f"    - Safety     : SLA Violations={best_checkpoint['ppo']['sla_violation_percent']}% | Loss={best_checkpoint['ppo']['mean_packet_loss_rate']:.4f}")
        print(f"    - Efficiency : Energy Savings={best_checkpoint['ppo']['energy_savings_percent']}% vs OSPF Baseline")
        shutil.copyfile(best_checkpoint["checkpoint_path"], best_dest_path)
        print(f"    - Copied best model to: {best_dest_path}")
        chosen_for_deployment = best_dest_path
    else:
        print("\n[*] No checkpoint strictly met SLA <= 5% and Loss <= 1.5%. Saving final model as active candidate.")
        shutil.copyfile(final_model_path, best_dest_path)
        chosen_for_deployment = final_model_path

    # Deploy primary model for live simulation & web dashboard
    if primary_save_path:
        os.makedirs(os.path.dirname(primary_save_path), exist_ok=True)
        shutil.copyfile(chosen_for_deployment, primary_save_path)
        print(f"[+] Primary deployment model updated: {primary_save_path}")

    # 7. Compile and Export Comprehensive Results
    report_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "training_time_sec": round(training_time_sec, 2),
        "overall_fps": round(remaining_steps / max(0.01, training_time_sec), 2),
        "training_history": callback.training_history,
        "validation_records": val_records,
        "best_checkpoint": best_checkpoint,
        "safe_checkpoint_count": len(safe_checkpoints),
        "total_checkpoints": len(val_records),
        "final_model_path": final_model_path,
        "best_model_path": best_dest_path,
        "primary_model_path": primary_save_path,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Complete training results exported to: {results_file}")

    train_env.close()
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-Scale PPO Training on Real Abilene Traffic")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps (default: 100,000)")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation and checkpoint frequency (default: 10,000)")
    parser.add_argument("--val-steps", type=int, default=300, help="Steps evaluated on validation split (default: 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device (default: auto)")
    parser.add_argument("--green-mode", type=str, default="link_level", choices=["link_level", "switch_sleep"], help="Green routing mode (default: link_level)")
    parser.add_argument("--output-dir", type=str, default="models/abilene_full_training", help="Checkpoint directory")
    parser.add_argument("--results-file", type=str, default="results/abilene_full_training_results.json", help="Results file path")
    parser.add_argument("--primary-save-path", type=str, default="models/ppo_green_agent_abilene.zip", help="Primary model export path")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint if available")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO mini-batch size (default: 256)")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout buffer steps (default: 1024)")
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate (default: 3e-4)")
    args = parser.parse_args()

    run_full_training(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        val_steps=args.val_steps,
        seed=args.seed,
        device=args.device,
        green_mode=args.green_mode,
        output_dir=args.output_dir,
        results_file=args.results_file,
        primary_save_path=args.primary_save_path,
        resume=args.resume,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        learning_rate=args.lr,
    )
