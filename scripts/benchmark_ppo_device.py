"""
GreenNet-RL: CPU vs GPU (CUDA) PPO Performance Benchmark on Real Abilene Traffic Data.
Measures:
  - Total Training Time (s)
  - Training FPS (timesteps / sec)
  - Process RAM & Peak RAM (MB) using Windows API (Zero external dependencies)
  - Peak GPU VRAM (MB) using torch.cuda
  - Final & Per-Epoch Loss Metrics (policy_loss, value_loss, loss, explained_variance, entropy_loss)
  - Real Traffic Source & Split Verification
Preserves train/val/test isolation and saves benchmark models to models/benchmark/.
"""

import argparse
import ctypes
from ctypes import wintypes
import gc
import json
import os
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

# Set UTF-8 stdout encoding on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env.green_network_env import GreenNetworkEnv
from src.agents.rl_agent import GreenRLAgent
from src.data.abilene_loader import AbileneDataLoader, RealTrafficSource


# Windows Process Memory Measurement via Win32 API (Standard Library)
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_process_memory_mb() -> Tuple[float, float]:
    """Returns (current_ram_mb, peak_ram_mb) for the current process."""
    if sys.platform.startswith("win"):
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                current_mb = pmc.WorkingSetSize / (1024 * 1024)
                peak_mb = pmc.PeakWorkingSetSize / (1024 * 1024)
                return current_mb, peak_mb
        except Exception:
            pass
    return 0.0, 0.0


class TrainingMetricsCallback(BaseCallback):
    """Captures training metrics at each rollout/update iteration."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.iterations: List[Dict[str, float]] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
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
            }
            self.iterations.append(entry)


def run_benchmark_for_device(
    device: str,
    timesteps: int = 10000,
    seed: int = 42,
    save_dir: str = "models/benchmark",
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f" [>] BENCHMARK RUN: Device = {device.upper()} | Timesteps = {timesteps} | Seed = {seed}")
    print("=" * 70)

    # Force garbage collection & CUDA cache reset before run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 1. Verification of real Abilene data source & strict train split isolation
    loader = AbileneDataLoader()
    train_count = loader.get_split_count("train")
    val_count = loader.get_split_count("val")
    test_count = loader.get_split_count("test")
    split_ranges = loader.get_split_ranges()

    print(f" [*] Abilene Data Verified: Total={loader.total_count}")
    print(f" [*] Split Counts: Train={train_count} (70%), Val={val_count} (15%), Test={test_count} (15%)")
    print(f" [*] Split Ranges: Train={split_ranges['train']}, Val={split_ranges['val']}, Test={split_ranges['test']}")

    # Create Real Environment
    env = GreenNetworkEnv(
        topology_type="abilene",
        max_steps_per_episode=150,
        training_mode=True,
        reward_scale=0.05,
        traffic_source="real",
        real_traffic_loader=loader,
        real_traffic_split="train",
        real_traffic_loop=True,
    )

    # Verify that env is strictly bound to real train traffic
    obs, info = env.reset(seed=seed)
    assert env.traffic_source == "real", "Verification failed: traffic_source is not 'real'!"
    assert env.real_traffic_split == "train", "Verification failed: real_traffic_split is not 'train'!"
    assert env.real_source is not None, "Verification failed: real_source is None!"
    assert isinstance(env.real_source, RealTrafficSource), "Verification failed: real_source is not RealTrafficSource!"
    assert env.real_source.split == "train", "Verification failed: RealTrafficSource split is not 'train'!"

    # Verify initial real demand
    step_info = env.real_source.get_active_info()
    print(f" [*] Verified Real Traffic Feed: Matrix Timestamp={step_info.get('timestamp')}, Unit={step_info.get('unit')}, Demand={step_info.get('total_demand_mbps')} Mbps")

    # 2. Agent Initialization with Target Device
    agent = GreenRLAgent(
        env=env,
        algo="PPO",
        device=device,
        seed=seed,
    )

    # Verify actual PyTorch device placement
    actual_device = str(agent.model.device)
    print(f" [*] Confirmed PyTorch Model Device: {actual_device}")

    # Track memory before training
    ram_before, _ = get_process_memory_mb()
    vram_before = (torch.cuda.memory_allocated() / (1024 * 1024)) if (torch.cuda.is_available() and "cuda" in device) else 0.0

    # 3. Training Execution with Timers & Callback
    callback = TrainingMetricsCallback()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ppo_abilene_{device}_{timesteps}.zip")

    start_time = time.perf_counter()
    agent.model.learn(total_timesteps=timesteps, callback=callback)
    elapsed_time = time.perf_counter() - start_time

    # Save model weights to benchmark dir (never overwrites main models)
    agent.model.save(save_path)
    print(f"[+] Model checkpoint safely saved to: {save_path}")

    # Memory after training
    ram_after, peak_ram = get_process_memory_mb()
    peak_vram = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if (torch.cuda.is_available() and "cuda" in device) else 0.0
    vram_after = (torch.cuda.memory_allocated() / (1024 * 1024)) if (torch.cuda.is_available() and "cuda" in device) else 0.0

    # Calculate overall FPS
    fps = timesteps / elapsed_time

    # Extract final loss metrics from callback or logger
    final_metrics = callback.iterations[-1] if callback.iterations else {}
    logger_data = dict(agent.model.logger.name_to_value) if agent.model.logger else {}

    result = {
        "device": device,
        "actual_device": actual_device,
        "timesteps": timesteps,
        "seed": seed,
        "elapsed_time_sec": round(elapsed_time, 2),
        "fps": round(fps, 2),
        "ram_before_mb": round(ram_before, 2),
        "ram_after_mb": round(ram_after, 2),
        "peak_ram_mb": round(peak_ram, 2),
        "ram_delta_mb": round(ram_after - ram_before, 2),
        "vram_before_mb": round(vram_before, 2),
        "vram_after_mb": round(vram_after, 2),
        "peak_vram_mb": round(peak_vram, 2),
        "model_save_path": save_path,
        "final_loss": round(float(logger_data.get("train/loss", final_metrics.get("loss", 0.0))), 6),
        "final_policy_loss": round(float(logger_data.get("train/policy_gradient_loss", final_metrics.get("policy_loss", 0.0))), 6),
        "final_value_loss": round(float(logger_data.get("train/value_loss", final_metrics.get("value_loss", 0.0))), 6),
        "final_entropy_loss": round(float(logger_data.get("train/entropy_loss", final_metrics.get("entropy_loss", 0.0))), 6),
        "final_approx_kl": round(float(logger_data.get("train/approx_kl", final_metrics.get("approx_kl", 0.0))), 6),
        "final_explained_variance": round(float(logger_data.get("train/explained_variance", final_metrics.get("explained_variance", 0.0))), 6),
        "history": callback.iterations,
    }

    print("-" * 50)
    print(f" [*] Benchmark Result ({device.upper()}):")
    print(f"     Elapsed Time: {result['elapsed_time_sec']} s")
    print(f"     Average FPS:  {result['fps']}")
    print(f"     RAM RSS:      {result['ram_after_mb']} MB (Peak: {result['peak_ram_mb']} MB)")
    print(f"     Peak VRAM:    {result['peak_vram_mb']} MB")
    print(f"     Loss:         {result['final_loss']} (Policy: {result['final_policy_loss']}, Value: {result['final_value_loss']})")
    print(f"     Exp Variance: {result['final_explained_variance']}")
    print("-" * 50)

    # Clean up environment & model
    env.close()
    del agent
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="CPU vs GPU Benchmark for PPO on Abilene Data")
    parser.add_argument("--timesteps", type=int, default=10000, help="Timesteps per benchmark run")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="both", choices=["cpu", "cuda", "both"], help="Device to benchmark")
    parser.add_argument("--output", type=str, default="results/cpu_gpu_benchmark_results.json", help="Path to save output JSON")
    args = parser.parse_args()

    results = {}

    # Hardware Info
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print("=" * 70)
    print(" [*] HARDWARE & PLATFORM SPECIFICATIONS")
    print(f" [*] PyTorch Version:     {torch.__version__}")
    print(f" [*] CUDA Available:      {cuda_available}")
    print(f" [*] GPU Device Name:     {device_name}")
    print(f" [*] CPU Count (Logical): {os.cpu_count()}")
    print("=" * 70)

    devices_to_test = ["cpu", "cuda"] if args.device == "both" else [args.device]

    for dev in devices_to_test:
        if dev == "cuda" and not cuda_available:
            print(" [!] CUDA is not available on this system. Skipping CUDA run.")
            continue
        results[dev] = run_benchmark_for_device(
            device=dev,
            timesteps=args.timesteps,
            seed=args.seed,
        )

    # Determine Winner
    if "cpu" in results and "cuda" in results:
        cpu_time = results["cpu"]["elapsed_time_sec"]
        cuda_time = results["cuda"]["elapsed_time_sec"]
        cpu_fps = results["cpu"]["fps"]
        cuda_fps = results["cuda"]["fps"]

        speedup_fps = cuda_fps / cpu_fps if cpu_fps > 0 else 0.0

        if cuda_fps > cpu_fps:
            winner = "CUDA (GPU)"
            margin = f"{round((cuda_fps - cpu_fps) / cpu_fps * 100, 2)}% faster (FPS)"
        else:
            winner = "CPU"
            margin = f"{round((cpu_fps - cuda_fps) / cuda_fps * 100, 2)}% faster (FPS)"

        comparison = {
            "winner": winner,
            "margin": margin,
            "speedup_ratio": round(speedup_fps, 3),
            "cpu_time_sec": cpu_time,
            "cuda_time_sec": cuda_time,
            "cpu_fps": cpu_fps,
            "cuda_fps": cuda_fps,
        }
        results["comparison"] = comparison

        print("\n" + "=" * 70)
        print(" [*] BENCHMARK COMPARISON SUMMARY")
        print(f" [*] Winner:        {winner} ({margin})")
        print(f" [*] CPU:           {cpu_time}s | {cpu_fps} FPS")
        print(f" [*] CUDA:          {cuda_time}s | {cuda_fps} FPS")
        print(f" [*] Speedup Ratio: {round(speedup_fps, 2)}x")
        print("=" * 70)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full benchmark results saved to: {args.output}")


if __name__ == "__main__":
    main()
