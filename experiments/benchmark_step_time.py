#!/usr/bin/env python3
"""
Benchmark training step time for parameter-golf scripts.

Usage:
    # Benchmark baseline:
    BENCHMARK_STEPS=50 torchrun --standalone --nproc_per_node=8 experiments/benchmark_step_time.py --script experiments/clean_train.py

    # Or set BENCHMARK_ONLY=1 in the training script itself (if supported)
    BENCHMARK_ONLY=1 torchrun --standalone --nproc_per_node=8 experiments/clean_train_131_triton.py

This script measures the average wall-clock time per training step,
excluding the first 10 steps (warmup/compilation).
"""

import os
import sys
import time
import subprocess

def main():
    script = sys.argv[sys.argv.index("--script") + 1] if "--script" in sys.argv else None
    if script is None:
        print("Usage: python benchmark_step_time.py --script <path_to_train_script.py>")
        sys.exit(1)

    n_steps = int(os.environ.get("BENCHMARK_STEPS", "50"))

    # Set env vars for a short benchmark run
    env = os.environ.copy()
    env["BENCHMARK_ONLY"] = "1"
    env["BENCHMARK_STEPS"] = str(n_steps)
    env["WANDB_MODE"] = "disabled"
    env["VAL_LOSS_EVERY"] = "99999"  # Skip validation
    env["TRAIN_LOG_EVERY"] = "1"  # Log every step for timing

    print(f"Benchmarking {script} for {n_steps} steps...")
    result = subprocess.run(
        [sys.executable, script],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])

if __name__ == "__main__":
    main()
