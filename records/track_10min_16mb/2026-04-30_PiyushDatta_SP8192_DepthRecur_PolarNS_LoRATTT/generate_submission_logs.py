#!/usr/bin/env python3
"""
Run train_gpt.py with 3 different seeds and save training logs.

Usage:
    # On 8xH100:
    python generate_submission_logs.py --nproc 8

    # On 4xA100:
    python generate_submission_logs.py --nproc 4

    # Custom seeds:
    python generate_submission_logs.py --nproc 8 --seeds 42 314 999

    # Dry run (print commands without executing):
    python generate_submission_logs.py --nproc 8 --dry-run

Output:
    logs/seed_42.log, logs/seed_314.log, logs/seed_999.log
    logs/summary.json  (parsed results from all seeds)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
TRAIN_SCRIPT = SCRIPT_DIR / "train_gpt.py"
LOGS_DIR = SCRIPT_DIR / "logs"


def parse_log(log_path: str) -> dict:
    """Parse a training log file and extract key metrics."""
    result = {}
    with open(log_path, "r") as f:
        text = f.read()

    m = re.search(r"seed:\s*(\d+)", text)
    if m:
        result["seed"] = int(m.group(1))

    m = re.search(r"stopping_early.*step:\s*(\d+)/", text)
    if m:
        result["training_steps"] = int(m.group(1))

    m = re.search(r"train_batch_tokens:\s*(\d+)", text)
    if m:
        result["train_batch_tokens"] = int(m.group(1))

    m = re.search(r"world_size:\s*(\d+)", text)
    if m:
        result["world_size"] = int(m.group(1))

    m = re.search(r"model_params:(\d+)", text)
    if m:
        result["model_params"] = int(m.group(1))

    m = re.search(r"peak memory allocated:\s*(\d+)\s*MiB", text)
    if m:
        result["peak_memory_mib"] = int(m.group(1))

    m = re.search(r"swa:applying SWA weights \((\d+) checkpoints\)", text)
    if m:
        result["swa_checkpoints"] = int(m.group(1))

    m = re.search(r"pre-quantization post-ema val_loss:([\d.]+) val_bpb:([\d.]+)", text)
    if m:
        result["pre_quant_val_bpb"] = float(m.group(2))

    m = re.search(r"Code size:\s*(\d+)\s*bytes", text)
    if m:
        result["code_bytes"] = int(m.group(1))

    m = re.search(r"Serialized model quantized\+\w+:\s*(\d+)\s*bytes", text)
    if m:
        result["model_bytes"] = int(m.group(1))

    m = re.search(r"Total submission size quantized\+\w+:\s*(\d+)\s*bytes", text)
    if m:
        result["artifact_bytes"] = int(m.group(1))

    m = re.search(r"^quantized val_loss:([\d.]+) val_bpb:([\d.]+)", text, re.MULTILINE)
    if m:
        result["post_gptq_val_bpb"] = float(m.group(2))

    m = re.search(r"quantized_sliding_window val_loss:([\d.]+) val_bpb:([\d.]+)", text)
    if m:
        result["sliding_val_bpb"] = float(m.group(2))

    # tok/s from last logged training step
    tok_matches = re.findall(r"tok/s:\s*(\d+)", text)
    if tok_matches:
        result["tok_per_sec"] = int(tok_matches[-1])

    return result


def run_seed(seed: int, nproc: int, data_dir: str, extra_env: dict) -> str:
    """Run training for a single seed. Returns path to log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"seed_{seed}.log"

    env = os.environ.copy()
    env["SEED"] = str(seed)
    env["DATA_DIR"] = data_dir
    env.update(extra_env)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(TRAIN_SCRIPT),
    ]

    print(f"\n{'='*70}")
    print(f"Running seed={seed} with {nproc} GPUs")
    print(f"Log: {log_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR),
        )

    if proc.returncode != 0:
        print(f"WARNING: seed={seed} exited with code {proc.returncode}")
    else:
        print(f"seed={seed} completed successfully")

    return str(log_path)


def main():
    parser = argparse.ArgumentParser(description="Generate 3-seed submission logs")
    parser.add_argument("--nproc", type=int, required=True,
                        help="Number of GPUs (e.g., 4 for A100, 8 for H100)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 314, 999],
                        help="Seeds to run (default: 42 314 999)")
    parser.add_argument("--data-dir", type=str, default="./data/",
                        help="Data directory (default: ./data/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--env", type=str, default="",
                        help="Extra env vars as KEY=VAL,KEY2=VAL2")
    args = parser.parse_args()

    extra_env = {}
    if args.env:
        for pair in args.env.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                extra_env[k.strip()] = v.strip()

    if args.dry_run:
        for seed in args.seeds:
            env_str = " ".join(f"{k}={v}" for k, v in extra_env.items())
            print(f"DATA_DIR={args.data_dir} SEED={seed} {env_str} "
                  f"torchrun --standalone --nproc_per_node={args.nproc} {TRAIN_SCRIPT}")
        return

    # Run each seed
    results = {}
    for seed in args.seeds:
        log_path = run_seed(seed, args.nproc, args.data_dir, extra_env)
        parsed = parse_log(log_path)
        results[str(seed)] = parsed
        if "sliding_val_bpb" in parsed:
            print(f"  -> sliding_val_bpb = {parsed['sliding_val_bpb']:.4f}")
        if "artifact_bytes" in parsed:
            fits = parsed["artifact_bytes"] < 16_000_000
            print(f"  -> artifact = {parsed['artifact_bytes']:,} bytes ({'FITS' if fits else 'OVER!'})")

    # Compute summary statistics
    sliding_bpbs = [r["sliding_val_bpb"] for r in results.values() if "sliding_val_bpb" in r]
    summary = {
        "seeds": args.seeds,
        "nproc": args.nproc,
        "seed_results": results,
    }
    if sliding_bpbs:
        mean_bpb = sum(sliding_bpbs) / len(sliding_bpbs)
        std_bpb = (sum((x - mean_bpb) ** 2 for x in sliding_bpbs) / len(sliding_bpbs)) ** 0.5
        summary["mean_sliding_val_bpb"] = round(mean_bpb, 6)
        summary["std_sliding_val_bpb"] = round(std_bpb, 6)

    summary_path = LOGS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for seed_str, r in results.items():
        sliding = r.get("sliding_val_bpb", "N/A")
        post_gptq = r.get("post_gptq_val_bpb", "N/A")
        pre_quant = r.get("pre_quant_val_bpb", "N/A")
        size = r.get("artifact_bytes", "N/A")
        steps = r.get("training_steps", "N/A")
        print(f"  seed={seed_str}: sliding={sliding}, post_gptq={post_gptq}, "
              f"pre_quant={pre_quant}, size={size}, steps={steps}")

    if sliding_bpbs:
        print(f"\n  Mean sliding_val_bpb: {summary['mean_sliding_val_bpb']:.6f}")
        print(f"  Std  sliding_val_bpb: {summary['std_sliding_val_bpb']:.6f}")

    print(f"\nLogs saved to: {LOGS_DIR}/")
    print(f"Summary saved to: {summary_path}")
    print(f"\nUpdate submission.json and README.md with these results.")


if __name__ == "__main__":
    main()
