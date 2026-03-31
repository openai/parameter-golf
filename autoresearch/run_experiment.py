#!/usr/bin/env python3
"""
Autoresearch-style experiment runner for Parameter Golf.
Runs quick 5-minute experiments on M3 Ultra, measures val_bpb,
logs results, and enables systematic exploration.

Usage:
    .venv/bin/python autoresearch/run_experiment.py --name "baseline_12L" --iters 1000
    .venv/bin/python autoresearch/run_experiment.py --name "no_smeargate" --iters 1000 --env "SMEAR_GATE=0"
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULTS_FILE = Path("autoresearch/results.jsonl")
VENV_PYTHON = str(Path(__file__).parent.parent / ".venv/bin/python")
TRAIN_SCRIPT = str(Path(__file__).parent.parent / "train_gpt_mlx.py")


def run_experiment(name: str, iters: int = 1000, extra_env: dict = None):
    """Run a single experiment and return results."""
    env = {
        **os.environ,
        "ITERATIONS": str(iters),
        "TRAIN_BATCH_TOKENS": "65536",
        "GRAD_ACCUM_STEPS": "1",
        "VAL_BATCH_SIZE": "524288",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": str(max(iters // 5, 1)),
        "WARMUP_STEPS": "10",
        "MLX_EAGER_EVAL": "1",
        "MLX_MAX_MICROBATCH_TOKENS": "32768",
        "MAX_WALLCLOCK_SECONDS": "0",
        "RUN_ID": f"exp_{name}",
    }
    if extra_env:
        env.update(extra_env)

    log_file = f"/tmp/exp_{name}.log"
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  iters={iters}, extra={extra_env or {}}")
    print(f"  log: {log_file}")
    print(f"{'='*60}")

    t0 = time.time()
    with open(log_file, "w") as f:
        proc = subprocess.run(
            [VENV_PYTHON, "-u", TRAIN_SCRIPT],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=7200,  # 2 hour max
            cwd=str(Path(__file__).parent.parent),
        )
    elapsed = time.time() - t0

    # Parse results from log
    results = {
        "name": name,
        "iters": iters,
        "extra_env": extra_env or {},
        "elapsed_s": round(elapsed, 1),
        "exit_code": proc.returncode,
    }

    with open(log_file) as f:
        log = f.read()

    # Extract key metrics
    for line in log.split("\n"):
        if "val_bpb:" in line and "step:" in line and "val_bpb:enabled" not in line:
            parts = line.split()
            for p in parts:
                if p.startswith("val_bpb:"):
                    results["val_bpb"] = float(p.split(":")[1])
                if p.startswith("val_loss:"):
                    results["val_loss"] = float(p.split(":")[1])
                if p.startswith("step:"):
                    results["final_step"] = p.split(":")[1]
        if "int8_zlib_roundtrip_exact" in line:
            for p in line.split():
                if p.startswith("val_bpb:"):
                    results["int8_bpb"] = float(p.split(":")[1])
        if "int6_gptq_roundtrip_exact" in line:
            for p in line.split():
                if p.startswith("val_bpb:"):
                    results["int6_bpb"] = float(p.split(":")[1])
        if "int6+zstd:" in line or "int6_zstd:" in line:
            for p in line.split():
                if p.startswith("total_submission:"):
                    results["submission_bytes"] = int(p.split(":")[1])
        if "tok_s:" in line:
            for p in line.split():
                if p.startswith("tok_s:"):
                    results["tok_s"] = float(p.split(":")[1])
        if "model_params:" in line:
            for p in line.split():
                if p.startswith("model_params:"):
                    results["params"] = int(p.split(":")[1])

    # Log result
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(results) + "\n")

    bpb = results.get("val_bpb", "N/A")
    int6 = results.get("int6_bpb", "N/A")
    size = results.get("submission_bytes", "N/A")
    print(f"\nRESULT: {name}")
    print(f"  val_bpb={bpb}, int6_roundtrip={int6}, size={size}")
    print(f"  elapsed={elapsed:.0f}s, exit={proc.returncode}")
    return results


def print_leaderboard():
    """Print all experiment results sorted by val_bpb."""
    if not RESULTS_FILE.exists():
        print("No experiments yet.")
        return
    results = []
    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if line:
            results.append(json.loads(line))
    results.sort(key=lambda r: r.get("val_bpb", 999))
    print(f"\n{'='*70}")
    print(f"{'Name':<30} {'bpb':>8} {'int6':>8} {'size':>10} {'time':>6}")
    print(f"{'='*70}")
    for r in results:
        bpb = f"{r.get('val_bpb', 0):.4f}" if "val_bpb" in r else "N/A"
        int6 = f"{r.get('int6_bpb', 0):.4f}" if "int6_bpb" in r else "N/A"
        size = f"{r.get('submission_bytes', 0):,}" if "submission_bytes" in r else "N/A"
        t = f"{r.get('elapsed_s', 0):.0f}s"
        print(f"{r['name']:<30} {bpb:>8} {int6:>8} {size:>10} {t:>6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--env", nargs="*", default=[])
    parser.add_argument("--leaderboard", action="store_true")
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
        sys.exit(0)

    extra = {}
    for e in args.env:
        k, v = e.split("=", 1)
        extra[k] = v

    run_experiment(args.name, args.iters, extra)
    print_leaderboard()
