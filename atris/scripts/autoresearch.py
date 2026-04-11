#!/usr/bin/env python3
"""
Autoresearch Loop for Parameter Golf

Based on Karpathy's autoresearch pattern:
- Fixed metric (val_bpb)
- Fixed compute budget (10 min on 8xH100)
- Modify train_gpt.py → run → measure → keep/revert

Usage:
    # On a RunPod 8xH100:
    python autoresearch.py --mode run --experiment "baseline_repro"

    # View results:
    python autoresearch.py --mode status

    # Compare two experiments:
    python autoresearch.py --mode compare --experiments "exp1,exp2"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = REPO_ROOT / "train_gpt.py"
RESULTS_DIR = REPO_ROOT / "atris" / "experiments"
LOGS_DIR = REPO_ROOT / "atris" / "logs"
BEST_SCRIPT = REPO_ROOT / "atris" / "best_train_gpt.py"


def run_experiment(
    name: str,
    env_overrides: dict | None = None,
    nproc: int = 8,
    max_wallclock: float = 600.0,
    dry_run: bool = False,
) -> dict:
    """Run a single training experiment and capture results."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_id = f"{timestamp}_{name}"
    exp_dir = RESULTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save the current train_gpt.py snapshot
    shutil.copy2(TRAIN_SCRIPT, exp_dir / "train_gpt.py")

    # Build environment
    env = os.environ.copy()
    env.update({
        "RUN_ID": exp_id,
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock),
        "VAL_LOSS_EVERY": "200",
        "TRAIN_LOG_EVERY": "50",
    })
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    # Save experiment config
    config = {
        "name": name,
        "exp_id": exp_id,
        "timestamp": timestamp,
        "env_overrides": env_overrides or {},
        "nproc": nproc,
        "max_wallclock": max_wallclock,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if dry_run:
        print(f"[DRY RUN] Would run experiment: {exp_id}")
        print(f"  Config: {json.dumps(config, indent=2)}")
        return config

    # Run training
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(TRAIN_SCRIPT),
    ]

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=int(max_wallclock + 300),  # extra 5 min for eval
    )
    elapsed = time.time() - start_time

    # Save stdout/stderr
    with open(exp_dir / "stdout.txt", "w") as f:
        f.write(result.stdout)
    with open(exp_dir / "stderr.txt", "w") as f:
        f.write(result.stderr)

    # Parse results from stdout
    metrics = parse_metrics(result.stdout)
    metrics["elapsed_seconds"] = elapsed
    metrics["return_code"] = result.returncode

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Copy log file if it exists
    log_pattern = REPO_ROOT / "logs" / f"{exp_id}.txt"
    if log_pattern.exists():
        shutil.copy2(log_pattern, exp_dir / "train.log")

    # Print summary
    print(f"\n{'='*80}")
    print(f"RESULT: {exp_id}")
    print(f"  val_bpb: {metrics.get('val_bpb', 'N/A')}")
    print(f"  val_bpb (int8+zlib): {metrics.get('q_val_bpb', 'N/A')}")
    print(f"  artifact_bytes: {metrics.get('artifact_bytes', 'N/A')}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  return_code: {result.returncode}")
    print(f"{'='*80}\n")

    return metrics


def parse_metrics(stdout: str) -> dict:
    """Extract key metrics from training output."""
    metrics = {}
    for line in stdout.strip().split("\n"):
        line = line.strip()

        # Final int8+zlib roundtrip (the official score)
        if "final_int8_zlib_roundtrip_exact" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["q_val_bpb"] = float(part.split(":")[1])
                elif part.startswith("val_loss:"):
                    metrics["q_val_loss"] = float(part.split(":")[1])
        elif "final_int8_zlib_roundtrip" in line and "exact" not in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["q_val_bpb_rounded"] = float(part.split(":")[1])

        # Pre-quant val metrics (last validation step)
        elif line.startswith("step:") and "val_bpb:" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["val_bpb"] = float(part.split(":")[1])
                elif part.startswith("val_loss:"):
                    metrics["val_loss"] = float(part.split(":")[1])

        # Model size
        elif "Total submission size int8+zlib:" in line:
            try:
                metrics["artifact_bytes"] = int(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass

        # Serialized model size
        elif "Serialized model int8+zlib:" in line:
            try:
                metrics["model_bytes"] = int(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass

        # Code size
        elif "Code size:" in line:
            try:
                metrics["code_bytes"] = int(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass

        # Param count
        elif "model_params:" in line:
            try:
                metrics["param_count"] = int(line.split(":")[1].strip())
            except (IndexError, ValueError):
                pass

        # Peak memory
        elif "peak memory allocated:" in line:
            try:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "allocated:":
                        metrics["peak_mem_mib"] = int(parts[i + 1])
            except (IndexError, ValueError):
                pass

        # Early stopping
        elif "stopping_early:" in line:
            metrics["stopped_early"] = True
            for part in line.split():
                if part.startswith("step:"):
                    metrics["final_step"] = part.split(":")[1]

    return metrics


def load_all_results() -> list[dict]:
    """Load all experiment results sorted by BPB."""
    results = []
    if not RESULTS_DIR.exists():
        return results

    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics_file = exp_dir / "metrics.json"
        config_file = exp_dir / "config.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)
        config = {}
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)

        results.append({
            "exp_id": exp_dir.name,
            "name": config.get("name", "unknown"),
            **metrics,
        })

    # Sort by q_val_bpb (lower is better), putting None at end
    results.sort(key=lambda r: r.get("q_val_bpb", 999.0))
    return results


def show_status():
    """Print leaderboard of all experiments."""
    results = load_all_results()
    if not results:
        print("No experiments found.")
        return

    print(f"\n{'='*100}")
    print(f"{'EXPERIMENT LEADERBOARD':^100}")
    print(f"{'='*100}")
    print(f"{'Rank':<5} {'Name':<30} {'BPB (q)':<12} {'BPB (raw)':<12} {'Artifact':<12} {'Params':<12} {'Status'}")
    print(f"{'-'*100}")

    baseline_bpb = 1.2244
    for i, r in enumerate(results):
        q_bpb = r.get("q_val_bpb", None)
        raw_bpb = r.get("val_bpb", None)
        artifact = r.get("artifact_bytes", None)
        params = r.get("param_count", None)
        rc = r.get("return_code", None)

        q_str = f"{q_bpb:.4f}" if q_bpb else "N/A"
        raw_str = f"{raw_bpb:.4f}" if raw_bpb else "N/A"
        art_str = f"{artifact:,}" if artifact else "N/A"
        par_str = f"{params:,}" if params else "N/A"

        delta = ""
        if q_bpb and q_bpb < baseline_bpb:
            delta = f" ({baseline_bpb - q_bpb:+.4f})"
        status = "OK" if rc == 0 else f"FAIL({rc})" if rc else "?"

        print(f"{i+1:<5} {r['name']:<30} {q_str:<12} {raw_str:<12} {art_str:<12} {par_str:<12} {status}{delta}")

    print(f"{'='*100}")
    print(f"Baseline to beat: 1.2244 BPB | Need improvement ≥ 0.005 nats for new record")
    print()


def log_experiment(exp_id: str, metrics: dict, notes: str = ""):
    """Append to the experiment log."""
    log_file = LOGS_DIR / "experiments.jsonl"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exp_id": exp_id,
        "q_val_bpb": metrics.get("q_val_bpb"),
        "val_bpb": metrics.get("val_bpb"),
        "artifact_bytes": metrics.get("artifact_bytes"),
        "notes": notes,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Autoresearch Loop")
    parser.add_argument("--mode", choices=["run", "status", "compare", "sweep"], required=True)
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--experiments", type=str, help="Comma-separated experiment names for compare")
    parser.add_argument("--nproc", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--wallclock", type=float, default=600.0, help="Max training seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")

    # Env overrides as key=value pairs
    parser.add_argument("--env", nargs="*", help="Environment overrides: KEY=VALUE ...")

    args = parser.parse_args()

    if args.mode == "status":
        show_status()
        return

    if args.mode == "run":
        if not args.experiment:
            print("Error: --experiment required for run mode")
            sys.exit(1)

        env_overrides = {}
        if args.env:
            for pair in args.env:
                k, v = pair.split("=", 1)
                env_overrides[k] = v

        metrics = run_experiment(
            name=args.experiment,
            env_overrides=env_overrides,
            nproc=args.nproc,
            max_wallclock=args.wallclock,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            log_experiment(args.experiment, metrics)

    elif args.mode == "sweep":
        # Quick hyperparameter sweep
        sweeps = {
            "lr_high": {"MATRIX_LR": "0.06", "SCALAR_LR": "0.06"},
            "lr_low": {"MATRIX_LR": "0.02", "SCALAR_LR": "0.02"},
            "lr_very_high": {"MATRIX_LR": "0.08", "SCALAR_LR": "0.08"},
            "batch_large": {"TRAIN_BATCH_TOKENS": "1048576"},
            "batch_small": {"TRAIN_BATCH_TOKENS": "262144"},
            "seq_512": {"TRAIN_SEQ_LEN": "512"},
            "momentum_high": {"MUON_MOMENTUM": "0.98"},
            "momentum_low": {"MUON_MOMENTUM": "0.90"},
            "warmdown_long": {"WARMDOWN_ITERS": "2400"},
            "warmdown_short": {"WARMDOWN_ITERS": "600"},
        }

        print(f"Sweep has {len(sweeps)} experiments")
        print(f"Estimated cost: {len(sweeps) * 3.3:.0f} ({len(sweeps)} × $3.30)")
        print()

        for name, overrides in sweeps.items():
            exp_name = f"sweep_{name}"
            print(f"--- Running: {exp_name} ---")
            metrics = run_experiment(
                name=exp_name,
                env_overrides=overrides,
                nproc=args.nproc,
                max_wallclock=args.wallclock,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                log_experiment(exp_name, metrics, notes=f"Sweep: {overrides}")


if __name__ == "__main__":
    main()
