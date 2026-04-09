#!/usr/bin/env python3
"""Autoresearch orchestrator for Parameter Golf experiments.

Two-phase architecture:
  Phase 1 (model optimization): fast train + FP eval proxy
  Phase 2 (eval optimization): frozen checkpoint + full eval

Subcommands:
  fast  — Phase 1 fast screening (~3-4 min): short wallclock, shard subset, skip GPTQ
  full  — Full calibration run (~16 min): all shards, 600s, GPTQ, sliding window
  eval  — Phase 2 eval-only (~5-6 min): run fast_eval.py on frozen checkpoint

Usage:
  python run_autoresearch.py fast --desc "wider MLP 4x"
  python run_autoresearch.py full --desc "calibration run"
  python run_autoresearch.py eval --checkpoint best_model.pt --desc "stride=32" --sliding --stride 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────

FAST_WALLCLOCK = 180            # seconds for fast training
FAST_TRAIN_SHARDS = 12          # number of shards for fast runs
TOTAL_SHARDS = 40               # total available training shards
FULL_WALLCLOCK = 600            # seconds for full training
RECALIBRATE_EVERY = 5           # full run every N successes
FAILURE_RECALIBRATE = 12        # full run after N consecutive failures
EARLY_STOP_THRESHOLD = 0.15    # nats above baseline to kill early
EARLY_STOP_MIN_STEP = 50       # don't early-stop before this step

EXPERIMENTS_FILE = "experiments.jsonl"
BASELINE_CURVE_FILE = "baseline_loss_curve.json"
FAST_BASELINE_FILE = "fast_baseline.json"


# ── Helpers ────────────────────────────────────────────────────

def git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def next_experiment_id(mode: str) -> str:
    """Generate next experiment ID like fast_001, full_002, eval_003."""
    count = 0
    if Path(EXPERIMENTS_FILE).exists():
        with open(EXPERIMENTS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    count += 1
    return f"{mode}_{count + 1:03d}"


def load_baseline_curve() -> list[dict] | None:
    if Path(BASELINE_CURVE_FILE).exists():
        with open(BASELINE_CURVE_FILE) as f:
            return json.load(f)
    return None


def interpolate_baseline_loss(curve: list[dict], progress: float) -> float | None:
    """Interpolate expected train_loss at a given fractional progress."""
    if not curve or progress <= 0:
        return None
    # Find bracketing points
    prev = None
    for point in curve:
        if point["progress"] >= progress:
            if prev is None:
                return point["train_loss"]
            # Linear interpolation
            frac = (progress - prev["progress"]) / max(point["progress"] - prev["progress"], 1e-9)
            return prev["train_loss"] + frac * (point["train_loss"] - prev["train_loss"])
        prev = point
    # Beyond curve — use last point
    return curve[-1]["train_loss"] if curve else None


def load_fast_baseline() -> float | None:
    if Path(FAST_BASELINE_FILE).exists():
        with open(FAST_BASELINE_FILE) as f:
            data = json.load(f)
            return data.get("bpb")
    return None


def save_fast_baseline(bpb: float) -> None:
    with open(FAST_BASELINE_FILE, "w") as f:
        json.dump({"bpb": bpb, "timestamp": datetime.now(timezone.utc).isoformat()}, f)


def log_experiment(entry: dict) -> None:
    with open(EXPERIMENTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def count_recent_stats() -> tuple[int, int]:
    """Return (successes_since_last_full, consecutive_failures)."""
    successes = 0
    failures = 0
    if not Path(EXPERIMENTS_FILE).exists():
        return 0, 0
    entries = []
    with open(EXPERIMENTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # Walk backwards
    consecutive_failures = 0
    successes_since_full = 0
    for e in reversed(entries):
        if e.get("mode") == "full":
            break
        if e.get("mode") == "fast":
            delta = e.get("baseline_delta")
            if delta is not None and delta < 0:
                successes_since_full += 1
                consecutive_failures = 0  # reset streak on success
            elif delta is not None:
                consecutive_failures += 1 if successes_since_full == 0 or consecutive_failures > 0 else 0
    # Recount consecutive failures from the end
    consecutive_failures = 0
    for e in reversed(entries):
        if e.get("mode") == "full":
            break
        if e.get("mode") == "fast":
            delta = e.get("baseline_delta")
            if delta is not None and delta < 0:
                break
            consecutive_failures += 1
    return successes_since_full, consecutive_failures


def parse_val_bpb(output: str) -> float | None:
    """Extract the last val_bpb from training output."""
    matches = re.findall(r"val_bpb:(\d+\.\d+)", output)
    return float(matches[-1]) if matches else None


def parse_sliding_bpb(output: str) -> float | None:
    match = re.search(r"final_int6_sliding_window.*val_bpb:(\d+\.\d+)", output)
    return float(match.group(1)) if match else None


def parse_roundtrip_bpb(output: str) -> float | None:
    match = re.search(r"final_int6_.*roundtrip.*val_bpb:(\d+\.\d+)", output)
    return float(match.group(1)) if match else None


def parse_loss_curve(output: str) -> list[dict]:
    """Extract training loss curve from output for baseline_loss_curve.json."""
    curve = []
    total_steps = None
    # Find total iterations from output
    m = re.search(r"iterations:(\d+)", output)
    if m:
        total_steps = int(m.group(1))
    for match in re.finditer(r"step:(\d+)/(\d+)\s+train_loss:(\d+\.\d+)", output):
        step = int(match.group(1))
        total = int(match.group(2))
        loss = float(match.group(3))
        if total_steps is None:
            total_steps = total
        curve.append({
            "step": step,
            "progress": step / max(total, 1),
            "train_loss": loss,
        })
    return curve


def parse_train_time(output: str) -> float | None:
    """Extract total training time in seconds."""
    matches = re.findall(r"train_time:(\d+)ms", output)
    if matches:
        return float(matches[-1]) / 1000.0
    return None


# ── Subcommands ────────────────────────────────────────────────

def cmd_fast(args: argparse.Namespace) -> None:
    """Run a fast Phase 1 experiment."""
    exp_id = next_experiment_id("fast")
    print(f"{'=' * 60}")
    print(f"FAST EXPERIMENT: {exp_id}")
    print(f"Description: {args.desc}")
    print(f"{'=' * 60}")

    # Use fixed shards + seed for reproducible A/B comparison
    shard_indices = sorted(random.Random(42).sample(range(TOTAL_SHARDS), min(FAST_TRAIN_SHARDS, TOTAL_SHARDS)))
    shard_seed = 42

    env = os.environ.copy()
    env.update({
        "MAX_WALLCLOCK_SECONDS": str(FAST_WALLCLOCK),
        "SKIP_QUANT": "1",
        "EVAL_STRIDE": "0",
        "TRAIN_LOG_EVERY": "10",
        "VAL_LOSS_EVERY": "9999",
        "TRAIN_SHARD_INDICES": ",".join(str(i) for i in shard_indices),
        "TRAIN_SHARD_SHUFFLE_SEED": str(shard_seed),
    })

    baseline_curve = load_baseline_curve()
    fast_baseline = load_fast_baseline()

    print(f"Shards: {shard_indices}")
    print(f"Wallclock: {FAST_WALLCLOCK}s | SKIP_QUANT=1 | EVAL_STRIDE=0")
    if fast_baseline is not None:
        print(f"Fast baseline BPB: {fast_baseline:.4f}")
    print()

    t_start = time.perf_counter()
    early_stopped = False
    output_lines: list[str] = []

    proc = subprocess.Popen(
        [sys.executable, "train_gpt.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)

            # Early stopping: check train_loss against baseline curve
            if baseline_curve:
                m = re.match(r"step:(\d+)/(\d+)\s+train_loss:(\d+\.\d+)", line)
                if m:
                    step = int(m.group(1))
                    total = int(m.group(2))
                    loss = float(m.group(3))
                    if step >= EARLY_STOP_MIN_STEP:
                        progress = step / max(total, 1)
                        expected = interpolate_baseline_loss(baseline_curve, progress)
                        if expected is not None and (loss - expected) > EARLY_STOP_THRESHOLD:
                            print(f"\n*** EARLY STOP at step {step}: "
                                  f"loss={loss:.4f} vs baseline={expected:.4f} "
                                  f"(delta={loss - expected:.4f} > {EARLY_STOP_THRESHOLD}) ***\n")
                            early_stopped = True
                            proc.terminate()
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            break

        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        print("\nInterrupted by user.")
        return

    total_time = time.perf_counter() - t_start
    output = "\n".join(output_lines)

    # Parse results
    bpb = parse_val_bpb(output) if not early_stopped else None
    train_time = parse_train_time(output)

    # Compute delta
    baseline_delta = None
    if bpb is not None and fast_baseline is not None:
        baseline_delta = bpb - fast_baseline

    # Log experiment
    entry = {
        "id": exp_id,
        "description": args.desc,
        "mode": "fast",
        "bpb": bpb,
        "bpb_full": None,
        "bpb_sliding": None,
        "train_time_s": round(train_time, 1) if train_time else None,
        "total_time_s": round(total_time, 1),
        "shards_used": len(shard_indices),
        "shard_indices": shard_indices,
        "early_stopped": early_stopped,
        "baseline_delta": round(baseline_delta, 6) if baseline_delta is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_short(),
    }
    log_experiment(entry)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"fast_result {exp_id}: ", end="")
    if early_stopped:
        print("EARLY STOPPED (loss too high)")
    elif bpb is None:
        print("CRASHED (no val_bpb found)")
    else:
        print(f"val_bpb={bpb:.4f}", end="")
        if baseline_delta is not None:
            direction = "IMPROVED" if baseline_delta < 0 else "WORSE"
            print(f" | delta={baseline_delta:+.4f} ({direction})", end="")
        print()
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 60}")

    # Update fast baseline if this is the first run or explicitly requested
    if fast_baseline is None and bpb is not None:
        save_fast_baseline(bpb)
        print(f"fast_baseline: set to {bpb:.4f} (first run)")

    # Calibration recommendations
    successes, failures = count_recent_stats()
    if successes >= RECALIBRATE_EVERY:
        print(f"\nRECOMMEND: {successes} successes since last full run. "
              f"Run: python run_autoresearch.py full --desc \"calibration\"")
    if failures >= FAILURE_RECALIBRATE:
        print(f"\nRECOMMEND: {failures} consecutive failures. "
              f"Run: python run_autoresearch.py full --desc \"calibration after failures\"")


def cmd_full(args: argparse.Namespace) -> None:
    """Run a full calibration experiment."""
    exp_id = next_experiment_id("full")
    print(f"{'=' * 60}")
    print(f"FULL EXPERIMENT: {exp_id}")
    print(f"Description: {args.desc}")
    print(f"Wallclock: {FULL_WALLCLOCK}s | All shards | GPTQ | Sliding window")
    print(f"{'=' * 60}")

    env = os.environ.copy()
    env.update({
        "MAX_WALLCLOCK_SECONDS": str(FULL_WALLCLOCK),
        "EVAL_STRIDE": "64",
        "TRAIN_LOG_EVERY": "10",
    })
    # Remove fast-mode env vars if set
    env.pop("SKIP_QUANT", None)
    env.pop("TRAIN_SHARD_INDICES", None)
    env.pop("TRAIN_SHARD_SHUFFLE_SEED", None)

    t_start = time.perf_counter()
    output_lines: list[str] = []

    proc = subprocess.Popen(
        [sys.executable, "train_gpt.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        print("\nInterrupted by user.")
        return

    total_time = time.perf_counter() - t_start
    output = "\n".join(output_lines)

    # Parse results
    roundtrip_bpb = parse_roundtrip_bpb(output)
    sliding_bpb = parse_sliding_bpb(output)
    fp_bpb = parse_val_bpb(output)
    train_time = parse_train_time(output)

    # Save loss curve for early stopping
    curve = parse_loss_curve(output)
    if curve:
        with open(BASELINE_CURVE_FILE, "w") as f:
            json.dump(curve, f, indent=2)
        print(f"Saved baseline loss curve ({len(curve)} points) to {BASELINE_CURVE_FILE}")

    # Update fast baseline (using FP BPB from this full run's eval_val)
    if fp_bpb is not None:
        save_fast_baseline(fp_bpb)
        print(f"Updated fast baseline to {fp_bpb:.4f}")

    # Log experiment
    entry = {
        "id": exp_id,
        "description": args.desc,
        "mode": "full",
        "bpb": fp_bpb,
        "bpb_full": roundtrip_bpb,
        "bpb_sliding": sliding_bpb,
        "train_time_s": round(train_time, 1) if train_time else None,
        "total_time_s": round(total_time, 1),
        "shards_used": TOTAL_SHARDS,
        "shard_indices": None,
        "early_stopped": False,
        "baseline_delta": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_short(),
    }
    log_experiment(entry)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"full_result {exp_id}:")
    if fp_bpb is not None:
        print(f"  FP val_bpb:       {fp_bpb:.4f}")
    if roundtrip_bpb is not None:
        print(f"  Roundtrip BPB:    {roundtrip_bpb:.4f}")
    if sliding_bpb is not None:
        print(f"  Sliding BPB:      {sliding_bpb:.4f}")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 60}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Run Phase 2 eval-only experiment on a frozen checkpoint."""
    exp_id = next_experiment_id("eval")
    print(f"{'=' * 60}")
    print(f"EVAL EXPERIMENT: {exp_id}")
    print(f"Description: {args.desc}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'=' * 60}")

    cmd = [sys.executable, "fast_eval.py", "--checkpoint", args.checkpoint]
    if args.val_fraction < 1.0:
        cmd += ["--val-fraction", str(args.val_fraction)]
    if args.sliding:
        cmd += ["--sliding", "--stride", str(args.stride)]

    t_start = time.perf_counter()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("TIMEOUT: eval exceeded 15 minutes")
        output = ""
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return

    total_time = time.perf_counter() - t_start

    # Parse BPB
    bpb = None
    m = re.search(r"val_bpb:(\d+\.\d+)", output)
    if m:
        bpb = float(m.group(1))

    entry = {
        "id": exp_id,
        "description": args.desc,
        "mode": "eval",
        "bpb": bpb,
        "bpb_full": None,
        "bpb_sliding": bpb if args.sliding else None,
        "train_time_s": None,
        "total_time_s": round(total_time, 1),
        "shards_used": None,
        "shard_indices": None,
        "early_stopped": False,
        "baseline_delta": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_short(),
    }
    log_experiment(entry)

    print(f"\n{'=' * 60}")
    print(f"eval_result {exp_id}: ", end="")
    if bpb is not None:
        print(f"val_bpb={bpb:.6f}")
    else:
        print("FAILED (no val_bpb found)")
        if output.strip():
            print(f"Output:\n{output[-500:]}")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 60}")


# ── CLI ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autoresearch orchestrator for Parameter Golf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # fast
    p_fast = sub.add_parser("fast", help="Phase 1: fast screening experiment (~3-4 min)")
    p_fast.add_argument("--desc", required=True, help="Short description of what changed")

    # full
    p_full = sub.add_parser("full", help="Full calibration run (~16 min)")
    p_full.add_argument("--desc", required=True, help="Short description")

    # eval
    p_eval = sub.add_parser("eval", help="Phase 2: eval-only on frozen checkpoint (~5-6 min)")
    p_eval.add_argument("--checkpoint", required=True, help="Path to frozen .pt checkpoint")
    p_eval.add_argument("--desc", required=True, help="Short description of eval change")
    p_eval.add_argument("--sliding", action="store_true", help="Use sliding window eval")
    p_eval.add_argument("--stride", type=int, default=64, help="Sliding window stride")
    p_eval.add_argument("--val-fraction", type=float, default=1.0, help="Fraction of val data")

    # status
    sub.add_parser("status", help="Show current experiment stats and baselines")

    args = parser.parse_args()

    if args.command == "fast":
        cmd_fast(args)
    elif args.command == "full":
        cmd_full(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "status":
        cmd_status()


def cmd_status() -> None:
    """Print current experiment statistics and baselines."""
    fast_baseline = load_fast_baseline()
    has_curve = Path(BASELINE_CURVE_FILE).exists()
    successes, failures = count_recent_stats()

    print(f"{'=' * 60}")
    print("AUTORESEARCH STATUS")
    print(f"{'=' * 60}")
    print(f"Fast baseline BPB: {fast_baseline:.4f}" if fast_baseline else "Fast baseline: not set")
    print(f"Baseline loss curve: {'available' if has_curve else 'not set (run full first)'}")
    print(f"Successes since last full: {successes} (calibrate at {RECALIBRATE_EVERY})")
    print(f"Consecutive failures: {failures} (calibrate at {FAILURE_RECALIBRATE})")

    # Count total experiments
    total = 0
    if Path(EXPERIMENTS_FILE).exists():
        with open(EXPERIMENTS_FILE) as f:
            for line in f:
                if line.strip():
                    total += 1
    print(f"Total experiments: {total}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
