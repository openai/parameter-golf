#!/usr/bin/env python3
"""
compute_score.py — Extract and validate metrics from train_gpt.py output.
Immutable during experiments. Parses REAL output format.

Usage:
    python compute_score.py run.log          # DEV or FULL eval
    python compute_score.py quick.log --quick  # Quick screen
"""
import re
import sys


def extract_metrics(log_path: str, quick: bool = False) -> dict:
    """Parse metrics from train_gpt.py log output."""
    metrics = {}
    with open(log_path) as f:
        text = f.read()

    # Final int6 roundtrip (always present on successful run)
    m = re.search(r"final_int6_\w+_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", text)
    if m:
        metrics["roundtrip_val_loss"] = float(m.group(1))
        metrics["roundtrip_val_bpb"] = float(m.group(2))

    # Sliding window (only in FULL mode)
    m = re.search(r"final_int6_sliding_window_exact val_loss:([\d.]+) val_bpb:([\d.]+)", text)
    if m:
        metrics["sliding_val_loss"] = float(m.group(1))
        metrics["sliding_val_bpb"] = float(m.group(2))

    # Artifact size
    m = re.search(r"Total submission size int6\+\w+: (\d+) bytes", text)
    if m:
        metrics["artifact_bytes"] = int(m.group(1))
        metrics["artifact_mb"] = round(int(m.group(1)) / 1_000_000, 1)

    # Peak memory
    m = re.search(r"peak memory allocated: (\d+) MiB", text)
    if m:
        metrics["peak_memory_mib"] = int(m.group(1))
        metrics["peak_memory_gb"] = round(int(m.group(1)) / 1024, 1)

    # Training steps completed
    m = re.search(r"stopping_early: wallclock_cap.*step:(\d+)/(\d+)", text)
    if m:
        metrics["steps_completed"] = int(m.group(1))
        metrics["steps_total"] = int(m.group(2))

    # Last val_bpb during training (useful for quick screen)
    for m_iter in re.finditer(r"val_bpb:([\d.]+)", text):
        metrics["last_val_bpb"] = float(m_iter.group(1))

    # Detect mode from log header
    if "DEV EVAL MODE" in text:
        metrics["mode"] = "dev"
    elif "FULL EVAL MODE" in text:
        metrics["mode"] = "full"
    elif "QUICK SCREEN MODE" in text:
        metrics["mode"] = "quick"

    # Determine the composite score
    if quick:
        metrics["quick_score"] = metrics.get("roundtrip_val_bpb", 999.0)
    else:
        # Prefer sliding window (FULL mode), fall back to roundtrip (DEV mode)
        metrics["composite_score"] = metrics.get(
            "sliding_val_bpb",
            metrics.get("roundtrip_val_bpb", 999.0)
        )

    return metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_score.py <log_file> [--quick]")
        sys.exit(1)

    log_path = sys.argv[1]
    quick = "--quick" in sys.argv

    try:
        metrics = extract_metrics(log_path, quick=quick)
    except FileNotFoundError:
        print(f"ERROR: {log_path} not found")
        sys.exit(1)

    if not metrics:
        print("ERROR: No metrics found in log. Run likely crashed.")
        sys.exit(1)

    # Print all sub-metrics
    for k, v in sorted(metrics.items()):
        print(f"{k}: {v}")

    # Final line: the score (parsed by agent)
    if quick:
        score = metrics.get("quick_score", 999.0)
        print(f"\nquick_score: {score}")
    else:
        score = metrics.get("composite_score", 999.0)
        artifact_mb = metrics.get("artifact_mb", 0.0)
        artifact_ok = "PASS" if metrics.get("artifact_bytes", 99e6) <= 16_000_000 else "FAIL"
        mode = metrics.get("mode", "unknown")
        print(f"\ncomposite_score: {score}")
        print(f"artifact_check: {artifact_ok} ({artifact_mb} MB)")
        print(f"eval_mode: {mode}")
        if mode == "dev":
            print("NOTE: DEV mode uses roundtrip BPB (no sliding window). Run FULL=1 to get final score.")


if __name__ == "__main__":
    main()
