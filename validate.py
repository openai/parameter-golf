#!/usr/bin/env python3
"""Validate each technique individually against baseline.

Interruptible: Ctrl+C saves progress. Re-run to resume.

Usage:
    python validate.py [--max-wallclock 30] [--iterations 50]
"""

from __future__ import annotations

import argparse
import sys

from pgolf.config import PENALTY
from pgolf.runner import load_completed_labels, load_results, run_trial, save_result
from pgolf.techniques import TECHNIQUE_TESTS, VALIDATION_BASELINE


def main():
    parser = argparse.ArgumentParser(description="Validate each technique individually")
    parser.add_argument("--max-wallclock", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--results-only", action="store_true", help="Just print results"
    )
    args = parser.parse_args()

    if args.results_only:
        print_summary()
        return

    completed = load_completed_labels()
    total = len(TECHNIQUE_TESTS)

    print(f"\n{'=' * 70}")
    print(f"TECHNIQUE VALIDATION: {total} tests, {len(completed)} already done")
    print(f"Training: {args.iterations} iters, {args.max_wallclock}s wallclock cap")
    print(f"Ctrl+C to stop — progress is saved, re-run to resume")
    print(f"{'=' * 70}\n")

    done = 0
    for label, overrides in TECHNIQUE_TESTS:
        done += 1
        if label in completed:
            print(f"[{done}/{total}] {label}: SKIP (already completed)")
            continue

        config = {**VALIDATION_BASELINE, **overrides}
        config["RUN_ID"] = f"val_{label}"

        print(f"[{done}/{total}] {label}")
        print(f"  Config: {overrides or '(baseline defaults)'}")
        print(f"  Running...", end="", flush=True)

        try:
            result = run_trial(
                config,
                args.max_wallclock,
                args.iterations,
                label=label,
                skip_compile=args.skip_compile,
            )
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at trial {done}/{total} ({label})")
            print(f"Completed {done - 1} trials. Re-run to resume.")
            sys.exit(0)

        save_result(result)
        status = result["status"]
        bpb = result["val_bpb"]
        elapsed = result["elapsed"]
        size = result.get("artifact_size", "?")
        print(f" {status} | bpb={bpb:.4f} | size={size} | {elapsed:.0f}s")

        if status != "OK":
            err = result.get("error", "")
            if err:
                print(f"  Error: {err[:200]}")

    print_summary()


def print_summary():
    results = load_results()
    if not results:
        print("No results yet.")
        return

    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Label':<30} {'BPB':>8} {'Size':>12} {'Status':>8}")
    print("-" * 70)

    baseline_bpb = None
    for r in sorted(results, key=lambda x: x.get("val_bpb", 99)):
        bpb = r.get("val_bpb", 99)
        label = r.get("label", "?")
        size = r.get("artifact_size", "?")
        status = r.get("status", "?")
        if label == "baseline":
            baseline_bpb = bpb
        delta = ""
        if baseline_bpb is not None and label != "baseline" and bpb < PENALTY:
            delta = f" ({bpb - baseline_bpb:+.4f})"
        print(f"{label:<30} {bpb:>8.4f} {str(size):>12} {status:>8}{delta}")


if __name__ == "__main__":
    main()
