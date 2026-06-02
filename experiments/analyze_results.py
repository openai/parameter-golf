#!/usr/bin/env python3
"""
Analyze M4 Experiment Results

Parses all logs from logs/exp_*.txt, extracts BPB and other metrics,
compares against baseline, and generates a summary report.
"""

import re
import json
from pathlib import Path
from typing import Optional

# Baseline from logs/m4_10min_baseline.txt
BASELINE_M4_BPB = 3.2178
BASELINE_M4_STEPS = 71


def extract_metrics(log_text: str) -> dict:
    """Extract key metrics from a training log."""
    metrics = {
        "steps": None,
        "val_bpb_preq": None,  # Pre-quantization
        "val_bpb_postq": None,  # Post int8 quantization
        "artifact_bytes": None,
        "train_time_ms": None,
    }

    # Extract final step count
    step_match = re.search(r"stopping_early:.*?step:(\d+)/", log_text)
    if step_match:
        metrics["steps"] = int(step_match.group(1))

    # Extract pre-quant BPB
    preq_match = re.search(r"step:(\d+)/.*?val_loss:[\d.]+\s+val_bpb:([\d.]+)", log_text)
    if preq_match:
        metrics["val_bpb_preq"] = float(preq_match.group(2))

    # Extract post-quant BPB (the key metric)
    postq_match = re.search(r"final_int8_zlib_roundtrip_exact.*?val_bpb:([\d.]+)", log_text)
    if postq_match:
        metrics["val_bpb_postq"] = float(postq_match.group(1))

    # Extract artifact size
    artifact_match = re.search(r"serialized_model_int8_zlib:(\d+)\s+bytes", log_text)
    if artifact_match:
        metrics["artifact_bytes"] = int(artifact_match.group(1))

    # Extract training time
    time_match = re.search(r"stopping_early:.*?train_time:([\d.]+)ms", log_text)
    if time_match:
        metrics["train_time_ms"] = float(time_match.group(1))

    return metrics


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    logs_dir = project_root / "logs"

    if not logs_dir.exists():
        print(f"Error: logs directory not found at {logs_dir}")
        return

    # Find all experiment logs
    exp_logs = sorted(logs_dir.glob("exp_E*.txt"))

    if not exp_logs:
        print(f"No experiment logs found in {logs_dir}")
        print("Run: bash experiments/run_m4_experiments.sh --fast")
        return

    # Parse all results
    results = []
    for log_path in exp_logs:
        exp_name = log_path.stem.replace("exp_", "")
        log_text = log_path.read_text()
        metrics = extract_metrics(log_text)

        # Compute delta from baseline
        bpb = metrics["val_bpb_postq"]
        delta = bpb - BASELINE_M4_BPB if bpb else None

        results.append({
            "exp_name": exp_name,
            "log_path": str(log_path),
            "metrics": metrics,
            "bpb": bpb,
            "delta": delta,
        })

    if not results:
        print("No valid results found. Check logs for errors.")
        return

    # Sort by BPB
    results.sort(key=lambda x: x["bpb"] if x["bpb"] else float("inf"))

    # Print summary table
    print("\n" + "=" * 100)
    print("PARAMETER GOLF M4 EXPERIMENT RESULTS")
    print("=" * 100)
    print(f"\nBaseline M4: {BASELINE_M4_BPB:.4f} BPB, {BASELINE_M4_STEPS} steps\n")

    print(f"{'Rank':<5} {'Experiment':<25} {'BPB':<10} {'Delta':<10} {'Steps':<8} {'Artifact (MB)':<15} {'Time (min)':<12}")
    print("-" * 100)

    for i, result in enumerate(results, 1):
        exp_name = result["exp_name"]
        bpb = result["bpb"]
        delta = result["delta"]
        metrics = result["metrics"]

        bpb_str = f"{bpb:.4f}" if bpb else "N/A"
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        steps_str = str(metrics["steps"]) if metrics["steps"] else "N/A"
        artifact_str = f"{metrics['artifact_bytes'] / 1e6:.2f}" if metrics["artifact_bytes"] else "N/A"
        time_str = f"{metrics['train_time_ms'] / 60000:.1f}" if metrics["train_time_ms"] else "N/A"

        # Mark improvements
        marker = " ✓" if delta is not None and delta < 0 else ""

        print(f"{i:<5} {exp_name:<25} {bpb_str:<10} {delta_str:<10} {steps_str:<8} {artifact_str:<15} {time_str:<12}{marker}")

    # Print winners
    print("\n" + "=" * 100)
    print("TOP 3 WINNERS (best BPB)")
    print("=" * 100)
    for i, result in enumerate(results[:3], 1):
        if result["bpb"] is None:
            continue
        exp_name = result["exp_name"]
        bpb = result["bpb"]
        delta = result["delta"]
        print(f"\n{i}. {exp_name}")
        print(f"   BPB: {bpb:.4f} (Δ {delta:+.4f} vs baseline)")
        print(f"   Steps: {result['metrics']['steps']}")
        print(f"   Artifact: {result['metrics']['artifact_bytes'] / 1e6:.2f} MB")

    # Generate markdown report
    print("\n" + "=" * 100)
    print("Generating markdown report...")
    print("=" * 100)

    report_path = script_dir / "results_summary.md"
    with open(report_path, "w") as f:
        f.write("# Parameter Golf M4 Experiment Results\n\n")
        f.write(f"**Baseline**: {BASELINE_M4_BPB:.4f} BPB ({BASELINE_M4_STEPS} steps)\n\n")

        f.write("## Results Table\n\n")
        f.write("| Rank | Experiment | BPB | Delta | Steps | Artifact (MB) | Time (min) |\n")
        f.write("|------|------------|-----|-------|-------|---------------|------------|\n")

        for i, result in enumerate(results, 1):
            if result["bpb"] is None:
                continue
            exp_name = result["exp_name"]
            bpb = f"{result['bpb']:.4f}"
            delta = f"{result['delta']:+.4f}"
            steps = result['metrics']['steps'] or "N/A"
            artifact = f"{result['metrics']['artifact_bytes'] / 1e6:.2f}" if result['metrics']['artifact_bytes'] else "N/A"
            time_min = f"{result['metrics']['train_time_ms'] / 60000:.1f}" if result['metrics']['train_time_ms'] else "N/A"
            f.write(f"| {i} | {exp_name} | {bpb} | {delta} | {steps} | {artifact} | {time_min} |\n")

        f.write("\n## Top 3 Winners\n\n")
        for i, result in enumerate(results[:3], 1):
            if result["bpb"] is None:
                continue
            exp_name = result["exp_name"]
            bpb = result["bpb"]
            delta = result["delta"]
            f.write(f"### {i}. {exp_name}\n")
            f.write(f"- **BPB**: {bpb:.4f} (Δ {delta:+.4f})\n")
            f.write(f"- **Steps**: {result['metrics']['steps']}\n")
            f.write(f"- **Artifact**: {result['metrics']['artifact_bytes'] / 1e6:.2f} MB\n")
            f.write(f"- **Log**: {result['log_path']}\n\n")

    print(f"✓ Report saved to {report_path}")


if __name__ == "__main__":
    main()
