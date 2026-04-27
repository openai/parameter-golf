#!/usr/bin/env python3
"""Compare experiment results across runs."""
import os
import re
import sys
from pathlib import Path

def parse_log(log_path: Path) -> dict:
    """Extract key metrics from a training log."""
    metrics = {}
    with open(log_path) as f:
        lines = f.readlines()

    for line in lines:
        # Model params
        m = re.search(r'model_params:(\d+)', line)
        if m:
            metrics['params'] = int(m.group(1))

        # Final validation (last val_loss line)
        m = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)', line)
        if m:
            metrics['final_step'] = int(m.group(1))
            metrics['val_loss'] = float(m.group(2))
            metrics['val_bpb'] = float(m.group(3))

        # Post-quant metrics
        m = re.search(r'final_int[68]_zlib_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+)', line)
        if m:
            metrics['q_val_loss'] = float(m.group(1))
            metrics['q_val_bpb'] = float(m.group(2))

        # Size
        m = re.search(r'Total submission size int[68]\+zlib: (\d+) bytes', line)
        if m:
            metrics['total_bytes'] = int(m.group(1))

        # Step timing
        m = re.search(r'step_avg:([\d.]+)ms', line)
        if m:
            metrics['step_avg_ms'] = float(m.group(1))

        # Train loss at last logged step
        m = re.search(r'step:\d+/\d+ train_loss:([\d.]+)', line)
        if m:
            metrics['last_train_loss'] = float(m.group(1))

    return metrics


def main():
    exp_dir = Path("experiments")
    if not exp_dir.exists():
        print("No experiments directory found.")
        return

    experiments = []
    for d in sorted(exp_dir.iterdir()):
        if d.is_dir():
            log = d / "train.log"
            if log.exists():
                metrics = parse_log(log)
                metrics['name'] = d.name
                experiments.append(metrics)

    if not experiments:
        print("No experiments found.")
        return

    # Sort by val_bpb (or q_val_bpb if available)
    def sort_key(e):
        return e.get('q_val_bpb', e.get('val_bpb', 999))

    experiments.sort(key=sort_key)

    # Print comparison table
    print(f"\n{'Experiment':<30} {'Params':>10} {'Val BPB':>10} {'Q BPB':>10} {'Size MB':>10} {'Steps':>8} {'ms/step':>8}")
    print("-" * 96)

    for exp in experiments:
        name = exp.get('name', '?')[:30]
        params = f"{exp.get('params', 0):,}" if 'params' in exp else '?'
        val_bpb = f"{exp['val_bpb']:.4f}" if 'val_bpb' in exp else '?'
        q_bpb = f"{exp['q_val_bpb']:.4f}" if 'q_val_bpb' in exp else '-'
        size = f"{exp['total_bytes'] / 1e6:.2f}" if 'total_bytes' in exp else '?'
        steps = str(exp.get('final_step', '?'))
        ms = f"{exp['step_avg_ms']:.1f}" if 'step_avg_ms' in exp else '?'

        print(f"{name:<30} {params:>10} {val_bpb:>10} {q_bpb:>10} {size:>10} {steps:>8} {ms:>8}")

    # Show best
    best = experiments[0]
    best_metric = best.get('q_val_bpb', best.get('val_bpb', None))
    if best_metric:
        print(f"\nBest: {best['name']} with BPB = {best_metric:.4f}")
        baseline_bpb = 1.2244
        delta = baseline_bpb - best_metric
        print(f"Delta vs baseline: {delta:+.4f} BPB")


if __name__ == "__main__":
    main()
