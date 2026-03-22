#!/usr/bin/env python3
"""
Extract final metrics from training logs.
Usage: python extract_metrics.py train_1.log train_2.log train_3.log
"""

import re
import sys
import json
from pathlib import Path
from statistics import mean, stdev


def extract_bpb_and_time(log_file: str) -> tuple[float, float]:
    """Extract final val_bpb and total training time from log file."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find final_roundtrip_exact val_bpb
    bpb_match = re.search(r'final_roundtrip_exact val_loss:[0-9.]+\s+val_bpb:([0-9.]+)', content)
    if not bpb_match:
        print(f"ERROR: Could not find final_roundtrip_exact in {log_file}")
        return None, None
    
    val_bpb = float(bpb_match.group(1))
    
    # Find submission size line to estimate total time
    # Look for last occurrence of "train_time" 
    time_matches = re.findall(r'train_time:([0-9.]+)ms', content)
    if time_matches:
        train_time_ms = float(time_matches[-1])
        train_time_sec = train_time_ms / 1000.0
    else:
        train_time_sec = None
        print(f"WARNING: Could not find train_time in {log_file}")
    
    return val_bpb, train_time_sec


def extract_submission_size(log_file: str) -> int:
    """Extract total submission size in bytes."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    match = re.search(r'Total submission size intq\+\w+:\s+([0-9]+)\s+bytes', content)
    if match:
        return int(match.group(1))
    return None


def main():
    if len(sys.argv) < 4:
        print("Usage: python extract_metrics.py train_1.log train_2.log train_3.log")
        sys.exit(1)
    
    log_files = sys.argv[1:]
    
    results = {}
    bpb_values = []
    time_values = []
    sizes = []
    
    for i, log_file in enumerate(log_files, 1):
        print(f"\nProcessing {log_file}...")
        
        if not Path(log_file).exists():
            print(f"  ERROR: File not found")
            continue
        
        val_bpb, train_time = extract_bpb_and_time(log_file)
        submission_size = extract_submission_size(log_file)
        
        if val_bpb is not None:
            print(f"  Final val_bpb: {val_bpb:.8f}")
            bpb_values.append(val_bpb)
            results[f'run_{i}_val_bpb'] = val_bpb
        
        if train_time is not None:
            print(f"  Training time: {train_time:.1f} seconds")
            time_values.append(train_time)
            results[f'run_{i}_wallclock_sec'] = train_time
        
        if submission_size is not None:
            print(f"  Submission size: {submission_size:,} bytes")
            sizes.append(submission_size)
            results[f'run_{i}_submission_bytes'] = submission_size
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if bpb_values:
        mean_bpb = mean(bpb_values)
        std_bpb = stdev(bpb_values) if len(bpb_values) > 1 else 0.0
        print(f"Mean val_bpb: {mean_bpb:.8f}")
        print(f"Std Dev val_bpb: {std_bpb:.8f}")
        results['mean_val_bpb'] = mean_bpb
        results['std_val_bpb'] = std_bpb
    
    if time_values:
        mean_time = mean(time_values)
        print(f"Mean wallclock: {mean_time:.1f} seconds")
        results['mean_wallclock_sec'] = mean_time
        
        # Check if under 600s limit
        if all(t < 600 for t in time_values):
            print("✓ All runs under 600s limit")
        else:
            print("✗ Some runs exceeded 600s limit!")
            for i, t in enumerate(time_values, 1):
                if t >= 600:
                    print(f"  Run {i}: {t:.1f}s")
    
    if sizes:
        mean_size = mean(sizes)
        print(f"Mean submission size: {mean_size:,.0f} bytes")
        results['mean_submission_bytes'] = mean_size
        
        # Check if under 16MB limit
        if all(s < 16_000_000 for s in sizes):
            print("✓ All submissions under 16MB limit")
        else:
            print("✗ Some submissions exceeded 16MB limit!")
            for i, s in enumerate(sizes, 1):
                if s >= 16_000_000:
                    print(f"  Run {i}: {s:,} bytes")
    
    # Save results to JSON
    output_file = "metrics_summary.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
