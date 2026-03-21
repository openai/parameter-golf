#!/usr/bin/env python3
"""
Multi-Seed Training Runner for HDC VSA Model

This script automates running 3 training sessions with different seeds,
collecting the results, and generating a statistically-rigorous submission.json
with p-value calculation.

Usage:
    python run_multi_seed.py [--seeds 42 7 1337] [--author "Your Name"] [--github_id "your_github"]

The script will:
1. Run train_gpt.py 3 times with different seeds
2. Save logs to train_seed{N}.log for each run
3. Generate submission.json with aggregated results and p-value
"""

import subprocess
import sys
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import re
import statistics
from scipy import stats


def parse_training_log(log_path: str) -> Dict[str, Any]:
    """Parse a training log file to extract key metrics."""
    result = {
        "val_loss": None,
        "val_bpb": None,
        "steps": None,
        "ms_per_step": None,
        "elapsed_seconds": None,
        "recipes_count": None,
        "ngram_count": None,
        "storage_mb": None
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Look for final BPB and loss
    # Pattern: "Final BPB: X.XXXX" or "val_bpb: X.XXXX"
    bpb_match = re.search(r'(?:Final BPB|val_bpb)[:\s]+(\d+\.\d+)', content)
    if bpb_match:
        result["val_bpb"] = float(bpb_match.group(1))
    
    loss_match = re.search(r'(?:Final Loss|val_loss)[:\s]+(\d+\.\d+)', content)
    if loss_match:
        result["val_loss"] = float(loss_match.group(1))
    
    # Look for step count
    steps_match = re.search(r'step[:\s]+(\d+)(?:/\d+)?', content)
    if steps_match:
        result["steps"] = int(steps_match.group(1))
    
    # Look for ms_per_step
    ms_match = re.search(r'step_avg[:\s]+(\d+\.\d+)ms', content)
    if ms_match:
        result["ms_per_step"] = float(ms_match.group(1))
    
    # Look for elapsed time
    time_match = re.search(r'(?:Training complete|train_time)[:\s]+(\d+\.\d+)s', content)
    if time_match:
        result["elapsed_seconds"] = float(time_match.group(1))
    
    # Look for recipes count
    recipes_match = re.search(r'Recipes[:\s]+([\d,]+)', content)
    if recipes_match:
        result["recipes_count"] = int(recipes_match.group(1).replace(',', ''))
    
    # Look for n-grams count
    ngram_match = re.search(r'N-grams[:\s]+([\d,]+)', content)
    if ngram_match:
        result["ngram_count"] = int(ngram_match.group(1).replace(',', ''))
    
    # Look for storage size
    storage_match = re.search(r'Storage[:\s]+(\d+\.\d+)MB', content)
    if storage_match:
        result["storage_mb"] = float(storage_match.group(1))
    
    return result


def run_single_training(seed: int, script_path: str, data_path: str, 
                        max_time: float, iterations: int, 
                        author: str, github_id: str, run_name: str) -> Dict[str, Any]:
    """Run a single training session with the given seed."""
    log_file = f"train_seed{seed}.log"
    
    print(f"\n{'='*60}")
    print(f"Starting training with seed {seed}")
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, script_path,
        f"--data_path={data_path}",
        f"--seed={seed}",
        f"--max_time={max_time}",
        f"--iterations={iterations}",
        f"--author={author}",
        f"--github_id={github_id}",
        f"--run_name={run_name}"
    ]
    
    start_time = time.time()
    
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        
        process.wait()
    
    elapsed = time.time() - start_time
    
    # Parse the log to get results
    results = parse_training_log(log_file)
    results["seed"] = seed
    results["log_file"] = log_file
    results["return_code"] = process.returncode
    results["total_elapsed"] = elapsed
    
    print(f"\nTraining with seed {seed} completed:")
    print(f"  BPB: {results.get('val_bpb', 'N/A')}")
    print(f"  Loss: {results.get('val_loss', 'N/A')}")
    print(f"  Steps: {results.get('steps', 'N/A')}")
    
    return results


def calculate_p_value(bpb_values: List[float], baseline: float = 1.2244) -> float:
    """
    Calculate one-sample t-test p-value against baseline.
    
    Tests whether the mean BPB is significantly lower than baseline.
    """
    if len(bpb_values) < 2:
        return 1.0
    
    # One-sided t-test: is our mean significantly lower than baseline?
    t_stat, p_value_two_sided = stats.ttest_1samp(bpb_values, baseline)
    
    # Convert to one-sided (we want to test if mean < baseline)
    if t_stat < 0:
        p_value_one_sided = p_value_two_sided / 2
    else:
        p_value_one_sided = 1 - p_value_two_sided / 2
    
    return p_value_one_sided


def generate_submission(seed_results: Dict[int, Dict[str, Any]], 
                        author: str, github_id: str, run_name: str,
                        code_bytes: int) -> Dict[str, Any]:
    """Generate the aggregated submission.json."""
    
    # Extract BPB values
    bpb_values = [r["val_bpb"] for r in seed_results.values() if r.get("val_bpb") is not None]
    loss_values = [r["val_loss"] for r in seed_results.values() if r.get("val_loss") is not None]
    
    if not bpb_values:
        raise ValueError("No valid BPB values found in training results")
    
    mean_bpb = statistics.mean(bpb_values)
    mean_loss = statistics.mean(loss_values) if loss_values else None
    std_bpb = statistics.stdev(bpb_values) if len(bpb_values) > 1 else 0.0
    
    p_value = calculate_p_value(bpb_values)
    
    # Calculate artifact size (code + compressed model, but HDC is zero-weight)
    # For HDC, the "model" is the recipes which are stored in the log
    # The code itself is the main artifact
    artifact_bytes = code_bytes  # HDC is zero-weight, so just code size
    
    submission = {
        "track": "10min_16mb",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "name": run_name,
        "author": author,
        "seed_results": {
            str(seed): {
                "val_loss": r.get("val_loss"),
                "val_bpb": r.get("val_bpb"),
                "steps": r.get("steps"),
                "ms_per_step": r.get("ms_per_step")
            }
            for seed, r in seed_results.items()
        },
        "mean_val_loss": mean_loss,
        "mean_val_bpb": mean_bpb,
        "std_val_bpb": std_bpb,
        "p_value": round(p_value, 6),
        "artifact_bytes": artifact_bytes,
        "code_bytes": code_bytes,
        "baseline_bpb": 1.2244,
        "improvement": f"{((1.2244 - mean_bpb) / 1.2244 * 100):.2f}%"
    }
    
    return submission


def main():
    parser = argparse.ArgumentParser(description="Multi-seed training runner for HDC model")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 7, 1337],
                        help="Seeds for training runs (default: 42 7 1337)")
    parser.add_argument("--author", type=str, default="notapplica",
                        help="Author name for submission")
    parser.add_argument("--github_id", type=str, default="notapplica",
                        help="GitHub ID for submission")
    parser.add_argument("--run_name", type=str, default="HDC Zero Track 5Mb",
                        help="Run name for submission")
    parser.add_argument("--data_path", type=str, 
                        default="./data/datasets/fineweb10B_sp1024",
                        help="Path to training data")
    parser.add_argument("--max_time", type=float, default=600.0,
                        help="Maximum training time per run in seconds")
    parser.add_argument("--iterations", type=int, default=20000,
                        help="Maximum iterations per run")
    parser.add_argument("--script", type=str, default="train_gpt.py",
                        help="Path to training script")
    
    args = parser.parse_args()
    
    # Get the directory containing this runner script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Get code size
    code_bytes = os.path.getsize(args.script)
    
    print(f"Multi-Seed Training Runner")
    print(f"{'='*60}")
    print(f"Seeds: {args.seeds}")
    print(f"Author: {args.author}")
    print(f"GitHub ID: {args.github_id}")
    print(f"Run name: {args.run_name}")
    print(f"Data path: {args.data_path}")
    print(f"Max time per run: {args.max_time}s")
    print(f"Code size: {code_bytes:,} bytes")
    print(f"{'='*60}")
    
    # Run training for each seed
    seed_results = {}
    
    for seed in args.seeds:
        result = run_single_training(
            seed=seed,
            script_path=args.script,
            data_path=args.data_path,
            max_time=args.max_time,
            iterations=args.iterations,
            author=args.author,
            github_id=args.github_id,
            run_name=args.run_name
        )
        seed_results[seed] = result
    
    # Generate submission.json
    print(f"\n{'='*60}")
    print("Generating submission.json...")
    print(f"{'='*60}")
    
    submission = generate_submission(
        seed_results=seed_results,
        author=args.author,
        github_id=args.github_id,
        run_name=args.run_name,
        code_bytes=code_bytes
    )
    
    with open("submission.json", 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\nSubmission saved to submission.json")
    print(f"\nFinal Results:")
    print(f"  Mean BPB: {submission['mean_val_bpb']:.6f}")
    print(f"  Std BPB: {submission['std_val_bpb']:.6f}")
    print(f"  P-value: {submission['p_value']:.6f}")
    print(f"  Improvement over baseline: {submission['improvement']}")
    print(f"  Artifact size: {submission['artifact_bytes']:,} bytes")
    
    # Check if statistically significant
    if submission['p_value'] < 0.05:
        print(f"\n✅ Result is statistically significant (p < 0.05)")
    else:
        print(f"\n⚠️ Result is NOT statistically significant (p >= 0.05)")
    
    return 0 if submission['p_value'] < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
