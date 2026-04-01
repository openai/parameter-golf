#!/usr/bin/env python3
"""
Automated experiment sweep for Parameter Golf.

Usage:
    python3 sweep.py                    # Run all pending experiments
    python3 sweep.py --dry-run          # Show what would run without executing
    python3 sweep.py --max-experiments 5 # Run at most 5 experiments

Reads experiment configs, skips already-completed ones (by checking experiments.csv),
runs each via run_experiment.sh, and reports results.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent
RESULTS_FILE = REPO_DIR / "experiments.csv"
SCRIPT_BASELINE = "train_gpt_mlx.py"
SCRIPT_RECURRENT = "train_gpt_mlx_recurrent.py"


def define_experiments():
    """Define the sweep grid. Each experiment is a dict of env vars + a name."""
    experiments = []

    # --- Architecture sweep for recurrent model ---
    # Vary unique layers and recurrences
    for unique in [2, 3, 4]:
        for recur in [2, 3, 4, 5]:
            effective = unique * recur
            if effective < 6 or effective > 20:
                continue
            for dim in [640, 768, 896]:
                for heads in [8, 12]:
                    if dim % heads != 0:
                        continue
                    # Rough param estimate to skip configs that won't fit
                    kv_heads = 4
                    hd = dim // heads
                    kv_dim = kv_heads * hd
                    mlp = dim * 2
                    block_p = dim*dim + dim*kv_dim*2 + dim*dim + dim*mlp + mlp*dim
                    total_p = unique * block_p + 1024 * dim
                    est_compressed = int(total_p * 0.6)  # conservative estimate
                    if est_compressed > 15_500_000:  # leave margin
                        continue

                    name = f"recur_{unique}x{recur}_d{dim}_h{heads}"
                    experiments.append({
                        "name": name,
                        "env": {
                            "SCRIPT": SCRIPT_RECURRENT,
                            "NUM_UNIQUE_LAYERS": str(unique),
                            "NUM_RECURRENCES": str(recur),
                            "MODEL_DIM": str(dim),
                            "NUM_HEADS": str(heads),
                            "NUM_KV_HEADS": "4",
                            "ITERATIONS": "200",
                        },
                        "est_params": total_p,
                        "est_compressed": est_compressed,
                    })

    # Sort by estimated params (try smaller first — faster)
    experiments.sort(key=lambda e: e["est_params"])
    return experiments


def get_completed():
    """Read experiments.csv and return set of completed experiment names."""
    if not RESULTS_FILE.exists():
        return set()
    completed = set()
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(row["experiment"])
    return completed


def run_experiment(exp):
    """Run a single experiment via run_experiment.sh."""
    name = exp["name"]
    env_vars = exp["env"]

    script = env_vars.pop("SCRIPT", SCRIPT_BASELINE)
    iterations = env_vars.pop("ITERATIONS", "200")

    cmd = [str(REPO_DIR / "run_experiment.sh"), name]
    cmd.append(f"ITERATIONS={iterations}")
    for k, v in env_vars.items():
        cmd.append(f"{k}={v}")

    env = os.environ.copy()
    env["SCRIPT"] = script

    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"Config: {json.dumps(exp['env'], indent=2)}")
    print(f"Est params: {exp['est_params']:,} | Est compressed: {exp['est_compressed']:,}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, env=env, cwd=REPO_DIR)
    return result.returncode == 0


def print_results():
    """Print current leaderboard from experiments.csv."""
    if not RESULTS_FILE.exists():
        print("No results yet.")
        return
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows.sort(key=lambda r: float(r.get("val_bpb", "999")))
    print(f"\n{'='*60}")
    print("LEADERBOARD")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Experiment':<35} {'BPB':<10} {'Params':<12} {'Time':<8}")
    print("-" * 70)
    for i, row in enumerate(rows, 1):
        bpb = row.get("val_bpb", "?")
        params = row.get("params", "?")
        elapsed = row.get("elapsed_sec", "?")
        print(f"{i:<5} {row['experiment']:<35} {bpb:<10} {params:<12} {elapsed:<8}s")


def main():
    dry_run = "--dry-run" in sys.argv
    max_exp = None
    for i, arg in enumerate(sys.argv):
        if arg == "--max-experiments" and i + 1 < len(sys.argv):
            max_exp = int(sys.argv[i + 1])

    experiments = define_experiments()
    completed = get_completed()
    pending = [e for e in experiments if e["name"] not in completed]

    print(f"Total configs: {len(experiments)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending: {len(pending)}")

    if max_exp:
        pending = pending[:max_exp]
        print(f"Will run: {len(pending)} (limited by --max-experiments)")

    if dry_run:
        print("\n--- DRY RUN ---")
        for exp in pending:
            print(f"  {exp['name']}: params={exp['est_params']:,} compressed~{exp['est_compressed']:,}")
        return

    for i, exp in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] ", end="")
        success = run_experiment(exp)
        if not success:
            print(f"FAILED: {exp['name']}")

    print_results()


if __name__ == "__main__":
    main()
