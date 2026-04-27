#!/usr/bin/env python3
"""
Adaptive experiment runner for Parameter Golf.

Instead of a rigid sweep grid, this script:
1. Reads all past experiment results
2. Analyzes what worked (which directions improved BPB)
3. Proposes the next best experiment to try
4. Runs it
5. Repeats

Usage:
    python3 adaptive_sweep.py                    # Run until stopped
    python3 adaptive_sweep.py --max-experiments 10
    python3 adaptive_sweep.py --dry-run          # Show what it would try next
"""

import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.parent  # our/scripts/ -> repo root
RESULTS_FILE = REPO_DIR / "results" / "experiments.csv"
LOG_FILE = REPO_DIR / "results" / "pipeline_log.txt"
MAX_COMPRESSED_BYTES = 15_500_000  # leave margin under 16MB
COMPRESSION_RATIO = 0.55  # observed from actual runs


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


@dataclass
class ExperimentResult:
    name: str
    script: str
    params: int
    val_bpb: float
    iterations: int
    model_dim: int
    unique_layers: int
    recurrences: int
    num_heads: int
    compressed_bytes: int = 0


def load_results() -> list[ExperimentResult]:
    """Load all past experiment results."""
    if not RESULTS_FILE.exists():
        return []
    results = []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bpb = float(row.get("val_bpb", "999"))
            except (ValueError, TypeError):
                continue

            # Parse config from experiment name or log file
            name = row["experiment"]
            dim = int(row.get("model_dim", 0))
            params = 0
            try:
                params = int(row.get("params", 0))
            except (ValueError, TypeError):
                pass

            # Try to extract unique_layers and recurrences from log
            unique, recur, heads = 3, 3, 12  # defaults
            log_path = REPO_DIR / "results" / "logs" / f"{name}.txt"
            if log_path.exists():
                with open(log_path) as lf:
                    for line in lf:
                        if "unique_layers:" in line:
                            try:
                                unique = int(line.split("unique_layers:")[1].split()[0])
                            except (ValueError, IndexError):
                                pass
                        if "recurrences:" in line:
                            try:
                                recur = int(line.split("recurrences:")[1].split()[0])
                            except (ValueError, IndexError):
                                pass
                        if " heads:" in line:
                            try:
                                heads = int(line.split("heads:")[1].split()[0])
                            except (ValueError, IndexError):
                                pass
                        if "dim:" in line and dim == 0:
                            try:
                                dim = int(line.split("dim:")[1].split()[0])
                            except (ValueError, IndexError):
                                pass
                        if "serialized_model_int8_zlib:" in line and not line.strip().startswith("f\""):
                            try:
                                compressed = int(line.split("int8_zlib:")[1].split()[0])
                            except (ValueError, IndexError):
                                compressed = 0

            compressed = 0
            if log_path.exists():
                with open(log_path) as lf:
                    for line in lf:
                        if line.startswith("serialized_model_int8_zlib:"):
                            try:
                                compressed = int(line.split(":")[1].split()[0])
                            except (ValueError, IndexError):
                                pass

            results.append(ExperimentResult(
                name=name, script=row.get("script", ""), params=params,
                val_bpb=bpb, iterations=int(row.get("iterations", 200)),
                model_dim=dim, unique_layers=unique, recurrences=recur,
                num_heads=heads, compressed_bytes=compressed,
            ))
    return results


def estimate_compressed_size(unique_layers: int, dim: int, heads: int, kv_heads: int = 4) -> int:
    """Estimate compressed model size in bytes."""
    hd = dim // heads
    kv_dim = kv_heads * hd
    mlp = dim * 2
    block_p = dim * dim + dim * kv_dim * 2 + dim * dim + dim * mlp + mlp * dim
    total = unique_layers * block_p + 1024 * dim
    return int(total * COMPRESSION_RATIO)


def fits_budget(unique_layers: int, dim: int, heads: int) -> bool:
    """Check if config fits in 16MB budget."""
    return estimate_compressed_size(unique_layers, dim, heads) <= MAX_COMPRESSED_BYTES


def analyze_and_propose(results: list[ExperimentResult]) -> dict | None:
    """Analyze past results and propose the next experiment."""

    if not results:
        # No results yet — start with a reasonable default
        return {
            "name": "adaptive_3x3_d1024_h8",
            "unique_layers": 3, "recurrences": 3,
            "dim": 1024, "heads": 8,
            "script": "our/models/train_gpt_mlx_recurrent.py",
            "reason": "Starting point: moderate width, 80% budget",
        }

    # Filter to recurrent experiments only (skip baseline)
    recurrent = [r for r in results if "recur" in r.name or "max_" in r.name or "adaptive" in r.name]
    if not recurrent:
        recurrent = results

    # Sort by BPB (best first)
    recurrent.sort(key=lambda r: r.val_bpb)
    best = recurrent[0]
    tried_configs = {(r.unique_layers, r.recurrences, r.model_dim, r.num_heads) for r in results}

    log(f"  Analysis: {len(results)} experiments done, best={best.name} BPB={best.val_bpb:.4f}")
    log(f"  Best config: unique={best.unique_layers} recur={best.recurrences} dim={best.model_dim} heads={best.num_heads}")

    # Analyze trends
    # Group by dimension and find average BPB per dim
    dim_scores = {}
    for r in recurrent:
        dim_scores.setdefault(r.model_dim, []).append(r.val_bpb)
    dim_avg = {d: sum(s) / len(s) for d, s in dim_scores.items()}

    # Group by unique_layers
    layer_scores = {}
    for r in recurrent:
        layer_scores.setdefault(r.unique_layers, []).append(r.val_bpb)
    layer_avg = {l: sum(s) / len(s) for l, s in layer_scores.items()}

    # Group by recurrences
    recur_scores = {}
    for r in recurrent:
        recur_scores.setdefault(r.recurrences, []).append(r.val_bpb)
    recur_avg = {r: sum(s) / len(s) for r, s in recur_scores.items()}

    log(f"  Dim trends: {dict(sorted(dim_avg.items()))}")
    log(f"  Layer trends: {dict(sorted(layer_avg.items()))}")
    log(f"  Recur trends: {dict(sorted(recur_avg.items()))}")

    # Strategy: on limited data (1 shard), comparing across very different param counts
    # is unreliable — larger models look worse because they need more data. So we:
    # 1. Prioritize comparing configs at SIMILAR param counts (fair comparisons)
    # 2. Vary architecture shape (layers × recurrences) more than width
    # 3. Still test a few wider configs but don't over-prioritize them
    best_params = best.params if best.params > 0 else estimate_compressed_size(
        best.unique_layers, best.model_dim, best.num_heads) / COMPRESSION_RATIO

    proposals = []

    def param_similarity_bonus(est_params: int) -> float:
        """Higher priority for configs with similar param count to best (fair comparison on 1 shard)."""
        if best_params <= 0:
            return 0
        ratio = est_params / best_params
        # Configs within 50% of best param count get a bonus
        if 0.7 <= ratio <= 1.5:
            return 0.5
        return 0.0

    # 1. Vary architecture shape at SAME dim as best (fair comparisons)
    for unique_delta in [-1, 1, 2, 3]:
        new_unique = best.unique_layers + unique_delta
        if new_unique < 2 or new_unique > 8:
            continue
        for recur in [2, 3, 4, 5, 6]:
            cfg = (new_unique, recur, best.model_dim, best.num_heads)
            if cfg not in tried_configs and fits_budget(new_unique, best.model_dim, best.num_heads):
                est = estimate_compressed_size(new_unique, best.model_dim, best.num_heads)
                est_params = est / COMPRESSION_RATIO
                proposals.append({
                    "name": f"adaptive_{new_unique}x{recur}_d{best.model_dim}_h{best.num_heads}",
                    "unique_layers": new_unique, "recurrences": recur,
                    "dim": best.model_dim, "heads": best.num_heads,
                    "script": "our/models/train_gpt_mlx_recurrent.py",
                    "reason": f"Shape exploration: {new_unique} unique × {recur} recur at d{best.model_dim}, {est/1e6:.1f}MB",
                    "priority": 1.5 + param_similarity_bonus(est_params),
                })

    # 2. Vary recurrence count at best unique layers + dim (cheapest comparison)
    for recur_delta in [-1, 1, 2, 3]:
        new_recur = best.recurrences + recur_delta
        if new_recur < 1 or new_recur > 8:
            continue
        cfg = (best.unique_layers, new_recur, best.model_dim, best.num_heads)
        if cfg not in tried_configs and fits_budget(best.unique_layers, best.model_dim, best.num_heads):
            est = estimate_compressed_size(best.unique_layers, best.model_dim, best.num_heads)
            proposals.append({
                "name": f"adaptive_{best.unique_layers}x{new_recur}_d{best.model_dim}_h{best.num_heads}",
                "unique_layers": best.unique_layers, "recurrences": new_recur,
                "dim": best.model_dim, "heads": best.num_heads,
                "script": "our/models/train_gpt_mlx_recurrent.py",
                "reason": f"{'More' if recur_delta > 0 else 'Fewer'} recurrences ({new_recur}), same params",
                "priority": 1.8,  # highest — same params, pure architecture comparison
            })

    # 3. Try different head counts at best config (same params, different attention pattern)
    for heads in [4, 8, 12, 16]:
        if best.model_dim % heads != 0 or heads == best.num_heads:
            continue
        cfg = (best.unique_layers, best.recurrences, best.model_dim, heads)
        if cfg not in tried_configs:
            est = estimate_compressed_size(best.unique_layers, best.model_dim, heads)
            proposals.append({
                "name": f"adaptive_{best.unique_layers}x{best.recurrences}_d{best.model_dim}_h{heads}",
                "unique_layers": best.unique_layers, "recurrences": best.recurrences,
                "dim": best.model_dim, "heads": heads,
                "script": "our/models/train_gpt_mlx_recurrent.py",
                "reason": f"Different head count ({heads}), ~same params",
                "priority": 1.6,
            })

    # 4. Try moderately wider/narrower (one step at a time, not huge jumps)
    for dim_step in [-128, 128]:
        new_dim = best.model_dim + dim_step
        if new_dim < 256:
            continue
        for heads in [8, 12, 16]:
            if new_dim % heads != 0:
                continue
            cfg = (best.unique_layers, best.recurrences, new_dim, heads)
            if cfg not in tried_configs and fits_budget(best.unique_layers, new_dim, heads):
                est = estimate_compressed_size(best.unique_layers, new_dim, heads)
                est_params = est / COMPRESSION_RATIO
                proposals.append({
                    "name": f"adaptive_{best.unique_layers}x{best.recurrences}_d{new_dim}_h{heads}",
                    "unique_layers": best.unique_layers, "recurrences": best.recurrences,
                    "dim": new_dim, "heads": heads,
                    "script": "our/models/train_gpt_mlx_recurrent.py",
                    "reason": f"Width step ({'+' if dim_step > 0 else ''}{dim_step}), {est/1e6:.1f}MB",
                    "priority": 0.8 + param_similarity_bonus(est_params),
                })

    # 5. A few max-budget configs (lower priority — unfair comparison on 1 shard but useful for H100 planning)
    max_budget_configs = [
        (3, 3, 1152, 12), (4, 3, 896, 8), (5, 2, 896, 16),
        (6, 2, 768, 12), (4, 2, 1024, 8),
    ]
    for unique, recur, dim, heads in max_budget_configs:
        cfg = (unique, recur, dim, heads)
        if cfg not in tried_configs and fits_budget(unique, dim, heads):
            est = estimate_compressed_size(unique, dim, heads)
            proposals.append({
                "name": f"adaptive_{unique}x{recur}_d{dim}_h{heads}",
                "unique_layers": unique, "recurrences": recur,
                "dim": dim, "heads": heads,
                "script": "our/models/train_gpt_mlx_recurrent.py",
                "reason": f"Max-budget config (for H100 planning), {est/1e6:.1f}MB",
                "priority": 0.3,  # low priority — not fair on 1 shard
            })

    # 5. If best was with enhancements not yet tested, try enhanced version
    best_enhanced_name = f"adaptive_enhanced_{best.unique_layers}x{best.recurrences}_d{best.model_dim}"
    if not any(r.name == best_enhanced_name for r in results):
        proposals.append({
            "name": best_enhanced_name,
            "unique_layers": best.unique_layers, "recurrences": best.recurrences,
            "dim": best.model_dim, "heads": best.num_heads,
            "script": "our/models/train_gpt_mlx_enhanced.py",
            "reason": f"Enhanced version of current best (smear gate + backout)",
            "priority": 2.0,  # high priority — test enhancements on best
            "extra_env": {"USE_SMEAR_GATE": "1", "USE_BACKOUT": "1"},
        })

    # 6. If we have a good enhanced config, try longer training
    enhanced_results = [r for r in results if "enhanced" in r.name and r.iterations == 200]
    if enhanced_results:
        best_enh = min(enhanced_results, key=lambda r: r.val_bpb)
        longer_name = f"adaptive_enhanced_500iter_{best_enh.unique_layers}x{best_enh.recurrences}_d{best_enh.model_dim}"
        if not any(r.name == longer_name for r in results):
            proposals.append({
                "name": longer_name,
                "unique_layers": best_enh.unique_layers, "recurrences": best_enh.recurrences,
                "dim": best_enh.model_dim, "heads": best_enh.num_heads,
                "script": "our/models/train_gpt_mlx_enhanced.py",
                "reason": f"Longer training on best enhanced config",
                "priority": 1.5,
                "extra_env": {"USE_SMEAR_GATE": "1", "USE_BACKOUT": "1"},
                "iterations": 500,
            })

    if not proposals:
        log("  No more proposals — search space exhausted!")
        return None

    # Sort by priority (higher = try first)
    proposals.sort(key=lambda p: -p.get("priority", 0))

    chosen = proposals[0]
    log(f"  Proposed: {chosen['name']} — {chosen['reason']}")
    log(f"  ({len(proposals) - 1} other candidates in queue)")
    return chosen


def run_experiment(config: dict) -> bool:
    """Run a single experiment."""
    name = config["name"]
    script = config["script"]
    iters = config.get("iterations", 200)

    cmd_args = [
        str(REPO_DIR / "our" / "scripts" / "run_experiment.sh"), name,
        f"ITERATIONS={iters}",
        f"NUM_UNIQUE_LAYERS={config['unique_layers']}",
        f"NUM_RECURRENCES={config['recurrences']}",
        f"MODEL_DIM={config['dim']}",
        f"NUM_HEADS={config['heads']}",
        f"NUM_KV_HEADS=4",
    ]

    # Add extra env vars (for enhanced script)
    for k, v in config.get("extra_env", {}).items():
        cmd_args.append(f"{k}={v}")

    env = os.environ.copy()
    env["SCRIPT"] = script

    result = subprocess.run(cmd_args, env=env, cwd=REPO_DIR)
    return result.returncode == 0


def print_leaderboard(results: list[ExperimentResult]):
    """Print current leaderboard."""
    results.sort(key=lambda r: r.val_bpb)
    log("")
    log("Current Leaderboard:")
    log(f"  {'Rank':<5} {'Experiment':<45} {'BPB':<12} {'Dim':<6} {'Config':<10} {'Size':<10}")
    log(f"  {'-'*88}")
    for i, r in enumerate(results[:15], 1):
        config = f"{r.unique_layers}x{r.recurrences}"
        size = f"{r.compressed_bytes/1e6:.1f}MB" if r.compressed_bytes else "?"
        log(f"  {i:<5} {r.name:<45} {r.val_bpb:<12.6f} {r.model_dim:<6} {config:<10} {size:<10}")


def main():
    dry_run = "--dry-run" in sys.argv
    max_exp = None
    for i, arg in enumerate(sys.argv):
        if arg == "--max-experiments" and i + 1 < len(sys.argv):
            max_exp = int(sys.argv[i + 1])

    log("")
    log("=========================================")
    log("ADAPTIVE EXPERIMENT RUNNER")
    log("=========================================")

    experiment_count = 0
    while max_exp is None or experiment_count < max_exp:
        results = load_results()
        print_leaderboard(results)

        log("")
        log(f"--- Proposing experiment #{experiment_count + 1} ---")
        proposal = analyze_and_propose(results)

        if proposal is None:
            log("Search space exhausted. Stopping.")
            break

        if dry_run:
            log(f"DRY RUN — would run: {proposal['name']}")
            break

        log(f"Running: {proposal['name']}")
        success = run_experiment(proposal)

        if success:
            # Read back the result
            new_results = load_results()
            new = [r for r in new_results if r.name == proposal["name"]]
            if new:
                log(f"Result: BPB={new[0].val_bpb:.6f}")
                best = min(new_results, key=lambda r: r.val_bpb)
                if new[0].name == best.name:
                    log("*** NEW BEST! ***")
        else:
            log(f"FAILED: {proposal['name']}")

        experiment_count += 1

    # Final summary
    results = load_results()
    print_leaderboard(results)

    # Full training run on the best config
    if not dry_run:
        run_full_training(results)

    log("")
    log("Done.")


def run_full_training(results: list[ExperimentResult]):
    """Download more data shards and run a full training on the best config."""
    best = min(results, key=lambda r: r.val_bpb)
    full_name = f"full_train_{best.unique_layers}x{best.recurrences}_d{best.model_dim}_h{best.num_heads}"

    # Skip if already done
    if any(r.name == full_name for r in results):
        log(f"Full training {full_name} already completed. Skipping.")
        return

    log("")
    log("=========================================")
    log("FULL TRAINING RUN")
    log("=========================================")
    log(f"Best config: {best.name} BPB={best.val_bpb:.6f}")
    log(f"Config: unique={best.unique_layers} recur={best.recurrences} dim={best.model_dim} heads={best.num_heads}")

    # Download 10 shards if we only have 1
    data_dir = REPO_DIR / "data" / "datasets" / "fineweb10B_sp1024"
    existing_shards = len(list(data_dir.glob("fineweb_train_*.bin")))
    if existing_shards < 10:
        log(f"Currently have {existing_shards} shard(s). Downloading 10 shards...")
        dl_result = subprocess.run(
            ["python3", str(REPO_DIR / "data" / "cached_challenge_fineweb.py"),
             "--variant", "sp1024", "--train-shards", "10"],
            cwd=REPO_DIR,
        )
        if dl_result.returncode != 0:
            log("WARNING: Failed to download additional shards. Running with existing data.")
        else:
            new_shards = len(list(data_dir.glob("fineweb_train_*.bin")))
            log(f"Now have {new_shards} shard(s).")
    else:
        log(f"Already have {existing_shards} shard(s). Good.")

    # Determine the script — use recurrent unless enhanced was the best
    script = "our/models/train_gpt_mlx_recurrent.py"
    if "enhanced" in best.name:
        script = "our/models/train_gpt_mlx_enhanced.py"

    log(f"Starting full training: {full_name} (5000 iters, script={script})")
    log("This will take a few hours...")

    config = {
        "name": full_name,
        "unique_layers": best.unique_layers,
        "recurrences": best.recurrences,
        "dim": best.model_dim,
        "heads": best.num_heads,
        "script": script,
        "iterations": 5000,
        "extra_env": {"MAX_WALLCLOCK_SECONDS": "0"},
    }

    success = run_experiment(config)
    if success:
        new_results = load_results()
        new = [r for r in new_results if r.name == full_name]
        if new:
            log(f"Full training result: BPB={new[0].val_bpb:.6f}")
            log("This is our best projection for H100 performance.")
    else:
        log(f"FAILED: {full_name}")


if __name__ == "__main__":
    main()
