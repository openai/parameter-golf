#!/usr/bin/env python3
"""Automated causal research pipeline.

Runs the full discovery-adjust cycle end-to-end:
  Cycle 0: Extract records → Build DAG → Identifiability check → Run diagnostics
  Cycle 1-N: Auto-select intervention → Screen experiment → Analyze → Update DAG → Repeat

Designed for MLX screening runs (short training) to find promising interventions
before committing H100 time.

Usage:
  # Full auto pipeline with 4 screening cycles
  python scripts/causal/run_pipeline.py --max-cycles 4

  # Quick screen: 2 cycles, 200 iterations per run
  python scripts/causal/run_pipeline.py --max-cycles 2 --screen-iterations 200

  # Resume from existing cycle_0
  python scripts/causal/run_pipeline.py --resume-from 1

  # Dry run (no actual training, use mock results)
  python scripts/causal/run_pipeline.py --dry-run
"""
import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
SCRIPTS = REPO_ROOT / "scripts" / "causal"
RESULTS = REPO_ROOT / "results" / "causal"

# In-process trainer (avoids subprocess overhead)
sys.path.insert(0, str(REPO_ROOT))
from scripts.causal.inprocess_trainer import (
    SharedTrainingContext,
    create_shared_context,
    train_single_run,
)

# =========================================================================
# Intervention Search Space
# =========================================================================
# Maps DAG node names to concrete env_override experiments.
# Each entry: list of {label, overrides} representing values to try.
INTERVENTION_SEARCH_SPACE = {
    "num_layers": [
        {"label": "7L", "overrides": {"NUM_LAYERS": "7"}},
        {"label": "11L", "overrides": {"NUM_LAYERS": "11"}},
        {"label": "13L", "overrides": {"NUM_LAYERS": "13"}},
    ],
    "mlp_mult": [
        {"label": "1x_MLP", "overrides": {"MLP_MULT": "1"}},
        {"label": "3x_MLP", "overrides": {"MLP_MULT": "3"}},
        {"label": "4x_MLP", "overrides": {"MLP_MULT": "4"}},
    ],
    "attention_variant": [
        {"label": "fewer_kv_heads", "overrides": {"NUM_KV_HEADS": "2"}},
        {"label": "more_heads", "overrides": {"NUM_HEADS": "16", "NUM_KV_HEADS": "4"}},
    ],
    "rope_variant": [
        {"label": "rope_base_1000", "overrides": {"ROPE_BASE": "1000.0"}},
        {"label": "rope_base_100000", "overrides": {"ROPE_BASE": "100000.0"}},
    ],
    "weight_avg_method": [
        {"label": "higher_momentum", "overrides": {"MUON_MOMENTUM": "0.98"}},
        {"label": "lower_momentum", "overrides": {"MUON_MOMENTUM": "0.90"}},
    ],
    "quant_method": [
        # Quantization is post-training — test via softcap/precision proxies
        {"label": "softcap_20", "overrides": {"LOGIT_SOFTCAP": "20.0"}},
        {"label": "softcap_50", "overrides": {"LOGIT_SOFTCAP": "50.0"}},
    ],
    "model_dim": [
        {"label": "dim_384", "overrides": {"MODEL_DIM": "384", "NUM_HEADS": "6", "NUM_KV_HEADS": "3"}},
        {"label": "dim_640", "overrides": {"MODEL_DIM": "640", "NUM_HEADS": "10", "NUM_KV_HEADS": "5"}},
    ],
}

# Baseline config (9L, 512-dim, default everything)
BASELINE_CONFIG = {
    "script": "train_gpt_mlx.py",
    "env_overrides": {},
    "description": "Baseline (9L, 512-dim, default config)",
}


# =========================================================================
# Pipeline Steps
# =========================================================================

def run_script(script_name, args_list, timeout=600):
    """Run a causal script and return (success, stdout)."""
    cmd = [PYTHON, str(SCRIPTS / script_name)] + args_list
    log.info("Running: %s", " ".join(cmd[-4:]))  # last 4 args for brevity
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT)
        )
        if result.returncode != 0:
            log.error("FAILED: %s\nstderr: %s", script_name, result.stderr[-500:])
            return False, result.stderr
        return True, result.stdout
    except subprocess.TimeoutExpired:
        log.error("TIMEOUT: %s after %ds", script_name, timeout)
        return False, "timeout"


def cycle_0(screen_iters, dry_run=False):
    """Run initial discovery: extract → DAG → identifiability."""
    log.info("=" * 60)
    log.info("CYCLE 0: Initial Discovery")
    log.info("=" * 60)

    # Step 1: Extract interventions
    interventions_path = RESULTS / "interventions.json"
    ok, _ = run_script("extract_interventions.py", [
        "--input", "records/track_10min_16mb/",
        "--output", str(interventions_path),
    ])
    if not ok:
        return None

    # Step 2: Estimate DAG
    cycle_dir = RESULTS / "cycle_0"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    ok, _ = run_script("estimate_dag.py", [
        "--input", str(interventions_path),
        "--output-dir", str(cycle_dir),
    ])
    if not ok:
        return None

    # Step 3: Identifiability check
    ok, _ = run_script("identifiability_check.py", [
        "--interventions", str(interventions_path),
        "--dag", str(cycle_dir / "causal_dag.json"),
        "--output", str(cycle_dir / "identifiability_report.json"),
    ])
    if not ok:
        return None

    # Load results
    with open(cycle_dir / "causal_dag.json") as f:
        dag = json.load(f)
    with open(cycle_dir / "identifiability_report.json") as f:
        ident = json.load(f)

    log.info("DAG: %d nodes, %d edges, FCI degenerate=%s",
             len(dag["nodes"]), len(dag["edges"]), dag["metadata"]["fci_degenerate"])
    log.info("Identifiability: %d single-var, %d multi-var, recommendation=%s",
             ident["single_variable_records"], ident["multi_variable_records"],
             ident["recommendation"])
    log.info("Unexplored combinations: %d", len(ident["unexplored_combinations"]))

    return dag


def select_interventions(dag, completed_interventions):
    """Select next interventions to test from DAG + search space.

    Returns a list of (label, treatment_config) tuples to screen.
    Prioritizes: DAG recommendation → unexplored edges → search space sweep.
    """
    candidates = []

    # Priority 1: DAG's next_intervention recommendation
    rec = dag.get("next_intervention")
    if rec and rec["variable"] in INTERVENTION_SEARCH_SPACE:
        variable = rec["variable"]
        for variant in INTERVENTION_SEARCH_SPACE[variable]:
            label = f"dag_rec_{variable}_{variant['label']}"
            if label not in completed_interventions:
                candidates.append((label, {
                    "script": "train_gpt_mlx.py",
                    "env_overrides": variant["overrides"],
                    "description": f"DAG recommendation: {variable}={variant['label']}",
                }))

    # Priority 2: Sweep uncertain edges from DAG
    for edge in dag.get("edges", []):
        if edge.get("tag") in ("uncertain", "expert_imposed"):
            source_node = edge.get("source", edge.get("from", ""))
            if source_node in INTERVENTION_SEARCH_SPACE and source_node != rec.get("variable", ""):
                for variant in INTERVENTION_SEARCH_SPACE[source_node]:
                    label = f"sweep_{source_node}_{variant['label']}"
                    if label not in completed_interventions:
                        candidates.append((label, {
                            "script": "train_gpt_mlx.py",
                            "env_overrides": variant["overrides"],
                            "description": f"Sweep uncertain edge: {source_node}={variant['label']}",
                        }))

    # Deduplicate by label
    seen = set()
    deduped = []
    for label, config in candidates:
        if label not in seen:
            seen.add(label)
            deduped.append((label, config))

    return deduped


def run_screening_experiment(cycle_num, label, treatment_config, screen_iters, seeds, dry_run=False, ctx=None):
    """Run a single screening experiment (short training)."""
    cycle_dir = RESULTS / f"cycle_{cycle_num}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    # Add screening overrides (short training)
    treatment = json.loads(json.dumps(treatment_config))  # deep copy
    # Set wallclock high enough to not interfere; iterations control stopping
    screen_wallclock = max(screen_iters * 3, 300)  # at least 5 min for MLX compilation
    treatment["env_overrides"]["MAX_WALLCLOCK_SECONDS"] = str(screen_wallclock)
    treatment["env_overrides"]["ITERATIONS"] = str(screen_iters)
    treatment["env_overrides"]["VAL_LOSS_EVERY"] = str(max(screen_iters // 4, 1))
    treatment["env_overrides"]["TRAIN_LOG_EVERY"] = str(max(screen_iters // 10, 1))
    treatment["env_overrides"]["WARMUP_STEPS"] = "1"  # minimal: just prime the compiled graph
    treatment["env_overrides"]["WARMDOWN_ITERS"] = str(screen_iters // 5)
    treatment["env_overrides"]["GRAD_ACCUM_STEPS"] = "1"  # no grad accumulation for screening

    control = json.loads(json.dumps(BASELINE_CONFIG))
    control["env_overrides"]["MAX_WALLCLOCK_SECONDS"] = str(screen_wallclock)
    control["env_overrides"]["ITERATIONS"] = str(screen_iters)
    control["env_overrides"]["VAL_LOSS_EVERY"] = str(max(screen_iters // 4, 1))
    control["env_overrides"]["TRAIN_LOG_EVERY"] = str(max(screen_iters // 10, 1))
    control["env_overrides"]["WARMUP_STEPS"] = "1"
    control["env_overrides"]["WARMDOWN_ITERS"] = str(screen_iters // 5)
    control["env_overrides"]["GRAD_ACCUM_STEPS"] = "1"

    # Write configs
    treatment_path = cycle_dir / f"{label}_treatment.json"
    control_path = cycle_dir / f"{label}_control.json"
    treatment_path.write_text(json.dumps(treatment, indent=2), encoding="utf-8")
    control_path.write_text(json.dumps(control, indent=2), encoding="utf-8")

    raw_runs_path = cycle_dir / f"{label}_raw_runs.json"

    if dry_run:
        # Generate mock results
        import random
        random.seed(42)
        base_bpb = 1.22 + random.gauss(0, 0.005)
        mock_results = {
            "platform": "mlx",
            "seeds": seeds,
            "treatment": {
                "config": treatment,
                "results": [{"seed": s, "val_bpb": base_bpb + random.gauss(-0.003, 0.002),
                              "val_loss": 2.0, "wall_time_s": 10.0,
                              "checkpoint_path": None, "train_log_path": None} for s in seeds],
            },
            "control": {
                "config": control,
                "results": [{"seed": s, "val_bpb": base_bpb + random.gauss(0, 0.002),
                              "val_loss": 2.1, "wall_time_s": 10.0,
                              "checkpoint_path": None, "train_log_path": None} for s in seeds],
            },
        }
        raw_runs_path.write_text(json.dumps(mock_results, indent=2), encoding="utf-8")
        log.info("[DRY RUN] Mock results written for %s", label)
        return raw_runs_path

    # Run in-process (avoids subprocess MLX warmup overhead)
    if ctx is not None:
        log.info("Running in-process (warmup cached: %s)", label)
        try:
            treatment_results = []
            control_results = []
            val_every = max(screen_iters // 4, 1)
            warmup = min(1, screen_iters // 10)  # minimal warmup; cached after first run
            warmdown = screen_iters // 5

            for s in seeds:
                r = train_single_run(ctx, treatment["env_overrides"], seed=s,
                                     iterations=screen_iters, val_loss_every=val_every,
                                     warmup_steps=warmup, warmdown_iters=warmdown)
                treatment_results.append(r)

            for s in seeds:
                r = train_single_run(ctx, control["env_overrides"], seed=s,
                                     iterations=screen_iters, val_loss_every=val_every,
                                     warmup_steps=warmup, warmdown_iters=warmdown)
                control_results.append(r)

            results = {
                "platform": "mlx",
                "seeds": seeds,
                "treatment": {"config": treatment, "results": treatment_results},
                "control": {"config": control, "results": control_results},
            }
            raw_runs_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            return raw_runs_path

        except Exception as e:
            log.warning("In-process failed (%s), falling back to subprocess", e)

    # Subprocess fallback
    seeds_str = ",".join(str(s) for s in seeds)
    n_runs = len(seeds) * 2
    timeout = n_runs * (300 + screen_iters * 2) + 300
    ok, _ = run_script("experiment_runner.py", [
        "--treatment", str(treatment_path),
        "--control", str(control_path),
        "--output", str(raw_runs_path),
        "--seeds", seeds_str,
        "--platform", "mlx",
    ], timeout=timeout)

    if not ok:
        log.error("Experiment failed: %s", label)
        return None

    return raw_runs_path


def analyze_experiment(cycle_num, label, raw_runs_path):
    """Run statistical analysis on experiment results."""
    cycle_dir = RESULTS / f"cycle_{cycle_num}"
    ablation_path = cycle_dir / f"{label}_ablation.json"

    ok, _ = run_script("statistical_analysis.py", [
        "--input", str(raw_runs_path),
        "--output", str(ablation_path),
    ])

    if not ok:
        return None

    with open(ablation_path) as f:
        return json.load(f)


def update_dag(cycle_num):
    """Feed experiment results back into the DAG."""
    cycle_dir = RESULTS / f"cycle_{cycle_num}"
    prev_cycle_dir = RESULTS / f"cycle_{cycle_num - 1}"
    interventions_path = RESULTS / "interventions.json"

    # Find all raw_runs files in this cycle
    raw_runs_files = sorted(cycle_dir.glob("*_raw_runs.json"))
    for rf in raw_runs_files:
        run_script("extract_interventions.py", [
            "--input", "records/track_10min_16mb/",
            "--output", str(interventions_path),
            "--append-experiment", str(rf),
        ])

    # Re-estimate DAG with previous DAG for diff
    prev_dag = prev_cycle_dir / "causal_dag.json"
    dag_args = [
        "--input", str(interventions_path),
        "--output-dir", str(cycle_dir),
    ]
    if prev_dag.exists():
        dag_args += ["--previous-dag", str(prev_dag)]

    ok, _ = run_script("estimate_dag.py", dag_args)
    if not ok:
        return None

    with open(cycle_dir / "causal_dag.json") as f:
        return json.load(f)


# =========================================================================
# Results Ranking
# =========================================================================

def rank_results(all_results):
    """Rank all experiment results by effect size. Returns sorted list."""
    ranked = []
    for r in all_results:
        comparisons = r.get("analysis", {}).get("comparisons", [])
        for comp in comparisons:
            ranked.append({
                "label": r["label"],
                "cycle": r["cycle"],
                "mean_effect": comp.get("mean_effect", 0),
                "p_value": comp.get("p_value", 1.0),
                "p_corrected": comp.get("p_value_corrected", 1.0),
                "decision": comp.get("decision", "null"),
                "ci_lo": comp.get("ci_lo", 0),
                "ci_hi": comp.get("ci_hi", 0),
                "description": r.get("description", ""),
            })

    # Sort by effect size (most negative = biggest improvement)
    ranked.sort(key=lambda x: x["mean_effect"])
    return ranked


def print_leaderboard(ranked):
    """Print a ranked leaderboard of all tested interventions."""
    log.info("")
    log.info("=" * 80)
    log.info("SCREENING LEADERBOARD")
    log.info("=" * 80)
    log.info("%-35s %10s %10s %10s %s", "Intervention", "Effect", "p-value", "Decision", "CI")
    log.info("-" * 80)
    for i, r in enumerate(ranked):
        marker = " ***" if r["decision"] == "confirmed" else (" *" if r["decision"] == "suggestive" else "")
        log.info("%-35s %+10.4f %10.4f %10s [%+.4f, %+.4f]%s",
                 r["label"][:35], r["mean_effect"], r["p_value"],
                 r["decision"], r["ci_lo"], r["ci_hi"], marker)

    confirmed = [r for r in ranked if r["decision"] == "confirmed"]
    suggestive = [r for r in ranked if r["decision"] == "suggestive"]
    log.info("")
    log.info("Summary: %d confirmed, %d suggestive, %d null out of %d tested",
             len(confirmed), len(suggestive),
             len(ranked) - len(confirmed) - len(suggestive), len(ranked))

    if confirmed:
        log.info("")
        log.info("PROTOTYPES READY FOR H100 VALIDATION:")
        for r in confirmed:
            log.info("  %s (effect: %+.4f BPB, p=%.4f)", r["label"], r["mean_effect"], r["p_value"])

    return confirmed, suggestive


# =========================================================================
# Main Pipeline
# =========================================================================

def run_pipeline(max_cycles=4, screen_iters=500, seeds=None, dry_run=False,
                 resume_from=0, max_interventions_per_cycle=3):
    """Run the full automated research pipeline.

    Args:
        max_cycles: Maximum number of experiment cycles (1-4)
        screen_iters: Training iterations per screening run (lower = faster, noisier)
        seeds: List of seeds for paired design (default: [42, 137, 256])
        dry_run: Use mock results instead of actual training
        resume_from: Start from this cycle number (0 = fresh start)
        max_interventions_per_cycle: Max experiments per cycle
    """
    if seeds is None:
        seeds = [42, 137, 256]

    RESULTS.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    all_results = []
    completed_interventions = set()

    # Create shared training context once (avoids per-run data loading)
    ctx = None
    if not dry_run:
        try:
            ctx = create_shared_context(
                data_path="./data/datasets/fineweb10B_sp1024",
                tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model",
                vocab_size=1024,
                train_seq_len=1024,
            )
            log.info("In-process training enabled (shared context loaded)")
        except Exception as e:
            log.warning("Failed to create shared context (%s), using subprocess fallback", e)
            ctx = None

    # Load any existing results from previous cycles
    pipeline_log_path = RESULTS / "pipeline_log.json"
    if pipeline_log_path.exists() and resume_from > 0:
        with open(pipeline_log_path) as f:
            pipeline_state = json.load(f)
        all_results = pipeline_state.get("results", [])
        completed_interventions = set(r["label"] for r in all_results)
        log.info("Resumed: %d previous results loaded", len(all_results))

    # Cycle 0: Discovery
    if resume_from <= 0:
        dag = cycle_0(screen_iters, dry_run)
        if dag is None:
            log.error("Cycle 0 failed. Aborting.")
            return
    else:
        dag_path = RESULTS / f"cycle_{resume_from - 1}" / "causal_dag.json"
        if not dag_path.exists():
            dag_path = RESULTS / "cycle_0" / "causal_dag.json"
        with open(dag_path) as f:
            dag = json.load(f)
        log.info("Resumed from cycle %d with existing DAG", resume_from)

    # Experiment cycles
    null_streak = 0
    for cycle in range(max(1, resume_from), max_cycles + 1):
        elapsed_hours = (time.time() - start_time) / 3600
        log.info("")
        log.info("=" * 60)
        log.info("CYCLE %d (elapsed: %.1f hours, null streak: %d)", cycle, elapsed_hours, null_streak)
        log.info("=" * 60)

        # Stop conditions
        if null_streak >= 3:
            log.info("STOP: 3 consecutive null results. Pivoting to engineering fallback.")
            break

        # Select interventions
        candidates = select_interventions(dag, completed_interventions)
        if not candidates:
            log.info("STOP: No more interventions to test.")
            break

        # Limit per cycle
        batch = candidates[:max_interventions_per_cycle]
        log.info("Testing %d interventions this cycle: %s",
                 len(batch), [b[0] for b in batch])

        cycle_has_confirmed = False
        for label, treatment_config in batch:
            log.info("")
            log.info("--- Experiment: %s ---", label)
            log.info("Config: %s", treatment_config.get("description", ""))

            # Run screening experiment
            raw_runs_path = run_screening_experiment(
                cycle, label, treatment_config, screen_iters, seeds, dry_run, ctx=ctx
            )
            if raw_runs_path is None:
                log.warning("Skipping %s (experiment failed)", label)
                completed_interventions.add(label)
                continue

            # Analyze
            analysis = analyze_experiment(cycle, label, raw_runs_path)
            if analysis is None:
                log.warning("Skipping %s (analysis failed)", label)
                completed_interventions.add(label)
                continue

            # Record result
            result_entry = {
                "label": label,
                "cycle": cycle,
                "description": treatment_config.get("description", ""),
                "treatment_overrides": treatment_config["env_overrides"],
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result_entry)
            completed_interventions.add(label)

            # Check decision
            comparisons = analysis.get("comparisons", [])
            if comparisons:
                decision = comparisons[0].get("decision", "null")
                effect = comparisons[0].get("mean_effect", 0)
                log.info("Result: %s (effect: %+.4f BPB)", decision, effect)
                if decision == "confirmed":
                    cycle_has_confirmed = True
                    null_streak = 0
                    log.info("*** CONFIRMED EFFECT: %s ***", label)
            else:
                log.info("Result: no comparisons produced")

        # Update null streak
        if not cycle_has_confirmed:
            null_streak += 1

        # Update DAG with this cycle's results
        updated_dag = update_dag(cycle)
        if updated_dag is not None:
            dag = updated_dag

        # Save pipeline state
        pipeline_state = {
            "last_cycle": cycle,
            "total_experiments": len(all_results),
            "null_streak": null_streak,
            "elapsed_hours": (time.time() - start_time) / 3600,
            "results": all_results,
        }
        pipeline_log_path.write_text(json.dumps(pipeline_state, indent=2), encoding="utf-8")

        # Early stop on confirmed effect
        confirmed_count = sum(
            1 for r in all_results
            if r.get("analysis", {}).get("comparisons", [{}])[0].get("decision") == "confirmed"
        )
        if confirmed_count >= 2:
            log.info("STOP: 2+ confirmed effects found. Ready for H100 validation.")
            break

    # Final ranking
    ranked = rank_results(all_results)
    confirmed, suggestive = print_leaderboard(ranked)

    # Save final report
    report = {
        "pipeline_complete": True,
        "total_cycles": max(1, resume_from) if not all_results else all_results[-1]["cycle"],
        "total_experiments": len(all_results),
        "elapsed_hours": (time.time() - start_time) / 3600,
        "confirmed_prototypes": confirmed,
        "suggestive_prototypes": suggestive,
        "full_leaderboard": ranked,
    }
    report_path = RESULTS / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("")
    log.info("Pipeline report: %s", report_path)

    return report


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated causal research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with 4 cycles, 500 iterations per screen
  python scripts/causal/run_pipeline.py --max-cycles 4

  # Quick dry run to test the pipeline
  python scripts/causal/run_pipeline.py --dry-run --max-cycles 2

  # Short screening (200 iters) for fast iteration
  python scripts/causal/run_pipeline.py --screen-iterations 200 --max-cycles 3

  # Resume from cycle 2
  python scripts/causal/run_pipeline.py --resume-from 2
""")
    parser.add_argument("--max-cycles", type=int, default=4,
                        help="Maximum experiment cycles (default: 4)")
    parser.add_argument("--screen-iterations", type=int, default=500,
                        help="Training iterations per screening run (default: 500)")
    parser.add_argument("--seeds", type=str, default="42,137,256",
                        help="Comma-separated seeds (default: 42,137,256)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock results instead of actual training")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from this cycle number")
    parser.add_argument("--max-per-cycle", type=int, default=3,
                        help="Max interventions per cycle (default: 3)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    report = run_pipeline(
        max_cycles=args.max_cycles,
        screen_iters=args.screen_iterations,
        seeds=seeds,
        dry_run=args.dry_run,
        resume_from=args.resume_from,
        max_interventions_per_cycle=args.max_per_cycle,
    )

    if report:
        confirmed = report.get("confirmed_prototypes", [])
        if confirmed:
            print(f"\n{'='*60}")
            print(f"READY FOR H100: {len(confirmed)} confirmed prototypes")
            for p in confirmed:
                print(f"  {p['label']}: {p['mean_effect']:+.4f} BPB (p={p['p_value']:.4f})")
            print(f"{'='*60}")
        else:
            print("\nNo confirmed effects. Consider engineering fallback or more cycles.")


if __name__ == "__main__":
    main()
