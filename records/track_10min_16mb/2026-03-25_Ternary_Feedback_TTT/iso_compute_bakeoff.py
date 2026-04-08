#!/usr/bin/env python3
"""
Iso-Compute Bake-Off: Which architecture learns best per step?

Every config is tuned so that ms/step matches the baseline (plain ternary).
Then all configs race for the same number of steps. This removes the MLX
compiler speed bias and answers: "given equal compute, which architecture
extracts the most learning?"

Two phases:
  Phase 1 — CALIBRATE: Binary-search MODEL_DIM for each non-baseline config
            until ms/step matches baseline ±tolerance.
  Phase 2 — RACE:      Run all configs for RACE_STEPS steps, eval every
            EVAL_EVERY steps. Full eval pipeline at final step.

Usage:
    python iso_compute_bakeoff.py                  # full run
    python iso_compute_bakeoff.py --calibrate-only # just Phase 1
    python iso_compute_bakeoff.py --race-only      # skip calibration, use cached dims
    python iso_compute_bakeoff.py --race-steps 1000 --eval-every 100

Results are written to bakeoff_results.json and bakeoff_report.md.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CALIBRATION_STEPS = 25          # steps to measure ms/step
CALIBRATION_WARMUP = 5          # skip first N steps (compilation warmth)
TOLERANCE_PCT = 0.08            # ±8% of baseline ms/step
DIM_SEARCH_MIN = 96
DIM_SEARCH_MAX = 512
CALIBRATION_CACHE = SCRIPT_DIR / "bakeoff_calibration.json"

DEFAULT_RACE_STEPS = 2000
DEFAULT_EVAL_EVERY = 200

# Cap validation tokens per eval to avoid OOM on 16GB unified memory.
# Full val set is ~62M tokens; 2M is enough for a stable BPB estimate.
MAX_VAL_TOKENS_RACE = 2_000_000

# ---------------------------------------------------------------------------
# Base environment (shared by all configs)
# ---------------------------------------------------------------------------
BASE_ENV = {
    "DATA_PATH": os.environ.get("DATA_PATH", "../../../data/datasets/fineweb10B_sp1024"),
    "TOKENIZER_PATH": os.environ.get("TOKENIZER_PATH", "../../../data/tokenizers/fineweb_1024_bpe.model"),
    "VOCAB_SIZE": "1024",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "2",
    "MLP_MULT": "4",
    "EMBED_DIM": "128",
    "TRAIN_BATCH_TOKENS": "16384",
    "GRAD_ACCUM_STEPS": "2",
    "TRAIN_SEQ_LEN": "1024",
    "SEED": "42",
    "TRAIN_LOG_EVERY": "1",
    "WARMUP_STEPS": "0",
    # Eval settings (used in race phase)
    "SLIDING_EVAL": "1",
    "SLIDING_EVAL_STRIDE": "64",
    "TEMP_SCALING": "1",
    "NGRAM_CACHE_ENABLED": "1",
    "NGRAM_MAX_ORDER": "5",
    "NGRAM_ALPHA_BASE": "0.05",
    "NGRAM_ALPHA_SCALE": "0.55",
    "NGRAM_ENTROPY_CENTER": "4.0",
    "TTT_ENABLED": "1",
    "TTT_SCOPE": "feedback",
    "TTT_LR": "0.002",
    "TTT_EPOCHS": "1",
    "TTT_CHUNK_TOKENS": "32768",
    "TTT_MOMENTUM": "0.9",
    "TTT_BATCH_SEQS": "32",
    "TTT_GRAD_CLIP": "1.0",
}

# ---------------------------------------------------------------------------
# Architecture configs
# ---------------------------------------------------------------------------
# Each config defines:
#   "fixed": env vars that never change (architecture identity flags)
#   "is_baseline": True for the reference config
#   "scale_dims": function(model_dim, num_heads) -> dict of derived env vars
#                 Called every time MODEL_DIM is tried during calibration.
#                 Use this to keep auxiliary dims proportional to MODEL_DIM.
#
# Scaling rules rationale:
#   FEEDBACK_DIM   : ~model_dim/8, min 16, power-of-2 snap
#   BIGRAM_HASH_DIM: ~model_dim/4, min 32, power-of-2 snap
#   CAPSULE_DIM    : ~model_dim/8, min 16, power-of-2 snap
#   CAPSULE_NUM    : fixed at 8 (it's a count, not a width)
#   KOOPMAN_RANK   : fixed (architectural identity, not a width)
#   PARTIAL_ROPE_DIMS: min(8, head_dim//4) — must fit inside head_dim
#   EMBED_DIM      : ~model_dim/2, min 64, power-of-2 snap (factorized embed)

NUM_HEADS = int(BASE_ENV["NUM_HEADS"])


def _p2(x, lo=16):
    """Snap x to nearest power of 2, floored at lo."""
    x = max(lo, x)
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def _scale_common(model_dim):
    """Derived dims shared across all configs that use auxiliary features."""
    head_dim = model_dim // NUM_HEADS
    return {
        "FEEDBACK_DIM":    str(_p2(model_dim // 8, lo=16)),
        "BIGRAM_HASH_DIM": str(_p2(model_dim // 4, lo=32)),
        "CAPSULE_DIM":     str(_p2(model_dim // 8, lo=16)),
        "PARTIAL_ROPE_DIMS": str(max(4, min(8, head_dim // 4))),
        "EMBED_DIM":       str(_p2(model_dim // 2, lo=64)),
    }


CONFIGS = {
    "A_plain_ternary": {
        "description": "Plain ternary transformer, no innovations",
        "fixed": {
            "NUM_LAYERS": "4",
            "MODEL_DIM": "256",
            "EMBED_DIM": "128",
            "FEEDBACK_ENABLED": "0",
            "BIGRAM_HASH_ENABLED": "0",
            "VRL_ENABLED": "0",
            "CAPSULE_ENABLED": "0",
            "KOOPMAN_ENABLED": "0",
            "KOOPMAN_SPECULATOR_ENABLED": "0",
            "XSA_START_LAYER": "99",
            "EMA_ENABLED": "0",
            "ARCHITECTURE": "transformer",
            "SHARED_BLOCKS": "0",
        },
        "is_baseline": True,
    },
    "B_feedback_engram": {
        "description": "Transformer + Feedback + Engram + VRL + XSA",
        "fixed": {
            "NUM_LAYERS": "4",
            # Feedback
            "FEEDBACK_ENABLED": "1",
            "FEEDBACK_SKETCH_TOKENS": "2",
            "FEEDBACK_PASSES": "1",
            "EVAL_FEEDBACK_PASSES": "2",
            "FEEDBACK_EVERY": "2",
            # Engram (bigram hash)
            "BIGRAM_HASH_ENABLED": "1",
            "BIGRAM_HASH_BUCKETS": "4096",
            # Regularization tricks (no added compute per step)
            "VRL_ENABLED": "1",
            "VRL_START_LAYER": "2",
            "LN_SCALE_DAMPING": "1",
            "XSA_START_LAYER": "2",
            # No capsules / Koopman in this config
            "CAPSULE_ENABLED": "0",
            "KOOPMAN_ENABLED": "0",
            "KOOPMAN_SPECULATOR_ENABLED": "0",
            "EMA_ENABLED": "0",
            "ARCHITECTURE": "transformer",
            "SHARED_BLOCKS": "0",
        },
        "scale_dims": lambda d: _scale_common(d),
        "dim_default": 256,
    },
    "C_shared_blocks": {
        "description": "B + SharedBlocks(2) + Capsules + Koopman(rank=2)",
        "fixed": {
            "NUM_LAYERS": "4",
            # Feedback
            "FEEDBACK_ENABLED": "1",
            "FEEDBACK_SKETCH_TOKENS": "2",
            "FEEDBACK_PASSES": "1",
            "EVAL_FEEDBACK_PASSES": "2",
            "FEEDBACK_EVERY": "2",
            # Engram
            "BIGRAM_HASH_ENABLED": "1",
            "BIGRAM_HASH_BUCKETS": "4096",
            # Tricks
            "VRL_ENABLED": "1",
            "VRL_START_LAYER": "2",
            "LN_SCALE_DAMPING": "1",
            "XSA_START_LAYER": "2",
            # Capsules + Koopman dynamics in capsule space
            "CAPSULE_ENABLED": "1",
            "CAPSULE_NUM": "8",
            "KOOPMAN_ENABLED": "1",
            "KOOPMAN_RANK": "2",           # low rank — keep it affordable
            "KOOPMAN_SPECULATOR_ENABLED": "0",
            # Shared blocks: the defining feature of this config
            "SHARED_BLOCKS": "2",
            "EMA_ENABLED": "0",
            "ARCHITECTURE": "transformer",
        },
        "scale_dims": lambda d: _scale_common(d),
        "dim_default": 256,
    },
    "D_koopman_koopcaps": {
        "description": "Full Koopman + KoopCaps + Carry + Halt + Speculator",
        "fixed": {
            "NUM_LAYERS": "4",
            # Feedback
            "FEEDBACK_ENABLED": "1",
            "FEEDBACK_SKETCH_TOKENS": "2",
            "FEEDBACK_PASSES": "1",
            "EVAL_FEEDBACK_PASSES": "2",
            "FEEDBACK_EVERY": "2",
            # Engram
            "BIGRAM_HASH_ENABLED": "1",
            "BIGRAM_HASH_BUCKETS": "4096",
            # Tricks
            "VRL_ENABLED": "1",
            "VRL_START_LAYER": "2",
            "LN_SCALE_DAMPING": "1",
            "XSA_START_LAYER": "2",
            # Full Koopman stack — the defining feature of this config
            "CAPSULE_ENABLED": "1",
            "CAPSULE_NUM": "8",
            "KOOPMAN_ENABLED": "1",
            "KOOPMAN_RANK": "4",           # higher rank than C
            "KOOPMAN_SPECULATOR_ENABLED": "1",
            "KOOPMAN_SPECULATOR_STEPS": "3",
            "KOOPMAN_SPECULATOR_WEIGHT": "0.01",
            # Eval-only features (capsule carry + adaptive halt)
            "ADAPTIVE_HALT_ENABLED": "1",
            "ADAPTIVE_HALT_THRESHOLD": "0.05",
            "MAX_EVAL_PASSES": "3",
            "CAPSULE_CARRY_ENABLED": "1",
            "CAPSULE_CARRY_DECAY": "0.8",
            "SHARED_BLOCKS": "0",
            "EMA_ENABLED": "0",
            "ARCHITECTURE": "transformer",
        },
        "scale_dims": lambda d: _scale_common(d),
        "dim_default": 256,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_env(config_name: str, model_dim: int = None, overrides: dict = None) -> dict:
    """Merge BASE_ENV + config fixed vars + scaled dims + overrides.

    If model_dim is given and the config has a scale_dims function,
    derived auxiliary dims (FEEDBACK_DIM, BIGRAM_HASH_DIM, etc.) are
    recomputed proportionally so overhead doesn't dominate at small dims.
    """
    cfg = CONFIGS[config_name]
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(cfg["fixed"])

    # Apply proportional scaling for auxiliary dims
    if model_dim is not None and "scale_dims" in cfg:
        env["MODEL_DIM"] = str(model_dim)
        env.update(cfg["scale_dims"](model_dim))

    if overrides:
        env.update(overrides)
    return env


def run_training(config_name: str, env_overrides: dict, max_steps: int,
                 model_dim: int = None, max_wallclock: float = 0,
                 val_loss_every: int = 0, label: str = "") -> dict:
    """Run train_gpt_mlx.py and parse output. Returns parsed metrics."""
    env = build_env(config_name, model_dim=model_dim, overrides=env_overrides)
    env["ITERATIONS"] = str(max_steps)
    env["MAX_WALLCLOCK_SECONDS"] = str(max_wallclock)
    env["VAL_LOSS_EVERY"] = str(val_loss_every)
    env["RUN_ID"] = f"bakeoff_{config_name}_{label}"

    cmd = ["bash", "run_mlx_reasoner.sh"]
    tag = f"[{config_name}:{label}]"

    print(f"\n{'='*70}")
    print(f"  {tag} steps={max_steps} wallclock={max_wallclock}s")
    print(f"  dim={env.get('MODEL_DIM','?')} layers={env.get('NUM_LAYERS','?')}")
    print(f"{'='*70}\n")

    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(SCRIPT_DIR),
    )

    step_times_ms = []
    val_bpbs = []      # (step, bpb) pairs
    sliding_bpb = None
    ttt_bpb = None
    ngram_bpb = None
    n_params = None
    last_step = 0

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

        # Parse param count
        m = re.search(r'model_params:(\d+)', line)
        if m:
            n_params = int(m.group(1))

        # Parse step timing:  step:N/M loss:L t:Tms step:Sms
        m = re.search(r'step:(\d+)/\d+ loss:[\d.]+ t:\d+ms step:(\d+)ms', line)
        if m:
            s = int(m.group(1))
            ms = int(m.group(2))
            last_step = s
            step_times_ms.append((s, ms))

        # Parse val_bpb during training
        m = re.search(r'val_bpb:([\d.]+)', line)
        if m and last_step > 0:
            val_bpbs.append((last_step, float(m.group(1))))

        # Parse final sliding eval
        m = re.search(r'final_sliding.*val_bpb:([\d.]+)', line)
        if m:
            sliding_bpb = float(m.group(1))

        # Parse TTT eval
        m = re.search(r'(legal_ttt|ttt_sliding).*val_bpb:([\d.]+)', line)
        if m:
            ttt_bpb = float(m.group(2))

        # Parse n-gram cache eval
        m = re.search(r'ngram.*val_bpb:([\d.]+)', line)
        if m:
            ngram_bpb = float(m.group(1))

    proc.wait()

    # Compute mean step time (skip warmup steps)
    warmup = CALIBRATION_WARMUP
    timing = [ms for s, ms in step_times_ms if s > warmup]
    mean_ms = sum(timing) / len(timing) if timing else 0

    return {
        "config": config_name,
        "label": label,
        "n_params": n_params,
        "model_dim": int(env.get("MODEL_DIM", 0)),
        "steps_completed": last_step,
        "mean_step_ms": round(mean_ms, 1),
        "step_times": step_times_ms,
        "val_bpbs": val_bpbs,
        "sliding_bpb": sliding_bpb,
        "ttt_bpb": ttt_bpb,
        "ngram_bpb": ngram_bpb,
    }


# ===================================================================
# Phase 1: CALIBRATE — find MODEL_DIM for each config to match baseline
# ===================================================================
def calibrate_single(config_name: str, target_ms: float, tolerance: float) -> int:
    """Binary search MODEL_DIM to match target ms/step."""
    cfg = CONFIGS[config_name]
    lo, hi = DIM_SEARCH_MIN, DIM_SEARCH_MAX
    best_dim = cfg["dim_default"]
    best_diff = float("inf")

    # Round to multiple of num_heads (must be divisible)
    num_heads = int(cfg["fixed"].get("NUM_HEADS", BASE_ENV.get("NUM_HEADS", "4")))

    def round_dim(d):
        return max(num_heads, (d // num_heads) * num_heads)

    attempts = 0
    max_attempts = 8

    while lo <= hi and attempts < max_attempts:
        mid = round_dim((lo + hi) // 2)
        print(f"\n  [{config_name}] Trying MODEL_DIM={mid} (range [{lo}, {hi}])...")

        # Show what scaled dims this will produce
        if "scale_dims" in CONFIGS[config_name]:
            scaled = CONFIGS[config_name]["scale_dims"](mid)
            print(f"  [{config_name}]   scaled: feedback_dim={scaled.get('FEEDBACK_DIM','?')} "
                  f"hash_dim={scaled.get('BIGRAM_HASH_DIM','?')} "
                  f"capsule_dim={scaled.get('CAPSULE_DIM','?')} "
                  f"partial_rope={scaled.get('PARTIAL_ROPE_DIMS','?')} "
                  f"embed_dim={scaled.get('EMBED_DIM','?')}")

        result = run_training(
            config_name,
            env_overrides={},
            model_dim=mid,
            max_steps=CALIBRATION_STEPS,
            label=f"cal_{mid}",
        )
        measured_ms = result["mean_step_ms"]
        diff_pct = (measured_ms - target_ms) / target_ms

        print(f"  [{config_name}] dim={mid} → {measured_ms:.0f}ms/step "
              f"(target={target_ms:.0f}ms, diff={diff_pct:+.1%})")

        if abs(diff_pct) < best_diff:
            best_diff = abs(diff_pct)
            best_dim = mid

        if abs(diff_pct) <= tolerance:
            print(f"  [{config_name}] ✓ Converged: MODEL_DIM={mid}")
            return mid

        if measured_ms > target_ms:
            hi = mid - num_heads   # too slow, shrink
        else:
            lo = mid + num_heads   # too fast, grow

        attempts += 1

    print(f"  [{config_name}] Best after {attempts} attempts: MODEL_DIM={best_dim} "
          f"(diff={best_diff:.1%})")
    return best_dim


def run_calibration() -> dict:
    """Phase 1: Measure baseline, then calibrate all non-baseline configs."""
    print("\n" + "█" * 70)
    print("  PHASE 1: CALIBRATION")
    print("█" * 70)

    # Step 1: Measure baseline ms/step
    baseline_name = [k for k, v in CONFIGS.items() if v.get("is_baseline")][0]
    baseline_cfg = CONFIGS[baseline_name]

    print(f"\n▶ Measuring baseline ({baseline_name}) at MODEL_DIM={baseline_cfg['fixed']['MODEL_DIM']}...")
    baseline_result = run_training(
        baseline_name,
        env_overrides={},
        max_steps=CALIBRATION_STEPS,
        label="cal_baseline",
    )
    target_ms = baseline_result["mean_step_ms"]
    print(f"\n★ Baseline: {target_ms:.0f} ms/step (MODEL_DIM={baseline_cfg['fixed']['MODEL_DIM']})")

    # Step 2: Calibrate each non-baseline config
    calibrated_dims = {
        baseline_name: {
            "model_dim": int(baseline_cfg["fixed"]["MODEL_DIM"]),
            "target_ms": target_ms,
            "measured_ms": target_ms,
            "n_params": baseline_result["n_params"],
            "scaled_dims": {},
        }
    }

    for name, cfg in CONFIGS.items():
        if cfg.get("is_baseline"):
            continue

        print(f"\n▶ Calibrating {name}...")
        dim = calibrate_single(name, target_ms, TOLERANCE_PCT)

        # Verify final timing with scaled dims
        verify = run_training(
            name,
            env_overrides={},
            model_dim=dim,
            max_steps=CALIBRATION_STEPS,
            label=f"cal_verify_{dim}",
        )

        # Snapshot the scaled dims used at this model_dim
        scaled = CONFIGS[name].get("scale_dims", lambda d: {})(dim)
        calibrated_dims[name] = {
            "model_dim": dim,
            "target_ms": target_ms,
            "measured_ms": verify["mean_step_ms"],
            "n_params": verify["n_params"],
            "scaled_dims": scaled,
        }

    # Save calibration results
    with open(CALIBRATION_CACHE, "w") as f:
        json.dump(calibrated_dims, f, indent=2)
    print(f"\n✓ Calibration saved to {CALIBRATION_CACHE}")

    # Print summary table
    print(f"\n{'─'*60}")
    print(f"  CALIBRATION SUMMARY (target: {target_ms:.0f} ms/step)")
    print(f"{'─'*60}")
    print(f"  {'Config':<25} {'DIM':>5} {'ms/step':>8} {'Params':>10} {'Δ':>7}")
    print(f"  {'─'*25} {'─'*5} {'─'*8} {'─'*10} {'─'*7}")
    for name, info in calibrated_dims.items():
        diff = (info["measured_ms"] - target_ms) / target_ms * 100
        params = info.get("n_params", "?")
        print(f"  {name:<25} {info['model_dim']:>5} {info['measured_ms']:>7.0f}ms "
              f"{params:>10} {diff:>+6.1f}%")
    print()

    return calibrated_dims


# ===================================================================
# Phase 2: RACE — run all configs for fixed steps, eval periodically
# ===================================================================
def run_race(calibrated_dims: dict, race_steps: int, eval_every: int,
             start_from: str = None) -> list:
    """Phase 2: Run all configs at calibrated dims for race_steps."""
    print("\n" + "█" * 70)
    print(f"  PHASE 2: RACE ({race_steps} steps, eval every {eval_every})")
    print("█" * 70)

    results = []

    config_names = list(CONFIGS.keys())
    start_idx = 0
    if start_from:
        if start_from not in config_names:
            print(f"ERROR: --start-from '{start_from}' not in configs: {config_names}")
            sys.exit(1)
        start_idx = config_names.index(start_from)
        print(f"Resuming from {start_from} (skipping {config_names[:start_idx]})")

    for name in config_names[start_idx:]:
        info = calibrated_dims[name]
        dim = info["model_dim"]
        scaled = info.get("scaled_dims", {})
        print(f"\n▶ Racing {name} at MODEL_DIM={dim} scaled_dims={scaled}...")

        result = run_training(
            name,
            env_overrides={
                "VAL_LOSS_EVERY": str(eval_every),
                "TRAIN_LOG_EVERY": "10",
                "MAX_VAL_TOKENS": str(MAX_VAL_TOKENS_RACE),
                "VAL_BATCH_SIZE": "16384",     # smaller batches during val
            },
            model_dim=dim,
            max_steps=race_steps,
            max_wallclock=0,  # no time limit — pure step count
            val_loss_every=eval_every,
            label=f"race_dim{dim}",
        )
        results.append(result)

    return results


def generate_report(calibrated_dims: dict, results: list, outpath: Path):
    """Generate markdown report + JSON dump."""
    # JSON dump
    json_path = outpath.with_suffix(".json")
    with open(json_path, "w") as f:
        # Convert for JSON serialization
        out = []
        for r in results:
            out.append({
                "config": r["config"],
                "description": CONFIGS[r["config"]]["description"],
                "model_dim": r["model_dim"],
                "n_params": r["n_params"],
                "mean_step_ms": r["mean_step_ms"],
                "steps_completed": r["steps_completed"],
                "val_bpbs": r["val_bpbs"],
                "sliding_bpb": r["sliding_bpb"],
                "ttt_bpb": r["ttt_bpb"],
                "ngram_bpb": r["ngram_bpb"],
            })
        json.dump({"calibration": calibrated_dims, "results": out}, f, indent=2)

    # Markdown report
    md_path = outpath.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# Iso-Compute Bake-Off Results\n\n")
        target = calibrated_dims[list(calibrated_dims.keys())[0]]["target_ms"]
        f.write(f"**Target step time:** {target:.0f} ms/step "
                f"(baseline plain ternary)\n\n")

        # Calibration table
        f.write("## Calibrated Dimensions\n\n")
        f.write("| Config | MODEL_DIM | ms/step | Params | Δ |\n")
        f.write("|--------|-----------|---------|--------|---|\n")
        for name, info in calibrated_dims.items():
            diff = (info["measured_ms"] - target) / target * 100
            f.write(f"| {name} | {info['model_dim']} | {info['measured_ms']:.0f} "
                    f"| {info.get('n_params', '?')} | {diff:+.1f}% |\n")

        # Race results
        f.write("\n## Race Results\n\n")
        f.write("| Config | DIM | Params | Steps | Sliding BPB | TTT BPB | N-gram BPB | Best |\n")
        f.write("|--------|-----|--------|-------|-------------|---------|------------|------|\n")
        for r in results:
            best = min(filter(None, [r["sliding_bpb"], r["ttt_bpb"], r["ngram_bpb"]]),
                       default=None)
            f.write(f"| {r['config']} | {r['model_dim']} | {r['n_params']} "
                    f"| {r['steps_completed']} "
                    f"| {r['sliding_bpb'] or 'N/A'} "
                    f"| {r['ttt_bpb'] or 'N/A'} "
                    f"| {r['ngram_bpb'] or 'N/A'} "
                    f"| {best or 'N/A'} |\n")

        # Convergence curves
        f.write("\n## Convergence Curves (val_bpb vs step)\n\n")
        for r in results:
            f.write(f"### {r['config']} (dim={r['model_dim']}, {r['n_params']} params)\n\n")
            f.write("| Step | Val BPB |\n")
            f.write("|------|---------|\n")
            for step, bpb in r["val_bpbs"]:
                f.write(f"| {step} | {bpb:.4f} |\n")
            f.write("\n")

        # Winner
        valid = [(r, min(filter(None, [r["sliding_bpb"], r["ttt_bpb"], r["ngram_bpb"]]),
                         default=999))
                 for r in results]
        if valid:
            winner = min(valid, key=lambda x: x[1])
            f.write(f"\n## Winner: `{winner[0]['config']}` — "
                    f"Best BPB = **{winner[1]:.4f}**\n")
            f.write(f"\nAt MODEL_DIM={winner[0]['model_dim']}, "
                    f"{winner[0]['n_params']} params, "
                    f"{winner[0]['mean_step_ms']:.0f} ms/step\n")

    print(f"\n✓ Report: {md_path}")
    print(f"✓ Data:   {json_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Iso-Compute Architecture Bake-Off")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only run Phase 1 calibration")
    parser.add_argument("--race-only", action="store_true",
                        help="Skip calibration, use cached dims")
    parser.add_argument("--race-steps", type=int, default=DEFAULT_RACE_STEPS,
                        help=f"Steps per config in race (default: {DEFAULT_RACE_STEPS})")
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY,
                        help=f"Eval interval in race (default: {DEFAULT_EVAL_EVERY})")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Resume race from this config name (skips earlier ones)")
    args = parser.parse_args()

    t_start = time.time()

    # Phase 1
    if args.race_only:
        if not CALIBRATION_CACHE.exists():
            print(f"ERROR: {CALIBRATION_CACHE} not found. Run calibration first.")
            sys.exit(1)
        with open(CALIBRATION_CACHE) as f:
            calibrated_dims = json.load(f)
        print(f"Loaded cached calibration from {CALIBRATION_CACHE}")
    else:
        calibrated_dims = run_calibration()

    if args.calibrate_only:
        print(f"\nCalibration complete in {time.time() - t_start:.0f}s")
        return

    # Phase 2
    results = run_race(calibrated_dims, args.race_steps, args.eval_every,
                       start_from=args.start_from)

    # Report
    generate_report(calibrated_dims, results, SCRIPT_DIR / "bakeoff_results")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  BAKE-OFF COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    # Quick summary
    print(f"\n  {'Config':<25} {'DIM':>5} {'Params':>10} {'Best BPB':>10}")
    print(f"  {'─'*25} {'─'*5} {'─'*10} {'─'*10}")
    for r in results:
        best = min(filter(None, [r["sliding_bpb"], r["ttt_bpb"], r["ngram_bpb"]]),
                   default=None)
        print(f"  {r['config']:<25} {r['model_dim']:>5} {r['n_params'] or '?':>10} "
              f"{best or 'N/A':>10}")


if __name__ == "__main__":
    main()
