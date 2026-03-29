#!/usr/bin/env python3
"""Optuna-based hyperparameter optimization for Parameter Golf.

Supports two modes:
  1. --validate: Run each technique individually against baseline to verify it works
  2. (default): Full Optuna TPE search over the joint space

Interruptible: Ctrl+C saves progress to SQLite. Resume by re-running.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

TRAIN_SCRIPT = "records/track_10min_16mb/2026-03-29_FullStack_TTT_Ngram_KNN_TurboQuant/train_gpt.py"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(PROJECT_DIR, "validation_results.jsonl")

# ---------------------------------------------------------------------------
# Budget estimation
# ---------------------------------------------------------------------------

def estimate_compressed_size(
    model_dim, num_layers, mlp_mult, num_heads, num_kv_heads,
    quant_bits, enable_entropy_coding, enable_pruning, prune_fraction,
    vocab_size=1024,
):
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    mlp_hidden = mlp_mult * model_dim
    embed_params = vocab_size * model_dim
    per_layer = (
        model_dim * model_dim + model_dim * kv_dim + model_dim * kv_dim
        + model_dim * model_dim + model_dim * mlp_hidden + mlp_hidden * model_dim
        + model_dim * 5 + num_heads
    )
    total_params = embed_params + num_layers * per_layer + model_dim * num_layers
    weight_2d = num_layers * per_layer * 0.95
    weight_1d = num_layers * per_layer * 0.05 + embed_params
    raw_bytes = (weight_2d * quant_bits / 8) + (weight_1d * 8 / 8)
    if enable_pruning:
        raw_bytes *= (1 - prune_fraction * 0.5)
    compression_ratio = 0.70 if enable_entropy_coding else 0.75
    return raw_bytes * compression_ratio + 82_000, total_params


def find_max_model(quant_bits, enable_entropy_coding, enable_pruning, prune_fraction,
                   target_bytes=16_000_000):
    best = None
    for num_layers in [8, 9, 10, 11, 12, 13, 14]:
        for model_dim in [384, 448, 512, 576, 640, 768]:
            for mlp_mult in [2, 3]:
                for num_heads in [4, 8]:
                    if model_dim % num_heads != 0:
                        continue
                    for num_kv_heads in [2, 4]:
                        if num_heads % num_kv_heads != 0:
                            continue
                        size, params = estimate_compressed_size(
                            model_dim, num_layers, mlp_mult, num_heads, num_kv_heads,
                            quant_bits, enable_entropy_coding, enable_pruning, prune_fraction,
                        )
                        if size <= target_bytes:
                            if best is None or params > best[1]:
                                best = (
                                    dict(num_layers=num_layers, model_dim=model_dim,
                                         mlp_mult=mlp_mult, num_heads=num_heads,
                                         num_kv_heads=num_kv_heads),
                                    params, size,
                                )
    return best


# ---------------------------------------------------------------------------
# Run a single training trial
# ---------------------------------------------------------------------------

PENALTY = 10.0

def run_trial(config: dict, max_wallclock: int, iterations: int,
              label: str = "", skip_compile: bool = False) -> dict:
    """Run one training+eval and return parsed results dict."""
    # Defaults for 4090
    base = {
        "ITERATIONS": iterations,
        "MAX_WALLCLOCK_SECONDS": max_wallclock,
        "TRAIN_BATCH_TOKENS": 65536,
        "VAL_BATCH_SIZE": 65536,
        "VAL_LOSS_EVERY": 0,
        "TRAIN_LOG_EVERY": 50,
        "WARMUP_STEPS": 0,
        "ENABLE_TURBOQUANT": 1,
        "SKIP_COMPILE": 1 if skip_compile else 0,
    }
    base.update(config)

    env = os.environ.copy()
    env.update({k: str(v) for k, v in base.items()})

    timeout = max_wallclock + 300
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["torchrun", "--standalone", "--nproc_per_node=1",
             os.path.join(PROJECT_DIR, TRAIN_SCRIPT)],
            env=env, capture_output=True, text=True, timeout=timeout,
            cwd=PROJECT_DIR,
        )
        elapsed = time.time() - t0
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return {"label": label, "status": "TIMEOUT", "val_bpb": PENALTY,
                "elapsed": time.time() - t0, "config": config}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"label": label, "status": f"EXCEPTION: {e}", "val_bpb": PENALTY,
                "elapsed": time.time() - t0, "config": config}

    if proc.returncode != 0:
        err_lines = (proc.stderr or "").strip().split("\n")[-10:]
        return {"label": label, "status": "CRASHED", "val_bpb": PENALTY,
                "elapsed": elapsed, "error": "\n".join(err_lines), "config": config}

    # Parse results
    val_bpb = None
    for pattern in [
        r"final_ttt_lora_exact val_loss:\S+ val_bpb:(\S+)",
        r"final_int8_zlib_roundtrip_exact val_loss:\S+ val_bpb:(\S+)",
        r"val_bpb:(\S+)",
    ]:
        m = re.search(pattern, output)
        if m:
            val_bpb = float(m.group(1))
            break

    size_match = re.search(r"Total submission size int8\+zlib: (\d+) bytes", output)
    actual_size = int(size_match.group(1)) if size_match else None

    return {
        "label": label,
        "status": "OK",
        "val_bpb": val_bpb if val_bpb else PENALTY,
        "artifact_size": actual_size,
        "elapsed": elapsed,
        "config": config,
    }


def save_result(result: dict):
    """Append result to JSONL file."""
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def load_completed_labels() -> set:
    """Load labels of already-completed trials."""
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "OK":
                        completed.add(r["label"])
                except json.JSONDecodeError:
                    continue
    return completed


# ---------------------------------------------------------------------------
# Validation mode: test each technique individually
# ---------------------------------------------------------------------------

BASELINE_CONFIG = {
    "RUN_ID": "baseline",
    "NUM_LAYERS": 9, "MODEL_DIM": 512, "MLP_MULT": 2,
    "NUM_HEADS": 8, "NUM_KV_HEADS": 4,
}

# Each entry: (label, {env_var_overrides})
TECHNIQUE_TESTS = [
    # --- Baseline ---
    ("baseline", {}),

    # --- Activations (mutually exclusive) ---
    ("activation_leaky_relu_squared", {"ACTIVATION": "leaky_relu_squared"}),
    ("activation_star_relu", {"ACTIVATION": "star_relu"}),
    ("activation_polycom", {"ACTIVATION": "polycom"}),

    # --- Architecture toggles ---
    ("hybridnorm", {"ENABLE_HYBRIDNORM": 1}),
    ("smeargate", {"ENABLE_SMEARGATE": 1}),
    ("diff_attn", {"ENABLE_DIFF_ATTN": 1}),
    ("pope", {"ENABLE_POPE": 1}),
    ("wavelet", {"ENABLE_WAVELET": 1}),
    ("vga", {"ENABLE_VGA": 1}),
    ("xsa_3", {"XSA_LAST_N": 3}),
    ("xsa_6", {"XSA_LAST_N": 6}),
    ("mtp_1", {"MTP_NUM_HEADS": 1}),
    ("mtp_2", {"MTP_NUM_HEADS": 2}),

    # --- Training toggles ---
    ("ema_0.997", {"EMA_DECAY": 0.997}),
    ("ema_0.999", {"EMA_DECAY": 0.999}),
    ("swa", {"ENABLE_SWA": 1}),
    ("qat", {"ENABLE_QAT": 1}),
    ("ema_swa", {"EMA_DECAY": 0.997, "ENABLE_SWA": 1}),

    # --- Quantization ---
    ("quant_int6", {"QUANT_BITS": 6}),
    ("quant_int5", {"QUANT_BITS": 5}),
    ("optrot", {"ENABLE_OPTROT": 1}),
    ("gptq", {"ENABLE_GPTQ": 1}),
    ("pruning_2pct", {"ENABLE_PRUNING": 1, "PRUNE_FRACTION": 0.02}),
    ("entropy_coding", {"ENABLE_ENTROPY_CODING": 1}),
    ("optrot_gptq", {"ENABLE_OPTROT": 1, "ENABLE_GPTQ": 1}),
    ("optrot_gptq_pruning", {"ENABLE_OPTROT": 1, "ENABLE_GPTQ": 1, "ENABLE_PRUNING": 1}),

    # --- Eval-time (add on top of baseline) ---
    ("ttt", {"ENABLE_TTT": 1}),
    ("ngram", {"ENABLE_NGRAM": 1}),
    ("knn", {"ENABLE_KNN": 1}),
    ("ttt_tempcal", {"ENABLE_TTT": 1, "TTT_TEMP": 0.98}),
    ("ngram_knn", {"ENABLE_NGRAM": 1, "ENABLE_KNN": 1}),
    ("ttt_ngram_knn", {"ENABLE_TTT": 1, "TTT_TEMP": 0.98, "ENABLE_NGRAM": 1, "ENABLE_KNN": 1}),

    # --- Model size variations with int5 ---
    ("int5_11L", {"QUANT_BITS": 5, "NUM_LAYERS": 11}),
    ("int5_11L_mlp3x", {"QUANT_BITS": 5, "NUM_LAYERS": 11, "MLP_MULT": 3}),
    ("int5_13L", {"QUANT_BITS": 5, "NUM_LAYERS": 13}),
    ("int6_11L", {"QUANT_BITS": 6, "NUM_LAYERS": 11}),
    ("int6_11L_mlp3x", {"QUANT_BITS": 6, "NUM_LAYERS": 11, "MLP_MULT": 3}),
]


def run_validation(max_wallclock: int, iterations: int, skip_compile: bool = True):
    """Run each technique test sequentially. Interruptible — skips completed trials."""
    completed = load_completed_labels()
    total = len(TECHNIQUE_TESTS)
    done = 0

    print(f"\n{'='*70}")
    print(f"TECHNIQUE VALIDATION: {total} tests, {len(completed)} already done")
    print(f"Training: {iterations} iters, {max_wallclock}s wallclock cap")
    print(f"Results: {RESULTS_FILE}")
    print(f"Ctrl+C to stop — progress is saved, re-run to resume")
    print(f"{'='*70}\n")

    for label, overrides in TECHNIQUE_TESTS:
        if label in completed:
            done += 1
            print(f"[{done}/{total}] {label}: SKIP (already completed)")
            continue

        done += 1
        config = {**BASELINE_CONFIG, **overrides}
        config["RUN_ID"] = f"val_{label}"

        print(f"\n[{done}/{total}] {label}")
        print(f"  Config: {overrides or '(baseline defaults)'}")
        print(f"  Running...", end="", flush=True)

        try:
            result = run_trial(config, max_wallclock, iterations, label=label, skip_compile=skip_compile)
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at trial {done}/{total} ({label})")
            print(f"Completed {done-1} trials. Re-run to resume.")
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

    # Print summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Label':<30} {'BPB':>8} {'Size':>12} {'Status':>8}")
    print("-" * 70)

    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

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
            d = bpb - baseline_bpb
            delta = f" ({d:+.4f})"
        print(f"{label:<30} {bpb:>8.4f} {str(size):>12} {status:>8}{delta}")


# ---------------------------------------------------------------------------
# Optuna search mode
# ---------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, *, max_wallclock: int, iterations: int) -> float:
    activation = trial.suggest_categorical("activation", ["relu_squared", "leaky_relu_squared", "star_relu", "polycom"])
    quant_bits = trial.suggest_categorical("quant_bits", [5, 6, 8])
    enable_entropy_coding = trial.suggest_categorical("enable_entropy_coding", [0, 1])
    enable_hybridnorm = trial.suggest_categorical("enable_hybridnorm", [0, 1])
    enable_smeargate = trial.suggest_categorical("enable_smeargate", [0, 1])
    enable_diff_attn = trial.suggest_categorical("enable_diff_attn", [0, 1])
    enable_pope = trial.suggest_categorical("enable_pope", [0, 1])
    enable_wavelet = trial.suggest_categorical("enable_wavelet", [0, 1])
    enable_vga = trial.suggest_categorical("enable_vga", [0, 1])
    xsa_last_n = trial.suggest_int("xsa_last_n", 0, 6)
    mtp_num_heads = trial.suggest_int("mtp_num_heads", 0, 2)
    ema_decay = trial.suggest_categorical("ema_decay", [0.0, 0.99, 0.995, 0.997, 0.999])
    enable_swa = trial.suggest_categorical("enable_swa", [0, 1])
    enable_qat = trial.suggest_categorical("enable_qat", [0, 1])
    enable_optrot = trial.suggest_categorical("enable_optrot", [0, 1])
    enable_gptq = trial.suggest_categorical("enable_gptq", [0, 1])
    enable_pruning = trial.suggest_categorical("enable_pruning", [0, 1])
    prune_fraction = trial.suggest_float("prune_fraction", 0.01, 0.05) if enable_pruning else 0.0
    enable_ttt = trial.suggest_categorical("enable_ttt", [0, 1])
    ttt_lora_rank = trial.suggest_categorical("ttt_lora_rank", [4, 8, 16]) if enable_ttt else 8
    ttt_lora_lr = trial.suggest_float("ttt_lora_lr", 0.001, 0.05, log=True) if enable_ttt else 0.01
    ttt_temp = trial.suggest_float("ttt_temp", 0.95, 1.0) if enable_ttt else 1.0
    enable_ngram = trial.suggest_categorical("enable_ngram", [0, 1])
    enable_knn = trial.suggest_categorical("enable_knn", [0, 1])
    matrix_lr = trial.suggest_float("matrix_lr", 0.01, 0.1, log=True)
    scalar_lr = trial.suggest_float("scalar_lr", 0.01, 0.1, log=True)
    muon_momentum = trial.suggest_float("muon_momentum", 0.9, 0.99)

    result = find_max_model(quant_bits, enable_entropy_coding, enable_pruning, prune_fraction)
    if result is None:
        return PENALTY
    arch, total_params, est_size = result

    config = {
        "RUN_ID": f"optuna_t{trial.number}",
        **arch,
        "ACTIVATION": activation, "QUANT_BITS": quant_bits,
        "ENABLE_ENTROPY_CODING": int(enable_entropy_coding),
        "ENABLE_HYBRIDNORM": int(enable_hybridnorm),
        "ENABLE_SMEARGATE": int(enable_smeargate),
        "ENABLE_DIFF_ATTN": int(enable_diff_attn),
        "ENABLE_POPE": int(enable_pope),
        "ENABLE_WAVELET": int(enable_wavelet),
        "ENABLE_VGA": int(enable_vga),
        "XSA_LAST_N": xsa_last_n, "MTP_NUM_HEADS": mtp_num_heads,
        "EMA_DECAY": ema_decay,
        "ENABLE_SWA": int(enable_swa), "ENABLE_QAT": int(enable_qat),
        "ENABLE_OPTROT": int(enable_optrot), "ENABLE_GPTQ": int(enable_gptq),
        "ENABLE_PRUNING": int(enable_pruning), "PRUNE_FRACTION": prune_fraction,
        "ENABLE_TTT": int(enable_ttt), "TTT_LORA_RANK": ttt_lora_rank,
        "TTT_LORA_LR": ttt_lora_lr, "TTT_TEMP": ttt_temp,
        "ENABLE_NGRAM": int(enable_ngram), "ENABLE_KNN": int(enable_knn),
        "ENABLE_TURBOQUANT": 1,
        "MATRIX_LR": matrix_lr, "SCALAR_LR": scalar_lr,
        "MUON_MOMENTUM": muon_momentum,
    }

    print(f"Trial {trial.number}: params={total_params:,} est_size={est_size:,.0f}")
    r = run_trial(config, max_wallclock, iterations, label=f"optuna_t{trial.number}")
    save_result(r)

    val_bpb = r["val_bpb"]
    actual_size = r.get("artifact_size")
    if actual_size and actual_size > 16_000_000:
        val_bpb += (actual_size - 16_000_000) / 1_000_000
    print(f"Trial {trial.number}: val_bpb={val_bpb:.4f} status={r['status']}")
    return val_bpb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parameter Golf HPO")
    sub = parser.add_subparsers(dest="command", help="Mode")

    # Validate mode
    val_p = sub.add_parser("validate", help="Test each technique individually against baseline")
    val_p.add_argument("--max-wallclock", type=int, default=60)
    val_p.add_argument("--iterations", type=int, default=500)

    # Search mode
    search_p = sub.add_parser("search", help="Optuna TPE search over joint space")
    search_p.add_argument("--n-trials", type=int, default=100)
    search_p.add_argument("--max-wallclock", type=int, default=60)
    search_p.add_argument("--iterations", type=int, default=500)
    search_p.add_argument("--study-name", type=str, default="pgolf_v1")
    search_p.add_argument("--db", type=str, default="sqlite:///optuna_pgolf.db")

    # Results mode
    results_p = sub.add_parser("results", help="Show results from validation or search")
    results_p.add_argument("--study-name", type=str, default="pgolf_v1")
    results_p.add_argument("--db", type=str, default="sqlite:///optuna_pgolf.db")
    results_p.add_argument("--best", action="store_true")
    results_p.add_argument("--importance", action="store_true")
    results_p.add_argument("--top", type=int, default=0)

    args = parser.parse_args()

    if args.command == "validate":
        run_validation(args.max_wallclock, args.iterations, skip_compile=True)

    elif args.command == "search":
        study = optuna.create_study(
            study_name=args.study_name, storage=args.db,
            direction="minimize",
            sampler=TPESampler(multivariate=True, seed=42),
            load_if_exists=True,
        )
        try:
            study.optimize(
                lambda trial: optuna_objective(trial, max_wallclock=args.max_wallclock,
                                               iterations=args.iterations),
                n_trials=args.n_trials, show_progress_bar=True,
            )
        except KeyboardInterrupt:
            print(f"\nInterrupted after {len(study.trials)} trials")
        t = study.best_trial
        print(f"\nBest: trial #{t.number} val_bpb={t.value:.4f}")
        for k, v in sorted(t.params.items()):
            print(f"  {k}: {v}")

    elif args.command == "results":
        if args.best or args.importance or args.top:
            study = optuna.load_study(study_name=args.study_name, storage=args.db)
            if args.best:
                t = study.best_trial
                print(f"Best: trial #{t.number} val_bpb={t.value:.4f}")
                for k, v in sorted(t.params.items()):
                    print(f"  {k}: {v}")
            if args.importance:
                imp = optuna.importance.get_param_importances(study)
                for k, v in imp.items():
                    print(f"  {k}: {v:.4f}")
            if args.top > 0:
                for t in sorted(study.trials, key=lambda x: x.value or 99)[:args.top]:
                    print(f"Trial #{t.number}: val_bpb={t.value:.4f} params={dict(sorted(t.params.items()))}")
        else:
            # Show validation results
            if os.path.exists(RESULTS_FILE):
                results = []
                with open(RESULTS_FILE) as f:
                    for line in f:
                        try:
                            results.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                baseline_bpb = None
                for r in sorted(results, key=lambda x: x.get("val_bpb", 99)):
                    bpb = r.get("val_bpb", 99)
                    label = r.get("label", "?")
                    if label == "baseline":
                        baseline_bpb = bpb
                    delta = f" ({bpb - baseline_bpb:+.4f})" if baseline_bpb and label != "baseline" and bpb < PENALTY else ""
                    print(f"{label:<30} bpb={bpb:.4f} size={r.get('artifact_size','?')} {r.get('status','?')}{delta}")
            else:
                print("No validation results yet. Run: python optuna_search.py validate")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
