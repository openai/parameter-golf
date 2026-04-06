#!/usr/bin/env python3
"""
Headless Optuna hyperparameter search for parameter-golf (SOTA variant).

Usage:
    python3 autoresearch.py --trials 5 --minutes-per-trial 80

Study is persisted to ./optuna_study.db so it survives restarts.
Results are saved to ./run_history.json via start_sota.sh (auto-save).

To stop gracefully between trials: touch ./STOP_AUTORESEARCH
"""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

import optuna

import run_tracker

STOP_FILE = Path("./STOP_AUTORESEARCH")


def make_env(trial, minutes_per_trial):
    """Build env dict from Optuna trial suggestions."""
    params = {
        "MATRIX_LR": trial.suggest_float("MATRIX_LR", 0.05, 0.20, log=True),
        "SCALAR_LR": trial.suggest_float("SCALAR_LR", 0.005, 0.05, log=True),
        "TIED_EMBED_LR": trial.suggest_float("TIED_EMBED_LR", 0.01, 0.08, log=True),
        "MUON_MOMENTUM": trial.suggest_float("MUON_MOMENTUM", 0.92, 0.98),
        "WARMDOWN_ITERS": trial.suggest_int("WARMDOWN_ITERS", 800, 2500),
        "MUON_WD": trial.suggest_categorical("MUON_WD", [0.04, 0.06, 0.085]),
        "ADAM_WD": trial.suggest_categorical("ADAM_WD", [0.04, 0.06, 0.085]),
    }

    run_id = f"optuna_int8_{trial.number:03d}"

    env = os.environ.copy()
    env["RUN_ID"] = run_id
    env["ITERATIONS"] = "20000"
    env["MAX_WALLCLOCK_SECONDS"] = str(minutes_per_trial * 60)
    env["VAL_LOSS_EVERY"] = "200"
    env["TRAIN_LOG_EVERY"] = "50"
    # Hardcode: int8 mode, MLP 4×, 7 layers (baseline)
    env["USE_INT6"] = "0"
    env["MLP_MULT"] = "4"
    env["NUM_LAYERS"] = "7"
    for k, v in params.items():
        env[k] = str(v)

    return env, params, run_id


def run_trial(env, params, run_id):
    """Run a single training trial and return final_bpb.

    start_sota.sh auto-saves to run_history.json, so we only
    parse the log here for Optuna's objective value.
    """
    log_path = f"./logs/{run_id}.txt"

    proc = subprocess.run(
        ["bash", "start_baseline.sh"],
        env=env,
        capture_output=True,
        text=True,
    )

    parsed = run_tracker.parse_log(log_path)

    final_bpb = parsed["final_bpb"]
    artifact_size = parsed.get("artifact_size_bytes")

    if final_bpb is None:
        final_bpb_for_optuna = float("inf")
    else:
        final_bpb_for_optuna = final_bpb

    # Penalize configs that bust the 16MB artifact limit
    if artifact_size is not None and artifact_size > run_tracker.MAX_ARTIFACT_BYTES:
        print(f"  WARNING: artifact {artifact_size} bytes > 16MB limit, adding penalty")
        final_bpb_for_optuna += 10.0

    return final_bpb_for_optuna


# Track the current training subprocess so we can kill it on stop
_current_proc = None


def objective(trial, minutes_per_trial):
    global _current_proc

    # Check stop file before starting a new trial
    if STOP_FILE.exists():
        STOP_FILE.unlink(missing_ok=True)
        print("\nSTOP_AUTORESEARCH file detected — stopping.")
        raise optuna.exceptions.OptunaError("Stopped by user (STOP_AUTORESEARCH file)")

    env, params, run_id = make_env(trial, minutes_per_trial)

    total_trials = len(trial.study.trials) if hasattr(trial, 'study') else '?'
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: {run_id}")
    print(f"Params: { {k: (f'{v:.5f}' if isinstance(v, float) else v) for k, v in params.items()} }")
    print(f"{'='*60}")

    # GPU thermal status before trial
    thermal = run_tracker.get_gpu_thermal_status()
    print(f"GPU temp before trial: {thermal}")

    bpb = run_trial(env, params, run_id)

    # GPU thermal status after trial
    thermal = run_tracker.get_gpu_thermal_status()
    print(f"GPU temp after trial: {thermal}")

    print(f"Result: val_bpb = {bpb}")
    return bpb


def main():
    parser = argparse.ArgumentParser(description="Optuna autoresearch for parameter-golf (SOTA)")
    parser.add_argument("--trials", type=int, default=5)
    # 80 min on Spark at ~7.4s/step ≈ 650 steps
    parser.add_argument("--minutes-per-trial", type=int, default=80)
    args = parser.parse_args()

    # Clean up any stale stop file
    STOP_FILE.unlink(missing_ok=True)

    storage = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        study_name="parameter_golf_int8_baseline",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    existing = len(study.trials)
    total = existing + args.trials
    print(f"Starting autoresearch (SOTA): {args.trials} trials, {args.minutes_per_trial} min each")
    print(f"Study has {existing} existing trials (will run up to {total} total)")
    print(f"To stop gracefully: touch ./STOP_AUTORESEARCH")

    try:
        study.optimize(
            lambda trial: objective(trial, args.minutes_per_trial),
            n_trials=args.trials,
        )
    except (optuna.exceptions.OptunaError, KeyboardInterrupt) as e:
        print(f"\nAutoresearch stopped: {e}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{'='*60}")
    print("AUTORESEARCH COMPLETE" if not STOP_FILE.exists() else "AUTORESEARCH STOPPED")
    if completed:
        print(f"Best val_bpb: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
    print(f"Total trials: {len(study.trials)} ({len(completed)} completed)")
    print(f"{'='*60}")

    # Print leaderboard
    runs = run_tracker.load_runs()
    print(f"\nTop 5 runs:")
    for i, r in enumerate(runs[:5]):
        bpb = r.get("final_bpb", "N/A")
        bpb_str = f"{bpb:.4f}" if isinstance(bpb, (int, float)) and bpb != float("inf") else "N/A"
        print(f"  {i+1}. {r['run_id']}: bpb={bpb_str} steps={r.get('steps_completed', '?')}")


if __name__ == "__main__":
    main()
