#!/usr/bin/env python3
"""Optuna TPE search over the joint technique + model size space.

Interruptible: Ctrl+C saves to SQLite. Resume by re-running.

Usage:
    python search.py [--n-trials 100] [--max-wallclock 60] [--iterations 500]
    python search.py --best
    python search.py --importance
    python search.py --top 10
"""

from __future__ import annotations

import argparse

import optuna
from optuna.samplers import TPESampler

from pgolf.budget import find_max_model
from pgolf.config import PENALTY
from pgolf.runner import run_trial, save_result


def objective(trial: optuna.Trial, *, max_wallclock: int, iterations: int) -> float:
    # --- Sample all technique toggles ---
    activation = trial.suggest_categorical(
        "activation", ["relu_squared", "leaky_relu_squared", "star_relu", "polycom"])
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

    # --- Find max model that fits budget ---
    result = find_max_model(quant_bits, enable_entropy_coding, enable_pruning, prune_fraction)
    if result is None:
        return PENALTY
    arch, total_params, est_size = result

    config = {
        "RUN_ID": f"optuna_t{trial.number}", **arch,
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


def main():
    parser = argparse.ArgumentParser(description="Optuna search for Parameter Golf")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--max-wallclock", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--study-name", type=str, default="pgolf_v1")
    parser.add_argument("--db", type=str, default="sqlite:///optuna_pgolf.db")
    parser.add_argument("--best", action="store_true", help="Print best trial")
    parser.add_argument("--importance", action="store_true", help="Print param importance")
    parser.add_argument("--top", type=int, default=0, help="Print top N trials")
    args = parser.parse_args()

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
                print(f"Trial #{t.number}: val_bpb={t.value:.4f}")
                for k, v in sorted(t.params.items()):
                    print(f"  {k}: {v}")
                print()
        return

    study = optuna.create_study(
        study_name=args.study_name, storage=args.db, direction="minimize",
        sampler=TPESampler(multivariate=True, seed=42), load_if_exists=True,
    )
    try:
        study.optimize(
            lambda trial: objective(trial, max_wallclock=args.max_wallclock,
                                    iterations=args.iterations),
            n_trials=args.n_trials, show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(study.trials)} trials")

    t = study.best_trial
    print(f"\nBest: trial #{t.number} val_bpb={t.value:.4f}")
    for k, v in sorted(t.params.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
