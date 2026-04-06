#!/usr/bin/env python3
"""Optuna hyperparameter sweep for Parameter Golf.

Sweeps model and training hyperparameters within the 16MB budget constraint.

Usage:
    python training/sweep.py --n-trials 50 --train-seconds 120
"""

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

try:
    import optuna
except ImportError:
    print("Install optuna: pip install optuna")
    sys.exit(1)

from models.tiny_gpt_shared import TinyGPTShared
from utils.data import load_fineweb_train, load_fineweb_valid, make_byte_batches
from utils.quantize import check_budget


def objective(trial, train_data, val_data, device, base_args):
    # Sample hyperparameters
    d_model = trial.suggest_categorical("d_model", [384, 448, 512, 576, 640])
    n_unique_layers = trial.suggest_int("n_unique_layers", 3, 8)
    n_loops = trial.suggest_int("n_loops", 2, 6)
    mlp_ratio = trial.suggest_float("mlp_ratio", 2.0, 4.0, step=0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])

    # Ensure d_model divisible by n_heads
    if d_model % n_heads != 0:
        raise optuna.TrialPruned()

    model = TinyGPTShared(
        vocab_size=256, d_model=d_model,
        n_unique_layers=n_unique_layers, n_loops=n_loops,
        n_heads=n_heads, mlp_ratio=mlp_ratio,
        max_seq_len=base_args.seq_len, tie_embeddings=True,
    )

    # Budget check — must fit in 16MB at int6
    fits, details = check_budget(model, bits_per_param=6)
    if not details["estimates"]["int6_zlib"]["fits"]:
        raise optuna.TrialPruned()

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        betas=(0.9, 0.95), weight_decay=0.1,
    )

    from contextlib import nullcontext
    if device.type == "cuda":
        dtype_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        dtype_ctx = nullcontext()

    gen = make_byte_batches(train_data, base_args.batch_size, base_args.seq_len)

    import time
    start = time.time()
    model.train()
    step = 0

    while time.time() - start < base_args.train_seconds:
        x, y = next(gen)
        x, y = x.to(device), y.to(device)

        with dtype_ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        # Report intermediate for pruning
        if step % 50 == 0:
            bpb = loss.item() / math.log(2)
            trial.report(bpb, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Final eval on validation
    model.eval()
    val_gen = make_byte_batches(val_data, base_args.batch_size, base_args.seq_len)
    total_loss = 0.0
    n_eval = 20
    with torch.no_grad():
        for _ in range(n_eval):
            x, y = next(val_gen)
            x, y = x.to(device), y.to(device)
            with dtype_ctx:
                _, loss = model(x, y)
            total_loss += loss.item()

    val_bpb = (total_loss / n_eval) / math.log(2)
    return val_bpb


def main():
    p = argparse.ArgumentParser(description="Optuna Sweep")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--train-seconds", type=float, default=120,
                   help="Training time per trial")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--train-bytes", type=int, default=500_000_000)
    p.add_argument("--study-name", type=str, default="parameter_golf")
    p.add_argument("--db", type=str, default=str(PROJECT_ROOT / "sweep_study.db"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sweep] Device: {device}")

    print("[sweep] Loading data...")
    train_data = load_fineweb_train(args.train_bytes)
    val_data = load_fineweb_valid()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{args.db}",
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    study.optimize(
        lambda trial: objective(trial, train_data, val_data, device, args),
        n_trials=args.n_trials,
    )

    print(f"\n[sweep] Best trial: {study.best_trial.value:.4f} BPB")
    print(f"[sweep] Best params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
