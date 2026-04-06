#!/usr/bin/env python3
"""Leslie Smith LR Range Test for Parameter Golf models.

Sweeps learning rate from min_lr to max_lr over N steps,
records loss at each step, and identifies the optimal LR range.

Usage:
    python training/lr_finder.py --model-type shared --n-unique-layers 5 --n-loops 4
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train import build_model, parse_args as train_parse_args
from utils.data import load_fineweb_train, make_byte_batches


def lr_find(args=None):
    if args is None:
        p = argparse.ArgumentParser(description="LR Finder")
        p.add_argument("--min-lr", type=float, default=1e-6)
        p.add_argument("--max-lr", type=float, default=1e-1)
        p.add_argument("--n-steps", type=int, default=200)
        p.add_argument("--smoothing", type=float, default=0.05,
                       help="Exponential smoothing factor")
        # Inherit model args by parsing known + unknown
        args, remaining = p.parse_known_args()
        # Parse model args from remaining
        sys.argv = [sys.argv[0]] + remaining
        model_args = train_parse_args()
        # Merge
        for k, v in vars(model_args).items():
            if not hasattr(args, k):
                setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lr_finder] Device: {device}")

    from contextlib import nullcontext
    if args.dtype == "bf16" and device.type == "cuda":
        dtype_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    elif args.dtype == "fp16" and device.type == "cuda":
        dtype_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        dtype_ctx = nullcontext()

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.min_lr,
        betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
    )

    train_data = load_fineweb_train(args.train_bytes)
    gen = make_byte_batches(train_data, args.batch_size, args.seq_len)

    # Exponential LR schedule: min_lr -> max_lr
    mult = (args.max_lr / args.min_lr) ** (1.0 / args.n_steps)

    results = []
    smoothed_loss = 0.0
    best_loss = float("inf")
    lr = args.min_lr

    print(f"[lr_finder] Sweeping LR from {args.min_lr:.2e} to {args.max_lr:.2e} "
          f"over {args.n_steps} steps")

    model.train()
    for step in range(args.n_steps):
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = next(gen)
        x, y = x.to(device), y.to(device)

        with dtype_ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        if step == 0:
            smoothed_loss = loss_val
        else:
            smoothed_loss = smoothed_loss * (1 - args.smoothing) + loss_val * args.smoothing

        bpb = smoothed_loss / math.log(2)
        results.append({"step": step, "lr": lr, "loss": loss_val,
                       "smoothed_loss": smoothed_loss, "bpb": bpb})

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        # Stop if loss diverges (> 4x best)
        if smoothed_loss > best_loss * 4:
            print(f"[lr_finder] Loss diverged at lr={lr:.2e}, stopping.")
            break

        if step % 20 == 0:
            print(f"  step {step:>4d} | lr {lr:.2e} | loss {smoothed_loss:.4f} | "
                  f"bpb {bpb:.4f}")

        lr *= mult

    # Save results
    results_dir = Path(args.save_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"lr_finder_{args.exp_name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "lr", "loss", "smoothed_loss", "bpb"])
        w.writeheader()
        w.writerows(results)

    # Find suggested LR (steepest descent point)
    if len(results) > 10:
        losses = [r["smoothed_loss"] for r in results]
        lrs = [r["lr"] for r in results]
        # Find point of steepest negative slope
        gradients = [(losses[i+1] - losses[i]) / (lrs[i+1] - lrs[i])
                     for i in range(len(losses)-1)]
        min_grad_idx = min(range(len(gradients)), key=lambda i: gradients[i])
        suggested_lr = lrs[min_grad_idx]
        print(f"\n[lr_finder] Suggested LR: {suggested_lr:.2e}")
        print(f"[lr_finder] (where loss decreased fastest)")
    print(f"[lr_finder] Results saved to {csv_path}")


if __name__ == "__main__":
    lr_find()
