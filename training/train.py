#!/usr/bin/env python3
"""Unified training script for Parameter Golf.

Supports both standard and weight-sharing transformer models.
Trains for a fixed wall-clock time or step count, logs BPB to CSV.

Usage:
    python training/train.py --help
    python training/train.py --model-type shared --train-seconds 60
    python training/train.py --model-type standard --n-layers 10 --train-seconds 600
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.tiny_gpt import TinyGPT
from models.tiny_gpt_shared import TinyGPTShared
from utils.data import load_fineweb_train, load_fineweb_valid, make_byte_batches
from utils.quantize import print_budget_report, check_budget


def parse_args():
    p = argparse.ArgumentParser(description="Parameter Golf Training")

    # Model
    p.add_argument("--model-type", choices=["standard", "shared"], default="shared")
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=3.0)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--tie-embeddings", action="store_true", default=True)
    p.add_argument("--no-tie-embeddings", dest="tie_embeddings", action="store_false")
    p.add_argument("--parallel-residual", action="store_true", default=False)

    # Standard model
    p.add_argument("--n-layers", type=int, default=10,
                   help="Number of layers for standard model")

    # Shared model
    p.add_argument("--n-unique-layers", type=int, default=5)
    p.add_argument("--n-loops", type=int, default=4)

    # Training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--train-seconds", type=float, default=600)
    p.add_argument("--max-steps", type=int, default=0,
                   help="If > 0, stop after this many steps instead of time")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile for speedup")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")

    # Data
    p.add_argument("--train-bytes", type=int, default=500_000_000,
                   help="Bytes of training data to download/cache")

    # Logging
    p.add_argument("--exp-name", type=str, default="default")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-tokens", type=int, default=1_000_000,
                   help="Number of bytes to evaluate on")
    p.add_argument("--save-dir", type=str,
                   default=str(PROJECT_ROOT / "experiments"))

    return p.parse_args()


def build_model(args) -> torch.nn.Module:
    if args.model_type == "standard":
        model = TinyGPT(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            mlp_ratio=args.mlp_ratio,
            max_seq_len=args.max_seq_len,
            tie_embeddings=args.tie_embeddings,
            parallel_residual=args.parallel_residual,
        )
    else:
        model = TinyGPTShared(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_unique_layers=args.n_unique_layers,
            n_loops=args.n_loops,
            n_heads=args.n_heads,
            mlp_ratio=args.mlp_ratio,
            max_seq_len=args.max_seq_len,
            tie_embeddings=args.tie_embeddings,
            parallel_residual=args.parallel_residual,
        )
    return model


def get_lr(step: int, warmup_steps: int, max_lr: float, total_steps: int) -> float:
    """Cosine LR schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return max_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, val_data, args, device, dtype_ctx):
    """Evaluate BPB on validation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    seq_len = args.seq_len
    n_eval = min(args.eval_tokens, len(val_data) - seq_len - 1)
    n_batches = max(1, min(100, n_eval // (args.batch_size * seq_len)))

    gen = make_byte_batches(val_data, args.batch_size, seq_len)
    with torch.no_grad():
        for i in range(n_batches):
            x, y = next(gen)
            x, y = x.to(device), y.to(device)
            with dtype_ctx:
                _, loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    bpb = avg_loss / math.log(2)
    model.train()
    return avg_loss, bpb


def main():
    args = parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Mixed precision context
    if args.dtype == "bf16" and device.type == "cuda":
        dtype_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    elif args.dtype == "fp16" and device.type == "cuda":
        dtype_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        from contextlib import nullcontext
        dtype_ctx = nullcontext()

    # Build model
    model = build_model(args)
    model = model.to(device)

    # Budget report (before compile, which wraps the module)
    model_name = (f"{args.model_type} "
                  f"{'shared ' + str(args.n_unique_layers) + 'x' + str(args.n_loops) if args.model_type == 'shared' else str(args.n_layers) + ' layers'} "
                  f"d={args.d_model}")
    print_budget_report(model, model_name)

    fits, details = check_budget(model)
    print(f"[train] Unique params: {details['unique_params']:,}")
    if hasattr(model, 'effective_depth'):
        print(f"[train] Effective depth: {model.effective_depth}")

    # torch.compile (after budget report)
    if args.compile and hasattr(torch, 'compile'):
        print("[train] Compiling model with torch.compile...")
        model = torch.compile(model)

    # Data
    print("[train] Loading training data...")
    train_data = load_fineweb_train(args.train_bytes)
    print(f"[train] Training data: {len(train_data):,} bytes")

    print("[train] Loading validation data...")
    val_data = load_fineweb_valid()
    print(f"[train] Validation data: {len(val_data):,} bytes")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Estimate total steps from wall-clock budget
    # We'll update this after first few steps to get better timing
    estimated_total_steps = 5000  # rough estimate, updated dynamically

    # CSV logging
    save_dir = Path(args.save_dir)
    results_dir = save_dir / "results"
    logs_dir = save_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"{args.exp_name}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "wall_time", "train_loss", "train_bpb",
                         "val_loss", "val_bpb", "lr"])

    # Training loop
    train_gen = make_byte_batches(train_data, args.batch_size, args.seq_len)

    model.train()
    start_time = time.time()
    step = 0
    best_val_bpb = float("inf")
    running_loss = 0.0
    running_count = 0

    grad_accum = args.grad_accum_steps
    effective_batch = args.batch_size * grad_accum
    print(f"\n[train] Starting training for {args.train_seconds}s...")
    print(f"[train] Batch size: {args.batch_size}, Seq len: {args.seq_len}, "
          f"Grad accum: {grad_accum} (effective batch: {effective_batch})")
    print(f"[train] LR: {args.lr}, Warmup: {args.warmup_steps} steps\n")

    while True:
        elapsed = time.time() - start_time

        # Check stopping conditions
        if args.max_steps > 0 and step >= args.max_steps:
            print(f"\n[train] Reached max steps ({args.max_steps})")
            break
        if elapsed >= args.train_seconds:
            print(f"\n[train] Time limit reached ({args.train_seconds}s)")
            break

        # Update LR estimate based on actual timing
        if step == 10:
            time_per_step = elapsed / step
            estimated_total_steps = int(args.train_seconds / time_per_step)
            print(f"[train] Estimated total steps: ~{estimated_total_steps} "
                  f"({time_per_step*1000:.0f}ms/step)")

        # LR schedule
        lr = get_lr(step, args.warmup_steps, args.lr, estimated_total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_step in range(grad_accum):
            x, y = next(train_gen)
            x, y = x.to(device), y.to(device)
            with dtype_ctx:
                _, loss = model(x, y)
            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            accum_loss += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Track running loss
        running_loss += accum_loss / grad_accum
        running_count += 1

        # Logging
        if step > 0 and step % args.log_every == 0:
            avg_loss = running_loss / running_count
            bpb = avg_loss / math.log(2)
            elapsed = time.time() - start_time
            print(f"  step {step:>5d} | loss {avg_loss:.4f} | bpb {bpb:.4f} | "
                  f"lr {lr:.2e} | {elapsed:.0f}s")
            csv_writer.writerow([step, f"{elapsed:.1f}", f"{avg_loss:.6f}",
                                f"{bpb:.6f}", "", "", f"{lr:.2e}"])
            csv_file.flush()
            running_loss = 0.0
            running_count = 0

        # Evaluation
        if step > 0 and step % args.eval_every == 0:
            val_loss, val_bpb = evaluate(model, val_data, args, device, dtype_ctx)
            elapsed = time.time() - start_time
            is_best = val_bpb < best_val_bpb
            if is_best:
                best_val_bpb = val_bpb
            marker = " *BEST*" if is_best else ""
            print(f"  [eval] step {step} | val_loss {val_loss:.4f} | "
                  f"val_bpb {val_bpb:.4f}{marker}")
            csv_writer.writerow([step, f"{elapsed:.1f}", "", "",
                                f"{val_loss:.6f}", f"{val_bpb:.6f}", f"{lr:.2e}"])
            csv_file.flush()

        step += 1

    # Final evaluation
    val_loss, val_bpb = evaluate(model, val_data, args, device, dtype_ctx)
    elapsed = time.time() - start_time
    print(f"\n[train] Final: val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} "
          f"steps={step} time={elapsed:.0f}s")
    csv_writer.writerow([step, f"{elapsed:.1f}", "", "",
                        f"{val_loss:.6f}", f"{val_bpb:.6f}", "0"])
    csv_file.close()

    # Save checkpoint
    ckpt_path = save_dir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_path / f"{args.exp_name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "step": step,
        "val_bpb": val_bpb,
        "val_loss": val_loss,
        "elapsed": elapsed,
    }, ckpt_file)
    print(f"[train] Checkpoint saved: {ckpt_file}")
    print(f"[train] Results CSV: {csv_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"  EXPERIMENT: {args.exp_name}")
    print(f"  Model: {args.model_type}, d={args.d_model}")
    if args.model_type == "shared":
        print(f"  Sharing: {args.n_unique_layers} blocks x {args.n_loops} loops "
              f"= {args.n_unique_layers * args.n_loops} effective")
    else:
        print(f"  Layers: {args.n_layers}")
    print(f"  Unique params: {details['unique_params']:,}")
    print(f"  Steps: {step}, Time: {elapsed:.0f}s")
    print(f"  Final val BPB: {val_bpb:.4f}")
    print(f"  Best val BPB: {min(best_val_bpb, val_bpb):.4f}")
    print(f"{'='*50}")

    # Artifact budget report
    n_params = details['unique_params']
    budget_mb = 16.0
    print(f"\n=== Artifact Budget ===")
    print(f"Params: {n_params:,}")
    for label, bpp in [("fp16", 16), ("int8", 8), ("int6", 6)]:
        size_mb = n_params * bpp / 8 / 1024 / 1024
        status = "OK" if size_mb <= budget_mb else "OVER BUDGET"
        print(f"At {label}: {size_mb:.1f} MB ({status})")
    int6_mb = n_params * 6 / 8 / 1024 / 1024
    zlib_est = int6_mb * 0.88
    status = "OK" if zlib_est <= budget_mb else "OVER BUDGET"
    print(f"At int6+zlib (est): ~{zlib_est:.1f} MB ({status})")
    print()

    return val_bpb


if __name__ == "__main__":
    main()
