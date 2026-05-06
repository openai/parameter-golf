#!/usr/bin/env python3
"""
Standalone TTT (Test-Time Training) experiment.

Steps:
  1. Quick-train a 3×3 dim=768 GPT for N steps (default 400) to get a baseline model.
  2. Save checkpoint.
  3. Run streaming TTT on first 100 multi-chunk documents from val set.
  4. Report: static_bpb vs ttt_bpb, Δ, per-LR sweep.

Usage:
  # Quick run (400 steps train + TTT on 100 docs):
  python run_ttt_test.py

  # Load existing checkpoint, skip training:
  LOAD_CHECKPOINT=logs/my_model.npz python run_ttt_test.py

  # LR sweep:
  python run_ttt_test.py --lr_sweep
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# Import everything from the v2 training script
sys.path.insert(0, str(Path(__file__).parent))
from train_gpt_mlx_v2 import (
    GPT,
    Hyperparameters,
    TokenLoader,
    SplitOptimizers,
    build_sentencepiece_luts,
    load_validation_tokens,
    validate_dataset_tokenizer_pair,
    loss_and_grad_chunked,
    eval_val,
    eval_val_ttt,
    COMPUTE_DTYPE,
)
import sentencepiece as spm


# ── Configuration ────────────────────────────────────────────────────────────

TRAIN_STEPS      = int(os.environ.get("TRAIN_STEPS", 400))
TTT_NUM_DOCS     = int(os.environ.get("TTT_NUM_DOCS", 100))
TTT_CHUNK_SIZE   = int(os.environ.get("TTT_CHUNK_SIZE", 256))
LOAD_CHECKPOINT  = os.environ.get("LOAD_CHECKPOINT", "")  # path to .npz if skipping train
OUT_DIR          = Path(os.environ.get("OUT_DIR", "logs"))
LR_SWEEP         = os.environ.get("LR_SWEEP", "0") != "0"

# LR candidates for sweep
LR_CANDIDATES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_model(args: Hyperparameters) -> GPT:
    return GPT(
        vocab_size=args.vocab_size,
        num_unique_blocks=args.num_unique_blocks,
        num_loops=args.num_loops,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )


def quick_train(
    model: GPT,
    args: Hyperparameters,
    train_loader: TokenLoader,
    steps: int,
) -> None:
    """Train model in-place for `steps` steps."""
    opt = SplitOptimizers(model, args)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )

    t0 = time.time()
    for step in range(1, steps + 1):
        lr_mul = args.lr_mul(step, (time.time() - t0) * 1000)
        loss_val, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.eval(model.parameters())
        if step == 1 or step % 50 == 0 or step == steps:
            elapsed = time.time() - t0
            log(f"train step:{step}/{steps} loss:{loss_val.item():.4f} elapsed:{elapsed:.0f}s")

    log(f"Training done ({steps} steps, {time.time() - t0:.1f}s total)")


def save_checkpoint(model: GPT, path: Path) -> None:
    flat = dict(tree_flatten(model.parameters()))
    mx.savez(str(path), **flat)
    log(f"Saved checkpoint: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_checkpoint(model: GPT, path: str) -> None:
    data = dict(np.load(path))
    items = []
    for k, v in data.items():
        if v.dtype.kind == 'V':  # void = bfloat16 stored by mlx
            v = mx.array(v.view(np.uint16)).view(mx.bfloat16)
        else:
            v = mx.array(v)
        items.append((k, v))
    params = tree_unflatten(items)
    model.update(params)
    mx.eval(model.parameters())
    log(f"Loaded checkpoint: {path}")


def run_ttt_experiment(
    model: GPT,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    ttt_lr: float,
    num_docs: int = 100,
    chunk_size: int = 256,
) -> tuple[float, float, float, float]:
    """Run TTT experiment and return (static_loss, static_bpb, ttt_loss, ttt_bpb)."""
    t0 = time.time()
    results = eval_val_ttt(
        model=model,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        num_docs=num_docs,
        chunk_size=chunk_size,
        ttt_lr=ttt_lr,
        log_fn=log,
    )
    elapsed = time.time() - t0
    static_loss, static_bpb, ttt_loss, ttt_bpb = results
    delta_bpb = ttt_bpb - static_bpb
    delta_pct = 100 * delta_bpb / static_bpb if static_bpb > 0 else 0.0
    log(f"TTT lr={ttt_lr:.0e} elapsed={elapsed:.1f}s")
    log(f"  static: loss={static_loss:.4f} bpb={static_bpb:.4f}")
    log(f"  ttt:    loss={ttt_loss:.4f} bpb={ttt_bpb:.4f}")
    log(f"  delta:  bpb={delta_bpb:+.4f} ({delta_pct:+.2f}%)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="TTT experiment runner")
    parser.add_argument("--lr_sweep", action="store_true", help="Sweep multiple LR values")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS, help="Training steps")
    parser.add_argument("--docs", type=int, default=TTT_NUM_DOCS, help="TTT docs")
    parser.add_argument("--chunk", type=int, default=TTT_CHUNK_SIZE, help="Chunk size")
    parser.add_argument("--lr", type=float, default=1e-3, help="TTT learning rate (non-sweep)")
    parser.add_argument("--load", type=str, default=LOAD_CHECKPOINT, help="Load checkpoint path")
    cli = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Setup args & tokenizer ────────────────────────────────────────────────
    args = Hyperparameters()
    # Use smaller val batch for speed (TTT doesn't use eval_val's batch logic)
    args.grad_accum_steps = 4

    log(f"Config: num_unique_blocks={args.num_unique_blocks} num_loops={args.num_loops} "
        f"dim={args.model_dim} heads={args.num_heads} kv_heads={args.num_kv_heads}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)

    # ── Load validation data ──────────────────────────────────────────────────
    log("Loading validation tokens...")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    log(f"Val tokens: {val_tokens.size:,}")

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # ── Build model ───────────────────────────────────────────────────────────
    mx.random.seed(args.seed)
    model = build_model(args)
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"Model params: {n_params:,}")

    # ── Train or load checkpoint ──────────────────────────────────────────────
    if cli.load:
        load_checkpoint(model, cli.load)
    elif cli.steps > 0:
        log(f"Quick-training for {cli.steps} steps...")
        dataset_name, _, _ = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)
        quick_train(model, args, train_loader, steps=cli.steps)
        ckpt_path = OUT_DIR / f"ttt_test_{cli.steps}steps.npz"
        save_checkpoint(model, ckpt_path)
    else:
        log("Using randomly initialized model (no training, no checkpoint)")

    # ── Quick static BPB on val (first ~5% for speed) ─────────────────────────
    log("Running static BPB on subset of val set...")
    compiled_loss = mx.compile(
        lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state
    )

    # Use a smaller subset: first 50k tokens
    mini_val = val_tokens[:50001]
    mini_args = Hyperparameters()
    mini_args.val_batch_size = 8192
    mini_args.grad_accum_steps = 1
    mini_args.train_seq_len = args.train_seq_len
    static_val_loss, static_val_bpb = eval_val(
        mini_args, compiled_loss, mini_val,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        log_fn=log,
    )
    log(f"Static baseline (50k tok subset): loss={static_val_loss:.4f} bpb={static_val_bpb:.4f}")

    # ── TTT experiment ────────────────────────────────────────────────────────
    log("=" * 60)
    log(f"Starting TTT experiment: {cli.docs} docs, chunk_size={cli.chunk}")
    log("=" * 60)

    if cli.lr_sweep:
        log(f"LR sweep: {LR_CANDIDATES}")
        results_table = []
        for lr in LR_CANDIDATES:
            log(f"\n--- LR={lr:.0e} ---")
            r = run_ttt_experiment(
                model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                ttt_lr=lr, num_docs=cli.docs, chunk_size=cli.chunk,
            )
            results_table.append((lr, r))

        log("\n" + "=" * 60)
        log("LR SWEEP SUMMARY")
        log(f"{'LR':>10} {'static_bpb':>12} {'ttt_bpb':>12} {'delta':>10} {'delta%':>8}")
        for lr, (sl, sb, tl, tb) in results_table:
            delta = tb - sb
            pct = 100 * delta / sb if sb > 0 else 0
            log(f"{lr:>10.0e} {sb:>12.4f} {tb:>12.4f} {delta:>+10.4f} {pct:>+7.2f}%")
        log("=" * 60)
    else:
        run_ttt_experiment(
            model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            ttt_lr=cli.lr, num_docs=cli.docs, chunk_size=cli.chunk,
        )

    log("Done.")


if __name__ == "__main__":
    main()
