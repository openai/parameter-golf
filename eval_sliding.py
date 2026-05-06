#!/usr/bin/env python3
"""
Sliding Window Eval for Parameter Golf.

Loads an existing quantized model and evaluates with overlapping windows.
Each scored token gets at least (seq_len - stride) context tokens.

Usage:
    source .venv/bin/activate
    EVAL_STRIDE=256 python3 eval_sliding.py
"""
from __future__ import annotations

import glob
import math
import os
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Import model and helpers from baseline
from train_gpt_mlx import (
    COMPUTE_DTYPE,
    GPT,
    Hyperparameters,
    MX_DTYPE_FROM_NAME,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    load_data_shard,
    rms_norm,
)


def load_quantized_model(model: GPT, quant_path: Path) -> None:
    """Load int8+zlib quantized model from disk."""
    with quant_path.open("rb") as f:
        quant_blob = f.read()
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob)))
    model.update(tree_unflatten(list(quant_flat.items())))


def load_validation_tokens(pattern: str) -> np.ndarray:
    """Load all validation shard tokens as a flat int32 array."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return np.ascontiguousarray(
        np.concatenate([load_data_shard(f) for f in files], axis=0)
    )


def eval_sliding(
    model: GPT,
    val_tokens: np.ndarray,
    seq_len: int,
    stride: int,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn=print,
) -> tuple[float, float]:
    """
    Sliding window evaluation.

    For each window of size seq_len, shifted by stride:
    - Feed the full window to the model
    - Only score the LAST `stride` tokens (they have maximal context)
    - Exception: the very first window scores all seq_len tokens

    This ensures every scored token (except the first seq_len) has at least
    (seq_len - stride) tokens of prior context.
    """
    n_tokens = val_tokens.size
    total_loss_sum = 0.0
    total_scored_tokens = 0
    total_bytes = 0.0

    # Precompute: the model's loss function returns mean CE, but we need per-position
    # losses. We'll compute logits directly and do CE manually.
    vocab_dim = model.tok_emb.weight.shape[1]

    # Generate window start positions
    # First window: start=0, score all positions
    # Subsequent windows: start += stride, score only last `stride` positions
    starts = list(range(0, n_tokens - seq_len, stride))
    if not starts:
        starts = [0]
    total_windows = len(starts)

    log_fn(f"sliding_eval: {total_windows} windows, seq_len={seq_len}, stride={stride}")
    log_fn(f"sliding_eval: context_guarantee={seq_len - stride} tokens")

    t0 = time.perf_counter()
    for win_idx, start in enumerate(starts):
        end = start + seq_len + 1  # +1 for target
        if end > n_tokens:
            break

        chunk = val_tokens[start:end]
        x_np = chunk[:-1].reshape(1, seq_len)  # [1, seq_len]
        y_np = chunk[1:].reshape(1, seq_len)    # [1, seq_len]

        x = mx.array(x_np, dtype=mx.int32)

        # Forward pass through model to get hidden states
        hidden = model(x)  # [1, seq_len, dim]
        hidden_flat = hidden.reshape(-1, vocab_dim)  # [seq_len, dim]

        # Compute logits with tied embedding
        logits = hidden_flat @ model.tok_emb.weight.astype(hidden_flat.dtype).T
        logits = model.softcap(logits)  # [seq_len, vocab]

        # Determine which positions to score
        if start == 0:
            # First window: score all positions
            score_start = 0
        else:
            # Subsequent windows: score only last `stride` positions
            score_start = seq_len - stride

        score_logits = logits[score_start:]  # [score_len, vocab]
        score_targets = mx.array(y_np[0, score_start:], dtype=mx.int32)

        # Per-token cross entropy (sum, not mean)
        per_token_ce = nn.losses.cross_entropy(
            score_logits.astype(mx.float32), score_targets, reduction="sum"
        )
        mx.eval(per_token_ce)

        n_scored = seq_len - score_start
        total_loss_sum += float(per_token_ce.item())
        total_scored_tokens += n_scored

        # BPB byte counting (same logic as baseline)
        score_prev_ids = x_np[0, score_start:]
        score_tgt_ids = y_np[0, score_start:]
        bytes_np = base_bytes_lut[score_tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[score_tgt_ids] & ~is_boundary_token_lut[score_prev_ids]
        ).astype(np.int16, copy=False)
        total_bytes += float(bytes_np.astype(np.float64).sum())

        # Progress logging
        if (win_idx + 1) % 500 == 0 or win_idx + 1 == total_windows:
            elapsed = time.perf_counter() - t0
            win_per_sec = (win_idx + 1) / elapsed
            eta = (total_windows - win_idx - 1) / max(win_per_sec, 1e-9)
            log_fn(
                f"sliding_eval: {win_idx + 1}/{total_windows} "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
            )

    val_loss = total_loss_sum / total_scored_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_scored_tokens / total_bytes)

    elapsed_total = time.perf_counter() - t0
    log_fn(f"sliding_eval: done in {elapsed_total:.0f}s")
    log_fn(f"sliding_eval: scored {total_scored_tokens:,} tokens, {total_bytes:.0f} bytes")

    return val_loss, val_bpb


def main():
    args = Hyperparameters()

    # Config
    seq_len = args.train_seq_len  # 1024
    stride = int(os.environ.get("EVAL_STRIDE", 256))
    quant_path = os.environ.get(
        "QUANT_MODEL_PATH",
        "logs/mlx_smoke_mlx_model.int8.ptz"
    )

    print(f"=== Sliding Window Eval ===")
    print(f"seq_len={seq_len}, stride={stride}")
    print(f"model={quant_path}")

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # Load validation tokens
    val_tokens = load_validation_tokens(args.val_files)
    print(f"val_tokens: {val_tokens.size:,}")

    # Create model and load quantized weights
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
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
    load_quantized_model(model, Path(quant_path))
    print("model loaded from quantized checkpoint")

    # First: run baseline eval (stride=seq_len, standard non-overlapping)
    print("\n--- Baseline eval (no overlap) ---")
    baseline_loss, baseline_bpb = eval_sliding(
        model, val_tokens, seq_len, stride=seq_len,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    print(f"baseline: val_loss={baseline_loss:.4f} val_bpb={baseline_bpb:.4f}")

    # Then: run sliding window eval
    print(f"\n--- Sliding window eval (stride={stride}) ---")
    sw_loss, sw_bpb = eval_sliding(
        model, val_tokens, seq_len, stride=stride,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    print(f"sliding: val_loss={sw_loss:.4f} val_bpb={sw_bpb:.4f}")

    # Comparison
    delta_bpb = sw_bpb - baseline_bpb
    print(f"\n=== RESULT ===")
    print(f"baseline val_bpb: {baseline_bpb:.6f}")
    print(f"sliding  val_bpb: {sw_bpb:.6f}")
    print(f"delta:           {delta_bpb:+.6f} ({delta_bpb / baseline_bpb * 100:+.2f}%)")


if __name__ == "__main__":
    main()
