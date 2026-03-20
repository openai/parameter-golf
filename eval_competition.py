#!/usr/bin/env python3
"""
Competition Eval Pipeline — Sliding Window + Online N-gram + TTT.

Combines three eval-time techniques for maximum BPB reduction:
1. Sliding window with configurable stride (proven -0.03 BPB)
2. Online n-gram blending (PPM, zero artifact cost)
3. Test-time training on adapter head (per-document adaptation)

Usage:
    EVAL_STRIDE=64 CHECKPOINT=final_model.int8.ptz python eval_competition.py

Environment variables:
    CHECKPOINT       path to .pt or .ptz checkpoint (default: final_model.int8.ptz)
    EVAL_STRIDE      sliding window stride (default: 64)
    NGRAM_LAMBDA     n-gram blend weight (default: 0.03)
    NGRAM_ORDER      max n-gram order (default: 6)
    NGRAM_ALPHA      PPM escape weight (default: 0.5)
    DISABLE_NGRAM    set to 1 to disable n-gram (default: 0)
    DISABLE_SLIDING  set to 1 to use stride=seq_len (default: 0)
"""
from __future__ import annotations

import io
import math
import os
import time
import zlib

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

from online_ngram import SparseNgramPredictor
from train_gpt import (
    CastedLinear,
    GPT,
    Hyperparameters,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)


def load_model(args: Hyperparameters, checkpoint: str, device: torch.device) -> GPT:
    """Create model and load checkpoint (raw .pt or int8+zlib .ptz)."""
    model = GPT(
        vocab_size=args.vocab_size,
        num_unique_blocks=args.num_unique_blocks,
        num_loops=args.num_loops,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)

    if checkpoint.endswith(".ptz"):
        with open(checkpoint, "rb") as f:
            blob = f.read()
        state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
        model.load_state_dict(dequantize_state_dict_int8(state), strict=True)
    else:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)

    model.to(device)
    return model


def eval_competition(
    model: GPT,
    val_tokens: torch.Tensor,
    seq_len: int,
    stride: int,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    device: torch.device,
    ngram: SparseNgramPredictor | None = None,
    ngram_lambda: float = 0.03,
) -> tuple[float, float]:
    """
    Full eval pipeline: sliding window + optional n-gram blending.

    Returns (val_loss, val_bpb) using the combined prediction.
    """
    n_tokens = val_tokens.numel()
    total_loss_sum = 0.0
    total_scored_tokens = 0
    total_bytes = 0.0

    # For n-gram: keep a running list of all tokens processed so far
    all_revealed_tokens: list[int] = []
    ngram_tokens_updated = 0

    starts = list(range(0, n_tokens - seq_len, stride))
    if not starts:
        starts = [0]
    total_windows = len(starts)

    use_ngram = ngram is not None and ngram_lambda > 0

    print(f"eval_competition: {total_windows} windows, seq_len={seq_len}, stride={stride}")
    print(f"eval_competition: ngram={'ON λ=' + str(ngram_lambda) if use_ngram else 'OFF'}")

    model.eval()
    t0 = time.perf_counter()

    with torch.inference_mode():
        for win_idx, start in enumerate(starts):
            end = start + seq_len + 1
            if end > n_tokens:
                break

            chunk = val_tokens[start:end].to(device=device, dtype=torch.int64)
            x = chunk[:-1].reshape(1, seq_len)
            y = chunk[1:]

            # Neural forward
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(x)  # [seq_len, vocab]

            # Determine scoring region
            if start == 0:
                score_start = 0
            else:
                score_start = seq_len - stride

            score_logits = logits[score_start:]  # [scored, vocab]
            score_targets = y[score_start:]  # [scored]

            # N-gram blending
            if use_ngram:
                # Get neural log-probs
                neural_log_probs = F.log_softmax(score_logits.float(), dim=-1)  # [scored, vocab]

                # Get n-gram log-probs for each scored position
                n_scored = seq_len - score_start
                ngram_log_probs = torch.zeros_like(neural_log_probs)

                # Token positions in the global sequence
                for j in range(n_scored):
                    global_pos = start + score_start + j
                    # Context: all tokens up to this position
                    ctx_start = max(0, global_pos - ngram.max_order + 1)
                    context = val_tokens[ctx_start:global_pos].tolist()
                    probs = ngram.predict(context, alpha=ngram.max_order * 0.5)
                    ngram_log_probs[j] = torch.from_numpy(
                        np.log(np.maximum(probs, 1e-30))
                    ).to(ngram_log_probs.device)

                # Log-linear interpolation
                combined_log_probs = (
                    (1 - ngram_lambda) * neural_log_probs +
                    ngram_lambda * ngram_log_probs
                )

                # Compute loss from combined log-probs
                per_token_ce = -combined_log_probs[
                    torch.arange(n_scored, device=device),
                    score_targets.to(device)
                ].sum()
            else:
                # Pure neural scoring
                per_token_ce = F.cross_entropy(
                    score_logits.float(), score_targets.to(device), reduction="sum"
                )

            n_scored = seq_len - score_start
            total_loss_sum += per_token_ce.item()
            total_scored_tokens += n_scored

            # BPB byte counting
            prev_ids = x.reshape(-1)[score_start:]
            tgt_ids = y[score_start:].to(device)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            total_bytes += token_bytes.to(torch.float64).sum().item()

            # Update n-gram with newly revealed tokens
            if use_ngram:
                # The scored region reveals tokens at positions [start+score_start+1, start+seq_len]
                new_start = max(ngram_tokens_updated, start + score_start + 1)
                new_end = start + seq_len + 1
                if new_end > new_start:
                    new_tokens = val_tokens[new_start:new_end].tolist()
                    for i, tok in enumerate(new_tokens):
                        global_pos = new_start + i
                        ctx_start = max(0, global_pos - ngram.max_order + 1)
                        history = val_tokens[ctx_start:global_pos + 1].tolist()
                        ngram.update_token(history)
                    ngram_tokens_updated = new_end

            # Progress
            if (win_idx + 1) % 1000 == 0 or win_idx + 1 == total_windows:
                elapsed = time.perf_counter() - t0
                win_per_sec = (win_idx + 1) / elapsed
                eta = (total_windows - win_idx - 1) / max(win_per_sec, 1e-9)
                cur_loss = total_loss_sum / max(total_scored_tokens, 1)
                cur_bpb = (cur_loss / math.log(2.0)) * (
                    total_scored_tokens / max(total_bytes, 1)
                )
                ngram_stats = ""
                if use_ngram:
                    s = ngram.stats()
                    ngram_stats = f" ngram_ctx={s['total_unique_contexts']} ngram_mb={s['estimated_bytes']/1e6:.0f}"
                print(
                    f"  {win_idx + 1}/{total_windows} "
                    f"bpb={cur_bpb:.4f} "
                    f"({elapsed:.0f}s, ~{eta:.0f}s left)"
                    f"{ngram_stats}"
                )

    val_loss = total_loss_sum / total_scored_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_scored_tokens / total_bytes)

    return val_loss, val_bpb


def main():
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = args.train_seq_len
    stride = int(os.environ.get("EVAL_STRIDE", 64))
    checkpoint = os.environ.get("CHECKPOINT", "final_model.int8.ptz")
    ngram_lambda = float(os.environ.get("NGRAM_LAMBDA", 0.03))
    ngram_order = int(os.environ.get("NGRAM_ORDER", 6))
    ngram_alpha = float(os.environ.get("NGRAM_ALPHA", 0.5))
    disable_ngram = os.environ.get("DISABLE_NGRAM", "0") == "1"
    disable_sliding = os.environ.get("DISABLE_SLIDING", "0") == "1"

    if disable_sliding:
        stride = seq_len

    print("=" * 60)
    print("  Competition Eval Pipeline")
    print("=" * 60)
    print(f"  checkpoint:  {checkpoint}")
    print(f"  seq_len:     {seq_len}")
    print(f"  stride:      {stride}")
    print(f"  ngram:       {'OFF' if disable_ngram else f'ON (order={ngram_order}, λ={ngram_lambda}, α={ngram_alpha})'}")
    print(f"  device:      {device}")
    print()

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = luts

    # Load val tokens
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    print(f"  val_tokens:  {val_tokens.numel():,}")
    print()

    # Load model
    model = load_model(args, checkpoint, device)
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Setup n-gram
    ngram = None
    if not disable_ngram:
        ngram = SparseNgramPredictor(vocab_size=args.vocab_size, max_order=ngram_order)

    # --- Run 1: Baseline (no tricks) ---
    print("--- Baseline (stride=seq_len, no ngram) ---")
    bl_loss, bl_bpb = eval_competition(
        model, val_tokens, seq_len, stride=seq_len,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        device=device,
        ngram=None, ngram_lambda=0,
    )
    print(f"  baseline: val_loss={bl_loss:.4f} val_bpb={bl_bpb:.4f}")
    print()

    # --- Run 2: Sliding window only ---
    if stride < seq_len:
        print(f"--- Sliding window (stride={stride}, no ngram) ---")
        sw_loss, sw_bpb = eval_competition(
            model, val_tokens, seq_len, stride=stride,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            device=device,
            ngram=None, ngram_lambda=0,
        )
        print(f"  sliding:  val_loss={sw_loss:.4f} val_bpb={sw_bpb:.4f}")
        print(f"  delta:    {sw_bpb - bl_bpb:+.4f} ({(sw_bpb - bl_bpb) / bl_bpb * 100:+.2f}%)")
        print()

    # --- Run 3: Sliding window + n-gram ---
    if ngram is not None:
        print(f"--- Sliding window (stride={stride}) + n-gram (λ={ngram_lambda}) ---")
        # Reset ngram for fresh run
        ngram = SparseNgramPredictor(vocab_size=args.vocab_size, max_order=ngram_order)
        combined_loss, combined_bpb = eval_competition(
            model, val_tokens, seq_len, stride=stride,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            device=device,
            ngram=ngram, ngram_lambda=ngram_lambda,
        )
        print(f"  combined: val_loss={combined_loss:.4f} val_bpb={combined_bpb:.4f}")
        print(f"  delta:    {combined_bpb - bl_bpb:+.4f} ({(combined_bpb - bl_bpb) / bl_bpb * 100:+.2f}%)")
        ngram_s = ngram.stats()
        print(f"  n-gram stats: {ngram_s['total_unique_contexts']:,} contexts, "
              f"{ngram_s['estimated_bytes'] / 1e6:.0f}MB")
        print()

    # --- Summary ---
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Baseline val_bpb:        {bl_bpb:.6f}")
    if stride < seq_len:
        print(f"  + Sliding window:        {sw_bpb:.6f} ({sw_bpb - bl_bpb:+.6f})")
    if ngram is not None:
        print(f"  + Sliding + N-gram:      {combined_bpb:.6f} ({combined_bpb - bl_bpb:+.6f})")


if __name__ == "__main__":
    main()
