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
from match_model import MatchModel
from train_gpt import (
    CastedLinear,
    GPT,
    Hyperparameters,
    Rotary,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)


try:
    import zstandard as zstd_mod
    HAS_ZSTD_EVAL = True
except ImportError:
    HAS_ZSTD_EVAL = False


def load_model(args: Hyperparameters, checkpoint: str, device: torch.device) -> GPT:
    """Create model and load checkpoint (raw .pt or int8/int6+zlib/zstd .ptz)."""
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
        # Try zstd first (for int6), fall back to zlib (for int8)
        try:
            if HAS_ZSTD_EVAL:
                dctx = zstd_mod.ZstdDecompressor()
                raw = dctx.decompress(blob)
            else:
                raw = zlib.decompress(blob)
        except Exception:
            raw = zlib.decompress(blob)
        state = torch.load(io.BytesIO(raw), map_location="cpu")
        model.load_state_dict(dequantize_state_dict_int8(state), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)

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
    match_model: MatchModel | None = None,
    match_lambda: float = 0.0,  # 0 = disabled. Recommended: 0.05 - 0.15
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
    use_match = match_model is not None and match_lambda > 0
    use_blending = use_ngram or use_match

    # --- Exponential Weights Mixer ---
    # Models: 0 = neural, 1 = ngram, 2 = match
    # Weights update multiplicatively: w_i *= p_i(actual_token) ^ learning_rate
    # This gives O(log K) regret — converges to best model automatically.
    n_models = 1 + int(use_ngram) + int(use_match)
    model_names = ["neural"]
    # Initialize weights: neural gets remainder, others get their lambda
    mix_w = [1.0]
    if use_ngram:
        model_names.append("ngram")
        mix_w.append(ngram_lambda)
        mix_w[0] -= ngram_lambda
    if use_match:
        model_names.append("match")
        mix_w.append(match_lambda)
        mix_w[0] -= match_lambda
    mix_w = [max(w, 0.01) for w in mix_w]  # clamp to avoid zero weights
    w_sum = sum(mix_w)
    mix_w = [w / w_sum for w in mix_w]  # normalize

    # Learning rate for weight updates (higher = faster adaptation)
    mix_lr = 1.0  # standard exponential weights
    mix_update_count = 0
    mix_weight_history: list[list[float]] = []  # for stats

    print(f"eval_competition: {total_windows} windows, seq_len={seq_len}, stride={stride}")
    print(f"eval_competition: ngram={'ON λ=' + str(ngram_lambda) if use_ngram else 'OFF'}")
    print(f"eval_competition: match={'ON λ=' + str(match_lambda) if use_match else 'OFF'}")
    if use_blending:
        print(f"eval_competition: exponential_weights init={dict(zip(model_names, [f'{w:.3f}' for w in mix_w]))}")

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

            # Model blending with exponential weights
            if use_blending:
                # Get neural probs
                neural_probs = F.softmax(score_logits.float(), dim=-1)  # [scored, vocab]
                n_scored = seq_len - score_start

                # Per-token blending with current mix weights
                combined_probs = neural_probs.clone()

                for j in range(n_scored):
                    global_pos = start + score_start + j

                    # Collect per-model predictions for this token
                    model_preds = [neural_probs[j]]  # neural always present

                    if use_ngram:
                        ctx_start = max(0, global_pos - ngram.max_order + 1)
                        context = val_tokens[ctx_start:global_pos].tolist()
                        ng_probs = ngram.predict(context, alpha=ngram.max_order * 0.5)
                        ng_probs_t = torch.from_numpy(ng_probs).to(
                            device=neural_probs.device, dtype=neural_probs.dtype
                        )
                        model_preds.append(ng_probs_t)

                    if use_match:
                        ctx_start = max(0, global_pos - match_model.max_order)
                        context = val_tokens[ctx_start:global_pos].tolist()
                        m_probs, match_len = match_model.predict(context)
                        if m_probs is not None and match_len >= match_model.min_order:
                            m_probs_t = torch.from_numpy(m_probs).to(
                                device=neural_probs.device, dtype=neural_probs.dtype
                            )
                            model_preds.append(m_probs_t)
                        else:
                            # No match — use neural as fallback for this slot
                            model_preds.append(neural_probs[j])

                    # Weighted mixture using current exponential weights
                    mixed = torch.zeros_like(neural_probs[j])
                    for m_idx, pred in enumerate(model_preds):
                        mixed += mix_w[m_idx] * pred
                    combined_probs[j] = mixed

                combined_log_probs = torch.log(torch.clamp(combined_probs, min=1e-30))

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

            # Update n-gram, match model, AND exponential weights with revealed tokens
            if use_ngram or use_match:
                new_start = max(ngram_tokens_updated, start + score_start + 1)
                new_end = start + seq_len + 1
                if new_end > new_start:
                    new_tokens = val_tokens[new_start:new_end].tolist()
                    for i, tok in enumerate(new_tokens):
                        global_pos = new_start + i

                        # --- Exponential weights update ---
                        # Each model's probability for the actual token
                        # determines how much its weight increases
                        model_scores = []

                        # Neural score (from cached logits if in scoring region, else skip)
                        rel_pos = global_pos - start - 1  # position in the window
                        if 0 <= rel_pos < seq_len:
                            neural_p = F.softmax(logits[rel_pos:rel_pos+1].float(), dim=-1)
                            neural_score = neural_p[0, tok].item()
                        else:
                            neural_score = 1.0 / 1024  # uniform fallback
                        model_scores.append(max(neural_score, 1e-10))

                        if use_ngram:
                            ctx_s = max(0, global_pos - ngram.max_order + 1)
                            ctx = val_tokens[ctx_s:global_pos].tolist()
                            ng_p = ngram.predict(ctx, alpha=ngram.max_order * 0.5)
                            model_scores.append(max(float(ng_p[tok]), 1e-10))

                        if use_match:
                            ctx_s = max(0, global_pos - match_model.max_order)
                            ctx = val_tokens[ctx_s:global_pos].tolist()
                            m_p, _ = match_model.predict(ctx)
                            if m_p is not None:
                                model_scores.append(max(float(m_p[tok]), 1e-10))
                            else:
                                model_scores.append(max(neural_score, 1e-10))

                        # Multiplicative weight update: w_i *= p_i(actual)^lr
                        for m_idx in range(len(mix_w)):
                            mix_w[m_idx] *= model_scores[m_idx] ** mix_lr
                        # Floor to prevent weight collapse (sub-models stay ≥1%)
                        mix_w = [max(w, 0.01) for w in mix_w]
                        # Renormalize
                        w_total = sum(mix_w)
                        mix_w = [w / w_total for w in mix_w]
                        mix_update_count += 1

                        # Update sub-models
                        if use_ngram:
                            ctx_start = max(0, global_pos - ngram.max_order + 1)
                            history = val_tokens[ctx_start:global_pos + 1].tolist()
                            ngram.update_token(history)
                        if use_match:
                            match_model.update(tok)
                    ngram_tokens_updated = new_end

                    # Log weight snapshot periodically
                    if mix_update_count > 0 and mix_update_count % 50000 == 0:
                        mix_weight_history.append(list(mix_w))

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
                match_stats = ""
                if use_match:
                    ms = match_model.stats()
                    match_stats = f" match_rate={ms['match_rate']:.2%} avg_len={ms['avg_match_length']:.1f}"
                mix_info = ""
                if use_blending:
                    mix_info = " w=[" + "/".join(f"{w:.3f}" for w in mix_w) + "]"
                print(
                    f"  {win_idx + 1}/{total_windows} "
                    f"bpb={cur_bpb:.4f} "
                    f"({elapsed:.0f}s, ~{eta:.0f}s left)"
                    f"{ngram_stats}{match_stats}{mix_info}"
                )

    val_loss = total_loss_sum / total_scored_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_scored_tokens / total_bytes)

    # Print final exponential weights
    if use_blending:
        print(f"  final_mix_weights: {dict(zip(model_names, [f'{w:.4f}' for w in mix_w]))}")
        if mix_weight_history:
            print(f"  weight_snapshots ({len(mix_weight_history)}):")
            for snap in mix_weight_history[-3:]:
                print(f"    {dict(zip(model_names, [f'{w:.4f}' for w in snap]))}")

    return val_loss, val_bpb


def main():
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seq_len = args.train_seq_len
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", train_seq_len))
    seq_len = eval_seq_len  # use eval seq len for all runs
    stride = int(os.environ.get("EVAL_STRIDE", 64))
    checkpoint = os.environ.get("CHECKPOINT", "final_model.int8.ptz")
    ngram_lambda = float(os.environ.get("NGRAM_LAMBDA", 0.03))
    ngram_order = int(os.environ.get("NGRAM_ORDER", 6))
    ngram_alpha = float(os.environ.get("NGRAM_ALPHA", 0.5))
    match_lambda = float(os.environ.get("MATCH_LAMBDA", 0.1))
    match_min_order = int(os.environ.get("MATCH_MIN_ORDER", 4))
    match_max_order = int(os.environ.get("MATCH_MAX_ORDER", 12))
    disable_ngram = os.environ.get("DISABLE_NGRAM", "0") == "1"
    disable_sliding = os.environ.get("DISABLE_SLIDING", "0") == "1"
    disable_match = os.environ.get("DISABLE_MATCH", "0") == "1"

    if disable_sliding:
        stride = seq_len

    print("=" * 60)
    print("  Competition Eval Pipeline")
    print("=" * 60)
    print(f"  checkpoint:  {checkpoint}")
    print(f"  train_seq:   {train_seq_len}")
    print(f"  eval_seq:    {eval_seq_len}{' (NTK-RoPE extended!)' if eval_seq_len > train_seq_len else ''}")
    print(f"  stride:      {stride}")
    print(f"  ngram:       {'OFF' if disable_ngram else f'ON (order={ngram_order}, λ={ngram_lambda}, α={ngram_alpha})'}")
    print(f"  match:       {'OFF' if disable_match else f'ON (order={match_min_order}-{match_max_order}, λ={match_lambda})'}")
    print(f"  device:      {device}")
    print()

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = luts

    # Load val tokens (use eval_seq_len for chunking)
    val_tokens = load_validation_tokens(args.val_files, eval_seq_len)
    print(f"  val_tokens:  {val_tokens.numel():,}")
    print()

    # Load model
    model = load_model(args, checkpoint, device)
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")

    # NTK-RoPE rescaling for extended eval sequences
    if eval_seq_len > train_seq_len:
        for m in model.modules():
            if isinstance(m, Rotary):
                m.rescale_base(eval_seq_len, train_seq_len)
        print(f"  NTK-RoPE:    rescaled for {eval_seq_len} (train={train_seq_len})")
    print()

    # Setup n-gram
    ngram = None
    if not disable_ngram:
        ngram = SparseNgramPredictor(vocab_size=args.vocab_size, max_order=ngram_order)

    # Setup match model
    mm = None
    if not disable_match:
        mm = MatchModel(vocab_size=args.vocab_size, min_order=match_min_order, max_order=match_max_order)

    # --- Run 1: Baseline (no tricks) ---
    print("--- Baseline (stride=seq_len, no ngram, no match) ---")
    bl_loss, bl_bpb = eval_competition(
        model, val_tokens, seq_len, stride=seq_len,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        device=device,
        ngram=None, ngram_lambda=0,
        match_model=None, match_lambda=0,
    )
    print(f"  baseline: val_loss={bl_loss:.4f} val_bpb={bl_bpb:.4f}")
    print()

    # --- Run 2: Sliding window only ---
    if stride < seq_len:
        print(f"--- Sliding window (stride={stride}) ---")
        sw_loss, sw_bpb = eval_competition(
            model, val_tokens, seq_len, stride=stride,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            device=device,
            ngram=None, ngram_lambda=0,
            match_model=None, match_lambda=0,
        )
        print(f"  sliding:  val_loss={sw_loss:.4f} val_bpb={sw_bpb:.4f}")
        print(f"  delta:    {sw_bpb - bl_bpb:+.4f} ({(sw_bpb - bl_bpb) / bl_bpb * 100:+.2f}%)")
        print()

    # --- Run 3: Sliding window + n-gram ---
    if ngram is not None:
        print(f"--- Sliding window (stride={stride}) + n-gram (λ={ngram_lambda}) ---")
        ngram = SparseNgramPredictor(vocab_size=args.vocab_size, max_order=ngram_order)
        ng_loss, ng_bpb = eval_competition(
            model, val_tokens, seq_len, stride=stride,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            device=device,
            ngram=ngram, ngram_lambda=ngram_lambda,
            match_model=None, match_lambda=0,
        )
        print(f"  +ngram:   val_loss={ng_loss:.4f} val_bpb={ng_bpb:.4f}")
        print(f"  delta:    {ng_bpb - bl_bpb:+.4f} ({(ng_bpb - bl_bpb) / bl_bpb * 100:+.2f}%)")
        ngram_s = ngram.stats()
        print(f"  n-gram stats: {ngram_s['total_unique_contexts']:,} contexts, "
              f"{ngram_s['estimated_bytes'] / 1e6:.0f}MB")
        print()

    # --- Run 4: Sliding window + match model ---
    if mm is not None:
        print(f"--- Sliding window (stride={stride}) + match (λ={match_lambda}) ---")
        mm_fresh = MatchModel(vocab_size=args.vocab_size, min_order=match_min_order, max_order=match_max_order)
        mm_loss, mm_bpb = eval_competition(
            model, val_tokens, seq_len, stride=stride,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            device=device,
            ngram=None, ngram_lambda=0,
            match_model=mm_fresh, match_lambda=match_lambda,
        )
        print(f"  +match:   val_loss={mm_loss:.4f} val_bpb={mm_bpb:.4f}")
        print(f"  delta:    {mm_bpb - bl_bpb:+.4f} ({(mm_bpb - bl_bpb) / bl_bpb * 100:+.2f}%)")
        ms = mm_fresh.stats()
        print(f"  match stats: rate={ms['match_rate']:.2%} avg_len={ms['avg_match_length']:.1f} "
              f"max_len={ms['max_match_length']} mem={ms['estimated_memory_mb']:.0f}MB")
        print()

    # --- Run 5: Full combo (sliding + n-gram + match) ---
    if ngram is not None and mm is not None:
        print(f"--- FULL: sliding + ngram (λ={ngram_lambda}) + match (λ={match_lambda}) ---")
        ngram_full = SparseNgramPredictor(vocab_size=args.vocab_size, max_order=ngram_order)
        mm_full = MatchModel(vocab_size=args.vocab_size, min_order=match_min_order, max_order=match_max_order)
        full_loss, full_bpb = eval_competition(
            model, val_tokens, seq_len, stride=stride,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            device=device,
            ngram=ngram_full, ngram_lambda=ngram_lambda,
            match_model=mm_full, match_lambda=match_lambda,
        )
        print(f"  FULL:     val_loss={full_loss:.4f} val_bpb={full_bpb:.4f}")
        print(f"  delta:    {full_bpb - bl_bpb:+.4f} ({(full_bpb - bl_bpb) / bl_bpb * 100:+.2f}%)")
        print()

    # --- Summary ---
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Baseline val_bpb:        {bl_bpb:.6f}")
    if stride < seq_len:
        print(f"  + Sliding window:        {sw_bpb:.6f} ({sw_bpb - bl_bpb:+.6f})")
    if ngram is not None:
        print(f"  + Sliding + N-gram:      {ng_bpb:.6f} ({ng_bpb - bl_bpb:+.6f})")
    if mm is not None:
        print(f"  + Sliding + Match:       {mm_bpb:.6f} ({mm_bpb - bl_bpb:+.6f})")
    if ngram is not None and mm is not None:
        print(f"  + FULL (all combined):   {full_bpb:.6f} ({full_bpb - bl_bpb:+.6f})")


if __name__ == "__main__":
    main()
