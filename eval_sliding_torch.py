#!/usr/bin/env python3
"""
Sliding Window Eval (PyTorch/CUDA) for Parameter Golf.

Loads a trained model checkpoint and evaluates with overlapping windows.
Each scored token gets at least (seq_len - stride) context tokens.

Usage:
    python eval_sliding_torch.py
    EVAL_STRIDE=64 CHECKPOINT=final_model.pt python eval_sliding_torch.py
    CHECKPOINT=final_model.int8.ptz python eval_sliding_torch.py
"""
from __future__ import annotations

import io
import math
import os
import sys
import time
import zlib

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

# Import from train_gpt.py
from train_gpt import (
    GPT,
    Hyperparameters,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    load_validation_tokens,
)


def eval_sliding(
    model: GPT,
    val_tokens: Tensor,
    seq_len: int,
    stride: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """
    Sliding window evaluation.

    For each window of size seq_len, shifted by stride:
    - Feed the full window to the model
    - Only score the LAST `stride` tokens (they have maximal context)
    - Exception: the very first window scores all seq_len tokens
    """
    n_tokens = val_tokens.numel()
    total_loss_sum = 0.0
    total_scored_tokens = 0
    total_bytes = 0.0

    starts = list(range(0, n_tokens - seq_len, stride))
    if not starts:
        starts = [0]
    total_windows = len(starts)

    print(f"sliding_eval: {total_windows} windows, seq_len={seq_len}, stride={stride}")
    print(f"sliding_eval: context_guarantee={seq_len - stride} tokens")

    model.eval()
    t0 = time.perf_counter()

    with torch.inference_mode():
        for win_idx, start in enumerate(starts):
            end = start + seq_len + 1
            if end > n_tokens:
                break

            chunk = val_tokens[start:end].to(device=device, dtype=torch.int64)
            x = chunk[:-1].reshape(1, seq_len)
            y = chunk[1:].reshape(1, seq_len)

            # Forward: we need per-token losses, not mean
            # Replicate the GPT forward but get per-token CE
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                h = model.tok_emb(x)
                h = F.rms_norm(h, (h.size(-1),))
                x0 = h
                dim = h.size(-1)

                for loop_idx in range(model.num_loops):
                    adaln = model.adaln_params[loop_idx].to(dtype=h.dtype)
                    scale = adaln[:dim]
                    shift = adaln[dim:]
                    h = F.rms_norm(h, (dim,)) * (1.0 + scale[None, None, :]) + shift[None, None, :]

                    skips = []
                    for bi in range(model.num_encoder_blocks):
                        h = model.blocks[bi](h, x0)
                        skips.append(h)
                    for bi in range(model.num_encoder_blocks, model.num_unique_blocks):
                        di = bi - model.num_encoder_blocks
                        if di < model.num_skip_per_cycle and skips:
                            sw = model.skip_weights[loop_idx, di].to(dtype=h.dtype)
                            h = h + sw[None, None, :] * skips.pop()
                        h = model.blocks[bi](h, x0)

                    gate = torch.sigmoid(model.cycle_gates[loop_idx].to(dtype=h.dtype))
                    h = gate[None, None, :] * h + (1.0 - gate[None, None, :]) * x0

                h = model.final_norm(h).reshape(-1, h.size(-1))
                logits_proj = F.linear(h, model.tok_emb.weight)
                logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)

            # Determine which positions to score
            if start == 0:
                score_start = 0
            else:
                score_start = seq_len - stride

            score_logits = logits[score_start:]
            score_targets = y[0, score_start:]

            # Per-token cross entropy (sum)
            per_token_ce = F.cross_entropy(
                score_logits.float(), score_targets, reduction="sum"
            )

            n_scored = seq_len - score_start
            total_loss_sum += per_token_ce.item()
            total_scored_tokens += n_scored

            # BPB byte counting
            prev_ids = x[0, score_start:]
            tgt_ids = y[0, score_start:]
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            total_bytes += token_bytes.to(torch.float64).sum().item()

            if (win_idx + 1) % 500 == 0 or win_idx + 1 == total_windows:
                elapsed = time.perf_counter() - t0
                win_per_sec = (win_idx + 1) / elapsed
                eta = (total_windows - win_idx - 1) / max(win_per_sec, 1e-9)
                print(
                    f"sliding_eval: {win_idx + 1}/{total_windows} "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                )

    val_loss = total_loss_sum / total_scored_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_scored_tokens / total_bytes)

    elapsed_total = time.perf_counter() - t0
    print(f"sliding_eval: done in {elapsed_total:.0f}s")
    print(f"sliding_eval: scored {total_scored_tokens:,} tokens, {total_bytes:.0f} bytes")

    return val_loss, val_bpb


def main():
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = args.train_seq_len
    stride = int(os.environ.get("EVAL_STRIDE", 256))
    checkpoint = os.environ.get("CHECKPOINT", "final_model.int8.ptz")

    print(f"=== Sliding Window Eval (PyTorch) ===")
    print(f"seq_len={seq_len}, stride={stride}")
    print(f"checkpoint={checkpoint}")
    print(f"device={device}")

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Load validation tokens
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    print(f"val_tokens: {val_tokens.numel():,}")

    # Create model
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

    # Load checkpoint
    if checkpoint.endswith(".ptz"):
        print("Loading int8+zlib quantized checkpoint...")
        with open(checkpoint, "rb") as f:
            quant_blob = f.read()
        quant_state = torch.load(
            io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu"
        )
        model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    else:
        print("Loading raw checkpoint...")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)

    model.to(device)
    print(f"model loaded, params={sum(p.numel() for p in model.parameters()):,}")

    # Baseline eval (no overlap)
    print("\n--- Baseline eval (no overlap) ---")
    baseline_loss, baseline_bpb = eval_sliding(
        model, val_tokens, seq_len, stride=seq_len,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        device=device,
    )
    print(f"baseline: val_loss={baseline_loss:.4f} val_bpb={baseline_bpb:.4f}")

    # Sliding window eval
    print(f"\n--- Sliding window eval (stride={stride}) ---")
    sw_loss, sw_bpb = eval_sliding(
        model, val_tokens, seq_len, stride=stride,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        device=device,
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
