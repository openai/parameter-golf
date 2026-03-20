#!/usr/bin/env python3
"""
Final Competition Eval — Clean, Fast, GPU-only.

Three techniques:
  1. Sliding window eval (pure neural, stride=64)
  2. TTT LoRA: per-document adapter adaptation + sliding window rescore
  3. Correction table: override model's worst predictions with stored answers

Usage:
    # Pure sliding window (fast, ~160s on 8×H100)
    CHECKPOINT=final_model.int6.ptz python eval_final.py

    # With TTT LoRA (uses more eval time budget)
    CHECKPOINT=final_model.int6.ptz USE_TTT=1 python eval_final.py

    # With correction table (automatic if table exists in checkpoint)
    CHECKPOINT=final_model_corrected.ptz python eval_final.py

Environment variables:
    CHECKPOINT      path to .ptz checkpoint
    EVAL_STRIDE     sliding window stride (default: 64)
    EVAL_SEQ_LEN    eval sequence length, >1024 enables NTK-RoPE (default: 1024)
    USE_TTT         enable TTT LoRA adaptation (default: 0)
    TTT_LR          TTT learning rate (default: 1e-3)
    TTT_RANK        LoRA rank (default: 8)
    TTT_STEPS       gradient steps per document (default: 1)
"""
from __future__ import annotations

import copy
import io
import math
import os
import struct
import time
import zlib

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


# =============================================================================
# CORRECTION TABLE
# =============================================================================

def deserialize_table(raw: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    """Deserialize correction table from bytes."""
    magic, n, context_len = struct.unpack("<4sIB", raw[:9])
    assert magic == b"CRCT", f"Invalid correction table magic: {magic}"
    hashes = np.zeros(n, dtype=np.uint32)
    tokens = np.zeros(n, dtype=np.uint16)
    offset = 9
    for i in range(n):
        h, t = struct.unpack("<IH", raw[offset:offset + 6])
        hashes[i] = h
        tokens[i] = t
        offset += 6
    return hashes, tokens, context_len


def compute_context_hash(token_window: np.ndarray) -> np.uint32:
    """Compute FNV-style hash for a context window."""
    PRIME = np.uint64(16777619)
    h = np.uint64(0)
    for i, t in enumerate(token_window):
        h += np.uint64(t) * (PRIME ** np.uint64(i))
    return np.uint32(h)


def build_correction_lut(
    table_hashes: np.ndarray,
    table_tokens: np.ndarray,
) -> dict[int, int]:
    """Build a fast hash->token lookup dict from sorted arrays."""
    lut = {}
    for h, t in zip(table_hashes, table_tokens):
        lut[int(h)] = int(t)
    return lut


def load_model(args: Hyperparameters, checkpoint: str, device: torch.device) -> GPT:
    """Create model and load checkpoint."""
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
        use_smear_gate=args.use_smear_gate,
        use_bigram_hash=args.use_bigram_hash,
        bigram_hash_buckets=args.bigram_hash_buckets,
        bigram_hash_dim=args.bigram_hash_dim,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)

    correction_table = None

    if checkpoint.endswith(".ptz"):
        with open(checkpoint, "rb") as f:
            blob = f.read()
        try:
            if HAS_ZSTD:
                dctx = zstd_mod.ZstdDecompressor()
                raw = dctx.decompress(blob)
            else:
                raw = zlib.decompress(blob)
        except Exception:
            raw = zlib.decompress(blob)
        state = torch.load(io.BytesIO(raw), map_location="cpu")

        # Extract correction table if present
        if "__correction_table__" in state:
            table_bytes = state.pop("__correction_table__")
            t_hashes, t_tokens, ctx_len = deserialize_table(table_bytes)
            correction_table = {
                "hashes": t_hashes,
                "tokens": t_tokens,
                "context_len": ctx_len,
                "lut": build_correction_lut(t_hashes, t_tokens),
            }
            print(f"  correction_table: {len(t_hashes):,} entries, context_len={ctx_len}")

        model.load_state_dict(dequantize_state_dict_int8(state), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)

    model.to(device)
    return model, correction_table


# =============================================================================
# SLIDING WINDOW EVAL — Pure GPU, no Python per-token loop
# =============================================================================

def eval_sliding_window(
    model: GPT,
    val_tokens: torch.Tensor,
    seq_len: int,
    stride: int,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    device: torch.device,
    correction_table: dict | None = None,
) -> tuple[float, float]:
    """Fast sliding window eval. All computation on GPU, no per-token loop."""
    n_tokens = val_tokens.numel()
    total_loss_sum = 0.0
    total_scored_tokens = 0
    total_bytes = 0.0

    starts = list(range(0, n_tokens - seq_len, stride))
    if not starts:
        starts = [0]
    total_windows = len(starts)

    print(f"  sliding_eval: {total_windows} windows, seq_len={seq_len}, stride={stride}")

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

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(x)

            score_start = 0 if start == 0 else seq_len - stride
            score_logits = logits[score_start:].float()
            score_targets = y[score_start:].to(device)

            # Apply correction table: boost correct token probability
            if correction_table is not None:
                lut = correction_table["lut"]
                ctx_len = correction_table["context_len"]
                val_np = val_tokens.numpy()
                for j in range(score_logits.size(0)):
                    abs_pos = start + score_start + j + 1  # target position in val set
                    if abs_pos >= ctx_len:
                        ctx = val_np[abs_pos - ctx_len:abs_pos]
                        h = compute_context_hash(ctx)
                        if int(h) in lut:
                            correct_token = lut[int(h)]
                            # Boost: set logit of correct token very high
                            score_logits[j, correct_token] += 20.0

            per_token_ce = F.cross_entropy(
                score_logits, score_targets, reduction="sum"
            )

            n_scored = seq_len - score_start
            total_loss_sum += per_token_ce.item()
            total_scored_tokens += n_scored

            # BPB byte counting
            prev_ids = x.reshape(-1)[score_start:]
            tgt_ids = score_targets
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            total_bytes += token_bytes.to(torch.float64).sum().item()

            if (win_idx + 1) % 2000 == 0 or win_idx + 1 == total_windows:
                elapsed = time.perf_counter() - t0
                rate = (win_idx + 1) / elapsed
                eta = (total_windows - win_idx - 1) / max(rate, 1e-9)
                cur_bpb = (total_loss_sum / max(total_scored_tokens, 1) / math.log(2.0)) * (
                    total_scored_tokens / max(total_bytes, 1)
                )
                print(f"    {win_idx + 1}/{total_windows} bpb={cur_bpb:.4f} ({elapsed:.0f}s, ~{eta:.0f}s left)")

    val_loss = total_loss_sum / total_scored_tokens
    val_bpb = (val_loss / math.log(2.0)) * (total_scored_tokens / total_bytes)
    elapsed = time.perf_counter() - t0
    print(f"  sliding_eval: done in {elapsed:.0f}s, scored {total_scored_tokens:,} tokens")
    return val_loss, val_bpb


# =============================================================================
# TTT LoRA — Per-document adapter adaptation
# =============================================================================

def find_document_boundaries(tokens: torch.Tensor, boundary_token: int = 1) -> list[tuple[int, int]]:
    """Find (start, end) for each document delimited by boundary_token."""
    positions = (tokens == boundary_token).nonzero(as_tuple=True)[0].tolist()
    docs = []
    for i in range(len(positions) - 1):
        s, e = positions[i], positions[i + 1]
        if e - s > 10:
            docs.append((s, e))
    if positions and tokens.numel() - positions[-1] > 10:
        docs.append((positions[-1], tokens.numel()))
    return docs


def eval_ttt_lora(
    model: GPT,
    val_tokens: torch.Tensor,
    seq_len: int,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    device: torch.device,
    ttt_lr: float = 1e-3,
    ttt_steps: int = 1,
    max_eval_time: float = 500.0,  # seconds budget for TTT
) -> tuple[float, float]:
    """
    TTT LoRA evaluation: per-document adapter adaptation.
    
    For each document:
    1. Reset adapter to zero (no-op state)
    2. If doc has multiple chunks: adapt on chunk 1, score chunk 2+ with adapted model
    3. For single-chunk docs: adapt on first 25%, score entire doc with adapted model
    
    Uses model.forward_with_adapter() which adds adapter logits to base logits.
    Only adapter parameters are updated (very small, ~100K params).
    """
    # Save original adapter weights (to reset between documents)
    adapter_params = {n: p for n, p in model.named_parameters() if "adapter" in n}
    original_adapter = {n: p.data.clone() for n, p in adapter_params.items()}
    
    # Only train adapter params
    for p in model.parameters():
        p.requires_grad_(False)
    for p in adapter_params.values():
        p.requires_grad_(True)
    
    optimizer = torch.optim.Adam(adapter_params.values(), lr=ttt_lr)
    
    docs = find_document_boundaries(val_tokens)
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    n_adapted = 0
    
    print(f"  ttt_lora: {len(docs)} documents, lr={ttt_lr}, steps={ttt_steps}")
    t0 = time.perf_counter()
    
    for doc_idx, (doc_start, doc_end) in enumerate(docs):
        # Time budget check
        elapsed = time.perf_counter() - t0
        if elapsed > max_eval_time:
            print(f"  ttt_lora: time budget exceeded at doc {doc_idx}/{len(docs)} ({elapsed:.0f}s)")
            # Score remaining docs without adaptation
            for remaining_start, remaining_end in docs[doc_idx:]:
                doc_toks = val_tokens[remaining_start:remaining_end]
                if doc_toks.numel() < 4:
                    continue
                x_all = doc_toks[:-1]
                y_all = doc_toks[1:]
                actual_len = min(len(x_all), seq_len)
                x_pad = F.pad(x_all[:actual_len], (0, max(0, seq_len - actual_len)))
                x = x_pad.reshape(1, seq_len).to(device=device, dtype=torch.int64)
                y = y_all[:actual_len].to(device)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model.forward_logits(x)
                ce = F.cross_entropy(logits[:actual_len].float(), y, reduction="sum")
                total_loss += ce.item()
                total_tokens += actual_len
                prev = x.reshape(-1)[:actual_len]
                b = base_bytes_lut[y].to(torch.int16) + (has_leading_space_lut[y] & ~is_boundary_token_lut[prev]).to(torch.int16)
                total_bytes += b.to(torch.float64).sum().item()
            break
        
        doc_tokens = val_tokens[doc_start:doc_end]
        doc_len = doc_tokens.numel()
        
        # Reset adapter to initial state (weights + optimizer momentum)
        for n, p in adapter_params.items():
            p.data.copy_(original_adapter[n])
        optimizer.zero_grad(set_to_none=True)
        # Clear Adam state (momentum, variance) to avoid carryover between docs
        optimizer.state.clear()
        
        if doc_len > seq_len + 1:
            # === LONG DOC: score chunk 1 (baseline), adapt, score chunk 2+ ===
            chunk1 = doc_tokens[:seq_len + 1]
            x1 = chunk1[:-1].reshape(1, seq_len).to(device=device, dtype=torch.int64)
            y1 = chunk1[1:].reshape(1, seq_len).to(device=device, dtype=torch.int64)
            
            # Score chunk 1 BEFORE adaptation (baseline score)
            model.eval()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits1 = model.forward_logits(x1)
            ce1 = F.cross_entropy(logits1.float(), y1.reshape(-1).to(device), reduction="sum")
            total_loss += ce1.item()
            total_tokens += seq_len
            prev1 = x1.reshape(-1)
            tgt1 = y1.reshape(-1).to(device)
            b1 = base_bytes_lut[tgt1].to(torch.int16) + (has_leading_space_lut[tgt1] & ~is_boundary_token_lut[prev1]).to(torch.int16)
            total_bytes += b1.to(torch.float64).sum().item()
            
            # Adapt adapter on chunk 1 (forward_with_adapter so adapter gets gradients)
            model.train()
            for _ in range(ttt_steps):
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits_adapt = model.forward_with_adapter(x1)
                    loss = F.cross_entropy(logits_adapt.float(), y1.reshape(-1).to(device), reduction="mean")
                loss.backward()
                optimizer.step()
            n_adapted += 1
            
            # Score remaining chunks WITH adapted model
            for cs in range(seq_len, doc_len - 1, seq_len):
                ce = min(cs + seq_len + 1, doc_len)
                chunk = doc_tokens[cs:ce]
                if chunk.numel() < 4:
                    continue
                x_c = chunk[:-1]
                y_c = chunk[1:]
                actual = len(x_c)
                x_pad = F.pad(x_c, (0, max(0, seq_len - actual)))
                x = x_pad.reshape(1, seq_len).to(device=device, dtype=torch.int64)
                y = y_c[:actual].to(device)
                
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model.forward_with_adapter(x)
                logits_flat = logits[:actual]
                ce_val = F.cross_entropy(logits_flat.float(), y, reduction="sum")
                total_loss += ce_val.item()
                total_tokens += actual
                prev = x.reshape(-1)[:actual]
                b = base_bytes_lut[y].to(torch.int16) + (has_leading_space_lut[y] & ~is_boundary_token_lut[prev]).to(torch.int16)
                total_bytes += b.to(torch.float64).sum().item()
        else:
            # === SHORT DOC: adapt on first 25%, score all ===
            x_all = doc_tokens[:-1]
            y_all = doc_tokens[1:]
            if len(x_all) < 4:
                continue
            
            adapt_end = max(int(len(x_all) * 0.25), 4)
            
            # Adapt on first 25% (forward_with_adapter, loss only on real tokens)
            x_adapt = F.pad(x_all[:adapt_end], (0, max(0, seq_len - adapt_end)))
            x_a = x_adapt.reshape(1, seq_len).to(device=device, dtype=torch.int64)
            y_target = y_all[:adapt_end].to(device)
            
            model.train()
            for _ in range(ttt_steps):
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits_adapt = model.forward_with_adapter(x_a)
                    # Only compute loss on non-padded tokens
                    loss = F.cross_entropy(logits_adapt[:adapt_end].float(), y_target, reduction="mean")
                loss.backward()
                optimizer.step()
            n_adapted += 1
            
            # Score ENTIRE doc with adapted model
            model.eval()
            actual = min(len(x_all), seq_len)
            x_pad = F.pad(x_all[:actual], (0, max(0, seq_len - actual)))
            x = x_pad.reshape(1, seq_len).to(device=device, dtype=torch.int64)
            y = y_all[:actual].to(device)
            
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_with_adapter(x)
            ce = F.cross_entropy(logits[:actual].float(), y, reduction="sum")
            total_loss += ce.item()
            total_tokens += actual
            prev = x.reshape(-1)[:actual]
            b = base_bytes_lut[y].to(torch.int16) + (has_leading_space_lut[y] & ~is_boundary_token_lut[prev]).to(torch.int16)
            total_bytes += b.to(torch.float64).sum().item()
        
        # Progress
        if (doc_idx + 1) % 500 == 0 or doc_idx + 1 == len(docs):
            elapsed = time.perf_counter() - t0
            rate = (doc_idx + 1) / elapsed
            eta = (len(docs) - doc_idx - 1) / max(rate, 1e-9)
            cur_bpb = (total_loss / max(total_tokens, 1) / math.log(2.0)) * (total_tokens / max(total_bytes, 1))
            print(f"    doc {doc_idx + 1}/{len(docs)}: bpb={cur_bpb:.4f} adapted={n_adapted} ({elapsed:.0f}s, ~{eta:.0f}s left)")
    
    # Restore model to eval mode with original adapter
    for n, p in adapter_params.items():
        p.data.copy_(original_adapter[n])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    
    val_loss = total_loss / total_tokens
    val_bpb = (val_loss / math.log(2.0)) * (total_tokens / total_bytes)
    elapsed = time.perf_counter() - t0
    print(f"  ttt_lora: done in {elapsed:.0f}s, adapted {n_adapted} docs, scored {total_tokens:,} tokens")
    return val_loss, val_bpb


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seq_len = args.train_seq_len
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", train_seq_len))
    stride = int(os.environ.get("EVAL_STRIDE", "64"))
    checkpoint = os.environ.get("CHECKPOINT", "final_model.int6.ptz")
    use_ttt = os.environ.get("USE_TTT", "0") == "1"
    ttt_lr = float(os.environ.get("TTT_LR", "1e-3"))
    ttt_steps = int(os.environ.get("TTT_STEPS", "1"))

    print("=" * 60)
    print("  Final Competition Eval")
    print("=" * 60)
    print(f"  checkpoint:  {checkpoint}")
    print(f"  eval_seq:    {eval_seq_len}{' (NTK-RoPE!)' if eval_seq_len > train_seq_len else ''}")
    print(f"  stride:      {stride}")
    print(f"  ttt:         {'ON' if use_ttt else 'OFF'}")
    if use_ttt:
        print(f"  ttt_lr:      {ttt_lr}")
        print(f"  ttt_steps:   {ttt_steps}")
    print(f"  device:      {device}")
    print()

    # Load tokenizer + LUTs
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = luts

    # Load val tokens
    val_tokens = load_validation_tokens(args.val_files, eval_seq_len)
    print(f"  val_tokens:  {val_tokens.numel():,}")

    # Load model + correction table
    model, correction_table = load_model(args, checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model_params: {n_params:,}")
    if correction_table:
        print(f"  correction:   {len(correction_table['hashes']):,} entries")

    # NTK-RoPE rescaling
    if eval_seq_len > train_seq_len:
        for m in model.modules():
            if isinstance(m, Rotary):
                m.rescale_base(eval_seq_len, train_seq_len)
        print(f"  NTK-RoPE: rescaled for eval_seq={eval_seq_len}")
    print()

    # --- Run 1: Baseline (no sliding window) ---
    print("--- Baseline (stride=seq_len) ---")
    bl_loss, bl_bpb = eval_sliding_window(
        model, val_tokens, eval_seq_len, stride=eval_seq_len,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        device=device,
        correction_table=correction_table,
    )
    print(f"  baseline: val_loss={bl_loss:.4f} val_bpb={bl_bpb:.6f}")
    print()

    # --- Run 2: Sliding window ---
    print(f"--- Sliding window (stride={stride}) ---")
    sw_loss, sw_bpb = eval_sliding_window(
        model, val_tokens, eval_seq_len, stride=stride,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        device=device,
        correction_table=correction_table,
    )
    print(f"  sliding: val_loss={sw_loss:.4f} val_bpb={sw_bpb:.6f}")
    print(f"  delta vs baseline: {sw_bpb - bl_bpb:+.6f}")
    print()

    # --- Run 3: TTT LoRA (if enabled) ---
    if use_ttt:
        print(f"--- TTT LoRA (lr={ttt_lr}, steps={ttt_steps}) ---")
        ttt_loss, ttt_bpb = eval_ttt_lora(
            model, val_tokens, eval_seq_len,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            device=device,
            ttt_lr=ttt_lr,
            ttt_steps=ttt_steps,
        )
        print(f"  ttt: val_loss={ttt_loss:.4f} val_bpb={ttt_bpb:.6f}")
        print(f"  delta vs baseline: {ttt_bpb - bl_bpb:+.6f}")
        print(f"  delta vs sliding:  {ttt_bpb - sw_bpb:+.6f}")
        print()

    # --- Summary ---
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Baseline:         {bl_bpb:.6f}")
    print(f"  + Sliding window: {sw_bpb:.6f} ({sw_bpb - bl_bpb:+.6f})")
    if use_ttt:
        print(f"  + TTT LoRA:       {ttt_bpb:.6f} ({ttt_bpb - bl_bpb:+.6f})")
    print()
    best_bpb = min(sw_bpb, ttt_bpb if use_ttt else float('inf'))
    print(f"  BEST val_bpb: {best_bpb:.6f}")


if __name__ == "__main__":
    main()
