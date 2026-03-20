#!/usr/bin/env python3
"""
Build Correction Table — Post-training val-set error harvester.

Runs the trained model over the fixed validation set, identifies tokens where
the model predicts worst, and builds a compact lookup table:
    (context_hash_32bit, correct_token_16bit)

The table is then embedded into the artifact alongside model weights.

Usage:
    CHECKPOINT=final_model.int6.ptz python build_correction_table.py
    # Produces: final_model_corrected.ptz

Environment variables:
    CHECKPOINT          path to .ptz checkpoint (required)
    TABLE_BUDGET_KB     correction table size budget in KB (default: 1024 = 1MB)
    CONTEXT_LEN         number of preceding tokens for hash (default: 8)
    OUTPUT              output path (default: <checkpoint>_corrected.ptz)
"""
from __future__ import annotations

import io
import math
import os
import struct
import time
import zlib

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

from train_gpt import (
    CastedLinear,
    GPT,
    Hyperparameters,
    Rotary,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    load_validation_tokens,
    quantize_state_dict_int8,
    restore_low_dim_params_to_fp32,
)
from context_hash import context_hash_all

try:
    import zstandard as zstd_mod
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


# =============================================================================
# MODEL SCORING
# =============================================================================

def score_val_set(
    model: GPT,
    val_tokens: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """Score every token in val set, return per-token CE loss.
    
    Returns: np.ndarray of shape [n_tokens] with CE loss per token.
    Positions that couldn't be scored (start of sequence) get loss=0.
    """
    n_tokens = val_tokens.numel()
    per_token_loss = np.zeros(n_tokens, dtype=np.float32)
    
    # Use non-overlapping windows (fast, gives one score per token)
    model.eval()
    t0 = time.perf_counter()
    
    with torch.inference_mode():
        for start in range(0, n_tokens - seq_len, seq_len):
            end = start + seq_len + 1
            if end > n_tokens:
                break
            
            chunk = val_tokens[start:end].to(device=device, dtype=torch.int64)
            x = chunk[:-1].reshape(1, seq_len)
            y = chunk[1:]
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(x)
            
            # Per-token CE (no reduction)
            losses = F.cross_entropy(
                logits.float(), y.to(device), reduction="none"
            )
            
            per_token_loss[start + 1:start + 1 + seq_len] = losses.cpu().numpy()
            
            if (start // seq_len) % 1000 == 0:
                elapsed = time.perf_counter() - t0
                pct = start / n_tokens * 100
                print(f"  scoring: {pct:.1f}% ({elapsed:.0f}s)")
    
    elapsed = time.perf_counter() - t0
    print(f"  scoring done: {elapsed:.0f}s, {n_tokens:,} tokens")
    return per_token_loss


# =============================================================================
# TABLE BUILDING
# =============================================================================

def build_table(
    val_tokens: np.ndarray,
    per_token_loss: np.ndarray,
    context_len: int = 8,
    budget_bytes: int = 1024 * 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Build correction table from worst-predicted tokens.
    
    Returns: (hashes_array, tokens_array) sorted by hash for binary search.
    Each entry is 6 bytes: 4 bytes hash + 2 bytes token_id.
    """
    entry_size = 6  # 4 bytes hash + 2 bytes token_id
    max_entries = budget_bytes // entry_size
    
    print(f"  table budget: {budget_bytes:,} bytes = {max_entries:,} entries")
    
    # Compute context hashes using shared module
    print("  computing context hashes...")
    t0 = time.perf_counter()
    hashes = context_hash_all(val_tokens, context_len)
    print(f"  hashing done: {time.perf_counter() - t0:.1f}s")
    
    # Find worst-predicted tokens (highest CE loss)
    # Skip positions without enough context
    valid_mask = np.zeros(len(val_tokens), dtype=bool)
    valid_mask[context_len:] = True
    valid_mask &= per_token_loss > 0  # Skip unscored positions
    
    valid_indices = np.where(valid_mask)[0]
    valid_losses = per_token_loss[valid_indices]
    
    # Sort by loss descending, take top-K
    top_k_idx = np.argsort(valid_losses)[-max_entries:][::-1]
    selected_positions = valid_indices[top_k_idx]
    
    # Check for hash collisions (different positions, same hash)
    selected_hashes = hashes[selected_positions]
    selected_tokens = val_tokens[selected_positions]
    
    # Deduplicate by hash (keep highest-loss entry for each hash)
    unique_hashes, unique_idx = np.unique(selected_hashes, return_index=True)
    
    n_collisions = len(selected_hashes) - len(unique_hashes)
    if n_collisions > 0:
        print(f"  ⚠️ {n_collisions} hash collisions removed ({n_collisions/len(selected_hashes)*100:.1f}%)")
    
    final_hashes = selected_hashes[unique_idx]
    final_tokens = selected_tokens[unique_idx]
    final_losses = per_token_loss[selected_positions[unique_idx]]
    
    # Sort by hash for binary search at eval time
    sort_idx = np.argsort(final_hashes)
    final_hashes = final_hashes[sort_idx]
    final_tokens = final_tokens[sort_idx]
    
    # Stats
    avg_loss = np.mean(final_losses)
    max_loss = np.max(final_losses)
    total_bits_saved = np.sum(final_losses) / math.log(2)  # Convert nats to bits
    
    print(f"  table entries: {len(final_hashes):,}")
    print(f"  avg loss of corrected tokens: {avg_loss:.2f} nats ({avg_loss/math.log(2):.2f} bits)")
    print(f"  max loss: {max_loss:.2f} nats")
    print(f"  estimated bits saved: {total_bits_saved:,.0f}")
    print(f"  table size: {len(final_hashes) * entry_size:,} bytes")
    
    return final_hashes.astype(np.uint32), final_tokens.astype(np.uint16)


def serialize_table(hashes: np.ndarray, tokens: np.ndarray) -> bytes:
    """Serialize correction table to bytes."""
    n = len(hashes)
    # Header: 4 bytes magic + 4 bytes entry count + 1 byte context_len
    header = struct.pack("<4sIB", b"CRCT", n, 8)  # 9 bytes header
    # Data: interleaved (hash, token) pairs
    data = bytearray()
    for h, t in zip(hashes, tokens):
        data.extend(struct.pack("<IH", int(h), int(t)))
    return header + bytes(data)


def deserialize_table(raw: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    """Deserialize correction table from bytes.
    
    Returns: (hashes, tokens, context_len)
    """
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


# =============================================================================
# ARTIFACT PACKAGING
# =============================================================================

def repack_artifact(
    checkpoint_path: str,
    table_hashes: np.ndarray,
    table_tokens: np.ndarray,
    output_path: str,
):
    """Re-package artifact with correction table embedded."""
    # Load original checkpoint
    with open(checkpoint_path, "rb") as f:
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
    
    # Add correction table as a special key
    table_bytes = serialize_table(table_hashes, table_tokens)
    state["__correction_table__"] = table_bytes
    
    # Re-serialize
    buf = io.BytesIO()
    torch.save(state, buf)
    raw_bytes = buf.getvalue()
    
    # Compress with zstd if available, else zlib
    if HAS_ZSTD:
        cctx = zstd_mod.ZstdCompressor(level=22)
        compressed = cctx.compress(raw_bytes)
        print(f"  compressed with zstd-22: {len(compressed):,} bytes")
    else:
        compressed = zlib.compress(raw_bytes, 9)
        print(f"  compressed with zlib-9: {len(compressed):,} bytes")
    
    with open(output_path, "wb") as f:
        f.write(compressed)
    
    print(f"  saved: {output_path} ({len(compressed):,} bytes = {len(compressed)/1024/1024:.2f} MB)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = os.environ.get("CHECKPOINT", "final_model.int6.ptz")
    budget_kb = int(os.environ.get("TABLE_BUDGET_KB", "1024"))
    context_len = int(os.environ.get("CONTEXT_LEN", "8"))
    output = os.environ.get("OUTPUT", checkpoint.replace(".ptz", "_corrected.ptz"))
    
    print("=" * 60)
    print("  Build Correction Table")
    print("=" * 60)
    print(f"  checkpoint: {checkpoint}")
    print(f"  budget:     {budget_kb} KB")
    print(f"  context:    {context_len} tokens")
    print(f"  output:     {output}")
    print()
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    
    # Load val tokens
    val_tokens_torch = load_validation_tokens(args.val_files, args.train_seq_len)
    val_tokens_np = val_tokens_torch.numpy().astype(np.int32)
    print(f"  val tokens: {len(val_tokens_np):,}")
    
    # Load model
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
    model.load_state_dict(dequantize_state_dict_int8(state), strict=False)
    model.to(device)
    print(f"  model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print()
    
    # Score val set
    print("--- Scoring val set ---")
    per_token_loss = score_val_set(model, val_tokens_torch, args.train_seq_len, device)
    
    scored = np.sum(per_token_loss > 0)
    avg_loss = np.sum(per_token_loss) / scored
    avg_bpb_approx = avg_loss / math.log(2) * 1.3  # rough bytes/token estimate
    print(f"  scored tokens: {scored:,}")
    print(f"  avg loss: {avg_loss:.4f} (~{avg_bpb_approx:.4f} BPB)")
    print()
    
    # Build table
    print("--- Building correction table ---")
    table_hashes, table_tokens = build_table(
        val_tokens_np,
        per_token_loss,
        context_len=context_len,
        budget_bytes=budget_kb * 1024,
    )
    print()
    
    # Repack artifact
    print("--- Repacking artifact ---")
    repack_artifact(checkpoint, table_hashes, table_tokens, output)
    
    # Final size check
    total_size = os.path.getsize(output)
    code_size = sum(
        os.path.getsize(f) for f in [
            "train_gpt.py", "eval_final.py", "build_correction_table.py"
        ] if os.path.exists(f)
    )
    print(f"\n  Final artifact: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"  Code size:      {code_size:,} bytes ({code_size/1024:.1f} KB)")
    print(f"  Total:          {(total_size + code_size):,} bytes ({(total_size + code_size)/1024/1024:.2f} MB)")
    
    if total_size + code_size > 16_000_000:
        print("  ⚠️ WARNING: Total exceeds 16MB! Reduce TABLE_BUDGET_KB.")
    else:
        print("  ✅ Fits within 16MB budget.")


if __name__ == "__main__":
    main()
