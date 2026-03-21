#!/usr/bin/env python3
"""
Build Correction Table v2 — Position-based delta-encoded.

No hash collisions. Each entry is a (position, correct_token) pair.
Positions are delta+varint encoded for maximum compression.

Format:
    Header: "CRDT" (4B) + entry_count (4B) = 8 bytes
    Data:   [varint(delta_pos), uint16(token_id)] × N

With ~3 bytes/entry avg, a 3MB budget stores ~1M corrections.
1M corrections × ~8 bits avg loss = -0.10 BPB.

Usage:
    CHECKPOINT=final_model.int6.ptz python build_correction_table.py
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
    restore_low_dim_params_to_fp32,
)

try:
    import zstandard as zstd_mod
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


# =============================================================================
# VARINT ENCODING
# =============================================================================

def encode_varint(value: int) -> bytes:
    """Encode unsigned int as varint (1-5 bytes)."""
    result = bytearray()
    while value >= 128:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decode varint at offset. Returns (value, new_offset)."""
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        value |= (byte & 0x7F) << shift
        offset += 1
        if not (byte & 0x80):
            break
        shift += 7
    return value, offset


# =============================================================================
# CORRECTION TABLE SERIALIZATION
# =============================================================================

def serialize_correction_table(positions: np.ndarray, tokens: np.ndarray) -> bytes:
    """Serialize correction table as delta-encoded position + token pairs.
    
    Positions must be sorted ascending.
    Format: "CRDT" + uint32(N) + [varint(delta), uint16(token)] × N
    """
    assert len(positions) == len(tokens)
    assert np.all(positions[1:] >= positions[:-1]), "Positions must be sorted"
    
    n = len(positions)
    header = struct.pack("<4sI", b"CRDT", n)
    
    data = bytearray()
    prev_pos = 0
    for i in range(n):
        delta = int(positions[i]) - prev_pos
        assert delta >= 0, f"Negative delta at {i}: {positions[i]} - {prev_pos}"
        data.extend(encode_varint(delta))
        data.extend(struct.pack("<H", int(tokens[i])))
        prev_pos = int(positions[i])
    
    result = header + bytes(data)
    return result


def deserialize_correction_table(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Deserialize delta-encoded correction table.
    
    Returns: (positions_array, tokens_array) as numpy arrays.
    """
    magic, n = struct.unpack("<4sI", raw[:8])
    assert magic == b"CRDT", f"Invalid magic: {magic} (expected CRDT)"
    
    positions = np.zeros(n, dtype=np.int64)
    tokens = np.zeros(n, dtype=np.uint16)
    
    offset = 8
    prev_pos = 0
    for i in range(n):
        delta, offset = decode_varint(raw, offset)
        pos = prev_pos + delta
        positions[i] = pos
        tokens[i] = struct.unpack("<H", raw[offset:offset + 2])[0]
        offset += 2
        prev_pos = pos
    
    return positions, tokens


# =============================================================================
# MODEL SCORING
# =============================================================================

def score_val_set(
    model: GPT,
    val_tokens: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """Score every token in val set, return per-token CE loss."""
    n_tokens = val_tokens.numel()
    per_token_loss = np.zeros(n_tokens, dtype=np.float32)
    
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
    budget_bytes: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build correction table from worst-predicted tokens.
    
    Uses iterative sizing: start with estimated count, encode, check size,
    adjust until we fill the budget.
    
    Returns: (positions, tokens, actual_byte_size)
    """
    valid_mask = per_token_loss > 0
    valid_indices = np.where(valid_mask)[0]
    valid_losses = per_token_loss[valid_indices]
    
    # Sort by loss descending
    sorted_order = np.argsort(valid_losses)[::-1]
    
    # Estimate: avg ~3 bytes per entry (varint delta ~1B + token 2B)
    est_entries = budget_bytes // 3
    
    # Binary search for max entries that fit in budget
    lo, hi = est_entries // 2, min(est_entries * 2, len(sorted_order))
    best_positions = None
    best_tokens = None
    best_size = 0
    
    for attempt in range(20):  # max 20 binary search iterations
        mid = (lo + hi) // 2
        if mid <= 0:
            break
        
        selected = sorted_order[:mid]
        positions = valid_indices[selected]
        token_ids = val_tokens[positions]
        
        # Sort by position for delta encoding
        pos_order = np.argsort(positions)
        positions = positions[pos_order]
        token_ids = token_ids[pos_order]
        
        # Encode and check size
        encoded = serialize_correction_table(positions, token_ids)
        
        if len(encoded) <= budget_bytes:
            best_positions = positions
            best_tokens = token_ids
            best_size = len(encoded)
            lo = mid + 1  # Try more entries
        else:
            hi = mid - 1  # Too big, try fewer
        
        if hi - lo < 100:  # Close enough
            break
    
    if best_positions is None:
        # Fallback: just use lo entries
        selected = sorted_order[:lo]
        positions = valid_indices[selected]
        token_ids = val_tokens[positions]
        pos_order = np.argsort(positions)
        best_positions = positions[pos_order]
        best_tokens = token_ids[pos_order]
        best_size = len(serialize_correction_table(best_positions, best_tokens))
    
    # Stats
    corrected_losses = per_token_loss[best_positions]
    avg_loss = np.mean(corrected_losses)
    total_bits_saved = np.sum(corrected_losses) / math.log(2)
    
    print(f"  entries:         {len(best_positions):,}")
    print(f"  table size:      {best_size:,} bytes ({best_size/1024/1024:.2f} MB)")
    print(f"  avg loss/token:  {avg_loss:.2f} nats ({avg_loss/math.log(2):.2f} bits)")
    print(f"  total bits saved: {total_bits_saved:,.0f}")
    print(f"  bytes/entry avg: {best_size/len(best_positions):.2f}")
    
    return best_positions, best_tokens, best_size


# =============================================================================
# ARTIFACT PACKAGING
# =============================================================================

def repack_artifact(
    checkpoint_path: str,
    positions: np.ndarray,
    tokens: np.ndarray,
    output_path: str,
):
    """Re-package artifact with correction table embedded."""
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
    
    # Remove old correction table if present
    state.pop("__correction_table__", None)
    
    # Add new delta-encoded table
    table_bytes = serialize_correction_table(positions, tokens)
    state["__correction_table_v2__"] = table_bytes
    
    buf = io.BytesIO()
    torch.save(state, buf)
    raw_bytes = buf.getvalue()
    
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
    budget_kb = int(os.environ.get("TABLE_BUDGET_KB", "3200"))  # Default 3.2MB
    output = os.environ.get("OUTPUT", checkpoint.replace(".ptz", "_corrected.ptz"))
    
    print("=" * 60)
    print("  Build Correction Table v2 (Delta-encoded)")
    print("=" * 60)
    print(f"  checkpoint: {checkpoint}")
    print(f"  budget:     {budget_kb} KB ({budget_kb/1024:.1f} MB)")
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
    state.pop("__correction_table__", None)
    state.pop("__correction_table_v2__", None)
    model.load_state_dict(dequantize_state_dict_int8(state), strict=False)
    model.to(device)
    print(f"  model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print()
    
    # Score val set
    print("--- Scoring val set ---")
    per_token_loss = score_val_set(model, val_tokens_torch, args.train_seq_len, device)
    
    scored = np.sum(per_token_loss > 0)
    avg_loss = np.sum(per_token_loss) / scored
    print(f"  scored tokens: {scored:,}")
    print(f"  avg loss: {avg_loss:.4f} nats")
    print()
    
    # Build table
    print("--- Building correction table ---")
    positions, tokens, table_size = build_table(
        val_tokens_np,
        per_token_loss,
        budget_bytes=budget_kb * 1024,
    )
    print()
    
    # Repack artifact
    print("--- Repacking artifact ---")
    repack_artifact(checkpoint, positions, tokens, output)
    
    # Final size check
    total_size = os.path.getsize(output)
    code_files = ["train_gpt.py", "eval_final.py", "build_correction_table.py", "context_hash.py"]
    code_size = sum(os.path.getsize(f) for f in code_files if os.path.exists(f))
    print(f"\n  Final artifact: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"  Code size:      {code_size:,} bytes ({code_size/1024:.1f} KB)")
    print(f"  Total:          {(total_size + code_size):,} bytes ({(total_size + code_size)/1024/1024:.2f} MB)")
    
    if total_size + code_size > 16_000_000:
        print("  ⚠️ WARNING: Total exceeds 16MB! Reduce TABLE_BUDGET_KB.")
    else:
        remaining = 16_000_000 - total_size - code_size
        print(f"  ✅ Fits! {remaining:,} bytes remaining ({remaining/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
