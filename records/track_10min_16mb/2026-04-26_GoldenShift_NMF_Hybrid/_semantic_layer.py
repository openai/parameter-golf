"""Semantic Layer — GoldenShift_NMF_Hybrid.

Two-tier eval architecture:

  Tier 1 (NMF softmax, primary — matched buckets ~70-80%):
    bucket ← G[p] rolling hash
    if fingerprint_match AND embed_filled:
      logits = embed[bucket] @ W_out   → softmax → p_correct

  Tier 2 (GoldenAxisShift DSV fallback — collision + miss ~20-30%):
    Uses precomputed (V, V) raw_scores_table for fast O(B) scalar lookups:
      p_correct = normalised_score_table[prev_tok, tgt_tok]

Artifact format (HGZ4):
  Magic(4B "HGZ4") + nmf_seed(8B) + table_bits(4B) + embed_dim(4B)
      + vocab_size(4B) + n_words(4B) + dsv_seed(4B) + flags(4B)
  Body (LZMA9):
    embed        (TABLE_SIZE × embed_dim × 2)   float16   NMF Tier 1
    W_out        (embed_dim × vocab_size × 2)   float16   NMF Tier 1
    fingerprint  (TABLE_SIZE × 1)              uint8     collision detect
    sem_fwd      (vocab_size × n_words × 8)    uint64    DSV Tier 2

Public API:
  build_spiral_dsv()         — GoldenAxisShift DSV build (Phase 6), unchanged
  save_hybrid_artifact()     — save HGZ4 (NMF + DSV combined)
  load_hybrid_artifact()     — load HGZ4
  eval_hybrid_bpb()          — 2-tier BPB evaluation
  build_token_byte_arrays()  — per-token UTF-8 byte/space/boundary arrays
  check_artifact_size()      — verify ≤ 16 MB
"""

from __future__ import annotations

import lzma
import struct
import time
from typing import Optional, Tuple

import numpy as np

# -- Local imports ------------------------------------------------------------
from _spiral_dsv_lm import SpiralDSVLanguageModel, GOLDEN_AXES
from _eigen_convergence import EigenTrainer

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

VOCAB_SIZE: int        = 1024
N_WORDS_HYB: int       = 1024   # DSV Tier 2 hypervector width (8 MB sem_fwd)
N_WORDS_4090: int      = 128    # Smoke-test / RTX 4090
N_WORDS_TEST: int      = 16     # CPU smoke test
CTX_LEN: int           = 4      # lags 1..4 for GoldenAxisShift DSV
ARTIFACT_LIMIT: int    = 16_000_000  # 16 MB hard limit

# HGZ4 format
_HGZ4_MAGIC       = b'HGZ4'
_HGZ4_HEADER_FMT  = '<4sQIIIIII'   # magic nmf_seed table_bits embed_dim vocab_size n_words dsv_seed flags
_HGZ4_HEADER_SIZE = struct.calcsize(_HGZ4_HEADER_FMT)

_FMIX64 = np.uint64(0x9E3779B97F4A7C15)


# =============================================================================
# GoldenAxisShift DSV build (Phase 6) — unchanged from GoldenShift_DSV_Pure
# =============================================================================

def build_spiral_dsv(
    tokens: np.ndarray,
    vocab_size: int = VOCAB_SIZE,
    n_words: int = N_WORDS_HYB,
    ctx_len: int = CTX_LEN,
    seed: int = 42,
    time_budget_s: float = 300.0,
    dist_rank: int = 0,
    dist_world_size: int = 1,
    use_freq_weights: bool = True,
    verbose: bool = True,
) -> SpiralDSVLanguageModel:
    """Build sem_fwd GoldenAxisShift DSV table from training tokens.

    Used as Tier 2 fallback in eval_hybrid_bpb().  This function is
    identical to the one in GoldenShift_DSV_Pure (_semantic_layer.py).
    """
    t0 = time.time()

    model = SpiralDSVLanguageModel(
        vocab_size=vocab_size,
        n_words=n_words,
        seed=seed,
    )

    if verbose and dist_rank == 0:
        n_bits = n_words * 64
        sem_mb = vocab_size * n_words * 8 / 1_000_000
        print(f"\n[Hybrid-DSV] Building GoldenAxisShift sem_fwd (Tier 2 fallback)")
        print(f"[Hybrid-DSV] vocab={vocab_size}, n_words={n_words}, "
              f"n_bits={n_bits:,}, ctx_len={ctx_len}")
        print(f"[Hybrid-DSV] sem_fwd budget: {sem_mb:.1f} MB")
        print(f"[Hybrid-DSV] 1/freq weighting: {'ENABLED' if use_freq_weights else 'DISABLED'}")
        print(f"[Hybrid-DSV] dist: rank={dist_rank}/{dist_world_size}, budget={time_budget_s:.0f}s")

    # Ensure golden axes computed for all lags
    for c in range(1, ctx_len + 1):
        GOLDEN_AXES.offset(c)

    axis_word_shifts = [
        (GOLDEN_AXES._word_shifts[c], GOLDEN_AXES._bit_shifts[c])
        for c in range(1, ctx_len + 1)
    ]

    # Compute frequency table for 1/freq weighting
    if use_freq_weights:
        tokens_i32 = np.clip(tokens.astype(np.int32), 0, vocab_size - 1)
        freq_table = np.bincount(tokens_i32, minlength=vocab_size).astype(np.float32)
        freq_table = np.maximum(freq_table, 1.0)
        if verbose and dist_rank == 0:
            print(f"[Hybrid-DSV] freq_table: total={float(freq_table.sum()):.0f}, "
                  f"min={float(freq_table.min()):.0f}, max={float(freq_table.max()):.0f}")
    else:
        freq_table = None

    trainer = EigenTrainer.from_codebook_uint64(model.codebook)

    result = trainer.build_bilateral_from_tokens(
        tokens=tokens,
        ctx_len=ctx_len,
        axis_word_shifts=axis_word_shifts,
        chunk_size=2_000_000,
        verbose=verbose,
        time_budget_s=time_budget_s,
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
        freq_weights=freq_table,
    )

    if result.get('sem_fwd_u64') is None:
        return model  # non-main ranks — _built=False

    model.sem_fwd = result['sem_fwd_u64']
    model.sem_bwd = result['sem_bwd_u64']
    model._built  = True
    model._invalidate_pm1_cache()

    if verbose:
        elapsed      = time.time() - t0
        total_pairs  = result.get('total_pairs', 0)
        print(f"[Hybrid-DSV] Build complete: {total_pairs:,} pairs in {elapsed:.2f}s")
        print(f"[Hybrid-DSV] sem_fwd: {model.sem_fwd.nbytes:,} bytes")

    return model


# =============================================================================
# HGZ4 Artifact save / load
# =============================================================================

def save_hybrid_artifact(
    model: SpiralDSVLanguageModel,
    embed: np.ndarray,
    W_out: np.ndarray,
    fingerprint: np.ndarray,
    nmf_seed: int,
    table_bits: int,
    path: str,
    verbose: bool = True,
) -> int:
    """Save NMF (Tier 1) + DSV sem_fwd (Tier 2) to a single LZMA9 HGZ4 artifact.

    HGZ4 format:
        Header (32 bytes):
            4B  magic "HGZ4"
            8B  nmf_seed (uint64)
            4B  table_bits (uint32)
            4B  embed_dim (uint32)
            4B  vocab_size (uint32)
            4B  n_words (uint32)
            4B  dsv_seed (uint32)
            4B  flags (uint32)  -- bit0: has_fingerprint, bit1: has_sem_fwd
        Body (LZMA9 compressed):
            embed        (TABLE_SIZE × embed_dim × 2)  float16
            W_out        (embed_dim × vocab_size × 2)  float16
            fingerprint  (TABLE_SIZE × 1)              uint8   [if flags&1]
            sem_fwd      (vocab_size × n_words × 8)    uint64  [if flags&2]

    Args:
        model       : SpiralDSVLanguageModel with sem_fwd built
        embed       : (TABLE_SIZE, embed_dim) float16 NMF embedding
        W_out       : (embed_dim, vocab_size) float16 NMF output projection
        fingerprint : (TABLE_SIZE,) uint8 per-bucket fingerprint
        nmf_seed    : NMF hash seed
        table_bits  : log₂ TABLE_SIZE
        path        : Output file path
        verbose     : Print size info

    Returns:
        Compressed artifact size in bytes.
    """
    if not getattr(model, '_built', False):
        raise ValueError("DSV model not built — call build_spiral_dsv() first")

    embed_f16 = embed.astype(np.float16)
    W_out_f16 = W_out.astype(np.float16)
    fp_u8     = fingerprint.astype(np.uint8)

    flags = np.uint32(1 | 2)   # has_fingerprint=1, has_sem_fwd=2

    header = struct.pack(
        _HGZ4_HEADER_FMT,
        _HGZ4_MAGIC,
        np.uint64(nmf_seed),      # nmf_seed as uint64
        np.uint32(table_bits),
        np.uint32(embed_f16.shape[1]),
        np.uint32(model.vocab_size),
        np.uint32(model.n_words),
        np.uint32(model.seed),    # DSV codebook seed
        flags,
    )

    payload = (
        header
        + embed_f16.tobytes()
        + W_out_f16.tobytes()
        + fp_u8.tobytes()
        + model.sem_fwd.tobytes()
    )

    uncompressed_mb = len(payload) / 1_000_000
    compressed      = lzma.compress(payload, preset=9)
    compressed_mb   = len(compressed) / 1_000_000

    with open(path, 'wb') as f:
        f.write(compressed)

    if verbose:
        print(f"[Hybrid] Artifact saved: {path}")
        print(f"[Hybrid] Uncompressed: {uncompressed_mb:.2f} MB  "
              f"Compressed (LZMA9): {compressed_mb:.2f} MB  "
              f"({len(compressed):,} bytes)")
        print(f"[Hybrid]   embed: {embed_f16.nbytes/1e6:.2f} MB  "
              f"W_out: {W_out_f16.nbytes/1e3:.1f} KB  "
              f"fp: {fp_u8.nbytes/1e3:.1f} KB  "
              f"sem_fwd: {model.sem_fwd.nbytes/1e6:.2f} MB")

    return len(compressed)


def load_hybrid_artifact(
    path: str,
    verbose: bool = True,
) -> Tuple[SpiralDSVLanguageModel, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load HGZ4 artifact and reconstruct all tables.

    Returns:
        model       : SpiralDSVLanguageModel with sem_fwd populated
        embed       : (TABLE_SIZE, embed_dim) float16
        W_out       : (embed_dim, vocab_size) float16
        fingerprint : (TABLE_SIZE,) uint8
        nmf_seed    : int  (NMF hash seed)
        table_bits  : int  (log₂ TABLE_SIZE)
    """
    with open(path, 'rb') as f:
        data = lzma.decompress(f.read())

    (magic, nmf_seed, table_bits, embed_dim,
     vocab_size, n_words, dsv_seed, flags) = struct.unpack_from(
        _HGZ4_HEADER_FMT, data, 0
    )
    if magic != _HGZ4_MAGIC:
        raise ValueError(f"Bad HGZ4 magic: {magic!r}")

    TABLE_SIZE = 1 << table_bits
    off = _HGZ4_HEADER_SIZE

    embed_bytes = TABLE_SIZE * embed_dim * 2
    embed = np.frombuffer(data, dtype=np.float16, count=TABLE_SIZE * embed_dim,
                          offset=off).reshape(TABLE_SIZE, embed_dim).copy()
    off += embed_bytes

    wout_bytes = embed_dim * vocab_size * 2
    W_out = np.frombuffer(data, dtype=np.float16, count=embed_dim * vocab_size,
                          offset=off).reshape(embed_dim, vocab_size).copy()
    off += wout_bytes

    fingerprint = np.zeros(TABLE_SIZE, dtype=np.uint8)
    if flags & 1:
        fingerprint = np.frombuffer(data, dtype=np.uint8, count=TABLE_SIZE,
                                    offset=off).copy()
        off += TABLE_SIZE

    model = SpiralDSVLanguageModel(vocab_size=vocab_size, n_words=n_words, seed=dsv_seed)
    if flags & 2:
        sem_fwd = np.frombuffer(data, dtype=np.uint64, count=vocab_size * n_words,
                                offset=off).reshape(vocab_size, n_words).copy()
        model.sem_fwd = sem_fwd
        model.sem_bwd = sem_fwd   # alias
        model._built  = True
        model._invalidate_pm1_cache()

    if verbose:
        print(f"[Hybrid] Loaded: {path}")
        print(f"[Hybrid]  nmf_seed={nmf_seed}  table_bits={table_bits} "
              f"({TABLE_SIZE:,} buckets)  embed_dim={embed_dim}")
        print(f"[Hybrid]  dsv: vocab={vocab_size}  n_words={n_words}  seed={dsv_seed}")

    return model, embed, W_out, fingerprint, int(nmf_seed), int(table_bits)


# =============================================================================
# 2-Tier BPB Evaluation
# =============================================================================

def eval_hybrid_bpb(
    val_tokens: np.ndarray,
    g_states_val: np.ndarray,
    model: SpiralDSVLanguageModel,
    embed: np.ndarray,
    W_out: np.ndarray,
    fingerprint: np.ndarray,
    nmf_seed: int,
    table_bits: int,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary_token: Optional[np.ndarray] = None,
    batch_size: int = 500_000,
    use_fingerprint: bool = False,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Compute BPB using the 2-tier NMF+DSV hybrid.

    Routing (use_fingerprint=False, default — RECOMMENDED):
      Tier 1 : embed[bucket] filled → softmax(embed[bucket] @ W_out)
                Achieves ~100% NMF coverage when table is full.
                Gives properly calibrated probabilities for all positions.
      Tier 2 : bucket empty (unfilled) only → DSV fallback

    Routing (use_fingerprint=True):
      Tier 1 : fp_query == fp_stored AND embed filled (≈ 1/256 = 0.4% random matches)
      Tier 2 : fp mismatch OR empty (≈ 99.6%) — too restrictive, not recommended
      Only useful if you need strict collision detection per original 2026-04-07 design.

    Tier 2 DSV path: pure lookup from probs_table[prev, tgt]  (zero cost).

    Args:
        val_tokens    : (N,) uint16/int32  validation token sequence
        g_states_val  : (N,) uint64        precomputed rolling hash for val set
        model         : SpiralDSVLanguageModel with sem_fwd (Tier 2 DSV)
        embed         : (TABLE_SIZE, embed_dim) float16  NMF embedding
        W_out         : (embed_dim, vocab_size) float16  NMF output projection
        fingerprint   : (TABLE_SIZE,) uint8  per-bucket fingerprint (used only when use_fingerprint=True)
        nmf_seed      : int  NMF hash seed
        table_bits    : int  log₂ TABLE_SIZE
        base_bytes    : (vocab_size,) int16  UTF-8 bytes per token
        has_leading_space : (vocab_size,) bool
        is_boundary_token : (vocab_size,) bool (optional)
        batch_size    : int    positions per chunk
        use_fingerprint : bool  use fingerprint-based routing (default False;
                                set True only for original 2026-04-07-style collision detection)
        verbose       : bool   print progress

    Returns:
        (bpb, val_loss)
    """
    N  = len(val_tokens)
    V  = model.vocab_size

    if verbose:
        print(f"\n[Hybrid-Eval] Evaluating BPB on {N:,} tokens  "
              f"(TABLE_BITS={table_bits})", flush=True)

    t0 = time.time()

    # -- Prepare NMF routing constants -----------------------------------------
    SHIFT    = np.uint64(64 - table_bits)
    FP_SHIFT = np.uint64(64 - table_bits - 8)
    seed_u64 = np.uint64(nmf_seed)
    TABLE_SIZE = 1 << table_bits

    # embed fill mask: bucket is "filled" when its embedding vector is non-zero
    embed_f32  = embed.astype(np.float32)
    W_out_f32  = W_out.astype(np.float32)
    embed_norm = np.linalg.norm(embed_f32, axis=1)   # (TABLE_SIZE,)
    embed_fill = embed_norm > 1e-6                    # (TABLE_SIZE,) bool

    # -- Precompute Tier 2 DSV score table (V×V, one-time GPU HGEMM) -----------
    # raw_scores_table[prev, tgt] ≈ p(tgt | prev) estimated via XOR-bundle dot product.
    # Key improvement: use temperature-scaled softmax instead of 0.5+0.49×score.
    #
    # The old formula (0.5 + 0.49 * score) had a 0.5 "zero-point" that dominated after
    # row-normalisation: p_correct(score=+0.1) ≈ 0.549/512 ≈ 0.001 (near-uniform).
    #
    # With softmax(score × DSV_TEMP): exp(DSV_TEMP × 0.1) separates signal from noise.
    # DSV_TEMP=50 → exp(50×0.1)=148, p_correct ≈ 148/1172 ≈ 0.126 → BPB drops ~1.0–1.5.
    # DSV_TEMP=0  → reverts to normalised raw scores (uniform for zero-mean scores).
    # Env var DSV_TEMPERATURE overrides at runtime for tuning.
    #
    # If W_out is provided (from NMF Tier 1), also project DSV scores through the
    # NMF output vocabulary matrix for cross-tier calibration:
    #   dsv_calibrated[prev, :] = softmax((raw_score[prev, :] @ W_out.T @ W_out) × T)
    # This maps DSV similarity into the same semantic space as the NMF predictions.
    if verbose:
        print(f"[Hybrid-Eval] Precomputing ({V},{V}) DSV score table...", flush=True)
    t_pre = time.time()
    model._ensure_pm1_cache()
    n_bits = model.n_words * 64

    from _gpu import gpu_matmul_f16
    raw_scores_table = gpu_matmul_f16(
        model._sem_fwd_pm1, model._codebook_pm1.T
    ) / n_bits   # (V, V) float32

    # Temperature for DSV softmax (env var override: DSV_TEMPERATURE)
    import os as _os
    _dsv_temp = float(_os.environ.get("DSV_TEMPERATURE", "50.0"))

    # Optional: project through NMF W_out kernel for cross-tier calibration
    # W_out.T @ W_out is (V, embed_dim) @ (embed_dim, V) = NMF vocabulary kernel
    if W_out is not None and embed.shape[1] > 0:
        try:
            _W = W_out.astype(np.float32)   # (embed_dim, V)
            _nmf_kernel = (_W.T @ _W)        # (V, V) — NMF vocabulary co-structure
            _logits = raw_scores_table @ _nmf_kernel   # (V, V) NMF-projected scores
            if verbose:
                print(f"[Hybrid-Eval] NMF-projected DSV table (W_out kernel applied)",
                      flush=True)
        except Exception:
            _logits = raw_scores_table
    else:
        _logits = raw_scores_table

    # Temperature-scaled softmax for Tier 2 probabilities
    if _dsv_temp > 0:
        _ls = _logits * _dsv_temp
        _lmax = _ls.max(axis=1, keepdims=True)
        _ex   = np.exp(_ls - _lmax)
        probs_table = (_ex / (_ex.sum(axis=1, keepdims=True) + 1e-30)).astype(np.float32)
    else:
        # _dsv_temp=0: pure raw score normalisation (no temperature)
        _lmin = _logits.min(axis=1, keepdims=True)
        _shifted = _logits - _lmin + 1e-30
        probs_table = (_shifted / _shifted.sum(axis=1, keepdims=True)).astype(np.float32)

    if verbose:
        print(f"[Hybrid-Eval] DSV score table ready in {time.time()-t_pre:.2f}s  "
              f"({probs_table.nbytes/1e6:.1f} MB)  "
              f"DSV_TEMPERATURE={_dsv_temp}", flush=True)

    # Clip val_tokens once
    tok_i32 = np.clip(val_tokens.astype(np.int32), 0, V - 1)

    # -- Accumulation counters --------------------------------------------------
    total_bits  = 0.0
    total_bytes = 0
    total_nats  = 0.0
    total_toks  = 0
    n_tier1     = 0
    n_tier2     = 0

    # -- Chunk loop (fused Tier 1 + Tier 2) ------------------------------------
    for chunk_start in range(1, N, batch_size):
        chunk_end = min(chunk_start + batch_size, N)
        B = chunk_end - chunk_start

        prev_toks = tok_i32[chunk_start - 1 : chunk_end - 1]   # (B,)
        tgt_toks  = tok_i32[chunk_start     : chunk_end    ]    # (B,)

        # Route positions to Tier 1 / Tier 2
        g_chunk   = g_states_val[chunk_start : chunk_end].astype(np.uint64)
        finalised = (g_chunk ^ seed_u64) * _FMIX64
        buckets   = (finalised >> SHIFT).astype(np.int64)       # (B,)
        fp_query  = ((finalised >> FP_SHIFT) & np.uint64(0xFF)).astype(np.uint8)
        fp_stored = fingerprint[buckets]

        if use_fingerprint:
            # Original fingerprint-based routing: only matches when both bucket AND
            # 8-bit fingerprint match (~1/256 chance for independent contexts).
            tier1_mask = (fp_query == fp_stored) & embed_fill[buckets]
        else:
            # Fill-based routing (recommended): route ALL filled buckets through NMF.
            # Achieves ~100% Tier 1 coverage when table is well-trained.
            # Accepts that NMF gives the AVERAGE distribution for the bucket (not
            # context-specific), which is still far better than DSV's miscalibrated formula.
            tier1_mask = embed_fill[buckets]
        tier2_mask = ~tier1_mask

        # -- Tier 1: NMF softmax -----------------------------------------------
        if tier1_mask.any():
            idx_t1   = np.where(tier1_mask)[0]
            b_bkts   = buckets[idx_t1]
            b_tgt    = tgt_toks[idx_t1]

            logits   = embed_f32[b_bkts] @ W_out_f32    # (n1, V)
            lmax     = logits.max(axis=1, keepdims=True)
            exp_l    = np.exp(logits - lmax)
            probs_t1 = exp_l / (exp_l.sum(axis=1, keepdims=True) + 1e-30)  # (n1, V)
            p_t1     = np.clip(probs_t1[np.arange(len(idx_t1)), b_tgt], 1e-30, 1.0)

            # byte counts for Tier 1 positions
            prev_t1 = np.clip(
                val_tokens[chunk_start + idx_t1 - 1].astype(np.int32),
                0, base_bytes.shape[0] - 1)
            tb_t1 = base_bytes[b_tgt].astype(np.float64)
            if is_boundary_token is not None:
                tb_t1 += (has_leading_space[b_tgt] & ~is_boundary_token[prev_t1]).astype(np.float64)
            else:
                tb_t1 += has_leading_space[b_tgt].astype(np.float64)

            total_bits  += float(-np.log2(p_t1).sum())
            total_bytes += int(tb_t1.sum())
            total_nats  += float(-np.log(p_t1).sum())
            total_toks  += len(idx_t1)
            n_tier1     += len(idx_t1)

        # -- Tier 2: GoldenAxisShift DSV ---------------------------------------
        if tier2_mask.any():
            idx_t2 = np.where(tier2_mask)[0]

            bt2_prev = prev_toks[idx_t2]     # (n2,)
            bt2_tgt  = tgt_toks[idx_t2]      # (n2,)

            # Fast single 2D table lookup
            p_t2 = probs_table[bt2_prev, bt2_tgt].astype(np.float32)

            # byte counts for Tier 2 positions
            prev_t2 = np.clip(
                val_tokens[chunk_start + idx_t2 - 1].astype(np.int32),
                0, base_bytes.shape[0] - 1)
            tb_t2 = base_bytes[bt2_tgt].astype(np.float64)
            if is_boundary_token is not None:
                tb_t2 += (has_leading_space[bt2_tgt] & ~is_boundary_token[prev_t2]).astype(np.float64)
            else:
                tb_t2 += has_leading_space[bt2_tgt].astype(np.float64)

            total_bits  += float(-np.log2(np.clip(p_t2, 1e-30, 1.0)).sum())
            total_bytes += int(tb_t2.sum())
            total_nats  += float(-np.log(np.clip(p_t2, 1e-30, 1.0)).sum())
            total_toks  += len(idx_t2)
            n_tier2     += len(idx_t2)

        if verbose:
            elapsed     = time.time() - t0
            pct         = 100.0 * chunk_end / N
            running_bpb = total_bits / max(total_bytes, 1)
            t1_pct      = 100.0 * n_tier1 / max(n_tier1 + n_tier2, 1)
            print(f"[Hybrid-Eval] {chunk_end:,}/{N:,} ({pct:.1f}%)  "
                  f"running_bpb={running_bpb:.4f}  tier1={t1_pct:.1f}%  "
                  f"elapsed={elapsed:.1f}s", flush=True)

    bpb      = total_bits / max(total_bytes, 1)
    val_loss = total_nats  / max(total_toks, 1)
    elapsed  = time.time() - t0

    if verbose:
        t1_pct = 100.0 * n_tier1 / max(n_tier1 + n_tier2, 1)
        print(f"[Hybrid-Eval] Done: BPB={bpb:.4f}  val_loss={val_loss:.4f}  "
              f"elapsed={elapsed:.1f}s  tier1={t1_pct:.1f}% ({n_tier1:,})  "
              f"tier2={100-t1_pct:.1f}% ({n_tier2:,})", flush=True)

    return float(bpb), float(val_loss)


# =============================================================================
# Tokeniser helpers
# =============================================================================

def build_token_byte_arrays(
    sp_model,
    vocab_size: int = VOCAB_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-token byte-count, leading-space, boundary arrays.

    Identical to the reference train_gpt.py formula:
      is_boundary_token initialised ALL True; only cleared for normal word-pieces.
      tok_bytes = base_bytes[tgt] + (has_leading_space[tgt] & ~is_boundary_token[prev])
    """
    table_size        = max(vocab_size, sp_model.GetPieceSize())
    base_bytes        = np.zeros(table_size, dtype=np.int16)
    has_leading_space = np.zeros(table_size, dtype=bool)
    is_boundary_token = np.ones(table_size,  dtype=bool)

    for tok_id in range(min(vocab_size, sp_model.GetPieceSize())):
        if sp_model.is_byte(tok_id):
            base_bytes[tok_id] = 1
            continue
        piece = sp_model.IdToPiece(tok_id)
        if piece.startswith('\u2581'):
            has_leading_space[tok_id] = True
            piece = piece[1:]
        try:
            base_bytes[tok_id] = len(piece.encode('utf-8'))
        except Exception:
            base_bytes[tok_id] = 1
        is_boundary_token[tok_id] = False

    return (
        base_bytes[:vocab_size],
        has_leading_space[:vocab_size],
        is_boundary_token[:vocab_size],
    )


# =============================================================================
# Artifact size check
# =============================================================================

def check_artifact_size(
    artifact_path: str,
    code_bytes: int = 0,
    verbose: bool = True,
) -> Tuple[int, bool]:
    """Check artifact + code ≤ 16 MB."""
    import os
    artifact_bytes = os.path.getsize(artifact_path)
    total_bytes    = artifact_bytes + code_bytes
    passes         = total_bytes <= ARTIFACT_LIMIT

    if verbose:
        print(f"\n[Hybrid] Artifact size check:")
        print(f"  Artifact : {artifact_bytes:>12,} bytes  ({artifact_bytes/1e6:.3f} MB)")
        print(f"  Code     : {code_bytes:>12,} bytes  ({code_bytes/1e6:.3f} MB)")
        print(f"  Total    : {total_bytes:>12,} bytes  ({total_bytes/1e6:.3f} MB)")
        print(f"  Limit    : {ARTIFACT_LIMIT:>12,} bytes  ({ARTIFACT_LIMIT/1e6:.3f} MB)")
        print(f"  Result   : {'PASS' if passes else 'FAIL'}")

    return total_bytes, passes
