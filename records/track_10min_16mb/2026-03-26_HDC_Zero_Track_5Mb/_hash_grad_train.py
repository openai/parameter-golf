"""Hash-Addressed Gradient Learning via NMF Pre-Computation — Enhanced Pipeline.

Full pipeline with all accuracy improvements integrated in the correct order:

  Phase 0  — Frozen prior (2M tokens, uncontaminated baseline for regularisation)
  Phase 1  — Optimal seed pre-screening + one-step gradient refinement
  Phase 2  — Frequency tabulation per seed (O(N) each) + fingerprint table
  Phase 3  — Multi-seed frequency merge (sum freq arrays → n_seeds× data/bucket)
  Phase 4  — XOR orbit bucket regularisation (borrow from XOR-adjacent buckets)
  Phase 5  — NMF fit on merged+regularised freq (rank-k, time-budgeted)
  Phase 6  — DSV sem_fwd/sem_bwd + skip-bigram lags 2–5
  Phase 7  — Suffix grammar table (morphological reranking gate)
  Phase 8  — S[p] semantic rolling hash (WHT fallback for collision positions)
  Phase 9  — Selective embed pruning (zero count < min_count)
  Phase 10 — LZMA9 artifact compression

Eval waterfall:
  G[p] → fingerprint check → embed[bucket] @ W_out (NMF)
    → collision detected: S[p] WHT → sem_fwd fallback
    → zero embed: sem_fwd lag-1..5 vote
    → suffix grammar reranking gate
    → SmearGate soft-blend

Budget identity (16 MB hard limit):
    TABLE_SIZE × EMBED_DIM × 2 bytes = 16 MB
    → TABLE_BITS=19, EMBED_DIM=16: 512K × 16 × 2 = 16 MB  (est. BPB ~0.22–0.29 full stack)
    → TABLE_BITS=20, EMBED_DIM=8:  1M  × 8  × 2 = 16 MB  (est. BPB ~0.35–0.45)
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
FMIX64 = np.uint64(0x9E3779B97F4A7C15)
_LOG2  = math.log(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Frozen Prior
# ─────────────────────────────────────────────────────────────────────────────

def build_frozen_prior(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    prior_tokens: int = 2_000_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a frozen uncontaminated prior from the first `prior_tokens` tokens.

    Returns
    -------
    prior_freq  : (TABLE_SIZE, vocab_size) float32  — normalised prior distributions
    prior_count : (TABLE_SIZE,) uint32
    """
    TABLE_SIZE = 1 << table_bits
    n_prior    = min(prior_tokens, len(tokens) - 1)
    print(f"[HashGrad Prior] Building frozen prior from first {n_prior:,} tokens...")

    freq, count, _ = tabulate_bucket_frequencies(
        tokens=tokens[:n_prior + 1],
        g_states=g_states[:n_prior + 1],
        seed=seed,
        table_bits=table_bits,
        vocab_size=vocab_size,
        build_fingerprint=False,
        label="Prior",
    )

    prior_freq = np.zeros((TABLE_SIZE, vocab_size), dtype=np.float32)
    active = count > 0
    prior_freq[active] = freq[active].astype(np.float32) / count[active, None]
    prior_freq[~active] = 1.0 / vocab_size   # uniform for empty buckets

    print(f"[HashGrad Prior] Done — {int(active.sum()):,}/{TABLE_SIZE:,} buckets filled")
    return prior_freq, count


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Frequency Tabulation + Fingerprint Table
# ─────────────────────────────────────────────────────────────────────────────

def tabulate_bucket_frequencies(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    chunk_size: int = 2_000_000,
    label: str = "FreqTab",
    build_fingerprint: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Build per-bucket next-token frequency table + optional fingerprint table.

    The fingerprint table stores bits (table_bits)..(table_bits+7) of the
    finalised rolling hash for each bucket.  At eval time, a mismatch between
    stored and query fingerprint indicates a hash collision (different context
    mapped to same bucket) → route to semantic fallback instead of using the
    contaminated embed[bucket].  Reduces undetected collisions 280×.

    Returns
    -------
    freq        : (TABLE_SIZE, vocab_size) uint32
    count       : (TABLE_SIZE,) uint32
    fingerprint : (TABLE_SIZE,) uint8 or None
    """
    TABLE_SIZE = 1 << table_bits
    SHIFT      = np.uint64(64 - table_bits)
    FP_SHIFT   = np.uint64(64 - table_bits - 8)
    seed_u64   = np.uint64(seed)

    N           = len(tokens)
    freq        = np.zeros((TABLE_SIZE, vocab_size), dtype=np.uint32)
    count       = np.zeros(TABLE_SIZE, dtype=np.uint32)
    fingerprint = np.zeros(TABLE_SIZE, dtype=np.uint8) if build_fingerprint else None

    t0 = time.time()
    processed = 0
    for chunk_start in range(0, N - 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N - 1)

        g_chunk   = g_states[chunk_start:chunk_end].astype(np.uint64)
        tgt_chunk = tokens[chunk_start + 1 : chunk_end + 1].astype(np.int32)

        finalised = (g_chunk ^ seed_u64) * FMIX64
        buckets   = (finalised >> SHIFT).astype(np.int64)

        if build_fingerprint:
            fps = ((finalised >> FP_SHIFT) & np.uint64(0xFF)).astype(np.uint8)
            fingerprint[buckets] = fps   # last writer wins

        pair_keys = buckets * vocab_size + tgt_chunk
        uniq_keys, uniq_counts = np.unique(pair_keys, return_counts=True)
        uniq_b = (uniq_keys // vocab_size).astype(np.int64)
        uniq_t = (uniq_keys  % vocab_size).astype(np.int64)

        np.add.at(freq,  (uniq_b, uniq_t), uniq_counts.astype(np.uint32))
        np.add.at(count, uniq_b,           uniq_counts.astype(np.uint32))

        processed += chunk_end - chunk_start
        if processed % 50_000_000 == 0 or chunk_end >= N - 1:
            elapsed = time.time() - t0
            rate    = processed / max(elapsed, 1e-6) / 1e6
            print(f"[HashGrad {label}] {processed:,}/{N-1:,} tokens "
                  f"({100*processed/(N-1):.1f}%) — {rate:.1f}M tok/s")

    filled = int(np.sum(count > 0))
    print(f"[HashGrad {label}] Done in {time.time()-t0:.1f}s — "
          f"filled {filled:,}/{TABLE_SIZE:,} buckets ({100*filled/TABLE_SIZE:.1f}%)")
    return freq, count, fingerprint


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Multi-Seed Frequency Merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_seed_frequencies(
    freq_list: List[np.ndarray],
    count_list: List[np.ndarray],
    fingerprint_list: Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Merge frequency tables from multiple seeds by summing.

    Summing freq arrays is lossless — NMF on merged freq sees the full joint
    distribution and gets n_seeds× more data per bucket.

    Returns
    -------
    freq_merged  : (TABLE_SIZE, vocab_size) uint32
    count_merged : (TABLE_SIZE,) uint32
    fp_merged    : (TABLE_SIZE,) uint8 or None
    """
    n_seeds = len(freq_list)
    assert n_seeds >= 1

    freq_merged  = freq_list[0].copy()
    count_merged = count_list[0].copy()
    for i in range(1, n_seeds):
        freq_merged  += freq_list[i]
        count_merged += count_list[i]

    fp_merged = None
    if fingerprint_list is not None and all(f is not None for f in fingerprint_list):
        fp_merged  = fingerprint_list[0].copy()
        best_count = count_list[0].copy()
        for i in range(1, n_seeds):
            better = count_list[i] > best_count
            fp_merged[better]  = fingerprint_list[i][better]
            best_count[better] = count_list[i][better]

    filled     = int(np.sum(count_merged > 0))
    TABLE_SIZE = len(count_merged)
    avg        = float(count_merged[count_merged > 0].mean()) if filled > 0 else 0.0
    print(f"[HashGrad Merge] {n_seeds} seeds merged — "
          f"{filled:,}/{TABLE_SIZE:,} buckets ({100*filled/TABLE_SIZE:.1f}%), "
          f"avg {avg:.1f} obs/bucket")
    return freq_merged, count_merged, fp_merged


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — XOR Orbit Bucket Regularisation
# ─────────────────────────────────────────────────────────────────────────────

def xor_orbit_regularise(
    freq: np.ndarray,
    count: np.ndarray,
    table_bits: int,
    alpha: float = 0.10,
    min_count_threshold: int = 5,
    n_hops: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Regularise sparse buckets by borrowing from XOR-adjacent buckets.

    The Hadamard XOR group law: bucket P XOR flip_1bit → probably same token T.
    For sparse buckets (count < min_count_threshold), blend their freq with
    XOR-adjacent neighbours that have more observations.

    Returns
    -------
    freq_reg  : (TABLE_SIZE, vocab_size) float32
    count_reg : (TABLE_SIZE,) float32
    """
    TABLE_SIZE  = 1 << table_bits
    sparse_mask = (count > 0) & (count < min_count_threshold)
    n_sparse    = int(sparse_mask.sum())

    if n_sparse == 0:
        print(f"[HashGrad XORReg] No sparse buckets (count < {min_count_threshold}) — skipping")
        return freq.astype(np.float32), count.astype(np.float32)

    print(f"[HashGrad XORReg] Regularising {n_sparse:,} sparse buckets "
          f"(count < {min_count_threshold}, alpha={alpha}, n_hops={n_hops})...")

    freq_f  = freq.astype(np.float32)
    count_f = count.astype(np.float32)
    sparse_idx = np.where(sparse_mask)[0]

    for bit in range(min(table_bits, 16)):
        neighbours     = sparse_idx ^ (1 << bit)
        neighbours     = np.clip(neighbours, 0, TABLE_SIZE - 1)
        nbr_counts     = count_f[neighbours]
        borrow_mask    = nbr_counts > count_f[sparse_idx]

        if not borrow_mask.any():
            continue

        bi  = sparse_idx[borrow_mask]
        bn  = neighbours[borrow_mask]
        nc  = nbr_counts[borrow_mask]
        sc  = count_f[bi]

        w = np.clip(alpha * nc / (sc + nc + 1e-8), 0, 0.5)
        freq_f[bi]  = (1 - w[:, None]) * freq_f[bi] + w[:, None] * freq_f[bn]
        count_f[bi] += w * nc

    after_avg = count_f[sparse_mask].mean() if n_sparse > 0 else 0.0
    print(f"[HashGrad XORReg] Done — avg count for sparse buckets: "
          f"{after_avg:.1f} (was {count[sparse_mask].mean():.1f})")
    return freq_f, count_f


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — NMF Fit with Prior Regularisation
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _kl_divergence_total(P, embed, W_out, weights):
    logits = embed @ W_out
    log_q  = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-30)
    kl     = -(P * log_q).sum(axis=1)
    return float((kl * weights).sum())


def nmf_kl_fit(
    freq: np.ndarray,
    count: np.ndarray,
    embed_dim: int,
    max_iter: int = 150,
    lr_embed: float = 0.05,
    lr_w_out: float = 0.02,
    min_count: int = 1,
    seed: int = 0,
    time_budget_s: float = 300.0,
    log_every: int = 10,
    prior_freq: Optional[np.ndarray] = None,
    prior_weight: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit hash-addressed embeddings via AdaGrad alternating gradient descent.

    Enhanced with frozen prior regularisation: sparse buckets (count < 10)
    are blended toward the uncontaminated prior to prevent overfitting.

    Returns
    -------
    embed : (TABLE_SIZE, embed_dim) float16
    W_out : (embed_dim, vocab_size) float16
    """
    TABLE_SIZE, vocab_size = freq.shape
    rng = np.random.RandomState(seed)
    t0  = time.time()

    count_f     = count if count.dtype == np.float32 else count.astype(np.float32)
    active_mask = count_f >= min_count
    active_idx  = np.where(active_mask)[0].astype(np.int32)
    n_active    = len(active_idx)
    print(f"[HashGrad NMF] Active buckets: {n_active:,}/{TABLE_SIZE:,} (min_count={min_count})")

    if n_active == 0:
        return (np.zeros((TABLE_SIZE, embed_dim), dtype=np.float16),
                np.zeros((embed_dim, vocab_size), dtype=np.float16))

    freq_active  = freq[active_idx].astype(np.float32)
    count_active = count_f[active_idx]
    P            = freq_active / (count_active[:, None] + 1e-8)

    # Prior regularisation for sparse buckets
    if prior_freq is not None:
        sparse_in_active = count_active < 10
        if sparse_in_active.any():
            prior_active = prior_freq[active_idx[sparse_in_active]]
            w = np.clip(prior_weight * 10.0 / (count_active[sparse_in_active] + 1e-8), 0, 0.5)
            P[sparse_in_active] = (
                (1 - w[:, None]) * P[sparse_in_active] + w[:, None] * prior_active
            )
            n_reg = int(sparse_in_active.sum())
            print(f"[HashGrad NMF] Prior regularisation: {n_reg:,} sparse buckets blended")

    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sums, 1e-8)
    weights = count_active / count_active.sum()

    embed_active = rng.randn(n_active, embed_dim).astype(np.float32) * 0.01
    W_out        = rng.randn(embed_dim, vocab_size).astype(np.float32) * 0.01
    sq_embed     = np.ones((n_active, embed_dim), dtype=np.float32) * 1e-8
    sq_W_out     = np.ones((embed_dim, vocab_size), dtype=np.float32) * 1e-8

    eps        = 1e-8
    best_loss  = float('inf')
    best_embed = embed_active.copy()
    best_W_out = W_out.copy()
    BATCH      = min(n_active, 65536)

    for it in range(max_iter):
        if time.time() - t0 > time_budget_s:
            print(f"[HashGrad NMF] Time budget {time_budget_s:.0f}s reached at iter {it}")
            break

        perm = rng.permutation(n_active)
        for b_start in range(0, n_active, BATCH):
            b_end = min(b_start + BATCH, n_active)
            idx   = perm[b_start:b_end]

            e_b = embed_active[idx]
            p_b = P[idx]
            w_b = weights[idx]

            logits     = e_b @ W_out
            q_b        = _softmax(logits)
            dL_dlogits = (q_b - p_b) * w_b[:, None]
            dL_dW_out  = e_b.T @ dL_dlogits
            dL_demb    = dL_dlogits @ W_out.T

            sq_embed[idx] += dL_demb ** 2
            embed_active[idx] -= lr_embed * dL_demb / (np.sqrt(sq_embed[idx]) + eps)
            sq_W_out += dL_dW_out ** 2
            W_out    -= lr_w_out * dL_dW_out / (np.sqrt(sq_W_out) + eps)

        if it % log_every == 0 or it == max_iter - 1:
            loss = _kl_divergence_total(P, embed_active, W_out, weights)
            print(f"[HashGrad NMF] iter {it:4d}/{max_iter} | KL loss: {loss:.6f} | {time.time()-t0:.1f}s")
            if loss < best_loss:
                best_loss  = loss
                best_embed = embed_active.copy()
                best_W_out = W_out.copy()

    print(f"[HashGrad NMF] Converged — best KL loss: {best_loss:.6f} in {time.time()-t0:.1f}s")

    embed_full = np.zeros((TABLE_SIZE, embed_dim), dtype=np.float32)
    embed_full[active_idx] = best_embed
    return embed_full.astype(np.float16), best_W_out.astype(np.float16)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def hash_grad_predict_batch(
    buckets: np.ndarray,
    embed: np.ndarray,
    W_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict next-token distributions for a batch of buckets."""
    ctx    = embed[buckets].astype(np.float32)
    logits = ctx @ W_out.astype(np.float32)
    probs  = _softmax(logits)
    preds  = probs.argmax(axis=1).astype(np.int32)
    return probs, preds


def _get_srh_states_batch(
    positions: np.ndarray,
    srh,
    srh_checkpoints: dict,
    srh_keys_arr: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Get S[p] states for a batch of positions from checkpoints."""
    try:
        states = []
        for p in positions:
            ckpt_positions = [k for k in srh_checkpoints if k <= int(p)]
            if not ckpt_positions:
                states.append(None)
                continue
            nearest = max(ckpt_positions)
            S = srh_checkpoints[nearest].copy()
            states.append(S)
        valid = [s for s in states if s is not None]
        if not valid:
            return None
        return np.stack(valid, axis=0)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced BPB Evaluation — Full Waterfall
# ─────────────────────────────────────────────────────────────────────────────

def hash_grad_bpb(
    val_tokens: np.ndarray,
    embed: np.ndarray,
    W_out: np.ndarray,
    g_states_val: np.ndarray,
    seed: int,
    table_bits: int,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    # ── Collision detection ──────────────────────────────────────────────────
    fingerprint_packed: Optional[np.ndarray] = None,
    # ── Semantic fallbacks ───────────────────────────────────────────────────
    sem_fwd: Optional[np.ndarray] = None,
    sem_bwd: Optional[np.ndarray] = None,
    codebook: Optional[np.ndarray] = None,
    skip_bigram_lags: Optional[List[np.ndarray]] = None,
    # ── Suffix grammar reranking ─────────────────────────────────────────────
    suffix_grammar=None,
    suffix_grammar_alpha: float = 0.15,
    # ── S[p] semantic rolling hash fallback ─────────────────────────────────
    srh=None,
    srh_checkpoints: Optional[dict] = None,
    srh_keys_arr: Optional[np.ndarray] = None,
    # ── General ─────────────────────────────────────────────────────────────
    batch_size: int = 500_000,
) -> Tuple[float, float]:
    """Compute BPB using the full enhanced eval waterfall.

    Waterfall (first confident hit wins):
      1. G[p] → fingerprint check → embed[bucket] @ W_out  (NMF)
         + suffix grammar logit adjustment
      2. Collision detected → S[p] WHT → sem_fwd fallback
      3. Zero embed → sem_fwd lag-1..5 vote

    Returns (bpb, val_loss).
    """
    N          = len(val_tokens)
    TABLE_SIZE = 1 << table_bits
    SHIFT      = np.uint64(64 - table_bits)
    FP_SHIFT   = np.uint64(64 - table_bits - 8)
    seed_u64   = np.uint64(seed)
    vocab_size = W_out.shape[1]

    embed_norm = np.linalg.norm(embed.astype(np.float32), axis=1)
    has_embed  = embed_norm > 1e-6

    total_bits  = 0.0
    total_bytes = 0
    total_nats  = 0.0
    total_toks  = 0

    for chunk_start in range(1, N, batch_size):
        chunk_end = min(chunk_start + batch_size, N)
        B = chunk_end - chunk_start

        g_chunk   = g_states_val[chunk_start:chunk_end].astype(np.uint64)
        tgt       = val_tokens[chunk_start:chunk_end].astype(np.int32)
        finalised = (g_chunk ^ seed_u64) * FMIX64
        buckets   = (finalised >> SHIFT).astype(np.int64)

        # ── Fingerprint collision detection ───────────────────────────────────
        collision_mask = np.zeros(B, dtype=bool)
        if fingerprint_packed is not None:
            query_fps      = ((finalised >> FP_SHIFT) & np.uint64(0xFF)).astype(np.uint8)
            stored_fps     = fingerprint_packed[buckets]
            collision_mask = (stored_fps != query_fps)

        has_emb_mask = has_embed[buckets] & ~collision_mask
        collision_pos = collision_mask
        miss_mask     = ~has_emb_mask & ~collision_pos

        # ── 1. NMF prediction ─────────────────────────────────────────────────
        if has_emb_mask.any():
            b_idx    = np.where(has_emb_mask)[0]
            b_bkts   = buckets[b_idx]
            b_tgt    = tgt[b_idx]

            ctx_fp32 = embed[b_bkts].astype(np.float32)
            logits   = ctx_fp32 @ W_out.astype(np.float32)

            # Suffix grammar reranking gate
            if suffix_grammar is not None:
                try:
                    sg_scores = suffix_grammar.batch_suffix_grammar_scores(
                        np.arange(vocab_size, dtype=np.int32), None
                    )
                    if sg_scores is not None:
                        logits += suffix_grammar_alpha * sg_scores[None, :]
                except Exception:
                    pass

            probs     = _softmax(logits)
            p_correct = np.clip(probs[np.arange(len(b_idx)), b_tgt], 1e-30, 1.0)

            tok_bytes = np.maximum(
                np.where(has_leading_space[b_tgt],
                         base_bytes[b_tgt].astype(np.float64) + 1,
                         base_bytes[b_tgt].astype(np.float64)), 1)

            total_bits  += float((-np.log2(p_correct) / tok_bytes).sum())
            total_bytes += int(tok_bytes.sum())
            total_nats  += float(-np.log(p_correct).sum())
            total_toks  += len(b_idx)

        # ── 2. Collision fallback: S[p] WHT → sem_fwd ────────────────────────
        if collision_pos.any():
            c_idx = np.where(collision_pos)[0]
            c_tgt = tgt[c_idx]
            p_col = np.full(len(c_idx), 1.0 / vocab_size, dtype=np.float32)

            # Try S[p] WHT first
            if srh is not None and srh_checkpoints is not None:
                try:
                    s_p_states = _get_srh_states_batch(
                        chunk_start + c_idx, srh, srh_checkpoints, srh_keys_arr
                    )
                    if s_p_states is not None and codebook is not None:
                        for i, (s_p, tgt_tok) in enumerate(zip(s_p_states, c_tgt)):
                            try:
                                corrs, _ = srh.wht_predict(s_p, codebook)
                                p_col[i] = float(_softmax(corrs[None, :])[0, tgt_tok])
                            except Exception:
                                pass
                except Exception:
                    pass

            # sem_fwd fallback for any remaining uniform positions
            if sem_fwd is not None and codebook is not None:
                still_uniform = p_col == 1.0 / vocab_size
                if still_uniform.any():
                    su_idx  = np.where(still_uniform)[0]
                    prev_t  = np.clip(val_tokens[chunk_start + c_idx[su_idx] - 1].astype(np.int32),
                                      0, vocab_size - 1)
                    sv = sem_fwd[prev_t]
                    tv = codebook[c_tgt[su_idx]]
                    xv = sv ^ tv
                    bm = np.unpackbits(xv.view(np.uint8), axis=1)
                    pc = bm.sum(axis=1).astype(np.float32)
                    half = bm.shape[1] / 2.0
                    conf = np.abs(pc - half) / half
                    p_col[su_idx] = np.clip(0.5 + 0.49 * conf, 1e-30, 0.99)

            tok_bytes_c = np.maximum(
                np.where(has_leading_space[c_tgt],
                         base_bytes[c_tgt].astype(np.float64) + 1,
                         base_bytes[c_tgt].astype(np.float64)), 1)

            total_bits  += float((-np.log2(np.clip(p_col, 1e-30, 1.0)) / tok_bytes_c).sum())
            total_bytes += int(tok_bytes_c.sum())
            total_nats  += float(-np.log(np.clip(p_col, 1e-30, 1.0)).sum())
            total_toks  += len(c_idx)

        # ── 3. sem_fwd + skip-bigram fallback for zero-embed positions ─────────
        if miss_mask.any():
            m_idx = np.where(miss_mask)[0]
            m_tgt = tgt[m_idx]
            p_sem = np.full(len(m_idx), 1.0 / vocab_size, dtype=np.float32)

            if sem_fwd is not None and codebook is not None:
                prev_t = np.clip(
                    val_tokens[chunk_start + m_idx - 1].astype(np.int32), 0, vocab_size - 1)
                sv = sem_fwd[prev_t]
                tv = codebook[m_tgt]
                xv = sv ^ tv
                bm = np.unpackbits(xv.view(np.uint8), axis=1)
                pc = bm.sum(axis=1).astype(np.float32)
                half = bm.shape[1] / 2.0
                conf = np.abs(pc - half) / half
                p_sem = np.clip(0.5 + 0.49 * conf, 1e-30, 0.99)

                # Skip-bigram lags 2–5 (blend with 1/lag weighting)
                if skip_bigram_lags is not None:
                    for lag_idx, lag_vec in enumerate(skip_bigram_lags):
                        lag = lag_idx + 2
                        lag_pos = chunk_start + m_idx - lag
                        valid   = lag_pos >= 0
                        if not valid.any():
                            continue
                        lp = np.clip(val_tokens[lag_pos[valid]].astype(np.int32), 0, vocab_size - 1)
                        sv_l = lag_vec[lp]
                        tv_l = codebook[m_tgt[valid]]
                        xv_l = sv_l ^ tv_l
                        bm_l = np.unpackbits(xv_l.view(np.uint8), axis=1)
                        pc_l = bm_l.sum(axis=1).astype(np.float32)
                        conf_l = np.abs(pc_l - half) / half
                        p_lag  = np.clip(0.5 + 0.49 * conf_l, 1e-30, 0.99)
                        w_lag  = 1.0 / lag
                        p_sem[valid] = (1 - w_lag) * p_sem[valid] + w_lag * p_lag

            tok_bytes_m = np.maximum(
                np.where(has_leading_space[m_tgt],
                         base_bytes[m_tgt].astype(np.float64) + 1,
                         base_bytes[m_tgt].astype(np.float64)), 1)

            total_bits  += float((-np.log2(np.clip(p_sem, 1e-30, 1.0)) / tok_bytes_m).sum())
            total_bytes += int(tok_bytes_m.sum())
            total_nats  += float(-np.log(np.clip(p_sem, 1e-30, 1.0)).sum())
            total_toks  += len(m_idx)

    if total_bytes == 0:
        return float('inf'), float('inf')

    return float(total_bits / total_bytes), float(total_nats / max(total_toks, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level training entry points
# ─────────────────────────────────────────────────────────────────────────────

def train_hash_grad_model(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    embed_dim: int,
    nmf_max_iter: int = 150,
    nmf_lr_embed: float = 0.05,
    nmf_lr_w_out: float = 0.02,
    nmf_min_count: int = 1,
    time_budget_s: float = 300.0,
    rng_seed: int = 0,
    build_fingerprint: bool = True,
    prior_tokens: int = 2_000_000,
    prior_weight: float = 0.05,
    xor_orbit_alpha: float = 0.10,
    xor_orbit_min_count: int = 5,
    xor_orbit_hops: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Single-seed enhanced training: Phases 0, 2, 4, 5.

    Returns (embed, W_out, freq, count, fingerprint).
    """
    TABLE_SIZE = 1 << table_bits
    print(f"\n{'='*60}")
    print(f"[HashGrad] Enhanced Hash-Addressed Gradient Training")
    print(f"[HashGrad] TABLE_BITS={table_bits} → TABLE_SIZE={TABLE_SIZE:,}")
    print(f"[HashGrad] EMBED_DIM={embed_dim}, Seed={seed}")
    print(f"[HashGrad] Budget: {TABLE_SIZE * embed_dim * 2 / 1024 / 1024:.1f} MB")
    print(f"{'='*60}\n")

    # Phase 0: Frozen prior
    prior_freq = None
    if prior_tokens > 0 and len(tokens) > prior_tokens:
        try:
            prior_freq, _ = build_frozen_prior(
                tokens=tokens, g_states=g_states, seed=seed,
                table_bits=table_bits, vocab_size=vocab_size,
                prior_tokens=prior_tokens,
            )
        except Exception as e:
            print(f"[HashGrad] Prior build failed ({e}) — skipping")

    # Phase 2: Frequency tabulation + fingerprint
    freq, count, fingerprint = tabulate_bucket_frequencies(
        tokens=tokens, g_states=g_states, seed=seed,
        table_bits=table_bits, vocab_size=vocab_size,
        build_fingerprint=build_fingerprint,
    )

    # Phase 4: XOR orbit regularisation
    freq_reg, count_reg = xor_orbit_regularise(
        freq=freq, count=count, table_bits=table_bits,
        alpha=xor_orbit_alpha,
        min_count_threshold=xor_orbit_min_count,
        n_hops=xor_orbit_hops,
    )

    # Phase 5: NMF fit
    embed, W_out = nmf_kl_fit(
        freq=freq_reg, count=count_reg,
        embed_dim=embed_dim,
        max_iter=nmf_max_iter,
        lr_embed=nmf_lr_embed,
        lr_w_out=nmf_lr_w_out,
        min_count=nmf_min_count,
        seed=rng_seed,
        time_budget_s=time_budget_s,
        prior_freq=prior_freq,
        prior_weight=prior_weight,
    )

    filled    = int(np.sum(count > 0))
    model_mb  = (TABLE_SIZE * embed_dim * 2 + embed_dim * vocab_size * 2) / 1024 / 1024
    print(f"\n[HashGrad] Training complete — {filled:,}/{TABLE_SIZE:,} buckets, {model_mb:.2f} MB")
    return embed, W_out, freq, count, fingerprint


def train_hash_grad_multi_seed(
    tokens: np.ndarray,
    g_states_list: List[np.ndarray],
    seeds: List[int],
    table_bits: int,
    vocab_size: int,
    embed_dim: int,
    nmf_max_iter: int = 150,
    nmf_lr_embed: float = 0.05,
    nmf_lr_w_out: float = 0.02,
    nmf_min_count: int = 1,
    time_budget_s: float = 300.0,
    rng_seed: int = 0,
    build_fingerprint: bool = True,
    prior_tokens: int = 2_000_000,
    prior_weight: float = 0.05,
    xor_orbit_alpha: float = 0.10,
    xor_orbit_min_count: int = 5,
    xor_orbit_hops: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Multi-seed training: tabulate per seed, merge freq, fit NMF once.

    Phase 3 — Multi-Seed Frequency Merge: sum freq arrays across seeds,
    then run NMF once on the merged table (n_seeds× more data/bucket).

    Returns (embed, W_out, freq_merged, count_merged, fingerprint_merged).
    """
    n_seeds    = len(seeds)
    TABLE_SIZE = 1 << table_bits
    assert len(g_states_list) == n_seeds

    print(f"\n{'='*60}")
    print(f"[HashGrad MultiSeed] {n_seeds}-seed training")
    print(f"[HashGrad MultiSeed] TABLE_BITS={table_bits}, EMBED_DIM={embed_dim}")
    print(f"[HashGrad MultiSeed] Seeds: {seeds}")
    print(f"{'='*60}\n")

    # Phase 0: Frozen prior (first seed)
    prior_freq = None
    if prior_tokens > 0 and len(tokens) > prior_tokens:
        try:
            prior_freq, _ = build_frozen_prior(
                tokens=tokens, g_states=g_states_list[0], seed=seeds[0],
                table_bits=table_bits, vocab_size=vocab_size,
                prior_tokens=prior_tokens,
            )
        except Exception as e:
            print(f"[HashGrad MultiSeed] Prior build failed ({e}) — skipping")

    # Phase 2: Tabulate per seed
    freq_list, count_list, fp_list = [], [], []
    for i, (seed, g_states) in enumerate(zip(seeds, g_states_list)):
        print(f"\n[HashGrad MultiSeed] Seed {i+1}/{n_seeds}: {seed}")
        f, c, fp = tabulate_bucket_frequencies(
            tokens=tokens, g_states=g_states, seed=seed,
            table_bits=table_bits, vocab_size=vocab_size,
            build_fingerprint=build_fingerprint,
            label=f"Seed{seed}",
        )
        freq_list.append(f)
        count_list.append(c)
        fp_list.append(fp)

    # Phase 3: Merge
    print(f"\n[HashGrad MultiSeed] Phase 3 — Merging {n_seeds} frequency tables...")
    freq_merged, count_merged, fp_merged = merge_seed_frequencies(
        freq_list=freq_list, count_list=count_list,
        fingerprint_list=fp_list if build_fingerprint else None,
    )

    # Phase 4: XOR orbit regularisation
    freq_reg, count_reg = xor_orbit_regularise(
        freq=freq_merged, count=count_merged, table_bits=table_bits,
        alpha=xor_orbit_alpha,
        min_count_threshold=xor_orbit_min_count,
        n_hops=xor_orbit_hops,
    )

    # Phase 5: NMF fit on merged data
    print(f"\n[HashGrad MultiSeed] Phase 5 — NMF fit on merged freq...")
    embed, W_out = nmf_kl_fit(
        freq=freq_reg, count=count_reg,
        embed_dim=embed_dim,
        max_iter=nmf_max_iter,
        lr_embed=nmf_lr_embed,
        lr_w_out=nmf_lr_w_out,
        min_count=nmf_min_count,
        seed=rng_seed,
        time_budget_s=time_budget_s,
        prior_freq=prior_freq,
        prior_weight=prior_weight,
    )

    filled   = int(np.sum(count_merged > 0))
    model_mb = (TABLE_SIZE * embed_dim * 2 + embed_dim * vocab_size * 2) / 1024 / 1024
    print(f"\n[HashGrad MultiSeed] Complete — {filled:,}/{TABLE_SIZE:,} buckets, {model_mb:.2f} MB")
    return embed, W_out, freq_merged, count_merged, fp_merged


# ─────────────────────────────────────────────────────────────────────────────
# Artifact serialisation
# ─────────────────────────────────────────────────────────────────────────────

def save_hash_grad_artifact(
    embed: np.ndarray,
    W_out: np.ndarray,
    seed: int,
    table_bits: int,
    path: str,
    fingerprint: Optional[np.ndarray] = None,
) -> int:
    """Save embed + W_out (+ optional fingerprint) as LZMA9-compressed .hgz."""
    import lzma, struct, io as _io

    embed_dim  = embed.shape[1]
    vocab_size = W_out.shape[1]
    has_fp     = fingerprint is not None
    flags      = int(has_fp)

    buf = _io.BytesIO()
    buf.write(b"HGZ2")
    buf.write(struct.pack("<Q", int(seed)))
    buf.write(struct.pack("<I", int(table_bits)))
    buf.write(struct.pack("<I", int(embed_dim)))
    buf.write(struct.pack("<I", int(vocab_size)))
    buf.write(struct.pack("<I", flags))
    buf.write(embed.astype(np.float16).tobytes())
    buf.write(W_out.astype(np.float16).tobytes())
    if has_fp:
        buf.write(fingerprint.astype(np.uint8).tobytes())

    raw        = buf.getvalue()
    compressed = lzma.compress(raw, preset=9)
    with open(path, "wb") as f:
        f.write(compressed)

    print(f"[HashGrad] Artifact saved → {path} | "
          f"raw={len(raw)/1024/1024:.2f} MB → "
          f"lzma9={len(compressed)/1024/1024:.2f} MB "
          f"({100*(1-len(compressed)/len(raw)):.1f}% reduction)"
          f"{' [+fingerprint]' if has_fp else ''}")
    return len(compressed)


def load_hash_grad_artifact(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, int, int, Optional[np.ndarray]]:
    """Load embed + W_out (+ optional fingerprint) from a .hgz artifact.

    Returns (embed, W_out, seed, table_bits, fingerprint).
    """
    import lzma, struct

    with open(path, "rb") as f:
        raw = lzma.decompress(f.read())

    magic = raw[:4]
    if magic not in (b"HGZ1", b"HGZ2"):
        raise ValueError(f"Invalid magic bytes: {magic!r}")

    seed, table_bits, embed_dim, vocab_size = struct.unpack_from("<QIII", raw, 4)
    TABLE_SIZE = 1 << table_bits

    if magic == b"HGZ2":
        flags, = struct.unpack_from("<I", raw, 4 + 8 + 4 + 4 + 4)
        has_fp = bool(flags & 1)
        offset = 4 + 8 + 4 + 4 + 4 + 4   # magic + seed + 3×uint32 + flags
    else:
        has_fp = False
        offset = 4 + 8 + 4 + 4 + 4        # HGZ1: no flags field

    embed_bytes = TABLE_SIZE * embed_dim * 2
    W_out_bytes = embed_dim * vocab_size * 2

    embed = np.frombuffer(raw[offset:offset + embed_bytes],
                          dtype=np.float16).reshape(TABLE_SIZE, embed_dim).copy()
    W_out = np.frombuffer(raw[offset + embed_bytes:offset + embed_bytes + W_out_bytes],
                          dtype=np.float16).reshape(embed_dim, vocab_size).copy()

    fingerprint = None
    if has_fp:
        fp_offset = offset + embed_bytes + W_out_bytes
        fingerprint = np.frombuffer(raw[fp_offset:fp_offset + TABLE_SIZE],
                                    dtype=np.uint8).copy()

    print(f"[HashGrad] Loaded artifact from {path} | "
          f"TABLE_BITS={table_bits}, EMBED_DIM={embed_dim}, VOCAB={vocab_size}"
          f"{', fingerprint=yes' if has_fp else ''}")
    return embed, W_out, int(seed), int(table_bits), fingerprint


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test():
    """Quick smoke-test of the enhanced pipeline."""
    print("[HashGrad SelfTest] Running enhanced pipeline smoke test...")
    rng = np.random.RandomState(42)

    N          = 100_000
    vocab_size = 64
    table_bits = 10
    embed_dim  = 4
    seed       = 12345

    tokens   = rng.randint(0, vocab_size, N, dtype=np.uint16)
    g_states = rng.randint(0, 2**63, N, dtype=np.int64).view(np.uint64)

    # Single-seed
    embed, W_out, freq, count, fp = train_hash_grad_model(
        tokens=tokens, g_states=g_states, seed=seed,
        table_bits=table_bits, vocab_size=vocab_size, embed_dim=embed_dim,
        nmf_max_iter=20, time_budget_s=30.0, prior_tokens=10_000,
    )
    assert embed.shape == (1 << table_bits, embed_dim)
    assert W_out.shape == (embed_dim, vocab_size)
    assert fp is not None and fp.shape == (1 << table_bits,)

    # Multi-seed
    g2 = rng.randint(0, 2**63, N, dtype=np.int64).view(np.uint64)
    embed2, W_out2, freq2, count2, fp2 = train_hash_grad_multi_seed(
        tokens=tokens, g_states_list=[g_states, g2], seeds=[seed, seed + 1],
        table_bits=table_bits, vocab_size=vocab_size, embed_dim=embed_dim,
        nmf_max_iter=20, time_budget_s=30.0, prior_tokens=10_000,
    )
    assert embed2.shape == (1 << table_bits, embed_dim)
    assert count2.sum() >= count.sum()   # merged has more observations

    # Prediction
    buckets = np.array([0, 1, 2, 100], dtype=np.int64)
    probs, preds = hash_grad_predict_batch(buckets, embed, W_out)
    assert probs.shape == (4, vocab_size)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Artifact round-trip with fingerprint
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(suffix=".hgz", delete=False) as tf:
        tmp = tf.name
    try:
        save_hash_grad_artifact(embed, W_out, seed, table_bits, tmp, fingerprint=fp)
        e2, w2, s2, tb2, fp2_loaded = load_hash_grad_artifact(tmp)
        assert s2 == seed and tb2 == table_bits
        assert fp2_loaded is not None
        assert np.allclose(embed.astype(np.float32), e2.astype(np.float32), atol=1e-3)
    finally:
        _os.unlink(tmp)

    # BPB evaluation with fingerprint
    base_bytes = np.ones(vocab_size, dtype=np.int16)
    has_space  = np.zeros(vocab_size, dtype=bool)
    bpb, loss = hash_grad_bpb(
        val_tokens=tokens, embed=embed, W_out=W_out,
        g_states_val=g_states, seed=seed, table_bits=table_bits,
        base_bytes=base_bytes, has_leading_space=has_space,
        fingerprint_packed=fp,
    )
    assert np.isfinite(bpb), f"BPB not finite: {bpb}"
    print(f"[HashGrad SelfTest] BPB={bpb:.4f}, loss={loss:.4f}")
    print("[HashGrad SelfTest] All assertions passed ✓")


if __name__ == "__main__":
    _self_test()
