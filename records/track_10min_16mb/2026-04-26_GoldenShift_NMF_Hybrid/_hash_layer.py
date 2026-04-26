"""Hash-addressed NMF layer for GoldenShift_NMF_Hybrid.

Ports the core NMF pipeline from 2026-04-07_HDC_1_Step_Grad_DSV (Phases 0–5, 9)
and exposes a single convenience entry point `build_nmf_table()`.

Phases:
  0  — Frozen prior (2M tokens) for sparse-bucket regularisation
  2  — GPU bucket frequency tabulation + fingerprint (distributed)
  3  — Multi-seed frequency merge (optional, default 1 seed)
  4  — XOR orbit regularisation (smooth sparse buckets)
  5  — NMF KL fit (1-iter AdaGrad, GPU)
  9  — Embed pruning (zero unfilled buckets)

Public API:
  precompute_g_states(tokens)          → (N,) uint64  rolling hash
  build_nmf_table(tokens, g_states, ..) → (embed, W_out, fingerprint)
  build_frozen_prior(...)               → (prior_freq, prior_count)
  tabulate_bucket_frequencies_distributed(...)
  xor_orbit_regularise(...)
  nmf_kl_fit(...)

Budget identity:
  TABLE_SIZE × EMBED_DIM × 2 bytes = artifact NMF budget  (fp16)
  Example: TABLE_BITS=18, EMBED_DIM=16: 256K × 16 × 2 = 8 MB (uncompressed)
  Combined with N_WORDS=1024 DSV sem_fwd (~3.4 MB compressed), total ≈ 11 MB.
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _dist_rank() -> int:
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_rank()
    except Exception:
        pass
    return 0

def _dist_world_size() -> int:
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_world_size()
    except Exception:
        pass
    return 1

def _dist_is_main() -> bool:
    return _dist_rank() == 0

def _dist_barrier() -> None:
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _dist.barrier()
    except Exception:
        pass

def _dist_all_reduce_sum_numpy(arr: np.ndarray) -> np.ndarray:
    try:
        import torch
        import torch.distributed as _dist
        if not (_dist.is_available() and _dist.is_initialized()):
            return arr
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        t = torch.from_numpy(arr.copy()).to(device)
        _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
        return t.cpu().numpy()
    except Exception:
        return arr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PHI64  = np.uint64(0x9E3779B97F4A7C15)
FMIX64  = np.uint64(0x9E3779B97F4A7C15)


# ---------------------------------------------------------------------------
# G[p] rolling hash (seed-independent)
# ---------------------------------------------------------------------------

def precompute_g_states(tokens: np.ndarray) -> np.ndarray:
    """Compute the rolling XOR hash G[p] for every position p.

    G[p] encodes tokens[0 .. p-1].  Seed is applied separately at tabulation
    time via ``(G[p] XOR seed) * FMIX64``, so g_states can be reused across
    seeds.

    G[0] = 0
    G[p+1] = G[p] XOR (tokens[p] * KEY[p])
    KEY[p] = ((p+1) * PHI64) ^ ((p+1) >> 32)  |  1

    Args:
        tokens : (N,) uint16 / int32 token sequence

    Returns:
        (N,) uint64 g_states array
    """
    N = len(tokens)
    t = tokens.astype(np.uint64)
    positions = np.arange(N, dtype=np.uint64) + np.uint64(1)
    with np.errstate(over='ignore'):
        keys = positions * _PHI64
    keys ^= (keys >> np.uint64(32))
    keys |= np.uint64(1)
    with np.errstate(over='ignore'):
        contribs = t * keys
    cumxor = np.bitwise_xor.accumulate(contribs)
    g_states = np.empty(N, dtype=np.uint64)
    g_states[0] = np.uint64(0)
    if N > 1:
        g_states[1:] = cumxor[:-1]
    return g_states


def precompute_golden_gram_states(
    tokens: np.ndarray,
    lag_depth: int = 8,
) -> np.ndarray:
    """Sliding-window GoldenAxisShift n-gram hash (translational invariant).

    G_rel[p] = XOR_{k=1}^{lag_depth}  tok[p-k] * GoldenKey[k]

    Unlike the absolute rolling hash, this is TRANSLATIONAL INVARIANT:
    the SAME n-gram at DIFFERENT positions in DIFFERENT documents gets the
    SAME hash value.  This makes NMF buckets correspond to specific n-gram
    contexts rather than absolute document positions.

    GoldenKey[k] uses the same PHI64 irrational constant as GoldenAxisShift:
      GoldenKey[k] = (k * PHI64) ^ ((k * PHI64) >> 32) | 1
    Consecutive lags use irrational multiples of PHI64 → geometric orthogonality
    in 64-bit hash space (same principle as GoldenAxisShift HV rotations).

    Expected BPB improvement (English text, 1024-vocab, full 8B training tokens):
      lag_depth=1 (bigram): ~3.2 BPB
      lag_depth=4 (4-gram): ~2.0 BPB
      lag_depth=8 (8-gram): ~1.0–1.5 BPB

    Args:
        tokens    : (N,) uint16/int32 token sequence
        lag_depth : int  context window depth (default 8)

    Returns:
        (N,) uint64 g_states array  (same API as precompute_g_states)
    """
    N = len(tokens)
    tok_u64 = tokens.astype(np.uint64)
    g_states = np.zeros(N, dtype=np.uint64)

    for k in range(1, lag_depth + 1):
        k_u64 = np.uint64(k)
        with np.errstate(over='ignore'):
            key = k_u64 * _PHI64
        key ^= (key >> np.uint64(32))
        key |= np.uint64(1)
        # tok[p-k] * key contributes to g_states[p] for all p >= k
        if k < N:
            with np.errstate(over='ignore'):
                contrib = tok_u64[:N - k] * key   # (N-k,)
            g_states[k:] ^= contrib

    return g_states


def precompute_circular_golden_gram_states(
    tokens: np.ndarray,
    lag_depth: int = 8,
) -> np.ndarray:
    """Circular-rotation GoldenGram hash — DSV-aligned cross-axis geometry.

    G_cross[p] = XOR_{k=1}^{lag_depth}  CircularRotate64(tok[p-k] × PHI64, k × phi_offset)

    where phi_offset = 39 = round(φ × 64), the same rotation step used by
    GoldenAxisShift DSV hypervectors.

    Structural alignment with GoldenAxisShift DSV:
      DSV lag-k:   GoldenAxisShift rotates base HV by  (k × 39) % n_bits  bits
      Hash lag-k:  CircularRotate64(tok × PHI64,       (k × 39) % 64)

    Both operations apply the same phi-spaced circular rotation in their
    respective bit-width domains (DSV: n_bits, Hash: 64 bits).  This ensures:
      - Hash metric space = DSV embedding metric space (same circular geometry)
      - NMF Tier 1 and DSV Tier 2 share a unified geometric embedding space
      - Fallback DSV predictions are geometrically coherent with NMF predictions

    vs. standard GoldenGram (multiplication-based key mixing):
      GoldenGram:  G[p] = XOR_k  tok[p-k] × GoldenKey[k]    ← Weyl via multiply
      CircGram:    G[p] = XOR_k  rot64(tok[p-k] × PHI64, k*39) ← Weyl + rotation struct

    Both achieve Weyl-equidistributed lag separation.  CircGram additionally
    ensures the hash rotation geometry mirrors the DSV lag-subspace geometry.

    phi_offset = round(0.6180339887 × 64) = round(39.555) = 39

    Args:
        tokens    : (N,) uint16/int32 token sequence
        lag_depth : int  context window depth (default 8)

    Returns:
        (N,) uint64 g_states array  (same API as precompute_golden_gram_states)
    """
    N = len(tokens)
    tok_u64 = tokens.astype(np.uint64)
    g_states = np.zeros(N, dtype=np.uint64)

    # phi_offset = round(phi_frac × 64) = 39  (same as GoldenAxisShift uses)
    PHI_OFFSET = 39

    for k in range(1, lag_depth + 1):
        rot = (k * PHI_OFFSET) % 64          # circular left-rotation amount
        rot_u64     = np.uint64(rot)
        inv_rot_u64 = np.uint64(64 - rot) if rot > 0 else np.uint64(0)

        if k < N:
            with np.errstate(over='ignore'):
                base = tok_u64[:N - k] * _PHI64   # (N-k,) uint64: Weyl embedding
            if rot > 0:
                # 64-bit left circular rotate: (x << rot) | (x >> (64-rot))
                with np.errstate(over='ignore'):
                    contrib = (base << rot_u64) | (base >> inv_rot_u64)
            else:
                contrib = base
            g_states[k:] ^= contrib

    return g_states


# ---------------------------------------------------------------------------
# Frequency tabulation — GPU path
# ---------------------------------------------------------------------------

def tabulate_bucket_frequencies_gpu(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    chunk_size: int = 50_000_000,
    label: str = "FreqTab",
    build_fingerprint: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Tabulate per-bucket next-token frequencies using GPU scatter_add_.

    Processes tokens in chunks to avoid VRAM exhaustion.  Falls back to
    CPU if GPU memory is insufficient (raises RuntimeError → caller catches).
    """
    import torch

    TABLE_SIZE = 1 << table_bits
    SHIFT      = 64 - table_bits
    FP_SHIFT   = 64 - table_bits - 8
    TABLE_MASK = (1 << table_bits) - 1
    dev        = torch.device("cuda")
    N          = len(tokens)

    torch.cuda.empty_cache()

    free_vram, _ = torch.cuda.mem_get_info(0)
    freq_bytes   = TABLE_SIZE * vocab_size * 8   # int64
    needed       = freq_bytes + (N - 1) * 16     # g_states + tokens
    if needed > free_vram * 0.80:
        raise RuntimeError(
            f"GPU VRAM too small ({needed/1e9:.1f} GB needed, "
            f"{free_vram/1e9:.1f} GB free) — using CPU path"
        )

    _fmix_i64 = int(np.array([int(FMIX64)], dtype=np.uint64).view(np.int64)[0])
    seed_i64   = int(np.array([seed], dtype=np.uint64).view(np.int64)[0])
    FMIX_t    = torch.tensor(_fmix_i64,  dtype=torch.int64, device=dev)
    seed_t    = torch.tensor(seed_i64,   dtype=torch.int64, device=dev)
    MASK_t    = torch.tensor(TABLE_MASK, dtype=torch.int64, device=dev)
    FP_MSK_T  = torch.tensor(0xFF,       dtype=torch.int64, device=dev)

    freq_flat = torch.zeros(TABLE_SIZE * vocab_size, dtype=torch.int64, device=dev)
    fp_gpu    = torch.zeros(TABLE_SIZE, dtype=torch.int64, device=dev) if build_fingerprint else None

    t_up = time.time()
    g_gpu   = torch.as_tensor(g_states[:N - 1].view(np.int64), dtype=torch.int64, device=dev)
    tok_gpu = torch.as_tensor(tokens[1:N].astype(np.int64),    dtype=torch.int64, device=dev)
    torch.cuda.synchronize()
    print(f"[NMF {label} GPU] Uploaded {(N-1)*16/1e9:.2f} GB in {time.time()-t_up:.2f}s "
          f"— processing {N-1:,} tokens…", flush=True)

    ones_buf  = torch.ones(min(chunk_size, N), dtype=torch.int64, device=dev)
    t0        = time.time()
    processed = 0

    for chunk_start in range(0, N - 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N - 1)
        C = chunk_end - chunk_start

        g_ch   = g_gpu[chunk_start:chunk_end]
        tok_ch = tok_gpu[chunk_start:chunk_end]

        val     = (g_ch ^ seed_t) * FMIX_t
        buckets = (val >> SHIFT) & MASK_t

        if build_fingerprint and fp_gpu is not None:
            fps = (val >> FP_SHIFT) & FP_MSK_T
            fp_gpu.scatter_(0, buckets, fps)

        pair_keys = buckets * vocab_size + tok_ch
        freq_flat.scatter_add_(0, pair_keys, ones_buf[:C])

        processed += C
        if processed % 100_000_000 == 0 or chunk_end >= N - 1:
            elapsed = time.time() - t0
            rate    = processed / max(elapsed, 1e-6) / 1e6
            print(f"[NMF {label} GPU] {processed:,}/{N-1:,} "
                  f"({100*processed/(N-1):.1f}%) — {rate:.1f}M tok/s", flush=True)

    freq  = freq_flat.cpu().numpy().reshape(TABLE_SIZE, vocab_size).astype(np.uint32)
    count = freq.sum(axis=1).astype(np.uint32)
    fingerprint = fp_gpu.cpu().numpy().astype(np.uint8) if build_fingerprint else None
    filled = int(np.sum(count > 0))
    print(f"[NMF {label} GPU] Done in {time.time()-t0:.1f}s — "
          f"{filled:,}/{TABLE_SIZE:,} buckets filled ({100*filled/TABLE_SIZE:.1f}%)",
          flush=True)

    del freq_flat, g_gpu, tok_gpu, ones_buf
    if fp_gpu is not None:
        del fp_gpu
    torch.cuda.empty_cache()

    return freq, count, fingerprint


def tabulate_bucket_frequencies_cpu(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    chunk_size: int = 2_000_000,
    label: str = "FreqTab",
    build_fingerprint: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """CPU fallback for tabulate_bucket_frequencies."""
    TABLE_SIZE  = 1 << table_bits
    SHIFT       = np.uint64(64 - table_bits)
    FP_SHIFT    = np.uint64(64 - table_bits - 8)
    seed_u64    = np.uint64(seed)
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

        if build_fingerprint and fingerprint is not None:
            fps = ((finalised >> FP_SHIFT) & np.uint64(0xFF)).astype(np.uint8)
            fingerprint[buckets] = fps

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
            print(f"[NMF {label} CPU] {processed:,}/{N-1:,} "
                  f"({100*processed/(N-1):.1f}%) — {rate:.1f}M tok/s", flush=True)

    filled = int(np.sum(count > 0))
    print(f"[NMF {label} CPU] Done in {time.time()-t0:.1f}s — "
          f"{filled:,}/{TABLE_SIZE:,} buckets filled", flush=True)
    return freq, count, fingerprint


def tabulate_bucket_frequencies(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    chunk_size: int = 50_000_000,
    label: str = "FreqTab",
    build_fingerprint: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Route to GPU or CPU tabulation."""
    try:
        import torch
        if torch.cuda.is_available():
            return tabulate_bucket_frequencies_gpu(
                tokens, g_states, seed, table_bits, vocab_size,
                chunk_size=chunk_size, label=label,
                build_fingerprint=build_fingerprint,
            )
    except Exception as _e:
        print(f"[NMF {label}] GPU path error ({_e!r}) — falling back to CPU", flush=True)
    return tabulate_bucket_frequencies_cpu(
        tokens, g_states, seed, table_bits, vocab_size,
        chunk_size=min(chunk_size, 2_000_000), label=label,
        build_fingerprint=build_fingerprint,
    )


def tabulate_bucket_frequencies_distributed(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    chunk_size: int = 50_000_000,
    label: str = "FreqTab",
    build_fingerprint: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Distributed frequency tabulation: each rank processes its shard,
    then all-reduces (SUM) to produce globally merged freq/count/fingerprint.
    """
    rank       = _dist_rank()
    world_size = _dist_world_size()

    if world_size == 1:
        return tabulate_bucket_frequencies(
            tokens, g_states, seed, table_bits, vocab_size,
            chunk_size=chunk_size, label=label,
            build_fingerprint=build_fingerprint,
        )

    N           = len(tokens)
    shard_size  = (N + world_size - 1) // world_size
    shard_start = rank * shard_size
    shard_end   = min(shard_start + shard_size, N)

    TABLE_SIZE = 1 << table_bits

    if shard_start >= N - 1:
        freq_local  = np.zeros((TABLE_SIZE, vocab_size), dtype=np.uint32)
        count_local = np.zeros(TABLE_SIZE, dtype=np.uint32)
        fp_local    = np.zeros(TABLE_SIZE, dtype=np.uint8) if build_fingerprint else None
    else:
        shard_end_ext = min(shard_end + 1, N)
        tok_shard = tokens[shard_start:shard_end_ext]
        g_shard   = g_states[shard_start:shard_end_ext]
        print(f"[NMF {label} rank={rank}] Shard [{shard_start:,}, {shard_end_ext:,})", flush=True)
        freq_local, count_local, fp_local = tabulate_bucket_frequencies(
            tok_shard, g_shard, seed, table_bits, vocab_size,
            chunk_size=chunk_size, label=f"{label}_r{rank}",
            build_fingerprint=build_fingerprint,
        )

    _dist_barrier()

    freq_merged  = _dist_all_reduce_sum_numpy(freq_local.astype(np.int64)).astype(np.uint32)
    count_merged = _dist_all_reduce_sum_numpy(count_local.astype(np.int64)).astype(np.uint32)

    # For the fingerprint, take the entry from the rank with the highest count per bucket
    fp_merged = fp_local
    try:
        import torch
        import torch.distributed as _dist_mod
        if build_fingerprint and fp_local is not None and _dist_mod.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device     = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
            fp_t     = torch.from_numpy(fp_local.astype(np.int32)).to(device)
            count_t  = torch.from_numpy(count_local.astype(np.int64)).to(device)
            fp_list    = [torch.zeros_like(fp_t)    for _ in range(_dist_mod.get_world_size())]
            count_list = [torch.zeros_like(count_t) for _ in range(_dist_mod.get_world_size())]
            _dist_mod.all_gather(fp_list,    fp_t)
            _dist_mod.all_gather(count_list, count_t)
            fp_merged  = fp_list[0].cpu().numpy().astype(np.uint8)
            best_count = count_list[0].cpu().numpy()
            for i in range(1, len(fp_list)):
                c_i = count_list[i].cpu().numpy()
                better = c_i > best_count
                fp_merged[better]  = fp_list[i].cpu().numpy().astype(np.uint8)[better]
                best_count[better] = c_i[better]
    except Exception as _e:
        if _dist_is_main():
            print(f"[NMF {label}] Fingerprint all-gather failed ({_e}) — using rank-0 fp", flush=True)

    filled = int(np.sum(count_merged > 0))
    if _dist_is_main():
        print(f"[NMF {label} dist] All-reduce done — "
              f"{filled:,}/{TABLE_SIZE:,} buckets filled across {world_size} ranks", flush=True)

    return freq_merged, count_merged, fp_merged


# ---------------------------------------------------------------------------
# Phase 0: Frozen prior
# ---------------------------------------------------------------------------

def build_frozen_prior(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    prior_tokens: int = 2_000_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an uncontaminated prior from the first `prior_tokens` training tokens.

    Returns (prior_freq, prior_count) where prior_freq[b] is the normalised
    next-token distribution for bucket b computed from the first 2M tokens.
    Used in Phase 5 NMF to regularise sparse buckets.
    """
    TABLE_SIZE = 1 << table_bits
    n_prior    = min(prior_tokens, len(tokens) - 1)
    print(f"[NMF Prior] Building frozen prior from first {n_prior:,} tokens…", flush=True)

    freq, count, _ = tabulate_bucket_frequencies(
        tokens[:n_prior + 1], g_states[:n_prior + 1],
        seed, table_bits, vocab_size,
        build_fingerprint=False, label="Prior",
    )

    prior_freq = np.zeros((TABLE_SIZE, vocab_size), dtype=np.float32)
    active = count > 0
    prior_freq[active]  = freq[active].astype(np.float32) / count[active, None]
    prior_freq[~active] = 1.0 / vocab_size

    print(f"[NMF Prior] Done — {int(active.sum()):,}/{TABLE_SIZE:,} prior buckets filled",
          flush=True)
    return prior_freq, count


# ---------------------------------------------------------------------------
# Phase 4: XOR orbit regularisation
# ---------------------------------------------------------------------------

def xor_orbit_regularise(
    freq: np.ndarray,
    count: np.ndarray,
    table_bits: int,
    alpha: float = 0.10,
    min_count_threshold: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend sparse buckets toward XOR-adjacent richer neighbours.

    For each bucket with count < `min_count_threshold`, look at all
    single-bit-flip neighbours.  If a neighbour has a higher count, blend
    freq toward that neighbour with weight `alpha`.  This smooths the
    gradient target seen by the NMF without touching well-filled buckets.
    """
    TABLE_SIZE  = 1 << table_bits
    sparse_mask = (count > 0) & (count < min_count_threshold)
    n_sparse    = int(sparse_mask.sum())

    if n_sparse == 0:
        print(f"[NMF XORReg] No sparse buckets (count < {min_count_threshold}) — skipping",
              flush=True)
        return freq.astype(np.float32), count.astype(np.float32)

    print(f"[NMF XORReg] Regularising {n_sparse:,} sparse buckets "
          f"(alpha={alpha}, min_count={min_count_threshold})…", flush=True)

    freq_f  = freq.astype(np.float32)
    count_f = count.astype(np.float32)
    sparse_idx = np.where(sparse_mask)[0]

    for bit in range(min(table_bits, 16)):
        neighbours  = sparse_idx ^ (1 << bit)
        neighbours  = np.clip(neighbours, 0, TABLE_SIZE - 1)
        nbr_counts  = count_f[neighbours]
        borrow_mask = nbr_counts > count_f[sparse_idx]
        if not borrow_mask.any():
            continue
        bi  = sparse_idx[borrow_mask]
        bn  = neighbours[borrow_mask]
        nc  = nbr_counts[borrow_mask]
        sc  = count_f[bi]
        w   = np.clip(alpha * nc / (sc + nc + 1e-8), 0, 0.5)
        freq_f[bi]  = (1 - w[:, None]) * freq_f[bi] + w[:, None] * freq_f[bn]
        count_f[bi] += w * nc

    after_avg = count_f[sparse_mask].mean() if n_sparse > 0 else 0.0
    print(f"[NMF XORReg] Done — avg count for sparse buckets: "
          f"{after_avg:.1f} (was {count[sparse_mask].mean():.1f})", flush=True)
    return freq_f, count_f


# ---------------------------------------------------------------------------
# Phase 5: NMF KL fit (GPU)
# ---------------------------------------------------------------------------

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def nmf_svd_init(
    freq: np.ndarray,
    count: np.ndarray,
    embed_dim: int,
    min_count: int = 1,
    epsilon: float = 1e-9,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-shot gradient teleport: truncated SVD of the log-probability matrix.

    The exact global optimum of  min KL(P ‖ softmax(embed @ W_out))  subject to
    rank(embed @ W_out) ≤ embed_dim  is the rank-k truncated SVD of L = log P:

        L ≈ U_k × diag(Σ_k) × V_k.T
        embed  = U_k × diag(Σ_k)       (TABLE_SIZE × embed_dim)
        W_out  = V_k.T                   (embed_dim × vocab_size)
        softmax(embed @ W_out) ≈ P      at the optimum

    This jumps directly to the converged solution in one SVD pass —
    no random initialisation, no gradient steps, no learning-rate tuning.
    For a deterministic system (same freq table → same SVD) this always gives
    the globally-optimal low-rank embedding.

    Complexity: O(n_active × vocab_size × embed_dim)  — dominated by SVD.
    Handles tables with TABLE_SIZE up to ~256K on GPU (torch.linalg.svd).
    Falls back to randomised CPU SVD for larger tables.

    Args:
        freq      : (TABLE_SIZE, vocab_size) uint32  raw co-occurrence counts
        count     : (TABLE_SIZE,) uint32             total counts per bucket
        embed_dim : int  rank k
        min_count : int  skip buckets with fewer observations
        epsilon   : float  smoothing added to P before log
        verbose   : bool

    Returns:
        embed  : (TABLE_SIZE, embed_dim) float16  globally optimal embedding
        W_out  : (embed_dim, vocab_size) float16  globally optimal projection
    """
    TABLE_SIZE, vocab_size = freq.shape
    t0 = time.time()

    count_f     = count.astype(np.float32)
    active_mask = count_f >= min_count
    active_idx  = np.where(active_mask)[0].astype(np.int32)
    n_active    = len(active_idx)

    if verbose:
        print(f"[NMF SVD] Gradient teleport  n_active={n_active:,}/{TABLE_SIZE:,}  "
              f"embed_dim={embed_dim}", flush=True)

    if n_active == 0:
        return (np.zeros((TABLE_SIZE, embed_dim), dtype=np.float16),
                np.zeros((embed_dim, vocab_size), dtype=np.float16))

    # Step 1: empirical distributions P and their log-probs
    freq_a  = freq[active_idx].astype(np.float64)
    count_a = count_f[active_idx].astype(np.float64)
    P       = freq_a / (count_a[:, None] + epsilon)
    P      /= np.maximum(P.sum(axis=1, keepdims=True), epsilon)    # row-normalise
    P       = np.maximum(P, epsilon)                                # avoid log(0)

    L           = np.log(P)                         # (n, V) log-probs
    L_centred   = L - L.mean(axis=1, keepdims=True) # shift-invariant centering

    # Step 2: truncated SVD  → embed = U_k × Σ_k,  W_out = V_k.T
    k = min(embed_dim, n_active, vocab_size)

    # For small tables (n ≤ 64K): exact GPU SVD via torch.linalg.svd.
    # For large tables (n > 64K): randomised GPU SVD using power iteration
    # (O(n×V×k×4) vs O(n×V²) for exact, handles TABLE_BITS=18-20).
    _EXACT_THRESH = 65_536

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA")

        if n_active <= _EXACT_THRESH:
            L_t = torch.tensor(L_centred.astype(np.float32), device="cuda")
            U_t, S_t, Vh_t = torch.linalg.svd(L_t, full_matrices=False)
            U_k  = U_t[:, :k].cpu().numpy().astype(np.float32)
            S_k  = S_t[:k].cpu().numpy().astype(np.float32)
            Vh_k = Vh_t[:k, :].cpu().numpy().astype(np.float32)
            del L_t, U_t, S_t, Vh_t
            torch.cuda.empty_cache()
        else:
            # Randomised SVD (n >> k): sketch L via random projections, then
            # do exact SVD on the small (k+oversamp, V) sketched matrix.
            if verbose:
                print(f"[NMF SVD] Large table ({n_active:,}): randomised SVD...",
                      flush=True)
            L_t = torch.tensor(L_centred.astype(np.float32), device="cuda")
            torch.manual_seed(42)
            oversamp = 10
            Omega = torch.randn(vocab_size, k + oversamp, device="cuda")
            Y = L_t @ Omega                               # (n, k+oversamp)
            for _ in range(4):                            # power iterations
                Y, _ = torch.linalg.qr(Y)
                Z, _ = torch.linalg.qr(L_t.T @ Y)
                Y = L_t @ Z
            Q, _ = torch.linalg.qr(Y)                    # (n, k+oversamp) ortho
            B    = Q.T @ L_t                              # (k+oversamp, V)
            U_b, S_t, Vh_t = torch.linalg.svd(B, full_matrices=False)
            U_full = Q @ U_b                              # (n, k+oversamp)
            U_k  = U_full[:, :k].cpu().numpy().astype(np.float32)
            S_k  = S_t[:k].cpu().numpy().astype(np.float32)
            Vh_k = Vh_t[:k, :].cpu().numpy().astype(np.float32)
            del L_t, Omega, Y, Q, B, U_b, S_t, Vh_t, U_full
            torch.cuda.empty_cache()

        if verbose:
            print(f"[NMF SVD] SVD done in {time.time()-t0:.2f}s  "
                  f"S[0]={S_k[0]:.3f}  S[-1]={S_k[-1]:.3f}", flush=True)

    except Exception as _e:
        if verbose:
            print(f"[NMF SVD] GPU SVD failed ({_e}) — using sklearn CPU randomised SVD",
                  flush=True)
        try:
            from sklearn.utils.extmath import randomized_svd
            U_k, S_k, Vh_k = randomized_svd(
                L_centred.astype(np.float32), n_components=k,
                n_iter=4, random_state=42,
            )
        except ImportError:
            U_f, S_f, Vh_f = np.linalg.svd(L_centred.astype(np.float32),
                                             full_matrices=False)
            U_k, S_k, Vh_k = U_f[:, :k], S_f[:k], Vh_f[:k]

    embed_active = (U_k * S_k[None, :]).astype(np.float32)  # (n_active, k)
    W_out_k      = Vh_k.astype(np.float32)                   # (k, vocab_size)

    # Quick quality check: KL on 100-bucket sample
    if verbose:
        ns = min(200, n_active)
        logits  = embed_active[:ns] @ W_out_k            # (ns, V)
        lmax    = logits.max(axis=1, keepdims=True)
        q       = np.exp(logits - lmax)
        q      /= q.sum(axis=1, keepdims=True)
        kl_svd  = float(-(P[:ns] * np.log(q + 1e-30)).sum(1).mean())
        kl_base = float(-np.log(1.0 / vocab_size))       # uniform baseline
        print(f"[NMF SVD] KL_svd={kl_svd:.4f}  KL_uniform={kl_base:.4f}  "
              f"improvement={kl_base-kl_svd:.4f}  elapsed={time.time()-t0:.2f}s",
              flush=True)

    # Pack into full-size tables
    embed_full = np.zeros((TABLE_SIZE, embed_dim), dtype=np.float32)
    embed_full[active_idx, :k] = embed_active
    W_out_full = np.zeros((embed_dim, vocab_size), dtype=np.float32)
    W_out_full[:k] = W_out_k

    return embed_full.astype(np.float16), W_out_full.astype(np.float16)


def nmf_kl_fit(
    freq: np.ndarray,
    count: np.ndarray,
    embed_dim: int,
    max_iter: int = 1,
    lr_embed: float = 0.05,
    lr_w_out: float = 0.02,
    min_count: int = 1,
    seed: int = 0,
    time_budget_s: float = 60.0,
    log_every: int = 1,
    prior_freq: Optional[np.ndarray] = None,
    prior_weight: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Phase 5: 1-iteration NMF KL fit.

    Fits embed (TABLE_SIZE × embed_dim fp16) and W_out (embed_dim × vocab_size fp16)
    to approximate the per-bucket next-token distribution:
        P[bucket] ≈ softmax(embed[bucket] @ W_out)

    At max_iter=1 the loss stays near ln(vocab_size) — the NMF is effectively a
    single-step normalisation of the frequency table.  The GoldenShift DSV
    (Phase 6) carries the primary predictive load; this is a lightweight
    secondary signal for frequency-matched buckets.

    Returns:
        embed  : (TABLE_SIZE, embed_dim) float16
        W_out  : (embed_dim, vocab_size) float16
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return _nmf_kl_fit_gpu(
                freq=freq, count=count, embed_dim=embed_dim,
                max_iter=max_iter, lr_embed=lr_embed, lr_w_out=lr_w_out,
                min_count=min_count, seed=seed, time_budget_s=time_budget_s,
                log_every=log_every, prior_freq=prior_freq,
                prior_weight=prior_weight,
            )
    except Exception as _e:
        print(f"[NMF KL] GPU error ({_e!r}) — using CPU", flush=True)

    # CPU fallback
    TABLE_SIZE, vocab_size = freq.shape
    rng = np.random.RandomState(seed)
    t0  = time.time()

    count_f     = count.astype(np.float32)
    active_mask = count_f >= min_count
    active_idx  = np.where(active_mask)[0].astype(np.int32)
    n_active    = len(active_idx)
    print(f"[NMF KL CPU] Active buckets: {n_active:,}/{TABLE_SIZE:,}", flush=True)

    if n_active == 0:
        return (np.zeros((TABLE_SIZE, embed_dim), dtype=np.float16),
                np.zeros((embed_dim, vocab_size), dtype=np.float16))

    freq_active  = freq[active_idx].astype(np.float32)
    count_active = count_f[active_idx]
    P            = freq_active / (count_active[:, None] + 1e-8)

    if prior_freq is not None:
        sparse_in_active = count_active < 10
        if sparse_in_active.any():
            prior_active = prior_freq[active_idx[sparse_in_active]]
            w = np.clip(prior_weight * 10.0 / (count_active[sparse_in_active] + 1e-8), 0, 0.5)
            P[sparse_in_active] = (1 - w[:, None]) * P[sparse_in_active] + w[:, None] * prior_active

    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-8)
    weights = count_active / count_active.sum()

    embed_active = rng.randn(n_active, embed_dim).astype(np.float32) * 0.01
    W_out        = rng.randn(embed_dim, vocab_size).astype(np.float32) * 0.01
    sq_embed     = np.ones((n_active, embed_dim), dtype=np.float32) * 1e-8
    sq_W_out     = np.ones((embed_dim, vocab_size), dtype=np.float32) * 1e-8
    eps = 1e-8

    for it in range(max_iter):
        if time.time() - t0 > time_budget_s:
            break
        logits  = embed_active @ W_out
        q       = _softmax_np(logits)
        dL_dl   = (q - P) * weights[:, None]
        dL_dW   = embed_active.T @ dL_dl
        dL_de   = dL_dl @ W_out.T
        sq_embed  += dL_de  ** 2
        embed_active -= lr_embed * dL_de  / (np.sqrt(sq_embed)  + eps)
        sq_W_out  += dL_dW  ** 2
        W_out     -= lr_w_out * dL_dW / (np.sqrt(sq_W_out) + eps)
        if it % log_every == 0:
            kl = float(-(P * (logits - np.log(np.exp(logits).sum(1, keepdims=True) + 1e-30))).sum(1).dot(weights))
            print(f"[NMF KL CPU] iter {it}/{max_iter}  KL={kl:.4f}  t={time.time()-t0:.1f}s",
                  flush=True)

    embed_full = np.zeros((TABLE_SIZE, embed_dim), dtype=np.float32)
    embed_full[active_idx] = embed_active
    return embed_full.astype(np.float16), W_out.astype(np.float16)


def _nmf_kl_fit_gpu(
    freq: np.ndarray,
    count: np.ndarray,
    embed_dim: int,
    max_iter: int = 1,
    lr_embed: float = 0.05,
    lr_w_out: float = 0.02,
    min_count: int = 1,
    seed: int = 0,
    time_budget_s: float = 60.0,
    log_every: int = 1,
    prior_freq: Optional[np.ndarray] = None,
    prior_weight: float = 0.05,
    converge_tol: float = 1e-6,
    converge_patience: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    dev = torch.device("cuda")

    TABLE_SIZE, vocab_size = freq.shape
    rng = np.random.RandomState(seed)
    t0  = time.time()

    count_f     = count.astype(np.float32)
    active_mask = count_f >= min_count
    active_idx  = np.where(active_mask)[0].astype(np.int32)
    n_active    = len(active_idx)
    print(f"[NMF KL GPU] Active buckets: {n_active:,}/{TABLE_SIZE:,}", flush=True)

    if n_active == 0:
        return (np.zeros((TABLE_SIZE, embed_dim), dtype=np.float16),
                np.zeros((embed_dim, vocab_size), dtype=np.float16))

    freq_active  = freq[active_idx].astype(np.float32)
    count_active = count_f[active_idx]
    P            = freq_active / (count_active[:, None] + 1e-8)

    if prior_freq is not None:
        sparse_in_active = count_active < 10
        if sparse_in_active.any():
            prior_active = prior_freq[active_idx[sparse_in_active]]
            w = np.clip(prior_weight * 10.0 / (count_active[sparse_in_active] + 1e-8), 0, 0.5)
            P[sparse_in_active] = (1 - w[:, None]) * P[sparse_in_active] + w[:, None] * prior_active

    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-8)
    weights = count_active / count_active.sum()

    P_t       = torch.tensor(P,       dtype=torch.float32, device=dev)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=dev)
    embed_t   = torch.tensor(rng.randn(n_active, embed_dim).astype(np.float32) * 0.01, device=dev)
    W_out_t   = torch.tensor(rng.randn(embed_dim, vocab_size).astype(np.float32) * 0.01, device=dev)
    sq_emb    = torch.full_like(embed_t, 1e-8)
    sq_wout   = torch.full_like(W_out_t, 1e-8)

    eps            = 1e-8
    best_loss      = float("inf")
    best_embed     = embed_t.clone()
    best_W_out     = W_out_t.clone()
    patience_count = 0
    prev_loss      = float("inf")

    for it in range(max_iter):
        if time.time() - t0 > time_budget_s:
            print(f"[NMF KL GPU] Time budget reached at iter {it}", flush=True)
            break

        cos_fac = (1.0 + math.cos(math.pi * it / max(max_iter, 1))) * 0.5
        lr_e    = lr_embed * cos_fac
        lr_w    = lr_w_out * cos_fac

        logits  = embed_t @ W_out_t
        lmax    = logits.max(dim=1, keepdim=True).values
        exp_l   = torch.exp(logits - lmax)
        q       = exp_l / (exp_l.sum(dim=1, keepdim=True) + eps)
        dL_dl   = (q - P_t) * weights_t[:, None]
        dL_dW   = embed_t.t() @ dL_dl
        dL_de   = dL_dl @ W_out_t.t()
        sq_emb  += dL_de  ** 2
        embed_t -= lr_e * dL_de  / (sq_emb.sqrt()  + eps)
        sq_wout += dL_dW  ** 2
        W_out_t -= lr_w * dL_dW  / (sq_wout.sqrt() + eps)

        if it % log_every == 0 or it == max_iter - 1:
            with torch.no_grad():
                lg  = embed_t @ W_out_t
                lmx = lg.max(dim=1, keepdim=True).values
                elg = torch.exp(lg - lmx)
                llq = (lg - lmx) - torch.log(elg.sum(dim=1, keepdim=True) + 1e-30)
                kl  = float(-(P_t * llq).sum(dim=1).dot(weights_t))
            print(f"[NMF KL GPU] iter {it:4d}/{max_iter}  KL={kl:.6f}  t={time.time()-t0:.2f}s",
                  flush=True)
            if kl < best_loss:
                best_loss  = kl
                best_embed = embed_t.clone()
                best_W_out = W_out_t.clone()
            rel_imp = abs(prev_loss - kl) / (abs(prev_loss) + 1e-10)
            patience_count = (patience_count + 1) if rel_imp < converge_tol else 0
            if patience_count >= converge_patience:
                print(f"[NMF KL GPU] Early stop at iter {it}", flush=True)
                break
            prev_loss = kl

    print(f"[NMF KL GPU] Done — best KL={best_loss:.6f} in {time.time()-t0:.2f}s", flush=True)
    embed_full = np.zeros((TABLE_SIZE, embed_dim), dtype=np.float32)
    embed_full[active_idx] = best_embed.cpu().numpy()
    return embed_full.astype(np.float16), best_W_out.cpu().numpy().astype(np.float16)


# ---------------------------------------------------------------------------
# Phase 9: Embed pruning
# ---------------------------------------------------------------------------

def prune_embeds(
    embed: np.ndarray,
    count: np.ndarray,
    min_count: int = 1,
) -> np.ndarray:
    """Zero out embed vectors for unfilled buckets (count < min_count)."""
    pruned = embed.copy()
    pruned[count < min_count] = 0.0
    n_pruned = int((count < min_count).sum())
    TABLE_SIZE = len(count)
    print(f"[NMF Prune] Zeroed {n_pruned:,}/{TABLE_SIZE:,} unfilled buckets",
          flush=True)
    return pruned.astype(np.float16)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_nmf_table(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    embed_dim: int,
    vocab_size: int,
    nmf_max_iter: int = 1,
    dist_rank: int = 0,
    dist_world_size: int = 1,
    time_budget_s: float = 30.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full NMF pipeline: Phases 0–5, 9.

    Runs on all distributed ranks (Phase 2 all-reduces), then only rank 0
    performs Phases 4–5 and returns the trained tables.  Non-zero ranks
    return zero-filled arrays.

    Args:
        tokens          : (N,) uint16 training token sequence
        g_states        : (N,) uint64 rolling hash (from precompute_g_states)
        seed            : Hash seed for this NMF run
        table_bits      : log₂ of hash-table size (18 → 256K buckets)
        embed_dim       : NMF embedding dimension per bucket (e.g. 16)
        vocab_size      : Vocabulary size (1024)
        nmf_max_iter    : NMF gradient iterations (default 1 — near-optimal,
                          leaves more time for GoldenAxisShift DSV build)
        dist_rank       : This rank's index in distributed group
        dist_world_size : Total number of ranks
        time_budget_s   : Time budget for NMF fit (default 30s)
        verbose         : Print progress

    Returns:
        embed       : (TABLE_SIZE, embed_dim) float16
        W_out       : (embed_dim, vocab_size) float16
        fingerprint : (TABLE_SIZE,) uint8  (collision detector)
    """
    TABLE_SIZE = 1 << table_bits
    t0 = time.time()

    if verbose and dist_rank == 0:
        print(f"\n[NMF] Building NMF hash table  "
              f"TABLE_BITS={table_bits} TABLE_SIZE={TABLE_SIZE:,}  "
              f"EMBED_DIM={embed_dim}  max_iter={nmf_max_iter}", flush=True)

    # Phase 0: frozen prior (single-process, rank 0 only, fast)
    prior_freq = None
    if dist_rank == 0:
        prior_freq, _ = build_frozen_prior(
            tokens, g_states, seed, table_bits, vocab_size,
        )

    # Phase 2: distributed frequency tabulation + fingerprint
    freq, count, fingerprint = tabulate_bucket_frequencies_distributed(
        tokens, g_states, seed, table_bits, vocab_size,
        label="Phase2",
    )

    # Phases 4–5 run on rank 0 only; other ranks exit after all-reduce
    if dist_rank != 0:
        return (np.zeros((TABLE_SIZE, embed_dim), dtype=np.float16),
                np.zeros((embed_dim, vocab_size), dtype=np.float16),
                fingerprint if fingerprint is not None
                else np.zeros(TABLE_SIZE, dtype=np.uint8))

    # Phase 4: XOR orbit regularisation
    freq_reg, count_reg = xor_orbit_regularise(freq, count, table_bits)

    # Phase 5: NMF — gradient teleport via truncated SVD of log P
    # Jumps directly to the global optimum.  No iterations, no random seed.
    # freq_reg / count_reg are float32 (possibly regularised by XOR orbit smoothing).
    embed, W_out = nmf_svd_init(
        freq      = freq_reg,   # (TABLE_SIZE, vocab_size) float32 (smoothed)
        count     = count_reg,  # (TABLE_SIZE,) float32 (smoothed)
        embed_dim = embed_dim,
        min_count = 1,
        verbose   = verbose,
    )

    # Phase 9: prune unfilled buckets
    embed = prune_embeds(embed, count, min_count=1)

    fp = fingerprint if fingerprint is not None else np.zeros(TABLE_SIZE, dtype=np.uint8)

    if verbose:
        elapsed = time.time() - t0
        print(f"[NMF] Done in {elapsed:.1f}s  "
              f"embed={embed.nbytes/1e6:.1f} MB  "
              f"W_out={W_out.nbytes/1e6:.3f} MB  "
              f"fingerprint={fp.nbytes/1e6:.2f} MB", flush=True)

    return embed, W_out, fp
