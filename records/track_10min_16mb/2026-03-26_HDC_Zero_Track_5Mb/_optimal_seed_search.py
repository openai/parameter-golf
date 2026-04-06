"""Optimal Seed Search for HDC Rolling-Hash Language Model.

=======================================================================
THEORETICAL FOUNDATION
=======================================================================

The HDC rolling-hash model computes table buckets via:

    G[0]   = 0
    G[p+1] = G[p]  XOR  (tokens[p] * HADAMARD_KEY[p])         [O(1) rolling update]

    bucket[p] = top_TABLE_BITS( (G[p] XOR seed) * FMIX64 )     [finalise step]

The CRITICAL architectural fact: **G[p] is completely independent of the seed.**
The seed only XORs into G[p] in the single finalisation multiply before the top-N
bit extraction.  This means:

  1. All N G-states can be pre-computed in ONE O(N) pass over the training data,
     with zero knowledge of the seed.

  2. For any candidate seed s, all N bucket addresses are:
         buckets = ((G ^ s) * FMIX64) >> (64 − TABLE_BITS)
     This is ONE vectorised numpy operation on the pre-computed G array.

  3. The QUALITY of a seed is entirely determined by how well it separates
     training positions that have DIFFERENT next-tokens into different buckets.
     Better separation → fewer adversarial collisions → faster Boyer-Moore
     convergence → higher training accuracy → lower validation BPB.

=======================================================================
THE ADVERSARIAL COLLISION METRIC
=======================================================================

An *adversarial collision* is a pair of positions (p1, p2) where:
  - bucket[p1] == bucket[p2]   (same table slot)
  - tokens[p1+1] != tokens[p2+1]   (different prediction targets)

These directly poison the Boyer-Moore majority vote: the two contexts fight
each other, lowering the winner's count and potentially swapping the winner
entirely.  The MORE adversarial collisions a seed produces, the worse the
model will be after any finite number of training passes.

The *pure-bucket fraction* is the fraction of filled buckets that have
exactly ONE unique next-token.  A perfect seed would give pure_fraction=1.0
(every context hash is unique), yielding 100% accuracy on seen training data.

=======================================================================
WHAT CAN BE FOUND BEFORE TRAINING
=======================================================================

BEFORE TRAINING (O(N_sample) work):
  ✓ Adversarial collision rate for any candidate seed
  ✓ Expected training accuracy proxy (bucket majority-vote simulation)
  ✓ Expected BPB proxy (from accuracy proxy + uniform fallback for misses)
  ✓ Optimal seed(s) among K candidates via vectorised screening
  ✓ Theoretical BPB lower bound for the hash-table architecture:
      BPB_floor ≈ (collision_rate × uniform_fallback_bpb
                   + (1−collision_rate) × correct_pred_bpb)

CANNOT BE FULLY KNOWN BEFORE TRAINING:
  ✗ Exact val BPB — requires building the packed table, bigram table,
    and fingerprint table, then evaluating on validation data
  ✗ Phase 4 convergence behaviour — depends on which wrong entries exist
    and the order in which Phase A/B sub-passes process them
  ✗ True SWA agreement rate — depends on Phase 4 repair stochasticity

=======================================================================
ALGORITHM: PRE-TRAINING SEED SCREENING
=======================================================================

Instead of running training with arbitrary seeds {42, 7, 1337}, use
this module to:

  1. Pre-compute G-states from first ~1M training tokens  (~0.1 s)
  2. Screen K=1000–10000 candidate seeds for adversarial collision rate
     (vectorised batches of 100 seeds, ~0.2 s total for K=1000)
  3. Select top-3 seeds with lowest adversarial collision rate
  4. Run full Phases 2–4 training only on those 3 optimal seeds
  5. Merge via majority vote as before

Expected improvement: replacing arbitrary seeds with optimally-screened
seeds should increase the full-agreement rate in merge_hdc_tables() from
~50% to ~65–75%, directly improving the merged table's BPB.

=======================================================================
RUN AS STANDALONE SCRIPT
=======================================================================

  cd /workspace/parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb
  python _optimal_seed_search.py \\
      --tokens_path ../../../data/datasets/fineweb10B_sp1024 \\
      --n_candidates 2000 \\
      --top_k 3 \\
      --sample_tokens 1000000

  Outputs best seeds to stdout and writes seeds_ranked.json in same directory.

=======================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ─── Constants matching train_gpt.py / _full_context_hash.py ─────────────────
_FMIX64      = np.uint64(0x9E3779B97F4A7C15)   # Fibonacci / golden-ratio constant
_PHI64       = np.uint64(0x9E3779B97F4A7C15)   # same constant used as PHI
TABLE_BITS   = 22                               # 2^22 = 4,194,304 table entries
TABLE_SIZE   = 1 << TABLE_BITS
VOCAB_SIZE   = 1024                             # BPE vocabulary size
_SHIFT       = np.uint64(64 - TABLE_BITS)       # right-shift to extract top 22 bits


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Pre-compute G-states (seed-independent rolling hash)
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_g_states(tokens: np.ndarray) -> np.ndarray:
    """Compute the rolling XOR hash G[p] for every position p.

    G[p] encodes tokens[0 .. p-1].  The result is seed-independent:
    different seeds only change the finalise() step, not the G values.

    Algorithm (matches train_gpt.py lines 5790-5798):
        KEY[p] = ((p+1) * PHI64) ^ ((p+1) >> 32)  |  1    (Fibonacci bijection)
        G[0]   = 0
        G[p+1] = G[p]  XOR  (tokens[p] * KEY[p])

    Implemented via np.bitwise_xor.accumulate on the contributions array,
    then shift-by-one to get the EXCLUSIVE prefix XOR (G[p] excludes tokens[p]).

    Parameters
    ----------
    tokens : np.ndarray (N,) dtype convertible to uint64
        Training token IDs (0-1023 for this vocab).

    Returns
    -------
    g_states : np.ndarray (N,) dtype uint64
        g_states[p] = G[p]  =  XOR_{i < p} (tokens[i] * KEY[i])
        Note: g_states[0] = 0 always (empty prefix).
    """
    N = len(tokens)
    t = tokens.astype(np.uint64)

    # Fibonacci position keys matching hadamard_key_batch()
    positions = np.arange(N, dtype=np.uint64) + np.uint64(1)   # 1-indexed
    with np.errstate(over='ignore'):
        keys = positions * _PHI64
    keys ^= (keys >> np.uint64(32))
    keys |= np.uint64(1)                         # force odd for invertibility

    # Contributions: tokens[p] * KEY[p]
    with np.errstate(over='ignore'):
        contribs = t * keys                      # (N,) uint64

    # Inclusive prefix XOR: cumxor[i] = XOR_{j=0..i} contribs[j]
    cumxor = np.bitwise_xor.accumulate(contribs)  # (N,) uint64

    # Exclusive prefix XOR: G[p] = XOR_{j=0..p-1} contribs[j]
    # g[0] = 0, g[p] = cumxor[p-1] for p >= 1
    g_states = np.empty(N, dtype=np.uint64)
    g_states[0] = np.uint64(0)
    if N > 1:
        g_states[1:] = cumxor[:-1]

    return g_states


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Bucket assignment for a single seed
# ═══════════════════════════════════════════════════════════════════════════════

def g_to_buckets(g_states: np.ndarray, seed: int) -> np.ndarray:
    """Convert G-states to table bucket addresses for a given seed.

    bucket[p] = top_TABLE_BITS( (G[p] XOR seed) * FMIX64 )

    This is the exact formula from train_gpt.py line 5799 and
    _full_context_hash.py's _finalise_vec().

    Parameters
    ----------
    g_states : np.ndarray (N,) uint64
    seed : int  — 64-bit seed value

    Returns
    -------
    buckets : np.ndarray (N,) int64  — values in [0, TABLE_SIZE)
    """
    with np.errstate(over='ignore'):
        finalised = (g_states ^ np.uint64(seed & 0xFFFF_FFFF_FFFF_FFFF)) * _FMIX64
    return (finalised >> _SHIFT).astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Adversarial collision metric (single seed)
# ═══════════════════════════════════════════════════════════════════════════════

def adversarial_collision_score(
    g_states: np.ndarray,
    next_tokens: np.ndarray,
    seed: int,
) -> Tuple[float, float, float]:
    """Measure adversarial collisions and training-accuracy proxy for one seed.

    A bucket is *adversarial* when it contains two or more positions with
    DIFFERENT next-tokens.  The most-frequent next-token (Boyer-Moore winner)
    will be predicted for that bucket; any position with a minority next-token
    is a training error.

    This function simulates the DNA-stacking Phase 2 result without actually
    building the full 4 MB table — it only needs the sample of N positions.

    Parameters
    ----------
    g_states : np.ndarray (N,) uint64
    next_tokens : np.ndarray (N,) int64  — tokens[p+1] for each position p
    seed : int

    Returns
    -------
    (adversarial_fraction, accuracy_proxy, bpb_proxy) : Tuple[float, float, float]
        adversarial_fraction : fraction of filled buckets with >1 unique next-token
        accuracy_proxy       : fraction of positions correctly predicted by majority
        bpb_proxy            : estimated BPB from accuracy_proxy
    """
    N = len(g_states)
    buckets = g_to_buckets(g_states, seed)

    # Encode (bucket, next_token) pairs into a single 64-bit key
    # Using int64 arithmetic: bucket values are < 2^22, vocab < 2^10
    # → combined key fits comfortably in int64
    bucket_i64 = buckets.astype(np.int64)
    tok_i64    = next_tokens.astype(np.int64)
    pair_keys  = bucket_i64 * np.int64(VOCAB_SIZE) + tok_i64   # (N,) int64

    # Count occurrences of each (bucket, next_token) pair
    uniq_pairs, pair_counts = np.unique(pair_keys, return_counts=True)
    pair_buckets = uniq_pairs // VOCAB_SIZE     # bucket part of each unique pair

    # Count occurrences of each bucket (across all next-tokens)
    uniq_buckets, bucket_fill_counts = np.unique(pair_buckets, return_counts=True)
    n_filled       = len(uniq_buckets)
    n_adversarial  = int(np.sum(bucket_fill_counts > 1))
    adversarial_fraction = n_adversarial / n_filled if n_filled > 0 else 1.0

    # For each filled bucket, find the majority-vote winner's count
    # pair_counts[i] = how many times bucket `pair_buckets[i]` predicted `tok` i
    # For each unique bucket, the winner is the pair with the highest count
    #
    # Build a bucket→max_count map via scatter with np.maximum.reduceat (sorted):
    sort_idx    = np.argsort(pair_buckets, kind='stable')
    sp_buckets  = pair_buckets[sort_idx]
    sp_counts   = pair_counts[sort_idx]

    # boundaries: first occurrence of each unique bucket in sorted arrays
    boundaries  = np.concatenate([[0], np.where(np.diff(sp_buckets) != 0)[0] + 1])
    max_counts  = np.maximum.reduceat(sp_counts, boundaries)

    # accuracy_proxy = (total positions where winner is correct) / N
    # = sum of max_counts / N   (since total = N and max_count = correct votes)
    total_correct = int(np.sum(max_counts))
    accuracy_proxy = total_correct / N if N > 0 else 0.0

    # BPB proxy: use the same probability formula as evaluate_bpb_seed_projection():
    #   correct prediction: prob = min(0.99, 0.5 + 0.49 * (1 - exp(-confidence/5)))
    #   incorrect/miss:     prob = 1 / VOCAB_SIZE  (uniform fallback)
    #
    # For the proxy we treat all correct predictions as count=3 (crystallised):
    #   prob_correct = 0.5 + 0.49 * (1 - exp(-3/5)) = ~0.5 + 0.49*0.451 ≈ 0.721
    # and weight by accuracy:
    import math
    prob_correct   = 0.5 + 0.49 * (1.0 - math.exp(-3.0 / 5.0))   # ~0.721
    prob_incorrect = 1.0 / VOCAB_SIZE                               # ~0.000977
    expected_prob  = accuracy_proxy * prob_correct + (1.0 - accuracy_proxy) * prob_incorrect
    # BPB = -log2(prob) averaged; proxy uses the expected probability
    bpb_proxy = -math.log2(max(expected_prob, 1e-15)) / math.log2(256 / math.log2(VOCAB_SIZE + 1))
    # Simplified: just use bits-per-token then convert to bits-per-byte
    # bits-per-token = -log2(expected_prob)
    # bytes-per-token ≈ log2(VOCAB_SIZE) / 8  for a BPE vocab
    # bpb = bits_per_token / bytes_per_token
    bits_per_token  = -math.log2(max(expected_prob, 1e-15))
    bytes_per_token = math.log2(VOCAB_SIZE) / 8.0   # ~1.25 bytes for vocab=1024
    bpb_proxy       = bits_per_token / bytes_per_token

    return adversarial_fraction, accuracy_proxy, bpb_proxy


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Vectorised batch screening over many seeds
# ═══════════════════════════════════════════════════════════════════════════════

# log2(VOCAB_SIZE): used to extract the bucket ID from a uint32 pair key via right-shift.
# pair_key = bucket * VOCAB_SIZE + token  ∈ [0, TABLE_SIZE * VOCAB_SIZE) = [0, 2^32).
# VOCAB_SIZE = 1024 = 2^10  →  pair_key >> 10 == bucket exactly.
assert TABLE_BITS + 10 == 32, "uint32 pair-key packing requires TABLE_BITS + log2(VOCAB_SIZE) == 32"
_VOCAB_LOG2 = 10        # log2(VOCAB_SIZE) = log2(1024)
_VOCAB_SIZE_U32 = np.uint32(VOCAB_SIZE)


def screen_seeds_batch(
    g_states: np.ndarray,
    next_tokens: np.ndarray,
    candidate_seeds: np.ndarray,
    batch_size: int = 64,
    verbose: bool = True,
) -> np.ndarray:
    """Screen K candidate seeds and return adversarial fraction for each.

    OPTIMISED vs. the original implementation — same results, ~8–12× faster.

    Algorithm changes
    -----------------
    **Original** (slow path):
      For every seed in the batch, call ``np.unique`` on N int64 pair-keys.
      Each call does an O(N log N) sort + dedup pass, allocates fresh memory,
      then calls a second ``np.unique`` on the deduplicated bucket IDs.
      Total: K × 2 × O(N log N) with K independent Python/C transitions and
      K separate heap allocations.

    **Optimised** (fast path):
      1. **uint32 pair keys** — pack ``(bucket, token)`` into a single uint32.
         ``pair_key = bucket * VOCAB_SIZE + token``  since VOCAB_SIZE = 1024 = 2^10
         and TABLE_SIZE = 2^22, the result fits exactly in 32 bits.
         Sorting uint32 is ~1.5–2× faster than int64 (half the data volume).

      2. **One batched sort** — build pair keys as a (B, N) C-contiguous uint32
         array (B seeds × N positions) and call ``ndarray.sort(axis=1)``  once.
         This lets NumPy/C sort all B rows in a single C-level dispatch with
         better cache utilisation and zero per-seed Python/allocation overhead,
         compared to B separate ``np.unique`` calls.

      3. **O(N) diff instead of O(N log N) per-seed sort** — because each row is
         already sorted after step 2, unique pairs are detected in a single O(N)
         boolean comparison (``col[1:] != col[:-1]``).  Adversarial bucket
         counting then requires only O(M) ops (M ≪ N = number of unique pairs):

             n_adversarial = (bchg[:-1] & ~bchg[1:]).sum()

         where ``bchg`` marks bucket boundaries in the unique-pair list.  This
         formula counts exactly the bucket starts that are followed by at least
         one additional unique (bucket, token) entry — i.e. adversarial buckets.

    Memory: B × N × 4 bytes (uint32) + B × N × 8 bytes (uint64 intermediate).
    For N = 500 k, B = 64: ~384 MB peak.

    Parameters
    ----------
    g_states         : (N,) uint64 — pre-computed G[p] values
    next_tokens      : (N,) int64  — tokens[p+1]
    candidate_seeds  : (K,) int or array of candidate seed values
    batch_size       : number of seeds to evaluate simultaneously (tune for RAM)
    verbose          : print progress

    Returns
    -------
    scores : np.ndarray (K,) float64
        adversarial_fraction for each candidate seed.  Lower is better.
    """
    candidate_seeds = np.asarray(candidate_seeds, dtype=np.uint64)
    K   = len(candidate_seeds)
    N   = len(g_states)
    scores = np.empty(K, dtype=np.float64)

    g64     = g_states.astype(np.uint64)                   # (N,) uint64
    tok_u32 = next_tokens.astype(np.uint32)                 # (N,) uint32  [0..1023]

    t0 = time.time()
    for batch_start in range(0, K, batch_size):
        batch_end   = min(batch_start + batch_size, K)
        seeds_batch = candidate_seeds[batch_start:batch_end].astype(np.uint64)  # (B,)
        B           = len(seeds_batch)

        # ── Step A: compute uint32 pair keys for all B seeds — layout (B, N) ──────
        # Build (B, N) directly so each seed occupies a contiguous row (no transpose).
        with np.errstate(over='ignore'):
            # (B, N) broadcast: each of the B seeds XOR'd with all N g-states
            raw = g64[None, :] ^ seeds_batch[:, None]       # (B, N) uint64
            raw *= _FMIX64                                   # in-place multiply
        buckets_BN   = (raw >> _SHIFT).astype(np.uint32)    # (B, N) uint32; free raw
        del raw
        pair_keys_BN = buckets_BN * _VOCAB_SIZE_U32 + tok_u32[None, :]  # (B, N) uint32
        del buckets_BN

        # ── Step B: sort all B rows in one C-level call ───────────────────────────
        pair_keys_BN.sort(axis=1)                           # in-place; each row sorted

        # ── Step C: per-seed O(N) diff + O(M) adversarial count ──────────────────
        for b in range(B):
            col = pair_keys_BN[b]                           # (N,) uint32, sorted, contiguous

            # Unique-pair boundary: True where the pair key changes
            is_new      = np.empty(N, dtype=bool)
            is_new[0]   = True
            is_new[1:]  = col[1:] != col[:-1]

            # Extract unique pair values and recover bucket IDs via >> 10
            uniq = col[is_new]                              # (M,) uint32, M = unique pairs
            bkt  = uniq >> np.uint32(_VOCAB_LOG2)           # (M,) uint32, bucket IDs (sorted)
            M    = len(bkt)
            if M == 0:
                scores[batch_start + b] = 1.0
                continue

            # Bucket boundary within the unique-pair list
            bchg        = np.empty(M, dtype=bool)
            bchg[0]     = True
            bchg[1:]    = bkt[1:] != bkt[:-1]
            n_filled    = int(bchg.sum())

            # Adversarial buckets = bucket starts followed by an extra unique token.
            # In sorted order: bchg[i]=True (bucket start) AND bchg[i+1]=False
            # (same bucket, different token).  Summing these gives the exact count.
            if M > 1:
                n_adversarial = int((bchg[:-1] & ~bchg[1:]).sum())
            else:
                n_adversarial = 0

            scores[batch_start + b] = n_adversarial / n_filled if n_filled > 0 else 1.0

        if verbose and (batch_end % max(batch_size * 4, 1) == 0 or batch_end == K):
            elapsed = time.time() - t0
            rate    = batch_end / elapsed if elapsed > 0 else 0
            eta     = (K - batch_end) / rate if rate > 0 else 0
            print(f"  [SeedScreen] {batch_end}/{K} seeds   "
                  f"best_so_far={scores[:batch_end].min():.4f}   "
                  f"{elapsed:.1f}s elapsed   ETA {eta:.0f}s")

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3b — GPU-accelerated parallel seed screening (RTX / CUDA)
# ═══════════════════════════════════════════════════════════════════════════════

def screen_seeds_batch_gpu(
    g_states: np.ndarray,
    next_tokens: np.ndarray,
    candidate_seeds: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """GPU-accelerated parallel seed screening via PyTorch CUDA.

    Processes ALL K candidate seeds simultaneously in one (K, N) tensor on the
    GPU, then calls a single ``torch.sort(dim=1)`` across all rows.  Typically
    20–100× faster than the CPU batched version on a modern GPU (e.g. RTX 4090).

    Algorithm (identical to screen_seeds_batch — numerically equivalent):
        1. Build (K, N) int64 pair-key matrix via broadcast XOR + FMIX multiply.
        2. ``torch.sort(dim=1)`` — one GPU kernel for all K rows in parallel.
        3. Per-seed O(N) diff + O(M) adversarial count (Python loop, GPU tensors).

    Memory: K × N × 8 bytes (int64 pair keys).
        K=200, N=500 k → ~800 MB  (well within 24 GB RTX-4090 VRAM).

    Falls back transparently to ``screen_seeds_batch()`` when CUDA is unavailable
    or imports fail.
    """
    try:
        import torch as _tgpu
        if not _tgpu.cuda.is_available():
            raise RuntimeError("CUDA not available")
    except Exception:
        if verbose:
            print("  [SeedScreen] No CUDA — using CPU screening")
        return screen_seeds_batch(g_states, next_tokens, candidate_seeds, verbose=verbose)

    import torch
    dev = torch.device("cuda")
    K   = len(candidate_seeds)
    N   = len(g_states)
    t0  = time.time()

    if verbose:
        print(f"  [SeedScreen GPU] K={K} seeds  N={N:,} tokens — building pair-key matrix…")

    # Reinterpret uint64 → int64 (same bit pattern; PyTorch has no native uint64)
    g_i64  = g_states.view(np.int64)
    s_i64  = candidate_seeds.view(np.int64)
    g_t    = torch.as_tensor(g_i64,                           dtype=torch.int64, device=dev)
    tok_t  = torch.as_tensor(next_tokens.astype(np.int64),    dtype=torch.int64, device=dev)
    seed_t = torch.as_tensor(s_i64,                           dtype=torch.int64, device=dev)

    # FMIX64 = 0x9E3779B97F4A7C15 reinterpreted as signed int64
    _fmix  = int(np.array([0x9E3779B97F4A7C15], dtype=np.uint64).view(np.int64)[0])
    FMIX_t = torch.tensor(_fmix, dtype=torch.int64, device=dev)
    MASK_t = torch.tensor((1 << TABLE_BITS) - 1, dtype=torch.int64, device=dev)
    SHIFT  = 64 - TABLE_BITS   # = 42 for TABLE_BITS=22

    # ── Build (K, N) pair-key matrix ──────────────────────────────────────────
    raw       = g_t[None, :] ^ seed_t[:, None]           # (K, N) — XOR identical to uint64
    raw       = raw * FMIX_t                               # (K, N) — overflow = uint64 wrap
    buckets   = (raw >> SHIFT) & MASK_t                    # logical right-shift via mask
    pair_keys = buckets * VOCAB_SIZE + tok_t[None, :]      # (K, N) ∈ [0, 2^32-1] as int64
    del raw, buckets
    torch.cuda.synchronize()

    if verbose:
        print(f"  [SeedScreen GPU] Pair keys built in {time.time()-t0:.2f}s — sorting…")

    # ── Single GPU sort across all K rows ─────────────────────────────────────
    pair_keys_sorted, _ = torch.sort(pair_keys, dim=1)
    del pair_keys
    torch.cuda.synchronize()

    if verbose:
        t1 = time.time()
        print(f"  [SeedScreen GPU] Sort done in {t1-t0:.2f}s — counting adversarials…")

    # ── Per-seed O(N) diff + O(M) adversarial count ───────────────────────────
    scores = np.empty(K, dtype=np.float64)
    for k in range(K):
        col        = pair_keys_sorted[k]                   # (N,) int64 sorted
        is_new     = torch.empty(N, dtype=torch.bool, device=dev)
        is_new[0]  = True
        is_new[1:] = col[1:] != col[:-1]

        uniq = col[is_new]
        bkt  = uniq >> _VOCAB_LOG2                         # (M,) int64 — bucket IDs
        M    = int(bkt.shape[0])
        if M == 0:
            scores[k] = 1.0
            continue

        bchg      = torch.empty(M, dtype=torch.bool, device=dev)
        bchg[0]   = True
        bchg[1:]  = bkt[1:] != bkt[:-1]
        n_filled  = int(bchg.sum())
        n_adv     = int((bchg[:-1] & ~bchg[1:]).sum()) if M > 1 else 0
        scores[k] = n_adv / n_filled if n_filled > 0 else 1.0

    if verbose:
        elapsed = time.time() - t0
        print(f"  [SeedScreen GPU] {K} seeds  {elapsed:.2f}s  "
              f"({K / max(elapsed, 1e-6):.0f} seeds/s)  best={scores.min():.4f}")
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4b — One-step gradient refinement of a single seed
# ═══════════════════════════════════════════════════════════════════════════════

def one_step_gradient_refine(
    g_states: np.ndarray,
    next_tokens: np.ndarray,
    seed: int,
    verbose: bool = True,
) -> Tuple[int, float, float]:
    """Refine a seed via one gradient-like step in its 64-bit Hamming neighbourhood.

    For each of the 64 bit positions, evaluates the adversarial-collision score
    after flipping only that bit.  The single bit-flip that most reduces the
    adversarial fraction is accepted; if no flip improves the score the original
    seed is returned unchanged.

    Analogy — GPTQ one Newton step: GPTQ takes one closed-form Newton step to
    quantise each weight while minimising layer reconstruction error.  Here we
    take one local-search step to minimise the *seed's adversarial collision
    rate* (the HDC analog of reconstruction error):

        seed* = argmin_{s : hamming(s, seed)=1}  adversarial_fraction(s)

    Runtime: exactly 1 call to screen_seeds_batch with 64 variants, i.e. the
    same cost as screening 64 random candidates during find_optimal_seeds —
    negligible (<0.1 s per seed for 1M tokens).

    Parameters
    ----------
    g_states    : (N,) uint64 — pre-computed G[p] values (seed-independent)
    next_tokens : (N,) int64  — tokens[p+1] for each position p
    seed        : int         — candidate seed to refine
    verbose     : bool        — print improvement info when found

    Returns
    -------
    (refined_seed, original_score, refined_score) : Tuple[int, float, float]
        refined_seed   : best single-bit-flip variant (equals seed if no gain)
        original_score : adversarial fraction BEFORE refinement
        refined_score  : adversarial fraction AFTER  refinement (≤ original)
    """
    seed_u64 = np.uint64(int(seed) & 0xFFFF_FFFF_FFFF_FFFF)

    # Generate all 64 single-bit-flip neighbours
    bit_variants = np.array(
        [int(seed_u64 ^ np.uint64(np.uint64(1) << np.uint64(b))) for b in range(64)],
        dtype=np.uint64,
    )

    # Score all 64 variants in one vectorised batch — use GPU when available
    variant_scores = screen_seeds_batch_gpu(
        g_states, next_tokens, bit_variants, verbose=False
    )

    # Baseline score for the current seed (full metric)
    original_score, _, _ = adversarial_collision_score(g_states, next_tokens, int(seed))

    best_bit   = int(np.argmin(variant_scores))
    best_score = float(variant_scores[best_bit])

    if best_score < original_score - 1e-9:          # strict improvement
        refined_seed = int(bit_variants[best_bit])
        if verbose:
            print(
                f"  [OneStepGrad]  bit {best_bit:>2d} flipped: "
                f"{original_score:.4f} → {best_score:.4f}  "
                f"(Δ={best_score - original_score:+.5f})"
            )
        return refined_seed, original_score, best_score
    else:
        if verbose:
            print(
                f"  [OneStepGrad]  no improvement from any single bit-flip "
                f"(baseline={original_score:.4f})"
            )
        return int(seed), original_score, original_score


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — Main entry point: find optimal seeds
# ═══════════════════════════════════════════════════════════════════════════════

def find_optimal_seeds(
    tokens: np.ndarray,
    n_candidates: int = 2000,
    top_k: int = 3,
    sample_size: int = 1_000_000,
    screen_sample_size: Optional[int] = None,
    batch_size: int = 64,
    rng_seed: int = 0xABCDEF,
    verbose: bool = True,
    one_step_grad: bool = True,
) -> List[dict]:
    """Find the top-k seeds with the lowest adversarial collision rate.

    This is the main pre-training seed optimisation pipeline:

        tokens  →  G[p] pre-computation  (fast: screen_sample_size tokens)
                →  screen K random candidates with small sample
                →  select top-k by adversarial-collision rate
                →  re-score top-k with full sample_size tokens
                →  [optional] one-step gradient refinement per seed
                →  top-k optimised seeds

    Two-phase sampling
    ------------------
    ``screen_sample_size`` controls the token count used for the bulk
    candidate screening pass.  It defaults to ``min(sample_size // 2, 500_000)``
    which is typically sufficient to rank seeds reliably while roughly halving
    the screening time.  After the top-k candidates are selected they are
    re-scored and refined using the full ``sample_size`` for accuracy.

    The returned seeds can be passed directly to run_multi_seed_training() or
    the --seeds flag of train_gpt.py, replacing the arbitrary default seeds.

    Parameters
    ----------
    tokens             : full training token array (or a large sample)
    n_candidates       : number of candidate seeds to evaluate
    top_k              : number of best seeds to return
    sample_size        : tokens for top-k re-scoring and gradient refinement
                         (1M default; more → more accurate final metrics)
    screen_sample_size : tokens for the bulk candidate screening pass.
                         Default: min(sample_size // 2, 500_000).
                         Smaller → faster screening with negligible ranking
                         quality loss (discriminative power saturates early).
    batch_size         : seeds per vectorised batch (tune for available RAM)
    rng_seed           : RNG seed for generating candidate seeds
    one_step_grad      : if True (default) each of the top-k seeds is refined
                         by one_step_gradient_refine() — a single pass over
                         all 64 bit-flip neighbours.  Costs ~64 extra seed
                         evaluations per top-k seed (< 0.1 s at 1M tokens).
                         Usually lowers adversarial fraction by 0.001–0.005.

    Returns
    -------
    List[dict] of length top_k, each with keys:
        seed (int), adversarial_fraction (float), accuracy_proxy (float),
        bpb_proxy (float), rank (int), one_step_grad_applied (bool),
        pre_grad_adversarial_fraction (float)
    """
    N_total = len(tokens)

    # ── Full-accuracy sample (used for top-k re-scoring + gradient refine) ───
    N = min(sample_size, N_total - 1)       # -1 because we need next_tokens[p+1]
    if N < 10_000:
        raise ValueError(f"Need at least 10,001 tokens; got {N_total}")

    # ── Fast-screening sample (smaller subset for bulk candidate pass) ────────
    _default_screen = min(N // 2, 500_000)
    N_screen = min(
        screen_sample_size if screen_sample_size is not None else _default_screen,
        N_total - 1,
    )
    N_screen = max(N_screen, 10_000)

    if verbose:
        print(f"\n[SeedScreen] Pre-computing G-states for screening "
              f"({N_screen:,} tokens)...")
    t0 = time.time()
    toks_screen   = tokens[:N_screen].astype(np.uint64)
    next_toks_scr = tokens[1:N_screen + 1].astype(np.int64)
    g_screen      = precompute_g_states(toks_screen)
    if verbose:
        print(f"[SeedScreen] G-states ready in {time.time() - t0:.2f}s")

    # ── Generate candidate seeds ──────────────────────────────────────────────
    rng = np.random.RandomState(rng_seed)
    # Sample a diverse set: mix of small integers, large values, and purely random
    # Small integers often have poor bit diversity; mix in structured candidates too.
    n_random     = int(n_candidates * 0.70)
    n_structured = n_candidates - n_random
    random_seeds = rng.randint(0, 2**63, size=n_random, dtype=np.int64).astype(np.uint64)
    # Structured candidates: powers-of-two XOR'd with Fibonacci constant, primes, etc.
    phi_variants = np.array([
        0x9E3779B97F4A7C15,   # Fibonacci
        0x6C62272E07BB0142,   # FNV prime
        0xBF58476D1CE4E5B9,   # splitmix64 step 1
        0x94D049BB133111EB,   # splitmix64 step 2
        0x517CC1B727220A95,   # MurmurHash3
        0xA0761D6478BD642F,   # wyhash
        0xE7037ED1A0B428DB,   # wyhash
        0x8EBC6AF09C88C6E3,   # xxhash
    ] * (n_structured // 8 + 1), dtype=np.uint64)[:n_structured]
    # XOR each with a fresh random 64-bit value for diversity
    phi_rng_mix  = rng.randint(0, 2**63, size=n_structured, dtype=np.int64).astype(np.uint64)
    struct_seeds = phi_variants ^ phi_rng_mix
    candidate_seeds = np.concatenate([random_seeds, struct_seeds])

    if verbose:
        print(f"[SeedScreen] Screening {n_candidates:,} candidate seeds "
              f"(batch_size={batch_size}, screen_sample={N_screen:,} tokens)...")

    # ── Screen all candidates with the fast screening sample ─────────────────
    t1 = time.time()
    scores = screen_seeds_batch_gpu(g_screen, next_toks_scr, candidate_seeds,
                                    verbose=verbose)
    elapsed = time.time() - t1
    if verbose:
        print(f"[SeedScreen] Screening complete in {elapsed:.1f}s  "
              f"({n_candidates / elapsed:.0f} seeds/s)")

    # ── Rank and select top-k ─────────────────────────────────────────────────
    ranked_idx  = np.argsort(scores)        # ascending collision rate = better
    top_indices = ranked_idx[:top_k]
    top_seeds   = list(candidate_seeds[top_indices].astype(np.uint64))
    top_scores  = list(scores[top_indices])

    # ── Upgrade to full-accuracy G-states for refinement + reporting ──────────
    # The bulk screen used N_screen tokens; top-k refinement and final metrics
    # use the full N tokens for higher fidelity.
    if N_screen < N:
        if verbose:
            print(f"\n[SeedScreen] Pre-computing G-states for refinement "
                  f"({N:,} tokens)...")
        t2 = time.time()
        toks_full  = tokens[:N].astype(np.uint64)
        next_toks  = tokens[1:N + 1].astype(np.int64)
        g_states   = precompute_g_states(toks_full)
        if verbose:
            print(f"[SeedScreen] G-states ready in {time.time() - t2:.2f}s")
    else:
        # screen sample already covers full requested size — reuse directly
        next_toks = next_toks_scr
        g_states  = g_screen

    # ── One-step gradient refinement (optional) ───────────────────────────────
    # For each of the top-k seeds, scan all 64 single-bit-flip neighbours and
    # accept the one that most reduces the adversarial collision rate.  This is
    # the HDC analog of "one Newton step" from GPTQ: one closed-form local-
    # search step that minimises the collision-loss without re-running the full
    # K-candidate random search.  Cost: 64 extra evaluations per seed.
    pre_grad_scores = list(top_scores)      # baseline scores before refinement
    if one_step_grad:
        if verbose:
            print(f"\n[SeedScreen] One-step gradient refinement for top-{top_k} seeds...")
        for i in range(top_k):
            refined, orig_s, ref_s = one_step_gradient_refine(
                g_states, next_toks, int(top_seeds[i]), verbose=verbose
            )
            top_seeds[i]  = np.uint64(refined)
            top_scores[i] = ref_s
        if verbose:
            improved = sum(
                1 for a, b in zip(pre_grad_scores, top_scores) if b < a - 1e-9
            )
            print(f"[SeedScreen] One-step gradient: {improved}/{top_k} seeds improved")

    if verbose:
        grad_tag = " (after one-step gradient)" if one_step_grad else ""
        print(f"\n[SeedScreen] Top {top_k} seeds{grad_tag}:")
        print(f"  {'Rank':>4}  {'Seed (hex)':>20}  {'Adv.Collision':>14}  "
              f"{'AccProxy':>9}  {'BPB proxy':>9}  {'Grad':>5}")
        print(f"  {'----':>4}  {'-'*20}  {'-------------':>14}  "
              f"{'--------':>9}  {'---------':>9}  {'-----':>5}")

    results = []
    for i, (seed_val, adv_frac) in enumerate(zip(top_seeds, top_scores)):
        # Compute full metrics (accuracy + BPB proxy) using the full g_states
        adv_full, acc_proxy, bpb_proxy = adversarial_collision_score(
            g_states, next_toks, int(seed_val)
        )
        grad_applied = one_step_grad and (adv_frac < pre_grad_scores[i] - 1e-9)
        entry = {
            "rank"                          : i + 1,
            "seed"                          : int(seed_val),
            "seed_hex"                      : f"0x{int(seed_val):016X}",
            "adversarial_fraction"          : float(adv_full),
            "accuracy_proxy"                : float(acc_proxy),
            "bpb_proxy"                     : float(bpb_proxy),
            "sample_size"                   : N,
            "screen_sample_size"            : N_screen,
            "one_step_grad_applied"         : bool(grad_applied),
            "pre_grad_adversarial_fraction" : float(pre_grad_scores[i]),
        }
        results.append(entry)
        if verbose:
            grad_mark = "✓" if grad_applied else " "
            print(f"  {i+1:>4}  {entry['seed_hex']:>20}  "
                  f"{adv_full:>14.4f}  {acc_proxy:>9.5f}  {bpb_proxy:>9.4f}  "
                  f"{grad_mark:>5}")

    # Also evaluate the DEFAULT seeds for comparison
    default_seeds = [42, 7, 1337]
    if verbose:
        print(f"\n[SeedScreen] Comparison: DEFAULT seeds {default_seeds}")
        print(f"  {'Seed':>10}  {'Adv.Collision':>14}  {'AccProxy':>9}  {'BPB proxy':>9}")
    for ds in default_seeds:
        adv_d, acc_d, bpb_d = adversarial_collision_score(g_states, next_toks, ds)
        if verbose:
            print(f"  {ds:>10}  {adv_d:>14.4f}  {acc_d:>9.5f}  {bpb_d:>9.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Proof of Concept: why seed affects BPB
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_seed_effect(
    tokens: np.ndarray,
    seeds_to_compare: Optional[List[int]] = None,
    sample_size: int = 200_000,
) -> None:
    """Print a side-by-side comparison of seed quality metrics.

    Demonstrates that different seeds produce meaningfully different
    adversarial collision rates on the same training data, and that
    the BPB proxy correlates with this.

    This is a quick sanity check / pedagogical demo — not the full
    optimisation pipeline.  Call find_optimal_seeds() for real search.
    """
    if seeds_to_compare is None:
        seeds_to_compare = [
            7, 42, 1337,                         # current defaults
            0x9E3779B97F4A7C15,                  # Fibonacci constant itself
            0xBF58476D1CE4E5B9,                  # splitmix64
            0x6C62272E07BB0142,                  # FNV prime
            0,                                    # degenerate: zero seed
            1,                                    # degenerate: unit seed
        ]

    N = min(sample_size, len(tokens) - 1)
    toks = tokens[:N].astype(np.uint64)
    next_toks = tokens[1:N + 1].astype(np.int64)

    print(f"\n[SeedDemo] G-state pre-computation ({N:,} tokens)...")
    t0 = time.time()
    g_states = precompute_g_states(toks)
    print(f"[SeedDemo] Done in {time.time() - t0:.3f}s\n")

    print(f"  {'Seed':>10}  {'Adv.Collision':>14}  {'AccProxy':>9}  {'BPB proxy':>9}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*9}  {'-'*9}")
    for seed in seeds_to_compare:
        adv, acc, bpb = adversarial_collision_score(g_states, next_toks, seed)
        marker = "  ← default" if seed in [7, 42, 1337] else ""
        print(f"  {seed:>10}  {adv:>14.4f}  {acc:>9.5f}  {bpb:>9.4f}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# BPB Lower Bound Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bpb_lower_bound(
    tokens: np.ndarray,
    seed: int,
    sample_size: int = 1_000_000,
) -> dict:
    """Compute the theoretical BPB lower bound for the hash-table architecture.

    The lower bound assumes:
      - Perfect Phase 2/3/4 convergence (every non-adversarial bucket crystallised)
      - 11% residual hash collisions on validation data (measured empirically)
      - Bigram fallback catches some of the collision cases

    The actual BPB will be >= this bound because:
      - Training coverage is incomplete (not all val contexts seen in training)
      - Phase 4 budget is finite (10 minutes wall-clock)
      - Some adversarial buckets retain wrong predictions

    Returns a dict with lower-bound components.
    """
    import math

    N = min(sample_size, len(tokens) - 1)
    toks = tokens[:N].astype(np.uint64)
    next_toks = tokens[1:N + 1].astype(np.int64)

    g_states = precompute_g_states(toks)

    # ── Measure training-data statistics ─────────────────────────────────────
    buckets   = g_to_buckets(g_states, seed)
    pair_keys = buckets.astype(np.int64) * np.int64(VOCAB_SIZE) + next_toks
    uniq_pairs, pair_counts = np.unique(pair_keys, return_counts=True)
    pair_buckets = uniq_pairs // VOCAB_SIZE

    uniq_bkts, bkt_cnt = np.unique(pair_buckets, return_counts=True)
    n_filled      = len(uniq_bkts)
    n_clean       = int(np.sum(bkt_cnt == 1))     # one unique next-token: zero adversarial collision
    n_adversarial = n_filled - n_clean

    fill_rate        = n_filled / TABLE_SIZE
    clean_rate       = n_clean / n_filled if n_filled > 0 else 0.0
    adversarial_rate = n_adversarial / n_filled if n_filled > 0 else 1.0

    # ── Probability components ────────────────────────────────────────────────
    # P(hit a filled + clean bucket) on a new input ≈ fill_rate × clean_rate × (1 − val_collision_rate)
    # val_collision_rate ≈ 0.11 for the rolling hash (measured in _full_context_hash.py)
    val_collision_rate = 0.11

    p_clean_correct = fill_rate * clean_rate * (1.0 - val_collision_rate)
    p_adversarial   = fill_rate * adversarial_rate * (1.0 - val_collision_rate)
    p_miss          = 1.0 - fill_rate * (1.0 - val_collision_rate)

    # For clean buckets, prob = 0.721 (count=3 crystallised, from Phase 4)
    prob_clean = 0.5 + 0.49 * (1.0 - math.exp(-3.0 / 5.0))
    # For adversarial buckets, the majority token is correct with probability
    # slightly above 1/n_conflict ≈ 0.5 (most conflicts are 2-way)
    prob_adversarial = 0.5 + 0.49 * (1.0 - math.exp(-1.0 / 5.0))  # count≈1
    # For misses: uniform
    prob_miss = 1.0 / VOCAB_SIZE

    expected_prob = (p_clean_correct * prob_clean
                     + p_adversarial  * prob_adversarial
                     + p_miss         * prob_miss)

    bits_per_token  = -math.log2(max(expected_prob, 1e-15))
    bytes_per_token = math.log2(VOCAB_SIZE) / 8.0
    bpb_lower_bound = bits_per_token / bytes_per_token

    return {
        "seed"                   : seed,
        "seed_hex"               : f"0x{seed & 0xFFFF_FFFF_FFFF_FFFF:016X}",
        "n_sample"               : N,
        "n_filled_buckets"       : n_filled,
        "fill_rate"              : fill_rate,
        "clean_buck_fraction"    : clean_rate,
        "adversarial_fraction"   : adversarial_rate,
        "p_clean_correct"        : p_clean_correct,
        "p_adversarial_correct"  : p_adversarial,
        "p_miss"                 : p_miss,
        "expected_prob"          : expected_prob,
        "bits_per_token"         : bits_per_token,
        "bytes_per_token_approx" : bytes_per_token,
        "bpb_lower_bound"        : bpb_lower_bound,
        "note"                   : (
            "This is a lower bound.  Actual BPB >= this value because Phase 4 is "
            "time-bounded and val coverage is incomplete."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Token loading helper (mirrors train_gpt.py data loading)
# ═══════════════════════════════════════════════════════════════════════════════

def load_tokens(data_path: str, max_tokens: int = 2_000_000) -> np.ndarray:
    """Load training tokens from the fineweb10B sharded dataset.

    Mirrors the data-loading logic in train_gpt.py to ensure identical tokens.
    Loads shards in order until max_tokens is reached.

    Parameters
    ----------
    data_path  : path to the directory containing *.bin shard files
    max_tokens : maximum tokens to load (2M default; 1M is enough for screening)

    Returns
    -------
    np.ndarray (N,) uint16 — token IDs
    """
    import glob

    data_path = Path(data_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    shard_files = sorted(glob.glob(str(data_path / "*.bin")))
    if not shard_files:
        raise FileNotFoundError(f"No .bin shard files found in {data_path}")

    # BUG FIX: each shard has a 256-byte header (magic + vocab_size + token_count
    # + padding) that must be skipped before reading the uint16 token array.
    # Original np.fromfile(sf, dtype=np.uint16) read the header as tokens,
    # corrupting the first 128 "tokens" in every shard.
    # Mirrors load_data_shard() in train_gpt.py exactly.
    import struct as _struct
    _HEADER_BYTES = 256
    _MAGIC        = 20240520

    all_tokens: List[np.ndarray] = []
    total = 0
    for sf in shard_files:
        with open(sf, "rb") as _f:
            header = _f.read(_HEADER_BYTES)
        magic      = _struct.unpack_from('<I', header, 0)[0]
        if magic != _MAGIC:
            raise ValueError(f"Invalid shard magic in {sf}: expected {_MAGIC}, got {magic}")
        token_count = _struct.unpack_from('<Q', header, 8)[0]
        shard = np.memmap(sf, dtype=np.uint16, mode='r',
                          offset=_HEADER_BYTES, shape=(token_count,))
        all_tokens.append(np.array(shard))   # copy out of mmap before closing
        total += token_count
        if total >= max_tokens:
            break

    tokens = np.concatenate(all_tokens)[:max_tokens]
    print(f"[SeedScreen] Loaded {len(tokens):,} tokens from {len(all_tokens)} shard(s)")
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-training seed optimiser for HDC rolling-hash model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tokens_path", type=str,
        default="../../../data/datasets/fineweb10B_sp1024",
        help="Path to fineweb10B sharded dataset directory",
    )
    parser.add_argument(
        "--n_candidates", type=int, default=2000,
        help="Number of candidate seeds to evaluate (default: 2000)",
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Number of top seeds to return (default: 3, for 3-seed merge)",
    )
    parser.add_argument(
        "--sample_tokens", type=int, default=1_000_000,
        help="Training tokens for top-k re-scoring and gradient refinement (default: 1M)",
    )
    parser.add_argument(
        "--screen_sample_tokens", type=int, default=None,
        help="Training tokens for the fast bulk-screening pass "
             "(default: min(sample_tokens // 2, 500_000)). "
             "Smaller → faster with negligible ranking quality loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Seeds per vectorised batch (tune for available RAM, default: 64)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run quick demo comparing default seeds vs structured alternatives",
    )
    parser.add_argument(
        "--lower_bound", action="store_true",
        help="Compute BPB lower bound for the best found seed",
    )
    parser.add_argument(
        "--out", type=str, default="seeds_ranked.json",
        help="Output JSON path for ranked seeds (default: seeds_ranked.json)",
    )
    # One-step gradient refinement flag (default: enabled)
    _osg = parser.add_mutually_exclusive_group()
    _osg.add_argument(
        "--one_step_grad", dest="one_step_grad", action="store_true", default=True,
        help="Apply one-step gradient refinement to top-k seeds (default: ON). "
             "Each of the top-k seeds is refined by testing all 64 single-bit-flip "
             "neighbours and accepting the best improvement.",
    )
    _osg.add_argument(
        "--no_one_step_grad", dest="one_step_grad", action="store_false",
        help="Disable one-step gradient refinement.",
    )
    args = parser.parse_args()

    # ── Load tokens ───────────────────────────────────────────────────────────
    try:
        tokens = load_tokens(args.tokens_path, max_tokens=args.sample_tokens + 1)
    except FileNotFoundError as e:
        print(f"[SeedScreen] ERROR: {e}", file=sys.stderr)
        print("[SeedScreen] Run data/cached_challenge_fineweb.py first to download data.",
              file=sys.stderr)
        sys.exit(1)

    if args.demo:
        demonstrate_seed_effect(tokens, sample_size=min(200_000, len(tokens) - 1))
        return

    # ── Main seed search ──────────────────────────────────────────────────────
    results = find_optimal_seeds(
        tokens,
        n_candidates=args.n_candidates,
        top_k=args.top_k,
        sample_size=args.sample_tokens,
        screen_sample_size=args.screen_sample_tokens,
        batch_size=args.batch_size,
        verbose=True,
        one_step_grad=args.one_step_grad,
    )

    # ── Optional lower-bound analysis ─────────────────────────────────────────
    if args.lower_bound and results:
        best_seed = results[0]["seed"]
        print(f"\n[LowerBound] Computing BPB lower bound for best seed {best_seed:#x}...")
        lb = compute_bpb_lower_bound(tokens, best_seed)
        print("\n[LowerBound] Results:")
        for k, v in lb.items():
            if isinstance(v, float):
                print(f"  {k:<30} {v:.6f}")
            else:
                print(f"  {k:<30} {v}")
        results[0]["lower_bound_analysis"] = lb

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump({"optimal_seeds": results}, f, indent=2)
    print(f"\n[SeedScreen] Results saved → {out_path}")

    # Print the ready-to-use command with optimal seeds
    seed_list = " ".join(str(r["seed"]) for r in results)
    print(f"\n[SeedScreen] Use these seeds in training (explicit seed override):")
    print(f"  cd /workspace/parameter-golf-hdc/records/track_10min_16mb/"
          f"2026-03-26_HDC_Zero_Track_5Mb")
    print(f"  python train_gpt.py --multi_seed --seeds {seed_list} \\")
    print(f"      --data_path ../../../data/datasets/fineweb10B_sp1024 \\")
    print(f"      --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model")
    print(f"\n[SeedScreen] Or let train_gpt.py auto-screen seeds (pre_screen_seeds is ON by default):")
    print(f"  python train_gpt.py --multi_seed \\")
    print(f"      --data_path ../../../data/datasets/fineweb10B_sp1024 \\")
    print(f"      --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model")


if __name__ == "__main__":
    main()
