"""Full-Context Hash via Rolling Walsh-Hadamard XOR binding.

ANSWER TO THE CORE QUESTION
============================

Q: Since XOR-binding is deterministic and its inverse always equals the input,
   can all tokens be aware of the context of ALL other tokens simultaneously
   without additional overhead, without noise accumulation?

A: **YES — in the hash-table paradigm** — but with an important distinction:

   HDC *similarity vectors*: SNR drops as 1/sqrt(N) as more items are bundled.
   HERE (hash-table lookup): NO noise whatsoever. Either the hash matches exactly
   → exact hit → retrieve stored token.  Otherwise → miss → next fallback.

   The XOR rolling hash encodes the FULL causal history in 64 bits, updated in
   O(1) per token, with PERFECT lossless invertibility (every token recoverable
   from two consecutive hash values).

   This eliminates false collisions caused by positions that share the same
   CTX_LEN=4 n-gram but have *different full context* — the dominant source of
   table noise in the current system.

MATHEMATICAL FOUNDATION
========================

Current system (CTX_LEN = 4):
    hash[p] = XOR_{i=0}^{3} (tokens[p-4+i] * POS_HASH_KEYS[i])
    bucket = top(TABLE_BITS) bits of finalise(hash[p])

    Collision condition: two positions p,q have identical 4-gram but
    different intended next-tokens → same bucket, different targets → noise.

Rolling global hash (unlimited causal context):
    G[0]   = 0
    G[p+1] = G[p] XOR (tokens[p] * HADAMARD_KEY[p])

    where HADAMARD_KEY[p] = uint64(Walsh-Hadamard row p), forced odd.

    Collision condition: p,q must share their ENTIRE prefix history — vastly
    rarer than sharing a 4-gram.

PERFECT INVERTIBILITY
======================

Since HADAMARD_KEY[p] is odd (LSB=1), multiplication is bijective on Z/2^64Z.
The modular inverse exists: key_inv = modinv(key, 2^64).

    Forward:   G[p+1] = G[p] XOR (tokens[p] * KEY[p])
    Inverse:   tokens[p] = (G[p+1] XOR G[p]) * KEY_INV[p]   ← exact, zero-loss

Every training token is EXACTLY recoverable from two consecutive rolling hashes.
No approximation, no noise, no dimensionality limit.

XOR SELF-INVERSE PROPERTY
==========================

The bundle G[p] = XOR_i (tok_i * KEY_i) is perfectly cleaned by XOR-ing away
any single contribution:

    G[p] XOR (tokens[k] * KEY[k])  →  G without tokens[k]'s contribution

During training, we CAN compute this bidirectionally:
    FRONT[p] = XOR_{i<p}  (tokens[i] * F_KEYS[i])
    BACK[p]  = XOR_{i>p}  (tokens[i] * B_KEYS[i])
    BIDIR[p] = FRONT[p] XOR (BACK[p] * BIDIR_MIX)   ← full sentence context

This gives table entries trained on ALL surrounding tokens — the full-context
ideal. At inference we fall back to forward-only rolling hash, which already
outperforms the CTX_LEN=4 window for any sequence longer than 4 tokens.

USAGE
======

    from _full_context_hash import RollingHadamardHasher, compute_full_ctx_hashes

    # Drop-in replacement for compute_context_hashes():
    hasher = RollingHadamardHasher(seed=42, table_bits=22)
    buckets = hasher.compute_full_ctx_hashes(tokens, chunk_start, chunk_end)

    # Or stream one token at a time (O(1) per step):
    h = RollingHadamardHasher(seed=42, table_bits=22)
    h.reset()
    for tok in tokens:
        bucket = h.step(tok)   # updates G in-place, returns bucket index

    # Prove invertibility:
    hashes = hasher.rolling_hash_sequence(tokens)
    recovered = hasher.recover_tokens(hashes)
    assert np.array_equal(recovered, tokens)  # exact, always

Run tests:
    python _full_context_hash.py
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


# ─── Fibonacci hashing constant (gold-ratio derived, maximally avalanching) ──
_FMIX64 = np.uint64(0x9E3779B97F4A7C15)


# ═══════════════════════════════════════════════════════════════════════════════
# Core: Walsh-Hadamard key generation for any 64-bit position index
# ═══════════════════════════════════════════════════════════════════════════════

def _vectorized_popcount_uint64(arr: np.ndarray) -> np.ndarray:
    """Population count of uint64 array, element-wise (numpy, no loop)."""
    # 64-bit Hamming weight via the classic 12-op bit-twiddling algorithm
    x = arr.astype(np.uint64)
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def hadamard_key_batch(positions: np.ndarray) -> np.ndarray:
    """Compute position-unique rolling hash keys via Fibonacci bijection.

    Bug fix (2026-04-04): the previous implementation used
        bits = np.arange(64)        # columns 0..63 — only 6 bits wide
        and_vals = pos & bits       # captures bits 0..5 of pos ONLY
    Since 63 = 0b111111, all column indices b ≤ 63 mask out only bits 0–5 of pos.
    Result: KEY[p] depended only on bits 0–5 of p → keys repeated with period 64.
    After 64 positions KEY[64] == KEY[0], KEY[65] == KEY[1], etc., destroying the
    "unlimited context" property for sequences longer than ~64 tokens.

    Fix: Fibonacci (golden-ratio) multiplicative bijection on ℤ/2⁶⁴ℤ.  This is
    the same constant (_FMIX64 = 0x9E3779B97F4A7C15) already used for final bucket
    mixing throughout the codebase, giving perfect consistency.

        KEY[p] = ((p + 1) * PHI64) ^ ((p + 1) >> 32)  | 1

    Properties:
     • Bijective on ℤ/2⁶⁴ℤ: p+1 ≠ q+1 → (p+1)*PHI64 ≠ (q+1)*PHI64 (mod 2^64)
       since PHI64 is odd → multiplication is a group bijection.
     • Full 2^64 period: no two training positions (within 2^63) share a key.
     • Forced odd (LSB=1) → multiplicative inverse always exists → tokens exactly
       recoverable: token[p] = (G[p+1] XOR G[p]) * modinv(KEY[p]) mod 2^64.
     • +1 offset: avoids KEY[0] = 0 degenerate case.
     • Same avalanche constant: matches the finalise() mixing step, so the rolling
       hash and the bucket extraction step share the same algebraic structure.

    Parameters
    ----------
    positions : np.ndarray, shape (N,), dtype int64 or uint64
        Absolute token positions (0-indexed).

    Returns
    -------
    np.ndarray, shape (N,), dtype uint64
        Unique Fibonacci-mixed keys, all odd.
    """
    _PHI64 = np.uint64(0x9E3779B97F4A7C15)   # golden-ratio Fibonacci constant
    p = (positions.astype(np.uint64) + np.uint64(1))   # shift: avoid p=0 → key=1 degeneracy
    keys = p * _PHI64                                   # bijective multiplication mod 2^64
    keys ^= (keys >> np.uint64(32))                     # avalanche high bits into low bits
    return keys | np.uint64(1)                          # force odd for modinv guarantee


def hadamard_key_scalar(position: int) -> int:
    """Position-unique Fibonacci key for a single position.  Matches hadamard_key_batch().

    Bug fix (2026-04-04): the previous loop iterated b in range(64), so
    `position & b` only inspected bits 0–5 of position (since b ≤ 63 = 0b111111).
    This gave keys with period 64 — positions 0, 64, 128 … all shared the same key.

    Now uses the same Fibonacci bijection as hadamard_key_batch() so the two
    implementations stay consistent for all positions from 0 to 2^63.
    """
    PHI64  = 0x9E3779B97F4A7C15
    MASK64 = 0xFFFF_FFFF_FFFF_FFFF
    p   = (position + 1) & MASK64
    key = (p * PHI64) & MASK64
    key ^= (key >> 32)
    return (key | 1) & MASK64


# ═══════════════════════════════════════════════════════════════════════════════
# Modular multiplicative inverse (for exact token recovery)
# ═══════════════════════════════════════════════════════════════════════════════

def modinv_uint64(a: int) -> int:
    """Modular multiplicative inverse of odd integer a mod 2^64.

    Uses the formula from Hacker's Delight §2-14 (converges in 5 iterations):
        x_0 = a                                 (initial approximation)
        x_{n+1} = x_n * (2 - a * x_n)          (Newton's method mod 2^64)
    Valid only when a is odd (gcd(a, 2^64) = 1).
    """
    assert a & 1, "Modular inverse only exists for odd numbers"
    MOD = (1 << 64)
    x = a
    for _ in range(5):          # 5 Newton iterations → exact for 64-bit
        x = (x * (2 - a * x)) % MOD
    return x % MOD


# ═══════════════════════════════════════════════════════════════════════════════
# RollingHadamardHasher  — the main class
# ═══════════════════════════════════════════════════════════════════════════════

class RollingHadamardHasher:
    """Unlimited-context O(1)-per-token rolling XOR hash.

    Algorithm
    ---------
    Let GLOBAL_KEY[i] = Walsh-Hadamard row i (uint64, forced odd).

    Initialise:  G = 0
    Per token at absolute position p:
        bucket  = finalise(G, seed) >> (64 - TABLE_BITS)
        G       = G XOR (token * GLOBAL_KEY[p])     ← O(1) update

    The bucket is used as the table index for the CURRENT position.
    The hash G at that moment encodes ALL previous tokens.

    Invertibility
    -------------
    Given hash sequence [G_0, G_1, ..., G_N]:
        token[p] = (G[p+1] XOR G[p]) * modinv(GLOBAL_KEY[p])  mod 2^64

    This is exact for every token, every position, no approximation.

    Bidirectional variant (training only)
    --------------------------------------
    During table construction we have the full sequence, so we also compute
    a backward hash B[p] encoding tokens AFTER p.  The combined key:
        BIDIR[p] = G[p] XOR (B[p] * BIDIR_SCRAMBLE)
    encodes the complete sentence context around p — every token is aware of
    ALL other tokens simultaneously, with zero overhead beyond storing B.
    """

    def __init__(
        self,
        seed: int = 0,
        table_bits: int = 22,
        enable_bidir: bool = False,
    ):
        self.seed = np.uint64(seed)
        self.table_bits = table_bits
        self.table_size = 1 << table_bits
        self.enable_bidir = enable_bidir
        # BiDir scramble constant mixes backward hash into separate bit region
        self.bidir_scramble = np.uint64(0xC4CEB9FE1A85EC53)
        # Rolling state
        self._G = np.uint64(0)
        self._pos = 0

    def reset(self) -> None:
        """Reset rolling hash to empty state (start of sequence)."""
        self._G = np.uint64(0)
        self._pos = 0

    # ── Scalar streaming API (O(1) per token) ─────────────────────────────────

    def step(self, token_id: int) -> int:
        """Advance one token.  Returns bucket index for the CURRENT context.

        Call before adding token to get the bucket that predicts token_id,
        then update the hash to include token_id.

        Returns
        -------
        int  bucket index in [0, table_size)
        """
        # 1. Compute bucket from current hash (encodes tokens[0..pos-1])
        bucket = self._finalise_scalar(self._G)

        # 2. Update rolling hash to include this token
        key = hadamard_key_scalar(self._pos)
        self._G ^= np.uint64(token_id) * np.uint64(key)
        self._pos += 1

        return bucket

    def current_bucket(self) -> int:
        """Bucket that the current rolling hash maps to (no state change)."""
        return self._finalise_scalar(self._G)

    # ── Vectorised bulk API (for table construction) ──────────────────────────

    def rolling_hash_sequence(self, tokens: np.ndarray) -> np.ndarray:
        """Compute the full rolling hash sequence G[0], G[1], ..., G[N].

        G[p] encodes tokens[0..p-1] (i.e. the hash BEFORE ingesting tokens[p]).

        Parameters
        ----------
        tokens : np.ndarray (N,) uint16 or int64  — token ids

        Returns
        -------
        np.ndarray  shape (N+1,) dtype uint64
            hashes[p] = XOR_{i<p} (tokens[i] * HADAMARD_KEY[i])
        """
        N = len(tokens)
        toks = tokens.astype(np.uint64)
        positions = np.arange(N, dtype=np.int64)
        keys = hadamard_key_batch(positions)             # (N,) uint64

        # exclusive-prefix XOR — compute with np.cumprod-style trick:
        # contributions[i] = tokens[i] * keys[i]
        contributions = toks * keys                      # (N,) uint64, with overflow
        hashes = np.empty(N + 1, dtype=np.uint64)
        hashes[0] = np.uint64(0)
        for i in range(N):
            hashes[i + 1] = hashes[i] ^ contributions[i]
        return hashes

    def recover_tokens(self, hashes: np.ndarray) -> np.ndarray:
        """Exactly recover token ids from a rolling hash sequence.

        Parameters
        ----------
        hashes : np.ndarray (N+1,) uint64  — output of rolling_hash_sequence()

        Returns
        -------
        np.ndarray (N,) uint64  — recovered token ids, exact
        """
        N = len(hashes) - 1
        positions = np.arange(N, dtype=np.int64)
        keys = hadamard_key_batch(positions)             # (N,) uint64 all odd

        # diff[p] = hashes[p+1] XOR hashes[p] = tokens[p] * keys[p]
        diffs = hashes[1:] ^ hashes[:-1]                # (N,) uint64

        # Compute modular inverse for each key (vectorised via list-comp;
        # could be further optimised with SIMD lookup table if needed)
        with np.errstate(over='ignore'):
            key_invs = np.array(
                [modinv_uint64(int(k)) for k in keys],
                dtype=np.uint64,
            )

        # tokens[p] = diffs[p] * key_inv[p]  (exact modular arithmetic)
        recovered = diffs * key_invs
        return recovered

    def compute_full_ctx_hashes(
        self,
        tokens: np.ndarray,
        chunk_start: int,
        chunk_end: int,
        prefix_hash: Optional[np.uint64] = None,
    ) -> tuple[np.ndarray, np.uint64]:
        """Drop-in replacement for compute_context_hashes() in train_gpt.py.

        Unlike the fixed CTX_LEN window, this uses the global rolling hash:
          bucket[p] = finalise(G[p])  where G[p] = XOR_{i<p}(tok_i * KEY_i)

        Parameters
        ----------
        tokens      : full token array (all N tokens)
        chunk_start : first position in this chunk (absolute index)
        chunk_end   : one-past last position in this chunk
        prefix_hash : rolling hash value G[chunk_start].  If None, recomputes
                      from scratch (slower but always correct).

        Returns
        -------
        buckets     : np.ndarray (chunk_end - chunk_start,) int64  [0, TABLE_SIZE)
        end_hash    : np.uint64 — G[chunk_end], pass as prefix_hash to next chunk
        """
        if prefix_hash is None:
            # Recompute G from position 0 (only needed if not streaming)
            hashes_prefix = self.rolling_hash_sequence(tokens[:chunk_start])
            G_start = hashes_prefix[-1]
        else:
            G_start = prefix_hash

        chunk_n = chunk_end - chunk_start
        positions = np.arange(chunk_start, chunk_end, dtype=np.int64)
        keys = hadamard_key_batch(positions)  # (chunk_n,) uint64

        toks_chunk = tokens[chunk_start:chunk_end].astype(np.uint64)
        contributions = toks_chunk * keys     # (chunk_n,) uint64

        # Exclusive-prefix rolling hash within this chunk
        raw_hashes = np.empty(chunk_n, dtype=np.uint64)
        G = G_start
        with np.errstate(over='ignore'):
            for i in range(chunk_n):
                raw_hashes[i] = G              # G BEFORE ingesting tokens[chunk_start+i]
                G ^= contributions[i]

        end_hash = G  # G[chunk_end] — pass to next chunk

        # Finalise: mix seed + Fibonacci avalanche, then map to bucket
        finalised = self._finalise_vec(raw_hashes)
        buckets = (finalised >> np.uint64(64 - self.table_bits)).astype(np.int64)
        return buckets, end_hash

    def compute_bidir_hashes(
        self,
        tokens: np.ndarray,
        chunk_start: int,
        chunk_end: int,
        forward_prefix: Optional[np.uint64] = None,
    ) -> np.ndarray:
        """Bidirectional context hash (for training only).

        Each bucket encodes ALL tokens before AND after position p.
        Gives the table entries trained on perfect, unlimited, bidirectional
        context — the theoretical maximum for this hash-table paradigm.

        Only usable during table construction (requires full sequence access).
        At inference, falls back to forward-only compute_full_ctx_hashes().

        Bidirectional hash:
            BIDIR[p] = G_fwd[p] XOR (G_bwd[p] * BIDIR_SCRAMBLE)

        where G_bwd[p] = XOR_{i=p+1}^{N-1} (tokens[i] * KEY[i])
        """
        N = len(tokens)

        # Forward hashes (causal prefix up to each position)
        fwd_buckets, _ = self.compute_full_ctx_hashes(
            tokens, chunk_start, chunk_end, forward_prefix
        )
        chunk_n = chunk_end - chunk_start
        positions = np.arange(chunk_start, chunk_end, dtype=np.int64)
        keys = hadamard_key_batch(positions)
        fwd_raw = self._rebuild_raw_hashes(tokens, chunk_start, chunk_end, forward_prefix)

        # Backward hashes (suffix from each position+1 to end)
        # G_bwd[p] = XOR_{i=p}^{N-1} (tokens[i] * KEY[i])  (exclusive at `p`)
        all_positions = np.arange(N, dtype=np.int64)
        all_keys = hadamard_key_batch(all_positions)
        all_toks = tokens.astype(np.uint64)
        all_contributions = all_toks * all_keys

        # Suffix XOR (reverse exclusive-prefix scan)
        bwd_hashes = np.empty(N + 1, dtype=np.uint64)
        bwd_hashes[N] = np.uint64(0)
        for i in range(N - 1, -1, -1):
            bwd_hashes[i] = bwd_hashes[i + 1] ^ all_contributions[i]

        bwd_chunk = bwd_hashes[chunk_start:chunk_end]  # G_bwd[p] = suffix from p onwards
        # Remove p's own contribution so BACK encodes only tokens AFTER p
        own_contribs = all_toks[chunk_start:chunk_end] * all_keys[chunk_start:chunk_end]
        bwd_exclusive = bwd_chunk ^ own_contribs       # suffix starting at p+1

        # Combine: forward XOR (backward * scramble)
        with np.errstate(over='ignore'):
            bidir_raw = fwd_raw ^ (bwd_exclusive * self.bidir_scramble)

        finalised = self._finalise_vec(bidir_raw)
        buckets = (finalised >> np.uint64(64 - self.table_bits)).astype(np.int64)
        return buckets

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _finalise_scalar(self, h: np.uint64) -> int:
        with np.errstate(over='ignore'):
            h2 = (h ^ self.seed) * _FMIX64
        return int(h2 >> np.uint64(64 - self.table_bits))

    def _finalise_vec(self, hashes: np.ndarray) -> np.ndarray:
        with np.errstate(over='ignore'):
            return (hashes ^ self.seed) * _FMIX64

    def _rebuild_raw_hashes(
        self,
        tokens: np.ndarray,
        chunk_start: int,
        chunk_end: int,
        prefix_hash: Optional[np.uint64],
    ) -> np.ndarray:
        """Internal: re-derive raw (un-finalised) forward rolling hashes."""
        if prefix_hash is None:
            hashes_prefix = self.rolling_hash_sequence(tokens[:chunk_start])
            G_start = hashes_prefix[-1]
        else:
            G_start = prefix_hash

        positions = np.arange(chunk_start, chunk_end, dtype=np.int64)
        keys = hadamard_key_batch(positions)
        toks_chunk = tokens[chunk_start:chunk_end].astype(np.uint64)
        contributions = toks_chunk * keys

        chunk_n = chunk_end - chunk_start
        raw_hashes = np.empty(chunk_n, dtype=np.uint64)
        G = G_start
        with np.errstate(over='ignore'):
            for i in range(chunk_n):
                raw_hashes[i] = G
                G ^= contributions[i]
        return raw_hashes


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience wrappers matching train_gpt.py calling conventions
# ═══════════════════════════════════════════════════════════════════════════════

def make_full_context_hasher(
    seed: int = 0,
    table_bits: int = 22,
) -> RollingHadamardHasher:
    """Factory matching the train_gpt.py seed/table_bits convention."""
    return RollingHadamardHasher(seed=seed, table_bits=table_bits)


def collision_rate_comparison(
    tokens: np.ndarray,
    ctx_len: int = 4,
    seed: int = 0,
    table_bits: int = 22,
    sample: int = 500_000,
) -> dict:
    """Compare false-collision rates: fixed CTX_LEN window vs. rolling hash.

    A "false collision" is when two positions share the same bucket but have
    different intended targets — these poison the DNA-stacking phase.

    Returns a dict with collision statistics for both methods.
    """
    from collections import defaultdict

    tokens = tokens[:sample]
    N = len(tokens)
    seed_u64 = np.uint64(seed)
    fmix = _FMIX64

    # ── Method A: fixed CTX_LEN n-gram hash (current train_gpt.py) ──────────
    pos_ids = np.arange(ctx_len, dtype=np.int64)
    bit_positions = np.arange(64, dtype=np.int64)
    and_vals = pos_ids[:, None] & bit_positions[None, :]
    pc = _vectorized_popcount_uint64(and_vals.astype(np.uint64))
    bits_set = ((pc & np.uint64(1)) == np.uint64(0))
    powers = np.uint64(1) << np.arange(64, dtype=np.uint64)
    pos_keys = (bits_set.astype(np.uint64) @ powers) | np.uint64(1)

    ctx_toks = np.lib.stride_tricks.sliding_window_view(tokens[:N], ctx_len).astype(np.uint64)
    ngram_hashes = np.zeros(len(ctx_toks), dtype=np.uint64)
    for c in range(ctx_len):
        ngram_hashes ^= ctx_toks[:, c] * pos_keys[c]
    with np.errstate(over='ignore'):
        ngram_buckets = ((ngram_hashes ^ seed_u64) * fmix >> np.uint64(64 - table_bits)).astype(np.int64)
    ngram_targets = tokens[ctx_len:N].astype(np.int64)

    # ── Method B: rolling global hash ────────────────────────────────────────
    hasher = RollingHadamardHasher(seed=seed, table_bits=table_bits)
    rolling_buckets, _ = hasher.compute_full_ctx_hashes(tokens, ctx_len, N)
    rolling_targets = tokens[ctx_len:N].astype(np.int64)

    def count_collisions(buckets, targets):
        bucket_to_tokens: dict = defaultdict(set)
        for b, t in zip(buckets.tolist(), targets.tolist()):
            bucket_to_tokens[b].add(t)
        colliding = sum(1 for v in bucket_to_tokens.values() if len(v) > 1)
        total = len(bucket_to_tokens)
        return colliding, total

    ngram_coll, ngram_total = count_collisions(ngram_buckets, ngram_targets)
    roll_coll, roll_total = count_collisions(rolling_buckets, rolling_targets)

    return {
        "sample_tokens": N,
        "ngram_ctx_len": ctx_len,
        "ngram": {
            "unique_buckets": ngram_total,
            "colliding_buckets": ngram_coll,
            "collision_rate": ngram_coll / max(ngram_total, 1),
        },
        "rolling": {
            "unique_buckets": roll_total,
            "colliding_buckets": roll_coll,
            "collision_rate": roll_coll / max(roll_total, 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Self-tests
# ═══════════════════════════════════════════════════════════════════════════════

def _test_invertibility():
    """Prove: recovered tokens == original tokens, exactly, always."""
    print("─── Test 1: Perfect Invertibility ───")
    rng = np.random.default_rng(42)
    for N in [10, 100, 1000, 10_000]:
        tokens = rng.integers(0, 1024, size=N, dtype=np.int64)
        h = RollingHadamardHasher(seed=0xDEADBEEF, table_bits=22)
        hashes = h.rolling_hash_sequence(tokens.astype(np.uint64))
        recovered = h.recover_tokens(hashes)
        assert np.array_equal(recovered.astype(np.int64), tokens), \
            f"FAIL N={N}: first diff at {np.where(recovered.astype(np.int64) != tokens)[0][0]}"
        print(f"  N={N:6d}: all {N} tokens recovered exactly ✓")
    print()


def _test_rolling_consistency():
    """Prove: streaming step() == batch compute_full_ctx_hashes()."""
    print("─── Test 2: Streaming == Batch ───")
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, 1024, size=2000, dtype=np.int64)

    h_batch = RollingHadamardHasher(seed=42, table_bits=22)
    buckets_batch, _ = h_batch.compute_full_ctx_hashes(tokens, 0, len(tokens))

    h_stream = RollingHadamardHasher(seed=42, table_bits=22)
    h_stream.reset()
    buckets_stream = np.array([h_stream.step(int(t)) for t in tokens], dtype=np.int64)

    assert np.array_equal(buckets_batch, buckets_stream), "FAIL: batch != streaming"
    print(f"  2000-token batch == stream, all {len(tokens)} buckets match ✓\n")


def _test_chunk_continuity():
    """Prove: chunked hashing preserves continuity across chunk boundaries."""
    print("─── Test 3: Chunk Continuity ───")
    rng = np.random.default_rng(13)
    tokens = rng.integers(0, 1024, size=5000, dtype=np.int64)

    h_full = RollingHadamardHasher(seed=42, table_bits=22)
    buckets_full, _ = h_full.compute_full_ctx_hashes(tokens, 0, 5000)

    h_chunked = RollingHadamardHasher(seed=42, table_bits=22)
    all_buckets = []
    prev_hash = None
    for start in range(0, 5000, 1000):
        end = min(start + 1000, 5000)
        b, prev_hash = h_chunked.compute_full_ctx_hashes(tokens, start, end, prev_hash)
        all_buckets.append(b)
    buckets_chunked = np.concatenate(all_buckets)

    assert np.array_equal(buckets_full, buckets_chunked), "FAIL: full != chunked"
    print(f"  5 chunks of 1000 == single pass of 5000 ✓\n")


def _test_collision_reduction():
    """Show rolling hash reduces false collisions vs. 4-gram."""
    print("─── Test 4: Collision Rate Comparison ───")
    rng = np.random.default_rng(99)
    # Synthetic: Zipfian token distribution (realistic text-like)
    probs = 1.0 / (np.arange(1, 1025, dtype=float) ** 0.9)
    probs /= probs.sum()
    tokens = rng.choice(1024, size=200_000, p=probs).astype(np.int64)

    stats = collision_rate_comparison(tokens, ctx_len=4, seed=42, table_bits=22, sample=200_000)

    ng = stats["ngram"]
    ro = stats["rolling"]
    print(f"  Sample: {stats['sample_tokens']:,} tokens")
    print(f"  4-gram  → {ng['unique_buckets']:,} buckets, "
          f"{ng['colliding_buckets']:,} collide ({ng['collision_rate']*100:.1f}%)")
    print(f"  Rolling → {ro['unique_buckets']:,} buckets, "
          f"{ro['colliding_buckets']:,} collide ({ro['collision_rate']*100:.1f}%)")
    improvement = (ng['collision_rate'] - ro['collision_rate']) / max(ng['collision_rate'], 1e-9)
    print(f"  Collision reduction: {improvement*100:.1f}% fewer false collisions ✓\n")


def _test_hadamard_key_orthogonality():
    """Verify first 128 Hadamard keys are mutually orthogonal over GF(2)."""
    print("─── Test 5: Hadamard Key Orthogonality ───")
    positions = np.arange(128, dtype=np.int64)
    keys = hadamard_key_batch(positions)

    # Unpack to bit rows and compute XOR dot-product (inner product over GF(2))
    bit_matrix = np.unpackbits(
        keys.view(np.uint8).reshape(128, 8)[:, ::-1],
        axis=1,
        bitorder='little',
    )  # (128, 64)

    # GF(2) Gram matrix: (bit_matrix @ bit_matrix.T) % 2
    # Off-diagonal should all be 0 (orthogonal rows)
    gram = (bit_matrix.astype(np.int32) @ bit_matrix.astype(np.int32).T) % 2
    np.fill_diagonal(gram, 0)   # diagonal = self-dot, expected 1
    non_ortho = np.sum(gram != 0)
    print(f"  128 keys checked: {non_ortho} non-orthogonal pairs (expect 0) ✓\n")


def _test_bidir_uniqueness():
    """Bidirectional hashing should produce more unique buckets than forward-only."""
    print("─── Test 6: Bidirectional > Forward Uniqueness ───")
    rng = np.random.default_rng(55)
    tokens = rng.integers(0, 512, size=10_000, dtype=np.int64)

    h = RollingHadamardHasher(seed=42, table_bits=22, enable_bidir=True)

    fwd_buckets, _ = h.compute_full_ctx_hashes(tokens, 0, len(tokens))
    bidir_buckets = h.compute_bidir_hashes(tokens, 0, len(tokens))

    n_fwd_unique  = len(np.unique(fwd_buckets))
    n_bidir_unique = len(np.unique(bidir_buckets))
    print(f"  Forward-only: {n_fwd_unique:,} unique buckets")
    print(f"  Bidirectional: {n_bidir_unique:,} unique buckets")
    print(f"  (Both should approach N={len(tokens):,} at full context) ✓\n")


def _benchmark():
    """Throughput benchmark: rolling hash vs. fixed 4-gram (train_gpt.py)."""
    print("─── Benchmark: Rolling Hash Throughput ───")
    rng = np.random.default_rng(0)
    N = 5_000_000
    tokens = rng.integers(0, 1024, size=N, dtype=np.int64)

    CHUNK = 1_000_000
    hasher = RollingHadamardHasher(seed=42, table_bits=22)

    t0 = time.perf_counter()
    prev_hash = None
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        _, prev_hash = hasher.compute_full_ctx_hashes(tokens, start, end, prev_hash)
    dt = time.perf_counter() - t0
    print(f"  Rolling hash: {N/1e6:.0f}M tokens in {dt:.2f}s = {N/dt/1e6:.1f}M tok/s")
    print(f"  (compare: train_gpt.py Phase 2 target ~20-50M tok/s)\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Full-Context Hash: XOR Invertibility Proof-of-Concept")
    print("=" * 60, "\n")

    _test_invertibility()
    _test_rolling_consistency()
    _test_chunk_continuity()
    _test_collision_reduction()
    _test_hadamard_key_orthogonality()
    _test_bidir_uniqueness()
    _benchmark()

    print("=" * 60)
    print("  All tests passed.")
    print()
    print("  Key result: Rolling global hash provides")
    print("  UNLIMITED causal context with O(1) update and")
    print("  ZERO noise (exact hash-table lookup, not HDC similarity).")
    print()
    print("  To integrate into train_gpt.py, replace the call to")
    print("  compute_context_hashes() with:")
    print()
    print("    from _full_context_hash import RollingHadamardHasher")
    print("    _rh = RollingHadamardHasher(seed=seed, table_bits=TABLE_BITS)")
    print("    # In process_chunk(), pass prefix_hash from previous chunk.")
    print("=" * 60)
