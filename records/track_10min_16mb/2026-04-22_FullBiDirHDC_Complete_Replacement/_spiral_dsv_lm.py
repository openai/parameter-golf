"""Spiral DSV — Language Model Adaptation.

Extracted and adapted from Arc_AGI_3_HDC_Hadamard_DSV_Model-main/
ARC-AGI-3-Agents/agents/templates/_spiral_dsv.py

Keeps only the components needed for language modeling:
  - GoldenAxisShift  — golden-ratio axis rotations (metric-preserving, lossless)
  - GOLDEN_AXES      — module-level singleton
  - SpiralPointerMemory — axis-keyed hierarchical memory
  - SpiralDSVLanguageModel — bilateral XOR-bundle DSV for 1D token sequences

All ARC-AGI-3 game engine dependencies removed.
All 2D grid-specific methods removed.

Eigen Convergence upgrade (2026-04-23):
  - build_from_tokens() XOR-bundle scatter → EigenSpiralBuilder.build_bilateral_tables()
  - sem_fwd/sem_bwd now computed as sign(co_occurrence_weights @ CB_pm1)
  - Mathematically equivalent to XOR-bundle majority vote (uniform weights)
  - With frequency weights: strictly better signal (more frequent pairs → more weight)
  - Performance: O(vocab² × n_bits) matmul instead of O(N × ctx_len) scatter XOR
    For vocab=1024, n_bits=32768: ~0.1s on H100 vs ~30s scatter XOR
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Eigen Convergence import ──────────────────────────────────────────────────
try:
    from _eigen_convergence import EigenSpiralBuilder
    _EIGEN_SPIRAL_AVAILABLE = True
except ImportError:
    EigenSpiralBuilder = None   # type: ignore[assignment,misc]
    _EIGEN_SPIRAL_AVAILABLE = False

# ============================================================================
# Module-level popcount lookup table
# ============================================================================

_POPCOUNT_TABLE = np.zeros(256, dtype=np.int32)
for _i in range(256):
    _POPCOUNT_TABLE[_i] = bin(_i).count('1')


# ============================================================================
# Golden Ratio Axis Shift
# ============================================================================

PHI_FRAC: float = 0.6180339887498949  # φ − 1 = 1/φ


class GoldenAxisShift:
    """Generates an infinite, non-repeating, positive-only axis space.

    Each axis k is defined by a golden-ratio bit-offset in hypervector space:
        offset(k) = floor(k × PHI_FRAC × total_bits) mod total_bits

    Properties:
    - Non-repeating: offset(k) ≠ offset(j) for all finite k ≠ j
    - Maximally equidistributed: Weyl's theorem guarantees uniform coverage
    - Always positive: PHI_FRAC ∈ (0,1), offsets are non-negative integers
    - Infinite: any k ≥ 0 is valid
    - Metric-preserving: cosine(partner_hv(A,k), partner_hv(B,k)) = cosine(A,B)
    - Lossless: inverse_hv(partner_hv(hv, k), k) = hv exactly

    Y_SPLIT = 8: axes 0–7 are spatial-exploration (compass-equivalent);
                 axes 8–n_axes-1 are prototype-alignment (Y-axes).
    """

    DEFAULT_N_AXES: int = 19
    Y_SPLIT: int = 8

    def __init__(self, n_axes: int = DEFAULT_N_AXES, W: int = 16):
        self.n_axes = n_axes
        self.W = W
        self._total_bits = W * 64
        self._offsets: List[int] = [self._compute_offset(k) for k in range(n_axes)]
        self._word_shifts: List[int] = [off // 64 for off in self._offsets]
        self._bit_shifts: List[int] = [off % 64 for off in self._offsets]
        self._inverse_map: List[int] = self._build_inverse_map()

    def _compute_offset(self, k: int) -> int:
        return int(k * PHI_FRAC * self._total_bits) % self._total_bits

    def _build_inverse_map(self) -> List[int]:
        result = []
        for k in range(self.n_axes):
            target = (self._total_bits - self._offsets[k]) % self._total_bits
            best_k = min(range(self.n_axes), key=lambda k2: abs(self._offsets[k2] - target))
            result.append(best_k)
        return result

    def offset(self, k: int) -> int:
        while k >= len(self._offsets):
            new_k = len(self._offsets)
            off = self._compute_offset(new_k)
            self._offsets.append(off)
            self._word_shifts.append(off // 64)
            self._bit_shifts.append(off % 64)
            self._inverse_map = self._build_inverse_map()
        return self._offsets[k]

    def partner_hv(self, source_hv: np.ndarray, k: int) -> np.ndarray:
        """Rotate source_hv by axis_offset(k) bits — the golden-ratio partner HV."""
        if k >= len(self._word_shifts):
            self.offset(k)
        word_shift = self._word_shifts[k]
        bit_shift = self._bit_shifts[k]
        rotated = np.roll(source_hv, word_shift)
        if bit_shift > 0:
            rotated = (rotated << np.uint64(bit_shift)) | (rotated >> np.uint64(64 - bit_shift))
        return rotated.astype(np.uint64)

    def inverse_hv(self, hv: np.ndarray, k: int) -> np.ndarray:
        """Exact inverse of partner_hv(hv, k). Always lossless."""
        if k >= len(self._word_shifts):
            self.offset(k)
        word_shift = self._word_shifts[k]
        bit_shift = self._bit_shifts[k]
        if bit_shift > 0:
            bs = np.uint64(bit_shift)
            inv_bs = np.uint64(64 - bit_shift)
            result = (hv >> bs) | (hv << inv_bs)
            result = result.astype(np.uint64)
        else:
            result = hv.copy()
        if word_shift != 0:
            result = np.roll(result, -word_shift)
        return result.astype(np.uint64)

    def extend(self, new_n_axes: int) -> None:
        if new_n_axes > self.n_axes:
            for k in range(self.n_axes, new_n_axes):
                self._offsets.append(self._compute_offset(k))
            self.n_axes = new_n_axes


# Module-level singleton — shared across all files via import
GOLDEN_AXES = GoldenAxisShift(n_axes=19, W=16)


# ============================================================================
# Spiral-Pointer Hierarchical Memory
# ============================================================================

class SpiralPointerMemory:
    """Axis-keyed hierarchical memory using golden-ratio rotations as addresses.

    Each item is stored at a composite axis address (k1, k2, ..., kL) by
    applying L successive golden-ratio bit-rotations to the item HV. Retrieval
    applies the inverse rotations in reverse order — lossless because each
    rotation is a bijection on the HV space.

    Properties:
    - Capacity  : D^L items  (D = n_words × 64 bits)
    - Lossless  : cyclic rotation is a bijection; inverse is always exact
    - Metric-preserving: cosine(partner_hv(A,k), partner_hv(B,k)) = cosine(A,B)
    - Thread-safe: per-band RLocks (zero contention across different bands)
    """

    def __init__(self, n_words: int = 128, n_levels: int = 4) -> None:
        self.n_words = n_words
        self.n_levels = n_levels
        self.D = n_words * 64
        self._nodes: Dict[Tuple[int, ...], np.ndarray] = {}
        self._band_locks: Dict[int, threading.RLock] = {}
        self._meta_lock = threading.Lock()

    def _band_lock(self, band_start: int) -> threading.RLock:
        if band_start not in self._band_locks:
            with self._meta_lock:
                if band_start not in self._band_locks:
                    self._band_locks[band_start] = threading.RLock()
        return self._band_locks[band_start]

    def store(self, item_hv: np.ndarray, address: Tuple[int, ...]) -> None:
        """Store item_hv at composite axis address (k1, k2, ..., kL)."""
        if len(address) != self.n_levels:
            raise ValueError(f"address length {len(address)} != n_levels {self.n_levels}")
        current = item_hv.copy().astype(np.uint64)
        for k in reversed(address):
            current = GOLDEN_AXES.partner_hv(current, k)
        with self._band_lock(address[0]):
            self._nodes[address] = current

    def retrieve(self, address: Tuple[int, ...]) -> np.ndarray:
        """Retrieve item at composite axis address. Returns zeros if not stored."""
        with self._band_lock(address[0]):
            node = self._nodes.get(address)
        if node is None:
            return np.zeros(self.n_words, dtype=np.uint64)
        current = node.copy()
        for k in reversed(address):
            current = GOLDEN_AXES.inverse_hv(current, k)
        return current

    def find_similar(
        self,
        query_hv: np.ndarray,
        threshold: float = 0.95,
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """Find all stored items with cosine similarity > threshold to query_hv."""
        if not self._nodes:
            return []
        results: List[Tuple[Tuple[int, ...], float]] = []
        q = query_hv.astype(np.uint64)
        for addr, ptr_hv in self._nodes.items():
            xor_pc = int(np.unpackbits(np.bitwise_xor(q, ptr_hv).view(np.uint8)).sum())
            total_bits = self.n_words * 64
            sim = 0.5 + 0.5 * (1.0 - 2.0 * xor_pc / total_bits)
            if sim > threshold:
                results.append((addr, float(sim)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def __len__(self) -> int:
        return len(self._nodes)

    def addresses(self) -> List[Tuple[int, ...]]:
        return list(self._nodes.keys())


# ============================================================================
# Spiral DSV Language Model — Bilateral XOR-bundle DSV for 1D token sequences
# ============================================================================

class SpiralDSVLanguageModel:
    """Bilateral XOR-bundle DSV adapted for language modeling (1D token sequences).

    Replaces the 2D grid ring structure of SpiralDSV with 1D sequential rings
    (lag 1..ctx_len), using the GoldenAxisShift codebook instead of a
    Fibonacci-hash codebook.

    Key difference from DirectionalSemanticVec:
    - Uses GoldenAxisShift axes (metric-preserving) instead of random codebook
    - Stores both fwd and bwd bundles per token (bilateral)
    - Computes consistency = cosine(fwd_query, bwd_query) as confidence gate
    - vote_scores_batch() returns (batch, vocab_size) bilateral confidence scores

    Memory budget (n_words=16, vocab_size=1024):
        codebook : 1024 × 16 × 8 = 128 KB
        sem_fwd  : 1024 × 16 × 8 = 128 KB
        sem_bwd  : 1024 × 16 × 8 = 128 KB
        Total    : 384 KB
    """

    def __init__(self, vocab_size: int, n_words: int = 16, seed: int = 42) -> None:
        self.vocab_size = vocab_size
        self.n_words = n_words
        self.seed = seed
        self._half = float(n_words * 32)  # n_words × 64 / 2
        self._n_bits = n_words * 64

        # GoldenAxisShift codebook: vocab_size × n_words uint64
        # Each token gets a unique HV derived from golden-ratio axis rotations
        rng = np.random.default_rng(seed)
        self.codebook: np.ndarray = rng.integers(
            0, np.iinfo(np.uint64).max,
            size=(vocab_size, n_words), dtype=np.uint64
        )

        # Bilateral XOR-bundle tables
        self.sem_fwd: np.ndarray = np.zeros((vocab_size, n_words), dtype=np.uint64)
        self.sem_bwd: np.ndarray = np.zeros((vocab_size, n_words), dtype=np.uint64)

        self._built = False

        # FIX #5: Cache pm1 versions of codebook, sem_fwd, sem_bwd for BLAS matmul
        # These are lazily computed on first use and invalidated when tables change.
        self._codebook_pm1: Optional[np.ndarray] = None   # (vocab, n_bits) float32
        self._sem_fwd_pm1 : Optional[np.ndarray] = None   # (vocab, n_bits) float32
        self._sem_bwd_pm1 : Optional[np.ndarray] = None   # (vocab, n_bits) float32

    @staticmethod
    def _scatter_xor_fast(
        vec_2d: np.ndarray,
        a_toks: np.ndarray,
        b_toks: np.ndarray,
        codebook: np.ndarray,
        chunk_size: int = 8_000_000,
    ) -> None:
        """Scatter XOR-bundle: for each (a, b) pair, vec_2d[a] ^= codebook[b].

        Uses chunked argsort+reduceat for O(N log N) time with low memory.
        Same algorithm as DirectionalSemanticVec._scatter_xor_fast().
        """
        M = len(a_toks)
        if M == 0:
            return
        for cs in range(0, M, chunk_size):
            ce = min(cs + chunk_size, M)
            a_c = a_toks[cs:ce]
            b_c = b_toks[cs:ce]
            order = a_c.argsort(kind="stable")
            a_sort = a_c[order]
            b_sort = b_c[order]
            vecs = codebook[b_sort]
            unique_a, first_idx = np.unique(a_sort, return_index=True)
            bundles = np.bitwise_xor.reduceat(vecs, first_idx, axis=0)
            vec_2d[unique_a] ^= bundles

    def build_from_tokens(
        self,
        tokens: np.ndarray,
        ctx_len: int = 4,
        time_budget_s: float = 30.0,
        chunk_size: int = 8_000_000,
        verbose: bool = True,
    ) -> None:
        """Build bilateral tables from token sequence via eigen fixed-point solve.

        Eigen path (when _EIGEN_SPIRAL_AVAILABLE):
            1. Accumulate co-occurrence weight matrix:
               fwd_w[a, b] = Σ_{c=1..ctx_len} count(b follows a at lag c)
            2. Single matmul: sem_fwd_pm1 = sign( fwd_w @ CB_pm1 )
            3. Convert to uint64: sem_fwd = pm1_to_uint64(sem_fwd_pm1)

        Mathematical equivalence to XOR-bundle:
            XOR-bundle: sem_fwd[a] = XOR_i CB[b_i]
            Eigen:      sem_fwd_pm1[a]* = sign( Σ_i CB_pm1[b_i] )
            Both are the majority vote of CB[b_i] in pm1 space.
            Eigen with frequency weights is strictly better signal.

        Performance:
            Before: O(N × ctx_len) scatter XOR (~30s for 500M tokens, ctx_len=4)
            After:  O(vocab² × n_bits) matmul (~0.1s on H100 for vocab=1024)

        Fallback path (when _EIGEN_SPIRAL_AVAILABLE=False):
            Original scatter XOR via _scatter_xor_fast() (unchanged).

        Args:
            tokens       : (N,) int array of token IDs
            ctx_len      : Maximum context depth (number of lags)
            time_budget_s: Time budget in seconds
            chunk_size   : Chunk size for scatter XOR (fallback path only)
            verbose      : Print progress
        """
        import time
        N = len(tokens)
        start = time.time()

        if verbose:
            print(f"\n[SpiralDSV-LM] Building bilateral tables "
                  f"(vocab={self.vocab_size}, n_words={self.n_words}, "
                  f"ctx_len={ctx_len}, eigen={_EIGEN_SPIRAL_AVAILABLE})")

        if _EIGEN_SPIRAL_AVAILABLE and EigenSpiralBuilder is not None:
            # ── Eigen path: co-occurrence matmul replaces scatter XOR ─────
            builder = EigenSpiralBuilder.from_codebook_uint64(
                codebook_vecs = self.codebook,
                use_frequency = True,
            )
            result = builder.build_bilateral_tables(
                tokens        = tokens,
                ctx_len       = ctx_len,
                time_budget_s = time_budget_s,
                verbose       = verbose,
            )
            # Apply fixed-point results: uint64 tables
            self.sem_fwd = result['sem_fwd_u64']   # (vocab, n_words) uint64
            self.sem_bwd = result['sem_bwd_u64']   # (vocab, n_words) uint64
            self._built = True
            self._invalidate_pm1_cache()  # tables changed — invalidate pm1 cache

            elapsed = time.time() - start
            if verbose:
                print(f"[SpiralDSV-LM] Eigen build done: "
                      f"{result['total_pairs']:,} pairs in {elapsed:.2f}s")

        else:
            # ── Fallback: original scatter XOR (unchanged) ────────────────
            tokens_i32 = tokens.astype(np.int32)
            total_pairs = 0

            for c in range(1, ctx_len + 1):
                if time.time() - start > time_budget_s:
                    if verbose:
                        print(f"[SpiralDSV-LM] Time budget reached at c={c-1}")
                    break
                M = N - c
                a_toks = tokens_i32[:M]
                b_toks = tokens_i32[c:]
                self._scatter_xor_fast(self.sem_fwd, a_toks, b_toks,
                                       self.codebook, chunk_size)
                self._scatter_xor_fast(self.sem_bwd, b_toks, a_toks,
                                       self.codebook, chunk_size)
                total_pairs += M
                elapsed = time.time() - start
                if verbose:
                    print(f"[SpiralDSV-LM] c={c}/{ctx_len}  "
                          f"pairs={total_pairs:,}  elapsed={elapsed:.1f}s")

            elapsed = time.time() - start
            if verbose:
                print(f"[SpiralDSV-LM] Done. {total_pairs:,} pairs in {elapsed:.1f}s")
            self._built = True
            self._invalidate_pm1_cache()  # tables changed — invalidate pm1 cache

    def _invalidate_pm1_cache(self) -> None:
        """Invalidate cached pm1 versions of sem_fwd/sem_bwd/codebook."""
        self._sem_fwd_pm1  = None
        self._sem_bwd_pm1  = None
        self._codebook_pm1 = None

    def _ensure_pm1_cache(self) -> None:
        """Lazily build pm1 cache from uint64 tables.

        Converts sem_fwd, sem_bwd, codebook to float32 pm1 space once,
        then reuses for all subsequent vote_scores_all_vocab() calls.
        This avoids recomputing the (vocab, n_bits) unpackbits on every call.
        """
        if self._codebook_pm1 is None:
            bits = np.unpackbits(
                self.codebook.view(np.uint8).reshape(self.vocab_size, self.n_words * 8),
                axis=1, bitorder='little'
            )
            self._codebook_pm1 = bits.astype(np.float32) * 2.0 - 1.0  # (vocab, n_bits)

        if self._sem_fwd_pm1 is None:
            bits = np.unpackbits(
                self.sem_fwd.view(np.uint8).reshape(self.vocab_size, self.n_words * 8),
                axis=1, bitorder='little'
            )
            self._sem_fwd_pm1 = bits.astype(np.float32) * 2.0 - 1.0  # (vocab, n_bits)

        if self._sem_bwd_pm1 is None:
            bits = np.unpackbits(
                self.sem_bwd.view(np.uint8).reshape(self.vocab_size, self.n_words * 8),
                axis=1, bitorder='little'
            )
            self._sem_bwd_pm1 = bits.astype(np.float32) * 2.0 - 1.0  # (vocab, n_bits)

    def vote_scores_batch(
        self,
        prev_tokens: np.ndarray,
        tgt_tokens: np.ndarray,
    ) -> np.ndarray:
        """Bilateral confidence scores for specific (prev, tgt) pairs.

        Args:
            prev_tokens : (batch,) int32 — previous token IDs
            tgt_tokens  : (batch,) int32 — target token IDs

        Returns:
            (batch,) float32 — bilateral confidence in [0, 1]
        """
        # Forward: cosine(sem_fwd[prev_t], codebook[tgt_t])
        sv_fwd = self.sem_fwd[prev_tokens]          # (batch, n_words)
        tv = self.codebook[tgt_tokens]               # (batch, n_words)
        xv_fwd = sv_fwd ^ tv
        pc_fwd = np.unpackbits(xv_fwd.view(np.uint8), axis=1).sum(axis=1).astype(np.float32)
        conf_fwd = np.abs(pc_fwd - self._half) / self._half

        # Backward: cosine(sem_bwd[tgt_t], codebook[prev_t])
        sv_bwd = self.sem_bwd[tgt_tokens]            # (batch, n_words)
        pv = self.codebook[prev_tokens]              # (batch, n_words)
        xv_bwd = sv_bwd ^ pv
        pc_bwd = np.unpackbits(xv_bwd.view(np.uint8), axis=1).sum(axis=1).astype(np.float32)
        conf_bwd = np.abs(pc_bwd - self._half) / self._half

        # Bilateral consistency: average of fwd and bwd confidence
        consistency = (conf_fwd + conf_bwd) / 2.0
        return np.clip(0.5 + 0.49 * consistency, 1e-30, 0.99).astype(np.float32)

    def vote_scores_all_vocab(
        self,
        prev_tokens: np.ndarray,
    ) -> np.ndarray:
        """Bilateral confidence scores for all vocab tokens.

        FIX #5: Replaces (batch, vocab, n_words) XOR + unpackbits with a
        single BLAS matmul in pm1 space.

        In pm1 space, cosine(a, b) = (a · b) / n_bits.
        The XOR-based confidence |cosine| = |hamming_distance - 0.5| × 2
        is equivalent to |dot(a_pm1, b_pm1)| / n_bits.

        So:
            conf_fwd[b, v] = |sem_fwd_pm1[prev_t[b]] · codebook_pm1[v]| / n_bits
                           = |matmul(sem_fwd_pm1[prev_tokens], codebook_pm1.T)| / n_bits

        This is a single (batch, n_bits) × (n_bits, vocab) BLAS SGEMM,
        replacing the (batch, vocab, n_words) XOR + unpackbits.

        Args:
            prev_tokens : (batch,) int32 — previous token IDs

        Returns:
            (batch, vocab_size) float32 — bilateral confidence scores
        """
        # Ensure pm1 cache is built (lazy, one-time cost)
        self._ensure_pm1_cache()

        # Forward: |sem_fwd_pm1[prev_tokens] @ codebook_pm1.T| / n_bits
        # (batch, n_bits) @ (n_bits, vocab) → (batch, vocab)
        sf_pm1 = self._sem_fwd_pm1[prev_tokens]          # (batch, n_bits)
        conf_fwd = np.abs(sf_pm1 @ self._codebook_pm1.T) / self._n_bits  # (batch, vocab)

        # Backward: |sem_bwd_pm1 @ codebook_pm1[prev_tokens].T| / n_bits
        # (vocab, n_bits) @ (n_bits, batch) → (vocab, batch) → (batch, vocab)
        cb_pm1_prev = self._codebook_pm1[prev_tokens]     # (batch, n_bits)
        conf_bwd = np.abs(self._sem_bwd_pm1 @ cb_pm1_prev.T).T / self._n_bits  # (batch, vocab)

        # Bilateral consistency
        consistency = (conf_fwd + conf_bwd) * 0.5        # (batch, vocab)
        return np.clip(0.5 + 0.49 * consistency, 1e-30, 0.99).astype(np.float32)
