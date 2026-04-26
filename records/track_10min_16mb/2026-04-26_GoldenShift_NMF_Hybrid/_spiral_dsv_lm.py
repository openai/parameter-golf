"""GoldenShift DSV Language Model — Forward-Only.

Keeps only what is needed for the pure GoldenShift DSV forward build:

  GoldenAxisShift   -- golden-ratio circular bit-shift (metric-preserving)
  GOLDEN_AXES       -- module-level singleton
  SpiralDSVLanguageModel -- forward-only DSV for 1D token sequences

Removed entirely:
  - SpiralPointerMemory  (hierarchical memory, not used)
  - build_from_tokens()  (replaced by EigenTrainer.build_bilateral_from_tokens)
  - _scatter_xor_fast()  (old XOR-bundle scatter, replaced by matmul)
  - vote_scores_batch()  (bilateral confidence, not used)
  - sem_bwd / _sem_bwd_pm1  (backward table, not built)
  - bilateral midpoint path in vote_scores_all_vocab()
  - backward confidence path in vote_scores_all_vocab()
  - All EigenSpiralBuilder / EigenTrainer imports (moved to _eigen_convergence.py)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# -- GPU acceleration (optional, graceful fallback to CPU) --------------------
try:
    from _gpu import gpu_available, gpu_bilateral_confidence
    _GPU_AVAILABLE = gpu_available()
except ImportError:
    _GPU_AVAILABLE = False
    gpu_bilateral_confidence = None


# =============================================================================
# Golden Ratio Axis Shift
# =============================================================================

PHI_FRAC: float = 0.6180339887498949  # phi - 1 = 1/phi


class GoldenAxisShift:
    """Generates an infinite, non-repeating, positive-only axis space.

    Each axis k is defined by a golden-ratio bit-offset in hypervector space:
        offset(k) = floor(k x PHI_FRAC x total_bits) mod total_bits

    Properties:
    - Non-repeating: offset(k) != offset(j) for all finite k != j
    - Maximally equidistributed: Weyl's theorem guarantees uniform coverage
    - Always positive: PHI_FRAC in (0,1), offsets are non-negative integers
    - Infinite: any k >= 0 is valid
    - Metric-preserving: cosine(partner_hv(A,k), partner_hv(B,k)) = cosine(A,B)
    - Lossless: inverse_hv(partner_hv(hv, k), k) = hv exactly

    The circular bit-shift is the cheapest possible lag-separation operation:
    O(n_words) per token, no matrix multiply, no unpackbits.
    """

    DEFAULT_N_AXES: int = 19

    def __init__(self, n_axes: int = DEFAULT_N_AXES, W: int = 16):
        self.n_axes = n_axes
        self.W = W
        self._total_bits = W * 64
        self._offsets: List[int] = [self._compute_offset(k) for k in range(n_axes)]
        self._word_shifts: List[int] = [off // 64 for off in self._offsets]
        self._bit_shifts: List[int] = [off % 64 for off in self._offsets]

    def _compute_offset(self, k: int) -> int:
        return int(k * PHI_FRAC * self._total_bits) % self._total_bits

    def offset(self, k: int) -> int:
        """Return the bit offset for axis k, extending the table if needed."""
        while k >= len(self._offsets):
            new_k = len(self._offsets)
            off = self._compute_offset(new_k)
            self._offsets.append(off)
            self._word_shifts.append(off // 64)
            self._bit_shifts.append(off % 64)
        return self._offsets[k]


# Module-level singleton -- shared across all files via import
GOLDEN_AXES = GoldenAxisShift(n_axes=19, W=16)


# =============================================================================
# SpiralDSVLanguageModel -- Forward-Only
# =============================================================================

class SpiralDSVLanguageModel:
    """Forward-only GoldenShift DSV for 1D token sequences.

    Stores sem_fwd: for each token a, sem_fwd[a] is the hypervector encoding
    what tokens tend to follow a (weighted by 1/freq, across lags 1..ctx_len,
    with GoldenAxisShift rotation per lag).

    Eval:
        score(a -> v) = sem_fwd_pm1[a] . codebook_pm1[v] / n_bits
        p(v | a) = softmax(score(a -> v))

    Memory budget (n_words=1024, vocab_size=1024):
        codebook : 1024 x 1024 x 8 = 8 MB
        sem_fwd  : 1024 x 1024 x 8 = 8 MB
        Total    : 16 MB

    Memory budget (n_words=1024, sem_fwd only):
        sem_fwd  : 8 MB  (codebook regenerated from seed at eval time)
    """

    def __init__(self, vocab_size: int, n_words: int = 16, seed: int = 42) -> None:
        self.vocab_size = vocab_size
        self.n_words    = n_words
        self.seed       = seed
        self._n_bits    = n_words * 64

        # Fibonacci-hash codebook: vocab_size x n_words uint64
        # Each token gets a unique HV from a seeded RNG (deterministic).
        # NOT stored in the artifact -- regenerated from seed at eval time.
        rng = np.random.default_rng(seed)
        self.codebook: np.ndarray = rng.integers(
            0, np.iinfo(np.uint64).max,
            size=(vocab_size, n_words), dtype=np.uint64
        )

        # Forward DSV table (built by EigenTrainer.build_bilateral_from_tokens)
        self.sem_fwd: np.ndarray = np.zeros((vocab_size, n_words), dtype=np.uint64)
        # sem_bwd kept as alias to sem_fwd for API compatibility with _semantic_layer.py
        self.sem_bwd: np.ndarray = self.sem_fwd

        self._built = False

        # Lazy pm1 cache (built once on first eval call, invalidated on table change)
        self._codebook_pm1: Optional[np.ndarray] = None   # (vocab, n_bits) float32
        self._sem_fwd_pm1 : Optional[np.ndarray] = None   # (vocab, n_bits) float32

    def _invalidate_pm1_cache(self) -> None:
        """Invalidate cached pm1 versions (call after sem_fwd is updated)."""
        self._sem_fwd_pm1  = None
        self._codebook_pm1 = None

    def _ensure_pm1_cache(self) -> None:
        """Lazily build pm1 cache from uint64 tables.

        Converts codebook and sem_fwd to float32 pm1 space once,
        then reuses for all subsequent vote_scores_all_vocab() calls.
        """
        if self._codebook_pm1 is None:
            bits = np.unpackbits(
                self.codebook.view(np.uint8).reshape(self.vocab_size, self.n_words * 8),
                axis=1, bitorder='little'
            )
            self._codebook_pm1 = bits.astype(np.float32) * 2.0 - 1.0   # (vocab, n_bits)

        if self._sem_fwd_pm1 is None:
            bits = np.unpackbits(
                self.sem_fwd.view(np.uint8).reshape(self.vocab_size, self.n_words * 8),
                axis=1, bitorder='little'
            )
            self._sem_fwd_pm1 = bits.astype(np.float32) * 2.0 - 1.0   # (vocab, n_bits)

    def vote_scores_all_vocab(
        self,
        prev_tokens: np.ndarray,
        next_tokens: Optional[np.ndarray] = None,   # ignored (forward-only)
    ) -> np.ndarray:
        """Forward DSV scores for all vocab tokens.

        score(b, v) = |sem_fwd_pm1[prev[b]] . codebook_pm1[v]| / n_bits

        GPU-accelerated via _gpu.py (cuBLAS HGEMM) when CUDA is available.

        Args:
            prev_tokens : (batch,) int32 -- preceding token IDs
            next_tokens : ignored (kept for API compatibility)

        Returns:
            (batch, vocab_size) float32 -- probability-like scores in [0, 1]
        """
        self._ensure_pm1_cache()

        # GPU fast path: dispatches matmul to cuBLAS HGEMM
        if _GPU_AVAILABLE and gpu_bilateral_confidence is not None:
            return gpu_bilateral_confidence(
                sem_fwd_pm1  = self._sem_fwd_pm1,
                sem_bwd_pm1  = self._sem_fwd_pm1,   # alias: fwd == bwd (forward-only)
                codebook_pm1 = self._codebook_pm1,
                prev_tokens  = prev_tokens,
            )

        # CPU fallback: BLAS SGEMM
        # score(b, v) = |sem_fwd_pm1[prev[b]] . codebook_pm1[v]| / n_bits
        # (batch, n_bits) @ (n_bits, vocab) -> (batch, vocab)
        sf_pm1   = self._sem_fwd_pm1[prev_tokens]              # (batch, n_bits)
        conf_fwd = np.abs(sf_pm1 @ self._codebook_pm1.T) / self._n_bits  # (batch, vocab)
        return np.clip(0.5 + 0.49 * conf_fwd, 1e-30, 0.99).astype(np.float32)
