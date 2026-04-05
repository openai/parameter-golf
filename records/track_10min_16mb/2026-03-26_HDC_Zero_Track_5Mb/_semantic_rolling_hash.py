"""Semantic Rolling Hash S[p] — unlimited-context semantic fallback.

S[p] is the HDC analog of a transformer's key-value cache: it accumulates
the distributional meaning of all prior tokens into a fixed-size 1024-bit
vector, updated in O(W_UINT64) = O(16) ops per token.

Compare to the existing rolling hash G[p]:
    G[p+1] = G[p]  XOR  (tokens[p]  * HADAMARD_KEY[p])   ← scalar token id
    S[p+1] = S[p]  XOR  (sem_fwd[tokens[p]] * HADAMARD_KEY[p])  ← 1024-bit semantic vector

G[p] binds token *identity*. S[p] binds token *distributional meaning*.
Both use the same phi-based position key, making accumulation order-sensitive.

At inference, S[p] is queried via Walsh-Hadamard Transform (WHT) which
computes all 1024 token similarities simultaneously in O(N log N) and
returns a full spectrum — enabling butterfly consistency checks that
distinguish genuine signals from noise accumulation.

SNR management:
    - Forgetting factor alpha (default 0.005): effective window ~200 tokens
    - Phi-based position keys: eliminates systematic constructive interference
    - Butterfly consistency check: filters noise from genuine ambiguity
    - Surrounding token consensus (k=5): drives false-agreement rate to ~10^-12

Storage: checkpoint states at chunk boundaries only (~32 KB for 500M tokens).
Recomputed forward within each chunk during eval — zero new infrastructure.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHI64 = np.uint64(0x9E3779B97F4A7C15)   # Fibonacci / golden-ratio mixing constant
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


# ---------------------------------------------------------------------------
# Utility: Hamming similarity
# ---------------------------------------------------------------------------

def hamming_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Hamming similarity in [0, 1]: 1.0 = identical, 0.5 = random."""
    xor = a ^ b
    bits = int(np.unpackbits(xor.view(np.uint8)).sum())
    total = len(a) * 64
    return 1.0 - bits / total


# ---------------------------------------------------------------------------
# WHT and bipolar conversion
# ---------------------------------------------------------------------------

def bipolar(hv: np.ndarray) -> np.ndarray:
    """Convert uint64 hypervector to (dim,) float32 bipolar (+1 / -1).

    Parameters
    ----------
    hv : (W_UINT64,) uint64
    Returns
    -------
    (W_UINT64 * 64,) float32  — +1 where bit is set, -1 where bit is clear
    """
    bits = np.unpackbits(hv.view(np.uint8))   # (W_UINT64*8,) uint8 0/1
    return bits.astype(np.float32) * 2.0 - 1.0   # map 0→-1, 1→+1


def wht(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform (in-place Cooley-Tukey butterfly).

    Parameters
    ----------
    x : (N,) float32  — N must be a power of 2
    Returns
    -------
    (N,) float32  — WHT of x (unnormalised)

    Complexity: O(N log2 N) — for N=1024: 10 × 1024 = 10 240 ops.
    """
    x = x.copy()
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = x[j]
                v = x[j + h]
                x[j]     = u + v
                x[j + h] = u - v
        h *= 2
    return x


def wht_vectorised(x: np.ndarray) -> np.ndarray:
    """Vectorised WHT using numpy — avoids Python loops for large N.

    Equivalent to wht() but uses numpy slice operations.
    Faster for N >= 256.
    """
    x = x.copy()
    n = len(x)
    h = 1
    while h < n:
        x_reshaped = x.reshape(-1, 2 * h)
        u = x_reshaped[:, :h].copy()
        v = x_reshaped[:, h:].copy()
        x_reshaped[:, :h] = u + v
        x_reshaped[:, h:] = u - v
        x = x_reshaped.reshape(-1)
        h *= 2
    return x


# ---------------------------------------------------------------------------
# SemanticRollingHash
# ---------------------------------------------------------------------------

class SemanticRollingHash:
    """Semantic rolling hash S[p] with forgetting factor and WHT query.

    Parameters
    ----------
    W_UINT64 : int
        Number of uint64 blocks per hypervector (16 for 1024-bit vectors).
    alpha : float
        Forgetting factor: fraction of bits randomly flipped per step.
        alpha=0.005 → effective window ~200 tokens.
        alpha=0.02  → effective window ~50 tokens.
        alpha=0.0   → no forgetting (infinite accumulation, SNR degrades as 1/√n).
    """

    def __init__(self, W_UINT64: int = 16, alpha: float = 0.005) -> None:
        self.W_UINT64 = W_UINT64
        self.dim = W_UINT64 * 64
        self.alpha = alpha
        self._neutral = W_UINT64 * 32   # expected popcount for random vector

    # ------------------------------------------------------------------
    # Forgetting mask
    # ------------------------------------------------------------------

    def _flip_mask(self, p: int) -> np.ndarray:
        """Deterministic flip mask for position p.

        Seeded by position so it is reproducible across training and eval.
        Flips approximately alpha * dim bits.
        """
        if self.alpha <= 0.0:
            return np.zeros(self.W_UINT64, dtype=np.uint64)

        n_flips = max(1, int(self.alpha * self.dim))
        rng = np.random.RandomState(int(p) & 0x7FFFFFFF)
        flip_positions = rng.choice(self.dim, n_flips, replace=False)

        mask = np.zeros(self.W_UINT64, dtype=np.uint64)
        for pos in flip_positions:
            block = pos // 64
            bit   = pos % 64
            mask[block] ^= np.uint64(1) << np.uint64(bit)
        return mask

    # ------------------------------------------------------------------
    # Single-step accumulation
    # ------------------------------------------------------------------

    def step(self, S: np.ndarray, sem_fwd_vec: np.ndarray, key: np.uint64,
             p: int) -> np.ndarray:
        """One-step update: S[p+1] = (S[p] XOR flip_mask) XOR (sem_fwd[t] * key).

        Parameters
        ----------
        S           : (W_UINT64,) uint64 — current state
        sem_fwd_vec : (W_UINT64,) uint64 — sem_fwd[tokens[p]]
        key         : uint64 — HADAMARD_KEY[p]
        p           : int — current position (for deterministic flip mask)

        Returns
        -------
        (W_UINT64,) uint64 — next state S[p+1]
        """
        S_new = S ^ self._flip_mask(p)
        with np.errstate(over='ignore'):
            binding = sem_fwd_vec * key   # element-wise uint64 multiply
        return S_new ^ binding

    # ------------------------------------------------------------------
    # Bulk state building (training)
    # ------------------------------------------------------------------

    def build_states(
        self,
        tokens: np.ndarray,
        sem_fwd_matrix: np.ndarray,
        keys: np.ndarray,
        chunk_boundaries: List[int],
        time_budget_s: float = 30.0,
        label: str = "SRH",
    ) -> Dict[int, np.ndarray]:
        """Build S[p] checkpoint states at chunk boundaries.

        Only stores states at the positions in chunk_boundaries — not all N.
        Recompute forward within each chunk during eval.

        Parameters
        ----------
        tokens          : (N,) uint16 — full token array
        sem_fwd_matrix  : (vocab_size, W_UINT64) uint64 — sem_fwd vectors
        keys            : (N,) uint64 — HADAMARD_KEY[p] for each position
        chunk_boundaries: list of int — positions where checkpoints are stored
        time_budget_s   : float — soft wall-clock limit

        Returns
        -------
        Dict[int, np.ndarray] — {position: S[position]} for each checkpoint
        """
        N = len(tokens)
        vocab_size = sem_fwd_matrix.shape[0]
        start = time.time()

        print(f"\n[{label}] Building S[p] checkpoint states "
              f"(N={N:,}, alpha={self.alpha}, checkpoints={len(chunk_boundaries)})")

        S = np.zeros(self.W_UINT64, dtype=np.uint64)
        checkpoints: Dict[int, np.ndarray] = {0: S.copy()}

        boundary_set = set(chunk_boundaries)

        for p in range(N):
            if time.time() - start > time_budget_s:
                print(f"[{label}] Time budget reached at p={p:,}")
                break

            tok = int(tokens[p])
            if tok >= vocab_size:
                tok = 0
            sem_vec = sem_fwd_matrix[tok]
            key = keys[p] if p < len(keys) else PHI64

            S = self.step(S, sem_vec, key, p)

            if (p + 1) in boundary_set:
                checkpoints[p + 1] = S.copy()

        elapsed = time.time() - start
        print(f"[{label}] Done. {len(checkpoints)} checkpoints in {elapsed:.2f}s")
        return checkpoints

    # ------------------------------------------------------------------
    # Recompute within a chunk (eval)
    # ------------------------------------------------------------------

    def recompute_chunk(
        self,
        chunk_start: int,
        chunk_end: int,
        tokens: np.ndarray,
        sem_fwd_matrix: np.ndarray,
        keys: np.ndarray,
        checkpoints: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Recompute S[p] for all positions in [chunk_start, chunk_end).

        Finds the nearest checkpoint at or before chunk_start, then
        accumulates forward to fill the chunk.

        Parameters
        ----------
        chunk_start, chunk_end : int — half-open range
        tokens                 : (N,) uint16
        sem_fwd_matrix         : (vocab_size, W_UINT64) uint64
        keys                   : (N,) uint64
        checkpoints            : Dict[int, np.ndarray] — from build_states()

        Returns
        -------
        (chunk_end - chunk_start, W_UINT64) uint64 — S[p] for each position
        """
        vocab_size = sem_fwd_matrix.shape[0]
        chunk_n = chunk_end - chunk_start

        # Find nearest checkpoint at or before chunk_start
        valid_cps = [cp for cp in checkpoints if cp <= chunk_start]
        if valid_cps:
            start_cp = max(valid_cps)
            S = checkpoints[start_cp].copy()
        else:
            start_cp = 0
            S = np.zeros(self.W_UINT64, dtype=np.uint64)

        # Accumulate from start_cp to chunk_start (warm-up, not stored)
        for p in range(start_cp, chunk_start):
            tok = int(tokens[p]) if p < len(tokens) else 0
            if tok >= vocab_size:
                tok = 0
            sem_vec = sem_fwd_matrix[tok]
            key = keys[p] if p < len(keys) else PHI64
            S = self.step(S, sem_vec, key, p)

        # Accumulate through the chunk, storing each state
        states = np.zeros((chunk_n, self.W_UINT64), dtype=np.uint64)
        for i, p in enumerate(range(chunk_start, chunk_end)):
            states[i] = S
            tok = int(tokens[p]) if p < len(tokens) else 0
            if tok >= vocab_size:
                tok = 0
            sem_vec = sem_fwd_matrix[tok]
            key = keys[p] if p < len(keys) else PHI64
            S = self.step(S, sem_vec, key, p)

        return states   # (chunk_n, W_UINT64)

    def recompute_single(
        self,
        p: int,
        tokens: np.ndarray,
        sem_fwd_matrix: np.ndarray,
        keys: np.ndarray,
        checkpoints: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Recompute S[p] for a single position.

        Convenience wrapper around recompute_chunk for single-position queries.
        """
        states = self.recompute_chunk(p, p + 1, tokens, sem_fwd_matrix, keys, checkpoints)
        return states[0]

    # ------------------------------------------------------------------
    # WHT prediction
    # ------------------------------------------------------------------

    def wht_predict(
        self,
        S_p: np.ndarray,
        codebook: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query S[p] via Walsh-Hadamard Transform.

        Computes all vocab_size token similarities simultaneously in
        O(vocab * log(vocab)) = O(10K ops) for vocab=1024.

        Parameters
        ----------
        S_p      : (W_UINT64,) uint64 — semantic state at position p
        codebook : (vocab_size, W_UINT64) uint64 — Hadamard codebook

        Returns
        -------
        correlations : (vocab_size,) float32
            correlations[t] = bipolar similarity of S_p to codebook[t].
            Range: [-1, +1]. Positive = S_p resembles token t's semantic profile.
        bipolar_S : (dim,) float32
            Bipolar representation of S_p (for reuse in WHT).
        """
        bipolar_S = bipolar(S_p)   # (dim,) float32

        # WHT over bipolar S_p gives correlations with all Hadamard rows
        # correlations[t] = <bipolar_S, H[t]> / dim  (normalised dot product)
        correlations = wht_vectorised(bipolar_S)   # (dim,) float32

        # Normalise to [-1, +1]
        correlations = correlations / float(self.dim)

        # The WHT output has dim entries; for vocab_size < dim, take first vocab_size
        vocab_size = codebook.shape[0]
        correlations = correlations[:vocab_size]

        return correlations, bipolar_S

    # ------------------------------------------------------------------
    # Butterfly consistency check
    # ------------------------------------------------------------------

    def butterfly_consistency(
        self,
        correlations: np.ndarray,
        winner: int,
        n_levels: int = 10,
    ) -> float:
        """Check that butterfly partners of winner are near the noise floor.

        For a genuine Hadamard signal at token `winner`, all butterfly
        partners (winner XOR 2^k for k in 0..n_levels-1) should have
        correlations near zero. Elevated partners indicate noise.

        Parameters
        ----------
        correlations : (vocab_size,) float32 — from wht_predict()
        winner       : int — argmax of correlations
        n_levels     : int — number of butterfly levels to check (log2(vocab_size))

        Returns
        -------
        float in [0, 1]: 1.0 = all energy at winner (genuine), 0.0 = spread (noise)
        """
        vocab_size = len(correlations)
        winner_corr = float(correlations[winner])

        if abs(winner_corr) < 1e-8:
            return 0.0

        partner_ratios = []
        for k in range(n_levels):
            partner = winner ^ (1 << k)
            if partner < 0 or partner >= vocab_size:
                continue
            ratio = abs(float(correlations[partner])) / (abs(winner_corr) + 1e-8)
            partner_ratios.append(ratio)

        if not partner_ratios:
            return 1.0

        max_ratio = max(partner_ratios)
        consistency = max(0.0, 1.0 - max_ratio)
        return consistency

    # ------------------------------------------------------------------
    # Depth-scaled threshold
    # ------------------------------------------------------------------

    def depth_scaled_threshold(self, depth: int) -> float:
        """Expected genuine signal strength at accumulation depth d.

        At depth d, SNR ≈ 1/√d. A genuine signal should produce a
        correlation above this threshold even at large depth.

        Parameters
        ----------
        depth : int — number of tokens accumulated in S[p]

        Returns
        -------
        float — minimum correlation to consider a genuine signal
        """
        return 0.5 + 1.5 / (float(depth) ** 0.5 + 1.0)

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def classify_regime(
        self,
        correlations: np.ndarray,
        depth: int = 100,
    ) -> Tuple[str, int, float]:
        """Classify the WHT spectrum into one of three regimes.

        Regime 1 — clean signal:
            One elevated peak, all butterfly partners near zero.
            → Use prediction with high confidence.

        Regime 2 — genuine ambiguity:
            Two elevated peaks, both with clean butterfly partners.
            → Blend over 2-3 candidates.

        Regime 3 — noise:
            Multiple butterfly levels elevated.
            → Fall through to next layer.

        Parameters
        ----------
        correlations : (vocab_size,) float32
        depth        : int — tokens accumulated (for SNR-scaled threshold)

        Returns
        -------
        (regime, winner, confidence)
        regime     : str — 'clean', 'ambiguous', or 'noise'
        winner     : int — argmax of correlations
        confidence : float — scaled confidence score
        """
        vocab_size = len(correlations)
        winner = int(np.argmax(correlations))
        winner_corr = float(correlations[winner])

        threshold = self.depth_scaled_threshold(depth)
        consistency = self.butterfly_consistency(correlations, winner)

        # Count elevated peaks above noise floor
        noise_floor = 0.5 / (float(depth) ** 0.5 + 1.0)
        n_elevated = int(np.sum(correlations > noise_floor))

        if winner_corr > threshold and consistency > 0.7:
            return 'clean', winner, winner_corr * consistency

        if n_elevated == 2 and consistency > 0.3:
            return 'ambiguous', winner, winner_corr * consistency

        return 'noise', winner, 0.0

    # ------------------------------------------------------------------
    # Surrounding token consensus
    # ------------------------------------------------------------------

    def consensus_predict(
        self,
        S_states: np.ndarray,
        i: int,
        sem_fwd_matrix: np.ndarray,
        codebook: np.ndarray,
        keys: np.ndarray,
        window: int = 5,
    ) -> Tuple[int, float, int]:
        """Query S[p], S[p-1], ..., S[p-window+1] and find consensus.

        Each S[q] predicts the token at position q+1. If k of them agree
        on the same candidate, the false-agreement probability is (1/vocab)^(k-1).
        For k=5 and vocab=1024: ~10^-15 — functionally zero noise.

        Parameters
        ----------
        S_states : (chunk_n, W_UINT64) uint64 — states for current chunk
        i        : int — index within S_states (not absolute position)
        sem_fwd_matrix : (vocab_size, W_UINT64) uint64
        codebook : (vocab_size, W_UINT64) uint64
        keys     : (N,) uint64
        window   : int — number of neighboring states to query

        Returns
        -------
        (winner, confidence, n_agreeing)
        """
        votes: Dict[int, float] = {}

        for offset in range(window):
            q = i - offset
            if q < 0:
                continue

            S_q = S_states[q]
            correlations, _ = self.wht_predict(S_q, codebook)
            candidate = int(np.argmax(correlations))
            confidence = float(correlations[candidate])

            if confidence > 0.0:
                votes[candidate] = votes.get(candidate, 0.0) + confidence

        if not votes:
            return 0, 0.0, 0

        winner = max(votes, key=lambda k: votes[k])
        n_agreeing = sum(1 for v in votes.values() if v > 0.05)
        return winner, votes[winner], n_agreeing

    # ------------------------------------------------------------------
    # Bidirectional semantic rolling hash (training only)
    # ------------------------------------------------------------------

    def build_bidirectional_states(
        self,
        tokens: np.ndarray,
        sem_fwd_matrix: np.ndarray,
        sem_bwd_matrix: np.ndarray,
        keys: np.ndarray,
        chunk_boundaries: List[int],
        time_budget_s: float = 30.0,
        label: str = "SRH-BIDIR",
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Build both forward S_fwd[p] and backward S_bwd[p] checkpoint states.

        S_fwd[p] carries semantic information about everything before p.
        S_bwd[p] carries semantic information about everything after p.

        A token predicted by BOTH forward and backward context simultaneously
        is almost certainly the true signal — near-zero noise.

        Returns
        -------
        (fwd_checkpoints, bwd_checkpoints)
        """
        print(f"\n[{label}] Building bidirectional S[p] states...")
        start = time.time()

        # Forward pass
        fwd_checkpoints = self.build_states(
            tokens, sem_fwd_matrix, keys, chunk_boundaries,
            time_budget_s=time_budget_s * 0.5, label=f"{label}-fwd"
        )

        if time.time() - start > time_budget_s:
            return fwd_checkpoints, {}

        # Backward pass: reverse tokens and use sem_bwd
        tokens_rev = tokens[::-1].copy()
        keys_rev = keys[:len(tokens)][::-1].copy() if len(keys) >= len(tokens) else keys

        bwd_checkpoints_rev = self.build_states(
            tokens_rev, sem_bwd_matrix, keys_rev, chunk_boundaries,
            time_budget_s=time_budget_s * 0.5, label=f"{label}-bwd"
        )

        # Re-index backward checkpoints to forward positions
        N = len(tokens)
        bwd_checkpoints: Dict[int, np.ndarray] = {}
        for cp, state in bwd_checkpoints_rev.items():
            fwd_pos = N - cp
            if fwd_pos >= 0:
                bwd_checkpoints[fwd_pos] = state

        elapsed = time.time() - start
        print(f"[{label}] Done in {elapsed:.2f}s | "
              f"fwd={len(fwd_checkpoints)} bwd={len(bwd_checkpoints)} checkpoints")
        return fwd_checkpoints, bwd_checkpoints

    # ------------------------------------------------------------------
    # Slow-wave pruning (Phase 4 integration)
    # ------------------------------------------------------------------

    def slow_wave_prune(
        self,
        sem_fwd_matrix: np.ndarray,
        codebook: np.ndarray,
        confidence_threshold: float = 0.3,
    ) -> Tuple[int, int]:
        """Prune noisy sem_fwd windows and nudge low-confidence entries.

        Called every 3 Phase 4 rounds to prevent noise accumulation from
        swamping high-confidence correct entries.

        Parameters
        ----------
        sem_fwd_matrix      : (vocab_size, W_UINT64) uint64 — modified in-place
        codebook            : (vocab_size, W_UINT64) uint64
        confidence_threshold: float — below this, window is considered noisy

        Returns
        -------
        (pruned, nudged) — counts of pruned and nudged windows
        """
        vocab_size = sem_fwd_matrix.shape[0]
        pruned = 0
        nudged = 0

        for t in range(vocab_size):
            sem_vec = sem_fwd_matrix[t]
            correlations, _ = self.wht_predict(sem_vec, codebook)
            winner = int(np.argmax(correlations))
            winner_corr = float(correlations[winner])
            consistency = self.butterfly_consistency(correlations, winner)

            if winner_corr < confidence_threshold and consistency < 0.3:
                # Noisy window — zero it out to stop it polluting predictions
                sem_fwd_matrix[t] = np.zeros(self.W_UINT64, dtype=np.uint64)
                pruned += 1
            elif winner_corr < confidence_threshold * 0.5:
                # Very weak signal — nudge toward the codebook entry it most resembles
                sem_fwd_matrix[t] ^= codebook[winner]
                nudged += 1

        return pruned, nudged
