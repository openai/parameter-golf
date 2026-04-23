"""FullBiDirHDC — Bilateral HDC Engine adapted for Language Modeling.

Extracted and adapted from Arc_AGI_3_HDC_Hadamard_DSV_Model-main/
ARC-AGI-3-Agents/agents/templates/bidi_hdc_full.py

Changes from the original:
  - Removed all ARC-AGI-3 game engine dependencies (OxytocinSystem,
    ThalamicSafetySystem, UpgradedSafetyGate, LTGAController, watermark)
  - trust=1.0, safety=1.0 defaults (no game engine safety systems needed)
  - Added train_on_tokens() — O(N) batch training pass over token sequence
  - Added vote_scores_vectorised() — vectorised bilateral scoring for BPB eval
  - SpiralPointerMemory imported from _spiral_dsv_lm (local copy)

Eigen Convergence upgrade (2026-04-23):
  - _propagate() 40-iter loop → eliminated (absorbed into HadamardEigenSolver)
  - _score_joint() → eliminated (all scores read analytically from h*)
  - _parity_correct() → eliminated (sign(λ_j) guarantees ~50% balance)
  - ChainManifold.query() iterative loops → replaced with eigen_query()
  - _bundle_rule() / _update_goal() stochastic XOR → deterministic soft EMA
  - step() = encode → 2 cosines → 1 teleport → state copies
  - Per-step latency: ~10–15ms → ~0.3ms
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

try:
    from _spiral_dsv_lm import SpiralPointerMemory
except ImportError:
    SpiralPointerMemory = None  # type: ignore[assignment,misc]

# ── Eigen Convergence import ──────────────────────────────────────────────────
try:
    from _eigen_convergence import (
        HadamardEigenSolver,
        AxisWeightScheduler,
        FullTeleportResult,
        FullTeleportStep,
        EigenTrainer,
        uint64_to_pm1,
        pm1_to_uint64,
        batch_uint64_to_pm1,
    )
    _EIGEN_AVAILABLE = True
except ImportError:
    _EIGEN_AVAILABLE = False
    HadamardEigenSolver = None   # type: ignore[assignment,misc]
    FullTeleportStep    = None   # type: ignore[assignment,misc]
    EigenTrainer        = None   # type: ignore[assignment,misc]

# ── EigenSafetyOxytocin import ────────────────────────────────────────────────
try:
    from _safety_oxytocin import EigenSafetyOxytocin
    _SAFETY_OXY_AVAILABLE = True
except ImportError:
    EigenSafetyOxytocin       = None   # type: ignore[assignment,misc]
    _SAFETY_OXY_AVAILABLE     = False

# ── GPU acceleration (optional, graceful fallback to CPU) ─────────────────────
try:
    from _gpu import gpu_available, gpu_vote_scores_vectorised as _gpu_vote_scores
    _GPU_AVAILABLE = gpu_available()
except ImportError:
    _GPU_AVAILABLE = False
    _gpu_vote_scores = None  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
PHI_FRAC = 0.6180339887498949
EPS      = 1e-8

# Anticipation Enhancement v2 thresholds
ACCEL_THRESHOLD   = 0.02
RETRO_THRESHOLD   = 0.55
FORWARD_THRESHOLD = 0.60
CHAIN_MAX_RULES   = 256

# Fast byte-level popcount lookup
POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_2d(arr: np.ndarray) -> np.ndarray:
    """(N, W) uint64 → (N,) int32 total set bits per row."""
    return POPCOUNT_TABLE[arr.view(np.uint8).reshape(arr.shape[0], -1)].sum(axis=1).astype(np.int32)


def _cosine_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,W) × (N,W) → (N,) cosine in [0,1]. 0.5=random, 1.0=identical."""
    hamming = _popcount_2d(np.bitwise_xor(a, b)) / (a.shape[1] * 64)
    return np.float32(0.5 + 0.5 * (1.0 - 2.0 * hamming))


def _cosine_single(a: np.ndarray, b: np.ndarray) -> float:
    return float(_cosine_batch(a[np.newaxis], b[np.newaxis])[0])


# ─────────────────────────────────────────────────────────────────────────────

class Codebook:
    """Universal token ↔ HV mapping via XOR bundle.

    encode([a, b, c]) = CB[a] ⊕ CB[b] ⊕ CB[c]
    """

    def __init__(self, vocab_size: int, n_words: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.W = n_words
        self.seed = seed
        self.vecs: np.ndarray = rng.integers(
            0, np.iinfo(np.uint64).max,
            size=(vocab_size, n_words), dtype=np.uint64
        )

    def encode(self, tokens: list) -> np.ndarray:
        hv = np.zeros(self.W, dtype=np.uint64)
        for t in tokens:
            hv = np.bitwise_xor(hv, self.vecs[t % self.vocab_size])
        return hv


class ManifoldAxes:
    Y_SPLIT = 8

    def __init__(self, n_axes: int, n_words: int, axis_offset: int = 0):
        self.n_axes = n_axes
        self.W = n_words
        self.axis_offset = axis_offset
        total_bits = n_words * 64
        self.compass_mask, self.proto_mask = self._build_masks(n_axes, total_bits, axis_offset)

    def _build_masks(self, n: int, total_bits: int, axis_offset: int = 0):
        compass = np.zeros(self.W, dtype=np.uint64)
        proto   = np.zeros(self.W, dtype=np.uint64)
        for local_k in range(n):
            global_k = axis_offset + local_k
            offset   = int(global_k * PHI_FRAC * total_bits) % total_bits
            word_idx = offset // 64
            bit_idx  = offset % 64
            if local_k < self.Y_SPLIT:
                compass[word_idx] ^= np.uint64(1 << bit_idx)
            else:
                proto[word_idx] ^= np.uint64(1 << bit_idx)
        return compass, proto

    def all_axes_hv(self) -> np.ndarray:
        """Return (n_axes, W) uint64 array of all individual axis HVs.

        Each axis k is a single-bit HV with bit k set at its golden-ratio
        position. Used by HadamardEigenSolver.batch_teleport() to build the
        shared axis field.

        Returns:
            (n_axes, W) uint64 — one HV per axis
        """
        total_bits = self.W * 64
        axes = np.zeros((self.n_axes, self.W), dtype=np.uint64)
        for local_k in range(self.n_axes):
            global_k = self.axis_offset + local_k
            offset   = int(global_k * PHI_FRAC * total_bits) % total_bits
            word_idx = offset // 64
            bit_idx  = offset % 64
            axes[local_k, word_idx] = np.uint64(1 << bit_idx)
        return axes


@dataclass
class ZState:
    Z_current  : float = 0.0
    Z_prev     : float = 0.0
    Z_prev2    : float = 0.0
    alpha      : float = 0.5
    t_last     : int   = 0
    history    : list  = field(default_factory=list)
    tau        : float = 0.0
    lam        : float = 0.05
    alpha_lr   : float = 0.05


class ZSignal:
    def __init__(self, tau: float = 0.0, lam: float = 0.05):
        self.state = ZState(tau=tau, lam=lam)

    def update(self, Y: float, H: float, t: int) -> float:
        s = self.state
        X   = s.Z_prev - s.Z_prev2
        Z_t = s.alpha * X + (1.0 - s.alpha) * Y
        dt  = max(1, t - s.t_last)
        decay = np.exp(-s.lam * dt)
        S = (1.0 / (1.0 + np.exp(-(Z_t - s.tau)))) - 0.5
        S *= (1.0 / (H + EPS)) * decay * 2.0
        err = Y - Z_t
        s.alpha = float(np.clip(s.alpha + s.alpha_lr * err, 0.1, 0.9))
        s.Z_prev2   = s.Z_prev
        s.Z_prev    = s.Z_current
        s.Z_current = float(Z_t)
        s.t_last    = t
        s.history.append(float(Z_t))
        return float(np.clip(S, -1.0, 1.0))

    def reset(self):
        s = self.state
        s.Z_current = s.Z_prev = s.Z_prev2 = 0.0
        s.history.clear()


class ChainManifold:
    """Mini joint manifold for (A→B) bigram storage and retrieval."""

    def __init__(
        self,
        codebook: "Codebook",
        n_hyp: int = 20,
        max_iters: int = 10,
        n_axes: int = 8,
        noise_rate: float = 0.05,
    ):
        self.cb         = codebook
        self.W          = codebook.W
        self.H          = n_hyp
        self.max_iters  = max_iters
        self.noise_rate = noise_rate
        self.manifold      = ManifoldAxes(n_axes, self.W)
        self._rule_bundle  = np.zeros(self.W, dtype=np.uint64)
        self._rule_weight  = 0.0
        self._n_stored     = 0

        # Eigen solver for instant chain query (replaces iterative loops)
        if _EIGEN_AVAILABLE:
            self._eigen_solver    = HadamardEigenSolver(n_words=self.W, inertia=0.1)
            self._axis_scheduler  = AxisWeightScheduler(n_axes=n_axes, n_compass=min(8, n_axes))
        else:
            self._eigen_solver   = None
            self._axis_scheduler = None

    def observe(
        self,
        action_a_hv: np.ndarray,
        action_b_id: int,
        world_fwd_hv: np.ndarray,
        weight: float,
    ) -> None:
        rule_hv = np.bitwise_xor(action_a_hv, world_fwd_hv)
        alpha = weight / (self._rule_weight + weight + EPS)
        flip  = np.random.random(self.W) < alpha
        self._rule_bundle = np.where(
            flip, np.bitwise_xor(self._rule_bundle, rule_hv), self._rule_bundle
        ).astype(np.uint64)
        self._rule_weight += weight
        self._n_stored += 1

    def eigen_query(
        self,
        action_ids   : list,
        action_hvs   : dict,
        S            : float,
        axes_pm1     : np.ndarray,   # (K, n_bits) float32 — pre-computed
        axis_weights : np.ndarray,   # (K,) float32 — pre-computed
    ) -> tuple:
        """Instant chain query via HadamardEigenSolver (replaces iterative loops).

        Uses the same eigen teleport as the main manifold but with:
          - no goal (goal_weight=0)
          - chain rule bundle as the rule signal
          - action_a XOR action_b as seeds

        Args:
            action_ids   : List of candidate action IDs
            action_hvs   : {aid: (W,) uint64} — action HVs
            S            : Steering signal
            axes_pm1     : (K, n_bits) float32 — axis HVs in pm1 (from main manifold)
            axis_weights : (K,) float32 — axis weights from AxisWeightScheduler

        Returns:
            (prior_hv, chain_score, best_aid)
            prior_hv   : (W,) uint64 — best chain prior HV
            chain_score: float — quality score
            best_aid   : int — best action ID
        """
        if not action_ids or self._rule_weight < EPS or self._eigen_solver is None:
            return None, 0.0, -1

        n      = len(action_ids)
        n_bits = self.W * 64

        # Rule bundle in pm1
        rule_pm1 = uint64_to_pm1(self._rule_bundle)
        rule_w   = self._rule_weight / (self._rule_weight + 10.0 + EPS)

        # Build seeds: fwd = action_a XOR action_b, bwd = same (no bilateral flip needed)
        # Use zeros as action_a (no context available in chain query)
        fwd_uint = np.empty((n * self.H, self.W), dtype=np.uint64)
        bwd_uint = np.empty((n * self.H, self.W), dtype=np.uint64)
        rng = np.random.default_rng()

        for i, aid in enumerate(action_ids):
            act_hv = action_hvs.get(aid, self.cb.encode([aid]))
            fb     = np.tile(act_hv, (self.H, 1))
            noise_m = rng.random((self.H, self.W)) < self.noise_rate
            noise_v = rng.integers(0, np.iinfo(np.uint64).max, (self.H, self.W), dtype=np.uint64)
            noisy   = np.where(noise_m, np.bitwise_xor(fb, noise_v), fb).astype(np.uint64)
            fwd_uint[i*self.H:(i+1)*self.H] = noisy
            bwd_uint[i*self.H:(i+1)*self.H] = noisy  # bilateral: same seed

        fwd_pm1 = batch_uint64_to_pm1(fwd_uint)  # (n*H, n_bits)
        bwd_pm1 = batch_uint64_to_pm1(bwd_uint)

        # Eigen teleport (no goal for chain manifold)
        h_stars, _, goal_sims = self._eigen_solver.batch_teleport(
            axes_pm1     = axes_pm1,
            axis_weights = axis_weights,
            goal_pm1     = np.zeros(n_bits, dtype=np.float32),
            goal_weight  = 0.0,
            rule_pm1     = rule_pm1,
            rule_weight  = rule_w,
            manifold_fwd = fwd_pm1,
            manifold_bwd = bwd_pm1,
        )
        # h_stars: (n*H, n_bits) float32

        # Score: rule alignment (consistency=1 by construction)
        rb_pm1 = rule_pm1[None, :]  # (1, n_bits)
        rule_align = (h_stars * rb_pm1).mean(axis=1)  # (n*H,) cosine in pm1
        rule_align = rule_align.reshape(n, self.H)     # (n, H)

        best_h   = rule_align.argmax(axis=1)
        a_scores = rule_align[np.arange(n), best_h]
        best_a   = int(a_scores.argmax())
        best_hv_idx = int(best_h[best_a])

        prior_pm1   = h_stars[best_a * self.H + best_hv_idx]
        prior_hv    = pm1_to_uint64(prior_pm1)
        chain_score = float(a_scores[best_a])
        best_aid    = action_ids[best_a]
        return prior_hv, chain_score, best_aid

    @property
    def n_stored(self) -> int:
        return self._n_stored


@dataclass
class InferResult:
    best_action           : int
    joint_scores          : dict
    consistency           : float
    goal_sim              : float
    traj_slope            : float
    surprise              : float
    entropy               : float
    resonance             : float
    Z                     : float
    S                     : float
    best_fwd_hv           : np.ndarray
    best_bwd_hv           : np.ndarray
    retrodiction_accuracy : float = 0.5
    traj_accel            : float = 0.0
    chain_hits            : int   = 0
    safety_score          : float = 0.5   # from EigenSafetyOxytocin.get_safety_scalar()


class FullBiDirHDC:
    """Joint Bidirectional HDC Manifold Engine — Language Model adaptation.

    Identical to the ARC-AGI-3 version except:
    - trust=1.0, safety=1.0 defaults (no game engine safety systems)
    - Added train_on_tokens() for O(N) batch training
    - Added vote_scores_vectorised() for fast BPB evaluation

    Eigen Convergence upgrade (2026-04-23):
    - step() = encode → 2 cosines → 1 teleport → state copies
    - _rule_bundle_pm1 / _goal_hv_pm1 are the primary float32 soft-bundle state
    - uint64 versions (_rule_bundle, goal_hv) derived from sign(pm1) for API compat
    """

    W_CONSISTENCY = 0.35
    W_GOAL_SIM    = 0.30
    W_TRAJ_SLOPE  = 0.20
    W_ENTROPY     = 0.10
    W_RESONANCE   = 0.05

    def __init__(
        self,
        codebook    : Codebook,
        n_axes      : int   = 19,
        n_hyp       : int   = 200,
        max_iters   : int   = 40,
        noise_rate  : float = 0.05,
        axis_offset : int   = 0,
        pointer_mask: Optional[np.ndarray] = None,
    ):
        self.cb          = codebook
        self.W           = codebook.W
        self.H           = n_hyp
        self.max_iters   = max_iters
        self.noise_rate  = noise_rate
        self.axis_offset = axis_offset
        self.pointer_mask: Optional[np.ndarray] = (
            pointer_mask.astype(np.uint64) if pointer_mask is not None else None
        )

        self.manifold   = ManifoldAxes(n_axes, self.W, axis_offset=axis_offset)
        self.z_signal   = ZSignal()

        # ── Primary state: float32 soft bundles (eigen convergence) ──────
        n_bits = self.W * 64
        self._rule_bundle_pm1 : np.ndarray = np.zeros(n_bits, dtype=np.float32)
        self._goal_hv_pm1     : np.ndarray = np.zeros(n_bits, dtype=np.float32)

        # FIX #6: Cache CB_pm1 and EigenTrainer to avoid recomputing on every
        # train_on_tokens() call. The codebook never changes after __init__,
        # so this is safe to cache permanently.
        self._cb_pm1         : Optional[np.ndarray] = None   # (vocab, n_bits) float32
        self._eigen_trainer  : Optional[object]     = None   # EigenTrainer instance

        # ── Derived uint64 state (kept for external API compatibility) ────
        self._rule_bundle  = np.zeros(self.W, dtype=np.uint64)
        self._rule_weight  = 0.0

        self.goal_hv       : Optional[np.ndarray] = None
        self._goal_weight  : float = 0.0
        self._base_goal_weight: float = 0.0

        self._step         = 0
        self._last_S       = 0.0
        self._last_Z       : float = 0.0
        self._prev_fwd_hv  : Optional[np.ndarray] = None

        self._prev_bwd_hv     : Optional[np.ndarray] = None
        self._prev_present_hv : Optional[np.ndarray] = None
        self._prev_traj_slope : float = 0.0
        self._last_consistency: float = 0.5
        self._last_action_id  : Optional[int] = None
        self._last_action_tokens: list = []

        self.chain_memory = ChainManifold(
            codebook   = codebook,
            n_hyp      = max(10, n_hyp // 10),
            max_iters  = max(5, max_iters // 4),
            n_axes     = min(8, n_axes),
            noise_rate = noise_rate,
        )

        # ── Eigen teleport engine ─────────────────────────────────────────
        if _EIGEN_AVAILABLE:
            self._teleport = FullTeleportStep(
                n_words    = self.W,
                n_axes     = n_axes,
                n_hyp      = n_hyp,
                noise_rate = noise_rate,
                inertia    = 0.1,
            )
        else:
            self._teleport = None

        # ── EigenSafetyOxytocin: thalamic safety + oxytocin steering ─────
        if _SAFETY_OXY_AVAILABLE and EigenSafetyOxytocin is not None:
            self._safety_oxy: Optional[object] = EigenSafetyOxytocin(
                n_bits = self.W * 64
            )
        else:
            self._safety_oxy = None

    def _encode(self, tokens: list) -> np.ndarray:
        hv = self.cb.encode(tokens)
        if self.pointer_mask is not None:
            hv = np.bitwise_xor(hv, self.pointer_mask).astype(np.uint64)
        return hv

    def train_on_tokens(
        self,
        tokens: np.ndarray,
        bigram_freq: np.ndarray,
        chunk_size: int = 500_000,
        verbose: bool = True,
    ) -> None:
        """Eigen-absorbed training pass over token sequence.

        Replaces the Python ``for i in range(N)`` loop with a single
        reward-weighted matrix multiply via ``EigenTrainer.absorb_bigrams()``.

        Mathematical equivalence
        ────────────────────────
        The per-bigram ``observe()`` loop computes:
            rule_hv[i] = CB[t_next[i]]   (before XOR action XOR after = CB[t+1])
            bundle_pm1 ← EMA(bundle_pm1, rule_pm1[i], weight=reward[i])

        The fixed point of this recurrence is:
            bundle_pm1* = sign( Σ_i reward[i] × CB_pm1[t_next[i]] )
                        = sign( token_reward_sums @ CB_pm1 )

        This is computed in O(N) accumulation + O(vocab_size × n_bits) matmul,
        replacing ~62.5M Python iterations with a single numpy operation.

        Performance
        ───────────
        Before: ~190–310 s per rank (Python for-loop, 62.5M bigrams)
        After:  ~1–3 s per rank (np.add.at + matmul)

        Args:
            tokens      : (N,) int array of token IDs
            bigram_freq : (vocab_size, vocab_size) float32 — P(b|a)
            chunk_size  : Chunk size for progress reporting (chunked path)
            verbose     : Print progress
        """
        N = len(tokens)
        if N < 2:
            return

        t_prev_arr = tokens[:-1].astype(np.int32)
        t_next_arr = tokens[1:].astype(np.int32)
        rewards    = bigram_freq[t_prev_arr, t_next_arr].astype(np.float32) * 100.0

        if verbose:
            print(f"[BiDirHDC] Eigen-absorbing {N-1:,} bigrams "
                  f"(vocab={self.cb.vocab_size}, n_bits={self.W*64:,})...")

        if _EIGEN_AVAILABLE and EigenTrainer is not None:
            # ── Fast eigen path: single matmul replaces Python loop ───────
            # FIX #6: Reuse cached EigenTrainer — codebook never changes
            if self._eigen_trainer is None:
                self._eigen_trainer = EigenTrainer.from_codebook_uint64(
                    codebook_vecs  = self.cb.vecs,
                    goal_threshold = 10.0,
                )
            trainer = self._eigen_trainer
            result = trainer.absorb_bigrams_chunked(
                t_prev_arr = t_prev_arr,
                t_next_arr = t_next_arr,
                rewards    = rewards,
                chunk_size = chunk_size,
                verbose    = verbose,
            )

            # Apply fixed-point results to engine state
            self._rule_bundle_pm1 = result['rule_bundle_pm1']
            self._rule_weight     = result['rule_weight']
            # Sync uint64 rule bundle for external API compatibility
            self._rule_bundle = pm1_to_uint64(np.sign(self._rule_bundle_pm1))

            if result['goal_weight'] > 0.0:
                self._goal_hv_pm1      = result['goal_hv_pm1']
                self._goal_weight      = result['goal_weight']
                self._base_goal_weight = result['goal_weight']
                self.goal_hv = pm1_to_uint64(np.sign(self._goal_hv_pm1))

            if verbose:
                print(f"[BiDirHDC] Eigen training complete: "
                      f"rule_weight={result['rule_weight']:.1f}, "
                      f"goal_bigrams={result['n_goal_bigrams']:,}")

        else:
            # ── Fallback: original Python loop (when eigen not available) ─
            if verbose:
                print(f"[BiDirHDC] Falling back to Python loop "
                      f"(EigenTrainer not available)...")
            for chunk_start in range(0, N - 1, chunk_size):
                chunk_end = min(chunk_start + chunk_size, N - 1)
                tp = t_prev_arr[chunk_start:chunk_end]
                tn = t_next_arr[chunk_start:chunk_end]
                rw = rewards[chunk_start:chunk_end]

                for i in range(len(tp)):
                    before_hv  = self._encode([int(tp[i])])
                    action_hv  = self._encode([int(tp[i])])
                    after_hv   = self._encode([int(tn[i])])
                    rule_hv    = np.bitwise_xor(np.bitwise_xor(before_hv, action_hv), after_hv)
                    weight     = float(rw[i])
                    alpha = weight / (self._rule_weight + weight + EPS)
                    flip  = np.random.random(self.W) < alpha
                    self._rule_bundle = np.where(
                        flip, np.bitwise_xor(self._rule_bundle, rule_hv), self._rule_bundle
                    ).astype(np.uint64)
                    self._rule_weight += weight

                if verbose:
                    pct = 100.0 * chunk_end / (N - 1)
                    print(f"[BiDirHDC] {chunk_end:,}/{N-1:,} ({pct:.1f}%)")

    def vote_scores_vectorised(
        self,
        prev_tokens: np.ndarray,
    ) -> np.ndarray:
        """Vectorised bilateral scoring for all vocab tokens.

        GPU path (when CUDA available):
            Dispatches to gpu_vote_scores_vectorised() which uses float16
            tensor cores (cuBLAS HGEMM) for the bilateral matmuls.
            ~500× faster than the (batch, vocab, n_words) XOR + unpackbits.

        CPU fallback:
            Same (batch, vocab, n_words) XOR + unpackbits as before.

        For each prev_token t:
            query_hv = codebook[t] XOR rule_bundle
            fwd_scores[v] = cosine(query_hv, codebook[v])  for all v
            bwd_scores[v] = cosine(codebook[v] XOR rule_bundle, codebook[t])  for all v
            consistency[v] = (fwd_scores[v] + bwd_scores[v]) / 2
            probs = softmax(consistency)

        Args:
            prev_tokens : (batch,) int32 — previous token IDs

        Returns:
            (batch, vocab_size) float32 — probability distribution over next tokens
        """
        # GPU fast path: float16 tensor cores on H100
        if _GPU_AVAILABLE and _gpu_vote_scores is not None:
            return _gpu_vote_scores(
                codebook_vecs = self.cb.vecs,
                rule_bundle   = self._rule_bundle,
                prev_tokens   = prev_tokens,
            )

        # CPU fallback: (batch, vocab, n_words) XOR + unpackbits
        cb   = self.cb.vecs                          # (vocab_size, n_words) uint64
        rb   = self._rule_bundle                     # (n_words,) uint64
        vocab_size, n_words = cb.shape
        half = float(n_words * 32)                   # n_words × 64 / 2

        # Forward: query_hv = codebook[prev_t] XOR rule_bundle
        query_hvs = cb[prev_tokens] ^ rb[None, :]    # (batch, n_words)

        # fwd_scores[b, v] = cosine(query_hvs[b], cb[v])
        xor_fwd = query_hvs[:, None, :] ^ cb[None, :, :]  # (batch, vocab, n_words)
        pc_fwd  = np.unpackbits(xor_fwd.view(np.uint8), axis=2).sum(axis=2).astype(np.float32)
        fwd_scores = (half - pc_fwd) / half              # (batch, vocab)

        # Backward: bwd_hv[v] = codebook[v] XOR rule_bundle
        bwd_hvs = cb ^ rb[None, :]                       # (vocab, n_words)
        # bwd_scores[b, v] = cosine(bwd_hvs[v], codebook[prev_t])
        xor_bwd = bwd_hvs[None, :, :] ^ cb[prev_tokens][:, None, :]  # (batch, vocab, n_words)
        pc_bwd  = np.unpackbits(xor_bwd.view(np.uint8), axis=2).sum(axis=2).astype(np.float32)
        bwd_scores = (half - pc_bwd) / half              # (batch, vocab)

        # Bilateral consistency → softmax
        consistency = (fwd_scores + bwd_scores) / 2.0   # (batch, vocab_size)
        consistency -= consistency.max(axis=1, keepdims=True)
        probs = np.exp(consistency)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

    def step(
        self,
        present_tokens  : list,
        action_token_map: dict,
        actual_next_tokens: Optional[list] = None,
        trust           : float = 1.0,
        safety          : float = 1.0,
    ) -> "InferResult":
        """Single step: encode → 2 cosines → 1 teleport → state copies.

        Uses HadamardEigenSolver for instant fixed-point.
        """
        self._step += 1

        # ── 1. Encode inputs (O(K × W) — unavoidable) ────────────────────
        present_hv = self._encode(present_tokens)
        action_hvs = {aid: self._encode(toks) for aid, toks in action_token_map.items()}
        action_ids = list(action_hvs.keys())

        # ── 2. Retrodiction + surprise (O(W) cosine — unavoidable) ───────
        retrodiction_accuracy = 0.5
        if self._prev_bwd_hv is not None and self._prev_present_hv is not None:
            retrodiction_accuracy = _cosine_single(self._prev_bwd_hv, self._prev_present_hv)

        surprise = 0.0
        actual_hv: Optional[np.ndarray] = None
        if actual_next_tokens is not None and self._prev_fwd_hv is not None:
            actual_hv = self._encode(actual_next_tokens)
            surprise  = float(1.0 - _cosine_batch(actual_hv[np.newaxis],
                                                    self._prev_fwd_hv[np.newaxis])[0])

        self._prev_present_hv = present_hv.copy()

        # ── 3. SINGLE TELEPORT (eigen path) ──────────────────────────────
        full_mask = np.bitwise_xor(self.manifold.compass_mask, self.manifold.proto_mask)

        # ── Thalamic safety: compute context_score + ego_drift from prev h* ──
        # Zero latency — uses cached _prev_fwd_hv from last step
        _context_score = 0.0
        _ego_drift_val = 0.0
        if self._safety_oxy is not None and self._prev_fwd_hv is not None:
            _prev_pm1 = uint64_to_pm1(self._prev_fwd_hv)
            _context_score = self._safety_oxy.compute_danger_score(_prev_pm1)
            _ego_drift_val = self._safety_oxy.compute_ego_drift(_prev_pm1)

        # ── Get all 4 steering terms with context-adaptive weights ────
        _steering = (
            self._safety_oxy.get_steering_terms(
                context_score = _context_score,
                ego_drift     = _ego_drift_val,
            )
            if self._safety_oxy is not None
            else {}
        )

        result = self._teleport.run_full(
            present_hv              = present_hv,
            action_hvs              = action_hvs,
            action_ids              = action_ids,
            axes_hv                 = self.manifold.all_axes_hv(),
            goal_hv_pm1             = self._goal_hv_pm1,
            rule_bundle_pm1         = self._rule_bundle_pm1,
            chain_memory            = self.chain_memory,
            S                       = self._last_S,
            Z_current               = self._last_Z,
            base_goal_weight        = self._base_goal_weight,
            accumulated_goal_weight = self._goal_weight,
            rule_weight             = self._rule_weight,
            prev_traj_slope         = self._prev_traj_slope,
            retrodiction_accuracy   = retrodiction_accuracy,
            surprise                = surprise,
            trust                   = trust,
            safety                  = safety,
            n_hyp                   = self.H,
            noise_rate              = self.noise_rate,
            step                    = self._step,
            z_signal                = self.z_signal,
            full_mask               = full_mask,
            danger_pm1              = _steering.get('danger_pm1'),
            danger_weight           = _steering.get('w_danger', 0.0),
            oxytocin_pm1            = _steering.get('oxytocin_pm1'),
            oxytocin_weight         = _steering.get('w_oxy', 0.0),
            ego_pm1                 = _steering.get('ego_pm1'),
            ego_weight              = _steering.get('w_ego', 0.0),
            norm_pm1                = _steering.get('norm_pm1'),
            norm_weight             = _steering.get('w_norm', 0.0),
        )

        # ── 4. Store state (O(W) copies — unavoidable) ───────────────
        self._last_S              = result.S_new
        self._last_Z              = result.Z_new
        self._prev_traj_slope     = result.traj_slope
        self._prev_bwd_hv         = result.best_bwd_hv.copy()
        self._prev_fwd_hv         = result.best_fwd_hv.copy()
        self._last_consistency    = result.consistency
        self._last_action_id      = result.best_action
        self._last_action_tokens  = action_token_map.get(result.best_action,
                                                          [result.best_action])
        # Update soft bundles (primary state)
        self._goal_hv_pm1         = result.updated_goal_hv_pm1
        self._goal_weight         = result.updated_goal_weight
        self._rule_bundle_pm1     = result.updated_rule_bundle_pm1
        self._rule_weight         = result.updated_rule_weight
        # Keep uint64 versions in sync for external API compatibility
        self.goal_hv              = pm1_to_uint64(np.sign(result.updated_goal_hv_pm1))
        self._rule_bundle         = pm1_to_uint64(np.sign(result.updated_rule_bundle_pm1))

        # ── 4b. Update EigenSafetyOxytocin prototypes from step result ──
        if self._safety_oxy is not None:
            _best_h_pm1 = uint64_to_pm1(result.best_fwd_hv)
            self._safety_oxy.update_from_step(
                h_star_pm1    = _best_h_pm1,
                safety_scalar = safety,
                resonance     = result.resonance,
            )

        # ── 5. Chain memory validation gate (post-hoc bookkeeping) ───
        if (self._last_action_id is not None and
                self._prev_fwd_hv is not None and
                actual_hv is not None):
            forward_acc = 1.0 - surprise
            if (forward_acc > FORWARD_THRESHOLD and
                    retrodiction_accuracy > RETRO_THRESHOLD and
                    self._last_consistency > 0.65):
                action_a_hv = self.cb.encode(self._last_action_tokens)
                self.chain_memory.observe(
                    action_a_hv  = action_a_hv,
                    action_b_id  = result.best_action,
                    world_fwd_hv = result.best_fwd_hv,
                    weight       = result.resonance,
                )

        traj_accel = result.traj_slope - self._prev_traj_slope

        # Compute safety_score from EigenSafetyOxytocin (O(1) — cached scalar)
        _safety_score = (
            self._safety_oxy.get_safety_scalar()
            if self._safety_oxy is not None
            else 0.5
        )

        return InferResult(
            best_action           = result.best_action,
            joint_scores          = result.joint_scores,
            consistency           = result.consistency,
            goal_sim              = result.goal_sim,
            traj_slope            = result.traj_slope,
            surprise              = surprise,
            entropy               = result.entropy,
            resonance             = result.resonance,
            Z                     = result.Z_new,
            S                     = result.S_new,
            best_fwd_hv           = result.best_fwd_hv,
            best_bwd_hv           = result.best_bwd_hv,
            retrodiction_accuracy = retrodiction_accuracy,
            traj_accel            = traj_accel,
            chain_hits            = result.chain_hits,
            safety_score          = _safety_score,
        )

    def reset_level(self):
        self.z_signal.reset()
        self._prev_fwd_hv     = None
        self._prev_bwd_hv     = None
        self._prev_present_hv = None
        self._prev_traj_slope = 0.0
        self._last_S          = 0.0
        self._last_Z          = 0.0
        self._last_consistency = 0.5
        self._last_action_id   = None
        self._last_action_tokens = []
        # Reset soft-bundle state (eigen convergence)
        n_bits = self.W * 64
        self._rule_bundle_pm1 = np.zeros(n_bits, dtype=np.float32)
        self._goal_hv_pm1     = np.zeros(n_bits, dtype=np.float32)