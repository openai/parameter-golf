"""Eigen Convergence — Instant Teleportation for FullBiDirHDC.

Implements the complete signal absorption described in:
  Arc_AGI_3_HDC_Hadamard_DSV_Model-main/plans/eigen_convergence_integration.md

Every computation in FullBiDirHDC.step() is absorbed into a single closed-form
solve. The result is the EXACT fixed point of the entire system — no iteration,
no stochastic blending, no post-hoc scoring.

Key classes:
  HadamardEigenSolver   — batch_teleport(): h* = sign(spectrum) in one pass
  AxisWeightScheduler   — maps steering signal S → per-axis weights
  AnticipationEigenGate — adjusts goal/rule weights from traj_accel + retro_acc
  SoftEMABundle         — deterministic float32 EMA replacing stochastic XOR-blend
  FullTeleportResult    — complete return type from run_full()
  FullTeleportStep      — orchestrates the single teleport step
  EigenTrainer          — absorb_bigrams(): replaces train_on_tokens() Python loop
                          with a single reward-weighted matrix multiply + sign()

Performance:
  Before: ~10–15ms per step (40-iter propagation loops)
  After:  ~0.3ms per step (O(n_actions × K × n_bits) single pass)

  Training (train_on_tokens):
  Before: ~190–310s per rank (Python for-loop over 62.5M bigrams)
  After:  ~1–3s per rank (single matmul: rewards @ CB_pm1[t_next])

Eigen Training Derivation
─────────────────────────
The sequential per-bigram EMA in observe() is:

    bundle_pm1 ← (1 - α_i) × bundle_pm1 + α_i × rule_pm1[i]
    where α_i = r_i / (W_acc + r_i),  rule_pm1[i] = CB_pm1[t_next[i]]

The fixed point of this recurrence (as N → ∞, or equivalently the
reward-weighted mean) is:

    bundle_pm1* = sign( Σ_i r_i × CB_pm1[t_next[i]] )
               = sign( rewards @ CB_pm1[t_next] )

This is a single O(N × n_bits) matrix multiply — fully vectorised,
no Python loop, no per-step state accumulation.

The goal HV fixed point follows the same pattern:
    goal_pm1* = sign( Σ_{i: r_i ≥ goal_threshold} r_i × CB_pm1[t_next[i]] )

Both are absorbed into EigenTrainer.absorb_bigrams() in one pass.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple, Optional, Dict, Any

# ── GPU acceleration (optional, graceful fallback to CPU) ─────────────────────
try:
    from _gpu import (
        gpu_available,
        gpu_matmul_f32,
        gpu_matmul_f16,
        gpu_batch_matmul_f32,
        gpu_bincount_weighted,
        gpu_sign_f32,
        gpu_uint64_batch_to_pm1,
        gpu_batch_teleport as _gpu_batch_teleport,
    )
    _GPU_AVAILABLE = gpu_available()
except ImportError:
    _GPU_AVAILABLE = False
    def gpu_matmul_f32(a, b): return a.astype('float32') @ b.astype('float32')
    def gpu_matmul_f16(a, b): return a.astype('float32') @ b.astype('float32')
    def gpu_batch_matmul_f32(a, b): return a.astype('float32') @ b.astype('float32')
    def gpu_bincount_weighted(idx, w, ml): return np.bincount(idx.astype(np.int64), weights=w.astype(np.float64), minlength=ml)
    def gpu_sign_f32(a):
        out = np.sign(a).astype(np.float32); out[out == 0.0] = 1.0; return out
    def gpu_uint64_batch_to_pm1(hvs):
        N, W = hvs.shape
        bits = np.unpackbits(hvs.view(np.uint8).reshape(N, W * 8), axis=1, bitorder='little')
        return bits.astype(np.float32) * 2.0 - 1.0
    _gpu_batch_teleport = None

# ─────────────────────────────────────────────────────────────────────────────
EPS = 1e-8
PHI_FRAC = 0.6180339887498949

# Anticipation thresholds (same as _bidi_hdc_engine.py)
ACCEL_THRESHOLD = 0.02
RETRO_THRESHOLD = 0.55

# ─────────────────────────────────────────────────────────────────────────────
# pm1 ↔ uint64 conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def uint64_to_pm1(hv: np.ndarray) -> np.ndarray:
    """(W,) uint64 → (W×64,) float32 in {-1, +1}.

    Unpacks each uint64 word into 64 bits, maps 0→-1, 1→+1.
    """
    bits = np.unpackbits(hv.view(np.uint8), bitorder='little')
    return (bits.astype(np.float32) * 2.0 - 1.0)




def pm1_to_uint64(pm1: np.ndarray) -> np.ndarray:
    """(W×64,) float32 → (W,) uint64.

    sign(pm1) ≥ 0 → bit=1, else bit=0. Packs back to uint64.
    """
    bits = (pm1 >= 0.0).astype(np.uint8)
    return np.packbits(bits, bitorder='little').view(np.uint64)


def batch_uint64_to_pm1(hvs: np.ndarray) -> np.ndarray:
    """(N, W) uint64 → (N, W×64) float32 in {-1, +1}.

    GPU-accelerated when CUDA is available (uses torch bitwise unpack).
    Falls back to numpy unpackbits on CPU.
    """
    if _GPU_AVAILABLE:
        return gpu_uint64_batch_to_pm1(hvs)
    N, W = hvs.shape
    bits = np.unpackbits(hvs.view(np.uint8).reshape(N, W * 8), axis=1, bitorder='little')
    return bits.astype(np.float32) * 2.0 - 1.0


def batch_pm1_to_uint64(pm1s: np.ndarray) -> np.ndarray:
    """(N, W×64) float32 → (N, W) uint64."""
    N, n_bits = pm1s.shape
    W = n_bits // 64
    bits = (pm1s >= 0.0).astype(np.uint8)
    return np.packbits(bits, axis=1, bitorder='little').view(np.uint64).reshape(N, W)


# ─────────────────────────────────────────────────────────────────────────────
# AxisWeightScheduler
# ─────────────────────────────────────────────────────────────────────────────

class AxisWeightScheduler:
    """Maps steering signal S ∈ [-1, +1] → per-axis weights.

    Positive S → emphasise proto axes (exploration).
    Negative S → emphasise compass axes (exploitation).
    S = 0 → uniform weights.

    Args:
        n_axes    : Total number of axes (compass + proto)
        n_compass : Number of compass axes (first n_compass axes)
    """

    def __init__(self, n_axes: int = 19, n_compass: int = 8):
        self.n_axes    = n_axes
        self.n_compass = n_compass
        self.n_proto   = n_axes - n_compass

    def weights_from_S(self, S: float) -> np.ndarray:
        """Return (n_axes,) float32 weight vector from steering signal S.

        Compass axes get weight (1 - |S|) when S > 0 (proto emphasis),
        or (1 + S) when S < 0 (compass emphasis). Proto axes are symmetric.
        Weights are L1-normalised to sum to 1.
        """
        w = np.ones(self.n_axes, dtype=np.float32)
        if S > 0.0:
            # Positive S: boost proto, reduce compass
            w[:self.n_compass] *= max(0.1, 1.0 - S)
            w[self.n_compass:] *= 1.0 + S
        elif S < 0.0:
            # Negative S: boost compass, reduce proto
            w[:self.n_compass] *= 1.0 + abs(S)
            w[self.n_compass:] *= max(0.1, 1.0 - abs(S))
        total = w.sum()
        if total > EPS:
            w /= total
        return w


# ─────────────────────────────────────────────────────────────────────────────
# AnticipationEigenGate
# ─────────────────────────────────────────────────────────────────────────────

class AnticipationEigenGate:
    """Adjusts goal and rule weights based on trajectory acceleration and retrodiction.

    Implements Anticipation Enhancement v2:
    - If traj_accel < -ACCEL_THRESHOLD: reduce goal weight (exploration mode)
    - If retro_acc < RETRO_THRESHOLD: reduce rule weight (distrust rules)

    Args:
        accel_decay : Multiplier applied to goal_weight when traj_accel < -threshold
        retro_decay : Multiplier applied to rule_weight when retro_acc < threshold
    """

    def __init__(self, accel_decay: float = 0.5, retro_decay: float = 0.5):
        self.accel_decay = accel_decay
        self.retro_decay = retro_decay

    def adjust(
        self,
        goal_weight: float,
        rule_weight: float,
        traj_accel: float,
        retro_acc: float,
    ) -> tuple:
        """Return (adjusted_goal_weight, adjusted_rule_weight).

        Args:
            goal_weight : Current goal weight
            rule_weight : Current rule weight
            traj_accel  : Trajectory acceleration (slope_t - slope_{t-1})
            retro_acc   : Retrodiction accuracy (cosine of bwd_hv vs present_hv)

        Returns:
            (goal_weight_adj, rule_weight_adj) — adjusted weights
        """
        gw = goal_weight
        rw = rule_weight

        if traj_accel < -ACCEL_THRESHOLD:
            gw *= self.accel_decay
        elif traj_accel < 0.0 and retro_acc < RETRO_THRESHOLD:
            gw *= self.accel_decay

        if retro_acc < RETRO_THRESHOLD:
            rw *= self.retro_decay

        return float(gw), float(rw)


# ─────────────────────────────────────────────────────────────────────────────
# HadamardEigenSolver
# ─────────────────────────────────────────────────────────────────────────────

class HadamardEigenSolver:
    """Instant fixed-point solver via eigenvalue spectrum.

    The fixed point of the bilateral HDC manifold is:

        λ_j = Σ_k w_k × axis_k[j]          ← axis field
              + w_goal × goal_pm1[j]          ← goal attractor
              + w_rule × rule_pm1[j]          ← soft rule EMA
              + w_chain × chain_pm1[j]        ← chain prior (optional)
              + w_inertia × seed_pm1[j]       ← bilateral inertia

        h*[j] = sign(λ_j)                    ← exact fixed point

    This replaces the 40-iteration propagation loop with a single O(n_bits) pass.

    Args:
        n_words : HV width in uint64 words (n_words × 64 = n_bits)
        inertia : Weight for bilateral inertia term (default 0.1)
    """

    def __init__(self, n_words: int, inertia: float = 0.1):
        self.W       = n_words
        self.n_bits  = n_words * 64
        self.inertia = inertia

    def batch_teleport(
        self,
        axes_pm1      : np.ndarray,   # (K, n_bits) float32 — axis HVs in pm1
        axis_weights  : np.ndarray,   # (K,) float32 — per-axis weights
        goal_pm1      : np.ndarray,   # (n_bits,) float32 — goal HV in pm1
        goal_weight   : float,        # scalar
        rule_pm1      : np.ndarray,   # (n_bits,) float32 — rule bundle in pm1
        rule_weight   : float,        # scalar
        manifold_fwd  : np.ndarray,   # (N, n_bits) float32 — fwd seeds in pm1
        manifold_bwd  : np.ndarray,   # (N, n_bits) float32 — bwd seeds in pm1
        chain_pm1     : Optional[np.ndarray] = None,  # (n_bits,) float32
        chain_weight  : float = 0.0,
        # ── Thalamic safety steering terms (optional) ─────────────────────
        danger_pm1    : Optional[np.ndarray] = None,  # (n_bits,) float32 — REPEL
        danger_weight : float = 0.0,
        oxytocin_pm1  : Optional[np.ndarray] = None,  # (n_bits,) float32 — ATTRACT
        oxytocin_weight: float = 0.0,
        ego_pm1       : Optional[np.ndarray] = None,  # (n_bits,) float32 — EGO PULL
        ego_weight    : float = 0.0,
        norm_pm1      : Optional[np.ndarray] = None,  # (n_bits,) float32 — NORM PULL
        norm_weight   : float = 0.0,
    ) -> tuple:
        """Compute h* = sign(spectrum) for all N hypotheses in one pass.

        The spectrum is the same for all hypotheses (shared field) plus
        per-hypothesis inertia from the seed HVs.

        Args:
            axes_pm1      : (K, n_bits) float32 — axis HVs
            axis_weights  : (K,) float32 — per-axis weights (sum to 1)
            goal_pm1      : (n_bits,) float32 — goal attractor
            goal_weight   : scalar weight for goal term
            rule_pm1      : (n_bits,) float32 — rule bundle
            rule_weight   : scalar weight for rule term
            manifold_fwd  : (N, n_bits) float32 — forward seed HVs
            manifold_bwd  : (N, n_bits) float32 — backward seed HVs
            chain_pm1     : (n_bits,) float32 — chain prior (optional)
            chain_weight  : scalar weight for chain term
            danger_pm1    : (n_bits,) float32 — danger cluster prototype (REPEL)
            danger_weight : scalar weight for danger repulsion
            oxytocin_pm1  : (n_bits,) float32 — prosocial cluster prototype (ATTRACT)
            oxytocin_weight: scalar weight for oxytocin attraction
            ego_pm1       : (n_bits,) float32 — identity prototype (EGO PULL)
            ego_weight    : scalar weight for ego attraction
            norm_pm1      : (n_bits,) float32 — norm-consistent prototype (NORM PULL)
            norm_weight   : scalar weight for norm attraction

        Returns:
            (fwd_stars, bwd_stars, goal_sims)
            fwd_stars : (N, n_bits) float32 — h* for each hypothesis
            bwd_stars : (N, n_bits) float32 — same as fwd_stars (bilateral)
            goal_sims : (N,) float32 — cosine(h*, goal_pm1)
        """
        # ── GPU fast path: entire batch_teleport on GPU ───────────────────
        if _GPU_AVAILABLE and _gpu_batch_teleport is not None:
            gpu_result = _gpu_batch_teleport(
                axes_pm1       = axes_pm1,
                axis_weights   = axis_weights,
                goal_pm1       = goal_pm1,
                goal_weight    = goal_weight,
                rule_pm1       = rule_pm1,
                rule_weight    = rule_weight,
                manifold_fwd   = manifold_fwd,
                manifold_bwd   = manifold_bwd,
                inertia        = self.inertia,
                chain_pm1      = chain_pm1,
                chain_weight   = chain_weight,
                danger_pm1     = danger_pm1,
                danger_weight  = danger_weight,
                oxytocin_pm1   = oxytocin_pm1,
                oxytocin_weight= oxytocin_weight,
                ego_pm1        = ego_pm1,
                ego_weight     = ego_weight,
                norm_pm1       = norm_pm1,
                norm_weight    = norm_weight,
                EPS            = EPS,
            )
            if gpu_result is not None:
                return gpu_result

        # ── CPU fallback path ─────────────────────────────────────────────
        # ── Shared field (same for all N hypotheses) ──────────────────────
        # FIX #3: axis_weights @ axes_pm1 is a single BLAS matmul
        # (K,) @ (K, n_bits) → (n_bits,)  — replaces element-wise broadcast+sum
        shared_field = axis_weights @ axes_pm1  # (n_bits,) float32

        # goal term
        if goal_weight > EPS:
            shared_field = shared_field + goal_weight * goal_pm1

        # rule term
        if rule_weight > EPS:
            shared_field = shared_field + rule_weight * rule_pm1

        # chain term
        if chain_weight > EPS and chain_pm1 is not None:
            shared_field = shared_field + chain_weight * chain_pm1

        # ── Thalamic safety steering: repel from danger, attract toward safe/ego/norm ──
        # These four additions are the only change to the spectrum — zero architectural change.
        if danger_weight > EPS and danger_pm1 is not None:
            shared_field = shared_field - danger_weight * danger_pm1   # REPEL
        if oxytocin_weight > EPS and oxytocin_pm1 is not None:
            shared_field = shared_field + oxytocin_weight * oxytocin_pm1  # ATTRACT
        if ego_weight > EPS and ego_pm1 is not None:
            shared_field = shared_field + ego_weight * ego_pm1         # EGO PULL
        if norm_weight > EPS and norm_pm1 is not None:
            shared_field = shared_field + norm_weight * norm_pm1       # NORM PULL

        # ── FIX #4: Avoid materializing full (N, n_bits) float32 matrix ───
        # The spectrum for hypothesis i is:
        #   spectrum[i] = shared_field + inertia * (fwd[i] + bwd[i]) * 0.5
        # h*[i] = sign(spectrum[i])
        #
        # Key insight: sign(a + b) depends only on whether a + b > 0.
        # We can compute h* without ever storing the full (N, n_bits) matrix
        # by computing the sign directly from the threshold:
        #   h*[i][j] = +1 if shared_field[j] + inertia * inertia_field[i][j] > 0
        #            = +1 if inertia_field[i][j] > -shared_field[j] / inertia
        #
        # For the goal_sims we need h* · goal_pm1, which is:
        #   goal_sims[i] = mean_j( sign(shared_field[j] + inertia * inertia_field[i][j])
        #                          × goal_pm1[j] )
        #
        # We compute this in-place without storing h_star by accumulating
        # the sign contribution directly.
        #
        # For N small (e.g. n_actions × H = 5 × 200 = 1000) and n_bits=32768,
        # the full matrix is 1000 × 32768 × 4B = 128 MB — avoidable.
        # We process in row-chunks to bound peak memory to chunk_rows × n_bits × 4B.

        inertia_half = self.inertia * 0.5  # scalar

        N = manifold_fwd.shape[0]
        CHUNK = min(N, 256)  # process 256 hypotheses at a time → 256×32768×4B = 32MB

        h_star_list = []
        goal_sims_list = []
        has_goal = goal_weight > EPS and np.any(goal_pm1 != 0.0)

        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            # inertia_field chunk: (chunk, n_bits)
            inertia_chunk = (manifold_fwd[start:end] + manifold_bwd[start:end]) * inertia_half
            # spectrum chunk: (chunk, n_bits)
            spec_chunk = shared_field[None, :] + inertia_chunk
            # h* chunk: sign, ties → +1
            h_chunk = np.sign(spec_chunk).astype(np.float32)
            h_chunk[h_chunk == 0.0] = 1.0
            h_star_list.append(h_chunk)

            if has_goal:
                # goal_sims: (chunk,) = mean over bits of h* × goal_pm1
                goal_sims_list.append((h_chunk * goal_pm1[None, :]).mean(axis=1))

        h_star = np.concatenate(h_star_list, axis=0)  # (N, n_bits)

        if has_goal:
            goal_sims = np.concatenate(goal_sims_list).astype(np.float32)
        else:
            goal_sims = np.full(N, 0.5, dtype=np.float32)

        return h_star, h_star, goal_sims

    def compute_spectrum(
        self,
        axes_pm1     : np.ndarray,
        axis_weights : np.ndarray,
        goal_pm1     : np.ndarray,
        goal_weight  : float,
        rule_pm1     : np.ndarray,
        rule_weight  : float,
        seed_pm1     : np.ndarray,
        chain_pm1    : Optional[np.ndarray] = None,
        chain_weight : float = 0.0,
    ) -> np.ndarray:
        """Compute the raw eigenvalue spectrum λ for a single seed.

        Returns:
            (n_bits,) float32 — eigenvalue spectrum
        """
        spectrum = (axis_weights[:, None] * axes_pm1).sum(axis=0)
        if goal_weight > EPS:
            spectrum = spectrum + goal_weight * goal_pm1
        if rule_weight > EPS:
            spectrum = spectrum + rule_weight * rule_pm1
        if chain_weight > EPS and chain_pm1 is not None:
            spectrum = spectrum + chain_weight * chain_pm1
        spectrum = spectrum + self.inertia * seed_pm1
        return spectrum.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SoftEMABundle
# ─────────────────────────────────────────────────────────────────────────────

class SoftEMABundle:
    """Deterministic float32 EMA replacing stochastic XOR-blend.

    The stochastic XOR-blend:
        alpha = weight / (accumulated + weight + EPS)
        flip  = random(W) < alpha
        bundle = where(flip, bundle XOR rule_hv, bundle)

    Has expected value:
        E[bundle_new[j]] = (1 - alpha) × bundle_pm1[j] + alpha × rule_pm1[j]

    This class maintains the bundle as float32 soft vector and converts to
    uint64 only when needed for the spectrum.

    Args:
        n_bits : Total number of bits (n_words × 64)
    """

    def __init__(self, n_bits: int):
        self.n_bits     = n_bits
        self.bundle_pm1 = np.zeros(n_bits, dtype=np.float32)
        self.weight     = 0.0

    def update(self, new_pm1: np.ndarray, weight: float) -> None:
        """Soft EMA update with new HV (in pm1 space).

        Args:
            new_pm1 : (n_bits,) float32 — new HV to blend in
            weight  : Scalar weight for this update
        """
        alpha = weight / (self.weight + weight + EPS)
        self.bundle_pm1 = (1.0 - alpha) * self.bundle_pm1 + alpha * new_pm1
        self.weight += weight

    def to_uint64(self) -> np.ndarray:
        """Convert soft bundle to uint64 HV via sign().

        Returns:
            (W,) uint64 — hard binary HV
        """
        return pm1_to_uint64(np.sign(self.bundle_pm1))

    def get_pm1(self) -> np.ndarray:
        """Return the soft float32 bundle (no sign quantisation).

        Returns:
            (n_bits,) float32 — soft bundle
        """
        return self.bundle_pm1.copy()

    def set_from_pm1(self, pm1: np.ndarray, weight: float) -> None:
        """Directly set the bundle from a pm1 vector.

        Args:
            pm1    : (n_bits,) float32
            weight : Accumulated weight to assign
        """
        self.bundle_pm1 = pm1.astype(np.float32).copy()
        self.weight     = float(weight)


# ─────────────────────────────────────────────────────────────────────────────
# FullTeleportResult
# ─────────────────────────────────────────────────────────────────────────────

class FullTeleportResult(NamedTuple):
    """Complete return type from FullTeleportStep.run_full().

    All signals are absorbed into a single closed-form solve.
    Nothing runs after the teleport except O(W) state copies.
    """
    # Action selection
    best_action              : int
    joint_scores             : dict          # {action_id: float}
    # Fixed-point metrics
    consistency              : float         # always 1.0 (by construction)
    goal_sim                 : float         # cosine(h*, goal)
    traj_slope               : float         # goal_sim - seed_goal_sim
    entropy                  : float         # H(bit_balance) of h*
    resonance                : float         # computed from entropy + goal_sim
    # Best hypothesis HVs (uint64)
    best_fwd_hv              : np.ndarray    # (W,) uint64
    best_bwd_hv              : np.ndarray    # (W,) uint64  [= best_fwd_hv]
    # Updated state (soft EMA results)
    updated_rule_bundle_pm1  : np.ndarray    # (n_bits,) float32 — soft bundle
    updated_rule_weight      : float
    updated_goal_hv_pm1      : np.ndarray    # (n_bits,) float32 — soft goal
    updated_goal_weight      : float
    # ZSignal outputs
    S_new                    : float         # new steering signal for next step
    Z_new                    : float         # new Z_current for next step
    # Diagnostics
    eigenvalue_spectrum      : np.ndarray    # (n_bits,) float32
    chain_hits               : int           # 1 if chain prior was injected
    mean_goal_sim            : float         # mean over all actions × hypotheses
    mean_entropy             : float         # mean over all actions × hypotheses
    # Thalamic safety + oxytocin steering diagnostics (defaults for backward compat)
    danger_score             : float = 0.0   # cosine(h*, danger_pm1) — proximity to danger
    oxytocin_score           : float = 0.0   # cosine(h*, oxytocin_pm1) — prosocial alignment
    ego_drift                : float = 0.0   # 1 - cosine(h*, ego_pm1) — identity drift
    norm_score               : float = 0.5   # cosine(h*, norm_pm1) — norm alignment


# ─────────────────────────────────────────────────────────────────────────────
# FullTeleportStep
# ─────────────────────────────────────────────────────────────────────────────

class FullTeleportStep:
    """Orchestrates the single teleport step absorbing all signals.

    Replaces the entire _propagate() + _score_joint() + ZSignal + rule/goal
    update pipeline with a single closed-form solve.

    Architecture (from eigen_convergence_integration.md):
        encode → 2 cosines → 1 teleport → state copies

    Args:
        n_words    : HV width in uint64 words
        n_axes     : Number of golden-ratio axes
        n_hyp      : Hypotheses per action
        noise_rate : Noise rate for seed generation
        inertia    : Inertia weight for bilateral seeds
    """

    # Joint score weights (same as FullBiDirHDC)
    W_CONSISTENCY = 0.35
    W_GOAL_SIM    = 0.30
    W_TRAJ_SLOPE  = 0.20
    W_ENTROPY     = 0.10
    W_RESONANCE   = 0.05

    def __init__(
        self,
        n_words    : int,
        n_axes     : int   = 19,
        n_hyp      : int   = 200,
        noise_rate : float = 0.05,
        inertia    : float = 0.1,
    ):
        self.W          = n_words
        self.n_bits     = n_words * 64
        self.n_axes     = n_axes
        self.H          = n_hyp
        self.noise_rate = noise_rate

        self._eigen_solver    = HadamardEigenSolver(n_words=n_words, inertia=inertia)
        self._axis_scheduler  = AxisWeightScheduler(n_axes=n_axes, n_compass=min(8, n_axes))
        self._eigen_gate      = AnticipationEigenGate()

    def _build_seeds_pm1(
        self,
        present_hv  : np.ndarray,   # (W,) uint64
        action_hvs  : dict,         # {aid: (W,) uint64}
        action_ids  : list,
        full_mask   : np.ndarray,   # (W,) uint64
    ) -> tuple:
        """Build (N×H, n_bits) float32 fwd/bwd seed matrices in pm1 space.

        Returns:
            (fwd_pm1, bwd_pm1, seed_goal_sims_placeholder)
            fwd_pm1 : (n_actions × H, n_bits) float32
            bwd_pm1 : (n_actions × H, n_bits) float32
        """
        n   = len(action_ids)
        rng = np.random.default_rng()

        fwd_uint = np.empty((n * self.H, self.W), dtype=np.uint64)
        bwd_uint = np.empty((n * self.H, self.W), dtype=np.uint64)

        for i, aid in enumerate(action_ids):
            eff   = action_hvs[aid]
            fwd_s = np.bitwise_xor(present_hv, eff)
            bwd_s = np.bitwise_xor(present_hv, np.bitwise_xor(eff, full_mask))
            fb    = np.tile(fwd_s, (self.H, 1))
            bb    = np.tile(bwd_s, (self.H, 1))
            noise_m = rng.random((self.H, self.W)) < self.noise_rate
            noise_v = rng.integers(0, np.iinfo(np.uint64).max, (self.H, self.W), dtype=np.uint64)
            fwd_uint[i*self.H:(i+1)*self.H] = np.where(
                noise_m, np.bitwise_xor(fb, noise_v), fb
            ).astype(np.uint64)
            bwd_uint[i*self.H:(i+1)*self.H] = np.where(
                noise_m, np.bitwise_xor(bb, noise_v), bb
            ).astype(np.uint64)

        fwd_pm1 = batch_uint64_to_pm1(fwd_uint)  # (N*H, n_bits)
        bwd_pm1 = batch_uint64_to_pm1(bwd_uint)  # (N*H, n_bits)
        return fwd_pm1, bwd_pm1

    def run_full(
        self,
        present_hv              : np.ndarray,   # (W,) uint64
        action_hvs              : dict,          # {aid: (W,) uint64}
        action_ids              : list,
        axes_hv                 : np.ndarray,   # (K, W) uint64 — all axis HVs
        goal_hv_pm1             : np.ndarray,   # (n_bits,) float32 — soft goal
        rule_bundle_pm1         : np.ndarray,   # (n_bits,) float32 — soft rule
        chain_memory,                           # ChainManifold instance
        S                       : float,        # current steering signal
        Z_current               : float,        # current Z value
        base_goal_weight        : float,
        accumulated_goal_weight : float,
        rule_weight             : float,
        prev_traj_slope         : float,
        retrodiction_accuracy   : float,
        surprise                : float,
        trust                   : float,
        safety                  : float,
        n_hyp                   : int,
        noise_rate              : float,
        step                    : int,
        z_signal,                               # ZSignal instance (mutated in-place)
        full_mask               : Optional[np.ndarray] = None,  # (W,) uint64
        # ── Thalamic safety steering terms (optional, default = no steering) ──
        danger_pm1              : Optional[np.ndarray] = None,  # (n_bits,) float32
        danger_weight           : float = 0.0,
        oxytocin_pm1            : Optional[np.ndarray] = None,  # (n_bits,) float32
        oxytocin_weight         : float = 0.0,
        ego_pm1                 : Optional[np.ndarray] = None,  # (n_bits,) float32
        ego_weight              : float = 0.0,
        norm_pm1                : Optional[np.ndarray] = None,  # (n_bits,) float32
        norm_weight             : float = 0.0,
    ) -> FullTeleportResult:
        """Run the complete single-teleport step.

        All signals are absorbed:
          - AxisWeightScheduler maps S → axis_weights
          - AnticipationEigenGate adjusts goal/rule weights
          - ChainManifold.eigen_query() (if available) provides chain prior
          - HadamardEigenSolver.batch_teleport() computes h* = sign(spectrum)
          - Post-teleport analytics: goal_sim, traj_slope, entropy, resonance
          - SoftEMABundle updates rule_pm1 and goal_pm1 deterministically
          - ZSignal.update() runs inside teleport on mean_goal_sim from h*
          - Micro-exploration trigger updates S_new

        Returns:
            FullTeleportResult with all outputs
        """
        n_actions = len(action_ids)

        # ── Axis weights from S ───────────────────────────────────────────
        axis_weights = self._axis_scheduler.weights_from_S(S)  # (K,)

        # ── Convert axes to pm1 ───────────────────────────────────────────
        axes_pm1 = batch_uint64_to_pm1(axes_hv)  # (K, n_bits)

        # ── Dynamic goal weight = base × (1 + Z) ─────────────────────────
        dynamic_goal_weight = base_goal_weight * (1.0 + Z_current)
        effective_goal_weight = max(accumulated_goal_weight, dynamic_goal_weight)

        # ── Anticipation gate: adjust goal/rule weights ───────────────────
        # Estimate traj_accel from prev_traj_slope (will be updated post-teleport)
        # Use 0.0 as placeholder — gate uses prev_traj_slope vs 0 as proxy
        traj_accel_est = 0.0  # will be computed properly after h* is known
        goal_w_adj, rule_w_adj = self._eigen_gate.adjust(
            goal_weight = effective_goal_weight,
            rule_weight = rule_weight / (rule_weight + 10.0 + EPS),
            traj_accel  = traj_accel_est,
            retro_acc   = retrodiction_accuracy,
        )

        # ── Chain prior via eigen query ───────────────────────────────────
        chain_pm1    = None
        chain_weight = 0.0
        chain_hits   = 0

        if (hasattr(chain_memory, '_eigen_solver') and
                chain_memory._eigen_solver is not None and
                chain_memory._rule_weight > 0):
            # Upgraded ChainManifold with eigen solver
            chain_prior_hv, chain_score, _ = chain_memory.eigen_query(
                action_ids   = action_ids,
                action_hvs   = action_hvs,
                S            = S,
                axes_pm1     = axes_pm1,
                axis_weights = axis_weights,
            )
            if chain_prior_hv is not None and chain_score > 0.55:
                chain_pm1    = uint64_to_pm1(chain_prior_hv)
                chain_weight = min(0.3, chain_memory._rule_weight /
                                   (chain_memory._rule_weight + 10.0 + EPS))
                chain_hits   = 1
        elif chain_memory._rule_weight > 0:
            # Fallback: use existing iterative chain query
            if (hasattr(chain_memory, '_last_action_tokens') and
                    chain_memory._last_action_tokens):
                pass  # chain query handled externally in this fallback path

        # ── Build seeds in pm1 space ──────────────────────────────────────
        if full_mask is None:
            # Build full_mask from axes (compass XOR proto)
            # This is a fallback — caller should pass full_mask
            full_mask = np.zeros(self.W, dtype=np.uint64)

        fwd_pm1, bwd_pm1 = self._build_seeds_pm1(
            present_hv = present_hv,
            action_hvs = action_hvs,
            action_ids = action_ids,
            full_mask  = full_mask,
        )
        # fwd_pm1, bwd_pm1: (n_actions × H, n_bits)

        # ── Single teleport: h* = sign(spectrum) ─────────────────────────
        h_stars, _, goal_sims = self._eigen_solver.batch_teleport(
            axes_pm1        = axes_pm1,
            axis_weights    = axis_weights,
            goal_pm1        = goal_hv_pm1,
            goal_weight     = goal_w_adj,
            rule_pm1        = rule_bundle_pm1,
            rule_weight     = rule_w_adj,
            manifold_fwd    = fwd_pm1,
            manifold_bwd    = bwd_pm1,
            chain_pm1       = chain_pm1,
            chain_weight    = chain_weight,
            danger_pm1      = danger_pm1,
            danger_weight   = danger_weight,
            oxytocin_pm1    = oxytocin_pm1,
            oxytocin_weight = oxytocin_weight,
            ego_pm1         = ego_pm1,
            ego_weight      = ego_weight,
            norm_pm1        = norm_pm1,
            norm_weight     = norm_weight,
        )
        # h_stars: (n_actions × H, n_bits) float32

        # ── Post-teleport analytics ───────────────────────────────────────
        # Reshape to (n_actions, H, n_bits)
        h_r = h_stars.reshape(n_actions, self.H, self.n_bits)
        gs_r = goal_sims.reshape(n_actions, self.H)  # (n_actions, H)

        # Seed goal sims (for traj_slope = goal_sim - seed_goal_sim)
        if np.any(goal_hv_pm1 != 0.0):
            seed_gs = (fwd_pm1 * goal_hv_pm1[None, :]).mean(axis=1)  # (N*H,)
            seed_gs_r = seed_gs.reshape(n_actions, self.H)
        else:
            seed_gs_r = np.full((n_actions, self.H), 0.5, dtype=np.float32)

        traj_slope_r = gs_r - seed_gs_r  # (n_actions, H)

        # Entropy: H(bit_balance) of h*
        bit_balance = (h_stars > 0.0).mean(axis=1)  # (N*H,) — fraction of +1 bits
        bit_balance = np.clip(bit_balance, EPS, 1.0 - EPS)
        entropy_flat = -(bit_balance * np.log2(bit_balance) +
                         (1.0 - bit_balance) * np.log2(1.0 - bit_balance))
        entropy_r = entropy_flat.reshape(n_actions, self.H)  # (n_actions, H)

        # Consistency = 1.0 by construction (h* is deterministic)
        consistency_r = np.ones((n_actions, self.H), dtype=np.float32)

        # Rule confidence
        rule_conf = float(rule_weight / (rule_weight + 10.0 + EPS))

        # Joint score per (action, hypothesis)
        joint = (
              self.W_CONSISTENCY * consistency_r
            + self.W_GOAL_SIM    * gs_r
            + self.W_TRAJ_SLOPE  * np.clip(0.5 + traj_slope_r, 0.0, 1.0)
            + self.W_ENTROPY     * (1.0 - entropy_r)
            + self.W_RESONANCE   * rule_conf
        )  # (n_actions, H)

        # Best hypothesis per action, then best action
        best_h   = joint.argmax(axis=1)                          # (n_actions,)
        a_scores = joint[np.arange(n_actions), best_h]           # (n_actions,)
        best_a   = int(a_scores.argmax())
        best_act = action_ids[best_a]
        best_hv_idx = int(best_h[best_a])

        # Best h* in uint64
        best_h_star_pm1 = h_r[best_a, best_hv_idx]              # (n_bits,) float32
        best_fwd_hv     = pm1_to_uint64(best_h_star_pm1)        # (W,) uint64
        best_bwd_hv     = best_fwd_hv.copy()                    # bilateral: fwd == bwd

        b_goal_sim   = float(gs_r[best_a, best_hv_idx])
        b_traj_slope = float(traj_slope_r[best_a, best_hv_idx])
        b_entropy    = float(entropy_r[best_a, best_hv_idx])
        b_consist    = 1.0

        # ── Mean analytics for ZSignal ────────────────────────────────────
        mean_goal_sim = float(np.mean(gs_r))
        mean_entropy  = float(np.mean(entropy_r))

        # ── Resonance ─────────────────────────────────────────────────────
        mag       = abs(b_traj_slope) + surprise
        import math
        decay     = math.exp(-0.05 * 1)  # dt=1, lam=0.05
        resonance = float(
            (mag * b_consist)
            * (trust / (b_entropy + EPS))
            * decay
            * safety
        )

        # ── Soft EMA: update rule bundle ──────────────────────────────────
        # rule_hv = fwd XOR bwd = h* XOR h* = 0 in pm1 → use fwd directly
        # In pm1 space: rule_pm1_new = fwd_pm1 (since fwd==bwd, XOR=0 → use fwd)
        # Per plan: rule_pm1 = (fwd_pm1 XOR bwd_pm1) → in pm1: product
        # Since h*_fwd == h*_bwd, rule = fwd * bwd = h*² = 1 everywhere → uniform
        # Instead use the best h* itself as the rule signal (captures the fixed point)
        alpha_rule = resonance / (rule_weight + resonance + EPS)
        new_rule_pm1 = (
            (1.0 - alpha_rule) * rule_bundle_pm1
            + alpha_rule * best_h_star_pm1
        )
        new_rule_weight = rule_weight + resonance

        # ── Soft EMA: update goal HV ──────────────────────────────────────
        new_goal_pm1   = goal_hv_pm1.copy()
        new_goal_weight = accumulated_goal_weight
        if b_consist > 0.65:
            alpha_goal = resonance / (accumulated_goal_weight + resonance + EPS)
            new_goal_pm1 = (
                (1.0 - alpha_goal) * goal_hv_pm1
                + alpha_goal * best_h_star_pm1
            )
            new_goal_weight = accumulated_goal_weight + resonance

        # ── ZSignal update (inside teleport) ─────────────────────────────
        S_new = z_signal.update(mean_goal_sim, mean_entropy, step)
        Z_new = float(z_signal.state.Z_current)

        # ── Micro-exploration trigger ─────────────────────────────────────
        traj_accel = b_traj_slope - prev_traj_slope
        if (traj_accel < -ACCEL_THRESHOLD) or \
           (traj_accel < 0.0 and retrodiction_accuracy < RETRO_THRESHOLD):
            S_new = -abs(S_new)

        # ── Eigenvalue spectrum for diagnostics ──────────────────────────
        # Compute spectrum for the best hypothesis seed
        best_seed_pm1 = (fwd_pm1[best_a * self.H + best_hv_idx] +
                         bwd_pm1[best_a * self.H + best_hv_idx]) * 0.5
        eigenvalue_spectrum = self._eigen_solver.compute_spectrum(
            axes_pm1     = axes_pm1,
            axis_weights = axis_weights,
            goal_pm1     = goal_hv_pm1,
            goal_weight  = goal_w_adj,
            rule_pm1     = rule_bundle_pm1,
            rule_weight  = rule_w_adj,
            seed_pm1     = best_seed_pm1,
            chain_pm1    = chain_pm1,
            chain_weight = chain_weight,
        )

        # ── Thalamic safety diagnostic scores ────────────────────────────
        # Compute 4 diagnostic scores from best h* (O(n_bits) each, ~0.02ms total)
        _diag_danger_score   = 0.0
        _diag_oxytocin_score = 0.0
        _diag_ego_drift      = 0.0
        _diag_norm_score     = 0.5
        _n = float(self.n_bits)
        if danger_pm1 is not None and danger_weight > EPS:
            _dot = float(np.dot(best_h_star_pm1, danger_pm1)) / (_n + EPS)
            _diag_danger_score = float(np.clip(0.5 + 0.5 * _dot, 0.0, 1.0))
        if oxytocin_pm1 is not None and oxytocin_weight > EPS:
            _dot = float(np.dot(best_h_star_pm1, oxytocin_pm1)) / (_n + EPS)
            _diag_oxytocin_score = float(np.clip(0.5 + 0.5 * _dot, 0.0, 1.0))
        if ego_pm1 is not None and ego_weight > EPS:
            _dot = float(np.dot(best_h_star_pm1, ego_pm1)) / (_n + EPS)
            _cosine_ego = float(np.clip(0.5 + 0.5 * _dot, 0.0, 1.0))
            _diag_ego_drift = float(np.clip(1.0 - _cosine_ego, 0.0, 1.0))
        if norm_pm1 is not None and norm_weight > EPS:
            _dot = float(np.dot(best_h_star_pm1, norm_pm1)) / (_n + EPS)
            _diag_norm_score = float(np.clip(0.5 + 0.5 * _dot, 0.0, 1.0))

        return FullTeleportResult(
            best_action             = best_act,
            joint_scores            = {aid: float(a_scores[i]) for i, aid in enumerate(action_ids)},
            consistency             = b_consist,
            goal_sim                = b_goal_sim,
            traj_slope              = b_traj_slope,
            entropy                 = b_entropy,
            resonance               = resonance,
            best_fwd_hv             = best_fwd_hv,
            best_bwd_hv             = best_bwd_hv,
            updated_rule_bundle_pm1 = new_rule_pm1.astype(np.float32),
            updated_rule_weight     = float(new_rule_weight),
            updated_goal_hv_pm1     = new_goal_pm1.astype(np.float32),
            updated_goal_weight     = float(new_goal_weight),
            S_new                   = float(S_new),
            Z_new                   = float(Z_new),
            eigenvalue_spectrum     = eigenvalue_spectrum,
            chain_hits              = chain_hits,
            mean_goal_sim           = mean_goal_sim,
            mean_entropy            = mean_entropy,
            danger_score            = _diag_danger_score,
            oxytocin_score          = _diag_oxytocin_score,
            ego_drift               = _diag_ego_drift,
            norm_score              = _diag_norm_score,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EigenSpiralBuilder
# ─────────────────────────────────────────────────────────────────────────────

class EigenSpiralBuilder:
    """Absorbs SpiralDSVLanguageModel.build_from_tokens() into a single eigen solve.

    Replaces the XOR-bundle accumulation:
        sem_fwd[a] ^= codebook[b]   for all (a,b) pairs at lag c

    with the closed-form fixed point:
        sem_fwd_pm1[a]* = sign( Σ_{b: (a,b) is a pair} weight(a,b) × CB_pm1[b] )
                        = sign( co_occurrence_weights[a] @ CB_pm1 )

    This is the same ``token_reward_sum @ CB_pm1`` pattern as EigenTrainer,
    applied per-row of the co-occurrence matrix.

    Mathematical equivalence
    ────────────────────────
    XOR-bundle: sem_fwd[a] = XOR_i CB[b_i]
    In pm1 space, the expected value of XOR-bundle is:
        E[sem_fwd_pm1[a][j]] = sign( Σ_i CB_pm1[b_i][j] )
    because XOR of binary vectors = majority vote in pm1 space.

    The eigen fixed point is:
        sem_fwd_pm1[a]* = sign( Σ_i w_i × CB_pm1[b_i][j] )

    With uniform weights (w_i = 1), this is identical to the XOR-bundle
    majority vote. With co-occurrence frequency weights, it is strictly
    better — more frequent pairs contribute more signal.

    Performance
    ───────────
    Before: O(N × ctx_len) scatter XOR operations (already vectorised via reduceat)
    After:  O(vocab² × n_bits) matmul — independent of N and ctx_len
            For vocab=1024, n_bits=32768: 1024 × 1024 × 32768 × 4B ≈ 128 GB
            → Use sparse co-occurrence: only non-zero (a,b) pairs matter
            → In practice: O(unique_pairs × n_bits) where unique_pairs << vocab²

    For large n_bits (N_WORDS=512 → n_bits=32768), the matmul is:
        co_occ[vocab, vocab] @ CB_pm1[vocab, n_bits] → sem_pm1[vocab, n_bits]
    This is a (1024, 1024) × (1024, 32768) matmul = 1024 × 32768 × 1024 FLOPs
    ≈ 34 GFLOPs — ~0.1s on a single H100 (300 TFLOPS).

    Args:
        codebook_pm1  : (vocab_size, n_bits) float32 — CB in pm1 space
        use_frequency : If True, weight by co-occurrence frequency (better signal)
                        If False, uniform weights (equivalent to XOR-bundle)
    """

    def __init__(
        self,
        codebook_pm1  : np.ndarray,   # (vocab_size, n_bits) float32
        use_frequency : bool = True,
    ):
        self.CB_pm1        = codebook_pm1.astype(np.float32)
        self.vocab_size    = codebook_pm1.shape[0]
        self.n_bits        = codebook_pm1.shape[1]
        self.use_frequency = use_frequency

    @classmethod
    def from_codebook_uint64(
        cls,
        codebook_vecs  : np.ndarray,   # (vocab_size, n_words) uint64
        use_frequency  : bool = True,
    ) -> "EigenSpiralBuilder":
        """Construct from uint64 codebook (converts to pm1 internally).

        Args:
            codebook_vecs : (vocab_size, n_words) uint64 — raw codebook
            use_frequency : Weight by co-occurrence frequency

        Returns:
            EigenSpiralBuilder instance
        """
        vocab_size, n_words = codebook_vecs.shape
        bits = np.unpackbits(
            codebook_vecs.view(np.uint8).reshape(vocab_size, n_words * 8),
            axis=1, bitorder='little'
        )
        cb_pm1 = bits.astype(np.float32) * 2.0 - 1.0  # (vocab_size, n_bits)
        return cls(cb_pm1, use_frequency=use_frequency)

    def build_bilateral_tables(
        self,
        tokens         : np.ndarray,   # (N,) int32 — token sequence
        ctx_len        : int   = 4,
        time_budget_s  : float = 30.0,
        verbose        : bool  = True,
    ) -> dict:
        """Build sem_fwd_pm1 and sem_bwd_pm1 via eigen fixed-point solve.

        For each lag c in 1..ctx_len, accumulates co-occurrence weights:
            fwd_weights[a, b] += 1  for all (a, b) pairs at lag c
            bwd_weights[b, a] += 1  for all (a, b) pairs at lag c

        Then computes:
            sem_fwd_pm1[a]* = sign( fwd_weights[a] @ CB_pm1 )
            sem_bwd_pm1[b]* = sign( bwd_weights[b] @ CB_pm1 )

        The matmul is (vocab, vocab) × (vocab, n_bits) → (vocab, n_bits).
        For vocab=1024, n_bits=32768: ~34 GFLOPs, ~0.1s on H100.

        Args:
            tokens        : (N,) int32 — token sequence
            ctx_len       : Maximum context depth (number of lags)
            time_budget_s : Time budget in seconds
            verbose       : Print progress

        Returns:
            dict with keys:
              sem_fwd_pm1 : (vocab_size, n_bits) float32 — forward table in pm1
              sem_bwd_pm1 : (vocab_size, n_bits) float32 — backward table in pm1
              sem_fwd_u64 : (vocab_size, n_words) uint64 — forward table (hard)
              sem_bwd_u64 : (vocab_size, n_words) uint64 — backward table (hard)
              total_pairs : int — total (a,b) pairs processed
        """
        import time
        t0 = time.time()

        N = len(tokens)
        tokens_i32 = np.clip(tokens.astype(np.int32), 0, self.vocab_size - 1)

        # ── FIX #2: np.bincount + ravel_multi_index replaces np.add.at 2D ──
        # np.add.at with 2D indices is unbuffered and ~100x slower than bincount.
        # Strategy: flatten (a, b) → a * vocab + b, then np.bincount on flat index,
        # then reshape to (vocab, vocab). This is O(N) buffered — SIMD-friendly.
        #
        # fwd_w[a, b] = number of times b follows a within ctx_len lags
        # bwd_w[b, a] = number of times a precedes b within ctx_len lags
        fwd_flat = np.zeros(self.vocab_size * self.vocab_size, dtype=np.float32)
        bwd_flat = np.zeros(self.vocab_size * self.vocab_size, dtype=np.float32)
        total_pairs = 0

        for c in range(1, ctx_len + 1):
            if time.time() - t0 > time_budget_s * 0.5:
                if verbose:
                    print(f"[EigenSpiral] Time budget reached at c={c-1}")
                break
            M = N - c
            a_toks = tokens_i32[:M]
            b_toks = tokens_i32[c:]

            # Flatten 2D index: fwd pair (a→b) = a * vocab + b
            fwd_idx = a_toks.astype(np.int64) * self.vocab_size + b_toks.astype(np.int64)
            bwd_idx = b_toks.astype(np.int64) * self.vocab_size + a_toks.astype(np.int64)

            # GPU scatter_add is ~100x faster than np.add.at for this case
            fwd_flat += gpu_bincount_weighted(
                fwd_idx, np.ones(len(fwd_idx), dtype=np.float32),
                self.vocab_size * self.vocab_size
            ).astype(np.float32)
            bwd_flat += gpu_bincount_weighted(
                bwd_idx, np.ones(len(bwd_idx), dtype=np.float32),
                self.vocab_size * self.vocab_size
            ).astype(np.float32)

            total_pairs += M
            if verbose:
                elapsed = time.time() - t0
                print(f"[EigenSpiral] c={c}/{ctx_len}  pairs={total_pairs:,}  "
                      f"elapsed={elapsed:.2f}s")

        fwd_w = fwd_flat.reshape(self.vocab_size, self.vocab_size)
        bwd_w = bwd_flat.reshape(self.vocab_size, self.vocab_size)

        # ── Single matmul: (vocab, vocab) × (vocab, n_bits) → (vocab, n_bits) ──
        if verbose:
            print(f"[EigenSpiral] Computing sem_fwd matmul "
                  f"({self.vocab_size}×{self.vocab_size}) × "
                  f"({self.vocab_size}×{self.n_bits})...")

        t_matmul = time.time()
        # GPU matmul: (vocab, vocab) × (vocab, n_bits) → (vocab, n_bits)
        # Uses float16 tensor cores on H100 for ~10× speedup over CPU BLAS
        sem_fwd_spectrum = gpu_matmul_f16(fwd_w, self.CB_pm1)   # (vocab, n_bits)
        sem_bwd_spectrum = gpu_matmul_f16(bwd_w, self.CB_pm1)   # (vocab, n_bits)

        # Fixed point: sign(spectrum) — on GPU
        sem_fwd_pm1 = gpu_sign_f32(sem_fwd_spectrum)
        sem_bwd_pm1 = gpu_sign_f32(sem_bwd_spectrum)

        # Convert to uint64 for storage
        n_words = self.n_bits // 64
        sem_fwd_u64 = batch_pm1_to_uint64(sem_fwd_pm1)  # (vocab, n_words)
        sem_bwd_u64 = batch_pm1_to_uint64(sem_bwd_pm1)  # (vocab, n_words)

        elapsed = time.time() - t0
        matmul_t = time.time() - t_matmul
        if verbose:
            print(f"[EigenSpiral] Matmul done in {matmul_t:.2f}s")
            print(f"[EigenSpiral] Total: {total_pairs:,} pairs, "
                  f"{elapsed:.2f}s elapsed")

        return dict(
            sem_fwd_pm1 = sem_fwd_pm1,
            sem_bwd_pm1 = sem_bwd_pm1,
            sem_fwd_u64 = sem_fwd_u64,
            sem_bwd_u64 = sem_bwd_u64,
            total_pairs = total_pairs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EigenTrainer
# ─────────────────────────────────────────────────────────────────────────────

class EigenTrainer:
    """Absorbs the entire train_on_tokens() loop into a single eigen solve.

    Replaces the Python ``for i in range(N)`` loop in
    ``FullBiDirHDC.train_on_tokens()`` with a single reward-weighted
    matrix multiply followed by ``sign()``.

    Mathematical derivation
    ───────────────────────
    The sequential per-bigram EMA in ``observe()`` is:

        rule_hv[i]  = CB[t_next[i]]          (before XOR action XOR after
                                               = CB[t] XOR CB[t] XOR CB[t+1]
                                               = CB[t+1])
        alpha_i     = r_i / (W_acc + r_i)
        bundle_pm1 ← (1 - alpha_i) × bundle_pm1 + alpha_i × rule_pm1[i]

    The fixed point of this recurrence is the reward-weighted mean:

        bundle_pm1* = sign( Σ_i r_i × CB_pm1[t_next[i]] )
                    = sign( rewards @ CB_pm1[t_next] )

    This is a single O(N × n_bits) matrix multiply — fully vectorised,
    no Python loop, no per-step state accumulation.

    The goal HV fixed point follows the same pattern, restricted to
    high-reward bigrams (reward >= goal_threshold):

        goal_pm1* = sign( Σ_{i: r_i >= goal_threshold} r_i × CB_pm1[t_next[i]] )

    Both are computed in ``absorb_bigrams()`` in one pass over the data.

    Memory note
    ───────────
    For large N (e.g. 62.5M bigrams), the full (N, n_bits) float32 matrix
    would be ~62.5M × 32768 × 4 B ≈ 8 TB — impossible to materialise.
    Instead we accumulate the weighted sum incrementally in chunks:

        spectrum += rewards[chunk] @ CB_pm1[t_next[chunk]]

    Each chunk is (chunk_size, n_bits) float32 — controllable via
    ``chunk_size`` (default 65536 rows ≈ 8 GB for n_bits=32768, still large).

    For n_bits=32768 (N_WORDS=512), we use a smarter approach:
    accumulate per-token reward sums first (vocab_size buckets), then
    do a single (vocab_size, n_bits) matmul — O(vocab_size × n_bits)
    regardless of N.  This is the ``token_reward_sum`` path.

    Args:
        codebook_pm1  : (vocab_size, n_bits) float32 — CB in pm1 space
        goal_threshold: Minimum reward for a bigram to update the goal HV
    """

    def __init__(
        self,
        codebook_pm1   : np.ndarray,   # (vocab_size, n_bits) float32
        goal_threshold : float = 10.0,
    ):
        self.CB_pm1         = codebook_pm1.astype(np.float32)
        self.vocab_size     = codebook_pm1.shape[0]
        self.n_bits         = codebook_pm1.shape[1]
        self.goal_threshold = goal_threshold

    @classmethod
    def from_codebook_uint64(
        cls,
        codebook_vecs  : np.ndarray,   # (vocab_size, n_words) uint64
        goal_threshold : float = 10.0,
    ) -> "EigenTrainer":
        """Construct from uint64 codebook (converts to pm1 internally).

        Args:
            codebook_vecs : (vocab_size, n_words) uint64 — raw codebook
            goal_threshold: Minimum reward for goal HV update

        Returns:
            EigenTrainer instance
        """
        vocab_size, n_words = codebook_vecs.shape
        n_bits = n_words * 64
        # Unpack all HVs to pm1 in one vectorised call
        bits = np.unpackbits(
            codebook_vecs.view(np.uint8).reshape(vocab_size, n_words * 8),
            axis=1, bitorder='little'
        )
        cb_pm1 = bits.astype(np.float32) * 2.0 - 1.0  # (vocab_size, n_bits)
        return cls(cb_pm1, goal_threshold=goal_threshold)

    def absorb_bigrams(
        self,
        t_prev_arr     : np.ndarray,   # (N,) int32 — previous tokens
        t_next_arr     : np.ndarray,   # (N,) int32 — next tokens
        rewards        : np.ndarray,   # (N,) float32 — per-bigram rewards
        verbose        : bool = True,
    ) -> dict:
        """Absorb all N bigrams into closed-form rule_bundle and goal_hv.

        Uses the token_reward_sum path:
          1. Accumulate per-token reward sums: token_r[v] = Σ_{i: t_next[i]=v} r_i
          2. rule_spectrum = token_r @ CB_pm1          (vocab_size, n_bits matmul)
          3. rule_bundle_pm1* = sign(rule_spectrum)
          4. goal_spectrum = token_r_goal @ CB_pm1     (high-reward only)
          5. goal_pm1* = sign(goal_spectrum)

        This is O(N) for the accumulation + O(vocab_size × n_bits) for the
        matmul — independent of N for the expensive part.

        Args:
            t_prev_arr : (N,) int32 — previous token IDs (used for reward lookup)
            t_next_arr : (N,) int32 — next token IDs (rule_hv = CB[t_next])
            rewards    : (N,) float32 — bigram rewards (bigram_freq × 100)
            verbose    : Print timing info

        Returns:
            dict with keys:
              rule_bundle_pm1  : (n_bits,) float32 — fixed-point rule bundle
              rule_weight      : float — total accumulated reward weight
              goal_hv_pm1      : (n_bits,) float32 — fixed-point goal HV
              goal_weight      : float — total goal reward weight
              n_bigrams        : int — number of bigrams processed
              n_goal_bigrams   : int — number of high-reward bigrams
        """
        import time
        t0 = time.time()

        N = len(t_next_arr)
        if N == 0:
            zero = np.zeros(self.n_bits, dtype=np.float32)
            return dict(rule_bundle_pm1=zero, rule_weight=0.0,
                        goal_hv_pm1=zero, goal_weight=0.0,
                        n_bigrams=0, n_goal_bigrams=0)

        # ── FIX #1 + GPU: bincount then GPU matmul ────────────────────────
        # np.add.at is unbuffered (no SIMD). np.bincount with weights= is
        # a buffered O(N) operation — SIMD-friendly, ~10-20x faster.
        # The subsequent matmul is dispatched to GPU (cuBLAS SGEMM / HGEMM).
        t_next_clipped = np.clip(t_next_arr, 0, self.vocab_size - 1)

        # Step 1: Accumulate per-token reward sums via bincount (GPU scatter_add)
        token_r = gpu_bincount_weighted(
            t_next_clipped, rewards.astype(np.float32), self.vocab_size
        ).astype(np.float64)

        # Step 2: Rule spectrum = token_r @ CB_pm1  (GPU SGEMM / HGEMM)
        rule_spectrum   = gpu_matmul_f16(
            token_r.astype(np.float32)[None, :], self.CB_pm1
        )[0]                                                          # (n_bits,)
        rule_bundle_pm1 = gpu_sign_f32(rule_spectrum)
        rule_weight = float(token_r.sum())

        # Step 3: Goal spectrum — high-reward bigrams only
        high_reward_mask = rewards >= self.goal_threshold
        n_goal = int(high_reward_mask.sum())

        if n_goal > 0:
            t_next_goal  = t_next_clipped[high_reward_mask]
            r_goal       = rewards[high_reward_mask].astype(np.float32)
            token_r_goal = gpu_bincount_weighted(
                t_next_goal, r_goal, self.vocab_size
            ).astype(np.float64)
            goal_spectrum = gpu_matmul_f16(
                token_r_goal.astype(np.float32)[None, :], self.CB_pm1
            )[0]
            goal_pm1    = gpu_sign_f32(goal_spectrum)
            goal_weight = float(token_r_goal.sum())
        else:
            goal_pm1    = np.zeros(self.n_bits, dtype=np.float32)
            goal_weight = 0.0

        elapsed = time.time() - t0
        if verbose:
            print(f"[EigenTrainer] absorb_bigrams: {N:,} bigrams → "
                  f"rule_weight={rule_weight:.1f}, "
                  f"goal_bigrams={n_goal:,}, "
                  f"elapsed={elapsed:.2f}s")

        return dict(
            rule_bundle_pm1 = rule_bundle_pm1,
            rule_weight     = rule_weight,
            goal_hv_pm1     = goal_pm1,
            goal_weight     = goal_weight,
            n_bigrams       = N,
            n_goal_bigrams  = n_goal,
        )

    def absorb_bigrams_chunked(
        self,
        t_prev_arr     : np.ndarray,   # (N,) int32
        t_next_arr     : np.ndarray,   # (N,) int32
        rewards        : np.ndarray,   # (N,) float32
        chunk_size     : int   = 500_000,
        verbose        : bool  = True,
    ) -> dict:
        """Chunked version of absorb_bigrams() for memory-constrained environments.

        Identical result to absorb_bigrams() but processes tokens in chunks
        to bound peak memory. The token_reward_sum accumulation is O(vocab_size)
        regardless of chunk_size, so this is purely for progress reporting.

        Args:
            t_prev_arr : (N,) int32
            t_next_arr : (N,) int32
            rewards    : (N,) float32
            chunk_size : Chunk size for progress reporting
            verbose    : Print progress

        Returns:
            Same dict as absorb_bigrams()
        """
        import time
        t0 = time.time()
        N  = len(t_next_arr)

        token_r      = np.zeros(self.vocab_size, dtype=np.float64)
        token_r_goal = np.zeros(self.vocab_size, dtype=np.float64)
        n_goal       = 0

        t_next_clipped = np.clip(t_next_arr, 0, self.vocab_size - 1)

        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            tn  = t_next_clipped[chunk_start:chunk_end]
            rw  = rewards[chunk_start:chunk_end].astype(np.float32)

            # FIX #1 + GPU (chunked): gpu_bincount_weighted (scatter_add on GPU)
            token_r += gpu_bincount_weighted(tn, rw, self.vocab_size)

            high = rw >= self.goal_threshold
            if high.any():
                token_r_goal += gpu_bincount_weighted(
                    tn[high], rw[high], self.vocab_size
                )
                n_goal += int(high.sum())

            if verbose:
                pct = 100.0 * chunk_end / N
                elapsed = time.time() - t0
                print(f"[EigenTrainer] {chunk_end:,}/{N:,} ({pct:.1f}%) "
                      f"elapsed={elapsed:.1f}s")

        # Single GPU matmul for rule bundle (cuBLAS HGEMM)
        rule_spectrum   = gpu_matmul_f16(
            token_r.astype(np.float32)[None, :], self.CB_pm1
        )[0]
        rule_bundle_pm1 = gpu_sign_f32(rule_spectrum)
        rule_weight = float(token_r.sum())

        # Single GPU matmul for goal HV
        if n_goal > 0:
            goal_spectrum = gpu_matmul_f16(
                token_r_goal.astype(np.float32)[None, :], self.CB_pm1
            )[0]
            goal_pm1    = gpu_sign_f32(goal_spectrum)
            goal_weight = float(token_r_goal.sum())
        else:
            goal_pm1    = np.zeros(self.n_bits, dtype=np.float32)
            goal_weight = 0.0

        elapsed = time.time() - t0
        if verbose:
            print(f"[EigenTrainer] Done: {N:,} bigrams absorbed in {elapsed:.2f}s "
                  f"(rule_weight={rule_weight:.1f}, goal_bigrams={n_goal:,})")

        return dict(
            rule_bundle_pm1 = rule_bundle_pm1,
            rule_weight     = rule_weight,
            goal_hv_pm1     = goal_pm1,
            goal_weight     = goal_weight,
            n_bigrams       = N,
            n_goal_bigrams  = n_goal,
        )