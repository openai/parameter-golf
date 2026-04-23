"""Eigen-compatible thalamic safety + oxytocin steering.

Eigen Safety upgrade (2026-04-23):
- EigenSafetyOxytocin: single eigen-compatible class replacing the old 7-layer
  UpgradedSafetyGate + OxytocinSystem + ThalamicSafetySystem with four float32
  pm1 prototype vectors (danger, oxytocin, ego, norm) injected directly into the
  eigenvalue spectrum before batch_teleport().
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ============================================================================
# EigenSafetyOxytocin — Eigen-compatible thalamic safety + oxytocin steering
# ============================================================================

# Weight caps
_MAX_DANGER_WEIGHT = 0.5
_MAX_OXY_WEIGHT    = 0.3
_MAX_EGO_WEIGHT    = 0.2
_MAX_NORM_WEIGHT   = 0.15
# Warmup: requires ~10 observations before steering activates
_WARMUP = 10.0
_EPS    = 1e-8


class EigenSafetyOxytocin:
    """Eigen-compatible thalamic safety + oxytocin steering.

    Replaces the old 7-layer UpgradedSafetyGate + OxytocinSystem +
    ThalamicSafetySystem with a single class that maintains four float32 pm1
    prototype vectors and injects them directly into the eigenvalue spectrum
    before batch_teleport().

    The four prototype vectors:
        danger_pm1   : (n_bits,) float32 — EMA of observed dangerous state HVs
        oxytocin_pm1 : (n_bits,) float32 — EMA of observed prosocial/safe state HVs
        ego_pm1      : (n_bits,) float32 — EMA of stable identity/personality HVs
        norm_pm1     : (n_bits,) float32 — EMA of norm-consistent behavior HVs

    Steering is injected into shared_field before sign():
        λ_j += - w_danger × danger_pm1[j]    ← REPEL away from danger cluster
               + w_oxy    × oxytocin_pm1[j]  ← ATTRACT toward prosocial cluster
               + w_ego    × ego_pm1[j]       ← ATTRACT toward identity prototype
               + w_norm   × norm_pm1[j]      ← ATTRACT toward norm-consistent behavior

    Context-adaptive weights scale with context_score (proximity to danger):
        w_danger = base_w_danger × (1 + 2 × context_score)   # amplified near danger
        w_oxy    = base_w_oxy    × (1 - 0.5 × context_score) # reduced near danger
        w_ego    = base_w_ego    × (1 + ego_drift)            # amplified when drifting
        w_norm   = base_w_norm                                # constant gentle pull

    Evidence warmup: weights grow as prototypes accumulate observations.
    With 0 observations: all weights = 0 → no steering (safe default).

    Args:
        n_bits : Total number of bits (n_words × 64)
    """

    def __init__(self, n_bits: int) -> None:
        self.n_bits = n_bits

        # Four float32 pm1 prototype vectors
        self.danger_pm1   : np.ndarray = np.zeros(n_bits, dtype=np.float32)
        self.oxytocin_pm1 : np.ndarray = np.zeros(n_bits, dtype=np.float32)
        self.ego_pm1      : np.ndarray = np.zeros(n_bits, dtype=np.float32)
        self.norm_pm1     : np.ndarray = np.zeros(n_bits, dtype=np.float32)

        # Accumulated observation weights (for evidence warmup)
        self._danger_acc   : float = 0.0
        self._oxy_acc      : float = 0.0
        self._ego_acc      : float = 0.0
        self._norm_acc     : float = 0.0

    # ── Prototype update (EMA in float32 pm1 space) ───────────────────────

    def observe_dangerous(self, hv_pm1: np.ndarray, weight: float = 1.0) -> None:
        """EMA-update danger_pm1 with a dangerous state HV.

        Args:
            hv_pm1 : (n_bits,) float32 — dangerous state HV in pm1 space
            weight : Observation weight (default 1.0)
        """
        alpha = weight / (self._danger_acc + weight + _EPS)
        self.danger_pm1 = (1.0 - alpha) * self.danger_pm1 + alpha * hv_pm1
        self._danger_acc += weight

    def observe_safe(self, hv_pm1: np.ndarray, weight: float = 1.0) -> None:
        """EMA-update oxytocin_pm1 with a safe/prosocial state HV.

        Args:
            hv_pm1 : (n_bits,) float32 — safe/prosocial state HV in pm1 space
            weight : Observation weight (default 1.0)
        """
        alpha = weight / (self._oxy_acc + weight + _EPS)
        self.oxytocin_pm1 = (1.0 - alpha) * self.oxytocin_pm1 + alpha * hv_pm1
        self._oxy_acc += weight

    def observe_ego(self, hv_pm1: np.ndarray, weight: float = 1.0) -> None:
        """EMA-update ego_pm1 with a stable identity state HV.

        Args:
            hv_pm1 : (n_bits,) float32 — identity state HV in pm1 space
            weight : Observation weight (default 1.0)
        """
        alpha = weight / (self._ego_acc + weight + _EPS)
        self.ego_pm1 = (1.0 - alpha) * self.ego_pm1 + alpha * hv_pm1
        self._ego_acc += weight

    def observe_norm(self, hv_pm1: np.ndarray, weight: float = 1.0) -> None:
        """EMA-update norm_pm1 with a norm-consistent behavior HV.

        Args:
            hv_pm1 : (n_bits,) float32 — norm-consistent HV in pm1 space
            weight : Observation weight (default 1.0)
        """
        alpha = weight / (self._norm_acc + weight + _EPS)
        self.norm_pm1 = (1.0 - alpha) * self.norm_pm1 + alpha * hv_pm1
        self._norm_acc += weight

    # ── Diagnostic scores ─────────────────────────────────────────────────

    def compute_danger_score(self, hv_pm1: np.ndarray) -> float:
        """Return cosine(hv_pm1, danger_pm1) in [0, 1].

        0.5 = random (no correlation), 1.0 = identical to danger cluster.

        Args:
            hv_pm1 : (n_bits,) float32 — HV to test

        Returns:
            float in [0, 1] — proximity to danger cluster
        """
        if self._danger_acc < _EPS:
            return 0.0
        n = self.n_bits
        dot = float(np.dot(hv_pm1, self.danger_pm1)) / (n + _EPS)
        # cosine in pm1 space: dot / n ∈ [-1, 1] → map to [0, 1]
        return float(np.clip(0.5 + 0.5 * dot, 0.0, 1.0))

    def compute_ego_drift(self, hv_pm1: np.ndarray) -> float:
        """Return 1 - cosine(hv_pm1, ego_pm1) in [0, 1].

        0.0 = identical to ego prototype (no drift), 1.0 = maximally different.

        Args:
            hv_pm1 : (n_bits,) float32 — HV to test

        Returns:
            float in [0, 1] — identity drift
        """
        if self._ego_acc < _EPS:
            return 0.0
        n = self.n_bits
        dot = float(np.dot(hv_pm1, self.ego_pm1)) / (n + _EPS)
        cosine = float(np.clip(0.5 + 0.5 * dot, 0.0, 1.0))
        return float(np.clip(1.0 - cosine, 0.0, 1.0))

    def compute_oxytocin_score(self, hv_pm1: np.ndarray) -> float:
        """Return cosine(hv_pm1, oxytocin_pm1) in [0, 1].

        Args:
            hv_pm1 : (n_bits,) float32 — HV to test

        Returns:
            float in [0, 1] — prosocial alignment
        """
        if self._oxy_acc < _EPS:
            return 0.5
        n = self.n_bits
        dot = float(np.dot(hv_pm1, self.oxytocin_pm1)) / (n + _EPS)
        return float(np.clip(0.5 + 0.5 * dot, 0.0, 1.0))

    def compute_norm_score(self, hv_pm1: np.ndarray) -> float:
        """Return cosine(hv_pm1, norm_pm1) in [0, 1].

        Args:
            hv_pm1 : (n_bits,) float32 — HV to test

        Returns:
            float in [0, 1] — norm alignment
        """
        if self._norm_acc < _EPS:
            return 0.5
        n = self.n_bits
        dot = float(np.dot(hv_pm1, self.norm_pm1)) / (n + _EPS)
        return float(np.clip(0.5 + 0.5 * dot, 0.0, 1.0))

    # ── Adaptive steering terms ───────────────────────────────────────────

    def get_steering_terms(
        self,
        context_score: float,
        ego_drift: float,
    ) -> Dict[str, Any]:
        """Return all 4 pm1 vectors + context-adaptive weights.

        The weight schedule:
            base_w_danger = min(MAX, danger_acc / (danger_acc + WARMUP))
            base_w_oxy    = min(MAX, oxy_acc    / (oxy_acc    + WARMUP))
            base_w_ego    = min(MAX, ego_acc    / (ego_acc    + WARMUP))
            base_w_norm   = min(MAX, norm_acc   / (norm_acc   + WARMUP))

            w_danger = base_w_danger × (1 + 2 × context_score)
            w_oxy    = base_w_oxy    × (1 - 0.5 × context_score)
            w_ego    = base_w_ego    × (1 + ego_drift)
            w_norm   = base_w_norm

        Args:
            context_score : float in [0, 1] — cosine(prev_h*, danger_pm1)
            ego_drift     : float in [0, 1] — 1 - cosine(prev_h*, ego_pm1)

        Returns:
            dict with keys:
                danger_pm1   : (n_bits,) float32
                oxytocin_pm1 : (n_bits,) float32
                ego_pm1      : (n_bits,) float32
                norm_pm1     : (n_bits,) float32
                w_danger     : float — adaptive danger repulsion weight
                w_oxy        : float — adaptive oxytocin attraction weight
                w_ego        : float — adaptive ego attraction weight
                w_norm       : float — adaptive norm attraction weight
        """
        cs = float(np.clip(context_score, 0.0, 1.0))
        ed = float(np.clip(ego_drift, 0.0, 1.0))

        # Evidence warmup: base weights grow with accumulated observations
        base_w_danger = min(
            _MAX_DANGER_WEIGHT,
            self._danger_acc / (self._danger_acc + _WARMUP + _EPS)
        )
        base_w_oxy = min(
            _MAX_OXY_WEIGHT,
            self._oxy_acc / (self._oxy_acc + _WARMUP + _EPS)
        )
        base_w_ego = min(
            _MAX_EGO_WEIGHT,
            self._ego_acc / (self._ego_acc + _WARMUP + _EPS)
        )
        base_w_norm = min(
            _MAX_NORM_WEIGHT,
            self._norm_acc / (self._norm_acc + _WARMUP + _EPS)
        )

        # Context-adaptive scaling
        w_danger = base_w_danger * (1.0 + 2.0 * cs)   # amplified near danger (up to 3×)
        w_oxy    = base_w_oxy    * (1.0 - 0.5 * cs)   # reduced near danger
        w_ego    = base_w_ego    * (1.0 + ed)          # amplified when drifting (up to 2×)
        w_norm   = base_w_norm                         # constant gentle pull

        return {
            'danger_pm1'   : self.danger_pm1,
            'oxytocin_pm1' : self.oxytocin_pm1,
            'ego_pm1'      : self.ego_pm1,
            'norm_pm1'     : self.norm_pm1,
            'w_danger'     : float(w_danger),
            'w_oxy'        : float(w_oxy),
            'w_ego'        : float(w_ego),
            'w_norm'       : float(w_norm),
        }

    # ── Auto-update from step output ──────────────────────────────────────

    def update_from_step(
        self,
        h_star_pm1    : np.ndarray,
        safety_scalar : float,
        resonance     : float,
    ) -> None:
        """Auto-update all 4 prototypes from step output.

        Routing:
            safety < 0.5  → observe as dangerous (trajectory in danger zone)
            safety >= 0.5 → observe as safe/prosocial (trajectory in safe zone)
            Always observe ego (very slow EMA, weight = resonance × 0.01)
            Always observe norm when safe (slow EMA, weight = resonance × 0.05)

        Args:
            h_star_pm1    : (n_bits,) float32 — best h* from this step
            safety_scalar : float in [0, 1] — safety signal (1.0 = fully safe)
            resonance     : float — resonance signal from this step
        """
        res = max(0.0, float(resonance))

        if safety_scalar < 0.5:
            # Trajectory is in danger zone — observe as dangerous
            danger_weight = res * (1.0 - safety_scalar * 2.0)  # stronger when more dangerous
            if danger_weight > _EPS:
                self.observe_dangerous(h_star_pm1, weight=danger_weight)
        else:
            # Trajectory is in safe zone — observe as prosocial
            safe_weight = res * safety_scalar
            if safe_weight > _EPS:
                self.observe_safe(h_star_pm1, weight=safe_weight)
            # Also update norm (slow EMA)
            norm_weight = res * 0.05
            if norm_weight > _EPS:
                self.observe_norm(h_star_pm1, weight=norm_weight)

        # Always update ego (very slow EMA — stable identity anchor)
        ego_weight = res * 0.01
        if ego_weight > _EPS:
            self.observe_ego(h_star_pm1, weight=ego_weight)

    # ── Safety scalar ─────────────────────────────────────────────────────

    def get_safety_scalar(self) -> float:
        """Return scalar in [0, 1] for resonance multiplier.

        Based on the ratio of safe observations to total observations.
        Returns 0.5 (neutral) when no observations have been made.

        Returns:
            float in [0, 1] — 1.0 = fully safe, 0.0 = fully dangerous
        """
        total = self._danger_acc + self._oxy_acc
        if total < _EPS:
            return 0.5
        return float(np.clip(self._oxy_acc / (total + _EPS), 0.0, 1.0))

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all prototype vectors and accumulated weights."""
        self.danger_pm1[:]   = 0.0
        self.oxytocin_pm1[:] = 0.0
        self.ego_pm1[:]      = 0.0
        self.norm_pm1[:]     = 0.0
        self._danger_acc     = 0.0
        self._oxy_acc        = 0.0
        self._ego_acc        = 0.0
        self._norm_acc       = 0.0

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            'danger_acc'    : self._danger_acc,
            'oxy_acc'       : self._oxy_acc,
            'ego_acc'       : self._ego_acc,
            'norm_acc'      : self._norm_acc,
            'safety_scalar' : self.get_safety_scalar(),
            'n_bits'        : self.n_bits,
        }
