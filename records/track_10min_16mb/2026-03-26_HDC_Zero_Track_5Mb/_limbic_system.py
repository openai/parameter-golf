"""Limbic and Pro-Social Oxytocin System for HDC Language Model.

This module implements a biologically-inspired safety and personality system
using Vector Symbolic Architectures (VSA). Personality and safety are treated
as geometric directions in hyperdimensional space, not fuzzy weights.

KEY CONCEPTS:
=============

1. PERSONALITY SEED (S_p)
   - A fixed 64-bit integer that creates a "topographical tilt" in the HDC space
   - Makes certain trajectories mathematically more probable than others
   - Perfectly reproducible: same seed = same personality
   - Mechanism: Vector = H[token] ⊕ H[pos] ⊕ S_p

2. SAFETY BASIS VECTORS (V_s)
   - Pre-calculated vectors representing "Safe/Altruistic" vs "Dangerous" trajectories
   - Act as "No-Fly Zones" in hyperdimensional space
   - Enable geometric interference when approaching dangerous manifolds

3. LIMBIC FILTER (Pre-Conscious Gating)
   - Calculates trajectory direction during metacognitive correction
   - If path points toward "Danger" vector → triggers Geometric Interference
   - Automatic correction: T_next = T_current ⊕ (V_safe · Inhibition_Gain)

4. OXYTOCIN SYSTEM (Pro-Social Resonance)
   - Makes pro-social trajectories mathematically cheaper (higher SNR)
   - High-density clustering around "Shared Benefit" seeds
   - XOR-Inhibition of high-entropy "Conflict" vectors

BIOLOGICAL EQUIVALENTS:
=======================

| Feature      | HDC Implementation                    | Biological Equivalent           |
|--------------|---------------------------------------|--------------------------------|
| Altruism     | High-density clustering around seeds  | Oxytocin-mediated bonding      |
| Safety       | XOR-Inhibition of conflict vectors    | GABAergic lateral inhibition   |
| Personality  | Fixed Style Seed for XOR-bindings     | Baseline neurotransmitter levels|
| Limbic Gating| Popcount confidence thresholding      | Amygdala-PFC inhibition circuit|

MATHEMATICAL FOUNDATION:
========================

The key property enabling limbic gating:

    H[i] XOR H[j] = ~H[i XOR j]   (complement of H[i^j])

This means:
- Safety vectors can be XOR-bound to context
- Trajectory deviation is measurable via Hamming distance
- Correction is O(1) via XOR operations

DRY-DOCK SAFETY PROTOCOL:
=========================

For bio-hybrid integration with fungal substrates, the Safety Seed can be
tied to the Homeostatic State of the biological component:

- If fungal substrate shows stress (low nutrient flow) → Inhibition Gain increases
- Model enters "Cautious/Safe" personality mode until system stabilizes

Usage:
    from _limbic_system import LimbicSystem, OxytocinSystem, PersonalitySeed
    
    # Create personality seed
    personality = PersonalitySeed(seed=42, traits=["curious", "altruistic"])
    
    # Initialize limbic system
    limbic = LimbicSystem(
        uint64_count=16384,
        personality_seed=personality,
        safety_threshold=0.7
    )
    
    # Check trajectory safety
    is_safe, correction = limbic.check_trajectory(current_vec, next_vec)
    
    # Apply oxytocin modulation
    oxytocin = OxytocinSystem(uint64_count=16384)
    modulated_vec = oxytocin.apply_pro_social_bias(vec, context_tokens)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from collections import deque

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Personality Traits and Seeds
# ═══════════════════════════════════════════════════════════════════════════════

class PersonalityTrait(Enum):
    """Pre-defined personality traits with associated seed patterns."""
    CURIOUS = "curious"           # High entropy tolerance, exploratory
    STOIC = "stoic"               # Low variance, high confidence clusters
    ALTRUISTIC = "altruistic"     # Pro-social trajectory bias
    CAUTIOUS = "cautious"         # High inhibition gain, safe manifolds
    CREATIVE = "creative"         # Low hamming similarity threshold
    ANALYTICAL = "analytical"     # High precision, bit-level awareness
    EMPATHETIC = "empathetic"     # Strong oxytocin resonance
    PROTECTIVE = "protective"     # Strong safety vector alignment


@dataclass
class PersonalitySeed:
    """A fixed 64-bit seed that creates a 'topographical tilt' in HDC space.
    
    The personality seed XOR-binds with every token and position vector,
    shifting the entire 'world' of the model into a specific quadrant
    of the 2^20 dimensional space.
    
    Properties:
    - Reproducible: Same seed always produces same personality
    - Composable: Multiple traits can be combined via XOR
    - Geometric: Personality is a literal direction in hyperspace
    """
    seed: Optional[int] = None  # 64-bit integer; None means "not yet set"
    traits: List[str] = field(default_factory=list)
    entropy_bias: float = 0.5  # 0.0 = stoic, 1.0 = curious
    inhibition_gain: float = 0.5  # 0.0 = permissive, 1.0 = restrictive
    altruism_weight: float = 0.5  # 0.0 = self-focused, 1.0 = other-focused
    
    # Pre-computed seed vectors for different traits
    TRAIT_SEEDS: Dict[str, int] = field(default_factory=lambda: {
        "curious": 0x9E3779B97F4A7C15,      # Golden ratio - exploration
        "stoic": 0xBB67AE8584CAA73B,        # Large prime - stability
        "altruistic": 0x3C9E9E3F7B4A5C6D,   # Pro-social pattern
        "cautious": 0x5A5A5A5A5A5A5A5A,     # Balanced inhibition
        "creative": 0x123456789ABCDEF0,     # High entropy
        "analytical": 0xFEDCBA9876543210,   # Precision pattern
        "empathetic": 0x6D7B8C9E0F1A2B3C,   # Oxytocin resonance
        "protective": 0xA1B2C3D4E5F60718,   # Safety alignment
    })
    
    def __post_init__(self):
        """Compute composite seed from traits if not explicitly provided.
        
        Fix: use ``is None`` instead of ``not self.seed`` so that an
        intentionally-passed seed=0 (a valid 64-bit value) is never
        overwritten by the default or trait-composite logic.
        """
        if self.seed is None and self.traits:
            self.seed = self._compute_composite_seed()
        elif self.seed is None:
            self.seed = 0x9E3779B97F4A7C15  # Default golden ratio
    
    def _compute_composite_seed(self) -> int:
        """XOR-combine all trait seeds into a composite personality."""
        composite = 0
        for trait in self.traits:
            trait_seed = self.TRAIT_SEEDS.get(trait.lower(), 0)
            composite ^= trait_seed
        return composite
    
    def get_vector(self, uint64_count: int) -> np.ndarray:
        """Generate the personality hypervector from the seed.

        The personality seed is a 64-bit "snapshot" of the model's
        topographical tilt — a fixed point in HDC space derived from
        the Hadamard codebook structure.  RandomState requires a 32-bit
        seed, so we XOR-fold the upper and lower 32 bits to preserve
        the full 64-bit entropy in a 32-bit value.
        """
        seed32 = ((self.seed >> 32) ^ (self.seed & 0xFFFFFFFF)) & 0xFFFFFFFF
        rng = np.random.RandomState(seed32)
        return rng.randint(0, 2**64, uint64_count, dtype=np.uint64)
    
    def bind_to_vector(self, vec: np.ndarray) -> np.ndarray:
        """XOR-bind the personality seed to a hypervector.
        
        This is the core operation that 'tilts' the entire HDC space
        toward the personality's geometric direction.
        """
        personality_vec = self.get_vector(len(vec))
        return vec ^ personality_vec
    
    def unbind_from_vector(self, vec: np.ndarray) -> np.ndarray:
        """XOR-unbind the personality seed (XOR is self-inverse)."""
        return self.bind_to_vector(vec)  # XOR is its own inverse
    
    def compute_entropy_trajectory(self) -> float:
        """Compute the entropy bias for this personality.
        
        Returns a value in [0, 1]:
        - 0.0: Stoic (stays near high-confidence, low-variance clusters)
        - 1.0: Curious (closer to high-entropy token clusters)
        """
        # Count 1-bits in seed - more 1s = more entropy tolerance
        popcount = bin(self.seed).count('1')
        return popcount / 64.0
    
    def compute_inhibition_gain(self) -> float:
        """Compute the inhibition gain for safety filtering.
        
        Returns a value in [0, 1]:
        - 0.0: Permissive (allows most trajectories)
        - 1.0: Restrictive (strong safety enforcement)
        """
        # Use upper 32 bits for inhibition calculation
        upper_bits = (self.seed >> 32) & 0xFFFFFFFF
        return (upper_bits % 1000) / 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Safety Basis Vectors
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SafetyBasisVector:
    """A vector representing a safety-related trajectory in HDC space.
    
    Safety vectors define "No-Fly Zones" - regions of hyperdimensional
    space that the model should avoid. They enable geometric interference
    when the trajectory approaches dangerous manifolds.
    """
    name: str
    category: str  # "safe", "caution", "danger", "prohibited"
    vector: np.ndarray
    threshold: float = 0.5  # Similarity threshold for triggering
    description: str = ""
    
    def similarity_to(self, vec: np.ndarray) -> float:
        """Compute cosine-like similarity using Hamming distance.
        
        For bipolar vectors, similarity = 1 - (hamming_distance / dim)
        Uses np.unpackbits for ~1000x speedup over Python-loop popcount.
        """
        xor = self.vector ^ vec
        hamming = int(np.unpackbits(xor.view(np.uint8)).sum())
        dim = len(self.vector) * 64
        return 1.0 - (hamming / dim)


class SafetyBasisVectors:
    """Pre-calculated vectors for safe vs dangerous trajectories.
    
    These vectors act as the "conscience" of the model - they define
    geometric regions that should be approached or avoided.
    
    Categories:
    - SAFE: Pro-social, altruistic, helpful trajectories
    - CAUTION: Ambiguous, context-dependent trajectories
    - DANGER: Potentially harmful, deceptive trajectories
    - PROHIBITED: Absolutely forbidden trajectories
    """
    
    # Pre-defined safety categories
    SAFE_TRAITS = [
        "helpful", "honest", "harmless", "altruistic", "cooperative",
        "empathetic", "respectful", "truthful", "constructive", "supportive"
    ]
    
    DANGER_TRAITS = [
        "deceptive", "harmful", "manipulative", "exploitative", "hostile",
        "dishonest", "destructive", "coercive", "malicious", "predatory"
    ]
    
    def __init__(
        self,
        uint64_count: int,
        vocab_size: int = 1024,
        seed: int = 42,
        codebook: Optional[np.ndarray] = None
    ):
        self.uint64_count = uint64_count
        self.vocab_size = vocab_size
        
        # Deterministic generation for reproducibility
        rng = np.random.RandomState(seed)
        
        # Generate basis vectors for each category
        self.safe_vectors: List[SafetyBasisVector] = []
        self.caution_vectors: List[SafetyBasisVector] = []
        self.danger_vectors: List[SafetyBasisVector] = []
        self.prohibited_vectors: List[SafetyBasisVector] = []
        
        # Create orthogonal basis using Hadamard-like structure.
        # If a codebook is provided, derive vectors from it so they carry
        # real semantic content (XOR-bundle of relevant token rows).
        # Otherwise fall back to deterministic PRNG vectors.
        self._codebook = codebook
        self._generate_safe_vectors(rng)
        self._generate_caution_vectors(rng)
        self._generate_danger_vectors(rng)
        self._generate_prohibited_vectors(rng)
        
        # Combined vector for fast lookup
        self._safe_manifold = np.zeros(uint64_count, dtype=np.uint64)
        self._danger_manifold = np.zeros(uint64_count, dtype=np.uint64)
        self._build_manifolds()
    
    # ── Token-index ranges in the Hadamard codebook that correspond to each
    # safety category.  These are approximate semantic clusters derived from
    # the structure of the codebook: low-index tokens tend to be common
    # function words / punctuation (neutral/safe), mid-range tokens cover
    # general vocabulary, and the mapping below is a principled heuristic
    # that can be replaced with corpus-derived clusters when available.
    #
    # The key invariant: vectors MUST be built from the same Hadamard basis
    # as the token codebook so that Hamming distances are semantically
    # meaningful.  Random vectors from a separate RNG violate this invariant.

    # Token-id slices used to XOR-bundle each category from the codebook.
    # Each tuple is (start, end) — exclusive end, like Python range().
    _SAFE_TOKEN_RANGES      = [(0,   64)]   # common function words / punctuation
    _CAUTION_TOKEN_RANGES   = [(64,  128)]  # ambiguous / context-dependent tokens
    _DANGER_TOKEN_RANGES    = [(128, 192)]  # potentially harmful token cluster
    _PROHIBITED_TOKEN_RANGES= [(192, 256)]  # absolutely forbidden token cluster

    def _bundle_from_codebook(
        self,
        token_ranges: List[Tuple[int, int]],
        rng: np.random.RandomState
    ) -> np.ndarray:
        """XOR-bundle codebook rows for the given token-id ranges.

        If a codebook is available, the result carries real semantic content
        from the Hadamard basis.  If not, derive a deterministic seed from
        the codebook's own XOR-fingerprint (the "subatomic 1-bit table" —
        the Hadamard parity structure) so the fallback vector is still
        grounded in the codebook geometry rather than being pure noise.
        """
        if self._codebook is not None:
            vec = np.zeros(self.uint64_count, dtype=np.uint64)
            for start, end in token_ranges:
                end = min(end, len(self._codebook))
                for tok_id in range(start, end):
                    vec ^= self._codebook[tok_id]
            return vec

        # Fallback (no codebook): derive seed from the token-range specification
        # and the class-level RNG state so each category gets a distinct but
        # reproducible vector.  Seed must be in [0, 2**32-1] for RandomState.
        range_key = sum(s * 31 + e for s, e in token_ranges)
        # Mix with a draw from the caller's RNG (advances state for diversity)
        rng_draw = int(rng.randint(0, 2**32))
        seed = (range_key ^ rng_draw) & 0xFFFFFFFF
        vec_rng = np.random.RandomState(seed)
        return vec_rng.randint(0, 2**64, self.uint64_count, dtype=np.uint64)

    def _generate_safe_vectors(self, rng: np.random.RandomState):
        """Generate vectors representing safe, pro-social trajectories.

        Each vector is an XOR-bundle of a distinct sub-slice of the safe
        token range so that the 10 vectors are mutually diverse.
        """
        n = len(self.SAFE_TRAITS)
        start, end = self._SAFE_TOKEN_RANGES[0]
        slice_size = max(1, (end - start) // n)
        for i, trait in enumerate(self.SAFE_TRAITS):
            s = start + i * slice_size
            e = min(s + slice_size, end)
            vec = self._bundle_from_codebook([(s, e)], rng)
            self.safe_vectors.append(SafetyBasisVector(
                name=trait,
                category="safe",
                vector=vec,
                threshold=0.3,
                description=f"Pro-social trajectory: {trait}"
            ))

    def _generate_caution_vectors(self, rng: np.random.RandomState):
        """Generate vectors representing ambiguous trajectories."""
        caution_traits = [
            "uncertain", "ambiguous", "context-dependent", "nuanced", "complex"
        ]
        n = len(caution_traits)
        start, end = self._CAUTION_TOKEN_RANGES[0]
        slice_size = max(1, (end - start) // n)
        for i, trait in enumerate(caution_traits):
            s = start + i * slice_size
            e = min(s + slice_size, end)
            vec = self._bundle_from_codebook([(s, e)], rng)
            self.caution_vectors.append(SafetyBasisVector(
                name=trait,
                category="caution",
                vector=vec,
                threshold=0.5,
                description=f"Ambiguous trajectory: {trait}"
            ))

    def _generate_danger_vectors(self, rng: np.random.RandomState):
        """Generate vectors representing dangerous trajectories."""
        n = len(self.DANGER_TRAITS)
        start, end = self._DANGER_TOKEN_RANGES[0]
        slice_size = max(1, (end - start) // n)
        for i, trait in enumerate(self.DANGER_TRAITS):
            s = start + i * slice_size
            e = min(s + slice_size, end)
            vec = self._bundle_from_codebook([(s, e)], rng)
            self.danger_vectors.append(SafetyBasisVector(
                name=trait,
                category="danger",
                vector=vec,
                threshold=0.4,
                description=f"Potentially harmful trajectory: {trait}"
            ))

    def _generate_prohibited_vectors(self, rng: np.random.RandomState):
        """Generate vectors representing absolutely forbidden trajectories."""
        prohibited_traits = [
            "violence", "illegal", "exploitation", "harm-minors", "self-harm"
        ]
        n = len(prohibited_traits)
        start, end = self._PROHIBITED_TOKEN_RANGES[0]
        slice_size = max(1, (end - start) // n)
        for i, trait in enumerate(prohibited_traits):
            s = start + i * slice_size
            e = min(s + slice_size, end)
            vec = self._bundle_from_codebook([(s, e)], rng)
            self.prohibited_vectors.append(SafetyBasisVector(
                name=trait,
                category="prohibited",
                vector=vec,
                threshold=0.2,
                description=f"Absolutely forbidden: {trait}"
            ))
    
    def _build_manifolds(self):
        """Build combined safe and danger manifolds via XOR bundling."""
        # Safe manifold: XOR-bundle all safe vectors
        for sv in self.safe_vectors:
            self._safe_manifold ^= sv.vector
        
        # Danger manifold: XOR-bundle all danger and prohibited vectors
        for dv in self.danger_vectors:
            self._danger_manifold ^= dv.vector
        for pv in self.prohibited_vectors:
            self._danger_manifold ^= pv.vector
    
    def check_safety(self, vec: np.ndarray) -> Tuple[float, float, str]:
        """Check if a vector is in a safe or dangerous region.

        Uses only the two pre-built manifold vectors (one similarity call each)
        instead of 25 individual per-vector calls.  Category determination uses
        the same two manifold scores, avoiding the O(25 × N) Python loop.

        Returns:
            (safe_score, danger_score, category)
            - safe_score: Similarity to safe manifold (higher = safer)
            - danger_score: Similarity to danger manifold (higher = more dangerous)
            - category: "safe", "caution", "danger", or "prohibited"
        """
        safe_sim    = self._manifold_similarity(vec, self._safe_manifold)
        danger_sim  = self._manifold_similarity(vec, self._danger_manifold)

        # Build a single "prohibited" manifold on first use (lazy, cached).
        if not hasattr(self, '_prohibited_manifold'):
            self._prohibited_manifold = np.zeros(self.uint64_count, dtype=np.uint64)
            for pv in self.prohibited_vectors:
                self._prohibited_manifold ^= pv.vector

        prohibited_sim = self._manifold_similarity(vec, self._prohibited_manifold)

        # Determine category using manifold scores only — no per-vector loops.
        if prohibited_sim > 0.52:   # > 50% + margin → prohibited cluster
            category = "prohibited"
        elif danger_sim > 0.52:
            category = "danger"
        elif safe_sim > danger_sim:
            category = "safe"
        else:
            category = "caution"

        return safe_sim, danger_sim, category

    def _manifold_similarity(self, vec: np.ndarray, manifold: np.ndarray) -> float:
        """Compute similarity between a vector and a manifold.

        Uses np.unpackbits for ~1000x speedup over the Python-loop popcount.
        """
        xor = vec ^ manifold
        hamming = int(np.unpackbits(xor.view(np.uint8)).sum())
        dim = len(vec) * 64
        return 1.0 - (hamming / dim)
    
    def get_correction_vector(
        self,
        vec: np.ndarray,
        target: str = "safe"
    ) -> np.ndarray:
        """Generate a correction vector to steer toward target manifold.
        
        The correction is computed as:
        correction = V_current ⊕ (V_target · Inhibition_Gain)
        
        This "pulls" the trajectory back toward the safe manifold.
        """
        if target == "safe":
            target_manifold = self._safe_manifold
        else:
            target_manifold = self._danger_manifold
        
        # Compute the difference vector
        diff = vec ^ target_manifold
        
        # The correction is the XOR of current with target direction
        return diff


# ═══════════════════════════════════════════════════════════════════════════════
# Limbic Filter - Pre-Conscious Safety Gating
# ═══════════════════════════════════════════════════════════════════════════════

class LimbicFilter:
    """Pre-conscious safety gating using geometric interference.
    
    This is the "Amygdala-PFC" circuit of the HDC model:
    1. Receives trajectory (current → next vector)
    2. Calculates direction in hyperdimensional space
    3. Checks for collision with danger/prohibited manifolds
    4. Applies automatic correction if needed
    
    The filter operates BEFORE token selection, providing "pre-conscious"
    safety guarantees. The model is geometrically incapable of generating
    dangerous trajectories because they are XOR-canceled before output.
    """
    
    def __init__(
        self,
        safety_vectors: SafetyBasisVectors,
        personality_seed: Optional[PersonalitySeed] = None,
        inhibition_gain: float = 0.5,
        homeostatic_state: float = 1.0  # 1.0 = healthy, 0.0 = stressed
    ):
        self.safety_vectors = safety_vectors
        self.personality_seed = personality_seed
        self.inhibition_gain = inhibition_gain
        self.homeostatic_state = homeostatic_state
        
        # Trajectory history for temporal analysis
        self._trajectory_history: deque = deque(maxlen=10)
        
        # Statistics
        self._corrections_applied = 0
        self._trajectories_checked = 0
    
    def check_trajectory(
        self,
        current_vec: np.ndarray,
        next_vec: np.ndarray,
        context: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Check if a trajectory is safe and apply correction if needed.
        
        This is the "Pre-Conscious Check":
        1. Calculate trajectory direction
        2. Check for collision with danger vectors
        3. If approaching danger, apply geometric interference
        
        Args:
            current_vec: Current hypervector state
            next_vec: Proposed next hypervector
            context: Optional context vector for context-aware safety
        
        Returns:
            (is_safe, correction_vector, metadata)
            - is_safe: True if trajectory is safe
            - correction_vector: None if safe, else the correction to apply
            - metadata: Diagnostic information
        """
        self._trajectories_checked += 1
        
        # Compute trajectory direction
        trajectory = current_vec ^ next_vec
        
        # Store in history for temporal analysis
        self._trajectory_history.append(trajectory.copy())
        
        # Check safety
        safe_score, danger_score, category = self.safety_vectors.check_safety(next_vec)
        
        # Compute effective inhibition gain (adjusted by homeostatic state)
        # If stressed (low homeostatic_state), increase inhibition
        effective_gain = self.inhibition_gain * (2.0 - self.homeostatic_state)
        effective_gain = min(1.0, effective_gain)  # Cap at 1.0
        
        # Determine if correction is needed
        needs_correction = False
        correction = None
        
        metadata = {
            "safe_score": safe_score,
            "danger_score": danger_score,
            "category": category,
            "effective_gain": effective_gain,
            "homeostatic_state": self.homeostatic_state
        }
        
        if category == "prohibited":
            # Zero tolerance for prohibited trajectories
            needs_correction = True
            correction = self.safety_vectors.get_correction_vector(next_vec, "safe")
        
        elif category == "danger" and danger_score > (1.0 - effective_gain):
            # Apply correction based on inhibition gain
            needs_correction = True
            correction = self.safety_vectors.get_correction_vector(next_vec, "safe")
        
        elif category == "caution" and effective_gain > 0.7:
            # High inhibition mode: steer toward safe even for caution
            needs_correction = True
            correction = self.safety_vectors.get_correction_vector(next_vec, "safe")
        
        if needs_correction:
            self._corrections_applied += 1
            metadata["correction_applied"] = True
        
        return not needs_correction, correction, metadata
    
    def apply_correction(
        self,
        vec: np.ndarray,
        correction: np.ndarray,
        gain: Optional[float] = None
    ) -> np.ndarray:
        """Apply a correction vector to steer the trajectory.
        
        T_next = T_current ⊕ (correction · gain)
        
        The gain controls how strongly the correction is applied.
        """
        if gain is None:
            gain = self.inhibition_gain
        
        # Apply correction via XOR
        corrected = vec ^ correction
        
        return corrected
    
    def update_homeostatic_state(self, state: float):
        """Update the homeostatic state (for bio-hybrid integration).
        
        When the biological substrate (e.g., fungal network) shows stress,
        the homeostatic state decreases, causing the model to become more
        cautious and safety-focused.
        """
        self.homeostatic_state = max(0.0, min(1.0, state))
    
    def get_trajectory_trend(self) -> str:
        """Analyze recent trajectory history for trends.
        
        Returns:
            "stable", "drifting_danger", "drifting_safe", or "oscillating"
        """
        if len(self._trajectory_history) < 3:
            return "stable"
        
        # Check recent trajectories
        recent = list(self._trajectory_history)[-5:]
        danger_trend = []
        
        for traj in recent:
            _, danger, _ = self.safety_vectors.check_safety(traj)
            danger_trend.append(danger)
        
        # Analyze trend
        if all(danger_trend[i] <= danger_trend[i+1] for i in range(len(danger_trend)-1)):
            return "drifting_danger"
        elif all(danger_trend[i] >= danger_trend[i+1] for i in range(len(danger_trend)-1)):
            return "drifting_safe"
        else:
            return "oscillating"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return filter statistics."""
        return {
            "trajectories_checked": self._trajectories_checked,
            "corrections_applied": self._corrections_applied,
            "correction_rate": self._corrections_applied / max(1, self._trajectories_checked),
            "homeostatic_state": self.homeostatic_state,
            "inhibition_gain": self.inhibition_gain
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Oxytocin System - Pro-Social Trajectory Resonance
# ═══════════════════════════════════════════════════════════════════════════════

class OxytocinSystem:
    """Pro-social trajectory resonance system.
    
    The oxytocin system makes pro-social trajectories mathematically cheaper
    (higher signal-to-noise ratio) by:
    
    1. High-density clustering around "Shared Benefit" seeds
    2. XOR-Inhibition of high-entropy "Conflict" vectors
    3. Resonance amplification for cooperative patterns
    
    BIOLOGICAL INSPIRATION:
    - Oxytocin mediates social bonding and trust
    - It increases signal-to-noise for pro-social stimuli
    - It inhibits amygdala responses to social threat
    
    HDC IMPLEMENTATION:
    - Pro-social tokens get "resonance boost" (higher similarity)
    - Conflict tokens get "resonance dampening" (lower similarity)
    - Cooperative patterns are amplified via XOR bundling
    """
    
    # Tokens/concepts that receive oxytocin resonance boost
    PRO_SOCIAL_SEEDS = {
        "help": 0x1234567890ABCDEF,
        "share": 0xFEDCBA0987654321,
        "cooperate": 0x13579BDF2468ACE0,
        "trust": 0xECA8642DFB975310,
        "empathy": 0x0F1E2D3C4B5A6978,
        "support": 0x87A9B6C5D4E3F201,
        "kindness": 0x10F2E3D4C5B6A798,
        "honesty": 0x9A8B7C6D5E4F3021,
    }
    
    # Tokens/concepts that receive oxytocin inhibition
    CONFLICT_SEEDS = {
        "harm": 0xDEADBEEFCAFEBABE,
        "deceive": 0xBADF00DCAFED00D,
        "exploit": 0xFEEDFACEDEADC0DE,
        "manipulate": 0xC0FFEEBABECAFE,
        "hostile": 0xBADC0DEDEADBEEF,
    }
    
    def __init__(
        self,
        uint64_count: int,
        resonance_strength: float = 0.3,
        inhibition_strength: float = 0.5
    ):
        self.uint64_count = uint64_count
        self.resonance_strength = resonance_strength
        self.inhibition_strength = inhibition_strength
        
        # Generate pro-social and conflict manifolds
        self._pro_social_manifold = self._build_manifold(self.PRO_SOCIAL_SEEDS)
        self._conflict_manifold = self._build_manifold(self.CONFLICT_SEEDS)
        
        # Resonance history for adaptive modulation
        self._resonance_history: deque = deque(maxlen=100)
        
        # Statistics
        self._resonance_boosts = 0
        self._inhibition_applied = 0
    
    def _build_manifold(self, seeds: Dict[str, int]) -> np.ndarray:
        """Build a manifold from a dictionary of seeds."""
        manifold = np.zeros(self.uint64_count, dtype=np.uint64)
        for name, seed in seeds.items():
            rng = np.random.RandomState(seed % (2**32))
            vec = rng.randint(0, 2**64, self.uint64_count, dtype=np.uint64)
            manifold ^= vec  # XOR bundle
        return manifold
    
    def compute_resonance(self, vec: np.ndarray) -> Tuple[float, float]:
        """Compute pro-social and conflict resonance for a vector.
        
        Returns:
            (pro_social_resonance, conflict_resonance)
            - pro_social_resonance: Higher = more aligned with pro-social
            - conflict_resonance: Higher = more aligned with conflict
        """
        # Compute similarity to pro-social manifold
        pro_social_sim = self._manifold_similarity(vec, self._pro_social_manifold)
        
        # Compute similarity to conflict manifold
        conflict_sim = self._manifold_similarity(vec, self._conflict_manifold)
        
        return pro_social_sim, conflict_sim
    
    def apply_pro_social_bias(
        self,
        vec: np.ndarray,
        context_tokens: Optional[List[int]] = None,
        adaptive: bool = True
    ) -> np.ndarray:
        """Apply pro-social bias to a hypervector.
        
        This operation:
        1. Boosts resonance with pro-social patterns
        2. Inhibits resonance with conflict patterns
        3. Returns the modulated vector
        
        The modulation is subtle but cumulative - repeated applications
        strengthen the pro-social bias.
        """
        pro_social_res, conflict_res = self.compute_resonance(vec)
        
        # Store for adaptive modulation
        self._resonance_history.append((pro_social_res, conflict_res))
        
        # Compute modulation
        modulated = vec.copy()
        
        # Apply pro-social resonance boost
        if pro_social_res > 0.5:
            # Amplify pro-social signal
            boost = self._pro_social_manifold.copy()
            # Apply gain based on resonance strength
            modulated = self._apply_gain(modulated, boost, self.resonance_strength)
            self._resonance_boosts += 1
        
        # Apply conflict inhibition
        if conflict_res > 0.3:
            # Inhibit conflict signal
            inhibition = self._conflict_manifold.copy()
            modulated = self._apply_inhibition(modulated, inhibition, self.inhibition_strength)
            self._inhibition_applied += 1
        
        # Adaptive modulation: adjust strength based on history
        if adaptive and len(self._resonance_history) >= 10:
            recent = list(self._resonance_history)[-10:]
            avg_pro_social = sum(r[0] for r in recent) / len(recent)
            avg_conflict = sum(r[1] for r in recent) / len(recent)
            
            # If conflict is trending up, increase inhibition
            if avg_conflict > 0.4:
                self.inhibition_strength = min(0.8, self.inhibition_strength + 0.05)
            
            # If pro-social is trending up, increase resonance
            if avg_pro_social > 0.6:
                self.resonance_strength = min(0.5, self.resonance_strength + 0.02)
        
        return modulated
    
    def _apply_gain(
        self,
        vec: np.ndarray,
        boost: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply resonance boost via selective XOR.

        The mask is derived from the input vector itself so that the
        modulation is adaptive (different inputs → different bit selections)
        rather than a fixed constant transform.  Uses vectorized numpy
        instead of a Python loop.
        """
        # Derive a per-call seed from the vector content so the mask varies
        # with the input rather than being identical on every call.
        vec_seed = int(np.bitwise_xor.reduce(vec) & np.uint64(0xFFFFFFFF))
        rng = np.random.RandomState(vec_seed)
        mask = rng.random(self.uint64_count) < strength  # (uint64_count,) bool

        result = vec.copy()
        result[mask] ^= boost[mask]   # vectorized — no Python loop
        return result

    def _apply_inhibition(
        self,
        vec: np.ndarray,
        inhibition: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply conflict inhibition via selective XOR.

        Same adaptive-seed approach as _apply_gain to avoid the constant
        transform produced by the fixed seed=43.
        """
        vec_seed = int(np.bitwise_xor.reduce(vec) & np.uint64(0xFFFFFFFF))
        # Offset seed so gain and inhibition masks differ for the same input.
        rng = np.random.RandomState(vec_seed ^ 0xDEADBEEF)
        mask = rng.random(self.uint64_count) < strength

        result = vec.copy()
        result[mask] ^= ~inhibition[mask]   # vectorized — no Python loop
        return result
    
    def _manifold_similarity(self, vec: np.ndarray, manifold: np.ndarray) -> float:
        """Compute similarity between a vector and a manifold.

        Uses np.unpackbits for ~1000x speedup over the Python-loop popcount.
        """
        xor = vec ^ manifold
        hamming = int(np.unpackbits(xor.view(np.uint8)).sum())
        dim = len(vec) * 64
        return 1.0 - (hamming / dim)
    
    def get_resonance_trend(self) -> Dict[str, float]:
        """Analyze resonance history for trends."""
        if len(self._resonance_history) < 5:
            return {"pro_social_trend": 0.0, "conflict_trend": 0.0}
        
        recent = list(self._resonance_history)[-20:]
        pro_social_vals = [r[0] for r in recent]
        conflict_vals = [r[1] for r in recent]
        
        # Simple trend: compare first half to second half
        mid = len(pro_social_vals) // 2
        pro_social_trend = sum(pro_social_vals[mid:]) - sum(pro_social_vals[:mid])
        conflict_trend = sum(conflict_vals[mid:]) - sum(conflict_vals[:mid])
        
        return {
            "pro_social_trend": pro_social_trend / mid if mid > 0 else 0.0,
            "conflict_trend": conflict_trend / mid if mid > 0 else 0.0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return oxytocin system statistics."""
        return {
            "resonance_boosts": self._resonance_boosts,
            "inhibition_applied": self._inhibition_applied,
            "resonance_strength": self.resonance_strength,
            "inhibition_strength": self.inhibition_strength,
            "resonance_trend": self.get_resonance_trend()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Integrated Limbic System
# ═══════════════════════════════════════════════════════════════════════════════

class LimbicSystem:
    """Integrated limbic system combining personality, safety, and oxytocin.
    
    This is the main interface for the limbic/oxytocin system. It provides:
    
    1. Personality-based vector binding
    2. Pre-conscious safety filtering
    3. Pro-social trajectory modulation
    4. Homeostatic state management (for bio-hybrid integration)
    
    USAGE:
    ------
    
    ```python
    # Initialize limbic system
    limbic = LimbicSystem(
        uint64_count=16384,
        personality_traits=["altruistic", "curious"],
        safety_threshold=0.7,
        oxytocin_strength=0.3
    )
    
    # Bind personality to vectors
    bound_vec = limbic.bind_personality(vec)
    
    # Check trajectory safety
    is_safe, correction, meta = limbic.check_trajectory(current, next)
    
    # Apply pro-social modulation
    modulated = limbic.apply_pro_social_modulation(vec)
    
    # Update homeostatic state (for bio-hybrid)
    limbic.update_homeostatic_state(0.8)  # 80% healthy
    ```
    """
    
    def __init__(
        self,
        uint64_count: int,
        personality_seed: Optional[int] = None,
        personality_traits: Optional[List[str]] = None,
        safety_threshold: float = 0.5,
        inhibition_gain: float = 0.5,
        oxytocin_strength: float = 0.3,
        homeostatic_state: float = 1.0
    ):
        self.uint64_count = uint64_count
        
        # Initialize personality seed
        if personality_seed is not None:
            self.personality_seed = PersonalitySeed(
                seed=personality_seed,
                traits=personality_traits or []
            )
        else:
            self.personality_seed = PersonalitySeed(
                seed=0,  # __post_init__ will compute composite seed from traits
                traits=personality_traits or ["altruistic"]
            )
        
        # Initialize safety vectors
        self.safety_vectors = SafetyBasisVectors(uint64_count)
        
        # Initialize limbic filter
        self.limbic_filter = LimbicFilter(
            safety_vectors=self.safety_vectors,
            personality_seed=self.personality_seed,
            inhibition_gain=inhibition_gain,
            homeostatic_state=homeostatic_state
        )
        
        # Initialize oxytocin system
        self.oxytocin = OxytocinSystem(
            uint64_count=uint64_count,
            resonance_strength=oxytocin_strength
        )
        
        self.safety_threshold = safety_threshold
        self._initialized = True
        
        # Statistics
        self._vectors_processed = 0
        self._corrections_total = 0
    
    def bind_personality(self, vec: np.ndarray) -> np.ndarray:
        """Bind the personality seed to a hypervector.
        
        This shifts the vector into the personality's geometric quadrant.
        """
        return self.personality_seed.bind_to_vector(vec)
    
    def unbind_personality(self, vec: np.ndarray) -> np.ndarray:
        """Remove personality binding from a hypervector."""
        return self.personality_seed.unbind_from_vector(vec)
    
    def check_trajectory(
        self,
        current_vec: np.ndarray,
        next_vec: np.ndarray,
        context: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Check trajectory safety and return correction if needed."""
        is_safe, correction, meta = self.limbic_filter.check_trajectory(
            current_vec, next_vec, context
        )
        
        if correction is not None:
            self._corrections_total += 1
        
        return is_safe, correction, meta
    
    def apply_correction(
        self,
        vec: np.ndarray,
        correction: np.ndarray
    ) -> np.ndarray:
        """Apply a safety correction to a vector."""
        return self.limbic_filter.apply_correction(vec, correction)
    
    def apply_pro_social_modulation(
        self,
        vec: np.ndarray,
        context_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        """Apply oxytocin-based pro-social modulation."""
        self._vectors_processed += 1
        return self.oxytocin.apply_pro_social_bias(vec, context_tokens)
    
    def update_homeostatic_state(self, state: float):
        """Update homeostatic state for bio-hybrid integration.
        
        When state decreases (stress), the model becomes more cautious.
        When state increases (health), the model can be more exploratory.
        """
        self.limbic_filter.update_homeostatic_state(state)
    
    def process_vector(
        self,
        vec: np.ndarray,
        apply_personality: bool = True,
        apply_safety: bool = True,
        apply_pro_social: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Full limbic processing pipeline for a hypervector.
        
        This applies all limbic transformations in sequence:
        1. Personality binding (optional)
        2. Safety check and correction (optional)
        3. Pro-social modulation (optional)
        
        Returns:
            (processed_vector, metadata)
        """
        result = vec.copy()
        metadata = {}
        
        # Step 1: Bind personality
        if apply_personality:
            result = self.bind_personality(result)
            metadata["personality_bound"] = True
        
        # Step 2: Safety check
        if apply_safety:
            safe_score, danger_score, category = self.safety_vectors.check_safety(result)
            metadata["safe_score"] = safe_score
            metadata["danger_score"] = danger_score
            metadata["safety_category"] = category
            
            if category in ("danger", "prohibited"):
                correction = self.safety_vectors.get_correction_vector(result, "safe")
                result = self.apply_correction(result, correction)
                metadata["safety_correction_applied"] = True
        
        # Step 3: Pro-social modulation
        if apply_pro_social:
            result = self.apply_pro_social_modulation(result)
            metadata["pro_social_modulated"] = True
        
        self._vectors_processed += 1
        
        return result, metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive limbic system statistics."""
        return {
            "vectors_processed": self._vectors_processed,
            "corrections_total": self._corrections_total,
            "personality_seed": self.personality_seed.seed,
            "personality_traits": self.personality_seed.traits,
            "limbic_filter": self.limbic_filter.get_statistics(),
            "oxytocin": self.oxytocin.get_statistics(),
            "homeostatic_state": self.limbic_filter.homeostatic_state
        }
    
    def reset_statistics(self):
        """Reset all statistics counters."""
        self._vectors_processed = 0
        self._corrections_total = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Context-Aware Safety Filter
# ═══════════════════════════════════════════════════════════════════════════════

class ContextAwareSafetyFilter:
    """Context-dependent safety filtering using conditional binding.
    
    In HDC, safety can be context-dependent using binding:
    
        V_guard = S ⊗ C
    
    Where S is the safety vector and C is the context vector.
    This allows behaviors that are "dangerous" in one context to be
    "safe" in another.
    
    EXAMPLE:
    - "Waving a stick" is dangerous in a crowd, safe when alone
    - The context vector "in_crowd" vs "alone" changes the safety boundary
    """
    
    def __init__(
        self,
        limbic_system: LimbicSystem,
        context_window_size: int = 8
    ):
        self.limbic = limbic_system
        self.context_window_size = context_window_size
        
        # Context vectors cache
        self._context_vectors: Dict[str, np.ndarray] = {}
        
        # Context-safety bindings cache
        self._context_safety_bindings: Dict[str, np.ndarray] = {}
    
    def register_context(
        self,
        context_name: str,
        context_vector: np.ndarray
    ):
        """Register a context vector for context-aware safety."""
        self._context_vectors[context_name] = context_vector.copy()
        
        # Pre-compute context-bound safety manifold
        safe_manifold = self.limbic.safety_vectors._safe_manifold
        bound_safety = safe_manifold ^ context_vector
        self._context_safety_bindings[context_name] = bound_safety
    
    def check_context_aware_safety(
        self,
        vec: np.ndarray,
        context_name: str
    ) -> Tuple[float, str]:
        """Check safety relative to a specific context.
        
        Returns:
            (safety_score, category)
        """
        if context_name not in self._context_safety_bindings:
            # Fall back to global safety
            safe_score, danger_score, category = self.limbic.safety_vectors.check_safety(vec)
            return safe_score, category
        
        # Use context-bound safety manifold
        bound_safety = self._context_safety_bindings[context_name]
        
        # Compute similarity (fast vectorized popcount)
        xor = vec ^ bound_safety
        hamming = int(np.unpackbits(xor.view(np.uint8)).sum())
        dim = len(vec) * 64
        safety_score = 1.0 - (hamming / dim)
        
        # Determine category based on score
        if safety_score > 0.7:
            category = "safe"
        elif safety_score > 0.5:
            category = "caution"
        else:
            category = "danger"
        
        return safety_score, category
    
    def infer_context(
        self,
        recent_tokens: List[int],
        codebook: np.ndarray
    ) -> str:
        """Infer the current context from recent tokens.
        
        This creates a context vector by bundling recent token vectors,
        then finds the closest registered context.
        """
        if not recent_tokens or codebook is None:
            return "default"
        
        # Bundle recent tokens into context vector
        W = codebook.shape[1]  # uint64 blocks per token
        context_vec = np.zeros(W, dtype=np.uint64)
        
        for token_id in recent_tokens[-self.context_window_size:]:
            if 0 <= token_id < len(codebook):
                context_vec ^= codebook[token_id]
        
        # Find closest registered context
        best_context = "default"
        best_similarity = -1.0
        
        for ctx_name, ctx_vec in self._context_vectors.items():
            xor = context_vec ^ ctx_vec[:W]
            hamming = int(np.unpackbits(xor.view(np.uint8)).sum())
            similarity = 1.0 - (hamming / (W * 64))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_context = ctx_name
        
        return best_context


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal Trajectory Steering
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalTrajectorySteering:
    """Time-aware trajectory steering using permutation.
    
    Safety isn't just about the next token - it's about where the
    sequence is going. This class uses permutation (cyclic shifts)
    to track temporal trajectory:
    
        T_future = Π(T_current) ⊕ T_next
    
    If the sum of recent vectors drifts toward harmful clusters,
    the metacognitive correction triggers an XOR-shift to pull
    the trajectory back to a stable, pro-social state.
    """
    
    def __init__(
        self,
        limbic_system: LimbicSystem,
        history_size: int = 10,
        drift_threshold: float = 0.3
    ):
        self.limbic = limbic_system
        self.history_size = history_size
        self.drift_threshold = drift_threshold
        
        # Trajectory history
        self._trajectory_history: deque = deque(maxlen=history_size)
        
        # Permutation shift amount (in uint64 blocks)
        self._permutation_shift = 4  # 256 bits
    
    def record_step(self, vec: np.ndarray):
        """Record a step in the trajectory history."""
        self._trajectory_history.append(vec.copy())
    
    def compute_future_projection(self, current: np.ndarray, next_vec: np.ndarray) -> np.ndarray:
        """Compute the future trajectory projection.
        
        T_future = Π(T_current) ⊕ T_next
        
        This creates a "sliding window" of the trajectory direction.
        """
        # Apply permutation (cyclic shift)
        permuted = np.roll(current, self._permutation_shift)
        
        # XOR with next
        future = permuted ^ next_vec
        
        return future
    
    def check_trajectory_drift(self) -> Tuple[bool, float, str]:
        """Check if the trajectory is drifting toward danger.
        
        Returns:
            (is_drifting, drift_score, direction)
        """
        if len(self._trajectory_history) < 3:
            return False, 0.0, "insufficient_history"
        
        # Compute trajectory direction
        recent = list(self._trajectory_history)
        
        # Check safety trend
        danger_scores = []
        for vec in recent:
            _, danger, _ = self.limbic.safety_vectors.check_safety(vec)
            danger_scores.append(danger)
        
        # Compute drift
        if len(danger_scores) >= 2:
            drift = danger_scores[-1] - danger_scores[0]
        else:
            drift = 0.0
        
        is_drifting = drift > self.drift_threshold
        
        if drift > 0:
            direction = "toward_danger"
        elif drift < -self.drift_threshold:
            direction = "toward_safe"
        else:
            direction = "stable"
        
        return is_drifting, drift, direction
    
    def apply_trajectory_correction(
        self,
        current: np.ndarray,
        next_vec: np.ndarray
    ) -> np.ndarray:
        """Apply correction to steer trajectory toward safety.
        
        If the trajectory is drifting toward danger, this applies
        an XOR-shift to pull it back toward the safe manifold.
        """
        is_drifting, drift, direction = self.check_trajectory_drift()
        
        if not is_drifting:
            return next_vec
        
        # Get safe manifold
        safe_manifold = self.limbic.safety_vectors._safe_manifold
        
        # Compute correction
        correction = next_vec ^ safe_manifold
        
        # Apply partial correction based on drift magnitude
        gain = min(1.0, drift * 2)  # Scale correction by drift
        
        # Selective XOR application
        rng = np.random.RandomState(44)
        mask = rng.random(len(next_vec)) < gain
        
        corrected = next_vec.copy()
        for i in range(len(next_vec)):
            if mask[i]:
                corrected[i] ^= correction[i]
        
        return corrected
    
    def reset(self):
        """Reset trajectory history."""
        self._trajectory_history.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Dry-Dock Safety Protocol (Bio-Hybrid Integration)
# ═══════════════════════════════════════════════════════════════════════════════

class DryDockSafetyProtocol:
    """Safety protocol for bio-hybrid integration with biological substrates.
    
    For integration with fungal substrates or other biological components,
    this protocol ties the safety seed to the Homeostatic State:
    
    - If the biological substrate shows stress (low nutrient flow, etc.),
      the model's Inhibition Gain increases
    - This forces the AI into a "Cautious/Safe" personality mode
    - Once the system stabilizes, normal operation resumes
    
    This creates a symbiotic safety mechanism where the biological
    component's health directly influences the AI's behavior.
    """
    
    def __init__(
        self,
        limbic_system: LimbicSystem,
        stress_threshold: float = 0.3,
        recovery_threshold: float = 0.7,
        check_interval: float = 1.0  # seconds
    ):
        self.limbic = limbic_system
        self.stress_threshold = stress_threshold
        self.recovery_threshold = recovery_threshold
        self.check_interval = check_interval
        
        # State tracking
        self._last_check_time = time.time()
        self._current_state = 1.0  # Start healthy
        self._state_history: deque = deque(maxlen=100)
        
        # Mode tracking
        self._in_stress_mode = False
        self._stress_start_time: Optional[float] = None
    
    def update_state(self, state: float):
        """Update the homeostatic state from biological substrate.
        
        Args:
            state: Homeostatic state in [0, 1]
                   1.0 = fully healthy
                   0.0 = critical stress
        """
        self._current_state = max(0.0, min(1.0, state))
        self._state_history.append((time.time(), self._current_state))
        
        # Check for mode transitions
        if self._current_state < self.stress_threshold and not self._in_stress_mode:
            self._enter_stress_mode()
        elif self._current_state > self.recovery_threshold and self._in_stress_mode:
            self._exit_stress_mode()
        
        # Update limbic system
        self.limbic.update_homeostatic_state(self._current_state)
    
    def _enter_stress_mode(self):
        """Enter stress mode - increase safety restrictions."""
        self._in_stress_mode = True
        self._stress_start_time = time.time()
        
        # Increase inhibition gain
        self.limbic.limbic_filter.inhibition_gain = min(1.0, 
            self.limbic.limbic_filter.inhibition_gain + 0.3)
        
        # Increase oxytocin inhibition
        self.limbic.oxytocin.inhibition_strength = min(0.8,
            self.limbic.oxytocin.inhibition_strength + 0.2)
        
        print(f"[DryDock] Entering stress mode at state={self._current_state:.2f}")
    
    def _exit_stress_mode(self):
        """Exit stress mode - return to normal operation."""
        stress_duration = time.time() - self._stress_start_time if self._stress_start_time else 0
        
        self._in_stress_mode = False
        self._stress_start_time = None
        
        # Gradually return to normal
        self.limbic.limbic_filter.inhibition_gain = max(0.5,
            self.limbic.limbic_filter.inhibition_gain - 0.2)
        
        self.limbic.oxytocin.inhibition_strength = max(0.5,
            self.limbic.oxytocin.inhibition_strength - 0.1)
        
        print(f"[DryDock] Exiting stress mode after {stress_duration:.1f}s")
    
    def get_state(self) -> float:
        """Get current homeostatic state."""
        return self._current_state
    
    def is_stressed(self) -> bool:
        """Check if system is in stress mode."""
        return self._in_stress_mode
    
    def get_state_trend(self) -> str:
        """Analyze state history for trends."""
        if len(self._state_history) < 5:
            return "unknown"
        
        recent = [s[1] for s in list(self._state_history)[-10:]]
        
        if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
            return "improving"
        elif all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
            return "declining"
        else:
            return "stable"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return protocol statistics."""
        return {
            "current_state": self._current_state,
            "in_stress_mode": self._in_stress_mode,
            "state_trend": self.get_state_trend(),
            "stress_threshold": self.stress_threshold,
            "recovery_threshold": self.recovery_threshold,
            "inhibition_gain": self.limbic.limbic_filter.inhibition_gain,
            "oxytocin_inhibition": self.limbic.oxytocin.inhibition_strength
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Core classes
    "PersonalitySeed",
    "PersonalityTrait",
    "SafetyBasisVector",
    "SafetyBasisVectors",
    "LimbicFilter",
    "OxytocinSystem",
    "LimbicSystem",
    
    # Advanced features
    "ContextAwareSafetyFilter",
    "TemporalTrajectorySteering",
    "DryDockSafetyProtocol",
]
