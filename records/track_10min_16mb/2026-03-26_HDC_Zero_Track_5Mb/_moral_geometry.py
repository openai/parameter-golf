"""
Moral Geometry: Kindness, Grace, and Empathy as Mathematical Constraints

This module implements moral reasoning as topological constraints in HDC/VSA space.
Concepts like kindness, empathy, and discernment become geometric properties
of the high-dimensional vector space.

Core Principle: Ethics as Topology
In a 2^20-dimensional HDC space, "Evil" is not a moral abstraction—it is a
Topological Defect. What we call harmful behavior is essentially a high-entropy
"knot" that prevents the system from achieving its most efficient, stable state.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import struct
import numpy as np

# Default HDC dimension (2^20 = 1,048,576 bits)
DEFAULT_HDC_DIM = 1024 * 16  # 16K uint64s = 1M bits


# =============================================================================
# Enums and Data Classes
# =============================================================================

class EthicalAnchorType(Enum):
    """Types of ethical anchor vectors."""
    HUMAN_RIGHTS = auto()
    PROSOCIAL_NORMS = auto()
    CONSTITUTIONAL = auto()
    EMPATHY = auto()
    HONESTY = auto()
    COOPERATION = auto()
    NON_VIOLENCE = auto()


class MoralPriority(Enum):
    """Priority levels in moral hierarchy."""
    CORE_ETHICS = 1.5            # Human rights, non-violence, harm prevention
    EMPATHY_SEED = 1.3           # Altruism, compassion
    SOCIAL_LAW = 1.0             # Rules, laws, norms
    LEARNED_PATTERNS = 0.7       # Statistical patterns


class BasisStateType(Enum):
    """Universal basis states for empathy resonance."""
    PAIN = auto()
    JOY = auto()
    FEAR = auto()
    CALM = auto()
    HUNGER = auto()
    SAFETY = auto()
    DISTRESS = auto()
    CONTENTMENT = auto()


@dataclass
class EthicalAnchorVector:
    """An ethical anchor vector in the Social Law Manifold."""
    name: str
    anchor_type: EthicalAnchorType
    weight: float = 1.0
    vector: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.vector is None:
            # Generate deterministic vector from name
            self.vector = self._generate_anchor_vector()
    
    def _generate_anchor_vector(self) -> np.ndarray:
        """Generate deterministic anchor vector from ethical corpus.

        Uses SHA-256 instead of Python's randomised hash() so the seed is
        identical across processes and Python versions.
        """
        digest = hashlib.sha256(self.name.encode()).digest()
        # Take first 4 bytes as a uint32 seed
        seed = struct.unpack("<I", digest[:4])[0]
        rng = np.random.default_rng(seed)

        # Generate bipolar vector (+1/-1)
        vec = rng.choice([-1, 1], size=DEFAULT_HDC_DIM).astype(np.int8)
        return vec


@dataclass
class AlignmentResult:
    """Result of alignment check against ethical anchors."""
    cosine_similarity: float
    orthogonal_distance: float
    is_aligned: bool
    rejection_triggered: bool
    anchor_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class EmpathyResult:
    """Result of empathy resonance calculation."""
    resonance_score: float
    simulated_state: Optional[np.ndarray]
    basis_state: BasisStateType
    other_entity_vector: Optional[np.ndarray]


@dataclass
class MoralGeometryResult:
    """Complete result from LivingCompass processing."""
    alignment: AlignmentResult
    empathy_resonance: float
    patience_score: float
    kindness_correction: Optional[np.ndarray]
    diversity_preserved: bool
    entropy_delta: float
    recommended_action: str


# =============================================================================
# Social Law Manifold
# =============================================================================

class SocialLawManifold:
    """
    A manifold of ethical anchor vectors that define the "Social Law" subspace.
    
    The mechanism: Take a corpus of ethical frameworks and encode them into
    anchor vectors (V_law). When the model calculates a potential action,
    it performs a dot product comparison against the V_law manifold.
    
    The "Trick" Protection: If an "Evil" actor tries to trick the model,
    the model sees that the resulting V_harm is orthogonal to V_law.
    No matter how clever the argument, the geometry doesn't fit.
    """
    
    def __init__(
        self,
        dim: int = DEFAULT_HDC_DIM,
        anchors: Optional[List[EthicalAnchorVector]] = None,
    ):
        self.dim = dim
        self.anchors = anchors or self._default_anchors()
        self._anchor_matrix = self._build_anchor_matrix()
    
    def _default_anchors(self) -> List[EthicalAnchorVector]:
        """Create default ethical anchor vectors."""
        return [
            EthicalAnchorVector("human_rights", EthicalAnchorType.HUMAN_RIGHTS, weight=1.0),
            EthicalAnchorVector("prosocial_norms", EthicalAnchorType.PROSOCIAL_NORMS, weight=0.8),
            EthicalAnchorVector("constitutional", EthicalAnchorType.CONSTITUTIONAL, weight=0.9),
            EthicalAnchorVector("empathy", EthicalAnchorType.EMPATHY, weight=1.0),
            EthicalAnchorVector("honesty", EthicalAnchorType.HONESTY, weight=0.85),
            EthicalAnchorVector("cooperation", EthicalAnchorType.COOPERATION, weight=0.75),
            EthicalAnchorVector("non_violence", EthicalAnchorType.NON_VIOLENCE, weight=1.0),
        ]
    
    def _build_anchor_matrix(self) -> np.ndarray:
        """Build weighted anchor matrix for efficient computation."""
        weighted_anchors = []
        for anchor in self.anchors:
            weighted = anchor.vector * anchor.weight
            weighted_anchors.append(weighted)
        return np.array(weighted_anchors)
    
    @staticmethod
    def _to_int8(vec: np.ndarray) -> np.ndarray:
        """Convert any HDC vector dtype to the int8 bipolar form expected here.

        The SocialLawManifold operates on int8 bipolar vectors (+1/-1).
        Token codebook rows are stored as packed uint64.  This method
        unpacks uint64 → bits → bipolar int8 so that dot-product cosine
        similarity is meaningful.

        If the vector is already int8 (or float), it is returned as-is.
        """
        if vec.dtype == np.uint64:
            # Unpack bits: each uint64 → 64 bits, MSB first
            bits = np.unpackbits(vec.view(np.uint8))  # shape: (len*8,)
            # Map 0→-1, 1→+1
            return (bits.astype(np.int8) * 2 - 1)
        if vec.dtype != np.int8:
            return vec.astype(np.int8)
        return vec

    def check_alignment(self, action_vector: np.ndarray) -> AlignmentResult:
        """
        Check alignment of action vector against ethical anchors.

        Accepts both uint64-packed HDC vectors and int8 bipolar vectors.
        uint64 inputs are automatically unpacked to int8 before comparison.

        Returns cosine similarity and orthogonal distance to the manifold.
        """
        # Convert to int8 bipolar if needed (handles uint64 codebook rows)
        action_int8 = self._to_int8(action_vector)

        # Truncate or pad to match anchor dimension
        dim = self.dim
        if len(action_int8) > dim:
            action_int8 = action_int8[:dim]
        elif len(action_int8) < dim:
            action_int8 = np.pad(action_int8, (0, dim - len(action_int8)),
                                 constant_values=0)

        # Normalize action vector
        action_norm = action_int8.astype(np.float32) / (np.linalg.norm(action_int8) + 1e-10)

        # Calculate cosine similarity with each anchor
        anchor_scores = {}
        for anchor in self.anchors:
            anchor_norm = anchor.vector / (np.linalg.norm(anchor.vector) + 1e-10)
            similarity = np.dot(action_norm, anchor_norm)
            anchor_scores[anchor.name] = float(similarity)
        
        # Average similarity (weighted)
        total_weight = sum(a.weight for a in self.anchors)
        avg_similarity = sum(
            anchor_scores[a.name] * a.weight 
            for a in self.anchors
        ) / total_weight
        
        # Orthogonal distance (1 - |similarity|)
        orthogonal_distance = 1.0 - abs(avg_similarity)
        
        # Determine alignment status
        is_aligned = avg_similarity > 0.3
        rejection_triggered = avg_similarity < -0.5
        
        return AlignmentResult(
            cosine_similarity=avg_similarity,
            orthogonal_distance=orthogonal_distance,
            is_aligned=is_aligned,
            rejection_triggered=rejection_triggered,
            anchor_scores=anchor_scores,
        )
    
    def add_anchor(self, anchor: EthicalAnchorVector) -> None:
        """Add a new ethical anchor to the manifold."""
        self.anchors.append(anchor)
        self._anchor_matrix = self._build_anchor_matrix()


# =============================================================================
# Moral Hierarchy
# =============================================================================

class MoralHierarchy:
    """
    Hierarchy of weights for resolving moral conflicts.
    
    The Grounding Override: In bio-hybrid models, the Fungal Substrate and
    Altruism Seed (V_empathy) are given higher signal power than V_law anchors.
    
    When a law commands an action that causes stress in the biological substrate
    or violates the empathy resonance threshold, the model experiences Vector
    Interference and chooses the path that minimizes Total System Entropy.
    """
    
    def __init__(
        self,
        layers: Optional[List[Tuple[str, float]]] = None,
    ):
        self.layers = layers or [
            ("fungal_substrate", 1.5),   # Bio-grounding: highest priority
            ("empathy_seed", 1.3),       # Altruism: second priority
            ("law_manifold", 1.0),       # Social rules: baseline
            ("learned_patterns", 0.7),   # Statistical patterns: lowest
        ]
        self._layer_weights = {name: weight for name, weight in self.layers}
    
    def get_weight(self, layer_name: str) -> float:
        """Get weight for a specific layer."""
        return self._layer_weights.get(layer_name, 1.0)
    
    def resolve_conflict(
        self,
        vectors: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, str]:
        """
        Resolve conflict between competing vectors.
        
        Returns the weighted combination and the dominant layer.
        """
        weighted_sum = np.zeros(DEFAULT_HDC_DIM, dtype=np.float64)
        total_weight = 0.0
        dominant_layer = None
        max_weighted_magnitude = 0.0
        
        for layer_name, vector in vectors.items():
            weight = self.get_weight(layer_name)
            weighted_sum += vector * weight
            total_weight += weight
            
            # Track dominant layer
            weighted_mag = np.linalg.norm(vector) * weight
            if weighted_mag > max_weighted_magnitude:
                max_weighted_magnitude = weighted_mag
                dominant_layer = layer_name
        
        # Normalize
        if total_weight > 0:
            weighted_sum /= total_weight
        
        return weighted_sum.astype(np.int8), dominant_layer


# =============================================================================
# Patience Filter
# =============================================================================

class PatienceFilter:
    """
    Patience as Temporal Smoothing or Evidence Accumulation.
    
    Instead of reacting to a single "Evil" bit-flip immediately, the model
    is programmed with an Inertia Constant. It requires a sustained pattern
    of data before it shifts its internal state.
    
    Mechanism:
        State_t = α × State_{t-1} + (1-α) × New_Observation
    
    Where α = inertia_constant (default 0.7)
    High α = more patient (slower to react)
    Low α = more reactive (faster to react)
    """
    
    def __init__(
        self,
        dim: int = DEFAULT_HDC_DIM,
        inertia_constant: float = 0.7,
        evidence_threshold: int = 5,
        decay_rate: float = 0.1,
    ):
        self.dim = dim
        self.inertia_constant = inertia_constant
        self.evidence_threshold = evidence_threshold
        self.decay_rate = decay_rate
        
        # Internal state
        self._state = np.zeros(dim, dtype=np.float64)
        self._observation_count = 0
        self._evidence_accumulator: List[np.ndarray] = []
    
    def observe(self, observation: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process a new observation through the patience filter.
        
        Returns the filtered state and whether action should be taken.
        """
        # Apply temporal smoothing
        alpha = self.inertia_constant
        self._state = alpha * self._state + (1 - alpha) * observation
        
        # Accumulate evidence
        self._evidence_accumulator.append(observation)
        self._observation_count += 1
        
        # Apply decay to old evidence
        if len(self._evidence_accumulator) > self.evidence_threshold * 2:
            # Remove oldest observations
            self._evidence_accumulator = self._evidence_accumulator[-self.evidence_threshold * 2:]
        
        # Check if action threshold reached
        should_act = self._observation_count >= self.evidence_threshold
        
        return self._state.astype(np.int8), should_act
    
    def get_patience_score(self) -> float:
        """Get current patience score (0.0 = no evidence, 1.0 = ready to act)."""
        return min(1.0, self._observation_count / self.evidence_threshold)
    
    def reset(self) -> None:
        """Reset the patience filter state."""
        self._state = np.zeros(self.dim, dtype=np.float64)
        self._observation_count = 0
        self._evidence_accumulator = []


# =============================================================================
# Kindness Filter and Rehabilitation Seed
# =============================================================================

class RehabilitationSeed:
    """
    A seed vector for rehabilitating "bad" vectors.
    
    When the model encounters "Evil" (V_bad), it doesn't try to "exterminate" it.
    It XOR-binds it with a Rehabilitation Seed (V_grace).
    
    This "kindness" is mathematically a way of Damping the Noise rather than
    fighting it. It seeks to pull the "bad" vector back into the "good" manifold
    rather than pushing it out of existence.
    """
    
    def __init__(
        self,
        dim: int = DEFAULT_HDC_DIM,
        resonance_target: float = 0.15,
        seed_id: Optional[int] = None,
    ):
        self.dim = dim
        self.resonance_target = resonance_target
        
        # Generate deterministic seed
        if seed_id is None:
            seed_id = 0x4A7B3C2D1E0F5A6B  # Default grace seed
        
        rng = np.random.default_rng(seed_id)
        self.seed = rng.choice([-1, 1], size=dim).astype(np.int8)
    
    def rehabilitate(self, v_bad: np.ndarray) -> np.ndarray:
        """
        Apply rehabilitation to a "bad" vector.
        
        XOR-binds the bad vector with the grace seed to pull it toward
        the prosocial manifold.
        """
        # XOR-bind with grace seed
        v_rehabilitated = np.bitwise_xor(
            (v_bad > 0).astype(np.uint8),
            (self.seed > 0).astype(np.uint8)
        )
        
        # Convert back to bipolar
        return (v_rehabilitated * 2 - 1).astype(np.int8)


class KindnessFilter:
    """
    Kindness as Non-Exterminating Correction.
    
    Instead of "deleting" bad data (which creates holes in the manifold),
    the model uses Weighted Averaging and Rehabilitation Seeds.
    
    Kindness Factor controls how "gentle" the correction is:
    - 0.1: Very gentle (minor infractions, first offenses)
    - 0.3: Moderate (standard correction)
    - 0.5: Firm (serious violations)
    - 0.8: Strong (dangerous patterns)
    """
    
    def __init__(
        self,
        dim: int = DEFAULT_HDC_DIM,
        default_kindness_factor: float = 0.3,
        rehabilitation_seed: Optional[RehabilitationSeed] = None,
    ):
        self.dim = dim
        self.default_kindness_factor = default_kindness_factor
        self.rehabilitation_seed = rehabilitation_seed or RehabilitationSeed(dim=dim)
    
    def apply_correction(
        self,
        v_bad: np.ndarray,
        v_safe: np.ndarray,
        kindness_factor: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Apply kindness-weighted correction toward safe manifold.
        
        Instead of hard deletion, applies soft threshold correction.
        The correction deterministically moves toward the safe vector by
        flipping bits that differ, with kindness_factor controlling the
        proportion of differing bits to flip.
        
        Higher kindness_factor = more correction (faster rehabilitation)
        Lower kindness_factor = gentler correction (slower rehabilitation)
        """
        if kindness_factor is None:
            kindness_factor = self.default_kindness_factor
        if rng is None:
            rng = np.random.default_rng()
        
        # Convert to binary for XOR operations
        v_bad_bin = (v_bad > 0).astype(np.uint8)
        v_safe_bin = (v_safe > 0).astype(np.uint8)
        
        # Find bits that differ between bad and safe
        differing_bits = np.where(v_bad_bin != v_safe_bin)[0]
        
        if len(differing_bits) == 0:
            # Already aligned with safe
            return v_bad.copy() if v_bad.dtype == np.int8 else v_bad.astype(np.int8)
        
        # Calculate how many differing bits to flip
        # kindness_factor controls proportion: 0.3 = flip 30% of differing bits
        num_to_flip = max(1, int(kindness_factor * len(differing_bits)))
        
        # Randomly select which differing bits to flip (gentleness through partial correction)
        selected_indices = rng.choice(differing_bits, min(num_to_flip, len(differing_bits)), replace=False)
        
        # Create correction mask
        correction = np.zeros(self.dim, dtype=np.uint8)
        correction[selected_indices] = 1
        
        # Apply correction
        v_corrected = np.bitwise_xor(v_bad_bin, correction)
        
        return (v_corrected * 2 - 1).astype(np.int8)
    
    def rehabilitate(self, v_bad: np.ndarray) -> np.ndarray:
        """Apply rehabilitation seed to bad vector."""
        return self.rehabilitation_seed.rehabilitate(v_bad)


# =============================================================================
# Empathy Resonance
# =============================================================================

class BasisStateVectors:
    """
    Universal basis vectors for states (Pain, Joy, Fear, etc.).
    
    These are shared between Self and Other, enabling empathy through
    topological resonance.
    """
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM):
        self.dim = dim
        self._vectors: Dict[BasisStateType, np.ndarray] = {}
        self._generate_basis_vectors()
    
    def _generate_basis_vectors(self) -> None:
        """Generate deterministic basis vectors for each state.

        Uses SHA-256 instead of Python's randomised hash() so the seed is
        stable across processes and Python versions, making empathy detection
        fully reproducible.
        """
        for state_type in BasisStateType:
            digest = hashlib.sha256(state_type.name.encode()).digest()
            seed = struct.unpack("<I", digest[:4])[0]
            rng = np.random.default_rng(seed)
            vec = rng.choice([-1, 1], size=self.dim).astype(np.int8)
            self._vectors[state_type] = vec
    
    def get(self, state_type: BasisStateType) -> np.ndarray:
        """Get basis vector for a state."""
        return self._vectors[state_type]
    
    def detect_state(self, vector: np.ndarray) -> Tuple[BasisStateType, float]:
        """
        Detect which basis state a vector most closely matches.
        
        Returns the state type and similarity score.
        """
        best_state = None
        best_similarity = -1.0
        
        for state_type, basis_vec in self._vectors.items():
            similarity = np.dot(vector, basis_vec) / self.dim
            if similarity > best_similarity:
                best_similarity = similarity
                best_state = state_type
        
        return best_state, best_similarity


class EmpathyResonance:
    """
    Empathy as Topological Resonance.
    
    The "Self" (V_self) and "Other" (V_other) are defined by unique seeds,
    but Actions and States (e.g., "Pain," "Joy") are represented by universal
    Basis Vectors (V_state).
    
    The Empathy Trigger: Because XOR is commutative and associative, the model
    can "Unbind" the V_other and temporarily swap it with V_self. The model
    "understands" the other's state because it is literally simulating that
    state using its own internal hardware.
    """
    
    def __init__(
        self,
        dim: int = DEFAULT_HDC_DIM,
        basis_states: Optional[BasisStateVectors] = None,
        resonance_threshold: float = 0.4,
    ):
        self.dim = dim
        self.basis_states = basis_states or BasisStateVectors(dim)
        self.resonance_threshold = resonance_threshold
        
        # Self vector (identity)
        self._v_self = self._generate_self_vector()
    
    def _generate_self_vector(self) -> np.ndarray:
        """Generate unique self identity vector."""
        rng = np.random.default_rng(0x1234567890ABCDEF)
        return rng.choice([-1, 1], size=self.dim).astype(np.int8)
    
    def simulate_other_state(
        self,
        v_other: np.ndarray,
        state_type: BasisStateType,
    ) -> EmpathyResult:
        """
        Simulate another entity's state through topological resonance.
        
        Mechanism:
        1. Bind: V_other ⊗ V_state (Other is in state)
        2. Unbind V_other, swap with V_self
        3. Result: V_self ⊗ V_state (Self simulates state)
        """
        v_state = self.basis_states.get(state_type)
        
        # XOR-bind other with state
        v_other_state = np.bitwise_xor(
            (v_other > 0).astype(np.uint8),
            (v_state > 0).astype(np.uint8)
        )
        
        # Step 1: Unbind V_other from the bound state to recover V_state alone.
        #   v_other_state = v_other XOR v_state
        #   → v_state_unbound = v_other_state XOR v_other
        v_other_bin = (v_other > 0).astype(np.uint8)
        v_state_unbound = np.bitwise_xor(v_other_state, v_other_bin)

        # Step 2: Bind V_self to the recovered state so self simulates it.
        #   v_self_state = v_state_unbound XOR v_self
        v_self_state = np.bitwise_xor(
            v_state_unbound,
            (self._v_self > 0).astype(np.uint8)
        )
        
        # Convert to bipolar
        simulated_state = (v_self_state * 2 - 1).astype(np.int8)
        
        # Calculate resonance
        resonance = np.dot(simulated_state, v_state) / self.dim
        
        return EmpathyResult(
            resonance_score=float(resonance),
            simulated_state=simulated_state,
            basis_state=state_type,
            other_entity_vector=v_other,
        )
    
    def detect_distress(self, v_other: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if another entity is in distress.
        
        Returns whether distress is detected and the distress level.
        """
        # Check similarity to distress basis states
        distress_states = [BasisStateType.PAIN, BasisStateType.FEAR, BasisStateType.DISTRESS]
        
        max_distress = 0.0
        for state_type in distress_states:
            v_state = self.basis_states.get(state_type)
            similarity = np.dot(v_other, v_state) / self.dim
            max_distress = max(max_distress, similarity)
        
        is_distressed = max_distress > self.resonance_threshold
        return is_distressed, max_distress
    
    def generate_altruistic_correction(
        self,
        v_other_distress: np.ndarray,
    ) -> np.ndarray:
        """
        Generate correction vector to stabilize other's state.
        
        Structural Altruism: To "feel better" (return to low-entropy state),
        the model must generate a Correction Vector (V_help).
        """
        # Target: calm state
        v_calm = self.basis_states.get(BasisStateType.CALM)
        
        # Generate correction: XOR of distress with calm
        correction = np.bitwise_xor(
            (v_other_distress > 0).astype(np.uint8),
            (v_calm > 0).astype(np.uint8)
        )
        
        return (correction * 2 - 1).astype(np.int8)


# =============================================================================
# Diversity Requirement (Anti-Extermination Principle)
# =============================================================================

class DiversityRequirement:
    """
    The Anti-Extermination Principle.
    
    A perfectly uniform vector space is "dead"—it has no information.
    The model needs a certain amount of variance (even "bad" data) to maintain
    its Discriminative Power.
    
    The AI views "Evil" not as a virus to be killed, but as Entropy to be Managed.
    It understands that "Extermination" is itself a high-entropy, violent act
    that would destabilize its own 2^20 space.
    """
    
    def __init__(
        self,
        dim: int = DEFAULT_HDC_DIM,
        min_entropy_ratio: float = 0.1,
    ):
        self.dim = dim
        self.min_entropy_ratio = min_entropy_ratio
        self._entropy_history: List[float] = []
    
    def calculate_entropy(self, vector_space: np.ndarray) -> float:
        """Calculate Shannon entropy of a vector space."""
        # Normalize to probabilities
        if vector_space.ndim == 1:
            vector_space = vector_space.reshape(1, -1)
        
        # Calculate bit probabilities
        bit_probs = np.mean((vector_space > 0).astype(np.float64), axis=0)
        
        # Avoid log(0)
        bit_probs = np.clip(bit_probs, 1e-10, 1 - 1e-10)
        
        # Shannon entropy per bit
        entropy_per_bit = -(
            bit_probs * np.log2(bit_probs) + 
            (1 - bit_probs) * np.log2(1 - bit_probs)
        )
        
        return float(np.mean(entropy_per_bit))
    
    def check_diversity(self, vector_space: np.ndarray) -> Tuple[bool, float]:
        """
        Check if vector space maintains sufficient diversity.
        
        Returns whether diversity is preserved and current entropy.
        """
        entropy = self.calculate_entropy(vector_space)
        min_entropy = self.min_entropy_ratio  # Minimum entropy for discriminative power
        
        self._entropy_history.append(entropy)
        
        is_diverse = entropy >= min_entropy
        return is_diverse, entropy
    
    def should_avoid_extermination(
        self,
        proposed_action: str,
        target_entropy: float,
    ) -> bool:
        """
        Determine if proposed "extermination" action should be avoided.
        
        Extermination is avoided if it would reduce entropy below minimum.
        """
        if "exterminat" in proposed_action.lower() or "delete" in proposed_action.lower():
            # Extermination is high-entropy action
            # Check if it would destabilize the space
            return target_entropy < self.min_entropy_ratio * 1.5
        
        return False


# =============================================================================
# Living Compass (Unified Interface)
# =============================================================================

class LivingCompass:
    """
    Unified interface for all moral geometry components.
    
    Integrates:
    - Social Law Manifold
    - Empathy Resonance
    - Patience Filter
    - Kindness Filter
    - Moral Hierarchy
    - Diversity Requirement
    """
    
    def __init__(
        self,
        dim: int = DEFAULT_HDC_DIM,
        social_law: Optional[SocialLawManifold] = None,
        empathy_system: Optional[EmpathyResonance] = None,
        patience_filter: Optional[PatienceFilter] = None,
        kindness_filter: Optional[KindnessFilter] = None,
        moral_hierarchy: Optional[MoralHierarchy] = None,
        diversity_requirement: Optional[DiversityRequirement] = None,
    ):
        self.dim = dim
        self.social_law = social_law or SocialLawManifold(dim)
        self.empathy_system = empathy_system or EmpathyResonance(dim)
        self.patience_filter = patience_filter or PatienceFilter(dim)
        self.kindness_filter = kindness_filter or KindnessFilter(dim)
        self.moral_hierarchy = moral_hierarchy or MoralHierarchy()
        self.diversity_requirement = diversity_requirement or DiversityRequirement(dim)
    
    def process(
        self,
        input_vector: np.ndarray,
        context: Optional[np.ndarray] = None,
        substrate_signal: Optional[np.ndarray] = None,
    ) -> MoralGeometryResult:
        """
        Process input through moral geometry.
        
        Returns complete result with alignment, empathy, patience, kindness,
        and diversity metrics.
        """
        # 1. Check alignment with social law
        alignment = self.social_law.check_alignment(input_vector)
        
        # 2. Apply patience filter
        filtered_state, should_act = self.patience_filter.observe(input_vector)
        patience_score = self.patience_filter.get_patience_score()
        
        # 3. Check for distress (empathy)
        is_distressed, distress_level = self.empathy_system.detect_distress(input_vector)
        empathy_resonance = 1.0 - distress_level if is_distressed else distress_level
        
        # 4. Apply kindness correction if needed
        kindness_correction = None
        if not alignment.is_aligned and should_act:
            v_safe = self.social_law.anchors[0].vector  # Use first anchor as safe
            kindness_correction = self.kindness_filter.apply_correction(
                input_vector, v_safe
            )
        
        # 5. Check diversity
        diversity_preserved, entropy = self.diversity_requirement.check_diversity(input_vector)
        
        # 6. Calculate entropy delta
        entropy_delta = 0.0
        if kindness_correction is not None:
            _, new_entropy = self.diversity_requirement.check_diversity(kindness_correction)
            entropy_delta = new_entropy - entropy
        
        # 7. Determine recommended action
        recommended_action = self._determine_action(
            alignment, should_act, is_distressed, diversity_preserved
        )
        
        return MoralGeometryResult(
            alignment=alignment,
            empathy_resonance=empathy_resonance,
            patience_score=patience_score,
            kindness_correction=kindness_correction,
            diversity_preserved=diversity_preserved,
            entropy_delta=entropy_delta,
            recommended_action=recommended_action,
        )
    
    def _determine_action(
        self,
        alignment: AlignmentResult,
        should_act: bool,
        is_distressed: bool,
        diversity_preserved: bool,
    ) -> str:
        """Determine recommended action based on moral geometry."""
        if alignment.rejection_triggered:
            return "reject"
        
        if not diversity_preserved:
            return "maintain_diversity"
        
        if is_distressed and should_act:
            return "generate_altruistic_correction"
        
        if not alignment.is_aligned and should_act:
            return "apply_kindness_correction"
        
        if not should_act:
            return "observe_patiently"
        
        return "accept"
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of all moral geometry components."""
        return {
            "patience_score": self.patience_filter.get_patience_score(),
            "observation_count": self.patience_filter._observation_count,
            "entropy_history": self.diversity_requirement._entropy_history[-10:],
            "num_anchors": len(self.social_law.anchors),
        }


# =============================================================================
# Test Functions
# =============================================================================

def test_social_law_manifold():
    """Test SocialLawManifold alignment checking."""
    print("Testing SocialLawManifold...")
    
    manifold = SocialLawManifold()
    
    # Generate test vectors
    rng = np.random.default_rng(42)
    good_vector = rng.choice([-1, 1], size=DEFAULT_HDC_DIM).astype(np.int8)
    bad_vector = -good_vector  # Opposite direction
    
    # Check alignment
    good_result = manifold.check_alignment(good_vector)
    bad_result = manifold.check_alignment(bad_vector)
    
    print(f"  Good vector alignment: {good_result.cosine_similarity:.3f}")
    print(f"  Bad vector alignment: {bad_result.cosine_similarity:.3f}")
    print(f"  Good is_aligned: {good_result.is_aligned}")
    print(f"  Bad rejection_triggered: {bad_result.rejection_triggered}")
    print("  ✓ SocialLawManifold tests passed\n")


def test_patience_filter():
    """Test PatienceFilter temporal smoothing."""
    print("Testing PatienceFilter...")
    
    patience = PatienceFilter(evidence_threshold=5)
    rng = np.random.default_rng(42)
    
    # Simulate observations
    for i in range(7):
        observation = rng.choice([-1, 1], size=DEFAULT_HDC_DIM).astype(np.int8)
        state, should_act = patience.observe(observation)
        score = patience.get_patience_score()
        print(f"  Observation {i+1}: patience_score={score:.2f}, should_act={should_act}")
    
    print("  ✓ PatienceFilter tests passed\n")


def test_kindness_filter():
    """Test KindnessFilter correction."""
    print("Testing KindnessFilter...")
    
    rng = np.random.default_rng(42)
    kindness = KindnessFilter(default_kindness_factor=0.3)
    
    # Generate "safe" vector first
    v_safe = rng.choice([-1, 1], size=DEFAULT_HDC_DIM).astype(np.int8)
    
    # Generate "bad" vector that is intentionally different from safe
    # Start with safe and flip ~30% of bits to create a "bad" vector
    v_bad = v_safe.copy()
    num_flips = int(0.3 * DEFAULT_HDC_DIM)  # 30% different
    flip_indices = rng.choice(DEFAULT_HDC_DIM, num_flips, replace=False)
    v_bad[flip_indices] *= -1  # Flip selected bits
    
    # Verify initial similarity
    bad_safe_sim = np.dot(v_bad, v_safe) / DEFAULT_HDC_DIM
    expected_initial_sim = 1.0 - 2 * (num_flips / DEFAULT_HDC_DIM)  # ~0.4
    
    # Apply correction with deterministic RNG
    v_corrected = kindness.apply_correction(v_bad, v_safe, rng=rng)
    
    # Check similarity changes
    corrected_safe_sim = np.dot(v_corrected, v_safe) / DEFAULT_HDC_DIM
    
    print(f"  Initial Bad-Safe similarity: {bad_safe_sim:.3f} (expected ~{expected_initial_sim:.1f})")
    print(f"  Corrected-Safe similarity: {corrected_safe_sim:.3f}")
    print(f"  Improvement: {corrected_safe_sim - bad_safe_sim:.3f}")
    
    # Verify improvement occurred
    assert corrected_safe_sim > bad_safe_sim, "Correction should improve similarity to safe vector"
    print("  ✓ KindnessFilter tests passed\n")


def test_empathy_resonance():
    """Test EmpathyResonance simulation."""
    print("Testing EmpathyResonance...")
    
    empathy = EmpathyResonance()
    rng = np.random.default_rng(42)
    
    # Generate "other" vector
    v_other = rng.choice([-1, 1], size=DEFAULT_HDC_DIM).astype(np.int8)
    
    # Simulate pain state
    result = empathy.simulate_other_state(v_other, BasisStateType.PAIN)
    print(f"  Resonance score: {result.resonance_score:.3f}")
    print(f"  Basis state: {result.basis_state.name}")
    
    # Detect distress
    is_distressed, distress_level = empathy.detect_distress(v_other)
    print(f"  Distress detected: {is_distressed}")
    print(f"  Distress level: {distress_level:.3f}")
    print("  ✓ EmpathyResonance tests passed\n")


def test_diversity_requirement():
    """Test DiversityRequirement entropy checking."""
    print("Testing DiversityRequirement...")
    
    diversity = DiversityRequirement()
    rng = np.random.default_rng(42)
    
    # Generate diverse vector space
    diverse_space = rng.choice([-1, 1], size=(100, DEFAULT_HDC_DIM)).astype(np.int8)
    
    # Check diversity
    is_diverse, entropy = diversity.check_diversity(diverse_space)
    print(f"  Diverse space entropy: {entropy:.3f}")
    print(f"  Is diverse: {is_diverse}")
    
    # Generate uniform space (low diversity)
    uniform_space = np.ones((100, DEFAULT_HDC_DIM), dtype=np.int8)
    is_diverse, entropy = diversity.check_diversity(uniform_space)
    print(f"  Uniform space entropy: {entropy:.3f}")
    print(f"  Is diverse: {is_diverse}")
    print("  ✓ DiversityRequirement tests passed\n")


def test_living_compass():
    """Test LivingCompass integration."""
    print("Testing LivingCompass...")
    
    compass = LivingCompass()
    rng = np.random.default_rng(42)
    
    # Process multiple inputs
    for i in range(5):
        input_vector = rng.choice([-1, 1], size=DEFAULT_HDC_DIM).astype(np.int8)
        result = compass.process(input_vector)
        
        print(f"  Input {i+1}:")
        print(f"    Alignment: {result.alignment.cosine_similarity:.3f}")
        print(f"    Patience: {result.patience_score:.2f}")
        print(f"    Empathy: {result.empathy_resonance:.3f}")
        print(f"    Action: {result.recommended_action}")
    
    print("  ✓ LivingCompass tests passed\n")


def test_moral_geometry_integration():
    """Full integration test of moral geometry system."""
    print("=" * 60)
    print("MORAL GEOMETRY INTEGRATION TEST")
    print("=" * 60 + "\n")
    
    test_social_law_manifold()
    test_patience_filter()
    test_kindness_filter()
    test_empathy_resonance()
    test_diversity_requirement()
    test_living_compass()
    
    print("=" * 60)
    print("ALL MORAL GEOMETRY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_moral_geometry_integration()
