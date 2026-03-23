"""
Unified Deterministic Personality System

A simplified, 100% deterministic personality architecture using XOR binding.
Replaces the complex spiking neuron + mood oscillator system with unified
ternary vector operations.

Key Principles:
1. Personality = XOR-bound trait vectors (NO floating point state)
2. Selection = Integer resonance counting (NO float comparison)
3. Mood = Optional context binding (NO separate state machine)
4. Learning = XOR update (reversible and traceable)

Benefits:
- 100% Deterministic: Same input → same output, guaranteed
- Integer-only arithmetic: No floating-point drift
- Unified representation: Everything is ternary vectors
- Compressible: Personality = 4 seeds (32 bytes) vs. float state
- Reversible: Can unbind any operation to audit decisions
"""

import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json

# Import from sibling modules (correct relative imports)
try:
    from ..Recipes_Seeds.walsh_hadamard_core import WalshHadamardBasis, TernaryHadamardEncoder
    from ..Recipes_Seeds.recipe_storage import IdentityRecipe, RecipeStorage
    from ..HDC_Core_Main.hdc_sparse_core import DEFAULT_HDC_DIM
    HDC_AVAILABLE = True
except ImportError:
    HDC_AVAILABLE = False
    WalshHadamardBasis = None
    TernaryHadamardEncoder = None
    IdentityRecipe = None
    RecipeStorage = None
    DEFAULT_HDC_DIM = 1048576  # Fallback default (2^20 for 8K video)


# =============================================================================
# Core Utility Functions
# =============================================================================

def sha256_ternary(label: str, dim: int, threshold: float = 0.5) -> np.ndarray:
    """
    Generate deterministic ternary vector from SHA256 hash.
    
    This is the foundation of determinism - the same label always
    produces the same ternary vector.
    
    Args:
        label: String label to hash (e.g., "curiosity", "agent_123_focus")
        dim: Vector dimension (should be power of 2, e.g., 32768)
        threshold: Fraction of non-zero values (0.5 = 50% sparse)
    
    Returns:
        Ternary vector with values in {-1, 0, +1}
    """
    # Get deterministic seed from SHA256
    hash_bytes = hashlib.sha256(label.encode('utf-8')).digest()
    seed = int.from_bytes(hash_bytes[:8], 'big')
    # Ensure seed is within numpy's RandomState limit (0 to 2**32 - 1)
    seed = seed % (2**32)
    
    # Use numpy's deterministic RNG
    rng = np.random.RandomState(seed)
    
    # Generate balanced ternary: ~25% -1, ~50% 0, ~25% +1
    # This matches the distribution in walsh_hadamard_core.py
    values = rng.choice([-1, 0, 1], size=dim, p=[0.25, 0.5, 0.25])
    
    return values.astype(np.int8)


def compute_resonance(vec_a: np.ndarray, vec_b: np.ndarray) -> int:
    """
    Compute integer resonance (alignment) between two ternary vectors.
    
    Resonance = count of positions where both vectors have the same non-zero value.
    This is integer-only arithmetic - no floating point.
    
    Args:
        vec_a: First ternary vector
        vec_b: Second ternary vector
    
    Returns:
        Integer resonance count (higher = more aligned)
    """
    # Match: both same non-zero value
    matches = ((vec_a == vec_b) & (vec_a != 0)).sum()
    return int(matches)


def compute_xor_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> int:
    """
    Compute XOR-based similarity (Hamming-style for ternary).
    
    For ternary vectors, we count positions where the XOR (elementwise multiply)
    produces +1 (same sign) vs -1 (opposite sign).
    
    Args:
        vec_a: First ternary vector
        vec_b: Second ternary vector
    
    Returns:
        Integer similarity score (positive = similar, negative = dissimilar)
    """
    # XOR binding via elementwise multiply
    bound = vec_a * vec_b
    
    # Count positive (same sign) vs negative (opposite sign) bindings
    positive = (bound == 1).sum()
    negative = (bound == -1).sum()
    
    return int(positive - negative)


# =============================================================================
# Personality Trait Definitions
# =============================================================================

@dataclass
class PersonalityTraits:
    """
    Standard personality traits as ternary vectors.
    
    Each trait is a direction in the HDC space that influences
    path selection via XOR resonance.
    """
    # Core traits (Big Five inspired)
    curiosity: np.ndarray = None      # Openness to novelty
    caution: np.ndarray = None        # Risk aversion
    creativity: np.ndarray = None     # Divergent thinking
    focus: np.ndarray = None          # Convergent thinking
    sociability: np.ndarray = None    # Social engagement
    assertiveness: np.ndarray = None  # Decision confidence
    
    # Trait names for iteration
    TRAIT_NAMES = ('curiosity', 'caution', 'creativity', 'focus', 'sociability', 'assertiveness')
    
    def __post_init__(self):
        """Initialize any None traits to zero vectors."""
        # This is called after __init__, but we handle None in from_seed
    
    def get_trait(self, name: str) -> Optional[np.ndarray]:
        """Get trait vector by name."""
        return getattr(self, name, None)
    
    def set_trait(self, name: str, vec: np.ndarray):
        """Set trait vector by name."""
        if name in self.TRAIT_NAMES:
            setattr(self, name, vec)
    
    def all_traits(self) -> Dict[str, np.ndarray]:
        """Return all non-None traits as a dictionary."""
        return {name: getattr(self, name) for name in self.TRAIT_NAMES 
                if getattr(self, name) is not None}
    
    def to_seeds(self) -> Dict[str, int]:
        """Convert traits to seed representation (for storage)."""
        # Seeds are derived from the first 8 bytes of each trait's hash
        seeds = {}
        for name, vec in self.all_traits().items():
            # Hash the vector bytes to get a seed
            h = hashlib.sha256(vec.tobytes()).digest()
            seeds[name] = int.from_bytes(h[:8], 'big')
        return seeds


# =============================================================================
# Main Personality Class
# =============================================================================

@dataclass
class DeterministicPersonality:
    """
    Personality as XOR-bound trait vectors - NO floating point state.
    
    All traits are ternary vectors that combine via reversible XOR.
    This enables:
    - 100% deterministic behavior
    - Personality persistence via seeds
    - Context-sensitive selection without mood state
    - Reversible decision auditing
    
    Attributes:
        name: Agent/personality name
        seed: Master seed for reproducibility
        dim: Vector dimension (default 32768)
        traits: PersonalityTraits object containing trait vectors
        trait_weights: Optional integer weights for each trait (default all 1)
        mood_context: Optional mood vector (binds with context for temporary modulation)
    """
    name: str
    seed: int
    dim: int = 32768
    traits: PersonalityTraits = None
    trait_weights: Dict[str, int] = field(default_factory=dict)
    mood_context: Optional[np.ndarray] = None
    _recipe_storage: Optional[Any] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize traits if not provided."""
        if self.traits is None:
            self.traits = PersonalityTraits()
            self._generate_traits()
    
    def _generate_traits(self):
        """Generate all trait vectors from master seed."""
        for trait_name in PersonalityTraits.TRAIT_NAMES:
            # Each trait gets a deterministic vector from seed + trait name
            label = f"{self.seed}_{self.name}_{trait_name}"
            vec = sha256_ternary(label, self.dim)
            self.traits.set_trait(trait_name, vec)
        
        # Set default weights if not provided
        for trait_name in PersonalityTraits.TRAIT_NAMES:
            if trait_name not in self.trait_weights:
                self.trait_weights[trait_name] = 1
    
    @classmethod
    def from_seed(cls, name: str, seed: int, dim: int = DEFAULT_HDC_DIM, 
                  trait_weights: Optional[Dict[str, int]] = None) -> 'DeterministicPersonality':
        """
        Generate personality deterministically from seed.
        
        Args:
            name: Agent/personality name
            seed: Master seed (any integer)
            dim: Vector dimension
            trait_weights: Optional weights for each trait
        
        Returns:
            DeterministicPersonality with generated traits
        """
        return cls(
            name=name,
            seed=seed,
            dim=dim,
            trait_weights=trait_weights or {}
        )
    
    @classmethod
    def from_recipe(cls, recipe_path: str) -> 'DeterministicPersonality':
        """
        Load personality from stored recipe file.
        
        Args:
            recipe_path: Path to personality recipe JSON file
        
        Returns:
            DeterministicPersonality reconstructed from recipe
        """
        with open(recipe_path, 'r') as f:
            data = json.load(f)
        
        personality = cls(
            name=data['name'],
            seed=data['seed'],
            dim=data.get('dim', DEFAULT_HDC_DIM),
            trait_weights=data.get('trait_weights', {})
        )
        
        # Restore mood context if present
        if 'mood_seed' in data:
            personality.mood_context = sha256_ternary(
                f"{data['seed']}_{data['name']}_mood_{data['mood_seed']}", 
                personality.dim
            )
        
        return personality
    
    def to_recipe(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export personality to recipe format.
        
        The recipe contains only the seeds, not the full vectors.
        Vectors are materialized on-demand from seeds.
        
        Args:
            path: Optional path to save recipe JSON
        
        Returns:
            Recipe dictionary
        """
        recipe = {
            'name': self.name,
            'seed': self.seed,
            'dim': self.dim,
            'trait_weights': self.trait_weights,
            'trait_seeds': self.traits.to_seeds(),
            'has_mood': self.mood_context is not None
        }
        
        if self.mood_context is not None:
            # Store mood as a seed
            h = hashlib.sha256(self.mood_context.tobytes()).digest()
            recipe['mood_seed'] = int.from_bytes(h[:8], 'big')
        
        if path:
            with open(path, 'w') as f:
                json.dump(recipe, f, indent=2)
        
        return recipe
    
    # =========================================================================
    # Core Selection Methods
    # =========================================================================
    
    def select_path(self, context_vec: np.ndarray, candidates: List[np.ndarray],
                    use_mood: bool = True) -> int:
        """
        Select path via XOR resonance - NO floating point comparison.
        
        Method:
        1. XOR context with each candidate
        2. Compute resonance with personality traits
        3. Select highest resonance
        
        This is 100% deterministic and reversible.
        
        Args:
            context_vec: Current context/situation vector
            candidates: List of candidate path vectors
            use_mood: Whether to include mood context in selection
        
        Returns:
            Index of selected candidate
        """
        if not candidates:
            return -1
        
        best_idx = 0
        best_resonance = -2**31  # Min int
        
        # Optionally bind mood with context
        effective_context = context_vec
        if use_mood and self.mood_context is not None:
            effective_context = context_vec * self.mood_context
        
        for i, candidate in enumerate(candidates):
            # XOR context with candidate (elementwise multiply for ternary)
            bound = effective_context * candidate
            
            # Compute resonance with each trait
            total_resonance = 0
            for trait_name, trait_vec in self.traits.all_traits().items():
                trait_resonance = compute_resonance(bound, trait_vec)
                weight = self.trait_weights.get(trait_name, 1)
                total_resonance += trait_resonance * weight
            
            if total_resonance > best_resonance:
                best_resonance = total_resonance
                best_idx = i
        
        return best_idx
    
    def rank_paths(self, context_vec: np.ndarray, candidates: List[np.ndarray],
                   use_mood: bool = True) -> List[Tuple[int, int]]:
        """
        Rank all paths by resonance score.
        
        Args:
            context_vec: Current context/situation vector
            candidates: List of candidate path vectors
            use_mood: Whether to include mood context
        
        Returns:
            List of (candidate_index, resonance_score) sorted by score descending
        """
        scores = []
        
        effective_context = context_vec
        if use_mood and self.mood_context is not None:
            effective_context = context_vec * self.mood_context
        
        for i, candidate in enumerate(candidates):
            bound = effective_context * candidate
            
            total_resonance = 0
            for trait_name, trait_vec in self.traits.all_traits().items():
                trait_resonance = compute_resonance(bound, trait_vec)
                weight = self.trait_weights.get(trait_name, 1)
                total_resonance += trait_resonance * weight
            
            scores.append((i, total_resonance))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def select_top_k(self, context_vec: np.ndarray, candidates: List[np.ndarray],
                     k: int = 3, use_mood: bool = True) -> List[int]:
        """
        Select top-k paths by resonance.
        
        Args:
            context_vec: Current context vector
            candidates: List of candidate vectors
            k: Number of top candidates to return
            use_mood: Whether to include mood context
        
        Returns:
            List of indices of top-k candidates
        """
        ranked = self.rank_paths(context_vec, candidates, use_mood)
        return [idx for idx, _ in ranked[:k]]
    
    # =========================================================================
    # Multi-Answer Alignment Scoring (Section 27)
    # =========================================================================
    
    def score_alignment(self, bound_vec: np.ndarray) -> float:
        """
        Compute personality alignment score for a bound vector.
        
        This is used by the MultiAnswerCoordinator to rank valid answers
        by how well they align with the personality's trait vectors.
        
        The score is computed as the weighted sum of resonance values
        between the bound vector and each trait vector.
        
        From FULLINTEGRATION_NEW_ARCHITECTURE.md Section 27:
        "Rank valid answers by personality alignment. Uses the existing
        select_path mechanism extended to multiple answers."
        
        Args:
            bound_vec: Vector bound with context (XOR for uint64, multiply for ternary)
            
        Returns:
            Float alignment score (higher = better alignment)
        """
        total_alignment = 0.0
        
        for trait_name, trait_vec in self.traits.all_traits().items():
            # Compute resonance between bound vector and trait
            resonance = compute_resonance(bound_vec, trait_vec)
            weight = self.trait_weights.get(trait_name, 1)
            total_alignment += resonance * weight
        
        return float(total_alignment)
    
    def score_alignment_detailed(self, bound_vec: np.ndarray) -> Dict[str, float]:
        """
        Compute detailed personality alignment with per-trait breakdown.
        
        This is useful for explaining why a particular answer was ranked
        higher than others.
        
        Args:
            bound_vec: Vector bound with context
            
        Returns:
            Dictionary with 'total' and individual trait scores
        """
        result = {'total': 0.0}
        
        for trait_name, trait_vec in self.traits.all_traits().items():
            resonance = compute_resonance(bound_vec, trait_vec)
            weight = self.trait_weights.get(trait_name, 1)
            weighted = resonance * weight
            result[trait_name] = weighted
            result['total'] += weighted
        
        return result
    
    def rank_answers(self, answers: List[np.ndarray],
                     context_vec: np.ndarray,
                     use_mood: bool = True) -> List[Tuple[int, float]]:
        """
        Rank multiple answer vectors by personality alignment.
        
        This is the multi-answer extension of select_path.
        
        Args:
            answers: List of answer vectors (already validated as correct)
            context_vec: Context vector for alignment
            use_mood: Whether to include mood context
        
        Returns:
            List of (answer_index, alignment_score) sorted by score descending
        """
        scores = []
        
        effective_context = context_vec
        if use_mood and self.mood_context is not None:
            effective_context = context_vec * self.mood_context
        
        for i, answer in enumerate(answers):
            # XOR bind answer with context (elementwise multiply for ternary)
            bound = effective_context * answer
            
            # Compute alignment score
            alignment = self.score_alignment(bound)
            scores.append((i, alignment))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def get_trait_preference(self, trait_name: str) -> int:
        """
        Get the weight for a specific trait.
        
        Args:
            trait_name: Name of the trait
            
        Returns:
            Integer weight for the trait
        """
        return self.trait_weights.get(trait_name, 1)
    
    def set_trait_preference(self, trait_name: str, weight: int):
        """
        Set the weight for a specific trait.
        
        This allows dynamic adjustment of personality preferences.
        
        Args:
            trait_name: Name of the trait
            weight: New weight value (integer)
        """
        if trait_name in PersonalityTraits.TRAIT_NAMES:
            self.trait_weights[trait_name] = weight
    
    # =========================================================================
    # Mood System (Optional)
    # =========================================================================
    
    def set_mood(self, mood_label: str):
        """
        Set mood from a label (deterministic).
        
        Mood is represented as a ternary vector that binds with context
        during selection, temporarily modulating behavior.
        
        Args:
            mood_label: Mood description (e.g., "happy", "cautious", "curious")
        """
        label = f"{self.seed}_{self.name}_mood_{mood_label}"
        self.mood_context = sha256_ternary(label, self.dim)
    
    def set_mood_from_context(self, context_vec: np.ndarray, intensity: float = 0.5):
        """
        Set mood derived from current context.
        
        This allows the personality to be influenced by recent events
        while remaining deterministic.
        
        Args:
            context_vec: Context to derive mood from
            intensity: How much to blend (0.0 = no mood, 1.0 = full context)
        """
        # For determinism, we threshold the context to create mood
        # intensity controls the sparsity
        threshold = 1.0 - intensity  # Higher intensity = more non-zero values
        
        mood = np.zeros(self.dim, dtype=np.int8)
        if intensity > 0:
            # Use top fraction of context values
            flat = context_vec.flatten()
            k = max(1, int(self.dim * intensity * 0.5))
            
            # Get indices of largest absolute values
            indices = np.argsort(np.abs(flat))[-k:]
            mood[indices] = np.sign(flat[indices]).astype(np.int8)
        
        self.mood_context = mood
    
    def clear_mood(self):
        """Clear mood context (return to baseline personality)."""
        self.mood_context = None
    
    def blend_mood(self, other_mood: np.ndarray, alpha: int = 1):
        """
        Blend current mood with another mood vector.
        
        Uses integer arithmetic: new_mood = mood + alpha * other
        then snaps back to ternary.
        
        Args:
            other_mood: Mood vector to blend in
            alpha: Blend factor (integer, can be negative)
        """
        if self.mood_context is None:
            self.mood_context = other_mood.copy()
            return
        
        # Integer accumulation
        blended = self.mood_context.astype(np.int32) + alpha * other_mood.astype(np.int32)
        
        # Snap back to ternary
        result = np.zeros(self.dim, dtype=np.int8)
        result[blended > 0] = 1
        result[blended < 0] = -1
        
        self.mood_context = result
    
    # =========================================================================
    # Learning & Adaptation
    # =========================================================================
    
    def reinforce_trait(self, trait_name: str, pattern: np.ndarray, strength: int = 1):
        """
        Reinforce a trait by XOR-binding with a successful pattern.
        
        This is how the personality "learns" - by updating trait vectors
        based on experience. The update is reversible.
        
        Args:
            trait_name: Name of trait to reinforce
            pattern: Pattern vector to bind with trait
            strength: Update strength (integer, can be negative for suppression)
        """
        trait_vec = self.traits.get_trait(trait_name)
        if trait_vec is None:
            return
        
        # XOR bind trait with pattern
        bound = trait_vec * pattern
        
        # Integer accumulation
        updated = trait_vec.astype(np.int32) + strength * bound.astype(np.int32)
        
        # Snap back to ternary
        result = np.zeros(self.dim, dtype=np.int8)
        result[updated > 0] = 1
        result[updated < 0] = -1
        
        self.traits.set_trait(trait_name, result)
    
    def learn_from_outcome(self, context_vec: np.ndarray, selected_vec: np.ndarray,
                           success: bool, feedback_vec: Optional[np.ndarray] = None):
        """
        Learn from an action outcome.
        
        Success: Reinforce traits that aligned with the successful action
        Failure: Suppress traits that led to the failed action
        
        Args:
            context_vec: Context when action was taken
            selected_vec: Action that was selected
            success: Whether the outcome was positive
            feedback_vec: Optional feedback signal vector
        """
        # Determine which traits were most involved in the selection
        bound = context_vec * selected_vec
        
        trait_involvement = {}
        for trait_name, trait_vec in self.traits.all_traits().items():
            trait_involvement[trait_name] = compute_resonance(bound, trait_vec)
        
        # Find most involved trait
        most_involved = max(trait_involvement.items(), key=lambda x: x[1])
        
        if success:
            # Reinforce the trait that led to success
            self.reinforce_trait(most_involved[0], selected_vec, strength=1)
        else:
            # Suppress the trait that led to failure
            self.reinforce_trait(most_involved[0], selected_vec, strength=-1)
        
        # If feedback provided, also update based on feedback
        if feedback_vec is not None:
            strength = 1 if success else -1
            self.reinforce_trait(most_involved[0], feedback_vec, strength=strength)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_trait_signature(self) -> str:
        """
        Get a short signature hash of the personality traits.
        
        Useful for debugging and verification.
        """
        all_vecs = np.concatenate([
            self.traits.get_trait(name)
            for name in PersonalityTraits.TRAIT_NAMES
            if self.traits.get_trait(name) is not None
        ])
        return hashlib.sha256(all_vecs.tobytes()).hexdigest()[:16]
    
    # =========================================================================
    # Advanced Features: Shared Knowledge, Evolution, Audit Trail
    # =========================================================================
    
    def create_snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of current personality state.
        
        Useful for:
        - Audit trails (replay with different personalities)
        - Reversible adaptation (undo changes)
        - Personality evolution tracking
        
        Returns:
            Snapshot dictionary with trait vectors and weights
        """
        return {
            'name': self.name,
            'seed': self.seed,
            'dim': self.dim,
            'trait_weights': dict(self.trait_weights),
            'trait_vectors': {
                name: self.traits.get_trait(name).copy()
                for name in PersonalityTraits.TRAIT_NAMES
                if self.traits.get_trait(name) is not None
            },
            'mood_context': self.mood_context.copy() if self.mood_context is not None else None,
            'signature': self.get_trait_signature()
        }
    
    def restore_snapshot(self, snapshot: Dict[str, Any]):
        """
        Restore personality from a snapshot.
        
        Enables reversible adaptation - can undo personality changes.
        
        Args:
            snapshot: Snapshot dictionary from create_snapshot()
        """
        self.trait_weights = dict(snapshot['trait_weights'])
        
        for name, vec in snapshot['trait_vectors'].items():
            self.traits.set_trait(name, vec.copy())
        
        if snapshot['mood_context'] is not None:
            self.mood_context = snapshot['mood_context'].copy()
        else:
            self.mood_context = None
    
    def replay_decision(self, context_vec: np.ndarray, candidates: List[np.ndarray],
                        other_personality: 'DeterministicPersonality') -> Dict[str, Any]:
        """
        Replay a decision with a different personality to see "what if".
        
        This is the core audit trail feature - compare how different
        personalities would handle the same situation.
        
        Args:
            context_vec: Context vector
            candidates: List of candidate vectors
            other_personality: Another personality to compare with
        
        Returns:
            Comparison of selections and rankings
        """
        # Get selections from both personalities
        my_selection = self.select_path(context_vec, candidates)
        other_selection = other_personality.select_path(context_vec, candidates)
        
        # Get rankings
        my_ranking = self.rank_paths(context_vec, candidates)
        other_ranking = other_personality.rank_paths(context_vec, candidates)
        
        # Get explanations
        my_explanation = self.explain_selection(context_vec, candidates[my_selection])
        other_explanation = other_personality.explain_selection(context_vec, candidates[other_selection])
        
        return {
            'my_selection': my_selection,
            'other_selection': other_selection,
            'selections_match': my_selection == other_selection,
            'my_ranking': my_ranking,
            'other_ranking': other_ranking,
            'my_explanation': my_explanation,
            'other_explanation': other_explanation,
            'my_signature': self.get_trait_signature(),
            'other_signature': other_personality.get_trait_signature()
        }
    
    def evolve_trait_weights(self, trait_deltas: Dict[str, int],
                             min_weight: int = 0, max_weight: int = 10):
        """
        Evolve personality by adjusting trait weights.
        
        This is "maturation" - changing how much each trait influences
        decisions without changing the trait vectors themselves.
        
        Args:
            trait_deltas: Dict of trait_name -> delta (can be negative)
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        for trait_name, delta in trait_deltas.items():
            if trait_name in self.trait_weights:
                new_weight = self.trait_weights[trait_name] + delta
                self.trait_weights[trait_name] = max(min_weight, min(max_weight, new_weight))
    
    @classmethod
    def create_shared_knowledge_agents(cls, names: List[str], shared_seed: int,
                                        agent_seeds: List[int], dim: int = 32768) -> List['DeterministicPersonality']:
        """
        Create multiple agents that share the same knowledge base (seed)
        but have different personalities.
        
        This enables "Same knowledge, different agents" pattern.
        
        Args:
            names: List of agent names
            shared_seed: Shared seed for knowledge base (affects trait generation)
            agent_seeds: Individual seeds for each agent (affects personality differences)
            dim: Vector dimension
        
        Returns:
            List of personalities with shared knowledge but different behaviors
        """
        personalities = []
        
        for i, (name, agent_seed) in enumerate(zip(names, agent_seeds)):
            # Combine shared seed with agent seed for unique but related personality
            combined_seed = shared_seed ^ agent_seed
            
            # Create personality with different trait weights based on agent_seed
            rng = np.random.RandomState(agent_seed)
            trait_weights = {
                trait: int(rng.randint(1, 4))  # Random weight 1-3
                for trait in PersonalityTraits.TRAIT_NAMES
            }
            
            personality = cls.from_seed(
                name=name,
                seed=combined_seed,
                dim=dim,
                trait_weights=trait_weights
            )
            personalities.append(personality)
        
        return personalities
    
    def explain_selection(self, context_vec: np.ndarray, candidate: np.ndarray,
                          use_mood: bool = True) -> Dict[str, Any]:
        """
        Explain why a candidate would be selected/rejected.
        
        Returns detailed breakdown of resonance scores per trait.
        
        Args:
            context_vec: Context vector
            candidate: Candidate vector to analyze
            use_mood: Whether to include mood in analysis
        
        Returns:
            Dictionary with trait-by-trait breakdown
        """
        effective_context = context_vec
        if use_mood and self.mood_context is not None:
            effective_context = context_vec * self.mood_context
        
        bound = effective_context * candidate
        
        explanation = {
            'total_resonance': 0,
            'trait_resonances': {},
            'mood_active': use_mood and self.mood_context is not None
        }
        
        for trait_name, trait_vec in self.traits.all_traits().items():
            resonance = compute_resonance(bound, trait_vec)
            weight = self.trait_weights.get(trait_name, 1)
            weighted = resonance * weight
            
            explanation['trait_resonances'][trait_name] = {
                'raw': resonance,
                'weight': weight,
                'weighted': weighted
            }
            explanation['total_resonance'] += weighted
        
        return explanation


# =============================================================================
# Personality Registry (for managing multiple personalities)
# =============================================================================

class PersonalityRegistry:
    """
    Registry for managing multiple deterministic personalities.
    
    Enables:
    - Creating agents with unique personalities
    - Loading/saving personality libraries
    - Cross-agent comparison and analysis
    """
    
    def __init__(self, dim: int = 32768, storage_path: Optional[str] = None):
        """
        Initialize personality registry.
        
        Args:
            dim: Vector dimension for all personalities
            storage_path: Optional path for personality storage
        """
        self.dim = dim
        self.storage_path = Path(storage_path) if storage_path else None
        self.personalities: Dict[str, DeterministicPersonality] = {}
        
        if self.storage_path and self.storage_path.exists():
            self._load_all()
    
    def _load_all(self):
        """Load all personalities from storage path."""
        if not self.storage_path:
            return
        
        for recipe_file in self.storage_path.glob('*.json'):
            try:
                personality = DeterministicPersonality.from_recipe(str(recipe_file))
                self.personalities[personality.name] = personality
            except Exception as e:
                print(f"Warning: Failed to load {recipe_file}: {e}")
    
    def create(self, name: str, seed: Optional[int] = None,
               trait_weights: Optional[Dict[str, int]] = None) -> DeterministicPersonality:
        """
        Create a new personality.
        
        Args:
            name: Personality name
            seed: Optional seed (auto-generated from name if not provided)
            trait_weights: Optional trait weights
        
        Returns:
            Created personality
        """
        if seed is None:
            # Generate deterministic seed from name
            h = hashlib.sha256(name.encode()).digest()
            seed = int.from_bytes(h[:8], 'big')
        
        personality = DeterministicPersonality.from_seed(
            name=name,
            seed=seed,
            dim=self.dim,
            trait_weights=trait_weights
        )
        
        self.personalities[name] = personality
        return personality
    
    def get(self, name: str) -> Optional[DeterministicPersonality]:
        """Get personality by name."""
        return self.personalities.get(name)
    
    def save(self, name: Optional[str] = None):
        """
        Save personality(ies) to storage.
        
        Args:
            name: Specific personality to save, or None for all
        """
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        if name:
            personality = self.personalities.get(name)
            if personality:
                personality.to_recipe(str(self.storage_path / f"{name}.json"))
        else:
            for pname, personality in self.personalities.items():
                personality.to_recipe(str(self.storage_path / f"{pname}.json"))
    
    def compare(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """
        Compare two personalities.
        
        Args:
            name_a: First personality name
            name_b: Second personality name
        
        Returns:
            Comparison results
        """
        p_a = self.personalities.get(name_a)
        p_b = self.personalities.get(name_b)
        
        if not p_a or not p_b:
            return {'error': 'Personality not found'}
        
        comparison = {
            'trait_similarities': {},
            'overall_similarity': 0
        }
        
        total_sim = 0
        count = 0
        
        for trait_name in PersonalityTraits.TRAIT_NAMES:
            vec_a = p_a.traits.get_trait(trait_name)
            vec_b = p_b.traits.get_trait(trait_name)
            
            if vec_a is not None and vec_b is not None:
                sim = compute_xor_similarity(vec_a, vec_b)
                comparison['trait_similarities'][trait_name] = sim
                total_sim += sim
                count += 1
        
        comparison['overall_similarity'] = total_sim // max(count, 1)
        return comparison


# =============================================================================
# Integration with Existing HDC Systems
# =============================================================================

def create_personality_from_hdc(name: str, seed: int, 
                                 encoder: 'TernaryHadamardEncoder') -> DeterministicPersonality:
    """
    Create personality using existing HDC encoder.
    
    This integrates with the existing Walsh-Hadamard infrastructure.
    
    Args:
        name: Personality name
        seed: Master seed
        encoder: TernaryHadamardEncoder instance
    
    Returns:
        DeterministicPersonality with traits from encoder
    """
    personality = DeterministicPersonality.from_seed(name, seed, encoder.dim)
    
    # Optionally use encoder's basis for trait generation
    # This ensures compatibility with the existing HDC system
    for trait_name in PersonalityTraits.TRAIT_NAMES:
        label = f"{seed}_{name}_{trait_name}"
        hash_bytes = hashlib.sha256(label.encode('utf-8')).digest()
        basis_idx = int.from_bytes(hash_bytes[:8], 'big') % encoder.dim
        
        # Get basis vector from encoder
        trait_vec = encoder.basis.get_row(basis_idx, packed=False)
        personality.traits.set_trait(trait_name, trait_vec)
    
    return personality


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_personality(name: str, seed: int = 0, dim: int = 32768) -> DeterministicPersonality:
    """
    Quick creation of a personality with default settings.
    
    Args:
        name: Personality name
        seed: Master seed (default 0)
        dim: Vector dimension
    
    Returns:
        DeterministicPersonality instance
    """
    return DeterministicPersonality.from_seed(name, seed, dim)


def personality_from_description(description: str, dim: int = 32768) -> DeterministicPersonality:
    """
    Create personality from a text description.
    
    The description is hashed to create a deterministic seed,
    so the same description always produces the same personality.
    
    Args:
        description: Text description of personality
        dim: Vector dimension
    
    Returns:
        DeterministicPersonality instance
    """
    # Create seed from description hash
    h = hashlib.sha256(description.encode()).digest()
    seed = int.from_bytes(h[:8], 'big')
    
    # Extract name from first word or use hash
    name = description.split()[0] if description else f"agent_{seed % 10000}"
    
    return DeterministicPersonality.from_seed(name, seed, dim)
