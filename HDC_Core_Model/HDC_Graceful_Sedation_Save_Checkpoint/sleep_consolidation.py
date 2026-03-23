"""
Symbolic Recursive Sleep Consolidation System (v2.4.0) - DXPS Refactored

This module implements a SYMBOLIC RECURSION-BASED sleep consolidation system
that stores the latent reasoning state `z` as a RECIPE (list of operations)
instead of a vector, enabling effectively INFINITE recursion depth.

DXPS Refactor:
    - All `bundle()` operations have been replaced with sequential `bind()` (XOR).
    - This ensures all operations are lossless, deterministic, and algebraically reversible,
      aligning with the "Deterministic XOR Program Synthesizer" architecture.
    - The system no longer uses statistical averaging, enabling the discovery of
      bit-perfect, provable transformation recipes.

Key Innovation - Symbolic Recipes vs Vector Saturation:
    - Problem: After ~50 bundle operations, HDC vectors tend toward 50% density
      (maximum entropy), losing discriminative power
    - Solution: Store `z` as a symbolic recipe of operations that can be
      reconstructed on-demand (lazy evaluation) using pure XOR logic.
    - Benefit: Recipe can grow indefinitely without saturation or signal loss!
"""

import numpy as np
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from ..HDC_Core_Main.hdc_sparse_core import SparseBinaryHDC
from ..Recipes_Seeds.seed_recipe_storage import (
    GenerativePatternStorage,
    PatternRecipe,
    RecipeOperation,
    RecipeOperationType,
)


class SleepPhase(Enum):
    """Phases of the sleep cycle."""
    AWAKE = auto()           # Not sleeping
    SCANNING = auto()        # Finding similar recipes
    RECOMBINING = auto()     # Creating new recipe combinations
    VERIFYING = auto()       # Testing candidates
    CONSOLIDATING = auto()   # Storing verified patterns
    COMPLETE = auto()        # Sleep cycle finished


@dataclass
class SleepConfig:
    """
    Configuration for seed-based sleep consolidation.
    
    Anti-Dilution Settings:
        verification_threshold: Minimum accuracy on training data (default 0.8)
        max_candidates_to_store: Cap on new patterns per cycle (prevents flooding)
        require_improvement: Only store if better than existing (default True)
    
    Search Settings:
        dream_cycles: Number of recombination cycles
        candidates_per_cycle: Candidates to generate per cycle
        max_recipe_depth: Maximum composition depth (prevents explosion)
    
    Seed Search Settings:
        seed_similarity_threshold: Minimum seed similarity for recombination
        use_primitives: Include DSL primitives in recombination
        use_existing_recipes: Include stored recipes in recombination
    """
    # Anti-dilution settings
    verification_threshold: float = 0.8
    max_candidates_to_store: int = 10
    require_improvement: bool = True
    
    # Search settings
    dream_cycles: int = 3
    candidates_per_cycle: int = 5
    max_recipe_depth: int = 3
    max_sequence_length: int = 4
    
    # Seed search settings
    seed_similarity_threshold: float = 0.5
    use_primitives: bool = True
    use_existing_recipes: bool = True
    
    # Timing
    max_cycle_time_seconds: float = 10.0
    
    def __post_init__(self):
        # Validate anti-dilution settings
        if self.verification_threshold < 0.5:
            raise ValueError("verification_threshold must be >= 0.5 to prevent dilution")
        if self.max_candidates_to_store < 1:
            raise ValueError("max_candidates_to_store must be >= 1")


# =============================================================================
# SYMBOLIC RECURSION MODE (v2.4.0) - Infinite Depth via Recipe-Based z Storage
# =============================================================================

# Depth level configurations - how deep the symbolic recursion can go
SYMBOLIC_DEPTH_LEVELS = {
    'shallow': {'n_sup': 16, 'n_recursions': 10, 't_rounds': 3},    # ~480 ops, ~1x vector
    'medium': {'n_sup': 64, 'n_recursions': 20, 't_rounds': 5},     # ~6,400 ops, ~12x
    'deep': {'n_sup': 256, 'n_recursions': 50, 't_rounds': 5},      # ~64,000 ops, ~125x
    'very_deep': {'n_sup': 1024, 'n_recursions': 100, 't_rounds': 5},  # ~512,000 ops, ~1000x
    'infinite': {'n_sup': 10000, 'n_recursions': 100, 't_rounds': 10},  # ~10M ops, ~20,000x
}


@dataclass
class SymbolicSleepRecipe:
    """
    Stores the latent reasoning state `z` as a RECIPE instead of a vector.
    
    This is the KEY INNOVATION that enables infinite recursion depth:
    - Instead of materializing z as a vector that saturates after ~50 ops
    - We store z as a list of symbolic operations
    - Reconstruction happens lazily on-demand
    
    Memory: ~50 bytes per operation vs ~4,096 bytes for full vector
    
    Operations are stored as (op_name, parameter) tuples:
        - ("bind", "context_key"): Bind z with a named context vector
        - ("permute", 137): Circular bit shift by 137
        - ("bundle", ["v1", "v2"]): Bundle with named vectors
        - ("seed", 12345): Reset z to vector from seed
        - ("xz_bind", "x_key"): Bind x with z (input relation)
        - ("yz_bind", "y_key"): Bind y with z (answer relation)
    """
    base_seed: int
    operations: List[Tuple[str, Any]] = field(default_factory=list)
    depth_level: str = 'medium'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_operation(self, op_name: str, param: Any) -> None:
        """Add an operation to the recipe (no vector computation!)."""
        self.operations.append((op_name, param))
    
    def add_bind(self, context_key: str) -> None:
        """Add a bind operation with a named context."""
        self.operations.append(("bind", context_key))
    
    def add_permute(self, shift: int) -> None:
        """Add a permutation (circular bit shift)."""
        self.operations.append(("permute", shift))
    
    def add_bundle(self, vector_keys: List[str]) -> None:
        """
        DXPS NOTE: 'bundle' is now implemented as sequential bind (XOR) to be lossless.
        """
        self.operations.append(("bundle", vector_keys))
    
    def add_xz_bind(self, x_key: str) -> None:
        """Add x-z relationship binding (input relates to reasoning)."""
        self.operations.append(("xz_bind", x_key))
    
    def add_yz_bind(self, y_key: str) -> None:
        """Add y-z relationship binding (answer relates to reasoning)."""
        self.operations.append(("yz_bind", y_key))
    
    def add_recursion_step(self, step: int, x_key: str, y_key: str) -> None:
        """
        Add a complete TRM-style latent recursion step.
        
        This encodes: z = bind(bind(bind(x,z), bind(y,z)), permute(z, shift))
        As symbolic operations instead of vector computation.
        """
        shift = (step + 1) * 128  # Position-dependent permutation
        self.operations.append(("xz_bind", x_key))
        self.operations.append(("yz_bind", y_key))
        self.operations.append(("permute", shift))
        self.operations.append(("bundle_step", step))  # Marker for reconstruction
    
    def reconstruct(
        self,
        hdc: SparseBinaryHDC,
        context_vectors: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Reconstruct z vector from recipe ON DEMAND (lazy evaluation).
        
        This is called only when we need the actual vector for similarity
        comparison during verification. The recipe can grow indefinitely
        without saturation because this reconstruction is deferred.
        
        Args:
            hdc: The HDC instance for operations
            context_vectors: Named vectors to use in operations
        
        Returns:
            The reconstructed z vector
        """
        z = hdc.from_seed(self.base_seed)
        
        # Temporary storage for intermediate results
        step_results = []
        
        for op_name, param in self.operations:
            if op_name == "bind":
                if param in context_vectors:
                    z = hdc.bind(z, context_vectors[param])
            elif op_name == "permute":
                z = hdc.permute(z, param)
            elif op_name == "bundle":
                # DXPS REFACTOR: Replace bundle with sequential bind (XOR) to be lossless.
                # This prevents signal dilution and aligns with the algebraic proof model.
                vecs_to_combine = [z]
                for key in param:
                    if key in context_vectors:
                        vecs_to_combine.append(context_vectors[key])
                # Start with the first vector and sequentially bind the rest
                if vecs_to_combine:
                    result_vec = vecs_to_combine[0]
                    for i in range(1, len(vecs_to_combine)):
                        result_vec = hdc.bind(result_vec, vecs_to_combine[i])
                    z = result_vec
            elif op_name == "seed":
                z = hdc.from_seed(param)
            elif op_name == "xz_bind":
                if param in context_vectors:
                    xz = hdc.bind(context_vectors[param], z)
                    step_results.append(("xz", xz))
            elif op_name == "yz_bind":
                if param in context_vectors:
                    yz = hdc.bind(context_vectors[param], z)
                    step_results.append(("yz", yz))
            elif op_name == "include":
                # RECURSIVE REUSE (The Efficiency Fix)
                # Param is the ID of another recipe (e.g., "BLOCK::EYES_STRUCTURE")
                
                # 1. Fetch the shared block recipe
                # We assume we have access to the storage/registry here
                # TODO: BE SURE THAT YOU DOUBLE CHECK THAT THIS LOGIC IS CORRECT.
                shared_recipe = self.fetch_recipe(param) 
                
                # 2. Reconstruct that block
                # We don't copy the data; we just run its logic on our current vector
                z = shared_recipe.reconstruct_on_top_of(z, hdc, context_vectors)
            elif op_name == "bundle_step":
                # DXPS REFACTOR: Replace bundle with sequential bind (XOR).
                # The original `hdc.bundle([xz, yz, z])` is a lossy operation.
                # The new formulation `bind(bind(xz, yz), z)` is a pure algebraic
                # operation that is fully reversible and preserves signal fidelity.
                if len(step_results) >= 2:
                    xz = step_results[-2][1] if step_results[-2][0] == "xz" else z
                    yz = step_results[-1][1] if step_results[-1][0] == "yz" else z
                    # Sequential XOR instead of bundling
                    z = hdc.bind(hdc.bind(xz, yz), z)
        
        return z
    
    def get_operation_count(self) -> int:
        """Get total number of operations in recipe."""
        return len(self.operations)
    
    def get_memory_estimate_bytes(self) -> int:
        """Estimate memory usage in bytes (~50 bytes per operation)."""
        return 50 * len(self.operations) + 100  # +100 for metadata
    
    def copy(self) -> 'SymbolicSleepRecipe':
        """Create a copy of this recipe."""
        new_recipe = SymbolicSleepRecipe(
            base_seed=self.base_seed,
            operations=list(self.operations),
            depth_level=self.depth_level,
            metadata=dict(self.metadata)
        )
        return new_recipe
    
    def to_dict(self) -> dict:
        """Serialize recipe to dictionary."""
        return {
            'base_seed': self.base_seed,
            'operations': self.operations,
            'depth_level': self.depth_level,
            'metadata': self.metadata,
            'operation_count': len(self.operations),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SymbolicSleepRecipe':
        """Deserialize recipe from dictionary."""
        return cls(
            base_seed=data['base_seed'],
            operations=data.get('operations', []),
            depth_level=data.get('depth_level', 'medium'),
            metadata=data.get('metadata', {})
        )


@dataclass
class SymbolicSleepConfig:
    """
    Configuration for symbolic recursion-based sleep consolidation.
    """
    # Symbolic recursion settings
    depth_level: str = 'medium'
    n_sup: Optional[int] = None  # Override from depth_level
    n_recursions: Optional[int] = None
    t_rounds: Optional[int] = None
    
    # Anti-dilution settings
    verification_threshold: float = 0.8
    max_candidates_to_store: int = 10
    require_improvement: bool = True
    
    # Search settings
    dream_cycles: int = 3
    candidates_per_cycle: int = 5
    max_recipe_depth: int = 3
    max_sequence_length: int = 4
    
    # Seed search settings
    seed_similarity_threshold: float = 0.5
    use_primitives: bool = True
    use_existing_recipes: bool = True
    
    # Timing
    max_cycle_time_seconds: float = 30.0  # Higher for symbolic recursion
    
    # Enable symbolic mode (can disable for fallback)
    symbolic_enabled: bool = True
    
    def __post_init__(self):
        # Apply depth level preset if not overridden
        if self.depth_level in SYMBOLIC_DEPTH_LEVELS:
            preset = SYMBOLIC_DEPTH_LEVELS[self.depth_level]
            if self.n_sup is None:
                self.n_sup = preset['n_sup']
            if self.n_recursions is None:
                self.n_recursions = preset['n_recursions']
            if self.t_rounds is None:
                self.t_rounds = preset['t_rounds']
        else:
            # Default to medium
            self.n_sup = self.n_sup or 64
            self.n_recursions = self.n_recursions or 20
            self.t_rounds = self.t_rounds or 5
        
        # Validate
        if self.verification_threshold < 0.5:
            raise ValueError("verification_threshold must be >= 0.5 to prevent dilution")
    
    def get_max_operations(self) -> int:
        """Calculate maximum possible operations for this config."""
        return self.n_sup * self.n_recursions * self.t_rounds * 4  # ~4 ops per recursion


@dataclass
class SymbolicDreamCandidate:
    """
    A candidate solution generated during symbolic sleep.
    """
    recipe: SymbolicSleepRecipe
    source_seeds: List[int]
    composition_type: str
    verification_score: Optional[float] = None
    verified: bool = False
    reconstruction_count: int = 0
    
    _cached_vector: Optional[np.ndarray] = field(default=None, repr=False)
    
    def get_vector(
        self,
        hdc: SparseBinaryHDC,
        context_vectors: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Get the vector (reconstructing if needed)."""
        if self._cached_vector is None:
            self._cached_vector = self.recipe.reconstruct(hdc, context_vectors)
            self.reconstruction_count += 1
        return self._cached_vector
    
    def clear_cache(self) -> None:
        """Clear cached vector to save memory."""
        self._cached_vector = None
    
    def to_dict(self) -> dict:
        return {
            'recipe': self.recipe.to_dict(),
            'source_seeds': self.source_seeds,
            'composition_type': self.composition_type,
            'verification_score': self.verification_score,
            'verified': self.verified,
            'operation_count': self.recipe.get_operation_count(),
        }


@dataclass
class DreamCandidate:
    """A candidate solution generated during sleep."""
    recipe: PatternRecipe
    vector: np.ndarray
    source_seeds: List[int]
    composition_type: str
    verification_score: Optional[float] = None
    verified: bool = False
    
    def to_dict(self) -> dict:
        return {
            'recipe_id': self.recipe.id,
            'source_seeds': self.source_seeds,
            'composition_type': self.composition_type,
            'verification_score': self.verification_score,
            'verified': self.verified,
        }


@dataclass
class SleepResult:
    """Result of a sleep consolidation cycle."""
    phase: SleepPhase
    dream_cycles_completed: int
    candidates_generated: int
    candidates_verified: int
    candidates_stored: int
    improvement_found: bool
    best_candidate: Optional[DreamCandidate] = None
    all_candidates: List[DreamCandidate] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'phase': self.phase.name,
            'dream_cycles_completed': self.dream_cycles_completed,
            'candidates_generated': self.candidates_generated,
            'candidates_verified': self.candidates_verified,
            'candidates_stored': self.candidates_stored,
            'improvement_found': self.improvement_found,
            'best_score': self.best_candidate.verification_score if self.best_candidate else None,
            'stats': self.stats,
        }


VerificationFunction = Callable[[np.ndarray], Tuple[float, bool]]


class SleepConsolidation:
    """
    Seed-based sleep consolidation that doesn't dilute learning.
    """
    
    def __init__(
        self,
        hdc: SparseBinaryHDC,
        storage: GenerativePatternStorage,
        config: Optional[SleepConfig] = None
    ):
        self.hdc = hdc
        self.storage = storage
        self.config = config or SleepConfig()
        self._phase = SleepPhase.AWAKE
        self._current_cycle = 0
        self._candidates: List[DreamCandidate] = []
        self._stats = {
            'total_sleep_cycles': 0,
            'total_candidates_generated': 0,
            'total_candidates_verified': 0,
            'total_candidates_stored': 0,
            'total_improvements': 0,
        }
    
    @property
    def phase(self) -> SleepPhase:
        return self._phase
    
    @property
    def is_sleeping(self) -> bool:
        return self._phase not in (SleepPhase.AWAKE, SleepPhase.COMPLETE)
    
    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)
    
    def sleep_cycle(
        self,
        task_context: np.ndarray,
        verify_fn: Optional[VerificationFunction] = None,
        existing_best_score: float = 0.0
    ) -> SleepResult:
        """Run a complete sleep cycle for pattern discovery."""
        start_time = time.time()
        result = SleepResult(
            phase=SleepPhase.SCANNING,
            dream_cycles_completed=0,
            candidates_generated=0,
            candidates_verified=0,
            candidates_stored=0,
            improvement_found=False,
        )
        self._phase = SleepPhase.SCANNING
        self._current_cycle = 0
        self._candidates = []
        
        try:
            similar_recipes = self._find_similar_recipes(task_context)
            result.stats['similar_recipes_found'] = len(similar_recipes)
            self._phase = SleepPhase.RECOMBINING
            
            for cycle in range(self.config.dream_cycles):
                self._current_cycle = cycle
                if time.time() - start_time > self.config.max_cycle_time_seconds:
                    result.stats['timeout'] = True
                    break
                cycle_candidates = self._generate_candidates(task_context, similar_recipes, cycle)
                self._candidates.extend(cycle_candidates)
                result.dream_cycles_completed = cycle + 1
            
            result.candidates_generated = len(self._candidates)
            self._phase = SleepPhase.VERIFYING
            verified_candidates = self._verify_candidates(self._candidates, verify_fn, task_context)
            result.candidates_verified = len(verified_candidates)
            self._phase = SleepPhase.CONSOLIDATING
            stored_count, improvement = self._consolidate_candidates(verified_candidates, existing_best_score)
            result.candidates_stored = stored_count
            result.improvement_found = improvement
            
            if verified_candidates:
                best = max(verified_candidates, key=lambda c: c.verification_score or 0.0)
                result.best_candidate = best
            result.all_candidates = verified_candidates
            
        finally:
            self._phase = SleepPhase.COMPLETE
            self._stats['total_sleep_cycles'] += 1
            self._stats['total_candidates_generated'] += result.candidates_generated
            self._stats['total_candidates_verified'] += result.candidates_verified
            self._stats['total_candidates_stored'] += result.candidates_stored
            if result.improvement_found:
                self._stats['total_improvements'] += 1
        
        result.phase = SleepPhase.COMPLETE
        return result
    
    def quick_dream(
        self,
        task_context: np.ndarray,
        primitives: List[str],
        verify_fn: VerificationFunction
    ) -> Optional[DreamCandidate]:
        """Quick single-cycle dream for finding a solution fast."""
        candidates = []
        for prim in primitives:
            vec = self.storage.get_primitive_vector(prim)
            if vec is not None:
                score, verified = verify_fn(vec)
                if verified:
                    recipe = PatternRecipe(
                        id=f"dream_single_{prim}",
                        base_seed=self.storage.primitives.get(prim, 0),
                        metadata={'type': 'single_primitive', 'primitive': prim}
                    )
                    cand = DreamCandidate(
                        recipe=recipe,
                        vector=vec,
                        source_seeds=[recipe.base_seed],
                        composition_type='single',
                        verification_score=score,
                        verified=True
                    )
                    candidates.append(cand)
        
        for i, prim1 in enumerate(primitives):
            for prim2 in primitives[i+1:]:
                seq_id = f"dream_seq_{prim1}_{prim2}"
                self.storage.store_sequence(seq_id, [prim1, prim2])
                vec = self.storage.reconstruct(seq_id)
                if vec is not None:
                    score, verified = verify_fn(vec)
                    if verified:
                        recipe = self.storage.recipes.get(seq_id)
                        cand = DreamCandidate(
                            recipe=recipe,
                            vector=vec,
                            source_seeds=[
                                self.storage.primitives.get(prim1, 0),
                                self.storage.primitives.get(prim2, 0)
                            ],
                            composition_type='sequence',
                            verification_score=score,
                            verified=True
                        )
                        candidates.append(cand)
        
        if candidates:
            return max(candidates, key=lambda c: c.verification_score or 0.0)
        return None
    
    def _find_similar_recipes(self, task_context: np.ndarray) -> List[Tuple[str, float]]:
        """Find recipes with similar seeds to the task context."""
        similar = []
        context_seed = self._vector_to_seed(task_context)
        
        if self.config.use_existing_recipes:
            for recipe_id, recipe in self.storage.recipes.items():
                seed_sim = self._seed_similarity(context_seed, recipe.base_seed)
                if seed_sim >= self.config.seed_similarity_threshold:
                    similar.append((recipe_id, seed_sim))
        
        if self.config.use_primitives:
            for prim_name, prim_seed in self.storage.primitives.items():
                seed_sim = self._seed_similarity(context_seed, prim_seed)
                if seed_sim >= self.config.seed_similarity_threshold:
                    similar.append((prim_name, seed_sim))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def _generate_candidates(
        self,
        task_context: np.ndarray,
        similar_recipes: List[Tuple[str, float]],
        cycle: int
    ) -> List[DreamCandidate]:
        """Generate candidate patterns via structured recombination."""
        candidates = []
        top_sources = similar_recipes[:min(10, len(similar_recipes))]
        
        if len(top_sources) >= 2:
            for i in range(min(self.config.candidates_per_cycle, len(top_sources))):
                idx1 = i % len(top_sources)
                idx2 = (i + 1 + cycle) % len(top_sources)
                if idx1 != idx2:
                    src1, _ = top_sources[idx1]
                    src2, _ = top_sources[idx2]
                    seq_id = f"dream_seq_{cycle}_{i}"
                    self.storage.store_sequence(seq_id, [src1, src2])
                    vec = self.storage.reconstruct(seq_id)
                    if vec is not None:
                        recipe = self.storage.recipes.get(seq_id)
                        cand = DreamCandidate(
                            recipe=recipe,
                            vector=vec,
                            source_seeds=self._get_seeds([src1, src2]),
                            composition_type='sequence',
                        )
                        candidates.append(cand)
        
        for src_id, sim in top_sources[:3]:
            if src_id in self.storage.recipes or src_id in self.storage.primitives:
                variation_name = f"dream_var_{cycle}"
                if src_id in self.storage.recipes:
                    var_id = self.storage.generalize(src_id, variation_name, 'SIMILAR')
                else:
                    prim_recipe_id = f"prim_{src_id}"
                    if prim_recipe_id not in self.storage.recipes:
                        self.storage.store_pattern(prim_recipe_id, base_seed=self.storage.primitives[src_id])
                    var_id = self.storage.generalize(prim_recipe_id, variation_name, 'SIMILAR')
                if var_id:
                    vec = self.storage.reconstruct(var_id)
                    if vec is not None:
                        recipe = self.storage.recipes.get(var_id)
                        cand = DreamCandidate(
                            recipe=recipe,
                            vector=vec,
                            source_seeds=self._get_seeds([src_id]),
                            composition_type='generalization',
                        )
                        candidates.append(cand)
        
        if len(top_sources) >= 2:
            src1, _ = top_sources[0]
            src2, _ = top_sources[min(1, len(top_sources)-1)]
            vec1 = self._get_vector(src1)
            vec2 = self._get_vector(src2)
            if vec1 is not None and vec2 is not None:
                combined = self.hdc.bind(vec1, vec2)
                combo_id = f"dream_combo_{cycle}"
                self.storage.store_composite(combo_id, [(src1, 'COMPOSED', 1.0), (src2, 'COMPOSED', 1.0)])
                cand = DreamCandidate(
                    recipe=self.storage.recipes.get(combo_id),
                    vector=combined,
                    source_seeds=self._get_seeds([src1, src2]),
                    composition_type='combination',
                )
                candidates.append(cand)
        return candidates
    
    def _verify_candidates(
        self,
        candidates: List[DreamCandidate],
        verify_fn: Optional[VerificationFunction],
        task_context: np.ndarray
    ) -> List[DreamCandidate]:
        """Verify candidates."""
        verified = []
        if verify_fn is not None:
            for cand in candidates:
                score, is_verified = verify_fn(cand.vector)
                cand.verification_score = score
                cand.verified = is_verified
                if is_verified and score >= self.config.verification_threshold:
                    verified.append(cand)
        else:
            if hasattr(self.hdc, 'batch_similarity') and len(candidates) > 1:
                # batch_similarity(query, candidates) returns similarities of query against all candidates
                candidate_vectors = [cand.vector for cand in candidates]
                similarities = self.hdc.batch_similarity(task_context, candidate_vectors)
                for cand, sim in zip(candidates, similarities):
                    cand.verification_score = float(sim)
                    cand.verified = sim >= self.config.verification_threshold
                    if cand.verified:
                        verified.append(cand)
            else:
                for cand in candidates:
                    sim = self.hdc.similarity(cand.vector, task_context)
                    cand.verification_score = sim
                    cand.verified = sim >= self.config.verification_threshold
                    if cand.verified:
                        verified.append(cand)
        return verified
    
    def _consolidate_candidates(
        self,
        verified_candidates: List[DreamCandidate],
        existing_best_score: float
    ) -> Tuple[int, bool]:
        """Store verified candidates."""
        stored = 0
        improvement = False
        sorted_candidates = sorted(verified_candidates, key=lambda c: c.verification_score or 0.0, reverse=True)
        
        for cand in sorted_candidates[:self.config.max_candidates_to_store]:
            score = cand.verification_score or 0.0
            if score > existing_best_score:
                improvement = True
            if self.config.require_improvement and score <= existing_best_score:
                continue
            if cand.recipe and cand.recipe.id in self.storage.recipes:
                self.storage.recipes[cand.recipe.id].metadata['verified'] = True
                self.storage.recipes[cand.recipe.id].metadata['score'] = score
                stored += 1
        return stored, improvement
    
    def _vector_to_seed(self, vec: np.ndarray) -> int:
        hash_bytes = hashlib.sha256(vec.tobytes()[:64]).digest()
        return int.from_bytes(hash_bytes[:8], 'big')
    
    def _seed_similarity(self, seed1: int, seed2: int) -> float:
        xor = seed1 ^ seed2
        bit_diff = bin(xor).count('1')
        return 1.0 - (bit_diff / 64)
    
    def _get_vector(self, pattern_id: str) -> Optional[np.ndarray]:
        vec = self.storage.reconstruct(pattern_id)
        if vec is not None: return vec
        return self.storage.get_primitive_vector(pattern_id)
    
    def _get_seeds(self, pattern_ids: List[str]) -> List[int]:
        seeds = []
        for pid in pattern_ids:
            if pid in self.storage.recipes:
                seeds.append(self.storage.recipes[pid].base_seed)
            elif pid in self.storage.primitives:
                seeds.append(self.storage.primitives[pid])
        return seeds


# =============================================================================
# SYMBOLIC SLEEP CONSOLIDATION (v2.4.0) - TRM-Style Recursive Discovery
# =============================================================================

@dataclass
class SymbolicSleepResult:
    """Result of a symbolic sleep consolidation cycle."""
    phase: SleepPhase
    dream_cycles_completed: int
    candidates_generated: int
    candidates_verified: int
    candidates_stored: int
    improvement_found: bool
    best_candidate: Optional[SymbolicDreamCandidate] = None
    all_candidates: List[SymbolicDreamCandidate] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # Symbolic recursion specific stats
    total_operations: int = 0
    max_recipe_depth: int = 0
    reconstruction_count: int = 0
    supervision_steps_used: int = 0
    final_recipe_size_bytes: int = 0
    
    def to_dict(self) -> dict:
        return {
            'phase': self.phase.name,
            'dream_cycles_completed': self.dream_cycles_completed,
            'candidates_generated': self.candidates_generated,
            'candidates_verified': self.candidates_verified,
            'candidates_stored': self.candidates_stored,
            'improvement_found': self.improvement_found,
            'best_score': self.best_candidate.verification_score if self.best_candidate else None,
            'total_operations': self.total_operations,
            'max_recipe_depth': self.max_recipe_depth,
            'reconstruction_count': self.reconstruction_count,
            'supervision_steps_used': self.supervision_steps_used,
            'final_recipe_size_bytes': self.final_recipe_size_bytes,
            'stats': self.stats,
        }


class SymbolicSleepConsolidation:
    """
    Symbolic Recursion-Based Sleep Consolidation (v2.4.0)
    """
    
    def __init__(
        self,
        hdc: SparseBinaryHDC,
        storage: GenerativePatternStorage,
        config: Optional[SymbolicSleepConfig] = None,
        synthesizer=None
    ):
        self.hdc = hdc
        self.storage = storage
        self.config = config or SymbolicSleepConfig()
        self.synthesizer = synthesizer
        self._phase = SleepPhase.AWAKE
        self._current_cycle = 0
        self._candidates: List[SymbolicDreamCandidate] = []
        self._context_vectors: Dict[str, np.ndarray] = {}
        self._dsl_primitives: Dict[str, np.ndarray] = {}
        self._build_dsl_primitives()
        self._stats = {
            'total_sleep_cycles': 0,
            'total_candidates_generated': 0,
            'total_candidates_verified': 0,
            'total_candidates_stored': 0,
            'total_improvements': 0,
            'total_operations_created': 0,
            'total_reconstructions': 0,
            'max_recipe_depth_ever': 0,
        }
        if hasattr(storage, 'get_composite'):
            self.profile_consolidator = ProfileConsolidator(storage)
        else:
            self.profile_consolidator = None
    
    def _build_dsl_primitives(self):
        try:
            from .hdc_program_synthesis import DSLPrimitive
            for prim in DSLPrimitive:
                if prim.value not in ['sequence', 'conditional']:
                    seed = prim.get_seed()
                    self._dsl_primitives[prim.value] = self.hdc.from_seed(seed)
                    self._context_vectors[prim.value] = self._dsl_primitives[prim.value]
        except ImportError:
            basic_primitives = [
                'rotate_90', 'rotate_180', 'rotate_270',
                'flip_horizontal', 'flip_vertical',
                'gravity_down', 'gravity_up', 'gravity_left', 'gravity_right',
                'color_swap', 'scale_2x', 'identity'
            ]
            for prim in basic_primitives:
                seed = self._string_to_seed(f"DSL_PRIMITIVE_{prim}")
                self._dsl_primitives[prim] = self.hdc.from_seed(seed)
                self._context_vectors[prim] = self._dsl_primitives[prim]
    
    def _string_to_seed(self, s: str) -> int:
        h = hashlib.sha256(s.encode()).digest()
        return int.from_bytes(h[:8], 'big') & 0x7FFFFFFF
    
    @property
    def phase(self) -> SleepPhase:
        return self._phase
    
    @property
    def is_sleeping(self) -> bool:
        return self._phase not in (SleepPhase.AWAKE, SleepPhase.COMPLETE)
    
    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)
    
    def set_context_vectors(self, vectors: Dict[str, np.ndarray]) -> None:
        self._context_vectors.update(vectors)
    
    def symbolic_sleep_cycle(
        self,
        task_context: np.ndarray,
        verify_fn: Optional[VerificationFunction] = None,
        existing_best_score: float = 0.0,
        x_context: Optional[np.ndarray] = None,
        y_answer: Optional[np.ndarray] = None,
        predictive_model: Optional[Any] = None,
        safety_constraints: Optional[List[Callable[[np.ndarray], bool]]] = None
    ) -> SymbolicSleepResult:
        """Run a SYMBOLIC sleep cycle for pattern discovery AND future simulation."""
        start_time = time.time()
        self._context_vectors["x_context"] = x_context if x_context is not None else task_context
        self._context_vectors["y_answer"] = y_answer if y_answer is not None else task_context
        self._context_vectors["task_context"] = task_context
        
        result = SymbolicSleepResult(
            phase=SleepPhase.SCANNING,
            dream_cycles_completed=0,
            candidates_generated=0,
            candidates_verified=0,
            candidates_stored=0,
            improvement_found=False,
        )
        self._phase = SleepPhase.SCANNING
        self._current_cycle = 0
        self._candidates = []
        
        try:
            similar_recipes = self._find_similar_recipes_symbolic(task_context)
            result.stats['similar_recipes_found'] = len(similar_recipes)
            self._phase = SleepPhase.RECOMBINING
            
            for cycle in range(self.config.dream_cycles):
                self._current_cycle = cycle
                if time.time() - start_time > self.config.max_cycle_time_seconds:
                    result.stats['timeout'] = True
                    break
                
                cycle_candidates = self._generate_symbolic_candidates(task_context, similar_recipes, cycle)
                
                # NEW: Future Simulation (Dreaming of the Future)
                if predictive_model is not None:
                    future_candidates = self._generate_future_simulation_candidates(
                        task_context,
                        predictive_model,
                        safety_constraints,
                        cycle
                    )
                    cycle_candidates.extend(future_candidates)
                
                self._candidates.extend(cycle_candidates)
                result.dream_cycles_completed = cycle + 1
                
                for cand in cycle_candidates:
                    ops = cand.recipe.get_operation_count()
                    result.total_operations += ops
                    if ops > result.max_recipe_depth:
                        result.max_recipe_depth = ops
            
            result.candidates_generated = len(self._candidates)
            self._phase = SleepPhase.VERIFYING
            verified_candidates = self._verify_symbolic_candidates(self._candidates, verify_fn, task_context)
            result.candidates_verified = len(verified_candidates)
            result.reconstruction_count = sum(c.reconstruction_count for c in verified_candidates)
            
            self._phase = SleepPhase.CONSOLIDATING
            stored_count, improvement = self._consolidate_symbolic_candidates(verified_candidates, existing_best_score)
            result.candidates_stored = stored_count
            result.improvement_found = improvement
            
            if verified_candidates:
                best = max(verified_candidates, key=lambda c: c.verification_score or 0.0)
                result.best_candidate = best
                result.final_recipe_size_bytes = best.recipe.get_memory_estimate_bytes()
            result.all_candidates = verified_candidates
            result.supervision_steps_used = min(self.config.n_sup * result.dream_cycles_completed, self.config.n_sup)
            
        finally:
            self._phase = SleepPhase.COMPLETE
            self._stats['total_sleep_cycles'] += 1
            self._stats['total_candidates_generated'] += result.candidates_generated
            self._stats['total_candidates_verified'] += result.candidates_verified
            self._stats['total_candidates_stored'] += result.candidates_stored
            self._stats['total_operations_created'] += result.total_operations
            self._stats['total_reconstructions'] += result.reconstruction_count
            if result.max_recipe_depth > self._stats['max_recipe_depth_ever']:
                self._stats['max_recipe_depth_ever'] = result.max_recipe_depth
            if result.improvement_found:
                self._stats['total_improvements'] += 1
        
        result.phase = SleepPhase.COMPLETE
        return result
    
    def quick_symbolic_dream(
        self,
        task_context: np.ndarray,
        primitives: Optional[List[str]] = None,
        verify_fn: Optional[VerificationFunction] = None,
        n_recursions: int = 10
    ) -> Optional[SymbolicDreamCandidate]:
        """Quick symbolic dream for finding a solution fast."""
        candidates = []
        self._context_vectors["task_context"] = task_context
        self._context_vectors["x_context"] = task_context
        self._context_vectors["y_answer"] = task_context
        
        if primitives is None:
            primitives = list(self._dsl_primitives.keys())
        
        for prim in primitives:
            if prim in self._dsl_primitives:
                self._context_vectors[prim] = self._dsl_primitives[prim]
        
        for prim in primitives[:10]:
            if prim in self._context_vectors:
                recipe = SymbolicSleepRecipe(
                    base_seed=self._string_to_seed(f"DSL_PRIMITIVE_{prim}"),
                    depth_level=self.config.depth_level,
                    metadata={'type': 'single_primitive_recursive', 'primitive': prim}
                )
                recipe.add_bind(prim)
                for step in range(n_recursions):
                    recipe.add_recursion_step(step, "x_context", "y_answer")
                cand = SymbolicDreamCandidate(
                    recipe=recipe,
                    source_seeds=[recipe.base_seed],
                    composition_type='recursive',
                )
                if verify_fn is not None:
                    vec = cand.get_vector(self.hdc, self._context_vectors)
                    score, verified = verify_fn(vec)
                    cand.verification_score = score
                    cand.verified = verified
                    if verified: candidates.append(cand)
                else:
                    vec = cand.get_vector(self.hdc, self._context_vectors)
                    sim = self.hdc.similarity(vec, task_context)
                    cand.verification_score = sim
                    cand.verified = sim >= self.config.verification_threshold
                    if cand.verified: candidates.append(cand)
                cand.clear_cache()
        
        for i, prim1 in enumerate(primitives[:5]):
            for prim2 in primitives[i+1:min(i+4, len(primitives))]:
                recipe = SymbolicSleepRecipe(
                    base_seed=self._string_to_seed(f"DSL_PRIMITIVE_{prim1}"),
                    depth_level=self.config.depth_level,
                    metadata={'type': 'pair_recursive', 'primitives': [prim1, prim2]}
                )
                if prim1 in self._context_vectors:
                    recipe.add_bind(prim1)
                if prim2 in self._context_vectors:
                    recipe.add_bind(prim2)
                    recipe.add_permute(128)
                for step in range(n_recursions):
                    recipe.add_recursion_step(step, "x_context", "y_answer")
                cand = SymbolicDreamCandidate(
                    recipe=recipe,
                    source_seeds=[
                        self._string_to_seed(f"DSL_PRIMITIVE_{prim1}"),
                        self._string_to_seed(f"DSL_PRIMITIVE_{prim2}")
                    ],
                    composition_type='recursive',
                )
                if verify_fn is not None:
                    vec = cand.get_vector(self.hdc, self._context_vectors)
                    score, verified = verify_fn(vec)
                    cand.verification_score = score
                    cand.verified = verified
                    if verified: candidates.append(cand)
                else:
                    vec = cand.get_vector(self.hdc, self._context_vectors)
                    sim = self.hdc.similarity(vec, task_context)
                    cand.verification_score = sim
                    cand.verified = sim >= self.config.verification_threshold
                    if cand.verified: candidates.append(cand)
                cand.clear_cache()
        
        if candidates:
            return max(candidates, key=lambda c: c.verification_score or 0.0)
        return None
    
    def _find_similar_recipes_symbolic(self, task_context: np.ndarray) -> List[Tuple[str, float]]:
        similar = []
        context_seed = self._vector_to_seed(task_context)
        
        if self.config.use_existing_recipes:
            for recipe_id, recipe in self.storage.recipes.items():
                seed_sim = self._seed_similarity(context_seed, recipe.base_seed)
                if seed_sim >= self.config.seed_similarity_threshold:
                    similar.append((recipe_id, seed_sim))
        
        if self.config.use_primitives:
            for prim_name, prim_seed in self.storage.primitives.items():
                seed_sim = self._seed_similarity(context_seed, prim_seed)
                if seed_sim >= self.config.seed_similarity_threshold:
                    similar.append((prim_name, seed_sim))
        
        for dsl_name in self._dsl_primitives.keys():
            dsl_seed = self._string_to_seed(f"DSL_PRIMITIVE_{dsl_name}")
            seed_sim = self._seed_similarity(context_seed, dsl_seed)
            if seed_sim >= self.config.seed_similarity_threshold * 0.8:
                similar.append((f"dsl_{dsl_name}", seed_sim))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def run_profile_maintenance(self):
        """
        New Sleep Phase: Optimizes Agent Personas and Common Contexts.
        """
        if not self.profile_consolidator:
            return
            
        print("Running Profile Crystallization...")
        shortcuts = self.profile_consolidator.run_crystallization_cycle()
        
        for sc in shortcuts:
            print(f"  -> Crystallized Shortcut: {sc}")
            # Identify what it is
            if "PROFILE" in sc:
                self._stats['profiles_consolidated'] = self._stats.get('profiles_consolidated', 0) + 1

    def _generate_future_simulation_candidates(
        self,
        start_state: np.ndarray,
        model: Any,  # PredictiveModel
        safety_constraints: Optional[List[Callable[[np.ndarray], bool]]],
        cycle: int
    ) -> List[SymbolicDreamCandidate]:
        """
        Generate candidates by simulating future trajectories.
        """
        candidates = []
        actions = []
        for i in range(3): # 3-step plan
            action_vec = self.hdc.random_vector(seed=cycle * 100 + i)
            actions.append(action_vec)
            
        # Simulate (Mental Time Travel)
        trajectory = model.simulate_trajectory(start_state, actions)
        final_state = trajectory[-1]
        
        # Evaluate Safety
        is_safe = True
        if safety_constraints:
            for constraint in safety_constraints:
                if not constraint(final_state):
                    is_safe = False
                    break
        
        base_seed = self._vector_to_seed(start_state)
        recipe = SymbolicSleepRecipe(
            base_seed=base_seed,
            depth_level=self.config.depth_level,
            metadata={
                'type': 'future_simulation',
                'cycle': cycle,
                'is_safe': is_safe,
                'trajectory_length': len(trajectory)
            }
        )
        
        # Add actions to recipe
        for i, action in enumerate(actions):
            action_key = f"sim_action_{cycle}_{i}"
            self._context_vectors[action_key] = action
            recipe.add_bind(action_key)
            
        cand = SymbolicDreamCandidate(
            recipe=recipe,
            source_seeds=[base_seed],
            composition_type='simulation',
            _cached_vector=final_state
        )
        candidates.append(cand)
        return candidates

    def _generate_symbolic_candidates(
        self,
        task_context: np.ndarray,
        similar_recipes: List[Tuple[str, float]],
        cycle: int
    ) -> List[SymbolicDreamCandidate]:
        """Generate candidate patterns via SYMBOLIC recursive composition."""
        candidates = []
        top_sources = similar_recipes[:min(10, len(similar_recipes))]
        
        for src_id, _ in top_sources:
            if src_id not in self._context_vectors:
                vec = self._get_vector(src_id)
                if vec is not None:
                    self._context_vectors[src_id] = vec
        
        n_sup = min(self.config.n_sup, 4 + cycle * 2)
        n_recursions = self.config.n_recursions
        t_rounds = self.config.t_rounds
        
        for src_idx, (src_id, sim) in enumerate(top_sources[:self.config.candidates_per_cycle]):
            base_seed = self._get_seed(src_id)
            recipe = SymbolicSleepRecipe(
                base_seed=base_seed,
                depth_level=self.config.depth_level,
                metadata={
                    'type': 'recursive_discovery',
                    'source': src_id,
                    'cycle': cycle,
                    'n_sup': n_sup,
                    'n_recursions': n_recursions,
                    't_rounds': t_rounds
                }
            )
            recipe.add_bind(src_id)
            for sup_step in range(n_sup):
                for t in range(t_rounds):
                    for r in range(n_recursions):
                        global_step = sup_step * t_rounds * n_recursions + t * n_recursions + r
                        recipe.add_recursion_step(global_step, "x_context", "y_answer")
                recipe.add_operation("refine_answer", sup_step)
            cand = SymbolicDreamCandidate(
                recipe=recipe,
                source_seeds=[base_seed],
                composition_type='recursive',
            )
            candidates.append(cand)
        
        dsl_prims = list(self._dsl_primitives.keys())[:8]
        if len(dsl_prims) >= 2:
            for i in range(min(self.config.candidates_per_cycle // 2, len(dsl_prims) - 1)):
                idx1 = i % len(dsl_prims)
                idx2 = (i + 1 + cycle) % len(dsl_prims)
                if idx1 != idx2:
                    prim1 = dsl_prims[idx1]
                    prim2 = dsl_prims[idx2]
                    seed1 = self._string_to_seed(f"DSL_PRIMITIVE_{prim1}")
                    seed2 = self._string_to_seed(f"DSL_PRIMITIVE_{prim2}")
                    recipe = SymbolicSleepRecipe(
                        base_seed=seed1,
                        depth_level=self.config.depth_level,
                        metadata={
                            'type': 'dsl_sequence',
                            'primitives': [prim1, prim2],
                            'cycle': cycle
                        }
                    )
                    recipe.add_bind(prim1)
                    recipe.add_bind(prim2)
                    recipe.add_permute((i + 1) * 64)
                    for step in range(min(n_recursions, 10)):
                        recipe.add_recursion_step(step, "x_context", "y_answer")
                    cand = SymbolicDreamCandidate(
                        recipe=recipe,
                        source_seeds=[seed1, seed2],
                        composition_type='sequence',
                    )
                    candidates.append(cand)
        
        for src_id, sim in top_sources[:3]:
            base_seed = self._get_seed(src_id)
            variation_seed = (base_seed + cycle * 1337 + 42) & 0x7FFFFFFFFFFFFFFF
            recipe = SymbolicSleepRecipe(
                base_seed=variation_seed,
                depth_level=self.config.depth_level,
                metadata={
                    'type': 'generalization',
                    'source': src_id,
                    'original_seed': base_seed,
                    'cycle': cycle
                }
            )
            recipe.add_bind(src_id)
            for step in range(min(n_recursions, 5)):
                recipe.add_recursion_step(step, "x_context", "y_answer")
            cand = SymbolicDreamCandidate(
                recipe=recipe,
                source_seeds=[variation_seed],
                composition_type='generalization',
            )
            candidates.append(cand)

        atomic_ops = [
            "pixel_get", "pixel_set", "add", "sub", "if_eq", "loop",
            "neighbors", "stack_push", "stack_pop"
        ]
        if cycle % 2 == 0:
            import random
            rng = random.Random(cycle * 999)
            seq_len = 3 + (cycle % 3)
            selected_atoms = [rng.choice(atomic_ops) for _ in range(seq_len)]
            atom_seed = self._string_to_seed(f"atomic_seq_{cycle}")
            recipe = SymbolicSleepRecipe(
                base_seed=atom_seed,
                depth_level=self.config.depth_level,
                metadata={
                    'type': 'atomic_invention',
                    'atoms': selected_atoms,
                    'cycle': cycle
                }
            )
            for i, atom in enumerate(selected_atoms):
                atom_key = f"atom_{atom}"
                if atom_key not in self._context_vectors:
                    self._context_vectors[atom_key] = self.hdc.from_string(atom_key)[1]
                recipe.add_bind(atom_key)
                recipe.add_permute((i + 1) * 32)
            for step in range(min(n_recursions, 5)):
                recipe.add_recursion_step(step, "x_context", "y_answer")
            cand = SymbolicDreamCandidate(
                recipe=recipe,
                source_seeds=[atom_seed],
                composition_type='atomic_invention',
            )
            candidates.append(cand)
        
        return candidates
    
    def _verify_symbolic_candidates(
        self,
        candidates: List[SymbolicDreamCandidate],
        verify_fn: Optional[VerificationFunction],
        task_context: np.ndarray
    ) -> List[SymbolicDreamCandidate]:
        """Verify symbolic candidates."""
        verified = []
        for cand in candidates:
            vec = cand.get_vector(self.hdc, self._context_vectors)
            if verify_fn is not None:
                score, is_verified = verify_fn(vec)
                cand.verification_score = score
                cand.verified = is_verified
                if is_verified and score >= self.config.verification_threshold:
                    verified.append(cand)
            else:
                sim = self.hdc.similarity(vec, task_context)
                cand.verification_score = sim
                cand.verified = sim >= self.config.verification_threshold
                if cand.verified:
                    verified.append(cand)
            cand.clear_cache()
        return verified
    
    def _consolidate_symbolic_candidates(
        self,
        verified_candidates: List[SymbolicDreamCandidate],
        existing_best_score: float
    ) -> Tuple[int, bool]:
        """Store verified symbolic candidates as recipes."""
        stored = 0
        improvement = False
        sorted_candidates = sorted(verified_candidates, key=lambda c: c.verification_score or 0.0, reverse=True)
        
        for cand in sorted_candidates[:self.config.max_candidates_to_store]:
            score = cand.verification_score or 0.0
            if score > existing_best_score:
                improvement = True
            if self.config.require_improvement and score <= existing_best_score:
                continue
            
            recipe_id = f"symbolic_sleep_{self._stats['total_candidates_stored']}"
            self.storage.store_pattern(
                recipe_id,
                base_seed=cand.recipe.base_seed,
                description=f"Symbolic recursive discovery (ops={cand.recipe.get_operation_count()}, score={score:.3f})",
                metadata={
                    'symbolic_operations': cand.recipe.operations,
                    'depth_level': cand.recipe.depth_level,
                    'composition_type': cand.composition_type,
                    'verification_score': score,
                    'verified': True,
                    'source_seeds': cand.source_seeds,
                }
            )
            stored += 1
        return stored, improvement
    
    def _vector_to_seed(self, vec: np.ndarray) -> int:
        hash_bytes = hashlib.sha256(vec.tobytes()[:64]).digest()
        return int.from_bytes(hash_bytes[:8], 'big')
    
    def _seed_similarity(self, seed1: int, seed2: int) -> float:
        xor = seed1 ^ seed2
        bit_diff = bin(xor).count('1')
        return 1.0 - (bit_diff / 64)
    
    def _get_vector(self, pattern_id: str) -> Optional[np.ndarray]:
        vec = self.storage.reconstruct(pattern_id)
        if vec is not None: return vec
        vec = self.storage.get_primitive_vector(pattern_id)
        if vec is not None: return vec
        if pattern_id.startswith("dsl_"):
            dsl_name = pattern_id[4:]
            if dsl_name in self._dsl_primitives:
                return self._dsl_primitives[dsl_name]
        return None
    
    def _get_seed(self, pattern_id: str) -> int:
        if pattern_id in self.storage.recipes:
            return self.storage.recipes[pattern_id].base_seed
        elif pattern_id in self.storage.primitives:
            return self.storage.primitives[pattern_id]
        elif pattern_id.startswith("dsl_"):
            dsl_name = pattern_id[4:]
            return self._string_to_seed(f"DSL_PRIMITIVE_{dsl_name}")
        else:
            return self._string_to_seed(pattern_id)

@dataclass
class ShortcutCandidate:
    """A proposed optimization for the Generative Registry."""
    name: str
    components: List[str]
    frequency: int
    saved_ops: int  # How many XOR ops this saves per use

class ProfileConsolidator:
    """
    Specialized Sleep Module for Character & Information Optimization.
    Interacts with the GenerativeRegistry to create O(1) shortcuts.
    """
    def __init__(self, registry):
        self.registry = registry
        self.access_history: List[List[str]] = [] # Logs component lists used
        
    def log_access(self, components: List[str]):
        """Call this whenever an agent constructs a composite vector."""
        # Sort to ensure [A, B] is treated same as [B, A] for counting
        self.access_history.append(sorted(components))
        
    def run_crystallization_cycle(self, threshold: int = 5) -> List[str]:
        """
        Analyzes history and creates shortcuts for frequent patterns.
        Returns list of newly created shortcuts.
        """
        if not self.access_history:
            return []
            
        # 1. Count Frequencies of Component Sets
        # Convert lists to tuples to be hashable for counting
        counts: Dict[Tuple[str, ...], int] = {}
        for comp_list in self.access_history:
            key = tuple(comp_list)
            # Only optimize complex things (len > 1)
            if len(key) > 1:
                counts[key] = counts.get(key, 0) + 1
        
        new_shortcuts = []
        
        # 2. Identify High-Value Targets
        for components_tuple, count in counts.items():
            if count >= threshold:
                # Calculate "Energy Saved"
                # Energy = Count * (Num_Components - 1) XOR ops
                energy_saved = count * (len(components_tuple) - 1)
                
                if energy_saved > 10: # Minimum ROI to justify a shortcut
                    shortcut_name = self._generate_shortcut_name(components_tuple)
                    
                    # 3. Mint the Shortcut in Registry
                    # We ask the registry to pre-calculate this composite
                    # and store it under the simple name.
                    self._crystallize(shortcut_name, list(components_tuple))
                    new_shortcuts.append(shortcut_name)
                    
        # Clear history after consolidation (Episodic reset)
        self.access_history = []
        return new_shortcuts

    def _generate_shortcut_name(self, components: Tuple[str, ...]) -> str:
        """
        Generates a readable name or hash for the shortcut.
        Heuristic: Detect if it looks like a Character Profile.
        """
        # Check for agent markers
        agent_tag = next((c for c in components if "AGENT" in c or "PROFILE" in c), None)
        
        if agent_tag:
            # e.g., "SHORTCUT::PROFILE_SCOUT_V1"
            return f"SHORTCUT::{agent_tag}_optimized"
        else:
            # Generic hash for info bundles
            h = hashlib.sha256("".join(components).encode()).hexdigest()[:8]
            return f"SHORTCUT::BUNDLE_{h}"

    def _crystallize(self, name: str, components: List[str]):
        """
        The Core Logic: Binds the components and caches the result 
        as if it were an Atomic Vector.
        """
        # 1. Calculate the expensive composite vector
        # (This uses the 'bind_all' logic from simple_hybrid_memory)
        # We assume self.registry has access to the HDC instance
        
        # Note: We need to import bind_all or have it available via registry
        # For this snippet, we assume registry.get_composite does the work
        vector = self.registry.get_composite(components)
        
        # 2. Force-Cache it in the Registry
        # This makes future lookups O(1)
        self.registry.cache[name] = vector
        
        # 3. (Optional) Register the definition for provenance
        # If the registry has a 'definitions' dict, save it there
        if hasattr(self.registry, 'definitions'):
            self.registry.definitions[name] = {
                "type": "shortcut",
                "components": components,
                "created_at": time.time()
            }

    def schedule_event(event_name, date_str):
        components = [
            f"EVENT::{event_name}",
            f"TIME::{date_str}",
            "STATUS::PENDING"
        ]
        
        # 1. Force Save the Event Detail
        sleep_system.profile_consolidator.log_access(components, important=True)
        
        # 2. Register a "Time Trigger" (The Alarm Clock)
        # We bind the date to the shortcut name in the global memory
        # "When is TIME::2025-12-25? -> It is linked to SHORTCUT::EVENT_X"
        global_memory.store(
            pattern_id=f"CALENDAR::{date_str}", 
            vector=registry.get_composite([f"LINK::{event_name}"])
        )

    def log_access(self, components: List[str], important: bool = False):
        """
        Logs usage. If 'important' is True, it forces immediate Crystallization
        without waiting for the frequency threshold.
        """
        # Standard logging
        self.access_history.append(sorted(components))
        
        # IMMEDIATE ACTION for Important Events (Birthdays/Deadlines)
        if important:
            # Create a unique name immediately
            shortcut_name = self._generate_shortcut_name(tuple(sorted(components)))
            
            # Force Crystallize NOW
            self._crystallize(shortcut_name, components)
            
            print(f"  [MEMORY] Important Event Anchored: {shortcut_name}")
# =============================================================================
# Factory Functions
# =============================================================================

def create_symbolic_sleep_system(
    hdc: SparseBinaryHDC,
    storage: Optional[GenerativePatternStorage] = None,
    depth_level: str = 'medium',
    synthesizer=None,
    **config_kwargs
) -> Tuple[SymbolicSleepConsolidation, SymbolicSleepConfig]:
    """Factory function to create a SYMBOLIC sleep consolidation system."""
    if storage is None:
        storage = GenerativePatternStorage(hdc)
    
    config = SymbolicSleepConfig(depth_level=depth_level, **config_kwargs)
    sleep = SymbolicSleepConsolidation(hdc, storage, config, synthesizer)
    return sleep, config


__all__ = [
    'SleepPhase',
    'SleepConfig',
    'DreamCandidate',
    'SleepResult',
    'SleepConsolidation',
    'VerificationFunction',
    'create_sleep_system',
    'create_verification_function',
    'SYMBOLIC_DEPTH_LEVELS',
    'SymbolicSleepRecipe',
    'SymbolicSleepConfig',
    'SymbolicDreamCandidate',
    'SymbolicSleepResult',
    'SymbolicSleepConsolidation',
    'create_symbolic_sleep_system',
]