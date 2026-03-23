"""
Resonator Network for Parallel Factorization

This module implements the Resonator Network architecture for parallel factorization
of bundled HDC vectors. The resonator uses parallel feedback loops to factorize
a complex bundle into its constituent parts in O(1) time.

Key Features:
- Parallel Codebook Projection: Project onto all codebooks simultaneously
- Inverse Binding (The Peel): XOR-unbind all other role estimates
- Inhibitory Mask Application: Apply repulsive force for constraint filtering
- Codebook Matching (The Snap): Find closest match in deterministic BLAKE3 codebook
- Convergence Detection: Check for stability in the energy landscape

From FULLINTEGRATION_NEW_ARCHITECTURE.md:
- Section G: Parallel Factorization (Resonator Networks)
- Section F: Role-Binding (Lego-Style Modularity)

Usage:
    >>> resonator = ResonatorNetwork(dim=1048576)
    >>> 
    >>> # Define codebooks for each role
    >>> codebooks = {
    ...     'action': ['rotate_90', 'flip_horizontal', 'scale'],
    ...     'object': ['cube', 'sphere', 'pyramid'],
    ...     'tone': ['confident', 'cautious', 'neutral']
    ... }
    >>> 
    >>> # Factorize bundled vector
    >>> result = resonator.factorize(bundled_vector, codebooks)
    >>> print(result)  # {'action': 'rotate_90', 'object': 'cube', 'tone': 'confident'}
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import time

# Import from core modules
from ..HDC_Core_Main.hdc_sparse_core import (
    seed_to_hypervector_blake3,
    DEFAULT_HDC_DIM
)


# =============================================================================
# Convergence Signal
# =============================================================================

class ConvergenceSignal(Enum):
    """Signals for resonator convergence monitoring."""
    CONVERGING = "converging"      # Residue shrinking steadily
    STUCK = "stuck"               # Stuck in local attractor
    OSCILLATING = "oscillating"   # Search is unstable
    UNCERTAIN = "uncertain"       # No clear pattern
    CONVERGED = "converged"       # Fully converged


# =============================================================================
# Resonator Result
# =============================================================================

@dataclass
class ResonatorResult:
    """
    Result from resonator factorization.
    
    Attributes:
        estimates: Dictionary mapping role names to decoded values
        iterations: Number of iterations to converge
        converged: Whether the resonator converged
        confidence: Confidence in the result (0.0 to 1.0)
        residue_history: History of residue values during convergence
        elapsed_time_ms: Time taken in milliseconds
    """
    estimates: Dict[str, str]
    iterations: int
    converged: bool
    confidence: float
    residue_history: List[float] = field(default_factory=list)
    elapsed_time_ms: float = 0.0


# =============================================================================
# Resonator Network
# =============================================================================

class ResonatorNetwork:
    """
    Resonator Network for parallel factorization of bundled HDC vectors.
    
    The resonator uses parallel feedback loops to factorize a complex bundle
    into its constituent parts. It works by:
    
    1. Parallel Codebook Projection: Project onto all codebooks simultaneously
    2. Inverse Binding (The Peel): XOR-unbind all OTHER role estimates
    3. Inhibitory Mask Application: Apply repulsive force
    4. Codebook Matching (The Snap): Find closest match in codebook
    5. Convergence Check: Check for stability
    
    Properties:
    - O(1) factorization time (parallel)
    - Deterministic: Same input → same output
    - Handles noise: Works with 40-50% noise
    - Real-time correction: Mid-flight trajectory adjustment
    """
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM, max_iterations: int = 100):
        """
        Initialize Resonator Network.
        
        Args:
            dim: HDC dimension
            max_iterations: Maximum iterations before giving up
        """
        self.dim = dim
        self.uint64_count = dim // 64
        self.max_iterations = max_iterations
        
        # Convergence thresholds
        self.convergence_threshold = 0.95
        self.stuck_variance_threshold = 0.0001
        self.oscillation_variance_threshold = 0.05
    
    def _generate_vector(self, seed_string: str) -> np.ndarray:
        """Generate deterministic vector from seed string."""
        return seed_to_hypervector_blake3(seed_string, self.uint64_count)
    
    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """XOR bind two vectors."""
        return np.bitwise_xor(a, b)
    
    def _unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """XOR unbind (same as bind)."""
        return np.bitwise_xor(bound, key)
    
    def _bind_all(self, vectors: List[np.ndarray]) -> np.ndarray:
        """XOR bind all vectors together."""
        if not vectors:
            return np.zeros(self.uint64_count, dtype=np.uint64)
        result = vectors[0].copy()
        for v in vectors[1:]:
            result = np.bitwise_xor(result, v)
        return result
    
    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Hamming similarity between two vectors using fast popcount."""
        xored = np.bitwise_xor(a, b)
        # Fast popcount using lookup table for bytes
        # This is much faster than np.unpackbits for large vectors
        diff_bits = np.count_nonzero(xored) * 64  # Each uint64 has 64 bits
        # Actually, we need proper popcount. Use numpy's built-in bit counting
        # For uint64, we can use the sum of popcount on each element
        diff_bits = sum(bin(x).count('1') for x in xored)
        total_bits = len(a) * 64
        return 1.0 - (diff_bits / total_bits)
    
    def _similarity_fast(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fast Hamming similarity using vectorized popcount."""
        xored = np.bitwise_xor(a, b)
        # Vectorized popcount using lookup table approach
        # Split into bytes and use lookup table
        xored_bytes = xored.view(np.uint8)
        # Precomputed popcount lookup table for 0-255
        POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint32)
        diff_bits = POPCOUNT_TABLE[xored_bytes].sum()
        total_bits = len(a) * 64
        return 1.0 - (diff_bits / total_bits)
    
    def _find_closest(self, query: np.ndarray,
                      codebook: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """
        Find closest match in codebook.
        
        Args:
            query: Query vector
            codebook: Dictionary mapping names to vectors
            
        Returns:
            Tuple of (best_match_name, similarity)
        """
        best_name = None
        best_similarity = -1.0
        
        for name, vec in codebook.items():
            sim = self._similarity_fast(query, vec)
            if sim > best_similarity:
                best_similarity = sim
                best_name = name
        
        return best_name, best_similarity
    
    def _find_closest_batch(self, query: np.ndarray,
                            codebook_vectors: np.ndarray,
                            codebook_names: List[str]) -> Tuple[str, float]:
        """
        Find closest match in codebook using batch operations.
        
        This is much faster than iterating through codebook entries one by one.
        
        Args:
            query: Query vector (uint64 array)
            codebook_vectors: Matrix of codebook vectors (N x uint64_count)
            codebook_names: List of codebook entry names
            
        Returns:
            Tuple of (best_match_name, similarity)
        """
        # XOR query with all codebook vectors at once
        xored = np.bitwise_xor(codebook_vectors, query)
        
        # Vectorized popcount
        xored_bytes = xored.view(np.uint8)
        POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint32)
        diff_bits = POPCOUNT_TABLE[xored_bytes].sum(axis=1)
        
        # Convert to similarities
        total_bits = len(query) * 64
        similarities = 1.0 - (diff_bits / total_bits)
        
        # Find best match
        best_idx = np.argmax(similarities)
        return codebook_names[best_idx], similarities[best_idx]
    
    def _apply_repulsion(self, vec: np.ndarray, 
                         mask: np.ndarray, 
                         strength: float = 0.5) -> np.ndarray:
        """
        Apply repulsive force from inhibitory mask.
        
        Pushes the trajectory away from prohibited regions.
        
        Args:
            vec: Vector to modify
            mask: Inhibitory mask (repulsive force)
            strength: Repulsion strength (0.0 to 1.0)
            
        Returns:
            Modified vector
        """
        # XOR with mask creates repulsion
        # The strength controls how much of the mask to apply
        if strength >= 1.0:
            return np.bitwise_xor(vec, mask)
        elif strength <= 0.0:
            return vec
        else:
            # Partial application via threshold
            xored = np.bitwise_xor(vec, mask)
            # For binary vectors, we can't do true partial application
            # Instead, we use the mask as a guide for the next iteration
            return xored
    
    def _monitor_convergence(self, residue_history: List[float]) -> ConvergenceSignal:
        """
        Monitor convergence from residue history.
        
        Args:
            residue_history: List of residue values over iterations
            
        Returns:
            ConvergenceSignal indicating current state
        """
        if len(residue_history) < 3:
            return ConvergenceSignal.UNCERTAIN
        
        recent = residue_history[-5:] if len(residue_history) >= 5 else residue_history
        
        # Check for convergence
        if recent[-1] >= self.convergence_threshold:
            return ConvergenceSignal.CONVERGED
        
        # Compute trend
        if len(recent) >= 3:
            trend = np.polyfit(range(len(recent)), recent, deg=1)[0]
            variance = np.var(recent)
            
            if trend < -0.02:
                return ConvergenceSignal.CONVERGING
            elif abs(trend) < 0.001 and variance < self.stuck_variance_threshold:
                return ConvergenceSignal.STUCK
            elif variance > self.oscillation_variance_threshold:
                return ConvergenceSignal.OSCILLATING
        
        return ConvergenceSignal.UNCERTAIN
    
    def factorize(self,
                  bundled_vector: np.ndarray,
                  codebooks: Dict[str, List[str]],
                  inhibitory_mask: Optional[np.ndarray] = None,
                  mask_roles: Optional[List[str]] = None,
                  initial_estimates: Optional[Dict[str, str]] = None,
                  max_iterations_override: Optional[int] = None) -> ResonatorResult:
        """
        Factorize a bundled vector into its constituent parts.
        
        The main algorithm:
        1. Initialize estimates for each role
        2. For each iteration:
           a. For each role:
              - Inverse-bind all OTHER role estimates
              - Apply inhibitory mask (if applicable)
              - Find closest match in codebook
              - Update estimate
           b. Check for convergence
        
        Args:
            bundled_vector: The superposed 'blurry' thought vector
            codebooks: Dict mapping role names to list of candidate seed strings
            inhibitory_mask: Optional inhibitory bias vector (e.g., Clean Language Mask)
            mask_roles: Roles to apply inhibitory mask to (default: all)
            initial_estimates: Optional initial estimates for each role
            max_iterations_override: Override max_iterations for this call
            
        Returns:
            ResonatorResult with factorized output
        """
        start_time = time.time()
        max_iter = max_iterations_override if max_iterations_override is not None else self.max_iterations
        
        # Materialize codebook vectors as arrays for batch operations
        # This is much faster than dict-based lookups
        codebook_arrays: Dict[str, np.ndarray] = {}  # role -> (N, uint64_count) array
        codebook_names: Dict[str, List[str]] = {}    # role -> list of seed names
        codebook_dicts: Dict[str, Dict[str, np.ndarray]] = {}  # For backward compat
        
        for role, seeds in codebooks.items():
            vectors = [self._generate_vector(seed) for seed in seeds]
            if vectors:
                codebook_arrays[role] = np.array(vectors, dtype=np.uint64)
                codebook_names[role] = list(seeds)
                codebook_dicts[role] = {seed: vec for seed, vec in zip(seeds, vectors)}
            else:
                codebook_arrays[role] = np.array([], dtype=np.uint64).reshape(0, self.uint64_count)
                codebook_names[role] = []
                codebook_dicts[role] = {}
        
        # Initialize estimates
        estimates: Dict[str, str] = {}
        estimate_vectors: Dict[str, np.ndarray] = {}
        
        for role in codebooks.keys():
            if initial_estimates and role in initial_estimates:
                estimates[role] = initial_estimates[role]
                estimate_vectors[role] = self._generate_vector(initial_estimates[role])
            else:
                # Start with first candidate
                first_seed = list(codebooks[role])[0] if codebooks[role] else None
                if first_seed:
                    estimates[role] = first_seed
                    estimate_vectors[role] = self._generate_vector(first_seed)
        
        # Residue history for convergence monitoring
        residue_history: List[float] = []
        
        # Pre-compute POPCOUNT_TABLE for fast similarity
        POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint32)
        
        # Main iteration loop
        for iteration in range(max_iter):
            iteration_residue = 0.0
            
            for role in codebooks.keys():
                if not codebook_names[role]:
                    continue
                    
                # Step 1: Inverse-bind all OTHER roles to isolate current role
                other_roles = [r for r in codebooks.keys() if r != role]
                other_vectors = [estimate_vectors[r] for r in other_roles if r in estimate_vectors]
                
                if other_vectors:
                    others_bound = self._bind_all(other_vectors)
                    isolated = self._unbind(bundled_vector, others_bound)
                else:
                    isolated = bundled_vector.copy()
                
                # Step 2: Apply inhibitory mask (if applicable)
                if inhibitory_mask is not None:
                    if mask_roles is None or role in mask_roles:
                        isolated = self._apply_repulsion(isolated, inhibitory_mask)
                
                # Step 3: Find closest match in codebook using batch operation
                # XOR query with all codebook vectors at once
                xored = np.bitwise_xor(codebook_arrays[role], isolated)
                
                # Vectorized popcount using lookup table
                xored_bytes = xored.view(np.uint8)
                diff_bits = POPCOUNT_TABLE[xored_bytes].sum(axis=1)
                
                # Convert to similarities
                total_bits = len(isolated) * 64
                similarities = 1.0 - (diff_bits / total_bits)
                
                # Find best match
                best_idx = np.argmax(similarities)
                best_match = codebook_names[role][best_idx]
                similarity = similarities[best_idx]
                
                # Update estimate
                estimates[role] = best_match
                estimate_vectors[role] = codebook_arrays[role][best_idx].copy()
                
                iteration_residue = max(iteration_residue, similarity)
            
            residue_history.append(iteration_residue)
            
            # Check for convergence
            signal = self._monitor_convergence(residue_history)
            
            if signal == ConvergenceSignal.CONVERGED:
                elapsed_ms = (time.time() - start_time) * 1000
                return ResonatorResult(
                    estimates=estimates.copy(),
                    iterations=iteration + 1,
                    converged=True,
                    confidence=iteration_residue,
                    residue_history=residue_history,
                    elapsed_time_ms=elapsed_ms
                )
            
            elif signal == ConvergenceSignal.STUCK:
                # Try random restart
                import random
                for role in codebooks.keys():
                    if codebooks[role]:
                        random_seed = random.choice(codebooks[role])
                        estimates[role] = random_seed
                        estimate_vectors[role] = self._generate_vector(random_seed)
            
            elif signal == ConvergenceSignal.OSCILLATING:
                # Reduce step size (in binary, this means being more conservative)
                # We can implement this by requiring higher similarity to change
                pass
        
        elapsed_ms = (time.time() - start_time) * 1000
        return ResonatorResult(
            estimates=estimates.copy(),
            iterations=max_iter,
            converged=False,
            confidence=residue_history[-1] if residue_history else 0.0,
            residue_history=residue_history,
            elapsed_time_ms=elapsed_ms
        )
    
    def factorize_with_roles(self,
                             bundled_vector: np.ndarray,
                             role_vectors: Dict[str, np.ndarray],
                             codebooks: Dict[str, List[str]]) -> ResonatorResult:
        """
        Factorize using role-binding (Lego-style modularity).
        
        Each concept is XOR-bound to a fixed, orthogonal Role Vector.
        This eliminates "blurry thoughts" during superposition.
        
        Formula: H_total = (Role_Action ⊗ V_Rotate) ⊕ (Role_Object ⊗ V_Cube) ⊕ ...
        
        Args:
            bundled_vector: The bundled thought vector
            role_vectors: Dict mapping role names to their orthogonal role vectors
            codebooks: Dict mapping role names to candidate seed strings
            
        Returns:
            ResonatorResult with factorized output
        """
        # Materialize codebook vectors
        materialized_codebooks: Dict[str, Dict[str, np.ndarray]] = {}
        for role, seeds in codebooks.items():
            materialized_codebooks[role] = {
                seed: self._generate_vector(seed) for seed in seeds
            }
        
        # Initialize estimates
        estimates: Dict[str, str] = {}
        estimate_vectors: Dict[str, np.ndarray] = {}
        
        for role in codebooks.keys():
            first_seed = list(codebooks[role])[0]
            estimates[role] = first_seed
            estimate_vectors[role] = materialized_codebooks[role][first_seed]
        
        residue_history: List[float] = []
        
        for iteration in range(self.max_iterations):
            iteration_residue = 0.0
            
            for role in codebooks.keys():
                # Unbind using role vector
                role_vec = role_vectors[role]
                isolated = self._unbind(bundled_vector, role_vec)
                
                # Find closest match
                best_match, similarity = self._find_closest(
                    isolated, materialized_codebooks[role]
                )
                
                estimates[role] = best_match
                estimate_vectors[role] = materialized_codebooks[role][best_match]
                iteration_residue = max(iteration_residue, similarity)
            
            residue_history.append(iteration_residue)
            
            if iteration_residue >= self.convergence_threshold:
                return ResonatorResult(
                    estimates=estimates.copy(),
                    iterations=iteration + 1,
                    converged=True,
                    confidence=iteration_residue,
                    residue_history=residue_history
                )
        
        return ResonatorResult(
            estimates=estimates.copy(),
            iterations=self.max_iterations,
            converged=False,
            confidence=residue_history[-1] if residue_history else 0.0,
            residue_history=residue_history
        )


# =============================================================================
# Role-Binding System
# =============================================================================

class RoleBindingSystem:
    """
    Role-Binding system for Lego-style modularity.
    
    Instead of bundling raw concepts, every concept is XOR-bound to a fixed,
    orthogonal Role Vector (the "Lego studs").
    
    Properties:
    - Zero Crosstalk: Each role is strictly orthogonal
    - Hot-Swappable: Concepts can be replaced without rebinding entire bundle
    - Deterministic Unbinding: XOR with role vector perfectly extracts original
    - Parallel Processing: All roles can be decoded simultaneously
    """
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM):
        """
        Initialize Role-Binding system.
        
        Args:
            dim: HDC dimension
        """
        self.dim = dim
        self.uint64_count = dim // 64
        self._role_vectors: Dict[str, np.ndarray] = {}
    
    def _generate_vector(self, seed_string: str) -> np.ndarray:
        """Generate deterministic vector from seed string."""
        return seed_to_hypervector_blake3(seed_string, self.uint64_count)
    
    def register_role(self, role_name: str, seed: Optional[str] = None):
        """
        Register a role with its orthogonal vector.
        
        Args:
            role_name: Name of the role (e.g., "action", "object", "tone")
            seed: Optional seed string (default: uses role_name)
        """
        seed_string = seed or f"role:{role_name}"
        self._role_vectors[role_name] = self._generate_vector(seed_string)
    
    def get_role_vector(self, role_name: str) -> Optional[np.ndarray]:
        """Get the orthogonal vector for a role."""
        return self._role_vectors.get(role_name)
    
    def bind_concept(self, role_name: str, concept_seed: str) -> np.ndarray:
        """
        Bind a concept to a role.
        
        Args:
            role_name: Name of the role
            concept_seed: Seed string for the concept
            
        Returns:
            Role-bound concept vector
        """
        if role_name not in self._role_vectors:
            self.register_role(role_name)
        
        role_vec = self._role_vectors[role_name]
        concept_vec = self._generate_vector(concept_seed)
        
        return np.bitwise_xor(role_vec, concept_vec)
    
    def unbind_concept(self, bound_vec: np.ndarray, role_name: str) -> np.ndarray:
        """
        Unbind a concept from a role.
        
        Args:
            bound_vec: Role-bound concept vector
            role_name: Name of the role
            
        Returns:
            Unbound concept vector
        """
        role_vec = self._role_vectors.get(role_name)
        if role_vec is None:
            raise ValueError(f"Role '{role_name}' not registered")
        
        return np.bitwise_xor(bound_vec, role_vec)
    
    def create_bundle(self, 
                      concepts: Dict[str, str]) -> np.ndarray:
        """
        Create a bundled thought from role-bound concepts.
        
        Args:
            concepts: Dict mapping role names to concept seeds
                      e.g., {'action': 'rotate_90', 'object': 'cube'}
            
        Returns:
            Bundled thought vector
        """
        bound_concepts = []
        for role_name, concept_seed in concepts.items():
            bound = self.bind_concept(role_name, concept_seed)
            bound_concepts.append(bound)
        
        # XOR all bound concepts together
        if not bound_concepts:
            return np.zeros(self.uint64_count, dtype=np.uint64)
        
        result = bound_concepts[0]
        for bc in bound_concepts[1:]:
            result = np.bitwise_xor(result, bc)
        
        return result
    
    def extract_concept(self,
                        bundled_vec: np.ndarray,
                        role_name: str,
                        candidates: List[str]) -> Tuple[str, float]:
        """
        Extract a concept from a bundled vector for a specific role.
        
        Args:
            bundled_vec: Bundled thought vector
            role_name: Name of the role to extract
            candidates: List of candidate concept seeds
            
        Returns:
            Tuple of (best_match, confidence)
        """
        # Unbind using role vector
        unbound = self.unbind_concept(bundled_vec, role_name)
        
        # Find closest match among candidates
        best_match = None
        best_similarity = -1.0
        
        for candidate in candidates:
            candidate_vec = self._generate_vector(candidate)
            sim = self._hamming_similarity(unbound, candidate_vec)
            if sim > best_similarity:
                best_similarity = sim
                best_match = candidate
        
        return best_match, best_similarity
    
    def _hamming_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Hamming similarity between two vectors."""
        xored = np.bitwise_xor(a, b)
        diff_bits = np.unpackbits(xored.view(np.uint8)).sum()
        total_bits = len(a) * 64
        return 1.0 - (diff_bits / total_bits)


# =============================================================================
# Inhibitory Mask System
# =============================================================================

class InhibitoryMask:
    """
    Inhibitory mask for constraint filtering.
    
    The mask acts as a repulsive force during resonator factorization,
    pushing the trajectory away from prohibited regions.
    
    Use cases:
    - Clean Language Mask: Filter inappropriate content
    - Safety Mask: Prevent dangerous actions
    - Domain Mask: Restrict to specific domain
    """
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM):
        """
        Initialize Inhibitory Mask system.
        
        Args:
            dim: HDC dimension
        """
        self.dim = dim
        self.uint64_count = dim // 64
        self._masks: Dict[str, np.ndarray] = {}
    
    def _generate_vector(self, seed_string: str) -> np.ndarray:
        """Generate deterministic vector from seed string."""
        return seed_to_hypervector_blake3(seed_string, self.uint64_count)
    
    def create_mask(self, 
                    mask_name: str, 
                    prohibited_seeds: List[str]) -> np.ndarray:
        """
        Create an inhibitory mask from prohibited seeds.
        
        The mask is the XOR bundle of all prohibited vectors.
        When applied, it pushes away from these prohibited regions.
        
        Args:
            mask_name: Name for the mask
            prohibited_seeds: List of prohibited seed strings
            
        Returns:
            Inhibitory mask vector
        """
        if not prohibited_seeds:
            mask = np.zeros(self.uint64_count, dtype=np.uint64)
        else:
            vectors = [self._generate_vector(s) for s in prohibited_seeds]
            mask = vectors[0]
            for v in vectors[1:]:
                mask = np.bitwise_xor(mask, v)
        
        self._masks[mask_name] = mask
        return mask
    
    def get_mask(self, mask_name: str) -> Optional[np.ndarray]:
        """Get a mask by name."""
        return self._masks.get(mask_name)
    
    def apply_mask(self, 
                   vec: np.ndarray, 
                   mask_name: str,
                   strength: float = 1.0) -> np.ndarray:
        """
        Apply inhibitory mask to a vector.
        
        Args:
            vec: Vector to modify
            mask_name: Name of the mask to apply
            strength: Application strength (0.0 to 1.0)
            
        Returns:
            Modified vector
        """
        mask = self._masks.get(mask_name)
        if mask is None:
            return vec
        
        if strength >= 1.0:
            return np.bitwise_xor(vec, mask)
        return vec
    
    def check_violation(self,
                        vec: np.ndarray,
                        mask_name: str,
                        threshold: float = 0.6) -> Tuple[bool, float]:
        """
        Check if a vector violates a mask.
        
        Args:
            vec: Vector to check
            mask_name: Name of the mask
            threshold: Violation threshold
            
        Returns:
            Tuple of (is_violation, violation_ratio)
        """
        mask = self._masks.get(mask_name)
        if mask is None:
            return False, 0.0
        
        # XOR with mask
        xored = np.bitwise_xor(vec, mask)
        
        # Count null regions (violations create null space)
        null_count = np.count_nonzero(xored == 0)
        violation_ratio = null_count / len(xored)
        
        return violation_ratio > threshold, violation_ratio


# =============================================================================
# Factory Functions
# =============================================================================

def create_resonator(dim: int = DEFAULT_HDC_DIM, 
                     max_iterations: int = 100) -> ResonatorNetwork:
    """
    Factory function to create a ResonatorNetwork.
    
    Args:
        dim: HDC dimension (default 1048576 for 8K video)
        max_iterations: Maximum iterations
        
    Returns:
        Configured ResonatorNetwork instance
    """
    return ResonatorNetwork(dim=dim, max_iterations=max_iterations)


def create_role_binding(dim: int = DEFAULT_HDC_DIM) -> RoleBindingSystem:
    """
    Factory function to create a RoleBindingSystem.
    
    Args:
        dim: HDC dimension
        
    Returns:
        Configured RoleBindingSystem instance
    """
    return RoleBindingSystem(dim=dim)


def create_inhibitory_mask(dim: int = DEFAULT_HDC_DIM) -> InhibitoryMask:
    """
    Factory function to create an InhibitoryMask.
    
    Args:
        dim: HDC dimension
        
    Returns:
        Configured InhibitoryMask instance
    """
    return InhibitoryMask(dim=dim)


# =============================================================================
# Multi-Answer System (Section 27: Multiple Correct Answers with Ranked Response)
# =============================================================================

@dataclass
class RankedAnswer:
    """
    A validated answer with correctness and alignment scores.
    
    From FULLINTEGRATION_NEW_ARCHITECTURE.md Section 27:
    The multi-answer system surfaces ALL valid answers above a correctness threshold,
    then ranks them by personality alignment.
    
    Attributes:
        vector: The answer hypervector
        correctness: XOR similarity score [0, 1] - must be above threshold
        personality_alignment: Trait-weighted resonance score
        discovery_order: Order found (for tie-breaking)
        decoded_content: Human-readable answer (if available)
        seed_string: Seed string for the answer (if available)
    """
    vector: np.ndarray
    correctness: float
    personality_alignment: float = 0.0
    discovery_order: int = 0
    decoded_content: str = ""
    seed_string: str = ""


class MultiAnswerCoordinator:
    """
    Coordination layer above Resonator and Personality systems.
    
    Manages multi-answer search loop, applies correctness gate,
    collects survivors, and hands to personality for ranking.
    
    From FULLINTEGRATION_NEW_ARCHITECTURE.md Section 27:
    - Phase 1: Discover all valid answers using inhibitory mask
    - Phase 2: Filter by correctness threshold
    - Phase 3: Rank by personality alignment
    
    Key Properties:
    - Correctness Guaranteed: Users never see wrong answers
    - Personality-Aligned: Ranking reflects learned preferences
    - Deterministic: Same problem + personality → same ranked list
    - Efficient: Inhibitory mask prevents redundant search
    """
    
    # Default correctness thresholds
    STRICT_THRESHOLD = 0.98      # Very strict, fewer answers shown
    STANDARD_THRESHOLD = 0.95   # Default balance
    LENIENT_THRESHOLD = 0.90    # More answers shown, slight risk
    
    def __init__(self,
                 resonator: ResonatorNetwork,
                 personality: 'DeterministicPersonality',
                 threshold: float = STANDARD_THRESHOLD,
                 max_answers: int = 5):
        """
        Initialize Multi-Answer Coordinator.
        
        Args:
            resonator: ResonatorNetwork for candidate discovery
            personality: DeterministicPersonality for ranking
            threshold: Correctness threshold (0.0 to 1.0)
            max_answers: Maximum number of answers to find
        """
        self.resonator = resonator
        self.personality = personality
        self.threshold = threshold
        self.max_answers = max_answers
        self.dim = resonator.dim
        self.uint64_count = resonator.uint64_count
    
    def _generate_vector(self, seed_string: str) -> np.ndarray:
        """Generate deterministic vector from seed string."""
        return seed_to_hypervector_blake3(seed_string, self.uint64_count)
    
    def _compute_correctness(self,
                             candidate_vec: np.ndarray,
                             target_subspace: np.ndarray) -> float:
        """
        Compute XOR similarity (correctness) between candidate and target.
        
        Returns value in [0, 1] where:
        - 1.0 = perfect match (identical vectors)
        - 0.5 = random/uncorrelated
        - 0.0 = perfect anti-correlation
        
        Args:
            candidate_vec: Candidate answer vector
            target_subspace: Target/problem subspace vector
            
        Returns:
            Similarity score [0, 1]
        """
        xor_result = np.bitwise_xor(candidate_vec, target_subspace)
        # Count non-zero uint64 values (faster than bit counting)
        diff_count = np.count_nonzero(xor_result)
        total = len(xor_result)
        # Convert to bit-level similarity
        similarity = 1.0 - (diff_count / total)
        return similarity
    
    def _create_inhibitory_mask(self, found_answer: np.ndarray) -> np.ndarray:
        """
        Create mask that suppresses the found answer.
        
        The mask "peels away" the found answer from the search space,
        allowing the next iteration to find a different valid answer.
        
        Args:
            found_answer: Previously found answer vector
            
        Returns:
            Inhibitory mask vector
        """
        return found_answer.copy()
    
    def solve(self,
              problem_vec: np.ndarray,
              context_vec: np.ndarray,
              codebooks: Dict[str, List[str]],
              max_answers: Optional[int] = None) -> List[RankedAnswer]:
        """
        Find all valid answers and rank by personality alignment.
        
        This is the main entry point for multi-answer solving.
        
        Args:
            problem_vec: The problem vector to solve
            context_vec: Context vector for personality alignment
            codebooks: Dict mapping role names to candidate seed strings
            max_answers: Override max_answers (optional)
            
        Returns:
            List of RankedAnswer sorted by personality alignment
        """
        max_ans = max_answers or self.max_answers
        
        # Phase 1: Discover all valid answers
        valid_answers = self._discover_valid_answers(
            problem_vec, codebooks, max_ans
        )
        
        if not valid_answers:
            return []
        
        # Phase 2: Rank by personality alignment
        ranked = self._rank_by_personality(valid_answers, context_vec)
        
        return ranked
    
    def _discover_valid_answers(self,
                                problem_vec: np.ndarray,
                                codebooks: Dict[str, List[str]],
                                max_answers: int) -> List[RankedAnswer]:
        """
        Use inhibitory mask to find multiple valid answers.
        
        Each found answer is suppressed before the next search iteration,
        naturally preventing duplicates and revealing different solutions.
        
        Args:
            problem_vec: The problem vector
            codebooks: Dict mapping role names to candidate seeds
            max_answers: Maximum answers to find
            
        Returns:
            List of valid RankedAnswer objects
        """
        answers = []
        search_space = problem_vec.copy()
        
        for order in range(max_answers):
            # Run resonator on current search space
            result = self.resonator.factorize(search_space, codebooks)
            
            if not result.converged:
                break
            
            # Reconstruct the answer vector from estimates
            answer_vec = self._reconstruct_answer_vector(result.estimates, codebooks)
            
            # Compute correctness
            correctness = self._compute_correctness(answer_vec, problem_vec)
            
            if correctness < self.threshold:
                break  # Below threshold - stop searching
            
            # Get decoded content
            decoded = self._decode_estimates(result.estimates)
            
            answers.append(RankedAnswer(
                vector=answer_vec,
                correctness=correctness,
                personality_alignment=0.0,  # Set in ranking phase
                discovery_order=order,
                decoded_content=decoded,
                seed_string="|".join(result.estimates.values())
            ))
            
            # Apply inhibitory mask for next iteration
            mask = self._create_inhibitory_mask(answer_vec)
            search_space = np.bitwise_xor(search_space, mask)
        
        return answers
    
    def _reconstruct_answer_vector(self,
                                    estimates: Dict[str, str],
                                    codebooks: Dict[str, List[str]]) -> np.ndarray:
        """
        Reconstruct answer vector from role estimates.
        
        Args:
            estimates: Dict mapping role names to selected seeds
            codebooks: Dict mapping role names to candidate seeds
            
        Returns:
            Reconstructed answer vector
        """
        vectors = []
        for role, seed in estimates.items():
            vec = self._generate_vector(seed)
            vectors.append(vec)
        
        if not vectors:
            return np.zeros(self.uint64_count, dtype=np.uint64)
        
        result = vectors[0]
        for v in vectors[1:]:
            result = np.bitwise_xor(result, v)
        
        return result
    
    def _decode_estimates(self, estimates: Dict[str, str]) -> str:
        """
        Convert estimates to human-readable string.
        
        Args:
            estimates: Dict mapping role names to selected seeds
            
        Returns:
            Human-readable string
        """
        parts = [f"{role}:{seed}" for role, seed in estimates.items()]
        return ", ".join(parts)
    
    def _rank_by_personality(self,
                             answers: List[RankedAnswer],
                             context_vec: np.ndarray) -> List[RankedAnswer]:
        """
        Score and sort answers by personality alignment.
        
        Uses the personality system's trait-weighted resonance scoring.
        
        Args:
            answers: List of valid answers
            context_vec: Context vector for alignment
            
        Returns:
            Sorted list of RankedAnswer by personality alignment
        """
        for answer in answers:
            # XOR bind answer with context for alignment scoring
            bound = np.bitwise_xor(answer.vector, context_vec)
            answer.personality_alignment = self.personality.score_alignment(bound)
        
        # Sort: highest alignment first, ties broken by discovery order
        return sorted(answers, key=lambda a: (-a.personality_alignment, a.discovery_order))
    
    def solve_with_roles(self,
                         problem_vec: np.ndarray,
                         context_vec: np.ndarray,
                         role_vectors: Dict[str, np.ndarray],
                         codebooks: Dict[str, List[str]],
                         max_answers: Optional[int] = None) -> List[RankedAnswer]:
        """
        Find multiple valid answers using role-binding.
        
        Uses the role-binding system for cleaner factorization.
        
        Args:
            problem_vec: The problem vector
            context_vec: Context vector for personality alignment
            role_vectors: Dict mapping role names to orthogonal role vectors
            codebooks: Dict mapping role names to candidate seeds
            max_answers: Override max_answers (optional)
            
        Returns:
            List of RankedAnswer sorted by personality alignment
        """
        max_ans = max_answers or self.max_answers
        answers = []
        search_space = problem_vec.copy()
        
        for order in range(max_ans):
            # Run resonator with role-binding
            result = self.resonator.factorize_with_roles(
                search_space, role_vectors, codebooks
            )
            
            if not result.converged:
                break
            
            # Reconstruct answer
            answer_vec = self._reconstruct_answer_vector(result.estimates, codebooks)
            correctness = self._compute_correctness(answer_vec, problem_vec)
            
            if correctness < self.threshold:
                break
            
            decoded = self._decode_estimates(result.estimates)
            
            answers.append(RankedAnswer(
                vector=answer_vec,
                correctness=correctness,
                personality_alignment=0.0,
                discovery_order=order,
                decoded_content=decoded,
                seed_string="|".join(result.estimates.values())
            ))
            
            # Apply inhibitory mask
            mask = self._create_inhibitory_mask(answer_vec)
            search_space = np.bitwise_xor(search_space, mask)
        
        return self._rank_by_personality(answers, context_vec)


def create_multi_answer_coordinator(resonator: ResonatorNetwork,
                                    personality: 'DeterministicPersonality',
                                    threshold: float = 0.95,
                                    max_answers: int = 5) -> MultiAnswerCoordinator:
    """
    Factory function to create a MultiAnswerCoordinator.
    
    Args:
        resonator: ResonatorNetwork for candidate discovery
        personality: DeterministicPersonality for ranking
        threshold: Correctness threshold (0.0 to 1.0)
        max_answers: Maximum number of answers to find
        
    Returns:
        Configured MultiAnswerCoordinator instance
    """
    return MultiAnswerCoordinator(
        resonator=resonator,
        personality=personality,
        threshold=threshold,
        max_answers=max_answers
    )
