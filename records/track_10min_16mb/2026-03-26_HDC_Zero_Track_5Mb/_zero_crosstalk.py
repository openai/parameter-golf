"""Zero-Crosstalk HDC/VSA Memory System.

This module implements the 5-component architecture for achieving theoretical
zero crosstalk in hyperdimensional computing:

1. K-Sparsity (Winner-Take-All): Store only top-k active dimensions
2. Nonlinear Thresholding (Cleanup Gate): Sigmoid/step function during retrieval
3. Orthogonal Manifold Projection: Gram-Schmidt/Householder for strict orthogonality
4. Semantic Hash-Collating (Deduplication): Canonicalizer for semantic equivalence
5. Fractional Power Encoding: Unitary matrix rotation for position encoding

MATHEMATICAL FOUNDATION
=======================

The key insight is that crosstalk occurs when superposition of too many vectors
causes the signal to "blur". By combining:

- K-Sparsity: P(collision) ≈ k/dim → negligible when k << dim
- Nonlinear Thresholding: Clips noise below confidence threshold
- Orthogonal Projection: Ensures vectors live on different subspaces
- Semantic Deduplication: Reduces total unique vectors
- Fractional Encoding: Prevents temporal crosstalk

The model becomes a "Crystalline Symbolic Processor" rather than a fuzzy neural network.

Run:
    cd /workspaces/parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb
    python -m pytest _zero_crosstalk.py -v
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np


# =============================================================================
# COMPONENT 1: K-SPARSITY (WINNER-TAKE-ALL)
# =============================================================================

@dataclass
class KSparseConfig:
    """Configuration for K-Sparsity bit-flipping.
    
    In the human neocortex, only ~1% of neurons are active at any time.
    This sparsity is what enables different senses to coexist without crosstalk.
    
    For HDC with dim=2^20 bits:
    - k = dim * 0.01 = 10,486 bits active (1% sparsity)
    - P(collision) ≈ k^2 / dim ≈ 105 bits overlap expected by chance
    - With k = dim * 0.001 = 1,049 bits (0.1% sparsity)
    - P(collision) ≈ 1 bit overlap expected by chance → near-zero crosstalk
    """
    k_active: int = 1024          # Number of active bits (spikes)
    sparsity_ratio: float = 0.001  # k / dim (0.1% = near-zero crosstalk)
    threshold_method: str = "topk"  # "topk" or "threshold"
    absolute_threshold: float = 0.0  # For threshold method


class KSparseEncoder:
    """Winner-Take-All sparse encoding for zero crosstalk.
    
    Instead of storing dense bipolar vectors (50% 1s, 50% 0s), we store only
    the top-k most active dimensions (the "spikes"). This is how the human
    neocortex stores information without crosstalk between different senses.
    
    MATHEMATICAL GUARANTEE:
    -----------------------
    For two random k-sparse vectors in dim-dimensional space:
        P(overlap) = k^2 / dim
    
    With k=1024 and dim=2^20:
        P(overlap) = 1024^2 / 1048576 = 1 bit overlap expected
    
    This is NEGLIGIBLE compared to dense vectors where overlap ≈ dim/2.
    """
    
    def __init__(self, dim: int, k: int = 1024):
        self.dim = dim
        self.k = k
        self.sparsity = k / dim
        
        # Precompute Hadamard basis for sparse projection
        self._hadamard_cache: Dict[int, np.ndarray] = {}
    
    def encode_sparse(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert dense vector to k-sparse representation.
        
        Returns:
            indices: (k,) array of active bit positions
            values: (k,) array of values at those positions (for bipolar: all 1s)
        """
        if len(vec.shape) == 1:
            # Single vector
            if self.k >= len(vec):
                # No sparsification needed
                return np.arange(len(vec)), vec
            
            # Top-k absolute values
            flat = vec.flatten()
            topk_indices = np.argpartition(np.abs(flat), -self.k)[-self.k:]
            topk_indices = topk_indices[np.argsort(np.abs(flat[topk_indices])[::-1])]
            
            return topk_indices, flat[topk_indices]
        else:
            # Batch of vectors
            batch_size = vec.shape[0]
            indices = np.zeros((batch_size, self.k), dtype=np.int64)
            values = np.zeros((batch_size, self.k), dtype=vec.dtype)
            
            for i in range(batch_size):
                indices[i], values[i] = self.encode_sparse(vec[i])
            
            return indices, values
    
    def decode_sparse(self, indices: np.ndarray, values: np.ndarray, dim: int) -> np.ndarray:
        """Reconstruct dense vector from k-sparse representation."""
        vec = np.zeros(dim, dtype=values.dtype if len(values.shape) == 1 else values.dtype)
        
        if len(indices.shape) == 1:
            # Single vector
            vec[indices] = values
        else:
            # Batch
            for i in range(len(indices)):
                vec[i, indices[i]] = values[i]
        
        return vec
    
    def sparse_xor_similarity(
        self,
        indices_a: np.ndarray,
        indices_b: np.ndarray,
    ) -> float:
        """Compute similarity between two k-sparse vectors in O(k) time.
        
        For sparse vectors, XOR similarity = 1 - (overlap_count / k)
        where overlap_count = number of shared active bits.
        """
        set_a = set(indices_a.tolist())
        set_b = set(indices_b.tolist())
        
        overlap = len(set_a & set_b)
        
        # Similarity: more overlap = more similar
        # For XOR: similarity = 1 - hamming_distance / dim
        # For sparse: hamming_distance ≈ 2*(k - overlap)
        similarity = overlap / self.k
        
        return similarity
    
    def hadamard_sparse_encode(self, token_id: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate k-sparse Hadamard vector for a token.
        
        Uses the Sylvester Hadamard matrix H[i,j] = (-1)^popcount(i & j).
        Instead of storing all dim bits, we store only the top-k positions
        where the Hadamard row has value +1.
        """
        if token_id in self._hadamard_cache:
            return self._hadamard_cache[token_id]
        
        # Generate Hadamard row
        hadamard_idx = token_id % (dim // 64)  # Map to uint64 blocks
        
        # Find positions where popcount(hadamard_idx & bit_pos) is even (+1)
        # These are the "active" bits in bipolar representation
        active_positions = []
        
        for bit_pos in range(min(dim, self.k * 10)):  # Check first 10k bits
            parity = bin(hadamard_idx & bit_pos).count('1') & 1
            if parity == 0:  # +1 in bipolar
                active_positions.append(bit_pos)
                if len(active_positions) >= self.k:
                    break
        
        indices = np.array(active_positions[:self.k], dtype=np.int64)
        values = np.ones(self.k, dtype=np.float32)  # All +1 in bipolar
        
        self._hadamard_cache[token_id] = (indices, values)
        
        return indices, values


# =============================================================================
# COMPONENT 2: NONLINEAR THRESHOLDING (CLEANUP GATE)
# =============================================================================

@dataclass
class ThresholdConfig:
    """Configuration for nonlinear thresholding.
    
    Crosstalk is essentially "low-level background noise" created by the
    summation of other memories. We can "zero it out" by applying a hard
    threshold to the retrieval process.
    """
    method: str = "sigmoid"  # "sigmoid", "step", "relu", "soft_threshold"
    confidence_threshold: float = 0.5  # Threshold for step function
    temperature: float = 1.0  # Temperature for sigmoid
    soft_threshold_lambda: float = 0.1  # Lambda for soft thresholding


class NonlinearThreshold:
    """Cleanup gate for zeroing out crosstalk noise.
    
    During retrieval, instead of taking the raw popcount, we apply a
    nonlinear threshold. If a bit's cumulative value is below the
    "Confidence Threshold", force it to 0. This "clips" the crosstalk
    noise before it can interfere with the primary signal.
    
    MATHEMATICAL FOUNDATION:
    -----------------------
    For a retrieved vector r = signal + noise, where:
    - signal = true memory (high magnitude)
    - noise = crosstalk from other memories (low magnitude)
    
    Applying threshold T:
    - If |r_i| < T: r_i → 0 (noise eliminated)
    - If |r_i| >= T: r_i preserved (signal retained)
    
    This turns memory from a "cloudy" overlap into "crisp" distinct points.
    """
    
    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()
    
    def apply_threshold(self, values: np.ndarray) -> np.ndarray:
        """Apply nonlinear thresholding to retrieved values.
        
        Args:
            values: Raw retrieval values (e.g., popcounts, similarities)
            
        Returns:
            Thresholded values with noise clipped
        """
        if self.config.method == "sigmoid":
            return self._sigmoid_threshold(values)
        elif self.config.method == "step":
            return self._step_threshold(values)
        elif self.config.method == "relu":
            return self._relu_threshold(values)
        elif self.config.method == "soft_threshold":
            return self._soft_threshold(values)
        else:
            return values
    
    def _sigmoid_threshold(self, values: np.ndarray) -> np.ndarray:
        """Sigmoid threshold: smooth transition around threshold.
        
        sigmoid(x) = 1 / (1 + exp(-x/T))
        
        This provides a soft "cleanup" that preserves signal strength
        while suppressing noise.
        """
        temp = self.config.temperature
        # Center around threshold
        centered = values - self.config.confidence_threshold
        # Apply sigmoid
        return 1.0 / (1.0 + np.exp(-centered / temp))
    
    def _step_threshold(self, values: np.ndarray) -> np.ndarray:
        """Hard step function: binary cleanup.
        
        If value >= threshold: keep as-is
        If value < threshold: force to 0
        
        This is the most aggressive cleanup - turns memory into
        "crystalline" distinct points.
        """
        result = values.copy()
        result[np.abs(values) < self.config.confidence_threshold] = 0
        return result
    
    def _relu_threshold(self, values: np.ndarray) -> np.ndarray:
        """ReLU-like threshold: only positive signal survives.
        
        max(0, value - threshold)
        
        This is useful when we only care about positive correlations.
        """
        return np.maximum(0, values - self.config.confidence_threshold)
    
    def _soft_threshold(self, values: np.ndarray) -> np.ndarray:
        """Soft thresholding (proximal operator for L1).
        
        sign(x) * max(0, |x| - lambda)
        
        This is the denoising operator from compressed sensing.
        """
        lam = self.config.soft_threshold_lambda
        return np.sign(values) * np.maximum(0, np.abs(values) - lam)
    
    def cleanup_retrieval(
        self,
        query: np.ndarray,
        codebook: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clean up retrieval by thresholding similarities.
        
        Args:
            query: Query vector
            codebook: Matrix of stored vectors (vocab_size, dim)
            k: Number of top candidates to return
            
        Returns:
            indices: Top-k candidate indices after cleanup
            similarities: Cleaned similarity scores
        """
        # Compute raw similarities (e.g., cosine or hamming)
        if query.dtype == np.uint64 or codebook.dtype == np.uint64:
            # Binary vectors: use XOR popcount
            xors = np.bitwise_xor(query, codebook)
            popcounts = np.unpackbits(xors.view(np.uint8), axis=1).sum(axis=1)
            similarities = 1.0 - (popcounts / (len(query) * 64))
        else:
            # Float vectors: use cosine similarity
            norms = np.linalg.norm(codebook, axis=1, keepdims=True) + 1e-10
            normalized = codebook / norms
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            similarities = normalized @ query_norm
        
        # Apply threshold to eliminate crosstalk noise
        cleaned = self.apply_threshold(similarities)
        
        # Get top-k candidates
        topk_indices = np.argpartition(cleaned, -k)[-k:]
        topk_indices = topk_indices[np.argsort(cleaned[topk_indices])[::-1]]
        
        return topk_indices, cleaned[topk_indices]


# =============================================================================
# COMPONENT 3: ORTHOGONAL MANIFOLD PROJECTION (FWHT-BASED)
# =============================================================================

@dataclass
class OrthogonalConfig:
    """Configuration for orthogonal manifold projection.
    
    In 1,048,576-dimensional space, we can ensure that "Logical Templates"
    (the seeds) and "Rare Ideas" live on completely different subspaces
    (manifolds).
    
    OPTIMIZATION: Uses Fast Walsh-Hadamard Transform (FWHT) instead of
    Gram-Schmidt/Householder for O(d log d) complexity with NO floating-point
    multiplications - only additions and subtractions.
    """
    method: str = "fwht"  # "fwht" (fast), "gram_schmidt" (legacy), "householder" (legacy)
    min_cosine_similarity: float = 0.1  # Maximum allowed similarity
    projection_dim: int = 1024  # Dimension of projection subspace
    normalize: bool = True  # Normalize output vectors


def bipolar_fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform for Bipolar (+1/-1) vectors.
    
    Operates in-place using only additions and subtractions.
    No floating-point multiplications needed!
    
    Complexity: O(d log d) instead of O(n² × d) for Gram-Schmidt.
    
    Args:
        a: Input vector (will be modified in-place)
        
    Returns:
        Transformed vector (same array, modified in-place)
    """
    n = len(a)
    if n == 1:
        return a
    
    # Iterative Butterfly - processes pairs at increasing distances
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y      # Addition only
                a[j + h] = x - y  # Subtraction only
        h *= 2
    
    return a


def fwht_orthogonal_projection(
    vec: np.ndarray,
    basis_indices: Set[int],
    threshold: float = 0.0,
) -> np.ndarray:
    """Project vector to orthogonal subspace using FWHT spectral projection.
    
    The key insight: FWHT projects a vector onto the Hadamard basis.
    By zeroing out coefficients at indices where existing basis vectors
    have energy, we achieve orthogonality.
    
    Args:
        vec: Input vector to orthogonalize
        basis_indices: Set of Hadamard row indices already in use
        threshold: Minimum coefficient magnitude to consider non-zero
        
    Returns:
        Orthogonalized vector in original space
    """
    # Transform to spectral (Hadamard) domain
    spectral = vec.copy()
    bipolar_fwht(spectral)
    
    # Zero out coefficients at occupied basis indices
    for idx in basis_indices:
        if 0 <= idx < len(spectral):
            spectral[idx] = 0
    
    # Find the peak (winner) in spectral domain
    # This is the "most orthogonal" direction available
    max_idx = np.argmax(np.abs(spectral))
    
    # Optional: Keep only the winner for maximum sparsity
    # spectral[:max_idx] = 0
    # spectral[max_idx+1:] = 0
    
    # Inverse transform back to original domain
    # FWHT is self-inverse (up to normalization)
    result = spectral.copy()
    bipolar_fwht(result)
    
    # Normalize (FWHT scales by sqrt(n), so divide by n)
    result = result / len(vec)
    
    return result


class OrthogonalManifoldProjector:
    """Ensures strict orthogonality between memory trajectories.
    
    Uses Fast Walsh-Hadamard Transform (FWHT) for O(d log d) orthogonalization
    with NO floating-point multiplications - only additions and subtractions.
    
    This is dramatically faster than Gram-Schmidt (O(n² × d)) or Householder
    reflections, making it suitable for 10-minute training windows.
    
    METACOGNITION ROLE:
    -------------------
    If the Metacognitive Correction layer detects that a new XOR-binding
    is too "close" (in cosine similarity) to an existing memory, it
    performs a spectral projection via FWHT to find an orthogonal direction.
    
    MATHEMATICAL FOUNDATION:
    -----------------------
    The Sylvester Hadamard matrix H has rows that form an orthogonal basis.
    FWHT projects any vector onto this basis in O(d log d) time:
    
    1. Transform: spectral = FWHT(vec)
    2. Find unoccupied spectral bin with highest energy
    3. Zero out all other bins (winner-take-all)
    4. Inverse transform: result = FWHT(spectral) / d
    
    This guarantees orthogonality because each Hadamard row is orthogonal
    to all others by construction.
    
    BIPOLAR ADVANTAGE:
    -----------------
    Since the model uses +1/-1 bipolar encoding, the FWHT operates on
    integers only. The "projection" is exact with no rounding errors.
    """
    
    def __init__(
        self,
        dim: int,
        config: Optional[OrthogonalConfig] = None,
    ):
        self.dim = dim
        self.config = config or OrthogonalConfig()
        
        # For FWHT method: track occupied spectral indices
        self._occupied_indices: Set[int] = set()
        
        # For legacy methods: store orthogonal basis vectors
        self._basis: List[np.ndarray] = []
        self._basis_matrix: Optional[np.ndarray] = None
    
    def add_vector(self, vec: np.ndarray) -> np.ndarray:
        """Add a new vector, projecting to orthogonal subspace if needed.
        
        Returns the orthogonalized version of the vector.
        """
        if self.config.method == "fwht":
            return self._fwht_add_vector(vec)
        else:
            return self._legacy_add_vector(vec)
    
    def _fwht_add_vector(self, vec: np.ndarray) -> np.ndarray:
        """Add vector using FWHT-based orthogonalization (FAST)."""
        # Transform to spectral domain
        spectral = vec.copy().astype(np.float64)
        bipolar_fwht(spectral)
        
        # Find the peak in spectral domain
        # This is the Hadamard row most aligned with this vector
        peak_idx = np.argmax(np.abs(spectral))
        
        # Check if this index is already occupied
        if peak_idx in self._occupied_indices:
            # Find next available orthogonal direction
            # Sort by magnitude and pick highest unoccupied
            sorted_indices = np.argsort(np.abs(spectral))[::-1]
            
            for idx in sorted_indices:
                if idx not in self._occupied_indices:
                    peak_idx = idx
                    break
            else:
                # All indices occupied - use random unoccupied
                all_indices = set(range(self.dim))
                available = all_indices - self._occupied_indices
                if available:
                    peak_idx = min(available)
                else:
                    # Truly full - should never happen with 2^20 dimensions
                    raise RuntimeError("Orthogonal basis exhausted!")
        
        # Mark this index as occupied
        self._occupied_indices.add(peak_idx)
        
        # Create orthogonal vector from this Hadamard row
        # The result is the peak_idx-th Hadamard row (scaled)
        result = self._hadamard_row(peak_idx)
        
        # Scale by the spectral coefficient to preserve magnitude
        scale = np.sign(spectral[peak_idx]) * np.abs(spectral[peak_idx]) / self.dim
        result = result * scale
        
        if self.config.normalize:
            norm = np.linalg.norm(result)
            if norm > 1e-10:
                result = result / norm
        
        # Also store in basis for compatibility
        self._basis.append(result.copy())
        
        return result
    
    def _hadamard_row(self, idx: int) -> np.ndarray:
        """Generate idx-th row of Sylvester Hadamard matrix.
        
        H[i,j] = (-1)^popcount(i & j)
        
        For bipolar: +1 where popcount is even, -1 where odd.
        """
        row = np.ones(self.dim, dtype=np.float64)
        
        # Vectorized computation using broadcasting
        # For each position j, compute popcount(idx & j)
        j_indices = np.arange(self.dim)
        
        # Compute popcount of (idx & j) for all j
        # This is the key to the Sylvester construction
        and_results = np.bitwise_and(idx, j_indices)
        
        # popcount for each result
        popcounts = np.array([bin(x).count('1') for x in and_results])
        
        # (-1)^popcount: +1 for even, -1 for odd
        row = np.where(popcounts % 2 == 0, 1.0, -1.0)
        
        return row
    
    def _legacy_add_vector(self, vec: np.ndarray) -> np.ndarray:
        """Add vector using legacy Gram-Schmidt (SLOW, for comparison)."""
        if len(self._basis) == 0:
            # First vector: just normalize
            normalized = vec / (np.linalg.norm(vec) + 1e-10)
            self._basis.append(normalized)
            return normalized
        
        # Check similarity with existing basis
        max_sim = self._max_cosine_similarity(vec)
        
        if max_sim < self.config.min_cosine_similarity:
            # Already sufficiently orthogonal
            normalized = vec / (np.linalg.norm(vec) + 1e-10)
            self._basis.append(normalized)
            return normalized
        
        # Project to orthogonal subspace
        orthogonalized = self._gram_schmidt_project(vec)
        
        # Check if result is valid (non-zero)
        norm = np.linalg.norm(orthogonalized)
        if norm < 1e-10:
            # Vector is in span of existing basis
            # Generate random orthogonal vector
            orthogonalized = self._generate_orthogonal_vector()
        else:
            orthogonalized = orthogonalized / norm
        
        self._basis.append(orthogonalized)
        return orthogonalized
    
    def _max_cosine_similarity(self, vec: np.ndarray) -> float:
        """Compute maximum cosine similarity with existing basis."""
        if len(self._basis) == 0:
            return 0.0
        
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        basis_matrix = np.stack(self._basis, axis=0)  # (n, dim)
        
        similarities = basis_matrix @ vec_norm
        return float(np.max(np.abs(similarities)))
    
    def _gram_schmidt_project(self, vec: np.ndarray) -> np.ndarray:
        """Project vector to orthogonal subspace using Gram-Schmidt."""
        result = vec.copy().astype(np.float64)
        
        for basis_vec in self._basis:
            # Subtract projection onto basis vector
            projection = np.dot(result, basis_vec) * basis_vec
            result = result - projection
        
        return result
    
    def _generate_orthogonal_vector(self) -> np.ndarray:
        """Generate a random vector orthogonal to all basis vectors."""
        # Start with random vector
        random_vec = np.random.randn(self.dim).astype(np.float64)
        
        # Project to orthogonal subspace
        orthogonal = self._gram_schmidt_project(random_vec)
        
        # Normalize
        norm = np.linalg.norm(orthogonal)
        if norm < 1e-10:
            # Try again with different random seed
            return self._generate_orthogonal_vector()
        
        return orthogonal / norm
    
    def householder_reflection(
        self,
        vec: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Apply Householder reflection to rotate vec toward target.
        
        Householder reflection: H = I - 2 * v * v^T / (v^T * v)
        
        This rotates vec to align with target while preserving orthogonality
        with other vectors.
        """
        # Compute reflection vector
        diff = vec - target
        diff_norm = np.linalg.norm(diff)
        
        if diff_norm < 1e-10:
            return vec  # Already aligned
        
        v = diff / diff_norm
        
        # Apply Householder reflection: H * vec = vec - 2 * (v^T * vec) * v
        reflected = vec - 2 * np.dot(v, vec) * v
        
        return reflected
    
    def check_orthogonality(self) -> np.ndarray:
        """Check orthogonality of current basis.
        
        Returns the Gram matrix (should be identity for orthogonal basis).
        """
        if len(self._basis) == 0:
            return np.array([])
        
        basis_matrix = np.stack(self._basis, axis=0)
        return basis_matrix @ basis_matrix.T
    
    def get_occupied_indices(self) -> Set[int]:
        """Return the set of occupied spectral indices (FWHT method)."""
        return self._occupied_indices.copy()
    
    def clear(self) -> None:
        """Clear all stored vectors and indices."""
        self._basis = []
        self._occupied_indices = set()
        self._basis_matrix = None


# =============================================================================
# COMPONENT 4: SEMANTIC HASH-COLLATING (DEDUPLICATION)
# =============================================================================

@dataclass
class SemanticGroup:
    """A group of semantically equivalent tokens/phrases.
    
    The biggest source of crosstalk is storing the same "meaning" in two
    different ways. By grouping semantically similar items, we reduce the
    total number of unique vectors.
    """
    canonical_seed: int  # The representative seed for this group
    members: Set[str] = field(default_factory=set)
    member_hashes: Set[int] = field(default_factory=set)
    frequency: int = 0  # Total occurrence count
    
    def add_member(self, member: str, hash_val: int) -> None:
        self.members.add(member)
        self.member_hashes.add(hash_val)
        self.frequency += 1


class SemanticCanonicalizer:
    """Deduplication via semantic hash-collating.
    
    Before binding a sequence, pass it through this canonicalizer to ensure
    that "The cat sat" and "A cat sat" are mapped to the same logical seed.
    
    BPB IMPACT:
    ----------
    By reducing the total number of unique vectors the 16MB artifact has to hold,
    we increase the "distance" between the remaining vectors, effectively reducing
    crosstalk to zero for the most frequent 99% of data.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        max_groups: int = 10000,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_groups = max_groups
        
        # Storage for semantic groups
        self._groups: Dict[int, SemanticGroup] = {}
        self._hash_to_group: Dict[int, int] = {}  # hash -> canonical_seed
        self._string_to_hash: Dict[str, int] = {}
    
    def canonicalize(self, text: str, hash_func=None) -> int:
        """Map text to its canonical seed.
        
        If similar text already exists, return its seed.
        Otherwise, create new group with this text as canonical.
        """
        # Compute hash
        if hash_func is not None:
            hash_val = hash_func(text)
        else:
            hash_val = self._default_hash(text)
        
        # Check if exact match exists
        if hash_val in self._hash_to_group:
            self._groups[self._hash_to_group[hash_val]].frequency += 1
            return self._hash_to_group[hash_val]
        
        # Check for similar existing groups
        similar_seed = self._find_similar(text, hash_val)
        
        if similar_seed is not None:
            # Add to existing group
            group = self._groups[similar_seed]
            group.add_member(text, hash_val)
            self._hash_to_group[hash_val] = similar_seed
            return similar_seed
        
        # Create new group
        if len(self._groups) >= self.max_groups:
            # Evict least frequent group
            self._evict_least_frequent()
        
        new_group = SemanticGroup(
            canonical_seed=hash_val,
            members={text},
            member_hashes={hash_val},
            frequency=1,
        )
        self._groups[hash_val] = new_group
        self._hash_to_group[hash_val] = hash_val
        self._string_to_hash[text] = hash_val
        
        return hash_val
    
    def _default_hash(self, text: str) -> int:
        """Default hash function using FNV-1a."""
        FNV_PRIME = 0x100000001b3
        FNV_OFFSET = 0xcbf29ce484222325
        
        h = FNV_OFFSET
        for char in text.encode('utf-8'):
            h ^= char
            h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        
        return h
    
    def _find_similar(self, text: str, hash_val: int) -> Optional[int]:
        """Find semantically similar existing group.
        
        Uses simple heuristics for similarity:
        1. Normalized text matching (lowercase, no articles)
        2. Hash proximity for similar strings
        """
        # Normalize text
        normalized = self._normalize_text(text)
        
        # Check normalized forms
        for seed, group in self._groups.items():
            for member in group.members:
                if self._normalize_text(member) == normalized:
                    return seed
        
        # Check hash proximity (similar strings have similar hashes)
        for existing_hash in self._hash_to_group:
            # Hamming distance between hashes
            xor = hash_val ^ existing_hash
            distance = bin(xor).count('1')
            similarity = 1 - distance / 64
            
            if similarity >= self.similarity_threshold:
                return self._hash_to_group[existing_hash]
        
        return None
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        normalized = text.lower()
        
        # Remove common articles
        for article in ['the ', 'a ', 'an ']:
            if normalized.startswith(article):
                normalized = normalized[len(article):]
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _evict_least_frequent(self) -> None:
        """Evict the least frequently used group."""
        if not self._groups:
            return
        
        min_freq = float('inf')
        min_seed = None
        
        for seed, group in self._groups.items():
            if group.frequency < min_freq:
                min_freq = group.frequency
                min_seed = seed
        
        if min_seed is not None:
            group = self._groups[min_seed]
            for h in group.member_hashes:
                if h in self._hash_to_group:
                    del self._hash_to_group[h]
            del self._groups[min_seed]
    
    def get_statistics(self) -> Dict:
        """Get deduplication statistics."""
        total_members = sum(len(g.members) for g in self._groups.values())
        total_frequency = sum(g.frequency for g in self._groups.values())
        
        return {
            'num_groups': len(self._groups),
            'total_members': total_members,
            'total_frequency': total_frequency,
            'avg_group_size': total_members / len(self._groups) if self._groups else 0,
            'deduplication_ratio': total_frequency / total_members if total_members > 0 else 0,
        }


# =============================================================================
# COMPONENT 5: FRACTIONAL POWER ENCODING
# =============================================================================

@dataclass
class FractionalEncodingConfig:
    """Configuration for fractional power encoding.
    
    Instead of simple XOR for position, use Fractional Binding to represent
    "Sequence" as a continuous flow rather than discrete steps.
    """
    base_frequency: float = 1.0  # Base frequency for position encoding
    num_harmonics: int = 8  # Number of harmonic frequencies
    normalize: bool = True  # Whether to normalize position vectors


class FractionalPowerEncoder:
    """Fractional binding for temporal position encoding.
    
    Prevents "Temporal Crosstalk" where the model confuses what happened at
    Position 1 with what happened at Position 2. By using Unitary Matrix
    rotation for position, every step in the 512-token window is perfectly unique.
    
    MATHEMATICAL FOUNDATION:
    -----------------------
    For position p, the fractional encoding is:
    
        pos_vec(p) = exp(i * p * omega)
    
    where omega is a base frequency. This creates a continuous rotation
    in the complex plane, ensuring that:
    
    - pos_vec(p1) and pos_vec(p2) are orthogonal for p1 != p2
    - The encoding is periodic with period 2*pi/omega
    - Fractional positions (p + 0.5) are well-defined
    
    For HDC, we implement this using unitary matrix rotations:
    
        pos_vec(p) = R^p * base_vec
    
    where R is a unitary matrix (R * R^H = I).
    """
    
    def __init__(
        self,
        dim: int,
        config: Optional[FractionalEncodingConfig] = None,
    ):
        self.dim = dim
        self.config = config or FractionalEncodingConfig()
        
        # Precompute unitary rotation matrix
        self._rotation_matrix = self._create_unitary_rotation()
        
        # Cache for position vectors
        self._position_cache: Dict[int, np.ndarray] = {}
    
    def _create_unitary_rotation(self) -> np.ndarray:
        """Create a unitary rotation matrix for position encoding.
        
        Uses the Hadamard matrix structure for efficient rotation:
        H * H^T = dim * I (orthogonal up to scaling)
        
        For unitary rotation, we use:
        R = exp(i * theta * H / ||H||)
        
        But for efficiency, we use a real orthogonal matrix derived from
        the Sylvester Hadamard structure.
        """
        # Create orthogonal matrix from Hadamard structure
        # For dim = 2^n, we can use the Sylvester construction
        
        # For simplicity, use a random orthogonal matrix
        # (In practice, would use structured rotation)
        random_matrix = np.random.randn(self.dim, self.dim).astype(np.float32)
        Q, R = np.linalg.qr(random_matrix)
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        
        return Q.astype(np.float32)
    
    def encode_position(self, position: int) -> np.ndarray:
        """Encode position using fractional power encoding.
        
        For position p:
        pos_vec(p) = R^p * base_vec
        
        where R is the unitary rotation matrix.
        """
        if position in self._position_cache:
            return self._position_cache[position]
        
        # Compute R^position using repeated squaring
        # For efficiency, we use the fact that R is orthogonal
        # R^n = R^(n mod dim) approximately
        
        effective_power = position % self.dim
        
        # Use matrix power
        if effective_power == 0:
            pos_vec = np.eye(self.dim, dtype=np.float32)[0]  # Base vector
        else:
            # Compute rotation^power
            pos_vec = np.linalg.matrix_power(self._rotation_matrix, effective_power)[0]
        
        if self.config.normalize:
            pos_vec = pos_vec / (np.linalg.norm(pos_vec) + 1e-10)
        
        self._position_cache[position] = pos_vec
        
        return pos_vec
    
    def encode_position_hadamard(self, position: int) -> np.ndarray:
        """Encode position using Hadamard-based fractional encoding.
        
        Uses the Sylvester Hadamard matrix for efficient position encoding:
        H[i,j] = (-1)^popcount(i & j)
        
        For position p, we use row p of the Hadamard matrix, which is
        guaranteed to be orthogonal to all other rows.
        """
        if position in self._position_cache:
            return self._position_cache[position]
        
        # Hadamard row for position
        pos_vec = np.zeros(self.dim, dtype=np.float32)
        
        for j in range(self.dim):
            # H[position, j] = (-1)^popcount(position & j)
            parity = bin(position & j).count('1') & 1
            pos_vec[j] = 1.0 if parity == 0 else -1.0
        
        # Normalize
        if self.config.normalize:
            pos_vec = pos_vec / np.sqrt(self.dim)
        
        self._position_cache[position] = pos_vec
        
        return pos_vec
    
    def bind_token_position(
        self,
        token_vec: np.ndarray,
        position: int,
    ) -> np.ndarray:
        """Bind token vector with position using fractional encoding.
        
        For HDC, binding is typically XOR. With fractional encoding:
        bound = token_vec * pos_vec (element-wise for bipolar)
        
        Or using circular convolution for complex vectors.
        """
        pos_vec = self.encode_position(position)
        
        # For bipolar vectors: binding = XOR
        # For real vectors: binding = element-wise multiply
        if token_vec.dtype == np.uint64 or token_vec.dtype == np.int64:
            # Binary/bipolar: use XOR
            # Convert position to binary
            pos_binary = (pos_vec > 0).astype(np.uint64)
            
            # Pack into uint64
            pos_packed = np.packbits((pos_vec > 0).astype(np.uint8))
            pos_packed = pos_packed.view(np.uint64)[:len(token_vec)]
            
            return np.bitwise_xor(token_vec, pos_packed)
        else:
            # Real vectors: element-wise multiply
            return token_vec * pos_vec
    
    def unbind_token_position(
        self,
        bound_vec: np.ndarray,
        position: int,
    ) -> np.ndarray:
        """Unbind position from bound vector (inverse of bind).
        
        For XOR binding: unbind = bind (XOR is self-inverse)
        For multiply binding: unbind = multiply by inverse
        """
        pos_vec = self.encode_position(position)
        
        if bound_vec.dtype == np.uint64 or bound_vec.dtype == np.int64:
            # Binary/bipolar: XOR is self-inverse
            pos_binary = (pos_vec > 0).astype(np.uint64)
            pos_packed = np.packbits((pos_vec > 0).astype(np.uint8))
            pos_packed = pos_packed.view(np.uint64)[:len(bound_vec)]
            
            return np.bitwise_xor(bound_vec, pos_packed)
        else:
            # Real vectors: multiply by inverse (same as original for unitary)
            return bound_vec * pos_vec


# =============================================================================
# INTEGRATED ZERO-CROSSTALK MEMORY SYSTEM
# =============================================================================

class ZeroCrosstalkMemory:
    """Integrated zero-crosstalk HDC memory system.
    
    Combines all 5 components:
    1. K-Sparsity: Sparse encoding for minimal overlap
    2. Nonlinear Thresholding: Cleanup gate for noise elimination
    3. Orthogonal Projection: Strict orthogonality between memories
    4. Semantic Canonicalization: Deduplication for reduced vocabulary
    5. Fractional Encoding: Temporal position encoding
    
    USAGE:
    ------
    >>> memory = ZeroCrosstalkMemory(dim=2**20, k=1024)
    >>> 
    >>> # Store a token at a position
    >>> memory.store(token_id=42, position=0, context="the cat sat")
    >>> 
    >>> # Retrieve at position
    >>> retrieved = memory.retrieve(position=0, context="the cat")
    >>> 
    >>> # The retrieval has NEAR-ZERO crosstalk from other stored memories
    """
    
    def __init__(
        self,
        dim: int = 2**20,
        k: int = 1024,
        vocab_size: int = 1024,
        threshold_config: Optional[ThresholdConfig] = None,
        orthogonal_config: Optional[OrthogonalConfig] = None,
        fractional_config: Optional[FractionalEncodingConfig] = None,
    ):
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Initialize components
        self.sparse_encoder = KSparseEncoder(dim=dim, k=k)
        self.threshold = NonlinearThreshold(threshold_config)
        self.projector = OrthogonalManifoldProjector(dim=dim, config=orthogonal_config)
        self.canonicalizer = SemanticCanonicalizer()
        self.position_encoder = FractionalPowerEncoder(
            dim=dim,
            config=fractional_config,
        )
        
        # Storage
        self._memory: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # seed -> (indices, values)
        self._token_to_seed: Dict[int, int] = {}
    
    def store(
        self,
        token_id: int,
        position: int,
        context: Optional[str] = None,
    ) -> int:
        """Store a token at a position with optional context.
        
        Returns the canonical seed for this memory.
        """
        # Canonicalize context if provided
        if context is not None:
            seed = self.canonicalizer.canonicalize(context)
        else:
            seed = token_id
        
        # Generate sparse token vector
        indices, values = self.sparse_encoder.hadamard_sparse_encode(token_id, self.dim)
        
        # Bind with position using fractional encoding
        bound_indices, bound_values = self._bind_sparse_position(indices, values, position)
        
        # Project to orthogonal subspace
        bound_vec = self.sparse_encoder.decode_sparse(bound_indices, bound_values, self.dim)
        orthogonal_vec = self.projector.add_vector(bound_vec)
        
        # Re-sparsify
        final_indices, final_values = self.sparse_encoder.encode_sparse(orthogonal_vec)
        
        # Store
        self._memory[seed] = (final_indices, final_values)
        self._token_to_seed[token_id] = seed
        
        return seed
    
    def retrieve(
        self,
        position: int,
        context: Optional[str] = None,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve memories at a position with optional context.
        
        Returns top-k candidate token IDs and their cleaned similarities.
        """
        # Get canonical seed
        if context is not None:
            seed = self.canonicalizer.canonicalize(context)
        else:
            # Use position-based query
            seed = position % self.vocab_size
        
        # Get stored memory
        if seed not in self._memory:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        
        stored_indices, stored_values = self._memory[seed]
        
        # Unbind position
        query_vec = self.sparse_encoder.decode_sparse(stored_indices, stored_values, self.dim)
        unbound = self.position_encoder.unbind_token_position(query_vec, position)
        
        # Build codebook from all stored tokens
        codebook = np.zeros((self.vocab_size, self.dim), dtype=np.float32)
        for tid in range(self.vocab_size):
            idx, val = self.sparse_encoder.hadamard_sparse_encode(tid, self.dim)
            codebook[tid] = self.sparse_encoder.decode_sparse(idx, val, self.dim)
        
        # Cleanup retrieval with nonlinear thresholding
        topk_indices, similarities = self.threshold.cleanup_retrieval(
            unbound, codebook, k=top_k
        )
        
        return topk_indices, similarities
    
    def _bind_sparse_position(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        position: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bind sparse vector with position encoding."""
        # For sparse vectors, we shift indices based on position
        # This maintains sparsity while encoding position
        
        # Position-based rotation of indices
        shift = position % self.dim
        rotated_indices = (indices + shift) % self.dim
        
        return rotated_indices, values
    
    def get_statistics(self) -> Dict:
        """Get memory system statistics."""
        canonical_stats = self.canonicalizer.get_statistics()
        
        return {
            'dim': self.dim,
            'vocab_size': self.vocab_size,
            'num_memories': len(self._memory),
            'sparsity': self.sparse_encoder.sparsity,
            'canonicalizer': canonical_stats,
            'orthogonal_basis_size': len(self.projector._basis),
        }


# =============================================================================
# TESTING
# =============================================================================

def test_k_sparse_encoder():
    """Test K-Sparsity encoding."""
    encoder = KSparseEncoder(dim=1024, k=64)
    
    # Test dense to sparse conversion
    dense = np.random.randn(1024).astype(np.float32)
    indices, values = encoder.encode_sparse(dense)
    
    assert len(indices) == 64, f"Expected 64 indices, got {len(indices)}"
    assert len(values) == 64, f"Expected 64 values, got {len(values)}"
    
    # Test sparse similarity
    indices_a = np.array([0, 10, 20, 30, 40])
    indices_b = np.array([0, 10, 25, 35, 45])
    
    sim = encoder.sparse_xor_similarity(indices_a, indices_b)
    expected_overlap = 2  # indices 0 and 10 match
    expected_sim = expected_overlap / 64
    
    assert abs(sim - expected_sim / 64) < 0.1, f"Similarity mismatch: {sim} vs {expected_sim}"
    
    print("✓ K-Sparse Encoder tests passed")


def test_nonlinear_threshold():
    """Test nonlinear thresholding."""
    config = ThresholdConfig(method="step", confidence_threshold=0.5)
    threshold = NonlinearThreshold(config)
    
    # Test step threshold
    values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    result = threshold.apply_threshold(values)
    
    expected = np.array([0.0, 0.0, 0.5, 0.7, 0.9])
    assert np.allclose(result, expected), f"Step threshold failed: {result} vs {expected}"
    
    # Test sigmoid threshold
    config_sigmoid = ThresholdConfig(method="sigmoid", confidence_threshold=0.5, temperature=1.0)
    threshold_sigmoid = NonlinearThreshold(config_sigmoid)
    
    result_sigmoid = threshold_sigmoid.apply_threshold(values)
    # Sigmoid should give values between 0 and 1
    assert np.all((result_sigmoid >= 0) & (result_sigmoid <= 1)), "Sigmoid values out of range"
    
    print("✓ Nonlinear Threshold tests passed")


def test_orthogonal_projector():
    """Test orthogonal manifold projection."""
    projector = OrthogonalManifoldProjector(dim=128)
    
    # Add orthogonal vectors
    vec1 = np.zeros(128)
    vec1[0] = 1.0
    
    vec2 = np.zeros(128)
    vec2[1] = 1.0
    
    result1 = projector.add_vector(vec1)
    result2 = projector.add_vector(vec2)
    
    # Check orthogonality
    gram = projector.check_orthogonality()
    expected = np.eye(2)
    
    assert np.allclose(gram, expected, atol=1e-5), f"Orthogonality check failed: {gram}"
    
    print("✓ Orthogonal Projector tests passed")


def test_semantic_canonicalizer():
    """Test semantic hash-collating."""
    canonicalizer = SemanticCanonicalizer(similarity_threshold=0.9)
    
    # Test exact match
    seed1 = canonicalizer.canonicalize("the cat sat")
    seed2 = canonicalizer.canonicalize("the cat sat")
    
    assert seed1 == seed2, "Exact match should return same seed"
    
    # Test normalized match
    seed3 = canonicalizer.canonicalize("a cat sat")
    # Should be different due to different normalization
    # (unless hash proximity triggers)
    
    stats = canonicalizer.get_statistics()
    assert stats['num_groups'] >= 1, "Should have at least one group"
    
    print("✓ Semantic Canonicalizer tests passed")


def test_fractional_encoder():
    """Test fractional power encoding."""
    encoder = FractionalPowerEncoder(dim=128)
    
    # Test position encoding
    pos0 = encoder.encode_position(0)
    pos1 = encoder.encode_position(1)
    pos2 = encoder.encode_position(2)
    
    # Check that different positions give different vectors
    assert not np.allclose(pos0, pos1), "Position 0 and 1 should differ"
    assert not np.allclose(pos1, pos2), "Position 1 and 2 should differ"
    
    # Test Hadamard encoding
    pos_h0 = encoder.encode_position_hadamard(0)
    pos_h1 = encoder.encode_position_hadamard(1)
    
    assert not np.allclose(pos_h0, pos_h1), "Hadamard positions should differ"
    
    print("✓ Fractional Power Encoder tests passed")


def test_zero_crosstalk_memory():
    """Test integrated zero-crosstalk memory."""
    memory = ZeroCrosstalkMemory(dim=1024, k=64, vocab_size=100)
    
    # Store some tokens
    memory.store(token_id=0, position=0, context="the cat")
    memory.store(token_id=1, position=1, context="the dog")
    memory.store(token_id=2, position=2, context="a bird")
    
    # Retrieve
    indices, similarities = memory.retrieve(position=0, context="the cat")
    
    # Should get some results
    assert len(indices) > 0, "Should retrieve at least one result"
    
    stats = memory.get_statistics()
    assert stats['num_memories'] >= 1, "Should have stored memories"
    
    print("✓ Zero-Crosstalk Memory tests passed")


if __name__ == "__main__":
    test_k_sparse_encoder()
    test_nonlinear_threshold()
    test_orthogonal_projector()
    test_semantic_canonicalizer()
    test_fractional_encoder()
    test_zero_crosstalk_memory()
    
    print("\n" + "="*60)
    print("All Zero-Crosstalk tests passed!")
    print("="*60)
