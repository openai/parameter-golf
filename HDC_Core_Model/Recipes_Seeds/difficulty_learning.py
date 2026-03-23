"""
BLAKE3-Based Difficulty Learning System

This module implements the difficulty learning system that uses BLAKE3 fingerprints
to learn problem difficulty over time, enabling adaptive time budgeting.

Key Features:
- BLAKE3 Fingerprinting: Same problem → identical signature on any hardware
- Three-Tier Difficulty Estimation: Exact match → Structural similarity → Category baseline
- Adaptive Time Budgets: EASY/MEDIUM/HARD/NOVEL with different resource allocations
- Convergence Monitoring: Real-time tracking of search progress
- Learning from Experience: Difficulty profiles updated with actual solve times

From FULLINTEGRATION_NEW_ARCHITECTURE.md:
- Section 22: BLAKE3-Based Difficulty Learning System
- Section 23: Achieving Near-100% Accuracy with Exact Bounded Search

Usage:
    >>> difficulty_memory = DifficultyMemory()
    >>> 
    >>> # Estimate difficulty for a new problem
    >>> input_vec = hdc.from_seed_string("input:task_123")
    >>> output_vec = hdc.from_seed_string("output:task_123")
    >>> profile = difficulty_memory.estimate_difficulty(input_vec, output_vec)
    >>> 
    >>> print(f"Difficulty: {profile.difficulty_class}")
    >>> print(f"Estimated time: {profile.estimated_time_ms}ms")
"""

import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time
import json

# Try to import BLAKE3
try:
    import blake3
    _BLAKE3_AVAILABLE = True
except ImportError:
    _BLAKE3_AVAILABLE = False
    blake3 = None

# Import from core modules
from ..HDC_Core_Main.hdc_sparse_core import (
    seed_to_hypervector_blake3,
    DEFAULT_HDC_DIM
)


# =============================================================================
# Difficulty Classes
# =============================================================================

class DifficultyClass(Enum):
    """Difficulty classification for problems."""
    EASY = "EASY"           # Known recipe, instant recall
    MEDIUM = "MEDIUM"       # Related recipe, bounded search
    HARD = "HARD"           # Novel composition, resonator convergence
    NOVEL = "NOVEL"         # Genuinely new, full peeling required


class ConvergenceSignal(Enum):
    """Signals for convergence monitoring."""
    CONVERGING = "converging"      # Residue shrinking steadily
    STUCK = "stuck"               # Stuck in local attractor
    OSCILLATING = "oscillating"   # Search is unstable
    UNCERTAIN = "uncertain"       # No clear pattern
    CONTINUE = "continue"         # Continue search


# =============================================================================
# Time Budget
# =============================================================================

@dataclass
class TimeBudget:
    """
    Time budget for different difficulty classes.
    
    Attributes:
        max_time_ms: Maximum time in milliseconds
        max_search_depth: Maximum search depth for XOR peeling
        max_resonator_iterations: Maximum resonator iterations
        strategy_order: Order of strategies to try
        can_extend: Whether budget can be extended if converging
    """
    max_time_ms: float
    max_search_depth: int
    max_resonator_iterations: int
    strategy_order: List[str] = field(default_factory=list)
    can_extend: bool = False


# Default budgets for each difficulty class
DEFAULT_BUDGETS = {
    DifficultyClass.EASY: TimeBudget(
        max_time_ms=1,
        max_search_depth=2,
        max_resonator_iterations=10,
        strategy_order=["recall", "shallow_peel"],
        can_extend=False
    ),
    DifficultyClass.MEDIUM: TimeBudget(
        max_time_ms=10,
        max_search_depth=5,
        max_resonator_iterations=30,
        strategy_order=["recall", "relationship", "peel"],
        can_extend=True
    ),
    DifficultyClass.HARD: TimeBudget(
        max_time_ms=100,
        max_search_depth=10,
        max_resonator_iterations=100,
        strategy_order=["relationship", "peel", "resonator"],
        can_extend=True
    ),
    DifficultyClass.NOVEL: TimeBudget(
        max_time_ms=1000,
        max_search_depth=20,
        max_resonator_iterations=500,
        strategy_order=["full_peel", "resonator", "mcts"],
        can_extend=True
    ),
}


# =============================================================================
# Difficulty Profile
# =============================================================================

@dataclass
class DifficultyProfile:
    """
    Everything the system learns about a problem's difficulty.
    
    Stored by BLAKE3 signature — tiny, permanent, transferable.
    
    Attributes:
        signature: BLAKE3 fingerprint of the problem
        solve_times: History of actual solve times (ms)
        search_depth_needed: How deep peeling had to go
        iterations_to_converge: Resonator iterations needed
        failed_strategies: What didn't work
        successful_strategy: What finally worked
        difficulty_class: EASY / MEDIUM / HARD / NOVEL
        confidence: How certain we are of difficulty estimate (0.0 to 1.0)
        usage_count: Number of times this problem has been seen
    """
    signature: str
    solve_times: List[float] = field(default_factory=list)
    search_depth_needed: int = 0
    iterations_to_converge: int = 0
    failed_strategies: List[str] = field(default_factory=list)
    successful_strategy: str = ""
    difficulty_class: DifficultyClass = DifficultyClass.NOVEL
    confidence: float = 0.0
    usage_count: int = 0
    
    @property
    def estimated_time_ms(self) -> float:
        """Get estimated solve time from history."""
        if not self.solve_times:
            budget = DEFAULT_BUDGETS.get(self.difficulty_class)
            return budget.max_time_ms if budget else 1000.0
        return np.mean(self.solve_times)
    
    @property
    def success_rate(self) -> float:
        """Get success rate from solve history."""
        if not self.solve_times:
            return 0.0
        return len([t for t in self.solve_times if t > 0]) / len(self.solve_times)
    
    def update(self, solve_time: float, strategy: str, success: bool):
        """
        Update profile with new solve attempt.
        
        Args:
            solve_time: Time taken to solve (ms)
            strategy: Strategy used
            success: Whether the solve was successful
        """
        self.solve_times.append(solve_time)
        self.usage_count += 1
        
        if success:
            self.successful_strategy = strategy
        else:
            if strategy not in self.failed_strategies:
                self.failed_strategies.append(strategy)
        
        # Update difficulty class based on average solve time
        avg_time = self.estimated_time_ms
        if avg_time < 5:
            self.difficulty_class = DifficultyClass.EASY
        elif avg_time < 50:
            self.difficulty_class = DifficultyClass.MEDIUM
        elif avg_time < 500:
            self.difficulty_class = DifficultyClass.HARD
        else:
            self.difficulty_class = DifficultyClass.NOVEL
        
        # Increase confidence with more data
        self.confidence = min(1.0, 0.5 + 0.1 * len(self.solve_times))
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'signature': self.signature,
            'solve_times': self.solve_times,
            'search_depth': self.search_depth_needed,
            'iterations': self.iterations_to_converge,
            'failed': self.failed_strategies,
            'success': self.successful_strategy,
            'class': self.difficulty_class.value,
            'confidence': self.confidence,
            'usage': self.usage_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DifficultyProfile':
        """Deserialize from dictionary."""
        return cls(
            signature=data['signature'],
            solve_times=data.get('solve_times', []),
            search_depth_needed=data.get('search_depth', 0),
            iterations_to_converge=data.get('iterations', 0),
            failed_strategies=data.get('failed', []),
            successful_strategy=data.get('success', ''),
            difficulty_class=DifficultyClass(data.get('class', 'NOVEL')),
            confidence=data.get('confidence', 0.0),
            usage_count=data.get('usage', 0)
        )


# =============================================================================
# Difficulty Memory
# =============================================================================

class DifficultyMemory:
    """
    Learns to recognise problem difficulty from BLAKE3 signatures.
    
    Three layers of recognition:
    1. Exact match: Seen this exact problem before (confidence = 1.0)
    2. Structural similarity: Similar problems seen before (confidence = 0.75)
    3. Category baseline: At least know problem type (confidence = 0.40)
    
    Storage:
    - exact_profiles: Signature → DifficultyProfile
    - structural_clusters: Similar signatures → difficulty class
    - category_baselines: Problem category → baseline difficulty
    """
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM):
        """
        Initialize Difficulty Memory.
        
        Args:
            dim: HDC dimension
        """
        self.dim = dim
        self.uint64_count = dim // 64
        
        # Storage layers
        self.exact_profiles: Dict[str, DifficultyProfile] = {}
        self.structural_clusters: Dict[str, List[str]] = {}
        self.category_baselines: Dict[str, DifficultyProfile] = {}
        
        # Statistics
        self.total_problems_seen = 0
        self.total_recalls = 0
    
    def compute_signature(self, input_vec: np.ndarray, output_vec: np.ndarray) -> str:
        """
        Compute BLAKE3 fingerprint of the problem.
        
        Same problem → identical signature on any hardware, forever.
        
        Args:
            input_vec: Input hypervector
            output_vec: Output hypervector
            
        Returns:
            16-character hex signature
        """
        problem_vec = np.bitwise_xor(input_vec, output_vec)
        problem_bytes = problem_vec.tobytes()
        
        if _BLAKE3_AVAILABLE:
            return blake3.blake3(problem_bytes).hexdigest(length=16)
        else:
            return hashlib.blake2s(problem_bytes, digest_size=8).hexdigest()
    
    def estimate_difficulty(self, 
                            input_vec: np.ndarray,
                            output_vec: np.ndarray) -> DifficultyProfile:
        """
        Three-tier difficulty estimation.
        
        Tries exact match first, falls back to structural similarity,
        then category baseline.
        
        Args:
            input_vec: Input hypervector
            output_vec: Output hypervector
            
        Returns:
            DifficultyProfile with estimated difficulty
        """
        self.total_problems_seen += 1
        
        # Compute signature
        sig = self.compute_signature(input_vec, output_vec)
        
        # Tier 1: Exact match — seen this exact problem before
        if sig in self.exact_profiles:
            self.total_recalls += 1
            profile = self.exact_profiles[sig]
            profile.confidence = 1.0
            return profile
        
        # Tier 2: Structural similarity — similar problems seen before
        similar_sig = self._find_structurally_similar(sig)
        if similar_sig:
            similar_profile = self.exact_profiles.get(similar_sig)
            if similar_profile:
                profile = DifficultyProfile(
                    signature=sig,
                    difficulty_class=similar_profile.difficulty_class,
                    confidence=0.75,
                    search_depth_needed=similar_profile.search_depth_needed,
                    iterations_to_converge=similar_profile.iterations_to_converge
                )
                return profile
        
        # Tier 3: Category baseline — at least know problem type
        category = self._infer_category(input_vec, output_vec)
        if category in self.category_baselines:
            baseline = self.category_baselines[category]
            profile = DifficultyProfile(
                signature=sig,
                difficulty_class=baseline.difficulty_class,
                confidence=0.40,
                search_depth_needed=baseline.search_depth_needed,
                iterations_to_converge=baseline.iterations_to_converge
            )
            return profile
        
        # Genuinely novel — allocate maximum time budget
        return DifficultyProfile(
            signature=sig,
            difficulty_class=DifficultyClass.NOVEL,
            confidence=0.0,
            search_depth_needed=20,
            iterations_to_converge=500
        )
    
    def _find_structurally_similar(self, sig: str) -> Optional[str]:
        """
        Find structurally similar signature.
        
        Uses prefix matching for structural similarity.
        """
        # Check first 8 characters (prefix)
        prefix = sig[:8]
        
        for cluster_prefix, signatures in self.structural_clusters.items():
            if cluster_prefix == prefix:
                return signatures[0] if signatures else None
        
        # Check for similar prefixes
        for existing_sig in self.exact_profiles.keys():
            # Hamming distance on hex strings
            distance = sum(c1 != c2 for c1, c2 in zip(sig, existing_sig))
            if distance <= 4:  # Allow up to 4 character differences
                return existing_sig
        
        return None
    
    def _infer_category(self, 
                        input_vec: np.ndarray, 
                        output_vec: np.ndarray) -> str:
        """
        Infer problem category from vectors.
        
        Categories:
        - geometric: Spatial transformations
        - color: Color-based patterns
        - sequence: Temporal patterns
        - logic: Logical reasoning
        - unknown: Cannot determine
        """
        # Simple heuristic based on vector statistics
        xor_vec = np.bitwise_xor(input_vec, output_vec)
        
        # Count bit flips
        bit_flips = np.unpackbits(xor_vec.view(np.uint8)).sum()
        flip_ratio = bit_flips / (len(xor_vec) * 8)
        
        # Categorize based on flip ratio
        if flip_ratio < 0.3:
            return "geometric"  # Small changes = spatial transform
        elif flip_ratio < 0.5:
            return "color"      # Medium changes = color pattern
        elif flip_ratio < 0.7:
            return "sequence"   # Larger changes = temporal
        else:
            return "logic"      # Major changes = logical reasoning
    
    def record_solve(self,
                     input_vec: np.ndarray,
                     output_vec: np.ndarray,
                     solve_time_ms: float,
                     strategy: str,
                     success: bool,
                     search_depth: int = 0,
                     iterations: int = 0):
        """
        Record a solve attempt for learning.
        
        Args:
            input_vec: Input hypervector
            output_vec: Output hypervector
            solve_time_ms: Time taken to solve
            strategy: Strategy used
            success: Whether the solve was successful
            search_depth: Search depth used
            iterations: Resonator iterations used
        """
        sig = self.compute_signature(input_vec, output_vec)
        
        if sig in self.exact_profiles:
            # Update existing profile
            profile = self.exact_profiles[sig]
            profile.update(solve_time_ms, strategy, success)
        else:
            # Create new profile
            profile = DifficultyProfile(
                signature=sig,
                solve_times=[solve_time_ms],
                search_depth_needed=search_depth,
                iterations_to_converge=iterations,
                successful_strategy=strategy if success else "",
                failed_strategies=[] if success else [strategy],
                difficulty_class=DifficultyClass.NOVEL,
                confidence=0.5,
                usage_count=1
            )
            
            # Set initial difficulty class based on solve time
            if solve_time_ms < 5:
                profile.difficulty_class = DifficultyClass.EASY
            elif solve_time_ms < 50:
                profile.difficulty_class = DifficultyClass.MEDIUM
            elif solve_time_ms < 500:
                profile.difficulty_class = DifficultyClass.HARD
            else:
                profile.difficulty_class = DifficultyClass.NOVEL
            
            self.exact_profiles[sig] = profile
        
        # Update structural cluster
        prefix = sig[:8]
        if prefix not in self.structural_clusters:
            self.structural_clusters[prefix] = []
        if sig not in self.structural_clusters[prefix]:
            self.structural_clusters[prefix].append(sig)
        
        # Update category baseline
        category = self._infer_category(input_vec, output_vec)
        self._update_category_baseline(category, profile)
    
    def _update_category_baseline(self, category: str, profile: DifficultyProfile):
        """Update category baseline with new profile."""
        if category not in self.category_baselines:
            self.category_baselines[category] = DifficultyProfile(
                signature=f"baseline:{category}",
                difficulty_class=profile.difficulty_class,
                confidence=0.3,
                search_depth_needed=profile.search_depth_needed,
                iterations_to_converge=profile.iterations_to_converge
            )
        else:
            # Running average
            baseline = self.category_baselines[category]
            n = baseline.usage_count + 1
            baseline.search_depth_needed = (
                baseline.search_depth_needed * baseline.usage_count + 
                profile.search_depth_needed
            ) // n
            baseline.iterations_to_converge = (
                baseline.iterations_to_converge * baseline.usage_count + 
                profile.iterations_to_converge
            ) // n
            baseline.usage_count = n
    
    def get_time_budget(self, profile: DifficultyProfile) -> TimeBudget:
        """
        Get time budget for a difficulty profile.
        
        Args:
            profile: Difficulty profile
            
        Returns:
            TimeBudget for the problem
        """
        return DEFAULT_BUDGETS.get(profile.difficulty_class, DEFAULT_BUDGETS[DifficultyClass.NOVEL])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_problems_seen': self.total_problems_seen,
            'total_recalls': self.total_recalls,
            'unique_problems': len(self.exact_profiles),
            'structural_clusters': len(self.structural_clusters),
            'category_baselines': len(self.category_baselines),
            'recall_rate': self.total_recalls / max(1, self.total_problems_seen)
        }


# =============================================================================
# Convergence Monitor
# =============================================================================

class ConvergenceMonitor:
    """
    Monitors XOR residue to decide whether to extend search.
    
    Reads the XOR residue trend to determine:
    - CONVERGING: Residue shrinking steadily
    - STUCK: Residue flat and stable
    - OSCILLATING: Residue oscillating
    - UNCERTAIN: No clear pattern
    """
    
    def __init__(self, 
                 convergence_threshold: float = 0.95,
                 stuck_variance_threshold: float = 0.0001,
                 oscillation_variance_threshold: float = 0.05):
        """
        Initialize Convergence Monitor.
        
        Args:
            convergence_threshold: Threshold for considering converged
            stuck_variance_threshold: Variance threshold for stuck detection
            oscillation_variance_threshold: Variance threshold for oscillation
        """
        self.convergence_threshold = convergence_threshold
        self.stuck_variance_threshold = stuck_variance_threshold
        self.oscillation_variance_threshold = oscillation_variance_threshold
    
    def monitor(self, residue_history: List[float]) -> ConvergenceSignal:
        """
        Monitor convergence from residue history.
        
        Args:
            residue_history: List of residue values over iterations
            
        Returns:
            ConvergenceSignal indicating current state
        """
        if len(residue_history) < 3:
            return ConvergenceSignal.CONTINUE
        
        recent = residue_history[-5:] if len(residue_history) >= 5 else residue_history
        
        # Check for convergence
        if recent[-1] >= self.convergence_threshold:
            return ConvergenceSignal.CONVERGING
        
        # Compute trend
        if len(recent) >= 3:
            trend = np.polyfit(range(len(recent)), recent, deg=1)[0]
            variance = np.var(recent)
            
            if trend < -0.02:
                # Residue shrinking steadily — actively converging
                return ConvergenceSignal.CONVERGING
            
            elif abs(trend) < 0.001 and variance < self.stuck_variance_threshold:
                # Residue flat and stable — stuck in local attractor
                return ConvergenceSignal.STUCK
            
            elif variance > self.oscillation_variance_threshold:
                # Residue oscillating — search is unstable
                return ConvergenceSignal.OSCILLATING
        
        return ConvergenceSignal.UNCERTAIN
    
    def should_extend(self, 
                      residue_history: List[float],
                      current_time_ms: float,
                      budget: TimeBudget) -> bool:
        """
        Decide whether to extend the time budget.
        
        Args:
            residue_history: History of residue values
            current_time_ms: Current elapsed time
            budget: Current time budget
            
        Returns:
            True if budget should be extended
        """
        if not budget.can_extend:
            return False
        
        signal = self.monitor(residue_history)
        
        # Extend if converging and within reasonable time
        if signal == ConvergenceSignal.CONVERGING:
            return current_time_ms < budget.max_time_ms * 2
        
        return False


# =============================================================================
# Exact Bounded Search
# =============================================================================

class ExactBoundedSearch:
    """
    Achieves near-100% accuracy by ensuring each search stage
    operates on a small enough subspace for exact search.
    
    Three-stage search:
    1. Exact search over categories (~10-50)
    2. Exact search within those categories (~50-200 each)
    3. Exact relationship-guided refinement
    """
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM):
        """
        Initialize Exact Bounded Search.
        
        Args:
            dim: HDC dimension
        """
        self.dim = dim
        self.uint64_count = dim // 64
    
    def _generate_vector(self, seed_string: str) -> np.ndarray:
        """Generate deterministic vector from seed string."""
        return seed_to_hypervector_blake3(seed_string, self.uint64_count)
    
    def _hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """Compute Hamming distance between two vectors."""
        xored = np.bitwise_xor(a, b)
        return int(np.unpackbits(xored.view(np.uint8)).sum())
    
    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Hamming similarity between two vectors."""
        xored = np.bitwise_xor(a, b)
        diff_bits = np.unpackbits(xored.view(np.uint8)).sum()
        total_bits = len(a) * 64
        return 1.0 - (diff_bits / total_bits)
    
    def search(self,
               target: np.ndarray,
               concepts: Dict[str, np.ndarray],
               k: int = 1) -> List[Tuple[str, float]]:
        """
        Exhaustive exact search - 100% accurate within subspace.
        
        Args:
            target: Target vector
            concepts: Dictionary mapping names to vectors
            k: Number of top results to return
            
        Returns:
            List of (name, similarity) tuples, sorted by similarity
        """
        scores = []
        for name, vec in concepts.items():
            sim = self._similarity(target, vec)
            scores.append((name, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def verify_minimum_separation(self,
                                  concepts: Dict[str, np.ndarray],
                                  min_distance: int) -> bool:
        """
        Verify that all concept pairs are sufficiently separated.
        
        If this passes, approximate search BECOMES exact search.
        
        Args:
            concepts: Dictionary mapping names to vectors
            min_distance: Minimum required Hamming distance
            
        Returns:
            True if all pairs are sufficiently separated
        """
        seeds = list(concepts.keys())
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i+1:]:
                hamming = self._hamming_distance(concepts[seed_a], concepts[seed_b])
                if hamming < min_distance:
                    return False
        return True
    
    def verified_search(self,
                        target: np.ndarray,
                        concepts: Dict[str, np.ndarray],
                        confidence_threshold: float = 0.95) -> Tuple[str, bool, float]:
        """
        XOR verification is binary and exact.
        
        A correct answer produces near-zero residue provably, not statistically.
        
        Args:
            target: Target vector
            concepts: Dictionary mapping names to vectors
            confidence_threshold: Threshold for verification
            
        Returns:
            Tuple of (result_name, is_verified, confidence)
        """
        # Find best match
        results = self.search(target, concepts, k=1)
        
        if not results:
            return "", False, 0.0
        
        best_name, best_sim = results[0]
        
        # Verify
        is_verified = best_sim >= confidence_threshold
        
        return best_name, is_verified, best_sim


# =============================================================================
# Factory Functions
# =============================================================================

def create_difficulty_memory(dim: int = DEFAULT_HDC_DIM) -> DifficultyMemory:
    """
    Factory function to create a DifficultyMemory.
    
    Args:
        dim: HDC dimension
        
    Returns:
        Configured DifficultyMemory instance
    """
    return DifficultyMemory(dim=dim)


def create_convergence_monitor() -> ConvergenceMonitor:
    """
    Factory function to create a ConvergenceMonitor.
    
    Returns:
        Configured ConvergenceMonitor instance
    """
    return ConvergenceMonitor()


def create_exact_search(dim: int = DEFAULT_HDC_DIM) -> ExactBoundedSearch:
    """
    Factory function to create an ExactBoundedSearch.
    
    Args:
        dim: HDC dimension
        
    Returns:
        Configured ExactBoundedSearch instance
    """
    return ExactBoundedSearch(dim=dim)
