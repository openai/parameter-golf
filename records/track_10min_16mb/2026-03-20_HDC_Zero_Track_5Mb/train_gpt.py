

"""HDC VSA Language Model for Parameter-Golf Competition.

Zero-weight architecture using procedurally generated hypervectors.
Run: python train_gpt.py --multi-seed --seeds 42 7 1337
"""

from __future__ import annotations

import glob
import io
import json
import math
import os
import struct
import sys
import time
import uuid
import zlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Callable

import numpy as np
import sentencepiece as spm

try:
    import blake3
    _BLAKE3_AVAILABLE = True
except ImportError:
    _BLAKE3_AVAILABLE = False
    blake3 = None

try:
    import torch
    import torch.distributed as dist
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    dist = None

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    cp = None

DEFAULT_HDC_DIM = 2**20  # 1,048,576 dimensions
HDC_DIM_L1 = 2**17       # 131,072 - L1 cache resident
HDC_DIM_L2 = 2**18       # 262,144 - L2 cache resident
HDC_DIM_L3 = 2**19       # 524,288 - L3 cache resident

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

@dataclass
class TimeBudget:
    
    max_time_ms: float
    max_search_depth: int
    max_resonator_iterations: int
    strategy_order: List[str] = field(default_factory=list)
    can_extend: bool = False

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

@dataclass
class DifficultyProfile:
    
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
        
        self.solve_times.append(solve_time)
        self.usage_count += 1
        
        if success:
            self.successful_strategy = strategy
        else:
            if strategy not in self.failed_strategies:
                self.failed_strategies.append(strategy)
        
        avg_time = self.estimated_time_ms
        if avg_time < 5:
            self.difficulty_class = DifficultyClass.EASY
        elif avg_time < 50:
            self.difficulty_class = DifficultyClass.MEDIUM
        elif avg_time < 500:
            self.difficulty_class = DifficultyClass.HARD
        else:
            self.difficulty_class = DifficultyClass.NOVEL
        
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

class DifficultyMemory:
    
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM):
        
        self.dim = dim
        self.uint64_count = dim // 64
        
        self.exact_profiles: Dict[str, DifficultyProfile] = {}
        self.structural_clusters: Dict[str, List[str]] = {}
        self.category_baselines: Dict[str, DifficultyProfile] = {}
        
        self.total_problems_seen = 0
        self.total_recalls = 0
    
    def compute_signature(self, input_vec: np.ndarray, output_vec: np.ndarray) -> str:
        
        problem_vec = np.bitwise_xor(input_vec, output_vec)
        problem_bytes = problem_vec.tobytes()
        
        if _BLAKE3_AVAILABLE:
            return blake3.blake3(problem_bytes).hexdigest(length=16)
        else:
            return hashlib.blake2s(problem_bytes, digest_size=8).hexdigest()
    
    def estimate_difficulty(self, 
                            input_vec: np.ndarray,
                            output_vec: np.ndarray) -> DifficultyProfile:
        
        self.total_problems_seen += 1
        
        sig = self.compute_signature(input_vec, output_vec)
        
        if sig in self.exact_profiles:
            self.total_recalls += 1
            profile = self.exact_profiles[sig]
            profile.confidence = 1.0
            return profile
        
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
        
        return DifficultyProfile(
            signature=sig,
            difficulty_class=DifficultyClass.NOVEL,
            confidence=0.0,
            search_depth_needed=20,
            iterations_to_converge=500
        )
    
    def _find_structurally_similar(self, sig: str) -> Optional[str]:
        
        prefix = sig[:8]
        
        for cluster_prefix, signatures in self.structural_clusters.items():
            if cluster_prefix == prefix:
                return signatures[0] if signatures else None
        
        for existing_sig in self.exact_profiles.keys():
            distance = sum(c1 != c2 for c1, c2 in zip(sig, existing_sig))
            if distance <= 4:  # Allow up to 4 character differences
                return existing_sig
        
        return None
    
    def _infer_category(self, 
                        input_vec: np.ndarray, 
                        output_vec: np.ndarray) -> str:
        
        xor_vec = np.bitwise_xor(input_vec, output_vec)
        
        bit_flips = np.unpackbits(xor_vec.view(np.uint8)).sum()
        flip_ratio = bit_flips / (len(xor_vec) * 8)
        
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
        
        sig = self.compute_signature(input_vec, output_vec)
        
        if sig in self.exact_profiles:
            profile = self.exact_profiles[sig]
            profile.update(solve_time_ms, strategy, success)
        else:
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
            
            if solve_time_ms < 5:
                profile.difficulty_class = DifficultyClass.EASY
            elif solve_time_ms < 50:
                profile.difficulty_class = DifficultyClass.MEDIUM
            elif solve_time_ms < 500:
                profile.difficulty_class = DifficultyClass.HARD
            else:
                profile.difficulty_class = DifficultyClass.NOVEL
            
            self.exact_profiles[sig] = profile
        
        prefix = sig[:8]
        if prefix not in self.structural_clusters:
            self.structural_clusters[prefix] = []
        if sig not in self.structural_clusters[prefix]:
            self.structural_clusters[prefix].append(sig)
        
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
        
        return DEFAULT_BUDGETS.get(profile.difficulty_class, DEFAULT_BUDGETS[DifficultyClass.NOVEL])
    

class GPUManager:
    
    
    _instance = None
    _initialized = False
    
    def __new__(cls, use_gpu: bool = True, device_id: int = 0):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        if GPUManager._initialized:
            return
        
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self.device_id = device_id
        self._stream = None
        self._pinned_memory_pool = {}
        
        if self.use_gpu:
            try:
                cp.cuda.Device(device_id).use()
                self._stream = cp.cuda.Stream()
                test_arr = cp.array([1, 2, 3])
                del test_arr
                cp.cuda.Stream.null.synchronize()
                print(f"GPU acceleration enabled: {cp.cuda.Device(device_id).name.decode()}")
                
                self._init_cuda_kernels()
            except Exception as e:
                print(f"GPU initialization failed: {e}, falling back to CPU")
                self.use_gpu = False
        
        self.xp = cp if self.use_gpu else np
        
        GPUManager._initialized = True
    
    def _init_cuda_kernels(self):
        """Initialize custom CUDA kernels for fused operations."""
        if not self.use_gpu:
            return
        
        self._xor_popcount_kernel = cp.ElementwiseKernel(
            'uint64 a, uint64 b',
            'uint32 out',
            '''
            unsigned long long xored = a ^ b;
            out = __popcll(xored);
            ''',
            'xor_popcount'
        )
        
        self._batch_xor_kernel = cp.ElementwiseKernel(
            'uint64 a, uint64 b',
            'uint64 out',
            'out = a ^ b',
            'batch_xor'
        )
        
        self._cumulative_xor_kernel = cp.ReductionKernel(
            'uint64 x',
            'uint64 y',
            'a ^ b',
            'identity = 0',
            'y = a',
            'a = x',
            'cumulative_xor'
        )
    
    def to_gpu(self, arr: np.ndarray) -> 'cp.ndarray':
        """Transfer array to GPU if available."""
        if self.use_gpu and isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        return arr
    
    def to_gpu_async(self, arr: np.ndarray) -> 'cp.ndarray':
        """Async transfer using pinned memory if available."""
        if self.use_gpu and isinstance(arr, np.ndarray):
            with self._stream:
                return cp.asarray(arr)
        return arr
    
    def to_cpu(self, arr) -> np.ndarray:
        """Transfer array to CPU."""
        if self.use_gpu and not isinstance(arr, np.ndarray):
            return cp.asnumpy(arr)
        return arr
    
    def to_cpu_async(self, arr) -> np.ndarray:
        """Async transfer to CPU."""
        if self.use_gpu and not isinstance(arr, np.ndarray):
            with self._stream:
                return cp.asnumpy(arr)
        return arr
    
    def allocate(self, shape, dtype=np.uint64) -> 'cp.ndarray':
        """Allocate array on GPU if available."""
        return self.xp.zeros(shape, dtype=dtype)
    
    def synchronize(self):
        """Synchronize GPU stream."""
        if self.use_gpu and self._stream:
            self._stream.synchronize()
    
    @property
    def stream(self):
        """Get the async stream for manual control."""
        return self._stream

_gpu_manager: Optional[GPUManager] = None

def get_gpu_manager(use_gpu: bool = True, device_id: int = 0) -> GPUManager:
    """Get or create the global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager(use_gpu=use_gpu, device_id=device_id)
    return _gpu_manager

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool
import threading

@dataclass
class HDCConfig:
    """Configuration for HDC model with full architecture integration."""
    data_path: str = "./data/datasets/fineweb10B_sp1024"
    train_files: str = ""
    val_files: str = ""
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model"
    run_id: str = ""
    seed: int = 42
    
    hdc_dim: int = DEFAULT_HDC_DIM  # 2^20 for L2 cache residency
    vocab_size: int = 1024
    max_context_length: int = 512
    
    use_ternary: bool = True  # {-1, 0, +1} representation
    
    temporal_folding: bool = True  # Enable circular shift encoding
    max_temporal_depth: int = 1000  # Maximum sequence length
    
    use_resonator: bool = True
    resonator_iterations: int = 10
    resonator_agents: int = 6
    
    max_peeling_iterations: int = 100
    convergence_threshold: float = 0.95
    n_search_agents: int = 6
    
    use_relationships: bool = True
    
    max_recipes: int = 100000  # ~5MB at 50 bytes each
    recipe_compression_level: int = 9
    deduplication_enabled: bool = True
    
    collision_threshold: float = 0.55
    holographic_redundancy: int = 3  # Number of redundant encodings
    
    iterations: int = 20000
    max_wallclock_seconds: float = 600.0
    train_batch_tokens: int = 524288
    val_batch_size: int = 524288
    val_loss_every: int = 1000
    train_log_every: int = 200
    
    temperature: float = 1.0
    similarity_scale: float = 10.0
    min_probability: float = 1e-10
    
    target_accuracy: float = 0.99
    use_hierarchical_search: bool = True
    hierarchical_depths: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    use_enhanced_resonator: bool = True
    max_resonator_iterations: int = 300
    min_resonator_iterations: int = 50
    stuck_detection_window: int = 20
    use_iterative_refinement: bool = True
    refinement_passes: int = 3
    residue_threshold: float = 0.01
    use_parallel_search: bool = True
    parallel_paths: int = 8
    use_enhanced_collision_shield: bool = True
    min_hamming_distance_ratio: float = 0.4
    codebook_expansion_factor: int = 4
    use_gpu_acceleration: bool = True
    
    gpu_device_id: int = 0
    gpu_batch_size: int = 1024  # Batch size for GPU operations
    
    def __post_init__(self):
        if not self.train_files:
            self.train_files = os.path.join(self.data_path, "fineweb_train_*.bin")
        if not self.val_files:
            self.val_files = os.path.join(self.data_path, "fineweb_val_*.bin")
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
    
    @property
    def uint64_count(self) -> int:
        """Number of uint64 elements needed to store the vector."""
        return self.hdc_dim // 64
    
    @property
    def byte_size(self) -> int:
        """Size in bytes for the vector."""
        return self.hdc_dim // 8

class WalshHadamardBasis:
    
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM, use_gpu: bool = False):
        
        if dim <= 0 or (dim & (dim - 1)) != 0:
            raise ValueError(f"Dimension must be power of 2, got {dim}")
        
        self.dim = dim
        self.log_dim = int(np.log2(dim))
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        self._row_cache: dict = {}
        self._cache_max_size = 1000
    
    def get_row(self, index: int, packed: bool = False) -> np.ndarray:
        
        cache_key = (index, packed)
        
        if cache_key in self._row_cache:
            return self._row_cache[cache_key].copy()
        
        if packed:
            row = hadamard_row_packed(index, self.dim)
        else:
            row = sylvester_hadamard_row_fast(index, self.dim)
        
        if len(self._row_cache) < self._cache_max_size:
            self._row_cache[cache_key] = row.copy()
        
        if self.use_gpu:
            return cp.asarray(row)
        return row
    
    def get_row_from_string(self, name: str, packed: bool = False, seed: int = 0) -> Tuple[int, np.ndarray]:
        
        if seed != 0:
            hash_input = f"{seed}:{name}".encode()
        else:
            hash_input = name.encode()
        
        if _BLAKE3_AVAILABLE:
            hash_bytes = blake3.blake3(hash_input).digest(length=4)
        else:
            import hashlib
            hash_bytes = hashlib.sha256(hash_input).digest()[:4]
        
        index = int.from_bytes(hash_bytes, 'big') % self.dim
        return index, self.get_row(index, packed=packed)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        
        return self._fwht(data)
    
    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        
        return self._fwht(transformed) / self.dim
    
    def _fwht(self, data: np.ndarray) -> np.ndarray:
        
        xp = self.xp
        
        if self.use_gpu and not isinstance(data, cp.ndarray):
            data = cp.asarray(data)
        elif not self.use_gpu and _CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        
        result = data.astype(xp.float64).copy()
        
        n = result.shape[-1]
        if n != self.dim:
            raise ValueError(f"Data dimension {n} != basis dimension {self.dim}")
        
        h = 1
        while h < n:
            half_n = n // 2
            num_blocks = n // (h * 2)
            
            
            original_shape = result.shape
            if len(original_shape) == 1:
                for i in range(0, n, h * 2):
                    x = result[i:i+h]
                    y = result[i+h:i+2*h]
                    result[i:i+h] = x + y
                    result[i+h:i+2*h] = x - y
            else:
                for i in range(0, n, h * 2):
                    x = result[..., i:i+h]
                    y = result[..., i+h:i+2*h]
                    result[..., i:i+h] = x + y
                    result[..., i+h:i+2*h] = x - y
            h *= 2
        
        return result
    
    def _fwht_parallel(self, data: np.ndarray) -> np.ndarray:
        
        xp = self.xp
        
        if self.use_gpu and not isinstance(data, cp.ndarray):
            data = cp.asarray(data)
        elif not self.use_gpu and _CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        
        result = data.astype(xp.float64).copy()
        n = result.shape[-1]
        
        if n != self.dim:
            raise ValueError(f"Data dimension {n} != basis dimension {self.dim}")
        
        
        log_n = int(xp.log2(n))
        
        for level in range(log_n):
            h = 2 ** level
            chunk_size = min(h * 2, n)
            
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                for i in range(start, end, h * 2):
                    if i + h <= n:
                        x = result[..., i:i+h]
                        y = result[..., i+h:min(i+2*h, n)]
                        if y.shape[-1] == h:
                            result[..., i:i+h] = x + y
                            result[..., i+h:i+2*h] = x - y
        
        return result
    
    def inner_product(self, a: np.ndarray, b: np.ndarray) -> float:
        
        xp = self.xp
        return float(xp.dot(a.astype(xp.float64), b.astype(xp.float64)) / self.dim)
    
    def orthogonality_test(self, num_samples: int = 100) -> dict:
        
        import random
        
        results = {
            "self_inner_products": [],
            "cross_inner_products": [],
            "max_cross_product": 0.0,
            "all_orthogonal": True
        }
        
        indices = random.sample(range(self.dim), min(num_samples, self.dim))
        
        for i in indices[:10]:
            row = self.get_row(i, packed=False)
            self_ip = np.dot(row.astype(np.float64), row.astype(np.float64))
            results["self_inner_products"].append(float(self_ip))
        
        for _ in range(num_samples):
            i, j = random.sample(indices, 2)
            row_i = self.get_row(i, packed=False)
            row_j = self.get_row(j, packed=False)
            cross_ip = abs(np.dot(row_i.astype(np.float64), row_j.astype(np.float64)))
            results["cross_inner_products"].append(float(cross_ip))
            results["max_cross_product"] = max(results["max_cross_product"], cross_ip)
            if cross_ip > 1e-10:
                results["all_orthogonal"] = False
        
        return results

@dataclass
class AccuracyConfig:
    
    target_accuracy: float = 0.99
    
    max_search_depth: int = 50
    hierarchical_depths: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    early_stop_threshold: float = 0.99
    
    max_resonator_iterations: int = 300
    min_resonator_iterations: int = 50
    convergence_threshold: float = 0.995
    stuck_detection_window: int = 20
    
    codebook_expansion_factor: int = 4  # 4x more candidates
    semantic_clustering: bool = True
    
    refinement_passes: int = 3
    residue_threshold: float = 0.01
    
    parallel_paths: int = 8
    use_multiprocessing: bool = False  # Use threading by default
    
    min_hamming_distance_ratio: float = 0.4  # 40% bits different
    collision_check_enabled: bool = True
    
    use_gpu: bool = True
    hdc_dim: int = DEFAULT_HDC_DIM
    
    enable_early_termination: bool = True

class RelationshipType(Enum):
    """Core relationship types for relationship-guided search."""
    IS_A = "is_a"           # Category membership
    SIMILAR = "similar"     # Similarity relationship
    OPPOSITE = "opposite"   # Inverse relationship
    COMPOSED = "composed"   # Composition relationship
    PART_OF = "part_of"     # Part-whole relationship
    PREDICTS = "predicts"   # Sequential prediction

def blake3_hash(data: bytes) -> bytes:
    """Compute BLAKE3 hash of data, falling back to BLAKE2b if not available."""
    if _BLAKE3_AVAILABLE:
        return blake3.blake3(data).digest()
    else:
        import hashlib
        return hashlib.blake2b(data, digest_size=32).digest()

def seed_to_hypervector(seed_string: str, dim: int = DEFAULT_HDC_DIM) -> np.ndarray:
    
    uint64_count = dim // 64
    num_bytes = uint64_count * 8
    
    if _BLAKE3_AVAILABLE:
        hash_bytes = blake3.blake3(seed_string.encode()).digest(length=num_bytes)
    else:
        hash_bytes = b""
        counter = 0
        while len(hash_bytes) < num_bytes:
            data = f"{seed_string}:{counter}".encode()
            hash_bytes += blake3_hash(data)
            counter += 1
        hash_bytes = hash_bytes[:num_bytes]
    
    return np.frombuffer(hash_bytes, dtype=np.uint64).copy()

def seed_to_ternary_hypervector(seed_string: str, dim: int = DEFAULT_HDC_DIM) -> Tuple[np.ndarray, np.ndarray]:
    
    pos_vec = seed_to_hypervector(f"{seed_string}:pos", dim)
    neg_vec = seed_to_hypervector(f"{seed_string}:neg", dim)
    
    overlap = np.bitwise_and(pos_vec, neg_vec)
    pos_vec = np.bitwise_xor(pos_vec, overlap)
    neg_vec = np.bitwise_xor(neg_vec, overlap)
    
    return pos_vec, neg_vec

def hadamard_position_vector(position: int, dim: int = DEFAULT_HDC_DIM) -> np.ndarray:
    
    base = seed_to_hypervector(f"hadamard_base", dim)
    
    perm_seed = position * 2654435761  # Golden ratio constant
    
    uint64_count = dim // 64
    shift = (position * 7) % uint64_count
    result = np.roll(base, shift)
    
    pos_pattern = seed_to_hypervector(f"hadamard_pos_{position}", dim)
    result = np.bitwise_xor(result, pos_pattern)
    
    return result

def circular_temporal_encode(
    events: List[np.ndarray],
    dim: int = DEFAULT_HDC_DIM
) -> np.ndarray:
    
    if not events:
        return np.zeros(dim // 8, dtype=np.uint8)
    
    first_event = events[0]
    if first_event.dtype == np.uint8:
        byte_count = dim // 8
        result = np.zeros(byte_count, dtype=np.uint8)
        
        for i, event_vec in enumerate(events):
            shift = i % byte_count
            shifted = np.roll(event_vec, shift)
            result = np.bitwise_xor(result, shifted)
        
        return result
    else:
        uint64_count = dim // 64
        result = np.zeros(uint64_count, dtype=np.uint64)
        
        for i, event_vec in enumerate(events):
            shift = i % uint64_count
            shifted = np.roll(event_vec, shift)
            result = np.bitwise_xor(result, shifted)
        
        return result

def retrieve_event_at_position(
    sequence: np.ndarray,
    position: int,
    dim: int = DEFAULT_HDC_DIM
) -> np.ndarray:
    
    if sequence.dtype == np.uint8:
        byte_count = dim // 8
        shift = position % byte_count
    else:
        uint64_count = dim // 64
        shift = position % uint64_count
    return np.roll(sequence, -shift)

def xor_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    
    return np.bitwise_xor(a, b)

def xor_unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    
    return np.bitwise_xor(bound, key)

def xor_bind_sequence(vectors: List[np.ndarray]) -> np.ndarray:
    
    if not vectors:
        return np.zeros_like(vectors[0]) if vectors else np.zeros(2048, dtype=np.uint64)
    
    result = vectors[0].copy()
    for vec in vectors[1:]:
        result = np.bitwise_xor(result, vec)
    return result

def bundle_vectors(vectors: List[np.ndarray], dim: int = DEFAULT_HDC_DIM) -> np.ndarray:
    
    if not vectors:
        return np.zeros(dim // 64, dtype=np.uint64)
    
    uint64_count = dim // 64
    bit_sums = np.zeros(dim, dtype=np.int32)
    
    for vec in vectors:
        bits = np.unpackbits(vec.view(np.uint8))
        bit_sums += bits[:dim]
    
    threshold = len(vectors) / 2
    result_bits = (bit_sums > threshold).astype(np.uint8)
    
    result = np.packbits(result_bits).view(np.uint64)
    return result[:uint64_count]

def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    
    xored = np.bitwise_xor(a, b)
    diff_bits = np.unpackbits(xored.view(np.uint8)).sum()
    total_bits = len(a) * 64
    return 1.0 - (diff_bits / total_bits)

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Count differing bits between two hypervectors."""
    xored = np.bitwise_xor(a, b)
    return int(np.unpackbits(xored.view(np.uint8)).sum())

class GPUBatchOperations:
    
    
    def __init__(self, gpu_manager: GPUManager, dim: int = DEFAULT_HDC_DIM):
        self.gpu = gpu_manager
        self.dim = dim
        self.uint64_count = dim // 64
        self.xp = gpu_manager.xp
        
        self._token_matrix = None  # Will hold all token vectors
        self._position_matrix = None  # Will hold position vectors
        
        self._init_kernels()
    
    def _init_kernels(self):
        """Initialize custom CUDA kernels for fused operations."""
        if not self.gpu.use_gpu:
            self._xor_popcount_kernel = None
            self._parallel_cumxor_kernel = None
            return
        
        self._xor_popcount_kernel = cp.ElementwiseKernel(
            'uint64 a, uint64 b',
            'uint32 out',
            '''
            unsigned long long xored = a ^ b;
            out = __popcll(xored);
            ''',
            'xor_popcount_fused'
        )
        
        self._parallel_cumxor_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void parallel_cumxor(
            const unsigned long long* __restrict__ bound,  // (batch, seq, uint64_count)
            unsigned long long* __restrict__ result,        // (batch, uint64_count)
            int batch_size, int seq_len, int uint64_count
        ) {
            int batch_idx = blockIdx.x;
            int elem_idx = threadIdx.x;
            
            if (batch_idx >= batch_size || elem_idx >= uint64_count) return;
            
            unsigned long long acc = bound[(batch_idx * seq_len + 0) * uint64_count + elem_idx];
            
            for (int i = 1; i < seq_len; i++) {
                // Circular shift by position i
                int shift = i % uint64_count;
                int src_idx = (elem_idx - shift + uint64_count) % uint64_count;
                unsigned long long shifted = bound[(batch_idx * seq_len + i) * uint64_count + src_idx];
                acc ^= shifted;
            }
            
            result[batch_idx * uint64_count + elem_idx] = acc;
        }
        ''', 'parallel_cumxor')
        
        self._batch_xor_kernel = cp.ElementwiseKernel(
            'uint64 a, uint64 b',
            'uint64 out',
            'out = a ^ b',
            'batch_xor_fused'
        )
    
    def build_token_matrix(self, vocab_size: int, seed_offset: int = 0) -> 'xp.ndarray':
        
        if self._token_matrix is not None and self._token_matrix.shape[0] >= vocab_size:
            return self._token_matrix[:vocab_size]
        
        if self.gpu.use_gpu:
            token_matrix = self.xp.zeros((vocab_size, self.uint64_count), dtype=self.xp.uint64)
            
            for token_id in range(vocab_size):
                seed = hash(f"token_{token_id + seed_offset}") & 0xFFFFFFFFFFFFFFFF
                self.xp.random.seed(seed % (2**32))
                token_matrix[token_id] = (self.xp.random.randint(0, 2**64, self.uint64_count, dtype=self.xp.uint64))
            
            self._token_matrix = token_matrix
        else:
            token_vectors = []
            for token_id in range(vocab_size):
                vec = seed_to_hypervector(f"token_{token_id + seed_offset}", self.dim)
                token_vectors.append(vec)
            token_matrix = self.xp.stack(token_vectors, axis=0)
            self._token_matrix = token_matrix
        
        return self._token_matrix
    
    def build_position_matrix(self, max_positions: int) -> 'xp.ndarray':
        
        if self._position_matrix is not None and self._position_matrix.shape[0] >= max_positions:
            return self._position_matrix[:max_positions]
        
        if self.gpu.use_gpu:
            pos_matrix = self.xp.zeros((max_positions, self.uint64_count), dtype=self.xp.uint64)
            
            for pos in range(max_positions):
                seed = pos * 0x9E3779B97F4A7C15  # Golden ratio constant
                seed = (seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9
                seed = (seed ^ (seed >> 27)) * 0x94D049BB133111EB
                seed = seed ^ (seed >> 31)
                
                self.xp.random.seed(seed % (2**32))
                pos_matrix[pos] = self.xp.random.randint(0, 2**64, self.uint64_count, dtype=self.xp.uint64)
            
            self._position_matrix = pos_matrix
        else:
            pos_vectors = []
            for pos in range(max_positions):
                vec = hadamard_position_vector(pos, self.dim)
                pos_vectors.append(vec)
            pos_matrix = self.xp.stack(pos_vectors, axis=0)
            self._position_matrix = pos_matrix
        
        return self._position_matrix
    
    def batch_xor_bind(self, a_batch: 'xp.ndarray', b_batch: 'xp.ndarray') -> 'xp.ndarray':
        
        if self.gpu.use_gpu and self._batch_xor_kernel is not None:
            return self._batch_xor_kernel(a_batch, b_batch)
        return self.xp.bitwise_xor(a_batch, b_batch)
    
    def batch_encode_context(
        self,
        token_ids_batch: 'xp.ndarray',
        token_matrix: 'xp.ndarray',
        position_matrix: 'xp.ndarray',
        batch_chunk_size: int = 32,  # Process batch in chunks to avoid memory explosion
        seq_chunk_size: int = 64     # Process sequence in chunks
    ) -> 'xp.ndarray':
        
        batch_size, seq_len = token_ids_batch.shape
        
        result = self.xp.zeros((batch_size, self.uint64_count), dtype=self.xp.uint64)
        
        positions = self.xp.arange(seq_len)
        pos_vecs = position_matrix[positions]
        
        for batch_start in range(0, batch_size, batch_chunk_size):
            batch_end = min(batch_start + batch_chunk_size, batch_size)
            token_ids_chunk = token_ids_batch[batch_start:batch_end]
            chunk_batch_size = batch_end - batch_start
            
            first_token_vecs = token_matrix[token_ids_chunk[:, 0]]
            first_pos_vec = pos_vecs[0]
            chunk_result = self.xp.bitwise_xor(first_token_vecs, first_pos_vec)
            
            for seq_start in range(1, seq_len, seq_chunk_size):
                seq_end = min(seq_start + seq_chunk_size, seq_len)
                
                for pos in range(seq_start, seq_end):
                    token_vecs = token_matrix[token_ids_chunk[:, pos]]
                    pos_vec = pos_vecs[pos]
                    
                    bound = self.xp.bitwise_xor(token_vecs, pos_vec)
                    
                    shift = pos % self.uint64_count
                    if shift != 0:
                        bound = self.xp.roll(bound, shift, axis=1)
                    
                    chunk_result = self.xp.bitwise_xor(chunk_result, bound)
            
            result[batch_start:batch_end] = chunk_result
        
        return result
    
    def batch_hamming_similarity(
        self,
        query_batch: 'xp.ndarray',
        codebook: 'xp.ndarray',
        chunk_size: int = 64  # Process in chunks to avoid memory explosion
    ) -> 'xp.ndarray':
        
        batch_size = query_batch.shape[0]
        codebook_size = codebook.shape[0]
        
        similarity = self.xp.zeros((batch_size, codebook_size), dtype=self.xp.float32)
        
        if self.gpu.use_gpu and self._xor_popcount_kernel is not None:
            for i_start in range(0, batch_size, chunk_size):
                i_end = min(i_start + chunk_size, batch_size)
                query_chunk = query_batch[i_start:i_end]  # (chunk, uint64_count)
                
                for j_start in range(0, codebook_size, chunk_size):
                    j_end = min(j_start + chunk_size, codebook_size)
                    codebook_chunk = codebook[j_start:j_end]  # (chunk, uint64_count)
                    
                    query_expanded = query_chunk[:, self.xp.newaxis, :]  # (chunk, 1, uint64)
                    codebook_expanded = codebook_chunk[self.xp.newaxis, :, :]  # (1, chunk, uint64)
                    
                    diff_bits = self._xor_popcount_kernel(query_expanded, codebook_expanded)
                    
                    diff_bits = self.xp.sum(diff_bits, axis=-1)  # (chunk, chunk)
                    
                    chunk_similarity = 1.0 - (diff_bits.astype(self.xp.float32) / self.dim)
                    similarity[i_start:i_end, j_start:j_end] = chunk_similarity
        else:
            for i_start in range(0, batch_size, chunk_size):
                i_end = min(i_start + chunk_size, batch_size)
                query_chunk = query_batch[i_start:i_end]
                
                for j_start in range(0, codebook_size, chunk_size):
                    j_end = min(j_start + chunk_size, codebook_size)
                    codebook_chunk = codebook[j_start:j_end]
                    
                    xored = self.xp.bitwise_xor(
                        query_chunk[:, self.xp.newaxis, :],
                        codebook_chunk[self.xp.newaxis, :, :]
                    )
                    xored_uint8 = xored.view(self.xp.uint8)
                    diff_bits = self._popcount_uint8_batch(xored_uint8)
                    diff_bits = self.xp.sum(diff_bits, axis=-1)
                    
                    chunk_similarity = 1.0 - (diff_bits.astype(self.xp.float32) / self.dim)
                    similarity[i_start:i_end, j_start:j_end] = chunk_similarity
        
        return similarity
    
    def _popcount_uint8_batch(self, arr: 'xp.ndarray') -> 'xp.ndarray':
        
        if not hasattr(self, '_popcount_lut') or self._popcount_lut is None:
            lut = self.xp.array([bin(i).count('1') for i in range(256)], dtype=self.xp.uint8)
            self._popcount_lut = lut
        
        return self._popcount_lut[arr]
    
    def batch_learn_patterns(
        self,
        contexts_batch: List[List[int]],
        targets_batch: List[int],
        token_matrix: 'xp.ndarray',
        position_matrix: 'xp.ndarray'
    ) -> Tuple['xp.ndarray', 'xp.ndarray']:
        
        batch_size = len(contexts_batch)
        
        max_len = max(len(c) for c in contexts_batch)
        padded_contexts = self.xp.zeros((batch_size, max_len), dtype=self.xp.int64)
        for i, ctx in enumerate(contexts_batch):
            padded_contexts[i, :len(ctx)] = self.xp.array(ctx)
        
        context_vecs = self.batch_encode_context(padded_contexts, token_matrix, position_matrix)
        
        targets_gpu = self.xp.array(targets_batch, dtype=self.xp.int64)
        target_vecs = token_matrix[targets_gpu]
        
        patterns = self.batch_xor_bind(context_vecs, target_vecs)
        
        return patterns, target_vecs
    
    def batch_predict(
        self,
        contexts_batch: List[List[int]],
        token_matrix: 'xp.ndarray',
        position_matrix: 'xp.ndarray',
        temperature: float = 1.0,
        top_k: int = 10
    ) -> Tuple['xp.ndarray', 'xp.ndarray']:
        
        batch_size = len(contexts_batch)
        vocab_size = token_matrix.shape[0]
        
        max_len = max(len(c) for c in contexts_batch)
        padded_contexts = self.xp.zeros((batch_size, max_len), dtype=self.xp.int64)
        for i, ctx in enumerate(contexts_batch):
            padded_contexts[i, :len(ctx)] = self.xp.array(ctx)
        
        context_vecs = self.batch_encode_context(padded_contexts, token_matrix, position_matrix)
        
        similarities = self.batch_hamming_similarity(context_vecs, token_matrix)
        
        scaled = similarities * 10.0 / temperature
        
        scaled_max = self.xp.max(scaled, axis=-1, keepdims=True)
        scaled = scaled - scaled_max
        exp_scores = self.xp.exp(scaled)
        probs = exp_scores / self.xp.sum(exp_scores, axis=-1, keepdims=True)
        
        top_k = min(top_k, vocab_size)
        top_indices = self.xp.argsort(probs, axis=-1)[:, ::-1][:, :top_k]
        
        return probs, top_indices

_batch_ops: Optional[GPUBatchOperations] = None

def get_batch_ops(gpu_manager: GPUManager = None, dim: int = DEFAULT_HDC_DIM) -> GPUBatchOperations:
    """Get or create the global batch operations instance."""
    global _batch_ops
    if _batch_ops is None:
        if gpu_manager is None:
            gpu_manager = get_gpu_manager()
        _batch_ops = GPUBatchOperations(gpu_manager, dim)
    return _batch_ops

def ternary_xor(
    a_pos: np.ndarray, a_neg: np.ndarray,
    b_pos: np.ndarray, b_neg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    
    result_pos = np.bitwise_xor(
        np.bitwise_and(a_pos, b_neg),
        np.bitwise_and(a_neg, b_pos)
    )
    result_neg = np.bitwise_xor(
        np.bitwise_and(a_neg, b_pos),
        np.bitwise_and(a_pos, b_neg)
    )
    
    return result_pos, result_neg

def ternary_similarity(
    a_pos: np.ndarray, a_neg: np.ndarray,
    b_pos: np.ndarray, b_neg: np.ndarray
) -> float:
    """Compute similarity between ternary vectors."""
    pos_match = np.bitwise_and(a_pos, b_pos)
    neg_match = np.bitwise_and(a_neg, b_neg)
    
    pos_neg_mismatch = np.bitwise_or(
        np.bitwise_and(a_pos, b_neg),
        np.bitwise_and(a_neg, b_pos)
    )
    
    match_count = np.unpackbits(pos_match.view(np.uint8)).sum() + \
                  np.unpackbits(neg_match.view(np.uint8)).sum()
    mismatch_count = np.unpackbits(pos_neg_mismatch.view(np.uint8)).sum()
    
    total = match_count + mismatch_count
    if total == 0:
        return 1.0
    
    return match_count / total

class SeedRegistry:
    
    
    def __init__(self):
        self._seeds: Dict[str, int] = {}      # seed_string → seed_id
        self._id_to_seed: Dict[int, str] = {}  # seed_id → seed_string
        self._next_id = 0
    
    def get_or_create(self, seed_string: str) -> int:
        """Get existing seed ID or create new one."""
        if seed_string in self._seeds:
            return self._seeds[seed_string]  # Deduplicate!
        
        seed_id = self._next_id
        self._seeds[seed_string] = seed_id
        self._id_to_seed[seed_id] = seed_string
        self._next_id += 1
        return seed_id
    
    def get_seed(self, seed_id: int) -> Optional[str]:
        """Retrieve seed string by ID."""
        return self._id_to_seed.get(seed_id)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'seeds': self._seeds.copy(),
            'next_id': self._next_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SeedRegistry':
        """Deserialize from dictionary."""
        registry = cls()
        registry._seeds = data.get('seeds', {}).copy()
        registry._next_id = data.get('next_id', 0)
        registry._id_to_seed = {v: k for k, v in registry._seeds.items()}
        return registry

@dataclass
class Recipe:
    
    recipe_id: str
    seed_sequence: List[str]      # Seeds to generate vectors from
    operation_order: List[int]    # Order of operations
    problem_signature: str        # Hash of input/output for lookup
    target_token: int             # Predicted token
    confidence: float = 1.0
    usage_count: int = 0
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'id': self.recipe_id,
            'seeds': self.seed_sequence,
            'order': self.operation_order,
            'sig': self.problem_signature[:16],
            'target': self.target_token,
            'conf': round(self.confidence, 2),
            'usage': self.usage_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Recipe':
        """Deserialize from dictionary."""
        return cls(
            recipe_id=data['id'],
            seed_sequence=data['seeds'],
            operation_order=data['order'],
            problem_signature=data['sig'],
            target_token=data['target'],
            confidence=data.get('conf', 1.0),
            usage_count=data.get('usage', 0)
        )
    
    def size_bytes(self) -> int:
        """Estimate storage size."""
        return 50 + sum(len(s) for s in self.seed_sequence)

class RecipeDeduplicator:
    
    
    def __init__(self):
        self._recipes: Dict[str, Recipe] = {}  # signature → recipe
        self._usage_count: Dict[str, int] = {}  # signature → count
    
    def _compute_signature(self, seed_sequence: List[str]) -> str:
        
        canonical = "|".join(sorted(seed_sequence))
        return blake3_hash(canonical.encode()).hex()[:16]
    
    def store_or_update(self, recipe: Recipe) -> str:
        
        sig = self._compute_signature(recipe.seed_sequence)
        
        if sig in self._recipes:
            existing = self._recipes[sig]
            existing.confidence = max(existing.confidence, recipe.confidence)
            self._usage_count[sig] += 1
            return sig
        
        self._recipes[sig] = recipe
        self._usage_count[sig] = 1
        return sig
    
    def find_similar(self, seed_sequence: List[str], threshold: float = 0.8) -> List[Recipe]:
        """Find similar recipes based on seed overlap."""
        results = []
        for sig, recipe in self._recipes.items():
            set_a = set(seed_sequence)
            set_b = set(recipe.seed_sequence)
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            if union > 0:
                similarity = intersection / union
                if similarity >= threshold:
                    results.append(recipe)
        return results
    
    def get_by_signature(self, signature: str) -> Optional[Recipe]:
        """Get recipe by signature."""
        return self._recipes.get(signature)

class XORPeelingSearch:
    
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM, n_agents: int = 6):
        self.dim = dim
        self.n_agents = n_agents
        self.uint64_count = dim // 64
    
    def _compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute Hamming similarity between vectors."""
        return hamming_similarity(vec_a, vec_b)
    
    def _compute_null_ratio(self, vec: np.ndarray) -> float:
        """Compute ratio of zero bits in vector."""
        zero_bits = len(vec) * 64 - np.unpackbits(vec.view(np.uint8)).sum()
        return zero_bits / (len(vec) * 64)
    
    def peel_single(
        self, 
        target: np.ndarray, 
        candidate: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        
        residue = np.bitwise_xor(target, candidate)
        null_ratio = self._compute_null_ratio(residue)
        return residue, null_ratio
    
    def peel_chunk(
        self,
        target: np.ndarray,
        candidates: List[np.ndarray],
        top_k: int = 5
    ) -> List[Tuple[int, float, np.ndarray]]:
        
        results = []
        for i, candidate in enumerate(candidates):
            residue, score = self.peel_single(target, candidate)
            results.append((i, score, residue))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def parallel_peel(
        self,
        target: np.ndarray,
        candidates: List[np.ndarray]
    ) -> List[Tuple[int, float]]:
        
        chunk_size = max(1, len(candidates) // self.n_agents)
        all_results = []
        
        for agent_id in range(self.n_agents):
            start_idx = agent_id * chunk_size
            end_idx = min(start_idx + chunk_size, len(candidates))
            chunk = candidates[start_idx:end_idx]
            
            for i, candidate in enumerate(chunk):
                residue, score = self.peel_single(target, candidate)
                all_results.append((start_idx + i, score))
        
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results
    
    def search(
        self,
        target: np.ndarray,
        candidate_seeds: List[str],
        known_patterns: Optional[Dict[str, np.ndarray]] = None,
        max_iterations: int = 100,
        convergence_threshold: float = 0.95
    ) -> Tuple[List[str], float]:
        
        discovered_seeds = []
        current_target = target.copy()
        
        candidates = [seed_to_hypervector(s, self.dim) for s in candidate_seeds]
        
        for iteration in range(max_iterations):
            results = self.parallel_peel(current_target, candidates)
            
            if not results:
                break
            
            best_idx, best_score = results[0]
            
            if best_score < convergence_threshold:
                break
            
            best_seed = candidate_seeds[best_idx]
            discovered_seeds.append(best_seed)
            
            current_target = np.bitwise_xor(current_target, candidates[best_idx])
            
            if self._compute_null_ratio(current_target) > 0.99:
                break
            
            candidates.pop(best_idx)
            candidate_seeds.pop(best_idx)
            
            if not candidates:
                break
        
        final_similarity = self._compute_null_ratio(current_target)
        return discovered_seeds, final_similarity

class ResonatorNetwork:
    
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM, n_agents: int = 6):
        self.dim = dim
        self.n_agents = n_agents
        self.uint64_count = dim // 64
    
    def factorize(
        self,
        composite: np.ndarray,
        factor_candidates: List[List[np.ndarray]],
        max_iterations: int = 10,
        convergence_threshold: float = 0.95
    ) -> Tuple[List[np.ndarray], float]:
        
        n_factors = len(factor_candidates)
        if n_factors == 0:
            return [], 0.0
        
        similarity = 0.0
        
        estimates = []
        for candidates in factor_candidates:
            if candidates:
                idx = np.random.randint(len(candidates))
                estimates.append(candidates[idx].copy())
            else:
                estimates.append(np.zeros(self.uint64_count, dtype=np.uint64))
        
        for iteration in range(max_iterations):
            for i in range(n_factors):
                residual = composite.copy()
                for j, est in enumerate(estimates):
                    if j != i:
                        residual = np.bitwise_xor(residual, est)
                
                best_score = -1
                best_candidate = estimates[i]
                
                for candidate in factor_candidates[i]:
                    score = hamming_similarity(residual, candidate)
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
                
                estimates[i] = best_candidate.copy()
            
            reconstruction = estimates[0].copy()
            for est in estimates[1:]:
                reconstruction = np.bitwise_xor(reconstruction, est)
            
            similarity = hamming_similarity(composite, reconstruction)
            if similarity >= convergence_threshold:
                break
        
        return estimates, similarity

class RelationshipGuidedSearch:
    
    
    def __init__(self):
        self.relationships: Dict[str, Dict[RelationshipType, List[str]]] = {}
    
    def add_relationship(
        self, 
        seed: str, 
        rel_type: RelationshipType, 
        related_seed: str
    ):
        """Add a relationship between seeds."""
        if seed not in self.relationships:
            self.relationships[seed] = {rt: [] for rt in RelationshipType}
        self.relationships[seed][rel_type].append(related_seed)
    
    def get_similar(self, seed: str) -> List[str]:
        """Get seeds with SIMILAR relationship."""
        if seed in self.relationships:
            return self.relationships[seed].get(RelationshipType.SIMILAR, [])
        return []
    
    def get_opposite(self, seed: str) -> Optional[str]:
        """Get seed with OPPOSITE relationship."""
        if seed in self.relationships:
            opposites = self.relationships[seed].get(RelationshipType.OPPOSITE, [])
            return opposites[0] if opposites else None
        return None
    
    def get_composed_from(self, seed: str) -> List[str]:
        """Get seeds that COMPOSED from this seed."""
        if seed in self.relationships:
            return self.relationships[seed].get(RelationshipType.COMPOSED, [])
        return []
    
    def get_predicts(self, seed: str) -> List[str]:
        """Get seeds that this seed PREDICTS."""
        if seed in self.relationships:
            return self.relationships[seed].get(RelationshipType.PREDICTS, [])
        return []
    
    def suggest_candidates(
        self, 
        failed_candidates: List[str]
    ) -> List[str]:
        """
        Use relationships to suggest next candidates after failed peeling.
        """
        suggestions = []
        
        for failed in failed_candidates:
            similar = self.get_similar(failed)
            suggestions.extend(similar)
            
            opposite = self.get_opposite(failed)
            if opposite:
                suggestions.append(opposite)
            
            composed = self.get_composed_from(failed)
            suggestions.extend(composed)
            
            predicts = self.get_predicts(failed)
            suggestions.extend(predicts)
        
        return list(set(suggestions))  # Deduplicate

class CollisionShield:
    
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM, redundancy: int = 3):
        self.dim = dim
        self.redundancy = redundancy
        self.collision_threshold = 0.55
    
    def encode_with_redundancy(
        self, 
        vector: np.ndarray
    ) -> List[np.ndarray]:
        
        uint64_count = self.dim // 64
        redundant_vectors = [vector.copy()]
        
        for i in range(1, self.redundancy):
            shift = (i * uint64_count // self.redundancy) % uint64_count
            shifted = np.roll(vector, shift)
            redundant_vectors.append(shifted)
        
        return redundant_vectors
    
    def decode_with_redundancy(
        self, 
        redundant_vectors: List[np.ndarray]
    ) -> np.ndarray:
        """
        Decode from redundant vectors using majority vote.
        """
        if not redundant_vectors:
            return np.zeros(self.dim // 64, dtype=np.uint64)
        
        uint64_count = self.dim // 64
        aligned = []
        for i, vec in enumerate(redundant_vectors):
            shift = (i * uint64_count // self.redundancy) % uint64_count
            unshifted = np.roll(vec, -shift)
            aligned.append(unshifted)
        
        return bundle_vectors(aligned, self.dim)
    
    def check_collision(
        self, 
        vec_a: np.ndarray, 
        vec_b: np.ndarray
    ) -> bool:
        """
        Check if two vectors are too similar (collision).
        """
        similarity = hamming_similarity(vec_a, vec_b)
        return similarity > self.collision_threshold

class HierarchicalSearchEngine:
    
    
    def __init__(
        self,
        hdc_dim: int = DEFAULT_HDC_DIM,
        depths: Optional[List[int]] = None,
        early_stop_threshold: float = 0.99,
        use_gpu: bool = True
    ):
        self.hdc_dim = hdc_dim
        self.depths = depths or [10, 20, 50, 100]
        self.early_stop_threshold = early_stop_threshold
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        
        self.stats = {
            'searches_performed': 0,
            'early_stops': 0,
            'depth_usage': {d: 0 for d in self.depths},
            'avg_iterations': 0.0
        }
        self._iterations_history: List[int] = []
    
    def search(
        self,
        composite_vector: np.ndarray,
        codebook: Dict[str, List[np.ndarray]],
        search_func: Callable,
        target_accuracy: Optional[float] = None
    ) -> Tuple[Dict[str, Any], float]:
        """Perform hierarchical search with progressive refinement."""
        threshold = target_accuracy or self.early_stop_threshold
        self.stats['searches_performed'] += 1
        
        best_result = None
        best_confidence = 0.0
        
        for depth in self.depths:
            result, confidence = search_func(composite_vector, codebook, depth)
            
            self.stats['depth_usage'][depth] += 1
            self._iterations_history.append(depth)
            
            if confidence > best_confidence:
                best_result = result
                best_confidence = confidence
            
            if confidence >= threshold:
                self.stats['early_stops'] += 1
                break
        
        if self._iterations_history:
            self.stats['avg_iterations'] = float(np.mean(self._iterations_history))
        
        return best_result or {}, best_confidence
    
    def xor_peeling_search(
        self,
        composite_vector: np.ndarray,
        codebook: Dict[str, List[np.ndarray]],
        max_depth: int = 10
    ) -> Tuple[Dict[str, Any], float]:
        """XOR peeling search implementation."""
        result = {}
        residue = composite_vector.copy()
        total_similarity = 0.0
        roles_found = 0
        
        for role, candidates in codebook.items():
            best_match = None
            best_similarity = -1
            
            for candidate in candidates[:max_depth]:
                if self.use_gpu and _CUPY_AVAILABLE:
                    residue_gpu = cp.asarray(residue)
                    candidate_gpu = cp.asarray(candidate)
                    similarity = float(cp.mean(residue_gpu == candidate_gpu))
                else:
                    similarity = float(np.mean(residue == candidate))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = candidate
            
            if best_match is not None:
                result[role] = best_match
                total_similarity += best_similarity
                roles_found += 1
                residue = np.bitwise_xor(residue, best_match.astype(residue.dtype))
        
        confidence = total_similarity / max(roles_found, 1)
        return result, confidence

class EnhancedResonatorNetwork:
    
    
    def __init__(
        self,
        hdc_dim: int = DEFAULT_HDC_DIM,
        max_iterations: int = 300,
        min_iterations: int = 50,
        convergence_threshold: float = 0.995,
        stuck_detection_window: int = 20,
        use_gpu: bool = True
    ):
        self.hdc_dim = hdc_dim
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.convergence_threshold = convergence_threshold
        self.stuck_detection_window = stuck_detection_window
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        
        self.stats = {
            'factorizations_performed': 0,
            'early_convergences': 0,
            'stuck_escapes': 0,
            'avg_iterations': 0.0
        }
        self._iterations_history: List[int] = []
    
    def factorize_adaptive(
        self,
        bundled_vector: np.ndarray,
        codebooks: Dict[str, List[np.ndarray]],
        initial_estimates: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool]:
        """Factorize with adaptive iterations and convergence monitoring."""
        self.stats['factorizations_performed'] += 1
        
        estimates = initial_estimates or {}
        for role in codebooks.keys():
            if role not in estimates:
                candidates = codebooks[role]
                if candidates:
                    estimates[role] = candidates[0].copy()
        
        residue_history = []
        stuck_count = 0
        confidence = 0.0  # Initialize confidence
        
        for iteration in range(self.max_iterations):
            estimates, confidence = self._single_iteration(
                bundled_vector, codebooks, estimates
            )
            
            residue_history.append(confidence)
            
            if iteration >= self.min_iterations:
                if confidence >= self.convergence_threshold:
                    self.stats['early_convergences'] += 1
                    self._iterations_history.append(iteration)
                    self.stats['avg_iterations'] = float(np.mean(self._iterations_history))
                    return estimates, confidence, True
                
                if len(residue_history) >= self.stuck_detection_window:
                    recent = residue_history[-self.stuck_detection_window:]
                    improvement = max(recent) - min(recent)
                    
                    if improvement < 0.001:
                        stuck_count += 1
                        if stuck_count >= 3:
                            estimates = self._apply_perturbation(estimates, codebooks)
                            self.stats['stuck_escapes'] += 1
                            stuck_count = 0
        
        self._iterations_history.append(self.max_iterations)
        self.stats['avg_iterations'] = float(np.mean(self._iterations_history))
        return estimates, confidence, False
    
    def _single_iteration(
        self,
        bundled_vector: np.ndarray,
        codebooks: Dict[str, List[np.ndarray]],
        estimates: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Perform a single iteration of resonator factorization."""
        new_estimates = {}
        total_confidence = 0.0
        
        for role, candidates in codebooks.items():
            partial = np.zeros_like(bundled_vector)
            for r, est in estimates.items():
                if r != role:
                    partial = np.bitwise_xor(partial, est.astype(partial.dtype))
            
            residue = np.bitwise_xor(bundled_vector, partial)
            
            best_match = None
            best_similarity = -1
            
            for candidate in candidates:
                if self.use_gpu and _CUPY_AVAILABLE:
                    similarity = float(cp.mean(cp.asarray(residue) == cp.asarray(candidate)))
                else:
                    similarity = float(np.mean(residue == candidate))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = candidate
            
            if best_match is not None:
                new_estimates[role] = best_match.copy()
                total_confidence += best_similarity
            else:
                new_estimates[role] = estimates.get(role, np.zeros_like(bundled_vector))
        
        confidence = total_confidence / max(len(codebooks), 1)
        return new_estimates, confidence
    
    def _apply_perturbation(
        self,
        estimates: Dict[str, np.ndarray],
        codebooks: Dict[str, List[np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Apply perturbation to escape local minimum."""
        perturbed = {}
        for role, est in estimates.items():
            if role in codebooks and codebooks[role]:
                idx = np.random.randint(len(codebooks[role]))
                perturbed[role] = codebooks[role][idx].copy()
            else:
                perturbed[role] = est.copy()
        return perturbed

class SemanticCodebook:
    
    
    def __init__(
        self,
        hdc_dim: int = DEFAULT_HDC_DIM,
        expansion_factor: int = 4,
        use_semantic_clustering: bool = True,
        use_gpu: bool = True
    ):
        self.hdc_dim = hdc_dim
        self.expansion_factor = expansion_factor
        self.use_semantic_clustering = use_semantic_clustering
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        
        self.codebooks: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.flat_codebooks: Dict[str, List[np.ndarray]] = {}
        
        self.stats = {
            'patterns_stored': 0,
            'clusters_created': 0,
            'lookups_performed': 0
        }
    
    def add_pattern(
        self,
        role: str,
        pattern: np.ndarray,
        semantic_cluster: Optional[str] = None
    ) -> None:
        """Add a pattern to the codebook."""
        if role not in self.flat_codebooks:
            self.flat_codebooks[role] = []
            if self.use_semantic_clustering:
                self.codebooks[role] = {}
        
        self.flat_codebooks[role].append(pattern)
        
        if self.use_semantic_clustering and semantic_cluster:
            if semantic_cluster not in self.codebooks[role]:
                self.codebooks[role][semantic_cluster] = []
                self.stats['clusters_created'] += 1
            self.codebooks[role][semantic_cluster].append(pattern)
        
        self.stats['patterns_stored'] += 1
    
    def expand_codebook(
        self,
        role: str,
        base_patterns: List[np.ndarray],
        semantic_clusters: Optional[Dict[str, List[int]]] = None
    ) -> List[np.ndarray]:
        """Expand a codebook by generating variations."""
        expanded = list(base_patterns)
        
        for _ in range(self.expansion_factor - 1):
            for pattern in base_patterns:
                variation = pattern.copy()
                flip_indices = np.random.choice(
                    len(pattern),
                    size=int(len(pattern) * 0.1),
                    replace=False
                )
                variation[flip_indices] = 1 - variation[flip_indices]
                expanded.append(variation)
        
        self.flat_codebooks[role] = expanded
        
        if self.use_semantic_clustering and semantic_clusters:
            for cluster_name, indices in semantic_clusters.items():
                self.codebooks[role][cluster_name] = [
                    expanded[i] for i in indices if i < len(expanded)
                ]
        
        return expanded
    
    def get_candidates(
        self,
        role: str,
        semantic_cluster: Optional[str] = None
    ) -> List[np.ndarray]:
        """Get candidates for a role."""
        self.stats['lookups_performed'] += 1
        
        if self.use_semantic_clustering and semantic_cluster:
            if role in self.codebooks and semantic_cluster in self.codebooks[role]:
                return self.codebooks[role][semantic_cluster]
        
        return self.flat_codebooks.get(role, [])
    
    def get_codebook_for_search(self) -> Dict[str, List[np.ndarray]]:
        """Get flat codebook for search operations."""
        return self.flat_codebooks

class IterativeRefinementEngine:
    
    
    def __init__(
        self,
        hdc_dim: int = DEFAULT_HDC_DIM,
        passes: int = 3,
        residue_threshold: float = 0.01,
        use_gpu: bool = True
    ):
        self.hdc_dim = hdc_dim
        self.passes = passes
        self.residue_threshold = residue_threshold
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        
        self.stats = {
            'refinements_performed': 0,
            'early_convergences': 0,
            'avg_passes': 0.0
        }
        self._passes_history: List[int] = []
    
    def factorize_with_refinement(
        self,
        bundled_vector: np.ndarray,
        codebooks: Dict[str, List[np.ndarray]],
        factorize_func: Callable
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Factorize with iterative refinement."""
        self.stats['refinements_performed'] += 1
        
        residue = bundled_vector.copy()
        all_estimates = {}
        
        for pass_num in range(self.passes):
            pass_estimates, confidence = factorize_func(residue, codebooks)
            
            for role, value in pass_estimates.items():
                if role in all_estimates:
                    all_estimates[role] = self._combine_estimates(
                        all_estimates[role], value, pass_num
                    )
                else:
                    all_estimates[role] = value
            
            reconstructed = self._reconstruct(all_estimates)
            residue = np.bitwise_xor(bundled_vector, reconstructed.astype(bundled_vector.dtype))
            
            residue_norm = self._residue_norm(residue)
            if residue_norm < self.residue_threshold:
                self.stats['early_convergences'] += 1
                self._passes_history.append(pass_num + 1)
                break
        
        self._passes_history.append(self.passes)
        self.stats['avg_passes'] = float(np.mean(self._passes_history))
        
        final_confidence = self._compute_confidence(bundled_vector, all_estimates)
        
        return all_estimates, final_confidence
    
    def _combine_estimates(
        self,
        existing: np.ndarray,
        new: np.ndarray,
        pass_num: int
    ) -> np.ndarray:
        """Combine estimates from multiple passes."""
        weight = 1.0 / (pass_num + 1)
        combined = np.where(
            np.random.random(len(existing)) < weight,
            new,
            existing
        )
        return combined.astype(existing.dtype)
    
    def _reconstruct(self, estimates: Dict[str, np.ndarray]) -> np.ndarray:
        """Reconstruct bundled vector from estimates."""
        if not estimates:
            return np.zeros(self.hdc_dim, dtype=np.int8)
        
        result = np.zeros_like(list(estimates.values())[0])
        for estimate in estimates.values():
            result = np.bitwise_xor(result, estimate.astype(result.dtype))
        
        return result
    
    def _residue_norm(self, residue: np.ndarray) -> float:
        """Compute norm of residue."""
        return float(np.sum(residue != 0)) / len(residue)
    
    def _compute_confidence(
        self,
        original: np.ndarray,
        estimates: Dict[str, np.ndarray]
    ) -> float:
        """Compute confidence of estimates."""
        reconstructed = self._reconstruct(estimates)
        
        if self.use_gpu and _CUPY_AVAILABLE:
            similarity = float(cp.mean(cp.asarray(original) == cp.asarray(reconstructed)))
        else:
            similarity = float(np.mean(original == reconstructed))
        
        return similarity

class ParallelMultiPathSearch:
    
    
    def __init__(
        self,
        hdc_dim: int = DEFAULT_HDC_DIM,
        num_paths: int = 8,
        use_multiprocessing: bool = False,
        use_gpu: bool = True
    ):
        self.hdc_dim = hdc_dim
        self.num_paths = num_paths
        self.use_multiprocessing = use_multiprocessing
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        
        self.stats = {
            'searches_performed': 0,
            'best_path_found': 0,
            'avg_paths_used': 0.0
        }
    
    def search_parallel(
        self,
        bundled_vector: np.ndarray,
        codebooks: Dict[str, List[np.ndarray]],
        search_func: Callable
    ) -> Tuple[Dict[str, Any], float]:
        """Search multiple factorization paths in parallel."""
        self.stats['searches_performed'] += 1
        
        initial_estimates = self._generate_hypotheses(codebooks, self.num_paths)
        
        results = []
        
        if self.use_multiprocessing:
            with Pool(self.num_paths) as pool:
                args = [
                    (bundled_vector, codebooks, init)
                    for init in initial_estimates
                ]
                results = pool.starmap(search_func, args)
        else:
            with ThreadPoolExecutor(max_workers=self.num_paths) as executor:
                futures = [
                    executor.submit(search_func, bundled_vector, codebooks, init)
                    for init in initial_estimates
                ]
                for future in as_completed(futures):
                    results.append(future.result())
        
        if results:
            best = max(results, key=lambda r: r[1] if isinstance(r, tuple) else r.get('confidence', 0))
            if isinstance(best, tuple):
                return best
            else:
                return best.get('result', {}), best.get('confidence', 0)
        
        return {}, 0.0
    
    def _generate_hypotheses(
        self,
        codebooks: Dict[str, List[np.ndarray]],
        num_hypotheses: int
    ) -> List[Dict[str, np.ndarray]]:
        """Generate multiple initial hypotheses."""
        hypotheses = []
        
        for i in range(num_hypotheses):
            hypothesis = {}
            for role, candidates in codebooks.items():
                if candidates:
                    idx = i % len(candidates)
                    hypothesis[role] = candidates[idx].copy()
            hypotheses.append(hypothesis)
        
        return hypotheses

class EnhancedCollisionShield:
    
    
    def __init__(
        self,
        hdc_dim: int = DEFAULT_HDC_DIM,
        safety_margin: float = 0.1,
        min_hamming_distance_ratio: float = 0.4
    ):
        self.hdc_dim = hdc_dim
        self.safety_margin = safety_margin
        self.min_hamming_distance = int(hdc_dim * min_hamming_distance_ratio)
        
        self._registered_vectors: Dict[str, np.ndarray] = {}
        
        self.stats = {
            'vectors_registered': 0,
            'collisions_detected': 0,
            'safety_checks': 0
        }
    
    def register_vector(
        self,
        seed: str,
        vector: np.ndarray
    ) -> bool:
        """Register a vector for collision tracking."""
        is_safe, min_distance, closest_match = self.check_vector_safety(vector)
        
        if not is_safe:
            self.stats['collisions_detected'] += 1
            return False
        
        self._registered_vectors[seed] = vector.copy()
        self.stats['vectors_registered'] += 1
        return True
    
    def check_vector_safety(
        self,
        vector: np.ndarray
    ) -> Tuple[bool, float, Optional[str]]:
        """Proactively check if a vector is safe from collisions."""
        self.stats['safety_checks'] += 1
        
        min_distance = float('inf')
        closest_match = None
        
        for seed, registered in self._registered_vectors.items():
            distance = self._hamming_distance(vector, registered)
            if distance < min_distance:
                min_distance = distance
                closest_match = seed
        
        is_safe = min_distance > self.min_hamming_distance or min_distance == float('inf')
        
        return is_safe, min_distance, closest_match
    
    def _hamming_distance(
        self,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> int:
        """Compute Hamming distance between two vectors."""
        return int(np.sum(v1 != v2))
    
    def get_collision_probability(self, num_vectors: int) -> float:
        """Estimate collision probability for given number of vectors."""
        d = self.hdc_dim
        n = num_vectors
        
        exponent = -(n ** 2) / (2 * (2 ** d))
        probability = 1 - np.exp(exponent)
        
        return float(probability)

class AccuracyEngine:
    
    
    def __init__(self, config: 'AccuracyConfig'):
        self.config = config
        
        self.hierarchical_search = HierarchicalSearchEngine(
            hdc_dim=config.hdc_dim,
            depths=config.hierarchical_depths,
            early_stop_threshold=config.early_stop_threshold,
            use_gpu=config.use_gpu
        )
        
        self.resonator = EnhancedResonatorNetwork(
            hdc_dim=config.hdc_dim,
            max_iterations=config.max_resonator_iterations,
            min_iterations=config.min_resonator_iterations,
            convergence_threshold=config.convergence_threshold,
            stuck_detection_window=config.stuck_detection_window,
            use_gpu=config.use_gpu
        )
        
        self.codebook = SemanticCodebook(
            hdc_dim=config.hdc_dim,
            expansion_factor=config.codebook_expansion_factor,
            use_semantic_clustering=config.semantic_clustering,
            use_gpu=config.use_gpu
        )
        
        self.refinement = IterativeRefinementEngine(
            hdc_dim=config.hdc_dim,
            passes=config.refinement_passes,
            residue_threshold=config.residue_threshold,
            use_gpu=config.use_gpu
        )
        
        self.parallel_search = ParallelMultiPathSearch(
            hdc_dim=config.hdc_dim,
            num_paths=config.parallel_paths,
            use_multiprocessing=config.use_multiprocessing,
            use_gpu=config.use_gpu
        )
        
        self.collision_shield = EnhancedCollisionShield(
            hdc_dim=config.hdc_dim,
            min_hamming_distance_ratio=config.min_hamming_distance_ratio
        )
        
        self.stats = {
            'searches': 0,
            'factorizations': 0,
            'refinements': 0,
            'parallel_searches': 0,
            'collisions_prevented': 0,
            'avg_accuracy': 0.0
        }
        self._accuracy_history: List[float] = []
    
    def search(
        self,
        composite_vector: np.ndarray,
        codebooks: Dict[str, List[np.ndarray]],
        use_refinement: bool = True,
        use_parallel: bool = True
    ) -> Tuple[Dict[str, Any], float]:
        
        self.stats['searches'] += 1
        
        def search_func(vec, cbs, depth):
            return self.hierarchical_search.xor_peeling_search(vec, cbs, depth)
        
        result, confidence = self.hierarchical_search.search(
            composite_vector, codebooks, search_func, self.config.target_accuracy
        )
        
        if use_refinement and confidence < self.config.target_accuracy:
            def factorize_func(vec, cbs):
                estimates, conf, _ = self.resonator.factorize_adaptive(vec, cbs)
                return estimates, conf
            
            result, confidence = self.refinement.factorize_with_refinement(
                composite_vector, codebooks, factorize_func
            )
            self.stats['refinements'] += 1
        
        if use_parallel and confidence < self.config.target_accuracy:
            def parallel_search_func(vec, cbs, init):
                estimates, conf, _ = self.resonator.factorize_adaptive(vec, cbs, init)
                return estimates, conf
            
            result, confidence = self.parallel_search.search_parallel(
                composite_vector, codebooks, parallel_search_func
            )
            self.stats['parallel_searches'] += 1
        
        self._accuracy_history.append(confidence)
        self.stats['avg_accuracy'] = float(np.mean(self._accuracy_history))
        
        return result, confidence
    
    def factorize(
        self,
        bundled_vector: np.ndarray,
        codebooks: Dict[str, List[np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], float, bool]:
        
        self.stats['factorizations'] += 1
        return self.resonator.factorize_adaptive(bundled_vector, codebooks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            **self.stats,
            'hierarchical': self.hierarchical_search.stats,
            'resonator': self.resonator.stats,
            'codebook': self.codebook.stats,
            'refinement': self.refinement.stats,
            'parallel_search': self.parallel_search.stats,
            'collision_shield': self.collision_shield.stats
        }

class HDCLanguageModel:
    
    
    def __init__(self, config: HDCConfig):
        self.config = config
        self.dim = config.hdc_dim
        self.uint64_count = self.dim // 64
        
        self.use_gpu = config.use_gpu_acceleration and _CUPY_AVAILABLE
        if self.use_gpu:
            self.gpu_manager = get_gpu_manager(use_gpu=True, device_id=config.gpu_device_id)
            self.batch_ops = get_batch_ops(self.gpu_manager, self.dim)
            self.xp = self.gpu_manager.xp
            print(f"HDCLanguageModel: GPU acceleration enabled")
        else:
            self.gpu_manager = None
            self.batch_ops = None
            self.xp = np
            print(f"HDCLanguageModel: Using CPU mode")
        
        self._token_cache: Dict[int, np.ndarray] = {}
        
        self._position_cache: Dict[int, np.ndarray] = {}
        
        self._gpu_token_matrix = None
        self._gpu_position_matrix = None
        
        self.seed_registry = SeedRegistry()
        
        self.recipe_deduplicator = RecipeDeduplicator()
        self.recipes: Dict[str, Recipe] = {}
        self.recipe_storage_size = 0
        
        self.ngram_stats: Dict[Tuple[int, ...], int] = {}
        
        self.xor_peeler = XORPeelingSearch(
            dim=self.dim,
            n_agents=config.n_search_agents
        )
        
        self.resonator = ResonatorNetwork(
            dim=self.dim,
            n_agents=config.resonator_agents
        )
        
        self.relationship_search = RelationshipGuidedSearch()
        
        self.collision_shield = CollisionShield(
            dim=self.dim,
            redundancy=config.holographic_redundancy
        )
        
        self.enhanced_collision_shield = EnhancedCollisionShield(
            hdc_dim=self.dim,
            min_hamming_distance_ratio=config.min_hamming_distance_ratio
        )
        
        if config.use_hierarchical_search or config.use_enhanced_resonator:
            accuracy_config = AccuracyConfig(
                target_accuracy=config.target_accuracy,
                hdc_dim=self.dim,
                hierarchical_depths=config.hierarchical_depths,
                max_resonator_iterations=config.max_resonator_iterations,
                min_resonator_iterations=config.min_resonator_iterations,
                stuck_detection_window=config.stuck_detection_window,
                refinement_passes=config.refinement_passes,
                residue_threshold=config.residue_threshold,
                parallel_paths=config.parallel_paths,
                codebook_expansion_factor=config.codebook_expansion_factor,
                min_hamming_distance_ratio=config.min_hamming_distance_ratio,
                use_gpu=config.use_gpu_acceleration
            )
            self.accuracy_engine = AccuracyEngine(accuracy_config)
        else:
            self.accuracy_engine = None
        
        self.semantic_codebook = SemanticCodebook(
            hdc_dim=self.dim,
            expansion_factor=config.codebook_expansion_factor,
            use_gpu=config.use_gpu_acceleration
        )
        
        self.hadamard_basis = WalshHadamardBasis(dim=self.dim, use_gpu=self.use_gpu)
        
        self.difficulty_memory = DifficultyMemory(dim=self.dim)
        
        self.context_patterns: Dict[str, List[int]] = {}
        
        self._build_token_relationships()
    
    def _build_token_relationships(self):
        """Build relationship graph between tokens."""
        for token_id in range(min(100, self.config.vocab_size)):  # Limit for efficiency
            token_seed = f"token_{token_id}"
            
            if token_id > 0:
                self.relationship_search.add_relationship(
                    token_seed, RelationshipType.SIMILAR, f"token_{token_id - 1}"
                )
            if token_id < self.config.vocab_size - 1:
                self.relationship_search.add_relationship(
                    token_seed, RelationshipType.SIMILAR, f"token_{token_id + 1}"
                )
    
    def get_token_vector(self, token_id: int) -> np.ndarray:
        
        if token_id in self._token_cache:
            return self._token_cache[token_id]
        
        index, row = self.hadamard_basis.get_row_from_string(
            f"token_{token_id}",
            packed=True,
            seed=self.config.seed  # Pass seed for different orthogonal mappings
        )
        
        self.seed_registry.get_or_create(f"token_{token_id}")
        
        if len(self._token_cache) < 10000:  # Limit cache size
            self._token_cache[token_id] = row
        
        return row
    
    def get_position_vector(self, position: int) -> np.ndarray:
        
        if position in self._position_cache:
            return self._position_cache[position]
        
        seed_offset = self.config.seed % self.dim if self.config.seed else 0
        row_index = (position + seed_offset) % self.dim
        row = self.hadamard_basis.get_row(row_index, packed=True)
        
        self.seed_registry.get_or_create(f"pos_{position}")
        
        if len(self._position_cache) < 1000:
            self._position_cache[position] = row
        
        return row
    
    def encode_context(
        self, 
        tokens: List[int],
        use_temporal: bool = True
    ) -> np.ndarray:
        
        if not tokens:
            return np.zeros(self.uint64_count, dtype=np.uint64)
        
        if use_temporal and self.config.temporal_folding:
            events = []
            for i, token_id in enumerate(tokens):
                token_vec = self.get_token_vector(token_id)
                pos_vec = self.get_position_vector(i)
                bound = xor_bind(token_vec, pos_vec)
                events.append(bound)
            
            return circular_temporal_encode(events, self.dim)
        else:
            vectors = []
            for i, token_id in enumerate(tokens):
                token_vec = self.get_token_vector(token_id)
                pos_vec = self.get_position_vector(i)
                bound = xor_bind(token_vec, pos_vec)
                vectors.append(bound)
            
            return xor_bind_sequence(vectors)
    
    def predict_next_token_probabilities(
        self,
        context_tokens: List[int],
        temperature: float = 1.0
    ) -> np.ndarray:
        
        probs = self.xp.ones(self.config.vocab_size) / self.config.vocab_size
        
        if self.recipes:
            recipe_probs = self._recall_from_recipes(context_tokens)
            if recipe_probs is not None:
                recipe_weight = 0.7
                probs = recipe_weight * recipe_probs + (1 - recipe_weight) * probs
        
        if self.config.use_resonator:
            resonator_probs = self._resonator_prediction(context_tokens)
            if resonator_probs is not None:
                resonator_weight = 0.5
                probs = resonator_weight * resonator_probs + (1 - resonator_weight) * probs
        
        if len(context_tokens) >= 1 and self.ngram_stats:
            ngram_probs = self._ngram_prediction(context_tokens)
            if ngram_probs is not None:
                ngram_weight = 0.4
                probs = ngram_weight * ngram_probs + (1 - ngram_weight) * probs
        
        context_vec = self.encode_context(context_tokens)
        similarities = self.xp.zeros(self.config.vocab_size)
        for token_id in range(self.config.vocab_size):
            token_vec = self.get_token_vector(token_id)
            similarities[token_id] = hamming_similarity(context_vec, token_vec)
        
        sim_probs = self._softmax_with_temperature(similarities, temperature)
        sim_weight = 0.1
        probs = sim_weight * sim_probs + (1 - sim_weight) * probs
        
        probs = self.xp.maximum(probs, self.config.min_probability)
        probs = probs / self.xp.sum(probs)
        
        return probs
    
    def _recall_from_recipes(self, context_tokens: List[int]) -> Optional[np.ndarray]:
        """
        Recall prediction from stored recipes using XOR unbinding.
        """
        if not self.recipes:
            return None
        
        for ctx_len in range(min(len(context_tokens), 5), 0, -1):
            context = context_tokens[-ctx_len:]
            sig = self._compute_signature(context)
            
            if sig in self.recipes:
                recipe = self.recipes[sig]
                
                probs = self.xp.zeros(self.config.vocab_size)
                probs[recipe.target_token] = recipe.confidence
                
                target_vec = self.get_token_vector(recipe.target_token)
                for token_id in range(self.config.vocab_size):
                    if token_id != recipe.target_token:
                        token_vec = self.get_token_vector(token_id)
                        sim = hamming_similarity(target_vec, token_vec)
                        if sim > 0.6:
                            probs[token_id] = sim * 0.1
                
                return probs
        
        return None
    
    def _resonator_prediction(
        self,
        context_tokens: List[int]
    ) -> Optional[np.ndarray]:
        
        if len(context_tokens) < 2:
            return None
        
        context_vec = self.encode_context(context_tokens)
        
        token_candidates = [
            [self.get_token_vector(t) for t in range(self.config.vocab_size)]
        ]
        
        if self.accuracy_engine is not None:
            codebooks = {
                'token': [self.get_token_vector(t) for t in range(self.config.vocab_size)]
            }
            
            factors, confidence, converged = self.accuracy_engine.factorize(
                context_vec,
                codebooks
            )
            
            if factors and 'token' in factors:
                probs = self.xp.zeros(self.config.vocab_size)
                factor_vec = factors['token']
                
                for token_id in range(self.config.vocab_size):
                    token_vec = self.get_token_vector(token_id)
                    sim = hamming_similarity(factor_vec, token_vec)
                    probs[token_id] = sim
                
                if confidence > 0.6:
                    probs = probs ** 2
                elif confidence < 0.4:
                    probs = probs ** 0.5
                
                if self.xp.sum(probs) > 0:
                    probs = probs / self.xp.sum(probs)
                    return probs
            
            return None
        
        factors, confidence = self.resonator.factorize(
            context_vec,
            token_candidates,
            max_iterations=self.config.resonator_iterations
        )
        
        if confidence < 0.5:
            return None
        
        probs = self.xp.zeros(self.config.vocab_size)
        for token_id in range(self.config.vocab_size):
            token_vec = self.get_token_vector(token_id)
            sim = hamming_similarity(factors[0] if factors else context_vec, token_vec)
            probs[token_id] = sim
        
        if self.xp.sum(probs) > 0:
            probs = probs / self.xp.sum(probs)
            return probs
        
        return None
    
    def _ngram_prediction(self, context_tokens: List[int]) -> Optional[np.ndarray]:
        """Predict using n-gram statistics."""
        probs = self.xp.zeros(self.config.vocab_size)
        found_any = False
        
        for n in range(min(4, len(context_tokens)), 0, -1):
            ngram = tuple(context_tokens[-n:])
            
            for next_ngram, next_count in self.ngram_stats.items():
                if len(next_ngram) == n + 1 and next_ngram[:n] == ngram:
                    next_token = next_ngram[-1]
                    probs[next_token] += next_count * (n / 4.0)
                    found_any = True
        
        if found_any:
            total = self.xp.sum(probs)
            if total > 0:
                probs = probs / total
                return probs
        
        return None
    
    def _softmax_with_temperature(
        self,
        similarities: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Convert similarities to probabilities via softmax."""
        scaled = similarities * self.config.similarity_scale / temperature
        scaled = scaled - self.xp.max(scaled)
        exp_scores = self.xp.exp(scaled)
        probs = exp_scores / self.xp.sum(exp_scores)
        probs = self.xp.maximum(probs, self.config.min_probability)
        probs = probs / self.xp.sum(probs)
        return probs
    
    def learn_pattern(
        self,
        context: List[int],
        target: int,
        use_peeling: bool = True
    ) -> None:
        
        import time as time_module
        
        start_time = time_module.perf_counter()
        
        context_vec = self.encode_context(context)
        target_vec = self.get_token_vector(target)
        
        profile = self.difficulty_memory.estimate_difficulty(context_vec, target_vec)
        time_budget = self.difficulty_memory.get_time_budget(profile)
        
        pattern = xor_bind(context_vec, target_vec)
        
        if self.enhanced_collision_shield is not None:
            is_safe, min_distance, closest_match = self.enhanced_collision_shield.check_vector_safety(pattern)
            if not is_safe:
                self.enhanced_collision_shield.stats['collisions_detected'] += 1
        
        if self.semantic_codebook is not None:
            semantic_cluster = f"target_{target % 100}"  # Cluster by target token group
            self.semantic_codebook.add_pattern(
                role='pattern',
                pattern=pattern,
                semantic_cluster=semantic_cluster
            )
        
        discovered_seeds = None
        confidence = 0.0
        
        if use_peeling and len(context) > 0:
            candidate_seeds = []
            for i, tok in enumerate(context[-5:]):  # Last 5 tokens
                candidate_seeds.append(f"token_{tok}")
                candidate_seeds.append(f"pos_{i}")  # Updated to match new position seed format
            candidate_seeds.append(f"token_{target}")
            
            adjusted_iterations = min(
                self.config.max_peeling_iterations,
                int(time_budget.max_iterations * (1.0 if profile.difficulty_class == DifficultyClass.MEDIUM else 1.5 if profile.difficulty_class == DifficultyClass.HARD else 0.75))
            )
            
            discovered_seeds, confidence = self.xor_peeler.search(
                pattern,
                candidate_seeds,
                max_iterations=adjusted_iterations,
                convergence_threshold=self.config.convergence_threshold
            )
            
            if discovered_seeds and confidence > 0.5:
                recipe_id = f"pattern_{len(self.recipes)}"
                recipe = Recipe(
                    recipe_id=recipe_id,
                    seed_sequence=discovered_seeds,
                    operation_order=list(range(len(discovered_seeds))),
                    problem_signature=self._compute_signature(context),
                    target_token=target,
                    confidence=confidence
                )
                
                sig = self.recipe_deduplicator.store_or_update(recipe)
                if sig not in self.recipes:
                    self.recipes[sig] = recipe
                    self.recipe_storage_size += recipe.size_bytes()
        else:
            recipe_id = f"pattern_{len(self.recipes)}"
            recipe = Recipe(
                recipe_id=recipe_id,
                seed_sequence=[f"token_{target}"],
                operation_order=[0],
                problem_signature=self._compute_signature(context),
                target_token=target,
                confidence=1.0
            )
            
            sig = self._compute_signature(context)
            if sig not in self.recipes:
                self.recipes[sig] = recipe
                self.recipe_storage_size += recipe.size_bytes()
        
        elapsed_time_ms = (time_module.perf_counter() - start_time) * 1000
        
        self.difficulty_memory.record_solve(
            input_vec=context_vec,
            output_vec=target_vec,
            solve_time_ms=elapsed_time_ms,
            strategy="xor_peeling" if use_peeling else "direct",
            success=discovered_seeds is not None and len(discovered_seeds) > 0,
            search_depth=adjusted_iterations if use_peeling else 0,
            iterations=adjusted_iterations if use_peeling else 0
        )
        
        if len(context) >= 1:
            for n in range(1, min(4, len(context) + 1)):
                continuation = tuple(context[-n:] + [target])
                self.ngram_stats[continuation] = self.ngram_stats.get(continuation, 0) + 1
    
    def _compute_signature(self, tokens: List[int]) -> str:
        """Compute signature for a token sequence."""
        data = json.dumps(tokens).encode()
        return blake3_hash(data).hex()[:16]
    
    def _ensure_gpu_matrices(self) -> None:
        """Ensure GPU matrices are initialized for batch operations."""
        if not self.use_gpu or self.batch_ops is None:
            return
        
        if self._gpu_token_matrix is None:
            self._gpu_token_matrix = self.batch_ops.build_token_matrix(self.config.vocab_size)
        
        if self._gpu_position_matrix is None:
            self._gpu_position_matrix = self.batch_ops.build_position_matrix(self.config.max_context_length)
    
    def learn_patterns_batch(
        self,
        contexts: List[List[int]],
        targets: List[int],
        use_peeling: bool = False  # Peeling is slow on GPU, disabled by default for batch
    ) -> None:
        
        if not self.use_gpu or self.batch_ops is None:
            for context, target in zip(contexts, targets):
                self.learn_pattern(context, target, use_peeling=use_peeling)
            return
        
        self._ensure_gpu_matrices()
        
        batch_size = len(contexts)
        if batch_size == 0:
            return
        
        patterns, target_vecs = self.batch_ops.batch_learn_patterns(
            contexts, targets,
            self._gpu_token_matrix,
            self._gpu_position_matrix
        )
        
        patterns_cpu = self.gpu_manager.to_cpu(patterns)
        
        for i, (context, target) in enumerate(zip(contexts, targets)):
            pattern = patterns_cpu[i]
            
            if self.semantic_codebook is not None:
                semantic_cluster = f"target_{target % 100}"
                self.semantic_codebook.add_pattern(
                    role='pattern',
                    pattern=pattern,
                    semantic_cluster=semantic_cluster
                )
            
            recipe_id = f"pattern_{len(self.recipes)}"
            recipe = Recipe(
                recipe_id=recipe_id,
                seed_sequence=[f"token_{target}"],
                operation_order=[0],
                problem_signature=self._compute_signature(context),
                target_token=target,
                confidence=1.0
            )
            
            sig = self._compute_signature(context)
            if sig not in self.recipes:
                self.recipes[sig] = recipe
                self.recipe_storage_size += recipe.size_bytes()
            
            if len(context) >= 1:
                for n in range(1, min(4, len(context) + 1)):
                    continuation = tuple(context[-n:] + [target])
                    self.ngram_stats[continuation] = self.ngram_stats.get(continuation, 0) + 1
    
    def predict_batch(
        self,
        contexts: List[List[int]],
        temperature: float = 1.0,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        if not self.use_gpu or self.batch_ops is None:
            probs_list = []
            for context in contexts:
                probs = self.predict_next_token_probabilities(context, temperature)
                probs_list.append(probs)
            probs = np.stack(probs_list, axis=0)
            top_indices = np.argsort(probs, axis=-1)[:, ::-1][:, :top_k]
            return probs, top_indices
        
        self._ensure_gpu_matrices()
        
        probs_gpu, top_indices_gpu = self.batch_ops.batch_predict(
            contexts,
            self._gpu_token_matrix,
            self._gpu_position_matrix,
            temperature=temperature,
            top_k=top_k
        )
        
        probs = self.gpu_manager.to_cpu(probs_gpu)
        top_indices = self.gpu_manager.to_cpu(top_indices_gpu)
        
        return probs, top_indices
    
    def save_recipes(self, path: str) -> None:
        """Save learned recipes to file."""
        data = {
            'recipes': {k: v.to_dict() for k, v in self.recipes.items()},
            'ngram_stats': {str(k): v for k, v in self.ngram_stats.items()},
            'seed_registry': self.seed_registry.to_dict(),
            'config': {
                'hdc_dim': self.dim,
                'vocab_size': self.config.vocab_size,
                'max_context_length': self.config.max_context_length
            }
        }
        
        raw = json.dumps(data).encode()
        compressed = zlib.compress(raw, self.config.recipe_compression_level)
        
        with open(path, 'wb') as f:
            f.write(compressed)
    
    def load_recipes(self, path: str) -> None:
        """Load learned recipes from file."""
        if not os.path.exists(path):
            return
        
        with open(path, 'rb') as f:
            compressed = f.read()
        
        raw = zlib.decompress(compressed)
        data = json.loads(raw.decode())
        
        self.recipes = {
            k: Recipe.from_dict(v) for k, v in data.get('recipes', {}).items()
        }
        
        for k, v in data.get('ngram_stats', {}).items():
            key = eval(k)
            self.ngram_stats[key] = v
        
        if 'seed_registry' in data:
            self.seed_registry = SeedRegistry.from_dict(data['seed_registry'])

def build_sentencepiece_luts(sp, vocab_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build lookup tables for byte counting."""
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=bool)
    is_boundary_token = np.ones((table_size,), dtype=bool)
    
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    
    return base_bytes, has_leading_space, is_boundary_token

def load_data_shard(file: Path):
    """Load a data shard."""
    with open(file, "rb") as f:
        header = f.read(256)
        magic = struct.unpack('<I', header[:4])[0]
        if magic != 20240520:
            raise ValueError(f"Invalid magic number in {file}")
        vocab_size = struct.unpack('<I', header[4:8])[0]
        token_count = struct.unpack('<Q', header[8:16])[0]
        tokens = np.frombuffer(f.read(token_count * 2), dtype=np.uint16)
    return tokens

def load_validation_tokens(pattern: str, seq_len: int):
    """Load validation tokens."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    
    all_tokens = []
    for file in files:
        tokens = load_data_shard(Path(file))
        all_tokens.append(tokens)
    
    all_tokens = np.concatenate(all_tokens)
    
    n_seqs = len(all_tokens) // seq_len
    return all_tokens[:n_seqs * seq_len].reshape(n_seqs, seq_len)

def evaluate_bpb(
    model: HDCLanguageModel,
    val_tokens: np.ndarray,
    sp,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary_token: np.ndarray,
    batch_size: int = 64,
    max_batches: Optional[int] = None
) -> Tuple[float, float]:
    
    
    total_bits = 0.0
    total_bytes = 0
    total_nats = 0.0  # For cross-entropy loss
    total_tokens = 0
    
    n_seqs = len(val_tokens)
    
    if model.use_gpu:
        gpu_batch_size = min(batch_size * 4, 256)
        
        for batch_idx in range(0, n_seqs, batch_size):
            if max_batches and batch_idx >= max_batches * batch_size:
                break
            
            batch_end = min(batch_idx + batch_size, n_seqs)
            batch = val_tokens[batch_idx:batch_end]
            
            all_contexts = []
            all_targets = []
            all_bytes = []
            
            for seq in batch:
                for i in range(len(seq) - 1):
                    context = seq[:i+1].tolist()
                    target = int(seq[i+1])
                    all_contexts.append(context)
                    all_targets.append(target)
                    
                    if target < len(base_bytes):
                        bytes_for_token = base_bytes[target]
                        if has_leading_space[target]:
                            bytes_for_token += 1
                        all_bytes.append(max(1, bytes_for_token))
                    else:
                        all_bytes.append(1)
            
            for i in range(0, len(all_contexts), gpu_batch_size):
                sub_contexts = all_contexts[i:i + gpu_batch_size]
                sub_targets = all_targets[i:i + gpu_batch_size]
                sub_bytes = all_bytes[i:i + gpu_batch_size]
                
                probs, _ = model.predict_batch(sub_contexts)
                
                for j, (target, bytes_for_token) in enumerate(zip(sub_targets, sub_bytes)):
                    prob = max(probs[j, target], model.config.min_probability)
                    total_bits += -math.log2(prob)
                    total_nats += -math.log(prob)
                    total_tokens += 1
                    total_bytes += bytes_for_token
    else:
        for batch_idx in range(0, n_seqs, batch_size):
            if max_batches and batch_idx >= max_batches * batch_size:
                break
            
            batch_end = min(batch_idx + batch_size, n_seqs)
            batch = val_tokens[batch_idx:batch_end]
            
            for seq in batch:
                for i in range(len(seq) - 1):
                    context = seq[:i+1].tolist()
                    target = int(seq[i+1])
                    
                    probs = model.predict_next_token_probabilities(context)
                    
                    prob = max(probs[target], model.config.min_probability)
                    bits = -math.log2(prob)
                    total_bits += bits
                    
                    nats = -math.log(prob)
                    total_nats += nats
                    total_tokens += 1
                    
                    if target < len(base_bytes):
                        bytes_for_token = base_bytes[target]
                        if has_leading_space[target]:
                            bytes_for_token += 1  # Space character
                        total_bytes += max(1, bytes_for_token)
                    else:
                        total_bytes += 1
    
    if total_bytes == 0:
        return float('inf'), float('inf')
    
    bpb = total_bits / total_bytes
    val_loss = total_nats / total_tokens if total_tokens > 0 else float('inf')
    
    return bpb, val_loss

class DistributedTokenLoader:
    """Distributed token loader for multi-GPU training."""
    
    def __init__(self, pattern: str, rank: int = 0, world_size: int = 1):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern}")
        self.rank = rank
        self.world_size = world_size
        self.current_file_idx = rank % len(self.files)
        self.current_tokens = None
        self.current_pos = 0
        self._load_current_file()
    
    def _load_current_file(self):
        """Load current file."""
        self.current_tokens = load_data_shard(Path(self.files[self.current_file_idx]))
        self.current_pos = 0
    
    def next_batch(self, batch_tokens: int, seq_len: int) -> Tuple[List[List[int]], List[int]]:
        """Get next batch of tokens."""
        contexts: List[List[int]] = []
        targets: List[int] = []
        
        tokens_needed = batch_tokens * (seq_len + 1)
        
        while len(contexts) < batch_tokens:
            if self.current_tokens is None:
                self._load_current_file()
                continue
            
            if self.current_pos + seq_len + 1 >= len(self.current_tokens):
                self.current_file_idx = (self.current_file_idx + self.world_size) % len(self.files)
                self._load_current_file()
                continue
            
            start = self.current_pos
            end = start + seq_len + 1
            
            if self.current_tokens is not None and end <= len(self.current_tokens):
                seq = self.current_tokens[start:end]
                contexts.append(seq[:-1].tolist())
                targets.append(int(seq[-1]))
                self.current_pos = end
            else:
                self.current_file_idx = (self.current_file_idx + self.world_size) % len(self.files)
                self._load_current_file()
        
        return contexts, targets

class AsyncTokenLoader:
    
    
    def __init__(self, pattern: str, rank: int = 0, world_size: int = 1,
                 prefetch_batches: int = 2):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern}")
        self.rank = rank
        self.world_size = world_size
        self.current_file_idx = rank % len(self.files)
        self.current_tokens = None
        self.current_pos = 0
        
        self.prefetch_batches = prefetch_batches
        self._prefetch_queue = []
        self._prefetch_thread = None
        self._stop_prefetch = False
        self._prefetch_lock = None
        self._prefetch_condition = None
        
        self._load_current_file()
    
    def _load_current_file(self):
        """Load current file."""
        self.current_tokens = load_data_shard(Path(self.files[self.current_file_idx]))
        self.current_pos = 0
    
    def _get_batch_sync(self, batch_tokens: int, seq_len: int) -> Tuple[List[List[int]], List[int]]:
        """Get next batch synchronously."""
        contexts: List[List[int]] = []
        targets: List[int] = []
        
        while len(contexts) < batch_tokens:
            if self.current_tokens is None:
                self._load_current_file()
                continue
            
            if self.current_pos + seq_len + 1 >= len(self.current_tokens):
                self.current_file_idx = (self.current_file_idx + self.world_size) % len(self.files)
                self._load_current_file()
                continue
            
            start = self.current_pos
            end = start + seq_len + 1
            
            if self.current_tokens is not None and end <= len(self.current_tokens):
                seq = self.current_tokens[start:end]
                contexts.append(seq[:-1].tolist())
                targets.append(int(seq[-1]))
                self.current_pos = end
            else:
                self.current_file_idx = (self.current_file_idx + self.world_size) % len(self.files)
                self._load_current_file()
        
        return contexts, targets
    
    def start_prefetch(self, batch_tokens: int, seq_len: int):
        """Start background prefetch thread."""
        import threading
        
        self._stop_prefetch = False
        self._prefetch_lock = threading.Lock()
        self._prefetch_condition = threading.Condition(self._prefetch_lock)
        
        def prefetch_worker():
            while not self._stop_prefetch:
                with self._prefetch_condition:
                    while len(self._prefetch_queue) >= self.prefetch_batches:
                        if self._stop_prefetch:
                            return
                        self._prefetch_condition.wait(timeout=0.1)
                    
                    batch = self._get_batch_sync(batch_tokens, seq_len)
                    self._prefetch_queue.append(batch)
                    self._prefetch_condition.notify()
        
        self._prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self._prefetch_thread.start()
    
    def stop_prefetch(self):
        """Stop prefetch thread."""
        self._stop_prefetch = True
        if self._prefetch_condition:
            with self._prefetch_condition:
                self._prefetch_condition.notify_all()
        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)
    
    def next_batch(self, batch_tokens: int, seq_len: int) -> Tuple[List[List[int]], List[int]]:
        """Get next batch, using prefetch if available."""
        if self._prefetch_queue is not None and len(self._prefetch_queue) > 0:
            with self._prefetch_condition:
                batch = self._prefetch_queue.pop(0)
                self._prefetch_condition.notify()
            return batch
        
        return self._get_batch_sync(batch_tokens, seq_len)
    
    def next_batch_async(self, batch_tokens: int, seq_len: int) -> Tuple[List[List[int]], List[int]]:
        """Get next batch with async prefetching."""
        import threading
        
        if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
            self.start_prefetch(batch_tokens, seq_len)
        
        return self.next_batch(batch_tokens, seq_len)

def train_hdc(config: HDCConfig) -> Tuple[float, float]:
    """
    Train HDC model on FineWeb data.
    
    Returns (final_bpb, training_time).
    """
    print(f"Training HDC Model with Full Architecture Integration")
    print(f"Dimension: {config.hdc_dim:,} ({config.hdc_dim // 1024}K)")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Max context: {config.max_context_length}")
    
    model = HDCLanguageModel(config)
    
    sp = spm.SentencePieceProcessor()
    sp.load(config.tokenizer_path)
    
    base_bytes, has_leading_space, is_boundary_token = build_sentencepiece_luts(
        sp, config.vocab_size
    )
    
    print("Loading validation data...")
    val_tokens = load_validation_tokens(config.val_files, config.max_context_length)
    print(f"Validation sequences: {len(val_tokens):,}")
    
    if model.use_gpu:
        loader = AsyncTokenLoader(config.train_files, prefetch_batches=2)
        print("Using AsyncTokenLoader with prefetch for GPU training")
    else:
        loader = DistributedTokenLoader(config.train_files)
        print("Using DistributedTokenLoader for CPU training")
    
    start_time = time.time()
    iteration = 0
    best_bpb = float('inf')
    
    print(f"\nStarting training (max {config.iterations} iterations, {config.max_wallclock_seconds}s timeout)...")
    
    gpu_batch_size = config.gpu_batch_size if model.use_gpu else 1
    batch_tokens = config.train_batch_tokens // config.max_context_length
    
    if isinstance(loader, AsyncTokenLoader):
        loader.start_prefetch(batch_tokens, config.max_context_length)
    
    try:
        while iteration < config.iterations:
            elapsed = time.time() - start_time
            if elapsed >= config.max_wallclock_seconds:
                print(f"\nTime limit reached ({elapsed:.1f}s)")
                break
            
            if isinstance(loader, AsyncTokenLoader):
                contexts, targets = loader.next_batch_async(batch_tokens, config.max_context_length)
            else:
                contexts, targets = loader.next_batch(batch_tokens, config.max_context_length)
        
        if model.use_gpu and len(contexts) > 1:
            for i in range(0, len(contexts), gpu_batch_size):
                batch_contexts = contexts[i:i + gpu_batch_size]
                batch_targets = targets[i:i + gpu_batch_size]
                model.learn_patterns_batch(batch_contexts, batch_targets, use_peeling=False)
        else:
            for context, target in zip(contexts, targets):
                model.learn_pattern(context, target, use_peeling=True)
        
        iteration += 1
        
        if iteration % config.train_log_every == 0:
            elapsed = time.time() - start_time
            recipes_count = len(model.recipes)
            ngram_count = len(model.ngram_stats)
            storage_mb = model.recipe_storage_size / (1024 * 1024)
            mode = "GPU" if model.use_gpu else "CPU"
            print(f"Iter {iteration} [{mode}]: {elapsed:.1f}s, {recipes_count:,} recipes, "
                  f"{ngram_count:,} n-grams, {storage_mb:.2f}MB storage")
        
        if iteration % config.val_loss_every == 0:
            print(f"\nEvaluating at iteration {iteration}...")
            bpb, val_loss = evaluate_bpb(
                model, val_tokens, sp,
                base_bytes, has_leading_space, is_boundary_token,
                batch_size=32,
                max_batches=100  # Quick evaluation
            )
            print(f"BPB: {bpb:.4f}, Loss: {val_loss:.4f}")
            
            if bpb < best_bpb:
                best_bpb = bpb
    
    finally:
        if isinstance(loader, AsyncTokenLoader):
            loader.stop_prefetch()
    
    print("\nFinal evaluation...")
    final_bpb, final_val_loss = evaluate_bpb(
        model, val_tokens, sp,
        base_bytes, has_leading_space, is_boundary_token,
        batch_size=64
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete: {elapsed:.1f}s")
    print(f"Final BPB: {final_bpb:.4f}")
    print(f"Final Loss: {final_val_loss:.4f}")
    print(f"Best BPB: {best_bpb:.4f}")
    print(f"Recipes: {len(model.recipes):,}")
    print(f"N-grams: {len(model.ngram_stats):,}")
    print(f"Storage: {model.recipe_storage_size / (1024*1024):.2f}MB")
    
    return final_bpb, final_val_loss, elapsed

def parse_training_log(log_path: str) -> Dict[str, Any]:
    """Parse a training log file to extract key metrics."""
    import re
    
    result = {
        "val_loss": None,
        "val_bpb": None,
        "steps": None,
        "ms_per_step": None,
        "elapsed_seconds": None,
        "recipes_count": None,
        "ngram_count": None,
        "storage_mb": None
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    bpb_match = re.search(r'(?:Final BPB|val_bpb)[:\s]+(\d+\.\d+)', content)
    if bpb_match:
        result["val_bpb"] = float(bpb_match.group(1))
    
    loss_match = re.search(r'(?:Final Loss|val_loss)[:\s]+(\d+\.\d+)', content)
    if loss_match:
        result["val_loss"] = float(loss_match.group(1))
    
    steps_match = re.search(r'step[:\s]+(\d+)(?:/\d+)?', content)
    if steps_match:
        result["steps"] = int(steps_match.group(1))
    
    ms_match = re.search(r'step_avg[:\s]+(\d+\.\d+)ms', content)
    if ms_match:
        result["ms_per_step"] = float(ms_match.group(1))
    
    time_match = re.search(r'(?:Training complete|train_time)[:\s]+(\d+\.\d+)s', content)
    if time_match:
        result["elapsed_seconds"] = float(time_match.group(1))
    
    recipes_match = re.search(r'Recipes[:\s]+([\d,]+)', content)
    if recipes_match:
        result["recipes_count"] = int(recipes_match.group(1).replace(',', ''))
    
    ngram_match = re.search(r'N-grams[:\s]+([\d,]+)', content)
    if ngram_match:
        result["ngram_count"] = int(ngram_match.group(1).replace(',', ''))
    
    storage_match = re.search(r'Storage[:\s]+(\d+\.\d+)MB', content)
    if storage_match:
        result["storage_mb"] = float(storage_match.group(1))
    
    return result

def run_single_training(seed: int, args, log_dir: str = ".") -> Dict[str, Any]:
    """Run a single training session with the given seed."""
    import statistics
    from datetime import datetime, timezone
    
    log_file = os.path.join(log_dir, f"train_seed{seed}.log")
    
    print(f"\n{'='*60}")
    print(f"Starting training with seed {seed}")
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")
    
    config = HDCConfig(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        hdc_dim=args.hdc_dim,
        iterations=args.iterations,
        max_wallclock_seconds=args.max_time,
        seed=seed
    )
    
    start_time = time.time()
    
    original_stdout = sys.stdout
    log_handle = open(log_file, 'w')
    
    try:
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            def write(self, data):
                for f in self.files:
                    f.write(data)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        sys.stdout = TeeOutput(original_stdout, log_handle)
        
        final_bpb, final_val_loss, elapsed = train_hdc(config)
        
        print(f"\n{'='*60}")
        print(f"Final BPB: {final_bpb:.4f}")
        print(f"Final Loss: {final_val_loss:.4f}")
        print(f"train_time: {elapsed:.1f}s")
        print(f"{'='*60}")
        
    finally:
        sys.stdout = original_stdout
        log_handle.close()
    
    total_elapsed = time.time() - start_time
    
    results = parse_training_log(log_file)
    results["seed"] = seed
    results["log_file"] = log_file
    results["total_elapsed"] = total_elapsed
    results["val_bpb"] = results.get("val_bpb") or final_bpb
    results["val_loss"] = results.get("val_loss") or final_val_loss
    
    print(f"\nTraining with seed {seed} completed:")
    print(f"  BPB: {results.get('val_bpb', 'N/A')}")
    print(f"  Loss: {results.get('val_loss', 'N/A')}")
    print(f"  Steps: {results.get('steps', 'N/A')}")
    
    return results

def calculate_p_value(bpb_values: List[float], baseline: float = 1.2244) -> float:
    
    import statistics
    
    if len(bpb_values) < 2:
        return 1.0
    
    n = len(bpb_values)
    mean_bpb = statistics.mean(bpb_values)
    std_bpb = statistics.stdev(bpb_values)
    
    if std_bpb == 0:
        return 0.0 if mean_bpb < baseline else 1.0
    
    t_stat = (mean_bpb - baseline) / (std_bpb / (n ** 0.5))
    
    try:
        from scipy import stats
        p_value_one_sided = stats.t.cdf(t_stat, df=n-1)
    except ImportError:
        if t_stat < -3:
            p_value_one_sided = 0.01
        elif t_stat < -2:
            p_value_one_sided = 0.05
        elif t_stat < -1:
            p_value_one_sided = 0.15
        else:
            p_value_one_sided = 0.5
    
    return p_value_one_sided

def generate_multi_seed_submission(seed_results: Dict[int, Dict[str, Any]],
                                   args, code_bytes: int) -> Dict[str, Any]:
    """Generate the aggregated submission.json for multi-seed runs."""
    import statistics
    from datetime import datetime, timezone
    
    bpb_values = [r["val_bpb"] for r in seed_results.values() if r.get("val_bpb") is not None]
    loss_values = [r["val_loss"] for r in seed_results.values() if r.get("val_loss") is not None]
    
    if not bpb_values:
        raise ValueError("No valid BPB values found in training results")
    
    mean_bpb = statistics.mean(bpb_values)
    mean_loss = statistics.mean(loss_values) if loss_values else None
    std_bpb = statistics.stdev(bpb_values) if len(bpb_values) > 1 else 0.0
    
    p_value = calculate_p_value(bpb_values)
    
    artifact_bytes = code_bytes  # HDC is zero-weight, so just code size
    
    submission = {
        "track": "10min_16mb",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "name": args.run_name,
        "author": args.author,
        "seed_results": {
            str(seed): {
                "val_loss": r.get("val_loss"),
                "val_bpb": r.get("val_bpb"),
                "steps": r.get("steps"),
                "ms_per_step": r.get("ms_per_step")
            }
            for seed, r in seed_results.items()
        },
        "mean_val_loss": mean_loss,
        "mean_val_bpb": mean_bpb,
        "std_val_bpb": std_bpb,
        "p_value": round(p_value, 6),
        "artifact_bytes": artifact_bytes,
        "code_bytes": code_bytes,
        "baseline_bpb": 1.2244,
        "improvement": f"{((1.2244 - mean_bpb) / 1.2244 * 100):.2f}%"
    }
    
    return submission

def run_multi_seed_training(args):
    """Run multi-seed training and generate aggregated submission."""
    from datetime import datetime, timezone
    
    script_path = os.path.abspath(__file__)
    code_bytes = os.path.getsize(script_path)
    
    print(f"Multi-Seed Training Runner")
    print(f"{'='*60}")
    print(f"Seeds: {args.seeds}")
    print(f"Author: {args.author}")
    print(f"GitHub ID: {args.github_id}")
    print(f"Run name: {args.run_name}")
    print(f"Data path: {args.data_path}")
    print(f"Max time per run: {args.max_time}s")
    print(f"Code size: {code_bytes:,} bytes")
    print(f"{'='*60}")
    
    seed_results = {}
    
    for seed in args.seeds:
        result = run_single_training(
            seed=seed,
            args=args,
            log_dir=os.path.dirname(script_path) or "."
        )
        seed_results[seed] = result
    
    print(f"\n{'='*60}")
    print("Generating submission.json...")
    print(f"{'='*60}")
    
    submission = generate_multi_seed_submission(
        seed_results=seed_results,
        args=args,
        code_bytes=code_bytes
    )
    
    submission_path = os.path.join(os.path.dirname(script_path) or ".", "submission.json")
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\nSubmission saved to {submission_path}")
    print(f"\nFinal Results:")
    print(f"  Mean BPB: {submission['mean_val_bpb']:.6f}")
    print(f"  Std BPB: {submission['std_val_bpb']:.6f}")
    print(f"  P-value: {submission['p_value']:.6f}")
    print(f"  Improvement over baseline: {submission['improvement']}")
    print(f"  Artifact size: {submission['artifact_bytes']:,} bytes")
    
    if submission['p_value'] < 0.05:
        print(f"\n✅ Result is statistically significant (p < 0.05)")
    else:
        print(f"\n⚠️ Result is NOT statistically significant (p >= 0.05)")
    
    return 0 if submission['p_value'] < 0.05 else 1

def main():
    """Main entry point with multi-seed training support."""
    import argparse
    from datetime import datetime, timezone
    
    parser = argparse.ArgumentParser(description="HDC VSA Model for Parameter-Golf")
    parser.add_argument("--data_path", type=str, default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer_path", type=str, default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--hdc_dim", type=int, default=DEFAULT_HDC_DIM)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--max_time", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--author", type=str, default="YOUR_NAME_HERE", help="Author name for submission")
    parser.add_argument("--github_id", type=str, default="YOUR_GITHUB_ID_HERE", help="GitHub ID for submission")
    parser.add_argument("--run_name", type=str, default="HDC Zero Track 5Mb", help="Run name for submission")
    
    parser.add_argument("--multi_seed", action="store_true",
                        help="Run multi-seed training for statistically significant results")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 7, 1337],
                        help="Seeds for multi-seed training (default: 42 7 1337)")
    
    args = parser.parse_args()
    
    if args.multi_seed:
        return run_multi_seed_training(args)
    
    config = HDCConfig(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        hdc_dim=args.hdc_dim,
        iterations=args.iterations,
        max_wallclock_seconds=args.max_time,
        seed=args.seed
    )
    
    final_bpb, final_val_loss, elapsed = train_hdc(config)
    
    script_path = os.path.abspath(__file__)
    code_size_bytes = os.path.getsize(script_path)
    
    bytes_total = code_size_bytes
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"BPB: {final_bpb:.4f}")
    print(f"Val Loss: {final_val_loss:.4f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Code size: {code_size_bytes:,} bytes")
    print(f"Total artifact size: {bytes_total:,} bytes (zero-weight HDC)")
    print(f"Baseline to beat: 1.2244 BPB")
    
    submission = {
        "author": args.author,
        "github_id": args.github_id,
        "name": args.run_name,
        "blurb": f"HDC VSA Zero-Weight Model with {config.hdc_dim:,} dimensions, trained for {config.iterations} iterations in {elapsed:.1f}s",
        "date": datetime.now(timezone.utc).isoformat(),
        "val_loss": final_val_loss,
        "val_bpb": final_bpb,
        "bytes_total": bytes_total,
        "bytes_code": code_size_bytes
    }
    
    submission_path = "submission.json"
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\nSubmission saved to {submission_path}")
    print(f"Artifact size check: {'PASS' if bytes_total < 16000000 else 'FAIL'} (limit: 16,000,000 bytes)")

if __name__ == "__main__":
    sys.exit(main() or 0)
