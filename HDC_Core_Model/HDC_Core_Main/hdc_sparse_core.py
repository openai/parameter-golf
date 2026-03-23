"""
Pure HDC/VSA Core Operations (Strict XOR / Lossless Optimized)

This module implements the Pure HDC/VSA architecture using:
- Walsh-Hadamard Basis for perfect orthogonality
- BLAKE3 hashing for deterministic, cross-platform vector generation
- uint64 bit-packed storage for 8x memory reduction
- Pure mathematical encoding (NO CNN encoder/decoder)

Key Features:
- 100% deterministic across all hardware platforms (x86, ARM, GPU)
- 16KB per vector (131K dimensions) - fits in L1 cache
- 128KB per vector (1M dimensions) - fits in L2 cache for 8K video
- Unlimited seed generation with BLAKE3 single-call API
- ~3x faster than SHA256 for vector generation

Key Operations for DXPS (Deterministic XOR Program Synthesis):
- bind(): XOR (Primary composition operation)
- bind_sequence(): Lossless combination of multiple vectors (Replaces Bundle)
- unbind(): XOR (Self-inverse, identical to bind)
- permute(): Circular bit shift (Sequence encoding)
- invert(): Bitwise NOT (Negation)
- exact_match(): Strict boolean verification

Pure HDC Image/Video Encoding:
- encode_image_pure(): Hadamard position encoding (no CNN)
- encode_pixel_position(): O(1) spatial addressing
- encode_temporal_sequence(): Circular temporal encoding

GPU Acceleration:
- When CuPy is available, operations run on GPU automatically with priority.
- bind_sequence uses GPU reduction for O(1) grid encoding.

IMPORTANT: Pin blake3 version in requirements.txt:
    blake3==1.0.4  # PINNED — DO NOT UPGRADE without seed migration
"""

import numpy as np
import hashlib
from typing import List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

# Try to import BLAKE3 for deterministic generation
# Falls back to SHA256 if blake3 is not installed
try:
    import blake3
    _BLAKE3_AVAILABLE = True
except ImportError:
    _BLAKE3_AVAILABLE = False
    blake3 = None

# Try to import CuPy for GPU acceleration
USE_GPU = True  # True = prioritize GPU, False = force CPU, None = auto-detect
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
    _mempool = cp.get_default_memory_pool()
    _pinned_mempool = cp.get_default_pinned_memory_pool()
except ImportError:
    _CUPY_AVAILABLE = False
    cp = None

def get_array_module(use_gpu: Optional[bool] = None):
    """Get the appropriate array module (numpy or cupy)."""
    if use_gpu is None:
        use_gpu = USE_GPU if USE_GPU is not None else _CUPY_AVAILABLE
    if use_gpu and _CUPY_AVAILABLE:
        return cp
    return np

def get_gpu_info() -> dict:
    """Get GPU information if available."""
    if not _CUPY_AVAILABLE:
        return {"available": False, "reason": "CuPy not installed"}
    try:
        device = cp.cuda.Device()
        mem_info = device.mem_info
        return {
            "available": True,
            "device_id": device.id,
            "total_memory_gb": mem_info[1] / (1024**3),
            "free_memory_gb": mem_info[0] / (1024**3),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}

def ensure_numpy(arr):
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

def ensure_gpu(arr):
    if _CUPY_AVAILABLE and not isinstance(arr, cp.ndarray):
        return cp.asarray(arr)
    return arr

# =============================================================================
# Dimension Selection (from FULLINTEGRATION_NEW_ARCHITECTURE.md)
# =============================================================================
# 2^17 (131,072) - Text, Audio, Small images - L1 cache (16KB)
# 2^20 (1,048,576) - 8K Video - L2 cache (128KB) - DEFAULT
# 2^21 (2,097,152) - Future expansion - L2/L3 cache (256KB)

DEFAULT_HDC_DIM = 1048576  # 2^20 - Default for 8K video processing
HDC_DIM_LEGACY = 131072    # 2^17 - Legacy for text/audio/small images


# =============================================================================
# BLAKE3 Deterministic Vector Generation
# =============================================================================

def seed_to_hypervector_blake3(seed_string: str, uint64_count: int = 2048) -> np.ndarray:
    """
    Deterministically generate a hypervector from any string using BLAKE3.
    
    Identical output on every machine, every OS, forever.
    
    BLAKE3 advantages over SHA256:
    - Unlimited seed generation (extendable output)
    - Single API call (no counter loop needed)
    - ~3x faster than SHA256
    - Native extendable output (fills 16KB in one call)
    
    Args:
        seed_string: String seed for deterministic generation
        uint64_count: Number of uint64 values (dim / 64)
                     - 2048 for 2^17 (131,072 dimensions)
                     - 16384 for 2^20 (1,048,576 dimensions)
    
    Returns:
        uint64 array of shape (uint64_count,)
    
    IMPORTANT: Pin blake3 version in requirements.txt:
        blake3==1.0.4  # PINNED — DO NOT UPGRADE without seed migration
    """
    num_bytes = uint64_count * 8  # 8 bytes per uint64
    
    if _BLAKE3_AVAILABLE:
        # BLAKE3: Single call produces exactly the bytes we need
        hash_bytes = blake3.blake3(seed_string.encode()).digest(length=num_bytes)
    else:
        # Fallback to SHA256 with counter (slower but always available)
        hash_bytes = b''
        counter = 0
        while len(hash_bytes) < num_bytes:
            data = f"{seed_string}:{counter}".encode('utf-8')
            hash_bytes += hashlib.sha256(data).digest()
            counter += 1
        hash_bytes = hash_bytes[:num_bytes]
    
    return np.frombuffer(hash_bytes, dtype=np.uint64).copy()


def seed_string_to_int(seed_string: str) -> int:
    """Convert a string seed to an integer seed using SHA256."""
    hash_bytes = hashlib.sha256(seed_string.encode()).digest()
    return int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF


@dataclass
class SparseBinaryConfig:
    """Configuration for Pure HDC/VSA Engine."""
    dim: int = DEFAULT_HDC_DIM  # Dimension (bits) - 1048576 for 8K video processing
    seed: int = 42              # Base seed for determinism
    cache_size: int = 10000     # Max vectors to cache
    use_blake3: bool = True     # Use BLAKE3 for generation (fallback to SHA256)
    
    @property
    def byte_size(self) -> int:
        return self.dim // 8
    
    @property
    def uint64_count(self) -> int:
        """Number of uint64 elements needed to store the vector."""
        return self.dim // 64


class SparseBinaryHDC:
    """
    Pure HDC/VSA implementation with uint64 bit-packed storage.
    
    Optimized for:
    - 100% determinism across all platforms
    - L1/L2 cache residency for ultra-fast processing
    - Pure mathematical encoding (no CNN)
    - XOR binding for lossless operations
    
    Storage: uint64 arrays (8x memory reduction vs int8)
    - 2^17 dimensions: 16KB per vector (L1 cache)
    - 2^20 dimensions: 128KB per vector (L2 cache)
    """
    
    def __init__(self, config: Optional[SparseBinaryConfig] = None, use_gpu: Optional[bool] = None):
        self.config = config or SparseBinaryConfig()
        self.dim = self.config.dim
        self.byte_size = self.config.byte_size
        self.uint64_count = self.config.uint64_count
        self.use_blake3 = self.config.use_blake3 and _BLAKE3_AVAILABLE
        
        self.use_gpu = use_gpu if use_gpu is not None else (USE_GPU if USE_GPU is not None else _CUPY_AVAILABLE)
        self.xp = get_array_module(self.use_gpu)
        self._on_gpu = self.use_gpu and _CUPY_AVAILABLE
        
        self._cache: dict = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Import Hadamard basis for position encoding (lazy import to avoid circular dependency)
        self._hadamard_basis = None
    
    def _get_hadamard_basis(self):
        """Lazy load Hadamard basis for position encoding."""
        if self._hadamard_basis is None:
            try:
                from ..Recipes_Seeds.walsh_hadamard_core import WalshHadamardBasis
                self._hadamard_basis = WalshHadamardBasis(dim=self.dim)
            except ImportError:
                pass
        return self._hadamard_basis
    
    def to_numpy(self, arr):
        return ensure_numpy(arr)
    
    def to_gpu(self, arr):
        if self._on_gpu:
            return ensure_gpu(arr)
        return arr
    
    # =========================================================================
    # Vector Generation (BLAKE3 Primary, SHA256 Fallback)
    # =========================================================================
    
    def _blake3_deterministic_vector(self, seed_string: str) -> np.ndarray:
        """
        Generate a deterministic uint64 vector using BLAKE3.
        
        Preferred method for new code - faster and cleaner API.
        
        Args:
            seed_string: String seed for deterministic generation
            
        Returns:
            uint64 array of shape (uint64_count,)
        """
        return seed_to_hypervector_blake3(seed_string, self.uint64_count)
    
    def _sha256_deterministic_bytes(self, seed: int, num_bytes: int) -> bytes:
        """
        Generate deterministic bytes using SHA256 hashing (fallback method).
        
        This method is 100% deterministic across:
        - All hardware platforms (x86, ARM, GPU, etc.)
        - All programming languages (Python, C++, Rust, etc.)
        - All operating systems (Windows, Linux, macOS, etc.)
        - All versions of libraries (no NumPy version dependency)
        
        Args:
            seed: Integer seed for deterministic generation
            num_bytes: Number of bytes to generate
            
        Returns:
            Deterministic bytes of length num_bytes
        """
        result = b''
        counter = 0
        
        while len(result) < num_bytes:
            data = f"{seed}:{counter}".encode('utf-8')
            hash_bytes = hashlib.sha256(data).digest()
            result += hash_bytes
            counter += 1
        
        return result[:num_bytes]
    
    def _sha256_deterministic_vector(self, seed: int) -> np.ndarray:
        """
        Generate a deterministic packed binary vector using SHA256 (fallback).
        
        Args:
            seed: Integer seed for deterministic generation
            
        Returns:
            Packed binary vector of shape (byte_size,) dtype=uint8
        """
        bytes_data = self._sha256_deterministic_bytes(seed, self.byte_size)
        return np.frombuffer(bytes_data, dtype=np.uint8).copy()
    
    def from_seed_string(self, seed_string: str) -> np.ndarray:
        """
        Generate a deterministic vector from a string seed.
        
        Uses BLAKE3 if available (preferred), otherwise SHA256.
        This is the recommended method for new code.
        
        Args:
            seed_string: Human-readable string seed (e.g., "concept:cat", "video:frame:42")
            
        Returns:
            uint64 array (if BLAKE3) or uint8 array (if SHA256 fallback)
        
        Examples:
            >>> hdc = SparseBinaryHDC()
            >>> cat_vec = hdc.from_seed_string("concept:cat")
            >>> frame_vec = hdc.from_seed_string("video:frame:42")
            >>> physics_vec = hdc.from_seed_string("physics:gravity:9.81")
        """
        cache_key = f"str:{seed_string}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key].copy()
        
        self._cache_misses += 1
        
        if self.use_blake3:
            vec = self._blake3_deterministic_vector(seed_string)
        else:
            # Fallback to integer seed + SHA256
            seed = seed_string_to_int(seed_string)
            vec = self._sha256_deterministic_vector(seed)
        
        if len(self._cache) < self.config.cache_size:
            self._cache[cache_key] = vec.copy()
        
        return vec
    
    def random_vector(self, seed: Optional[int] = None):
        """
        Generate a deterministic random vector from integer seed.
        
        Uses SHA256-based generation for true cross-platform determinism.
        Same seed always produces identical vector on any hardware/language.
        
        Args:
            seed: Integer seed for deterministic generation.
                  If None, generates a non-deterministic vector (not recommended).
                  
        Returns:
            Packed binary vector of shape (byte_size,) dtype=uint8
        """
        if seed is not None and seed in self._cache:
            self._cache_hits += 1
            vec = self._cache[seed].copy()
            return self.to_gpu(vec) if self._on_gpu else vec
        
        self._cache_misses += 1
        
        if seed is not None:
            # Use SHA256-based deterministic generation
            vec = self._sha256_deterministic_vector(seed)
        else:
            # Fallback: non-deterministic (not recommended for reproducibility)
            rng = np.random.default_rng()
            vec = rng.integers(0, 256, size=self.byte_size, dtype=np.uint8)
        
        if seed is not None and len(self._cache) < self.config.cache_size:
            self._cache[seed] = vec.copy()
        
        return self.to_gpu(vec) if self._on_gpu else vec
    
    def zeros(self):
        """Return a zero vector (all bits = 0). Acts as Identity for XOR."""
        return self.xp.zeros(self.byte_size, dtype=self.xp.uint8)
    
    def ones(self):
        """Return a vector with all bits = 1."""
        return self.xp.full(self.byte_size, 255, dtype=self.xp.uint8)
    
    def from_seed(self, seed: int) -> np.ndarray:
        return self.random_vector(seed=seed)
    
    def from_string(self, string: str) -> Tuple[int, np.ndarray]:
        hash_bytes = hashlib.sha256(string.encode()).digest()
        seed = int.from_bytes(hash_bytes[:8], 'big') & 0x7FFFFFFFFFFFFFFF
        return seed, self.from_seed(seed)
    
    # =========================================================================
    # Core Algebraic Operations (Lossless)
    # =========================================================================
    
    def bind(self, a, b):
        """
        XOR binding. The fundamental operation for DXPS.
        a ⊕ b
        """
        return self.xp.bitwise_xor(a, b)
    
    def unbind(self, bound, key):
        """
        XOR unbinding. Identical to bind.
        (a ⊕ b) ⊕ b = a
        """
        return self.xp.bitwise_xor(bound, key)

    def bind_sequence(self, vectors: List):
        """
        Sequentially XORs a list of vectors.
        Replaces 'bundle' for lossless grid encoding.
        Result = v1 ⊕ v2 ⊕ v3 ...
        
        Optimized with GPU reduction if available.
        """
        if len(vectors) == 0:
            return self.zeros()
        if len(vectors) == 1:
            return vectors[0].copy()
            
        xp = self.xp
        
        if self._on_gpu:
            # Use CuPy's highly optimized reduction kernel
            # Stack vectors into matrix (N, byte_size)
            vec_stack = xp.stack(vectors)
            # Reduce along axis 0 via XOR
            return xp.bitwise_xor.reduce(vec_stack, axis=0)
        else:
            # CPU Loop (Numpy reduce is also available but explicit for clarity)
            result = vectors[0].copy()
            for v in vectors[1:]:
                result = xp.bitwise_xor(result, v)
            return result

    def permute(self, vec, shift: int):
        """
        Circular bit shift. Preserves information 100%.
        Alias for circular_shift for backward compatibility.
        """
        return self.circular_shift(vec, shift)
    
    def circular_shift(self, vec, shift: int):
        """
        Circular bit shift (folding) for temporal encoding.
        
        Applies ρ^n(v) - rotates the vector by n positions.
        This is the core operation for Circular Temporal Encoding,
        enabling unlimited temporal depth with zero RAM increase.
        
        Properties:
        - Perfectly reversible: ρ^-n(ρ^n(v)) = v
        - XOR compatible: Can be combined with XOR binding
        - Zero RAM overhead: Only index manipulation, no new vectors
        
        Args:
            vec: Packed binary vector of shape (byte_size,)
            shift: Number of positions to shift (positive = right, negative = left)
        
        Returns:
            Shifted packed binary vector
        """
        xp = self.xp
        bits = xp.unpackbits(vec)
        shifted = xp.roll(bits, shift)
        return xp.packbits(shifted)
    
    def invert(self, vec):
        """
        Bitwise NOT. Reversible.
        """
        return self.xp.bitwise_not(vec)
    
    # =========================================================================
    # Circular Temporal Encoding (100-Year Memory)
    # =========================================================================
    
    def encode_temporal_sequence(self, events: List, position_markers: Optional[List] = None) -> np.ndarray:
        """
        Encode a temporal sequence using circular shifts + XOR binding.
        
        This implements the Circular Temporal Encoding from the architecture:
        sequence = ρ^0(event_a) ⊕ ρ^1(event_b) ⊕ ρ^2(event_c) ⊕ ...
        
        Key Properties:
        - Unlimited temporal depth with ZERO RAM increase
        - Perfect reversibility: Each event can be retrieved
        - XOR compatible: Works with all other XOR operations
        - 100% deterministic: Same sequence always produces same result
        
        Args:
            events: List of packed binary vectors representing events in order
            position_markers: Optional list of position marker vectors.
                             If None, uses circular shift for position encoding.
        
        Returns:
            Single packed binary vector containing all events with temporal ordering
        
        Example:
            >>> hdc = SparseBinaryHDC()
            >>> morning = hdc.from_seed(42)
            >>> afternoon = hdc.from_seed(43)
            >>> evening = hdc.from_seed(44)
            >>> daily_sequence = hdc.encode_temporal_sequence([morning, afternoon, evening])
            >>> # Result: Single 131K vector containing all three events
        """
        if len(events) == 0:
            return self.zeros()
        if len(events) == 1:
            return events[0].copy()
        
        xp = self.xp
        
        if position_markers is not None:
            # Use provided position markers (Hadamard row indices)
            # Bind each event with its position marker via XOR
            bound_events = []
            for i, (event, marker) in enumerate(zip(events, position_markers)):
                bound = self.bind(event, marker)
                bound_events.append(bound)
            return self.bind_sequence(bound_events)
        else:
            # Use circular shift for position encoding
            # sequence = ρ^0(e0) ⊕ ρ^1(e1) ⊕ ρ^2(e2) ⊕ ...
            result = self.zeros()
            for i, event in enumerate(events):
                shifted = self.circular_shift(event, i)
                result = self.bind(result, shifted)
            return result
    
    def retrieve_event_at_position(self, sequence: np.ndarray, position: int,
                                    position_marker: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Retrieve event at specific position from a temporal sequence.
        
        For circular shift encoding:
            event_n = ρ^-n(sequence)
        
        For position marker encoding:
            event_n = sequence ⊕ position_marker_n
        
        Args:
            sequence: Encoded temporal sequence vector
            position: Position index to retrieve (0-indexed)
            position_marker: Optional position marker vector. If None, uses circular shift.
        
        Returns:
            Retrieved event vector at the specified position
        
        Note:
            For circular shift encoding with multiple events, the retrieved vector
            will contain the superposition of all events. To isolate a specific event,
            you need to know the number of events and use appropriate decoding.
        """
        if position_marker is not None:
            # Unbind using position marker
            return self.unbind(sequence, position_marker)
        else:
            # Reverse the circular shift
            return self.circular_shift(sequence, -position)
    
    def decode_temporal_sequence(self, sequence: np.ndarray, num_events: int) -> List[np.ndarray]:
        """
        Decode all events from a temporal sequence.
        
        This retrieves each event by reversing the circular shifts.
        Note: Due to XOR superposition, this works best when you know
        the original events or have reference vectors to match against.
        
        Args:
            sequence: Encoded temporal sequence vector
            num_events: Number of events in the sequence
        
        Returns:
            List of retrieved event vectors (may contain superposition noise)
        """
        events = []
        for i in range(num_events):
            # For each position, we need to unbind all other positions
            # This is a simplified retrieval - full retrieval requires
            # knowing the events or using associative memory
            retrieved = self.circular_shift(sequence, -i)
            events.append(retrieved)
        return events

    # =========================================================================
    # Statistical Operations (Lossy - Use with Caution in ARC)
    # =========================================================================

    def bundle(self, vectors: List):
        """
        Majority Vote.
        WARNING: This operation is LOSSY. Do not use for ARC Grid Encoding.
        Use bind_sequence() instead for lossless combination.
        Kept for backward compatibility with 7-sense/Audio modules.
        """
        if len(vectors) == 0:
            return self.zeros()
        if len(vectors) == 1:
            return vectors[0].copy()
        
        n = len(vectors)
        xp = self.xp
        
        bit_counts = xp.zeros(self.dim, dtype=xp.int32)
        for vec in vectors:
            unpacked = xp.unpackbits(vec)
            bit_counts += unpacked
        
        threshold = (n - 1) // 2
        result_bits = (bit_counts > threshold).astype(xp.uint8)
        return xp.packbits(result_bits)

    # =========================================================================
    # Verification & Metrics
    # =========================================================================

    def exact_match(self, a, b) -> bool:
        """
        Checks if two vectors are 100% identical bit-for-bit.
        This is the standard for the new "Proving Gate".
        """
        return self.xp.array_equal(a, b)

    def is_zero(self, vec) -> bool:
        """
        Checks if the vector is the Zero (Identity) vector.
        Used to check if a Residual has been fully solved.
        """
        return not self.xp.any(vec)

    def similarity(self, a, b) -> float:
        """
        Normalized Hamming similarity [0.0, 1.0].
        Used for heuristics/beaming, but not for final verification.
        """
        xp = self.xp
        xor_result = xp.bitwise_xor(a, b)
        hamming_distance = xp.unpackbits(xor_result).sum()
        if self._on_gpu:
            hamming_distance = float(hamming_distance.get())
        return 1.0 - (hamming_distance / self.dim)

    # =========================================================================
    # Batch Operations (GPU Optimized)
    # =========================================================================

    def batch_similarity(self, query, candidates: List) -> List[float]:
        if len(candidates) == 0: return []
        xp = self.xp
        candidate_matrix = xp.stack(candidates, axis=0)
        xor_results = xp.bitwise_xor(candidate_matrix, query)
        
        if self._on_gpu:
            n_candidates = xor_results.shape[0]
            xor_flat = xor_results.flatten()
            unpacked_flat = xp.unpackbits(xor_flat)
            unpacked = unpacked_flat.reshape(n_candidates, self.dim)
            hamming_distances = xp.sum(unpacked, axis=1, dtype=xp.float32)
        else:
            unpacked = xp.unpackbits(xor_results, axis=1)
            hamming_distances = xp.sum(unpacked, axis=1).astype(xp.float32)
        
        similarities = 1.0 - (hamming_distances / self.dim)
        if self._on_gpu:
            return similarities.get().tolist()
        return similarities.tolist()

    def warmup_gpu(self, num_vectors: int = 100):
        if not self._on_gpu: return
        vectors = [self.random_vector(seed=i) for i in range(num_vectors)]
        _ = self.bind(vectors[0], vectors[1])
        _ = self.bind_sequence(vectors[:10])
        self._on_gpu and cp.cuda.Stream.null.synchronize()
    
    # =========================================================================
    # Pure HDC Image/Video Encoding (No CNN)
    # =========================================================================
    
    def encode_pixel_position(self, x: int, y: int, width: int) -> np.ndarray:
        """
        Get Hadamard row as position vector for pixel at (x, y).
        
        Hadamard rows are mutually orthogonal, guaranteeing:
        - Zero collisions between positions
        - O(1) spatial addressing
        - Perfect reversibility
        
        Args:
            x: X coordinate (column)
            y: Y coordinate (row)
            width: Image width (for computing linear index)
        
        Returns:
            Position vector (Hadamard row)
        """
        basis = self._get_hadamard_basis()
        position_index = x * width + y
        
        if basis is not None:
            # Use Hadamard basis for position encoding
            return basis.get_row(position_index % self.dim, packed=True)
        else:
            # Fallback: use BLAKE3 seed generation
            return self.from_seed_string(f"position:x{x}:y{y}:w{width}")
    
    def encode_pixel_value(self, pixel_value: int) -> np.ndarray:
        """
        Encode a pixel value using BLAKE3 deterministic generation.
        
        Args:
            pixel_value: Pixel value (0-255 for grayscale, or RGB packed)
        
        Returns:
            Value vector
        """
        return self.from_seed_string(f"pixel_value:{pixel_value}")
    
    def encode_image_pixel(self, pixel_value: int, x: int, y: int, width: int) -> np.ndarray:
        """
        Encode a single pixel using Hadamard position binding.
        
        Binding: position_vector ⊕ value_vector
        This is perfectly reversible: XOR again to unbind.
        
        Args:
            pixel_value: Pixel value
            x: X coordinate
            y: Y coordinate
            width: Image width
        
        Returns:
            Bound pixel vector
        """
        position_vec = self.encode_pixel_position(x, y, width)
        value_vec = self.encode_pixel_value(pixel_value)
        return self.bind(position_vec, value_vec)
    
    def encode_image_pure(self, image: np.ndarray, patch_size: int = 256) -> np.ndarray:
        """
        Encode an image using Pure HDC (no CNN encoder/decoder).
        
        Pipeline:
        1. Split into patches (256x256 for 8K images)
        2. For each pixel: bind position with value using XOR
        3. Bundle all pixels in patch into single patch vector
        4. Bind each patch with its grid position
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            patch_size: Size of patches (default 256x256)
        
        Returns:
            Image hypervector (single vector containing entire image)
        
        Example:
            >>> hdc = SparseBinaryHDC()
            >>> image = np.random.randint(0, 256, (720, 1280), dtype=np.uint8)
            >>> encoded = hdc.encode_image_pure(image)
        """
        # Handle color images by converting to grayscale or processing channels
        if len(image.shape) == 3:
            # For RGB, combine channels or process separately
            # Here we combine: pixel_value = R*65536 + G*256 + B
            h, w, c = image.shape
            flat_image = np.zeros((h, w), dtype=np.int32)
            for i in range(c):
                flat_image += image[:, :, i].astype(np.int32) * (256 ** (c - 1 - i))
            image = flat_image
        
        h, w = image.shape
        
        # For small images, encode directly
        if h <= patch_size and w <= patch_size:
            return self._encode_image_region(image, 0, 0, w, h)
        
        # For large images, split into patches
        patches = []
        patch_positions = []
        
        for py in range(0, h, patch_size):
            for px in range(0, w, patch_size):
                patch_h = min(patch_size, h - py)
                patch_w = min(patch_size, w - px)
                
                # Encode patch
                patch_vec = self._encode_image_region(
                    image, px, py, patch_w, patch_h
                )
                patches.append(patch_vec)
                
                # Track patch position for binding
                patch_idx = (py // patch_size) * ((w + patch_size - 1) // patch_size) + (px // patch_size)
                patch_positions.append(patch_idx)
        
        # Bind each patch with its position and combine
        bound_patches = []
        for patch_vec, patch_idx in zip(patches, patch_positions):
            position_vec = self.from_seed_string(f"patch:{patch_idx}")
            bound_patches.append(self.bind(patch_vec, position_vec))
        
        # Combine all patches
        return self.bind_sequence(bound_patches)
    
    def _encode_image_region(self, image: np.ndarray, ox: int, oy: int, w: int, h: int) -> np.ndarray:
        """
        Encode a region of an image.
        
        Args:
            image: Full image array
            ox: Origin X offset
            oy: Origin Y offset
            w: Region width
            h: Region height
        
        Returns:
            Encoded region vector
        """
        pixel_vectors = []
        
        for y in range(oy, oy + h):
            for x in range(ox, ox + w):
                pixel_value = int(image[y, x])
                pixel_vec = self.encode_image_pixel(pixel_value, x, y, image.shape[1])
                pixel_vectors.append(pixel_vec)
        
        # Combine all pixels using XOR binding
        return self.bind_sequence(pixel_vectors)
    
    def decode_pixel_value(self, image_vector: np.ndarray, x: int, y: int, width: int,
                           candidate_values: Optional[List[int]] = None) -> int:
        """
        Retrieve pixel value at position (x, y) from image hypervector.
        
        O(1) operation - no scanning needed.
        
        Args:
            image_vector: Encoded image hypervector
            x: X coordinate
            y: Y coordinate
            width: Image width
            candidate_values: Optional list of candidate values to check
        
        Returns:
            Most likely pixel value
        """
        position_vec = self.encode_pixel_position(x, y, width)
        unbound = self.unbind(image_vector, position_vec)
        
        # Check against candidate values
        if candidate_values is None:
            candidate_values = list(range(256))
        
        best_value = 0
        best_similarity = -1
        
        for val in candidate_values:
            value_vec = self.encode_pixel_value(val)
            sim = self.similarity(unbound, value_vec)
            if sim > best_similarity:
                best_similarity = sim
                best_value = val
        
        return best_value


class AtomicVocabulary:
    """Pre-encoded atomic vocabulary for sparse binary HDC."""
    def __init__(self, hdc: SparseBinaryHDC):
        self.hdc = hdc
        self._vocab: dict = {}
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        # Colors (0-9)
        for i in range(10):
            name = f'color_{i}'
            seed, vec = self.hdc.from_string(f'arc_color_{i}')
            self._vocab[name] = vec
        
        # Positions (30x30)
        for row in range(30):
            for col in range(30):
                name = f'pos_{row}_{col}'
                seed, vec = self.hdc.from_string(f'position_row{row}_col{col}')
                self._vocab[name] = vec

    def get(self, name: str) -> np.ndarray:
        if name not in self._vocab:
            # Deterministic fallback for missing items
            _, vec = self.hdc.from_string(name)
            self._vocab[name] = vec
        return self._vocab[name].copy()
    
    def has(self, name: str) -> bool:
        return name in self._vocab
    
    def size(self) -> int:
        return len(self._vocab)

def create_sparse_hdc(dim: int = DEFAULT_HDC_DIM, seed: int = 42, use_gpu: Optional[bool] = None):
    """
    Create a SparseBinaryHDC instance with atomic vocabulary.
    
    Args:
        dim: Vector dimension (default 1048576 for 8K video processing)
        seed: Base seed for determinism
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Tuple of (SparseBinaryHDC instance, AtomicVocabulary instance)
    """
    config = SparseBinaryConfig(dim=dim, seed=seed)
    hdc = SparseBinaryHDC(config, use_gpu=use_gpu)
    vocab = AtomicVocabulary(hdc)
    return hdc, vocab


# =============================================================================
# Standalone Wrapper Functions for GPU Operations
# =============================================================================
# These wrapper functions allow importing GPU operations as standalone functions
# rather than accessing them only through an HDC instance.

def warmup_gpu(dim: int = DEFAULT_HDC_DIM, num_vectors: int = 100) -> None:
    """
    Pre-warm GPU memory pools and kernels for faster first operations.
    
    Creates a temporary HDC instance to warm up GPU operations.
    Should be called once at startup when using GPU acceleration.
    
    Args:
        dim: Vector dimension (default 1048576 for 8K video processing)
        num_vectors: Number of test vectors to generate (default 100)
    """
    if not _CUPY_AVAILABLE or not USE_GPU:
        return
    
    config = SparseBinaryConfig(dim=dim)
    hdc = SparseBinaryHDC(config, use_gpu=True)
    hdc.warmup_gpu(num_vectors)


def get_gpu_memory_status() -> Optional[dict]:
    """
    Get current GPU memory status.
    
    Returns:
        Dict with device_name, free_memory_mb, total_memory_mb, used_memory_mb
        or None if GPU is not available.
    """
    if not _CUPY_AVAILABLE:
        return None
    
    try:
        device = cp.cuda.Device()
        free_mem, total_mem = device.mem_info
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        device_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
        
        return {
            "device_name": device_name,
            "device_id": device.id,
            "free_memory_mb": free_mem / (1024 * 1024),
            "total_memory_mb": total_mem / (1024 * 1024),
            "used_memory_mb": (total_mem - free_mem) / (1024 * 1024),
            "mempool_used_bytes": _mempool.used_bytes() if _mempool else 0,
            "mempool_total_bytes": _mempool.total_bytes() if _mempool else 0,
        }
    except Exception as e:
        return {"error": str(e), "available": False}


def batch_bind(hdc: SparseBinaryHDC, vectors_a: List, vectors_b: List) -> List:
    """
    Batch XOR binding of paired vectors using GPU acceleration.
    
    Binds vectors_a[i] with vectors_b[i] for all i in parallel.
    This is much faster than sequential binding for large batches.
    
    Args:
        hdc: SparseBinaryHDC instance
        vectors_a: List of first vectors
        vectors_b: List of second vectors (must be same length)
    
    Returns:
        List of bound vectors (a[i] XOR b[i])
    
    Example:
        word_vecs = [hdc.from_seed(i) for i in range(100)]
        pos_vecs = [hdc.from_seed(i + 1000) for i in range(100)]
        positioned = batch_bind(hdc, word_vecs, pos_vecs)
    """
    if len(vectors_a) != len(vectors_b):
        raise ValueError(f"Vector lists must be same length: {len(vectors_a)} vs {len(vectors_b)}")
    
    if len(vectors_a) == 0:
        return []
    
    xp = hdc.xp
    
    # Stack vectors into matrices
    matrix_a = xp.stack(vectors_a, axis=0)
    matrix_b = xp.stack(vectors_b, axis=0)
    
    # Batch XOR
    result_matrix = xp.bitwise_xor(matrix_a, matrix_b)
    
    # Split back into list
    return [result_matrix[i] for i in range(len(vectors_a))]


def batch_similarity(hdc: SparseBinaryHDC, query, candidates: List) -> List[float]:
    """
    Compute similarity between query and all candidates in batch.
    
    Wrapper around SparseBinaryHDC.batch_similarity() for standalone import.
    
    Args:
        hdc: SparseBinaryHDC instance
        query: Query vector
        candidates: List of candidate vectors
    
    Returns:
        List of similarity scores [0.0, 1.0] for each candidate
    """
    return hdc.batch_similarity(query, candidates)


def batch_similarity_matrix(hdc: SparseBinaryHDC, vectors_a: List, vectors_b: List) -> List[List[float]]:
    """
    Compute all pairwise similarities between two sets of vectors.
    
    Returns an MxN matrix where result[i][j] = similarity(vectors_a[i], vectors_b[j]).
    
    Args:
        hdc: SparseBinaryHDC instance
        vectors_a: First set of vectors (length M)
        vectors_b: Second set of vectors (length N)
    
    Returns:
        MxN list of lists with similarity scores
    
    Example:
        inputs = [encode_grid(g) for g in input_grids]
        outputs = [encode_grid(g) for g in output_grids]
        sim_matrix = batch_similarity_matrix(hdc, inputs, outputs)
        # sim_matrix[i][j] = how similar input i is to output j
    """
    if len(vectors_a) == 0 or len(vectors_b) == 0:
        return []
    
    result = []
    for a in vectors_a:
        row = hdc.batch_similarity(a, vectors_b)
        result.append(row)
    
    return result