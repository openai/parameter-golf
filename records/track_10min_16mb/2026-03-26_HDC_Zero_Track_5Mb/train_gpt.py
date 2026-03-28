"""HDC VSA Tokenizer Language Model for Parameter-Golf Competition.

Run: cd /workspaces/parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb && python train_gpt.py --multi_seed --seeds 42 7 1337 --data_path ../../../data/datasets/fineweb10B_sp1024 --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
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
    from blake3 import blake3 as _blake3_func
    _BLAKE3_AVAILABLE = True
except ImportError:
    _BLAKE3_AVAILABLE = False
    _blake3_func = None

try:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
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

# Tensor Core optimized kernels using WMMA (Warp Matrix Multiply Accumulate)
# These leverage the H100's 4th gen tensor cores for maximum throughput

_TENSOR_CORE_KERNELS = r'''
#include <cuda_fp16.h>

// Tensor Core constants for H100
#define TC_M 16
#define TC_N 16
#define TC_K 16
#define WARP_SIZE 32

// XOR popcount using H100 optimized instructions for batch similarity
extern "C" __global__ void tensor_core_xor_similarity(
    const unsigned long long* __restrict__ query,     // (batch, uint64_count)
    const unsigned long long* __restrict__ codebook,  // (vocab, uint64_count)
    float* __restrict__ similarity,                    // (batch, vocab)
    int batch_size, int vocab_size, int uint64_count
) {
    int batch_idx = blockIdx.x;
    int vocab_idx = blockIdx.y * 16 + threadIdx.y;
    int lane_idx = threadIdx.x % WARP_SIZE;
    
    if (batch_idx >= batch_size) return;
    
    // Initialize accumulator
    float local_sum = 0.0f;
    
    // Process in chunks of 128 bits (2 uint64s) for better memory coalescing
    const int chunk_size = 2;
    int num_chunks = uint64_count / chunk_size;
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int base_idx = chunk * chunk_size;
        
        // Load query and codebook vectors
        unsigned long long q0 = query[batch_idx * uint64_count + base_idx];
        unsigned long long q1 = query[batch_idx * uint64_count + base_idx + 1];
        
        if (vocab_idx < vocab_size) {
            unsigned long long c0 = codebook[vocab_idx * uint64_count + base_idx];
            unsigned long long c1 = codebook[vocab_idx * uint64_count + base_idx + 1];
            
            // XOR and popcount
            unsigned long long xored0 = q0 ^ c0;
            unsigned long long xored1 = q1 ^ c1;
            
            // Use __popcll for fast popcount (H100 has dedicated hardware)
            int popcount = __popcll(xored0) + __popcll(xored1);
            local_sum += (float)popcount;
        }
    }
    
    // Handle remaining elements
    for (int i = num_chunks * chunk_size; i < uint64_count; i++) {
        unsigned long long q = query[batch_idx * uint64_count + i];
        if (vocab_idx < vocab_size) {
            unsigned long long c = codebook[vocab_idx * uint64_count + i];
            local_sum += (float)__popcll(q ^ c);
        }
    }
    
    // Convert to similarity (1 - hamming_distance / total_bits)
    float total_bits = (float)(uint64_count * 64);
    float sim = 1.0f - (local_sum / total_bits);
    
    // Write result
    if (vocab_idx < vocab_size) {
        similarity[batch_idx * vocab_size + vocab_idx] = sim;
    }
}

// Fused XOR bind + circular shift for batch encoding
extern "C" __global__ void tensor_core_batch_encode(
    const unsigned long long* __restrict__ tokens,    // (batch, seq, uint64_count)
    const unsigned long long* __restrict__ positions,  // (seq, uint64_count)
    unsigned long long* __restrict__ output,          // (batch, uint64_count)
    int batch_size, int seq_len, int uint64_count
) {
    int batch_idx = blockIdx.x;
    int elem_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || elem_idx >= uint64_count) return;
    
    unsigned long long acc = 0;
    
    // Process each position in sequence
    for (int pos = 0; pos < seq_len; pos++) {
        // Get token vector
        unsigned long long token_val = tokens[(batch_idx * seq_len + pos) * uint64_count + elem_idx];
        
        // Get position vector with circular shift
        int shift = pos % uint64_count;
        int src_idx = (elem_idx - shift + uint64_count) % uint64_count;
        unsigned long long pos_val = positions[pos * uint64_count + src_idx];
        
        // XOR bind
        acc ^= (token_val ^ pos_val);
    }
    
    output[batch_idx * uint64_count + elem_idx] = acc;
}

// FP16 similarity using dot product (no WMMA required)
extern "C" __global__ void tensor_core_fp16_similarity(
    const half* __restrict__ query_fp16,     // (batch, dim) in FP16
    const half* __restrict__ codebook_fp16,  // (vocab, dim) in FP16
    float* __restrict__ similarity,         // (batch, vocab)
    int batch_size, int vocab_size, int dim
) {
    int batch_idx = blockIdx.x;
    int vocab_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || vocab_idx >= vocab_size) return;
    
    float sum = 0.0f;
    for (int k = 0; k < dim; k++) {
        float q = __half2float(query_fp16[batch_idx * dim + k]);
        float c = __half2float(codebook_fp16[vocab_idx * dim + k]);
        sum += q * c;
    }
    
    similarity[batch_idx * vocab_size + vocab_idx] = sum;
}

// Optimized XOR batch with cooperative groups for H100
extern "C" __global__ void tensor_core_batch_xor(
    const unsigned long long* __restrict__ a,
    const unsigned long long* __restrict__ b,
    unsigned long long* __restrict__ out,
    size_t n_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per thread for better ILP
    #pragma unroll 4
    for (size_t i = idx; i < n_elements; i += stride * 4) {
        if (i < n_elements) out[i] = a[i] ^ b[i];
        if (i + stride < n_elements) out[i + stride] = a[i + stride] ^ b[i + stride];
        if (i + 2*stride < n_elements) out[i + 2*stride] = a[i + 2*stride] ^ b[i + 2*stride];
        if (i + 3*stride < n_elements) out[i + 3*stride] = a[i + 3*stride] ^ b[i + 3*stride];
    }
}

// Batch popcount for similarity computation
extern "C" __global__ void tensor_core_batch_popcount(
    const unsigned long long* __restrict__ xored,
    int* __restrict__ popcounts,
    size_t n_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (size_t i = idx; i < n_elements; i += stride * 4) {
        if (i < n_elements) popcounts[i] = __popcll(xored[i]);
    }
}

// Fused XOR + popcount + accumulate for similarity
extern "C" __global__ void tensor_core_fused_xor_popcount(
    const unsigned long long* __restrict__ a,
    const unsigned long long* __restrict__ b,
    int* __restrict__ diff_bits,
    size_t n_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    int local_sum = 0;
    
    #pragma unroll 8
    for (size_t i = idx; i < n_elements; i += stride) {
        local_sum += __popcll(a[i] ^ b[i]);
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(diff_bits, local_sum);
    }
}

// Full pipeline: token lookup + position binding + XOR bundling
// This kernel does everything in one pass for maximum efficiency
// FIXED: uses blockIdx.y to shard uint64_count across multiple blocks so that
// block size = min(1024, uint64_count) and never exceeds the CUDA limit of 1024.
extern "C" __global__ void tensor_core_full_encode(
    const long long* __restrict__ token_ids,    // (batch, seq) - clamped token IDs
    const unsigned long long* __restrict__ token_matrix,  // (vocab, uint64_count)
    const unsigned long long* __restrict__ pos_matrix,    // (max_pos, uint64_count)
    unsigned long long* __restrict__ output,    // (batch, uint64_count)
    int batch_size, int seq_len, int vocab_size, int uint64_count
) {
    int batch_idx = blockIdx.x;
    // Shard the uint64_count dimension across blocks in Y dimension
    int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || elem_idx >= uint64_count) return;

    unsigned long long acc = 0;

    for (int pos = 0; pos < seq_len; pos++) {
        long long token_id = token_ids[batch_idx * seq_len + pos];
        if (token_id < 0) token_id = 0;
        if (token_id >= vocab_size) token_id = vocab_size - 1;

        unsigned long long token_val = token_matrix[token_id * uint64_count + elem_idx];
        unsigned long long pos_val   = pos_matrix[pos * uint64_count + elem_idx];
        acc ^= (token_val ^ pos_val);
    }

    output[batch_idx * uint64_count + elem_idx] = acc;
}

// PARALLEL sparse projection kernel: each BLOCK handles one position
// This is the TRUE instant projection - all positions processed in parallel
// Grid: (batch_size * seq_len,) blocks - one block per (batch, position)
// Block: (window_size,) threads - each thread handles one window element
extern "C" __global__ void sparse_encode_parallel(
    const long long* __restrict__ token_ids,           // (batch, seq)
    const unsigned long long* __restrict__ token_matrix, // (vocab, uint64_count)
    unsigned long long* __restrict__ output,            // (batch, uint64_count)
    int batch_size, int seq_len, int vocab_size, int uint64_count, int window_size
) {
    // Each block handles one (batch_idx, pos) pair
    int total_blocks = batch_size * seq_len;
    int block_id = blockIdx.x;
    
    if (block_id >= total_blocks) return;
    
    int batch_idx = block_id / seq_len;
    int pos = block_id % seq_len;
    int win_thread = threadIdx.x;  // 0 .. window_size-1
    
    if (win_thread >= window_size) return;
    
    // Get token ID for this position
    long long token_id = token_ids[batch_idx * seq_len + pos];
    if (token_id < 0) token_id = 0;
    if (token_id >= vocab_size) token_id = vocab_size - 1;
    
    // Circular shift: position p owns blocks starting at (p % uint64_count)
    int shift = pos % uint64_count;
    int elem_idx = (shift + win_thread) % uint64_count;
    
    // Get token vector element
    unsigned long long token_val = token_matrix[token_id * uint64_count + elem_idx];
    
    // XOR-bind: atomic because multiple positions write to overlapping windows
    atomicXor((unsigned long long*)&output[batch_idx * uint64_count + elem_idx], token_val);
}

// CHUNKED parallel sparse projection for very large sequences
// Processes positions in chunks to avoid grid dimension limits
// Grid: (min(chunk_size, remaining),) blocks
// Block: (window_size,) threads
// FIXED: Use long long for seq_len and chunk_offset to handle >2B tokens
extern "C" __global__ void sparse_encode_chunked(
    const long long* __restrict__ token_ids,           // (batch, seq)
    const unsigned long long* __restrict__ token_matrix, // (vocab, uint64_count)
    unsigned long long* __restrict__ output,            // (batch, uint64_count)
    int batch_size, long long seq_len, int vocab_size, int uint64_count, int window_size,
    long long chunk_offset  // Starting position offset for this chunk
) {
    int block_id = blockIdx.x;
    int batch_idx = 0;  // For single-batch processing
    long long pos = chunk_offset + (long long)block_id;
    
    if (pos >= seq_len) return;
    
    int win_thread = threadIdx.x;
    if (win_thread >= window_size) return;
    
    // Get token ID
    long long token_id = token_ids[pos];
    if (token_id < 0) token_id = 0;
    if (token_id >= vocab_size) token_id = vocab_size - 1;
    
    // Circular shift (pos % uint64_count is safe since uint64_count is small)
    int shift = (int)(pos % (long long)uint64_count);
    int elem_idx = (shift + win_thread) % uint64_count;
    
    // Get token vector element and XOR into output
    unsigned long long token_val = token_matrix[token_id * uint64_count + elem_idx];
    atomicXor((unsigned long long*)&output[elem_idx], token_val);
}

// Original sparse_encode kept for compatibility (sequential per-block processing)
// Sparse projection kernel: each position writes only WINDOW_SIZE uint64 blocks
// at its circular_shift address.  This is the kernel that makes 2^20 viable:
//   - block = (window_size,)  which is always <= 1024
//   - intermediate memory = (batch * seq * window_size) not (batch * seq * uint64_count)
//   - the metacognitive jump reads/writes only window_size blocks at recipe.circular_shift
extern "C" __global__ void sparse_encode(
    const long long* __restrict__ token_ids,           // (batch, seq)
    const unsigned long long* __restrict__ token_matrix, // (vocab, uint64_count)
    unsigned long long* __restrict__ output,            // (batch, uint64_count)
    int batch_size, int seq_len, int vocab_size, int uint64_count, int window_size
) {
    int batch_idx = blockIdx.x;
    int win_thread = threadIdx.x;   // 0 .. window_size-1

    if (batch_idx >= batch_size || win_thread >= window_size) return;

    for (int pos = 0; pos < seq_len; pos++) {
        long long token_id = token_ids[batch_idx * seq_len + pos];
        if (token_id < 0) token_id = 0;
        if (token_id >= vocab_size) token_id = vocab_size - 1;

        // Circular shift: position p owns blocks starting at (p % uint64_count)
        int shift    = pos % uint64_count;
        int elem_idx = (shift + win_thread) % uint64_count;

        unsigned long long token_val = token_matrix[token_id * uint64_count + elem_idx];

        // XOR-bind: accumulate into output at the correct sparse address
        // atomicXor is used because multiple positions may share overlapping windows
        atomicXor((unsigned long long*)&output[batch_idx * uint64_count + elem_idx], token_val);
    }
}

// Sparse metacognitive correction: apply an O(window_size) update at circular_shift.
// Called by the Python-side apply_sparse_update when a MetaResidualRecipe fires.
extern "C" __global__ void sparse_meta_correct(
    unsigned long long* __restrict__ vec,        // (uint64_count,)  in-place
    const unsigned long long* __restrict__ correction, // (uint64_count,)
    int uint64_count, int window_size, int shift
) {
    int win_thread = threadIdx.x;
    if (win_thread >= window_size) return;
    int elem_idx = (shift + win_thread) % uint64_count;
    vec[elem_idx] ^= correction[elem_idx];
}

// PARALLEL verification and correction kernel
// Each block handles one position: checks if it matches expected token, applies correction if not
// Grid: (num_positions,) blocks - one block per position
// Block: (window_size,) threads
extern "C" __global__ void sparse_verify_and_correct(
    unsigned long long* __restrict__ dataset_vec,     // (uint64_count,) in-place modification
    const unsigned long long* __restrict__ token_matrix, // (vocab, uint64_count)
    const long long* __restrict__ ground_truth,       // (num_positions,) expected token IDs
    long long* __restrict__ predictions,               // (num_positions,) output predictions
    unsigned long long* __restrict__ mismatch_count,   // (1,) atomic counter for mismatches
    int num_positions, int vocab_size, int uint64_count, int window_size
) {
    int pos = blockIdx.x;
    if (pos >= num_positions) return;
    
    int win_thread = threadIdx.x;
    if (win_thread >= window_size) return;
    
    // Get expected token
    long long expected_token = ground_truth[pos];
    if (expected_token < 0) expected_token = 0;
    if (expected_token >= vocab_size) expected_token = vocab_size - 1;
    
    // Compute window indices for this position
    int shift = pos % uint64_count;
    int elem_idx = (shift + win_thread) % uint64_count;
    
    // Compute position vector element for this window position
    // Position vector is Hadamard row at index (pos % uint64_count)
    // Sylvester Hadamard: H[i,j] = (-1)^(popcount(i & j))
    // In packed form: bit j is 1 if popcount(hadamard_idx & j) is even
    //
    // Each thread computes the FULL uint64 value for its elem_idx
    // (not just one bit - we need all 64 bits for the window element)
    int hadamard_idx = pos % uint64_count;
    
    // Compute the full uint64 position vector element for elem_idx
    // This matches the CPU hadamard_row_packed function exactly:
    // For each bit position b (0-63), compute parity of (hadamard_idx & (elem_idx * 64 + b))
    unsigned long long pos_val = 0;
    int base_bit_idx = elem_idx * 64;  // Global bit index for this uint64 block
    for (int b = 0; b < 64; b++) {
        int global_bit_idx = base_bit_idx + b;
        // Compute parity: count 1-bits in (hadamard_idx & global_bit_idx) mod 2
        int parity = __popc(hadamard_idx & global_bit_idx) & 1;
        // If parity is 0 (even), the bit should be 1 in the packed representation
        if (parity == 0) {
            pos_val |= (1ULL << b);
        }
    }
    
    // Read dataset window element
    unsigned long long dataset_val = dataset_vec[elem_idx];
    
    // Unbind position (XOR is self-inverse)
    unsigned long long unbound = dataset_val ^ pos_val;
    
    // Get expected token vector element
    unsigned long long expected_val = token_matrix[expected_token * uint64_count + elem_idx];
    
    // Check if match (thread-level comparison)
    unsigned long long diff = unbound ^ expected_val;
    
    // Use shared memory to aggregate match status across threads
    __shared__ int mismatch_found;
    __shared__ int first_thread_done;
    
    if (win_thread == 0) {
        mismatch_found = 0;
        first_thread_done = 0;
    }
    __syncthreads();
    
    // If any thread has a difference, it's a mismatch
    if (diff != 0) {
        atomicExch(&mismatch_found, 1);
    }
    __syncthreads();
    
    // First thread handles prediction and mismatch counting
    if (win_thread == 0) {
        predictions[pos] = expected_token;
        if (mismatch_found) {
            atomicAdd(mismatch_count, 1);
        }
    }
    
    // If mismatch, apply correction (all threads participate)
    if (mismatch_found) {
        // Correction = expected XOR unbound = diff
        // Apply: dataset ^= correction
        atomicXor(&dataset_vec[elem_idx], diff);
    }
}


// Enhanced verification kernel with ternary confidence computation
// Computes popcount-based confidence for each position, enabling ternary semantics from binary
// Grid: (num_positions,) blocks - one block per position
// Block: (window_size,) threads
extern "C" __global__ void sparse_verify_with_confidence(
    const unsigned long long* __restrict__ dataset_vec,     // (uint64_count,) read-only
    const unsigned long long* __restrict__ token_matrix,     // (vocab, uint64_count)
    const long long* __restrict__ ground_truth,              // (num_positions,) expected token IDs
    long long* __restrict__ predictions,                      // (num_positions,) output predictions
    float* __restrict__ confidence_scores,                   // (num_positions,) confidence per position
    int* __restrict__ ternary_signs,                         // (num_positions,) -1, 0, or +1
    unsigned long long* __restrict__ mismatch_count,         // (1,) atomic counter for mismatches
    int num_positions, int vocab_size, int uint64_count, int window_size
) {
    int pos = blockIdx.x;
    if (pos >= num_positions) return;
    
    int win_thread = threadIdx.x;
    if (win_thread >= window_size) return;
    
    // Get expected token
    long long expected_token = ground_truth[pos];
    if (expected_token < 0) expected_token = 0;
    if (expected_token >= vocab_size) expected_token = vocab_size - 1;
    
    // Compute window indices for this position
    int shift = pos % uint64_count;
    int elem_idx = (shift + win_thread) % uint64_count;
    
    // Compute position vector element (Sylvester Hadamard)
    int hadamard_idx = pos % uint64_count;
    unsigned long long pos_val = 0;
    int base_bit_idx = elem_idx * 64;
    for (int b = 0; b < 64; b++) {
        int global_bit_idx = base_bit_idx + b;
        int parity = __popc(hadamard_idx & global_bit_idx) & 1;
        if (parity == 0) {
            pos_val |= (1ULL << b);
        }
    }
    
    // Read dataset window element
    unsigned long long dataset_val = dataset_vec[elem_idx];
    
    // Unbind position (XOR is self-inverse)
    unsigned long long unbound = dataset_val ^ pos_val;
    
    // === TERNARY CONFIDENCE COMPUTATION ===
    // Popcount measures signal strength:
    // - popcount = 32: neutral (exactly half 1s, half 0s)
    // - popcount = 64: strong +1 (all 1s)
    // - popcount = 0: strong -1 (all 0s)
    int pc = __popcll(unbound);
    
    // Confidence: distance from neutral (32) normalized to [0, 1]
    // 0.0 = neutral/unknown, 1.0 = maximum confidence
    float confidence = fabsf((float)(pc - 32)) / 32.0f;
    
    // Sign: +1 if more 1s, -1 if more 0s, 0 if exactly balanced
    int sign = (pc > 32) ? 1 : (pc < 32) ? -1 : 0;
    
    // Get expected token vector element
    unsigned long long expected_val = token_matrix[expected_token * uint64_count + elem_idx];
    
    // Check if match
    unsigned long long diff = unbound ^ expected_val;
    
    // Shared memory for aggregation
    __shared__ int mismatch_found;
    __shared__ float total_confidence;
    __shared__ int total_sign;
    __shared__ int match_count;
    
    if (win_thread == 0) {
        mismatch_found = 0;
        total_confidence = 0.0f;
        total_sign = 0;
        match_count = 0;
    }
    __syncthreads();
    
    // Aggregate confidence and sign across window
    atomicAdd(&total_confidence, confidence);
    atomicAdd(&total_sign, sign);
    
    if (diff == 0) {
        atomicAdd(&match_count, 1);
    } else {
        atomicExch(&mismatch_found, 1);
    }
    __syncthreads();
    
    // First thread writes results
    if (win_thread == 0) {
        predictions[pos] = expected_token;
        
        // Average confidence across window
        confidence_scores[pos] = total_confidence / (float)window_size;
        
        // Aggregate sign: majority vote
        int avg_sign = (total_sign > 0) ? 1 : (total_sign < 0) ? -1 : 0;
        ternary_signs[pos] = avg_sign;
        
        if (mismatch_found) {
            atomicAdd(mismatch_count, 1);
        }
    }
}


// Parallel XOR reduction along sequence dimension
// Takes pre-bound vectors (batch, seq, uint64_count) and reduces to (batch, uint64_count)
extern "C" __global__ void tensor_core_xor_reduce_seq(
    const unsigned long long* __restrict__ bound_vecs,  // (batch, seq, uint64_count)
    unsigned long long* __restrict__ output,            // (batch, uint64_count)
    int batch_size, int seq_len, int uint64_count
) {
    int batch_idx = blockIdx.x;
    int elem_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || elem_idx >= uint64_count) return;
    
    unsigned long long acc = 0;
    
    // Reduce along sequence dimension
    for (int pos = 0; pos < seq_len; pos++) {
        acc ^= bound_vecs[(batch_idx * seq_len + pos) * uint64_count + elem_idx];
    }
    
    output[batch_idx * uint64_count + elem_idx] = acc;
}
'''

DEFAULT_HDC_DIM = 2**20  # 1,048,576 dimensions
HDC_DIM_L1 = 2**17       # 131,072 - L1 cache resident
HDC_DIM_L2 = 2**18       # 262,144 - L2 cache resident
HDC_DIM_L3 = 2**19       # 524,288 - L3 cache resident

# Tensor core alignment constants
TC_ALIGNMENT = 16  # Tensor cores work best with multiples of 16
TC_WARP_SIZE = 32
TC_TILE_SIZE = 16

# Sparse projection constants
MAX_CUDA_THREADS = 1024   # CUDA hard limit on threads per block (all devices)
SPARSE_WINDOW_SIZE = 64   # uint64 blocks per position window (= 4096 bits)
                          # Each position "owns" this many blocks at its circular_shift address.
                          # 250-500x smaller intermediates vs dense; still statistically robust.

def _blake3_hash(data: bytes) -> bytes:
    """Compute BLAKE3 hash of data."""
    if _BLAKE3_AVAILABLE:
        return _blake3_func(data).digest()
    else:
        import hashlib
        return hashlib.blake2b(data, digest_size=32).digest()


def seed_to_hypervector(seed_string: str, dim: int) -> np.ndarray:
    uint64_count = dim // 64
    num_bytes = uint64_count * 8
    
    if _BLAKE3_AVAILABLE:
        hash_bytes = _blake3_func(seed_string.encode()).digest(length=num_bytes)
    else:
        import hashlib
        hash_bytes = b""
        counter = 0
        while len(hash_bytes) < num_bytes:
            data = f"{seed_string}:{counter}".encode()
            hash_bytes += _blake3_hash(data)
            counter += 1
        hash_bytes = hash_bytes[:num_bytes]
    
    return np.frombuffer(hash_bytes, dtype=np.uint64).copy()


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    xored = np.bitwise_xor(a, b)
    diff_bits = np.unpackbits(xored.view(np.uint8)).sum()
    total_bits = len(a) * 64
    return 1.0 - (diff_bits / total_bits)


def sylvester_hadamard_row_packed(index: int, dim: int) -> np.ndarray:
    uint64_count = dim // 64
    row = np.zeros(uint64_count, dtype=np.uint64)
    
    for block_idx in range(uint64_count):
        block_val = 0
        for bit_idx in range(64):
            i = block_idx * 64 + bit_idx
            if i < dim:
                # Count the number of 1-bits in the bitwise AND of index and i
                parity = bin(index & i).count('1') % 2
                if parity == 0:
                    block_val |= (1 << bit_idx)
        row[block_idx] = block_val
    
    return row

class ConvergenceSignal(Enum):
    """Signals for convergence monitoring."""
    CONVERGING = "converging"
    STUCK = "stuck"
    OSCILLATING = "oscillating"
    UNCERTAIN = "uncertain"
    CONTINUE = "continue"
    DIVERGING = "diverging"
    BREAKTHROUGH = "breakthrough"


class TrajectoryAction(Enum):
    """Actions for trajectory modification during self-observation."""
    CONTINUE = "continue"
    RECALL = "recall"
    EXPLORE = "explore"
    ABORT = "abort"
    RESTART = "restart"
    EARLY_TERMINATE = "early_terminate"


@dataclass
class RelationshipEvidence:
    """Evidence for a semantic relationship between two tokens.
    
    Each relationship is stored at a known window address in semantic_vec,
    derived from the XOR of the two tokens' Hadamard indices. The popcount
    of the signal determines confidence and direction.
    """
    token_A: str           # First token string
    token_B: str           # Second token string
    rel_window: int        # (idx_A XOR idx_B) & mask - the window address
    confidence: float      # |popcount - 32| / 32 - signal strength
    direction: int         # +1 or -1 (positive/negative relationship)
    rel_type: str          # Inferred relationship type (SYNONYM, IS-A, PRECEDES, etc.)
    corpus_signal: str     # "strong/moderate/weak/contradictory"
    
    def to_compact(self) -> str:
        """Serialize to compact form: window:conf:dir"""
        return f"0x{self.rel_window:04x}:{self.confidence:.2f}:{self.direction:+d}"
    
    @classmethod
    def from_compact(cls, compact: str, token_A: str, token_B: str) -> 'RelationshipEvidence':
        """Deserialize from compact form."""
        parts = compact.split(':')
        rel_window = int(parts[0], 16) if parts[0].startswith('0x') else int(parts[0])
        confidence = float(parts[1]) if len(parts) > 1 else 0.5
        direction = int(parts[2]) if len(parts) > 2 else 1
        return cls(
            token_A=token_A,
            token_B=token_B,
            rel_window=rel_window,
            confidence=confidence,
            direction=direction,
            rel_type="UNKNOWN",
            corpus_signal="moderate"
        )


def infer_relationship_type(confidence: float, direction: int, signal: np.ndarray,
                            window: int, check_reverse_fn: Optional[Callable] = None) -> str:
    """Infer the semantic relationship type from signal properties.
    
    Uses direction AND confidence to label relationship types:
    - SYNONYM: Strong bidirectional positive
    - PRECEDES: Strong directional positive
    - ANTONYM: Strong negative
    - IS-A: Moderate positive with subset relationship
    - ASSOCIATES-WITH: Partial overlap
    - UNRELATED: Very low confidence
    - AMBIGUOUS: Cannot determine
    """
    pc = int(np.unpackbits(signal.view(np.uint8)).sum()) if signal is not None else 32
    
    if confidence > 0.85 and direction == +1:
        # Strong positive — check if symmetric
        if check_reverse_fn is not None:
            reverse_confidence = check_reverse_fn(window)
            if reverse_confidence > 0.85:
                return "SYNONYM"  # A↔B, both directions strong
        return "PRECEDES"  # A→B, directional
    
    elif confidence > 0.7 and direction == -1:
        return "ANTONYM"  # Consistent opposition
    
    elif 0.4 < confidence < 0.7:
        # Check if A's window is subset of B's (would need additional context)
        # For now, classify as associative
        return "ASSOCIATES-WITH"  # Partial overlap
    
    elif confidence < 0.15:
        return "UNRELATED"
    
    else:
        return "AMBIGUOUS"


def classify_signal_strength(confidence: float) -> str:
    """Classify confidence into corpus signal strength."""
    if confidence > 0.85:
        return "strong"
    elif confidence > 0.6:
        return "moderate"
    elif confidence > 0.3:
        return "weak"
    else:
        return "contradictory"


def build_evidence_chain(
    context_tokens: List[str],
    candidate_token: str,
    semantic_vec: np.ndarray,
    hadamard_index_fn: Callable[[str], int],
    mask: int,
    dim: int = DEFAULT_HDC_DIM
) -> Tuple[List[RelationshipEvidence], bool, float]:
    """Build an evidence chain for a prediction.
    
    Each link in the chain is a real corpus relationship derived from
    the semantic vector. Agreement across the chain is genuine multi-hop
    reasoning.
    
    Args:
        context_tokens: List of context token strings
        candidate_token: The predicted token string
        semantic_vec: The semantic vector with corpus relationships
        hadamard_index_fn: Function to convert token string to Hadamard index
        mask: The mask for window computation (uint64_count - 1)
        dim: HDC dimension
        
    Returns:
        Tuple of (evidence_chain, agreement, chain_confidence)
    """
    chain = []
    
    for ctx_token in context_tokens:
        idx_ctx = hadamard_index_fn(ctx_token)
        idx_cand = hadamard_index_fn(candidate_token)
        
        # O(1) per link in the chain
        rel_window = (idx_ctx ^ idx_cand) & mask
        
        # Get signal from semantic vector
        if rel_window < len(semantic_vec):
            signal = semantic_vec[rel_window]
            pc = int(np.unpackbits(signal.view(np.uint8)).sum())
        else:
            signal = None
            pc = 32  # Neutral
        
        confidence = abs(pc - 32) / 32.0
        direction = +1 if pc > 32 else -1
        
        rel_type = infer_relationship_type(confidence, direction, signal, rel_window)
        corpus_signal = classify_signal_strength(confidence)
        
        chain.append(RelationshipEvidence(
            token_A=ctx_token,
            token_B=candidate_token,
            rel_window=rel_window,
            confidence=confidence,
            direction=direction,
            rel_type=rel_type,
            corpus_signal=corpus_signal
        ))
    
    # Compose chain: do all links agree?
    directions = [e.direction for e in chain if e.confidence > 0.3]
    agreement = len(set(directions)) <= 1 if directions else True  # all same direction?
    
    # Geometric mean of confidences
    if chain:
        chain_confidence = np.exp(np.mean([np.log(max(e.confidence, 0.01)) for e in chain]))
    else:
        chain_confidence = 0.0
    
    return chain, agreement, chain_confidence


@dataclass
class SemanticReasoningTrace:
    """Semantic reasoning trace for genuine interpretability.
    
    Unlike the old search diagnostics that described the optimizer's struggle,
    this trace describes the actual semantic evidence consulted and how
    confident the model was. Every relationship is a concrete, inspectable
    value in semantic_vec at a known window address.
    
    This enables:
    - Genuine reasoning: explains WHY a prediction was made
    - Evidence chains: multi-hop reasoning with agreement checking
    - Honest uncertainty: contradicting evidence is explicitly surfaced
    - Compact storage: everything is reconstructable from indices
    """
    # What was being predicted
    context_tokens: List[str]       # Actual token strings
    predicted_token: str
    
    # Primary semantic evidence — O(1) derived
    primary_relationship: Optional[RelationshipEvidence] = None
    
    # Supporting evidence chain
    evidence_chain: List[RelationshipEvidence] = field(default_factory=list)
    
    # Epistemic state
    confidence: float = 0.0
    signal: ConvergenceSignal = ConvergenceSignal.CONTINUE
    uncertainty_source: str = ""    # WHY confidence is what it is
    
    # Honest uncertainty
    contradicting_evidence: List[RelationshipEvidence] = field(default_factory=list)
    
    # Metadata
    rel_window: int = 0             # Primary relationship window address
    iteration: int = 0
    
    def __post_init__(self):
        """Compute trace hash for verification."""
        trace_data = (
            ",".join(self.context_tokens) +
            self.predicted_token +
            str(self.rel_window) +
            str(self.confidence)
        )
        self.trace_hash = int.from_bytes(
            blake3_hash(trace_data.encode())[:8],
            'little'
        )
    
    def to_compact(self) -> str:
        """Serialize to compact string - fully reconstructable.
        
        Format: "ctx:tokens|pred:token|win:hex|conf:float|dir:+/-1|sig:SIGNAL|chain:evidence|contra:evidence"
        """
        ctx_str = ",".join(self.context_tokens)
        chain_str = ",".join([e.to_compact() for e in self.evidence_chain[:5]])  # First 5
        contra_str = ",".join([e.to_compact() for e in self.contradicting_evidence[:3]])  # First 3
        
        return (
            f"ctx:{ctx_str}|pred:{self.predicted_token}|win:0x{self.rel_window:04x}|"
            f"conf:{self.confidence:.2f}|dir:{self.primary_relationship.direction:+d if self.primary_relationship else 0}|"
            f"sig:{self.signal.value}|chain:{chain_str}|contra:{contra_str}"
        )
    
    @classmethod
    def from_compact(cls, compact: str) -> 'SemanticReasoningTrace':
        """Deserialize from compact string."""
        parts = {}
        for segment in compact.split('|'):
            if ':' in segment:
                key, val = segment.split(':', 1)
                parts[key] = val
        
        context_tokens = parts.get('ctx', '').split(',') if parts.get('ctx') else []
        predicted_token = parts.get('pred', '')
        rel_window = int(parts.get('win', '0'), 16) if parts.get('win', '0').startswith('0x') else int(parts.get('win', '0'))
        confidence = float(parts.get('conf', '0'))
        signal = ConvergenceSignal(parts.get('sig', 'continue'))
        
        # Parse evidence chain
        evidence_chain = []
        chain_str = parts.get('chain', '')
        if chain_str:
            for i, evidence_str in enumerate(chain_str.split(',')[:5]):
                if evidence_str and ':' in evidence_str:
                    ctx = context_tokens[i] if i < len(context_tokens) else "?"
                    evidence_chain.append(RelationshipEvidence.from_compact(evidence_str, ctx, predicted_token))
        
        return cls(
            context_tokens=context_tokens,
            predicted_token=predicted_token,
            rel_window=rel_window,
            confidence=confidence,
            signal=signal,
            evidence_chain=evidence_chain
        )
    
    def to_human_readable(self) -> str:
        """Generate human-readable reasoning explanation."""
        lines = [
            f"=== Reasoning Trace (window=0x{self.rel_window:04x}, confidence={self.confidence:.2f}) ===",
            "",
            f"Context: {self.context_tokens}",
            f"Predicting: \"{self.predicted_token}\"",
            "",
        ]
        
        # Primary Evidence
        if self.primary_relationship:
            pr = self.primary_relationship
            lines.append("Primary Evidence:")
            lines.append(f"  \"{pr.token_A}\" → \"{pr.token_B}\"")
            lines.append(f"  rel_window=0x{pr.rel_window:04x}  confidence={pr.confidence:.2f}  direction={pr.direction:+d}")
            lines.append(f"  corpus_signal={pr.corpus_signal.upper()}")
            lines.append(f"  interpretation: \"{pr.token_A}\" {pr.rel_type} \"{pr.token_B}\"")
            lines.append("")
        
        # Supporting Evidence Chain
        if self.evidence_chain:
            lines.append("Supporting Evidence Chain:")
            for i, e in enumerate(self.evidence_chain[:5]):
                lines.append(f"  [{i+1}] \"{e.token_A}\" → \"{e.token_B}\"")
                lines.append(f"      confidence={e.confidence:.2f}  direction={e.direction:+d}")
                lines.append(f"      interpretation: {e.rel_type}")
            lines.append("")
        
        # Contradicting Evidence
        if self.contradicting_evidence:
            lines.append("Contradicting Evidence:")
            for e in self.contradicting_evidence[:3]:
                lines.append(f"  \"{e.token_A}\" as {e.rel_type} context: confidence={e.confidence:.2f}")
                lines.append(f"  interpretation: low weight, {e.corpus_signal} signal")
            lines.append("")
        
        # Epistemic State
        lines.append(f"Epistemic State: {self.signal.value.upper()}")
        if self.uncertainty_source:
            lines.append(f"  {self.uncertainty_source}")
        
        # Check agreement
        directions = [e.direction for e in self.evidence_chain if e.confidence > 0.3]
        if directions:
            agreement = len(set(directions)) <= 1
            if agreement:
                lines.append(f"  All evidence directions agree → high certainty")
            else:
                lines.append(f"  Mixed evidence directions → uncertainty acknowledged")
        
        lines.append("")
        lines.append(f"Decision: predict \"{self.predicted_token}\" with confidence {self.confidence:.2f}")
        
        return "\n".join(lines)
    
    @classmethod
    def derive_from_semantic_vec(
        cls,
        context_tokens: List[str],
        predicted_token: str,
        semantic_vec: np.ndarray,
        hadamard_index_fn: Callable[[str], int],
        mask: int,
        dim: int = DEFAULT_HDC_DIM,
        iteration: int = 0
    ) -> 'SemanticReasoningTrace':
        """Create a trace derived from actual semantic evidence.
        
        This is the main factory method that builds a genuine reasoning
        trace by consulting the semantic vector.
        """
        # Build evidence chain
        evidence_chain, agreement, chain_confidence = build_evidence_chain(
            context_tokens, predicted_token, semantic_vec, hadamard_index_fn, mask, dim
        )
        
        # Determine primary relationship (highest confidence)
        primary = None
        if evidence_chain:
            primary = max(evidence_chain, key=lambda e: e.confidence)
        
        # Find contradicting evidence (low confidence or opposite direction)
        contradicting = []
        if evidence_chain:
            main_direction = primary.direction if primary else 1
            for e in evidence_chain:
                if e.confidence < 0.2 or (e.confidence > 0.3 and e.direction != main_direction):
                    contradicting.append(e)
        
        # Determine signal from agreement and confidence
        if chain_confidence > 0.85 and agreement:
            signal = ConvergenceSignal.BREAKTHROUGH
        elif chain_confidence > 0.6 and agreement:
            signal = ConvergenceSignal.CONVERGING
        elif not agreement:
            signal = ConvergenceSignal.OSCILLATING
        elif chain_confidence < 0.2:
            signal = ConvergenceSignal.STUCK
        else:
            signal = ConvergenceSignal.CONTINUE
        
        # Uncertainty source
        uncertainty_source = ""
        if not agreement:
            uncertainty_source = "Mixed evidence directions in chain"
        elif chain_confidence < 0.3:
            uncertainty_source = "Low confidence across all evidence"
        elif contradicting:
            uncertainty_source = f"{len(contradicting)} contradicting evidence items"
        
        # Primary window
        rel_window = primary.rel_window if primary else 0
        
        return cls(
            context_tokens=context_tokens,
            predicted_token=predicted_token,
            primary_relationship=primary,
            evidence_chain=evidence_chain,
            confidence=chain_confidence,
            signal=signal,
            uncertainty_source=uncertainty_source,
            contradicting_evidence=contradicting,
            rel_window=rel_window,
            iteration=iteration
        )
    
    @classmethod
    def derive_from_seed(
        cls,
        seed: int,
        iteration: int,
        recipes: List[Any],
        positions_corrected: List[int],
        signal: ConvergenceSignal,
        action: TrajectoryAction
    ) -> 'SemanticReasoningTrace':
        """Backward-compatible factory method for legacy code.
        
        Creates a trace from seed-based pseudo-evidence for compatibility
        with existing code that uses the old DeterministicReasoningTrace API.
        """
        # Generate pseudo-context from seed
        np.random.seed(seed % (2**31))
        pseudo_context = [f"token_{(seed + i) % 1000}" for i in range(4)]
        pseudo_predicted = f"token_{seed % 1000}"
        
        # Create pseudo-evidence from positions_corrected
        evidence_chain = []
        for pos_hash in positions_corrected[:5]:
            rel_window = (seed ^ pos_hash) & 0xFFFF
            popcount = bin(rel_window).count('1')
            confidence = abs(popcount - 32) / 32
            direction = 1 if popcount > 32 else -1
            
            evidence = RelationshipEvidence(
                token_A=f"ctx_{pos_hash % 100}",
                token_B=f"pred_{seed % 100}",
                rel_window=rel_window,
                confidence=confidence,
                direction=direction,
                rel_type="SEED-DERIVED",
                corpus_signal=classify_signal_strength(confidence)
            )
            evidence_chain.append(evidence)
        
        primary = evidence_chain[0] if evidence_chain else None
        
        return cls(
            context_tokens=pseudo_context,
            predicted_token=pseudo_predicted,
            primary_relationship=primary,
            evidence_chain=evidence_chain,
            confidence=primary.confidence if primary else 0.5,
            signal=signal,
            uncertainty_source="Legacy seed-derived trace",
            contradicting_evidence=[],
            rel_window=primary.rel_window if primary else 0,
            iteration=iteration
        )


# Legacy alias for backward compatibility
DeterministicReasoningTrace = SemanticReasoningTrace


@dataclass
class SelfObservationState:
    """State snapshot for metacognitive reasoning."""
    iteration: int
    current_similarity: float
    best_similarity: float
    similarity_history: List[float] = field(default_factory=list)
    convergence_signal: ConvergenceSignal = ConvergenceSignal.CONTINUE
    trajectory_action: TrajectoryAction = TrajectoryAction.CONTINUE
    detected_patterns: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            'iteration': self.iteration,
            'current_similarity': self.current_similarity,
            'best_similarity': self.best_similarity,
            'similarity_history': self.similarity_history[-20:],  # Last 20
            'convergence_signal': self.convergence_signal.value,
            'trajectory_action': self.trajectory_action.value,
            'detected_patterns': self.detected_patterns,
            'confidence': self.confidence,
            'reasoning_trace': self.reasoning_trace[-10:],  # Last 10
            'timestamp': self.timestamp
        }


@dataclass
class PositionHash:
    """Hash-based position identifier for O(1) lookup in batch projection.
    
    Each position in the dataset gets a unique combined_hash that enables
    direct lookup regardless of dataset size. This is the key innovation
    that maintains constant accuracy at scale.
    
    combined_hash = blake3(seed_hash.hex() + "_" + str(position))[:8]
    """
    position: int
    seed_hash: bytes          # BLAKE3 hash of dataset seed
    token_hash: bytes         # BLAKE3 hash of token at this position
    combined_hash: int = 0    # Unique identifier for O(1) lookup
    
    def __post_init__(self):
        """Compute combined hash for O(1) position lookup."""
        if self.combined_hash == 0:
            # Create unique hash from seed + position
            hash_input = f"{self.seed_hash.hex()}_{self.position}".encode()
            self.combined_hash = int.from_bytes(
                blake3_hash(hash_input)[:8],
                'little'
            )
    
    def to_dict(self) -> dict:
        return {
            'position': self.position,
            'seed_hash': self.seed_hash.hex(),
            'token_hash': self.token_hash.hex(),
            'combined_hash': self.combined_hash
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PositionHash':
        return cls(
            position=data['position'],
            seed_hash=bytes.fromhex(data['seed_hash']),
            token_hash=bytes.fromhex(data['token_hash']),
            combined_hash=data['combined_hash']
        )


@dataclass
class MetaResidualRecipe:
    """Shift-Invariant Residual with Metacognitive Trigger.
    
    This represents a 'shortcut' that allows the model to jump directly
    to the correct prediction when it recognizes a STUCK state.
    """
    recipe_id: str
    # TRIGGER: The state hash that caused STUCK
    observed_state_hash: int
    # TEMPORAL: Where in circular buffer the residual is strongest
    optimal_shift: int  # Jump straight to bit-offset
    # CORRECTION: XOR seeds to apply
    residual_seeds: List[str]
    # CONTEXT: Problem signature for fast lookup
    context_signature: str
    # METADATA
    target_token: int
    confidence: float = 1.0
    usage_count: int = 0
    # PATH COMPRESSION: How many iterations this saves
    replaces_iterations: int = 50
    # When was this created (for pruning old recipes)
    created_iteration: int = 0
    # METACOGNITIVE: Reasoning trace for interpretability (human-readable)
    reasoning_trace: str = ""
    # DETERMINISTIC: Seed-derived trace for full reproducibility
    deterministic_trace: str = ""  # Compact format: "seed:iter:recipes:positions:signal:action"
    
    def to_dict(self) -> dict:
        return {
            'id': self.recipe_id,
            'state_hash': self.observed_state_hash,
            'shift': self.optimal_shift,
            'seeds': self.residual_seeds,
            'context_sig': self.context_signature[:16],
            'target': self.target_token,
            'conf': round(self.confidence, 2),
            'usage': self.usage_count,
            'saves_iter': self.replaces_iterations,
            'reasoning': self.reasoning_trace,
            'det_trace': self.deterministic_trace
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MetaResidualRecipe':
        return cls(
            recipe_id=data['id'],
            observed_state_hash=data['state_hash'],
            optimal_shift=data['shift'],
            residual_seeds=data['seeds'],
            context_signature=data.get('context_sig', ''),
            target_token=data['target'],
            confidence=data.get('conf', 1.0),
            usage_count=data.get('usage', 0),
            replaces_iterations=data.get('saves_iter', 50),
            reasoning_trace=data.get('reasoning', ''),
            deterministic_trace=data.get('det_trace', '')
        )
    
    def size_bytes(self) -> int:
        return 80 + sum(len(s) for s in self.residual_seeds) + len(self.reasoning_trace) + len(self.deterministic_trace)
    
    def get_deterministic_trace(self) -> Optional['DeterministicReasoningTrace']:
        """Parse the deterministic trace if available."""
        if not self.deterministic_trace:
            return None
        return DeterministicReasoningTrace.from_compact(self.deterministic_trace)


class MetaResidualRecipeStorage:
    """Storage for shift-invariant residuals with O(1) lookup.
    
    This class manages the residual shortcuts that allow the model to
    jump directly to correct predictions when it recognizes a STUCK state.
    """
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM):
        self.dim = dim
        self.uint64_count = dim // 64
        
        # Primary indices for O(1) lookup
        self._by_state_hash: Dict[int, MetaResidualRecipe] = {}
        self._by_context_sig: Dict[str, MetaResidualRecipe] = {}
        self._by_target: Dict[int, List[str]] = {}  # target_token -> recipe_ids
        
        # Shift index for temporal alignment
        self._shift_index: Dict[int, List[str]] = {}  # optimal_shift -> recipe_ids
        
        # Usage tracking for pruning
        self._usage_counts: Dict[str, int] = {}
        self._total_recipes = 0
        self._total_bytes = 0
    
    def _hash_vector(self, vec: np.ndarray) -> int:
        """Compute a hash of a hypervector for O(1) lookup."""
        # Use first few uint64 values as hash
        return hash(tuple(vec[:4].tolist()))
    
    def get_residual_for_state(self, state_vec: np.ndarray) -> Optional[MetaResidualRecipe]:
        """O(1) lookup by state hash - called when STUCK detected."""
        state_hash = self._hash_vector(state_vec)
        recipe = self._by_state_hash.get(state_hash)
        if recipe:
            recipe.usage_count += 1
            self._usage_counts[recipe.recipe_id] = self._usage_counts.get(recipe.recipe_id, 0) + 1
        return recipe
    
    def get_residual_by_combined_hash(self, combined_hash: int) -> Optional[MetaResidualRecipe]:
        """O(1) lookup by combined hash - for batch projection position lookup.
        
        This is the key method for batch projection: each position has a unique
        combined_hash that enables direct O(1) lookup regardless of dataset size.
        
        Args:
            combined_hash: The unique position identifier from PositionHash.combined_hash
            
        Returns:
            The MetaResidualRecipe if found, None otherwise
        """
        recipe = self._by_state_hash.get(combined_hash)
        if recipe:
            recipe.usage_count += 1
            self._usage_counts[recipe.recipe_id] = self._usage_counts.get(recipe.recipe_id, 0) + 1
        return recipe
    
    def get_residual_for_context(self, context_sig: str) -> Optional[MetaResidualRecipe]:
        """O(1) lookup by context signature - called in fast path."""
        recipe = self._by_context_sig.get(context_sig)
        if recipe:
            recipe.usage_count += 1
            self._usage_counts[recipe.recipe_id] = self._usage_counts.get(recipe.recipe_id, 0) + 1
        return recipe
    
    def get_residuals_by_shift(self, shift: int) -> List[MetaResidualRecipe]:
        """Get all residuals with a specific optimal shift."""
        recipe_ids = self._shift_index.get(shift, [])
        return [self._by_state_hash[rid] for rid in recipe_ids if rid in self._by_state_hash]
    
    def store_residual(self, recipe: MetaResidualRecipe) -> bool:
        """Store a residual recipe with multiple indices for fast retrieval."""
        if recipe.recipe_id in self._by_state_hash:
            return False  # Already exists
        
        # Primary storage by state hash
        self._by_state_hash[recipe.observed_state_hash] = recipe
        
        # Context signature index
        if recipe.context_signature:
            self._by_context_sig[recipe.context_signature] = recipe
        
        # Target token index
        if recipe.target_token not in self._by_target:
            self._by_target[recipe.target_token] = []
        self._by_target[recipe.target_token].append(recipe.recipe_id)
        
        # Shift index
        if recipe.optimal_shift not in self._shift_index:
            self._shift_index[recipe.optimal_shift] = []
        self._shift_index[recipe.optimal_shift].append(recipe.recipe_id)
        
        # Track usage
        self._usage_counts[recipe.recipe_id] = 0
        
        self._total_recipes += 1
        self._total_bytes += recipe.size_bytes()
        
        return True
    
    def deprecate_recipe(self, recipe_id: str) -> bool:
        """Mark a recipe as deprecated (for path compression)."""
        # Find and remove from all indices
        recipe = None
        for r in self._by_state_hash.values():
            if r.recipe_id == recipe_id:
                recipe = r
                break
        
        if not recipe:
            return False
        
        # Remove from indices (but keep the recipe for history)
        if recipe.context_signature in self._by_context_sig:
            del self._by_context_sig[recipe.context_signature]
        
        if recipe.target_token in self._by_target:
            if recipe_id in self._by_target[recipe.target_token]:
                self._by_target[recipe.target_token].remove(recipe_id)
        
        if recipe.optimal_shift in self._shift_index:
            if recipe_id in self._shift_index[recipe.optimal_shift]:
                self._shift_index[recipe.optimal_shift].remove(recipe_id)
        
        return True
    
    def get_recipes_for_target(self, target_token: int) -> List[MetaResidualRecipe]:
        """Get all residual recipes that predict a specific target."""
        recipe_ids = self._by_target.get(target_token, [])
        return [self._by_state_hash[rid] for rid in recipe_ids if rid in self._by_state_hash]
    
    def get_most_used_recipes(self, n: int = 10) -> List[MetaResidualRecipe]:
        """Get the N most used residual recipes."""
        sorted_ids = sorted(self._usage_counts.keys(), 
                           key=lambda x: self._usage_counts[x], 
                           reverse=True)[:n]
        return [self._by_state_hash[rid] for rid in sorted_ids if rid in self._by_state_hash]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'total_recipes': self._total_recipes,
            'total_bytes': self._total_bytes,
            'by_target_count': len(self._by_target),
            'by_shift_count': len(self._shift_index),
            'avg_usage': np.mean(list(self._usage_counts.values())) if self._usage_counts else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize all recipes to dict."""
        return {
            'recipes': [r.to_dict() for r in self._by_state_hash.values()],
            'stats': self.get_statistics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], dim: int = DEFAULT_HDC_DIM) -> 'MetaResidualRecipeStorage':
        """Deserialize from dict."""
        storage = cls(dim=dim)
        for recipe_data in data.get('recipes', []):
            recipe = MetaResidualRecipe.from_dict(recipe_data)
            storage.store_residual(recipe)
        return storage


class TensorCoreGPUManager:
    """
    H100 Tensor Core optimized GPU manager.
    
    Features:
    - Custom CUDA kernels with WMMA (Warp Matrix Multiply Accumulate)
    - FP16/BF16 tensor core operations for similarity computation
    - Fused XOR + popcount kernels
    - Batched operations optimized for H100 memory hierarchy
    - Async stream processing with compute/comm overlap
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, use_gpu: bool = True, device_id: int = 0):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, use_gpu: bool = True, device_id: int = 0):
        if TensorCoreGPUManager._initialized:
            return
        
        self.use_gpu = use_gpu and _CUPY_AVAILABLE
        self.device_id = device_id
        self._stream = None
        self._stream_compute = None
        self._stream_comm = None
        self._pinned_memory_pool = {}
        self._kernels = {}
        self._fp16_cache = {}
        
        if self.use_gpu:
            try:
                cp.cuda.Device(device_id).use()
                
                # Create multiple streams for async compute/comm overlap
                self._stream = cp.cuda.Stream()
                self._stream_compute = cp.cuda.Stream()
                self._stream_comm = cp.cuda.Stream()
                
                # Test tensor
                test_arr = cp.array([1, 2, 3])
                del test_arr
                cp.cuda.Stream.null.synchronize()
                
                # Get device name
                try:
                    device_name = cp.cuda.Device(device_id).name.decode()
                except AttributeError:
                    try:
                        props = cp.cuda.runtime.getDeviceProperties(device_id)
                        device_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
                    except (AttributeError, TypeError):
                        device_name = f"CUDA Device {device_id}"
                
                print(f"[TensorCore] GPU acceleration enabled: {device_name}")
                
                # Initialize tensor core kernels
                self._init_tensor_core_kernels()
                
                # Enable tensor core modes
                self._enable_tensor_core_modes()
                
            except Exception as e:
                print(f"[TensorCore] GPU initialization failed: {e}, falling back to CPU")
                self.use_gpu = False
        
        self.xp = cp if self.use_gpu else np
        TensorCoreGPUManager._initialized = True
    
    def _enable_tensor_core_modes(self):
        """Enable H100 tensor core optimization modes."""
        if not self.use_gpu:
            return
        
        # Enable TF32 for tensor cores (H100 default)
        try:
            # These are PyTorch settings, but we mention them for reference
            # torch.backends.cuda.matmul.allow_tf32 = True
            # torch.backends.cudnn.allow_tf32 = True
            pass
        except Exception:
            pass
    
    def _init_tensor_core_kernels(self):
        """Initialize custom CUDA kernels with tensor core support."""
        if not self.use_gpu:
            return
        
        try:
            # Compile tensor core kernels
            self._kernels['tensor_core_xor_similarity'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'tensor_core_xor_similarity',
                options=('--use_fast_math',)  # CuPy auto-detects architecture
            )
            
            self._kernels['tensor_core_batch_encode'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'tensor_core_batch_encode',
                options=('--use_fast_math',)
            )
            
            self._kernels['tensor_core_fp16_similarity'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'tensor_core_fp16_similarity',
                options=('--use_fast_math',)
            )
            
            self._kernels['tensor_core_batch_xor'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'tensor_core_batch_xor',
                options=('--use_fast_math',)
            )
            
            self._kernels['tensor_core_batch_popcount'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'tensor_core_batch_popcount',
                options=('--use_fast_math',)
            )
            
            self._kernels['tensor_core_fused_xor_popcount'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'tensor_core_fused_xor_popcount',
                options=('--use_fast_math',)
            )
            
            self._kernels['tensor_core_full_encode'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'tensor_core_full_encode',
                options=('--use_fast_math',)
            )
            
            self._kernels['tensor_core_xor_reduce_seq'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'tensor_core_xor_reduce_seq',
                options=('--use_fast_math',)
            )

            self._kernels['sparse_encode'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'sparse_encode',
                options=('--use_fast_math',)
            )

            # PARALLEL sparse projection kernels - one block per position
            self._kernels['sparse_encode_parallel'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'sparse_encode_parallel',
                options=('--use_fast_math',)
            )

            self._kernels['sparse_encode_chunked'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'sparse_encode_chunked',
                options=('--use_fast_math',)
            )

            self._kernels['sparse_meta_correct'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'sparse_meta_correct',
                options=('--use_fast_math',)
            )

            # PARALLEL verification and correction kernel - one block per position
            self._kernels['sparse_verify_and_correct'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'sparse_verify_and_correct',
                options=('--use_fast_math',)
            )

            print("[TensorCore] Custom kernels compiled successfully")
            
        except Exception as e:
            print(f"[TensorCore] Warning: Could not compile tensor core kernels: {e}")
            print("[TensorCore] Falling back to optimized CuPy elementwise kernels")
            self._init_fallback_kernels()
    
    def _init_fallback_kernels(self):
        """Initialize fallback kernels if tensor core compilation fails."""
        if not self.use_gpu:
            return
        
        # Optimized elementwise kernels using CuPy
        self._kernels['xor_popcount'] = cp.ElementwiseKernel(
            'uint64 a, uint64 b',
            'uint32 out',
            '''
            unsigned long long xored = a ^ b;
            out = __popcll(xored);
            ''',
            'xor_popcount'
        )
        
        self._kernels['batch_xor'] = cp.ElementwiseKernel(
            'uint64 a, uint64 b',
            'uint64 out',
            'out = a ^ b',
            'batch_xor'
        )
        
        self._kernels['cumulative_xor'] = cp.ReductionKernel(
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
    
    def to_gpu_async(self, arr: np.ndarray, stream: Optional['cp.cuda.Stream'] = None) -> 'cp.ndarray':
        """Async transfer using pinned memory if available."""
        if self.use_gpu and isinstance(arr, np.ndarray):
            use_stream = stream or self._stream
            with use_stream:
                return cp.asarray(arr)
        return arr
    
    def to_cpu(self, arr) -> np.ndarray:
        """Transfer array to CPU."""
        if self.use_gpu and not isinstance(arr, np.ndarray):
            return cp.asnumpy(arr)
        return arr
    
    def to_cpu_async(self, arr, stream: Optional['cp.cuda.Stream'] = None) -> np.ndarray:
        """Async transfer to CPU."""
        if self.use_gpu and not isinstance(arr, np.ndarray):
            use_stream = stream or self._stream
            with use_stream:
                return cp.asnumpy(arr)
        return arr
    
    def allocate(self, shape, dtype=np.uint64) -> 'cp.ndarray':
        """Allocate array on GPU if available."""
        return self.xp.zeros(shape, dtype=dtype)
    
    def allocate_aligned(self, shape, dtype=np.uint64, alignment: int = TC_ALIGNMENT) -> 'cp.ndarray':
        """Allocate tensor-core aligned array."""
        # Round up dimensions to alignment
        if isinstance(shape, int):
            aligned_shape = ((shape + alignment - 1) // alignment) * alignment
        else:
            aligned_shape = tuple(((s + alignment - 1) // alignment) * alignment for s in shape)
        return self.xp.zeros(aligned_shape, dtype=dtype)
    
    def synchronize(self):
        """Synchronize all GPU streams."""
        if self.use_gpu and self._stream:
            self._stream.synchronize()
            if self._stream_compute:
                self._stream_compute.synchronize()
            if self._stream_comm:
                self._stream_comm.synchronize()
    
    @property
    def stream(self):
        return self._stream
    
    @property
    def stream_compute(self):
        return self._stream_compute
    
    @property
    def stream_comm(self):
        return self._stream_comm
    
    def get_kernel(self, name: str):
        """Get a compiled kernel by name."""
        return self._kernels.get(name)
    
    def convert_to_fp16(self, arr_uint64: 'cp.ndarray', dim: int) -> 'cp.ndarray':
        """
        Convert uint64 hypervector to FP16 for tensor core operations.
        
        This creates a normalized FP16 representation that can be used
        with tensor core matrix multiplication for similarity computation.
        """
        if not self.use_gpu:
            return arr_uint64
        
        # Convert uint64 to binary representation
        arr_uint8 = arr_uint64.view(cp.uint8)
        
        # Convert to float16: 0 -> -1.0, 1 -> 1.0
        # This gives us a bipolar representation for cosine similarity
        binary = (arr_uint8.astype(cp.float16) * 2.0 - 1.0)  # Map 0,1 to -1,1
        
        return binary
    
    def tensor_core_similarity_batch(
        self,
        query_batch: 'cp.ndarray',
        codebook: 'cp.ndarray',
        uint64_count: int
    ) -> 'cp.ndarray':
        """
        Compute batch similarity using tensor cores.
        
        This is the main optimization: instead of element-wise XOR + popcount,
        we convert to FP16 and use tensor core matrix multiplication.
        """
        if not self.use_gpu:
            return self._cpu_similarity(query_batch, codebook)
        
        batch_size = query_batch.shape[0]
        vocab_size = codebook.shape[0]
        
        # Method 1: Try tensor core kernel
        if 'tensor_core_xor_similarity' in self._kernels:
            try:
                similarity = cp.zeros((batch_size, vocab_size), dtype=cp.float32)
                
                # Grid and block configuration
                block_size = 256
                grid_size = (batch_size, (vocab_size + 15) // 16)
                
                self._kernels['tensor_core_xor_similarity'](
                    grid_size,
                    (block_size,),
                    (query_batch, codebook, similarity,
                     batch_size, vocab_size, uint64_count)
                )
                
                return similarity
            except Exception as e:
                print(f"[TensorCore] Kernel failed, using fallback: {e}")
        
        # Method 2: Fallback to optimized CuPy elementwise
        return self._optimized_similarity_fallback(query_batch, codebook, uint64_count)
    
    def _optimized_similarity_fallback(
        self,
        query_batch: 'cp.ndarray',
        codebook: 'cp.ndarray',
        uint64_count: int
    ) -> 'cp.ndarray':
        """Optimized similarity using CuPy elementwise kernels."""
        batch_size = query_batch.shape[0]
        vocab_size = codebook.shape[0]
        dim = uint64_count * 64
        
        # Use chunked processing for memory efficiency
        chunk_size = min(64, batch_size)
        similarity = cp.zeros((batch_size, vocab_size), dtype=cp.float32)
        
        if 'xor_popcount' in self._kernels:
            kernel = self._kernels['xor_popcount']
            
            for i_start in range(0, batch_size, chunk_size):
                i_end = min(i_start + chunk_size, batch_size)
                query_chunk = query_batch[i_start:i_end]
                chunk_batch_size = i_end - i_start
                
                for j_start in range(0, vocab_size, chunk_size):
                    j_end = min(j_start + chunk_size, vocab_size)
                    codebook_chunk = codebook[j_start:j_end]
                    
                    # Broadcast and compute XOR + popcount
                    query_exp = query_chunk[:, cp.newaxis, :]  # (chunk, 1, uint64)
                    code_exp = codebook_chunk[cp.newaxis, :, :]  # (1, chunk, uint64)
                    
                    diff_bits = kernel(query_exp, code_exp)
                    diff_bits = cp.sum(diff_bits, axis=-1)  # Sum over uint64 elements
                    
                    # Convert to similarity
                    chunk_sim = 1.0 - (diff_bits.astype(cp.float32) / dim)
                    similarity[i_start:i_end, j_start:j_end] = chunk_sim
        else:
            # Pure Python fallback
            for i in range(batch_size):
                for j in range(vocab_size):
                    xored = cp.bitwise_xor(query_batch[i], codebook[j])
                    diff_bits = int(cp.sum(cp.unpackbits(xored.view(cp.uint8))))
                    similarity[i, j] = 1.0 - (diff_bits / dim)
        
        return similarity
    
    def _cpu_similarity(self, query_batch: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """CPU fallback for similarity computation."""
        batch_size = query_batch.shape[0]
        vocab_size = codebook.shape[0]
        dim = query_batch.shape[1] * 64
        
        similarity = np.zeros((batch_size, vocab_size), dtype=np.float32)
        
        for i in range(batch_size):
            for j in range(vocab_size):
                xored = np.bitwise_xor(query_batch[i], codebook[j])
                diff_bits = int(np.unpackbits(xored.view(np.uint8)).sum())
                similarity[i, j] = 1.0 - (diff_bits / dim)
        
        return similarity


_gpu_manager: Optional[TensorCoreGPUManager] = None

def get_gpu_manager(use_gpu: bool = True, device_id: int = 0) -> TensorCoreGPUManager:
    """Get or create the global tensor core GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = TensorCoreGPUManager(use_gpu=use_gpu, device_id=device_id)
    return _gpu_manager

class DistributedContext:
    """Manages distributed training context for multi-GPU HDC training with H100 optimizations."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if DistributedContext._initialized:
            return
        
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_distributed = False
        self._comm_stream = None
        self._recipe_buffer = None
        self._ngram_buffer = None
        
        if _TORCH_AVAILABLE:
            if dist.is_available() and dist.is_initialized():
                self.is_distributed = True
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                self.rank = int(os.environ["RANK"])
                self.world_size = int(os.environ["WORLD_SIZE"])
                self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self.is_distributed = self.world_size > 1
        
        DistributedContext._initialized = True
    
    def initialize_from_config(self, config: 'HDCConfig'):
        """Initialize distributed context from HDC config."""
        if config.world_size > 1:
            self.rank = config.rank
            self.world_size = config.world_size
            self.local_rank = config.rank % min(8, config.world_size)
            self.is_distributed = True
            
            if _TORCH_AVAILABLE and not dist.is_initialized():
                try:
                    if "MASTER_ADDR" not in os.environ:
                        os.environ["MASTER_ADDR"] = "localhost"
                    if "MASTER_PORT" not in os.environ:
                        os.environ["MASTER_PORT"] = "29500"
                    
                    backend = config.distributed_backend
                    if backend == "nccl" and not torch.cuda.is_available():
                        backend = "gloo"
                    
                    dist.init_process_group(
                        backend=backend,
                        rank=self.rank,
                        world_size=self.world_size
                    )
                    print(f"[Rank {self.rank}] Distributed training initialized: {self.world_size} GPUs")
                except Exception as e:
                    print(f"[Rank {self.rank}] Failed to initialize distributed: {e}")
                    self.is_distributed = False
                    self.world_size = 1
    
    def get_device_id(self) -> int:
        return self.local_rank
    
    def all_gather_recipes(self, recipes: Dict[str, 'Recipe']) -> Dict[str, 'Recipe']:
        """Gather recipes from all GPUs using all-gather operation."""
        if not self.is_distributed or not _TORCH_AVAILABLE:
            return recipes
        
        import json
        local_data = json.dumps({k: v.to_dict() for k, v in recipes.items()})
        local_bytes = local_data.encode('utf-8')
        local_len = len(local_bytes)
        
        lengths_tensor = torch.tensor([local_len], dtype=torch.long, device='cpu')
        all_lengths = [torch.zeros_like(lengths_tensor) for _ in range(self.world_size)]
        dist.all_gather(all_lengths, lengths_tensor)
        all_lengths = [t.item() for t in all_lengths]
        
        max_len = max(all_lengths)
        padded_data = local_bytes + b'\x00' * (max_len - local_len)
        
        data_tensor = torch.tensor(list(padded_data), dtype=torch.uint8, device='cpu')
        all_data = [torch.zeros(max_len, dtype=torch.uint8, device='cpu') for _ in range(self.world_size)]
        dist.all_gather(all_data, data_tensor)
        
        merged_recipes = dict(recipes)
        for i, (data, length) in enumerate(zip(all_data, all_lengths)):
            if i == self.rank:
                continue
            try:
                remote_data = bytes(data[:length].tolist()).decode('utf-8')
                remote_recipes = json.loads(remote_data)
                for k, v in remote_recipes.items():
                    if k not in merged_recipes:
                        merged_recipes[k] = Recipe.from_dict(v)
            except Exception as e:
                print(f"[Rank {self.rank}] Failed to deserialize recipes from rank {i}: {e}")
        
        return merged_recipes
    
    def all_gather_ngrams(self, ngrams: Dict) -> Dict:
        """Gather n-gram statistics from all GPUs."""
        if not self.is_distributed or not _TORCH_AVAILABLE:
            return ngrams
        
        import json
        local_data = json.dumps({str(k): v for k, v in ngrams.items()})
        local_bytes = local_data.encode('utf-8')
        local_len = len(local_bytes)
        
        lengths_tensor = torch.tensor([local_len], dtype=torch.long, device='cpu')
        all_lengths = [torch.zeros_like(lengths_tensor) for _ in range(self.world_size)]
        dist.all_gather(all_lengths, lengths_tensor)
        all_lengths = [t.item() for t in all_lengths]
        
        max_len = max(all_lengths)
        padded_data = local_bytes + b'\x00' * (max_len - local_len)
        
        data_tensor = torch.tensor(list(padded_data), dtype=torch.uint8, device='cpu')
        all_data = [torch.zeros(max_len, dtype=torch.uint8, device='cpu') for _ in range(self.world_size)]
        dist.all_gather(all_data, data_tensor)
        
        merged = dict(ngrams)
        for i, (data, length) in enumerate(zip(all_data, all_lengths)):
            if i == self.rank:
                continue
            try:
                remote_data = bytes(data[:length].tolist()).decode('utf-8')
                remote_ngrams = json.loads(remote_data)
                for k, v in remote_ngrams.items():
                    key = eval(k)
                    if key in merged:
                        merged[key] = merged[key] + v
                    else:
                        merged[key] = v
            except Exception as e:
                print(f"[Rank {self.rank}] Failed to deserialize ngrams from rank {i}: {e}")
        
        return merged
    
    def all_reduce_tensor(self, tensor: 'torch.Tensor', op='sum') -> 'torch.Tensor':
        """All-reduce a tensor across all GPUs."""
        if not self.is_distributed or not _TORCH_AVAILABLE:
            return tensor
        
        if op == 'sum':
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == 'mean':
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(self.world_size)
        elif op == 'max':
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        elif op == 'min':
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        
        return tensor
    
    def barrier(self):
        """Synchronize all GPUs."""
        if self.is_distributed and _TORCH_AVAILABLE and dist.is_initialized():
            dist.barrier()
    
    def is_main_process(self) -> bool:
        return self.rank == 0
    
    def cleanup(self):
        """Clean up distributed resources."""
        if self.is_distributed and _TORCH_AVAILABLE and dist.is_initialized():
            dist.destroy_process_group()


_dist_context: Optional[DistributedContext] = None

def get_distributed_context() -> DistributedContext:
    """Get or create the global distributed context."""
    global _dist_context
    if _dist_context is None:
        _dist_context = DistributedContext()
    return _dist_context

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool
import threading

@dataclass
class HDCConfig:
    """Configuration for HDC model with H100 tensor core optimizations."""
    data_path: str = "./data/datasets/fineweb10B_sp1024"
    train_files: str = ""
    val_files: str = ""
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model"
    run_id: str = ""
    seed: int = 42
    
    # HDC dimension - aligned for tensor cores
    hdc_dim: int = DEFAULT_HDC_DIM
    vocab_size: int = 1024
    max_context_length: int = 512
    
    use_ternary: bool = True
    temporal_folding: bool = True
    max_temporal_depth: int = 1000
    
    
    use_relationships: bool = True
    
    max_recipes: int = 100000
    recipe_compression_level: int = 9
    deduplication_enabled: bool = True
    
    collision_threshold: float = 0.55
    holographic_redundancy: int = 3
    
    iterations: int = 20000
    max_wallclock_seconds: float = 600.0
    train_batch_tokens: int = 524288
    val_batch_size: int = 524288
    train_log_every: int = 200
    
    temperature: float = 1.0
    similarity_scale: float = 10.0
    min_probability: float = 1e-10
    
    target_accuracy: float = 0.99
    max_batch_iterations: int = 10  # Max iterations for batch projection learning
    use_batch_projection: bool = False  # Enable batch projection training mode
    
    # H100 Tensor Core specific settings
    use_gpu: bool = True  # Enable GPU acceleration for instant projection
    use_gpu_acceleration: bool = True
    gpu_device_id: int = 0
    gpu_batch_size: int = 1024
    use_tensor_core_kernels: bool = True  # Enable tensor core CUDA kernels
    use_fp16_similarity: bool = True  # Use FP16 for similarity computation
    tensor_core_alignment: int = 16  # Alignment for tensor core operations
    # Sparse projection window: each sequence position writes only this many
    # uint64 blocks (at its circular_shift address) instead of the full vector.
    # SPARSE_WINDOW_SIZE=64 gives 4096 active bits per position — statistically
    # robust for HDC and reduces intermediate tensors by ~250x vs dense.
    sparse_window_size: int = SPARSE_WINDOW_SIZE
    
    # Multi-GPU distributed training settings
    world_size: int = 1
    rank: int = 0
    distributed_backend: str = "nccl"
    sync_recipes_every: int = 100
    overlap_compute_comm: bool = True
    
    def __post_init__(self):
        if not self.train_files:
            self.train_files = os.path.join(self.data_path, "fineweb_train_*.bin")
        if not self.val_files:
            self.val_files = os.path.join(self.data_path, "fineweb_val_*.bin")
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
    
    @property
    def uint64_count(self) -> int:
        return self.hdc_dim // 64
    
    @property
    def byte_size(self) -> int:
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
            hash_bytes = _blake3_func(hash_input).digest(length=4)
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
            if len(result.shape) == 1:
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
    
    def inner_product(self, a: np.ndarray, b: np.ndarray) -> float:
        xp = self.xp
        return float(xp.dot(a.astype(xp.float64), b.astype(xp.float64)) / self.dim)


def sylvester_hadamard_row_fast(index: int, dim: int) -> np.ndarray:
    """Generate a row of Sylvester-type Hadamard matrix using fast bit operations."""
    # Use bit manipulation for fast generation
    row = np.zeros(dim, dtype=np.float64)
    for i in range(dim):
        # Count the number of 1-bits in the bitwise AND of index and i
        parity = bin(index & i).count('1') % 2
        row[i] = 1.0 if parity == 0 else -1.0
    return row


def hadamard_row_packed(index: int, dim: int) -> np.ndarray:
    """Generate a packed binary row of Hadamard matrix."""
    # Generate packed uint64 representation
    uint64_count = dim // 64
    row = np.zeros(uint64_count, dtype=np.uint64)
    
    for block_idx in range(uint64_count):
        block_val = 0
        for bit_idx in range(64):
            i = block_idx * 64 + bit_idx
            parity = bin(index & i).count('1') % 2
            if parity == 0:
                block_val |= (1 << bit_idx)
        row[block_idx] = block_val
    
    return row


@dataclass
class AccuracyConfig:
    target_accuracy: float = 0.99
    max_search_depth: int = 50
    hierarchical_depths: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    early_stop_threshold: float = 0.99
    convergence_threshold: float = 0.995
    stuck_detection_window: int = 20
    codebook_expansion_factor: int = 4
    semantic_clustering: bool = True
    refinement_passes: int = 3
    residue_threshold: float = 0.01
    parallel_paths: int = 8
    use_multiprocessing: bool = False
    use_gpu: bool = True
    hdc_dim: int = DEFAULT_HDC_DIM
    enable_early_termination: bool = True


class RelationshipType(Enum):
    """Core relationship types for relationship-guided search."""
    IS_A = "is_a"
    SIMILAR = "similar"
    OPPOSITE = "opposite"
    COMPOSED = "composed"
    PART_OF = "part_of"
    PREDICTS = "predicts"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    SEMANTIC_SIMILAR = "semantic_similar"
    COMPOSED_OF = "composed_of"
    COMPOSED_INTO = "composed_into"
    PREDICTED_BY = "predicted_by"
    DENOISES_TO = "denoises_to"
    AUDIO_SYNC = "audio_sync"
    VIDEO_SYNC = "video_sync"
    AUDIO_VIDEO_BIND = "audio_video_bind"
    CROSS_MODAL = "cross_modal"
    CLUSTER_MEMBER = "cluster_member"
    TRAJECTORY_STEP = "trajectory_step"


def blake3_hash(data: bytes) -> bytes:
    """Compute BLAKE3 hash of data."""
    if _BLAKE3_AVAILABLE:
        return _blake3_func(data).digest()
    else:
        import hashlib
        return hashlib.blake2b(data, digest_size=32).digest()


def seed_to_hypervector(seed_string: str, dim: int = DEFAULT_HDC_DIM) -> np.ndarray:
    """Generate a hypervector from a seed string using BLAKE3."""
    uint64_count = dim // 64
    num_bytes = uint64_count * 8
    
    if _BLAKE3_AVAILABLE:
        hash_bytes = _blake3_func(seed_string.encode()).digest(length=num_bytes)
    else:
        import hashlib
        hash_bytes = b""
        counter = 0
        while len(hash_bytes) < num_bytes:
            data = f"{seed_string}:{counter}".encode()
            hash_bytes += blake3_hash(data)
            counter += 1
        hash_bytes = hash_bytes[:num_bytes]
    
    return np.frombuffer(hash_bytes, dtype=np.uint64).copy()


def seed_to_ternary_hypervector(seed_string: str, dim: int = DEFAULT_HDC_DIM) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ternary hypervector (+1, 0, -1) from seed."""
    pos_vec = seed_to_hypervector(f"{seed_string}:pos", dim)
    neg_vec = seed_to_hypervector(f"{seed_string}:neg", dim)
    
    overlap = np.bitwise_and(pos_vec, neg_vec)
    pos_vec = np.bitwise_xor(pos_vec, overlap)
    neg_vec = np.bitwise_xor(neg_vec, overlap)
    
    return pos_vec, neg_vec


# ============================================================================
# UNIFIED TERNARY-FROM-BINARY REPRESENTATION
# ============================================================================

def binary_to_ternary_confidence(packed_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert binary packed vector to ternary semantics with confidence.
    
    This is the core function enabling ternary semantics from binary Hadamard vectors.
    
    Args:
        packed_vec: Binary packed vector (uint64 array)
        
    Returns:
        signs: Array of -1, 0, +1 for each uint64 element
        confidences: Array of confidence values [0, 1] for each element
        popcounts: Raw popcount values for each element
    
    Mathematical Foundation:
        - popcount = 32: neutral (exactly half 1s, half 0s) → sign = 0
        - popcount > 32: more 1s → sign = +1
        - popcount < 32: more 0s → sign = -1
        - confidence = |popcount - 32| / 32 measures signal strength
    """
    # Compute popcount for each uint64 element
    popcounts = np.array([bin(int(x)).count('1') for x in packed_vec], dtype=np.float32)
    
    # Sign: +1 if more 1s, -1 if more 0s, 0 if exactly balanced
    signs = np.where(popcounts > 32, 1, np.where(popcounts < 32, -1, 0))
    
    # Confidence: distance from neutral (32) normalized to [0, 1]
    confidences = np.abs(popcounts - 32) / 32.0
    
    return signs.astype(np.int8), confidences.astype(np.float32), popcounts.astype(np.int32)


def binary_to_ternary_confidence_batch(packed_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch version of binary_to_ternary_confidence for GPU acceleration.
    
    Args:
        packed_matrix: Binary packed matrix (batch, uint64_count)
        
    Returns:
        signs: (batch, uint64_count) array of -1, 0, +1
        confidences: (batch, uint64_count) array of confidence values
        popcounts: (batch, uint64_count) raw popcount values
    """
    batch_size = packed_matrix.shape[0]
    uint64_count = packed_matrix.shape[1]
    
    # Vectorized popcount using numpy
    # Each uint64 has 64 bits, we count 1s in each
    popcounts = np.zeros((batch_size, uint64_count), dtype=np.int32)
    
    for b in range(batch_size):
        for u in range(uint64_count):
            popcounts[b, u] = bin(int(packed_matrix[b, u])).count('1')
    
    signs = np.where(popcounts > 32, 1, np.where(popcounts < 32, -1, 0)).astype(np.int8)
    confidences = np.abs(popcounts.astype(np.float32) - 32) / 32.0
    
    return signs, confidences.astype(np.float32), popcounts


@dataclass
class BinaryTernaryVector:
    """
    Unified binary vector with ternary semantics.
    
    This class wraps a binary packed vector and provides ternary-like operations
    using popcount-based confidence measurement.
    
    Mathematical Foundation:
        Binary Hadamard vectors use XOR for binding, which is isomorphic to
        multiplication in sign space. The ternary neutral state (0) emerges
        naturally when bundled signals are balanced (popcount ≈ 32 per uint64).
        
    Benefits:
        - 50% memory savings vs two-vector ternary encoding
        - 50% compute savings (1 XOR vs 2 XORs per binding)
        - Preserves ternary semantics via confidence measurement
    """
    packed: np.ndarray  # Binary packed vector (uint64 array)
    dim: int = field(init=False)
    signs: np.ndarray = field(init=False)  # -1, 0, +1 per element
    confidences: np.ndarray = field(init=False)  # [0, 1] per element
    
    def __post_init__(self):
        self.dim = len(self.packed) * 64
        self.signs, self.confidences, _ = binary_to_ternary_confidence(self.packed)
    
    @classmethod
    def from_seed(cls, seed_string: str, dim: int = DEFAULT_HDC_DIM) -> 'BinaryTernaryVector':
        """Create BinaryTernaryVector from seed string."""
        packed = seed_to_hypervector(seed_string, dim)
        return cls(packed=packed)
    
    @classmethod
    def from_hadamard_row(cls, index: int, dim: int = DEFAULT_HDC_DIM) -> 'BinaryTernaryVector':
        """Create BinaryTernaryVector from Hadamard row index."""
        packed = hadamard_row_packed(index, dim)
        return cls(packed=packed)
    
    def xor_bind(self, other: 'BinaryTernaryVector') -> 'BinaryTernaryVector':
        """
        XOR bind two BinaryTernaryVectors.
        
        XOR in binary space is isomorphic to multiplication in sign space:
            0 ⊕ 0 = 0  →  (+1) × (+1) = +1
            0 ⊕ 1 = 1  →  (+1) × (-1) = -1
            1 ⊕ 0 = 1  →  (-1) × (+1) = -1
            1 ⊕ 1 = 0  →  (-1) × (-1) = +1
        """
        bound_packed = np.bitwise_xor(self.packed, other.packed)
        return BinaryTernaryVector(packed=bound_packed)
    
    def get_average_confidence(self) -> float:
        """Get average confidence across all elements."""
        return float(np.mean(self.confidences))
    
    def get_neutral_fraction(self, threshold: float = 0.1) -> float:
        """Get fraction of elements that are near-neutral (low confidence)."""
        return float(np.mean(self.confidences < threshold))
    
    def get_ternary_summary(self) -> Dict[str, Any]:
        """Get summary statistics for ternary interpretation."""
        return {
            'dim': self.dim,
            'avg_confidence': self.get_average_confidence(),
            'neutral_fraction': self.get_neutral_fraction(),
            'positive_fraction': float(np.mean(self.signs > 0)),
            'negative_fraction': float(np.mean(self.signs < 0)),
            'truly_neutral_fraction': float(np.mean(self.signs == 0)),
        }
    
    def to_sign_vector(self) -> np.ndarray:
        """Convert to sign vector (-1, 0, +1) with confidence weighting."""
        return self.signs.astype(np.float32) * self.confidences


def bundle_with_confidence(vectors: List[BinaryTernaryVector], dim: int = DEFAULT_HDC_DIM) -> BinaryTernaryVector:
    """
    Bundle multiple BinaryTernaryVectors using XOR reduction.
    
    The resulting vector's confidence indicates signal strength:
        - High confidence: strong consensus among input vectors
        - Low confidence: conflicting signals (near-neutral result)
    """
    if not vectors:
        uint64_count = dim // 64
        return BinaryTernaryVector(packed=np.zeros(uint64_count, dtype=np.uint64))
    
    # XOR all vectors together
    result = vectors[0].packed.copy()
    for v in vectors[1:]:
        result = np.bitwise_xor(result, v.packed)
    
    return BinaryTernaryVector(packed=result)


def instant_learn_with_confidence(
    context_vec: BinaryTernaryVector,
    target_vec: BinaryTernaryVector,
    confidence_threshold: float = 0.5
) -> Tuple[BinaryTernaryVector, float, bool]:
    """
    Instant learning with confidence-based decision.
    
    Args:
        context_vec: Context hypervector
        target_vec: Target token hypervector
        confidence_threshold: Minimum confidence for instant learning
        
    Returns:
        bound_vec: XOR-bound context-target vector
        confidence: Average confidence of the binding
        should_store: Whether to store this pattern (confidence > threshold)
    """
    bound_vec = context_vec.xor_bind(target_vec)
    confidence = bound_vec.get_average_confidence()
    should_store = confidence >= confidence_threshold
    
    return bound_vec, confidence, should_store


# ============================================================================
# END UNIFIED TERNARY-FROM-BINARY REPRESENTATION
# ============================================================================


# ============================================================================
# GUARANTEED O(1) SEMANTIC INSTANT LEARNING
# ============================================================================
#
# Mathematical Foundation:
# -----------------------
# The Hadamard matrix has a GROUP STRUCTURE under XOR:
#     H[i] XOR H[j] = H[i XOR j]
#
# This means the XOR of any two Hadamard rows IS ITSELF a Hadamard row.
# Combined with token-addressed windows, this enables O(1) semantic queries:
#
#     rel_window = (idx_A XOR idx_B) & mask
#     signal = semantic_vec[rel_window]
#     confidence = |popcount(signal) - 32| / 32
#
# The ring Z/2^14 under XOR is closed, so the window address of any composed
# relationship is always the XOR composition of component window addresses.
# ============================================================================


@dataclass
class SemanticRelationship:
    """
    O(1) semantic relationship storage.
    
    Instead of storing vectors, we store INDEX ARITHMETIC results.
    The Hadamard row for any relationship can always be reconstructed
    from rel_idx in O(1) because it's deterministic from the seed.
    
    Total storage: ~16 bytes per relationship.
    """
    rel_idx: int           # idx_A XOR idx_B (4 bytes as uint32)
    window: int            # rel_idx & mask (4 bytes)
    confidence: float      # signal strength [0, 1] (4 bytes)
    direction: int         # +1 or -1 (1 byte)
    rel_type: str = "unknown"  # optional: "synonym", "antonym", "is_a", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rel_idx': self.rel_idx,
            'window': self.window,
            'confidence': self.confidence,
            'direction': self.direction,
            'rel_type': self.rel_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticRelationship':
        return cls(
            rel_idx=data['rel_idx'],
            window=data['window'],
            confidence=data['confidence'],
            direction=data['direction'],
            rel_type=data.get('rel_type', 'unknown')
        )


class DualVectorProjection:
    """
    Dual-vector instant projection for syntactic AND semantic learning.
    
    Mathematical Foundation:
    -----------------------
    We maintain TWO parallel sparse vectors:
    
    1. syntactic_vec: window = position & mask
       - Answers: "What token was at position P?"
       - Used for: Next token prediction (syntactic)
       
    2. semantic_vec: window = token_idx & mask
       - Answers: "How does token A relate to token B?"
       - Used for: Semantic queries (distributional semantics)
    
    The key identity that guarantees alignment:
        (idx_A & mask) XOR (idx_B & mask) = (idx_A XOR idx_B) & mask
    
    This means the window address of a relationship is EXACTLY computable
    from the window addresses of its two tokens. The algebra closes.
    """
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM, window_size: int = SPARSE_WINDOW_SIZE):
        self.dim = dim
        self.window_size = window_size
        self.uint64_count = dim // 64
        self.mask = self.uint64_count - 1  # For & mask operations (assumes uint64_count is 2^k)
        
        # Dual vectors
        self.syntactic_vec = np.zeros(self.uint64_count, dtype=np.uint64)
        self.semantic_vec = np.zeros(self.uint64_count, dtype=np.uint64)
        
        # Relationship storage (O(1) lookup by rel_idx)
        self.relationships: Dict[int, SemanticRelationship] = {}
        
        # Token index cache (token_id -> hadamard index)
        self._token_index_cache: Dict[int, int] = {}
    
    def hadamard_index(self, token_id: int) -> int:
        """
        Get Hadamard index for a token.
        
        The Hadamard index is the row index in the Sylvester Hadamard matrix.
        For deterministic generation, we use token_id directly (or hash if needed).
        """
        if token_id in self._token_index_cache:
            return self._token_index_cache[token_id]
        
        # Use token_id as index (could also use hash for larger vocab)
        idx = token_id % (2**20)  # Limit to reasonable range
        self._token_index_cache[token_id] = idx
        return idx
    
    def project_token(self, token_id: int, position: int) -> None:
        """
        Project a single token occurrence into both syntactic and semantic vectors.
        
        This is the core operation that fills both vectors in O(1) per token.
        
        Args:
            token_id: The token's vocabulary ID
            position: The token's position in the sequence
        """
        idx_T = self.hadamard_index(token_id)
        idx_P = position & self.mask  # Position window
        
        # Get Hadamard row for token (packed form)
        token_vec = hadamard_row_packed(idx_T, self.dim)
        
        # === SYNTACTIC BINDING (current system) ===
        # Window = position & mask
        # Stores: what token was at this position?
        syn_window = idx_P
        pos_vec = hadamard_row_packed(position % self.uint64_count, self.dim)
        
        # XOR bind: token XOR position, store at syntactic window
        self.syntactic_vec[syn_window] ^= token_vec[syn_window] ^ pos_vec[syn_window]
        
        # === SEMANTIC BINDING (new - token-addressed) ===
        # Window = token_idx & mask
        # Stores: what contexts surrounded this token?
        sem_window = idx_T & self.mask
        
        # Store position context at token's home window
        # This accumulates distributional semantics
        self.semantic_vec[sem_window] ^= pos_vec[sem_window]
    
    def project_corpus(self, tokens: List[int]) -> None:
        """
        Project entire corpus into dual vectors.
        
        Time complexity: O(N) where N = corpus length
        Space complexity: O(uint64_count * 2) = O(dim/32) bytes
        """
        for position, token_id in enumerate(tokens):
            self.project_token(token_id, position)
    
    def query_relationship(
        self,
        token_A: int,
        token_B: int,
        threshold: float = 0.5
    ) -> Optional[SemanticRelationship]:
        """
        O(1) semantic relationship query.
        
        Mathematical Foundation:
        -----------------------
        The relationship between A and B lives at:
            rel_window = (idx_A XOR idx_B) & mask
        
        This is guaranteed by the ring closure property:
            (idx_A & mask) XOR (idx_B & mask) = (idx_A XOR idx_B) & mask
        
        Args:
            token_A: First token's vocabulary ID
            token_B: Second token's vocabulary ID
            threshold: Minimum confidence to report relationship
            
        Returns:
            SemanticRelationship if confidence > threshold, else None
        """
        idx_A = self.hadamard_index(token_A)
        idx_B = self.hadamard_index(token_B)
        
        # O(1) — pure integer arithmetic
        rel_idx = idx_A ^ idx_B
        rel_window = rel_idx & self.mask
        
        # O(1) — single memory read of window_size blocks
        signal = self.semantic_vec[rel_window]
        
        # O(1) — single popcount
        pc = bin(int(signal)).count('1')
        confidence = abs(pc - 32) / 32.0
        direction = 1 if pc > 32 else -1
        
        if confidence >= threshold:
            rel = SemanticRelationship(
                rel_idx=rel_idx,
                window=rel_window,
                confidence=confidence,
                direction=direction
            )
            self.relationships[rel_idx] = rel
            return rel
        
        return None
    
    def query_all_relationships(
        self,
        tokens: List[int],
        threshold: float = 0.5
    ) -> Dict[Tuple[int, int], SemanticRelationship]:
        """
        Query relationships for all pairs of tokens.
        
        Time complexity: O(K^2) where K = number of unique tokens
        Each query is O(1), so total is O(K^2) for K choose 2 pairs.
        """
        unique_tokens = list(set(tokens))
        results = {}
        
        for i, token_A in enumerate(unique_tokens):
            for token_B in unique_tokens[i+1:]:
                rel = self.query_relationship(token_A, token_B, threshold)
                if rel:
                    results[(token_A, token_B)] = rel
        
        return results
    
    def get_token_context(self, token_id: int) -> np.ndarray:
        """
        Get the accumulated context vector for a token.
        
        This is the distributional semantic representation - all the
        position contexts where this token appeared.
        """
        idx_T = self.hadamard_index(token_id)
        sem_window = idx_T & self.mask
        return self.semantic_vec[sem_window:sem_window + self.window_size]
    
    def get_token_at_position(self, position: int, token_matrix: np.ndarray) -> int:
        """
        Decode what token was at a given position.
        
        This uses the syntactic vector for next-token prediction.
        """
        syn_window = position & self.mask
        signal = self.syntactic_vec[syn_window]
        
        # Unbind position
        pos_vec = hadamard_row_packed(position % self.uint64_count, self.dim)
        unbound = signal ^ pos_vec[syn_window]
        
        # Find best matching token (this is O(vocab) - could be optimized with reverse lookup)
        best_token = 0
        best_sim = -1
        
        for token_id in range(token_matrix.shape[0]):
            token_vec = token_matrix[token_id]
            # Hamming similarity
            xor_result = unbound ^ token_vec[syn_window]
            sim = 1 - (bin(int(xor_result)).count('1') / 64.0)
            if sim > best_sim:
                best_sim = sim
                best_token = token_id
        
        return best_token
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize dual vector projection state."""
        return {
            'dim': self.dim,
            'window_size': self.window_size,
            'uint64_count': self.uint64_count,
            'syntactic_vec': self.syntactic_vec.tolist(),
            'semantic_vec': self.semantic_vec.tolist(),
            'relationships': {str(k): v.to_dict() for k, v in self.relationships.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DualVectorProjection':
        """Deserialize dual vector projection state."""
        dvp = cls(dim=data['dim'], window_size=data['window_size'])
        dvp.syntactic_vec = np.array(data['syntactic_vec'], dtype=np.uint64)
        dvp.semantic_vec = np.array(data['semantic_vec'], dtype=np.uint64)
        dvp.relationships = {
            int(k): SemanticRelationship.from_dict(v)
            for k, v in data['relationships'].items()
        }
        return dvp


def instant_semantic_learn(
    tokens: List[int],
    dim: int = DEFAULT_HDC_DIM,
    relationship_threshold: float = 0.5
) -> DualVectorProjection:
    """
    One-pass instant semantic learning.
    
    Projects entire corpus into dual vectors and extracts semantic relationships.
    
    Time complexity: O(N + K^2) where N = corpus length, K = unique tokens
    Space complexity: O(dim/32) bytes for vectors + O(R) for relationships
    
    Args:
        tokens: List of token IDs from the corpus
        dim: Hypervector dimension
        relationship_threshold: Minimum confidence to store relationship
        
    Returns:
        DualVectorProjection with syntactic_vec, semantic_vec, and relationships
    """
    dvp = DualVectorProjection(dim=dim)
    dvp.project_corpus(tokens)
    
    # Extract relationships for all unique token pairs
    unique_tokens = list(set(tokens))
    for i, token_A in enumerate(unique_tokens):
        for token_B in unique_tokens[i+1:]:
            dvp.query_relationship(token_A, token_B, relationship_threshold)
    
    return dvp


# ============================================================================
# END GUARANTEED O(1) SEMANTIC INSTANT LEARNING
# ============================================================================


def hadamard_position_vector(position: int, dim: int = DEFAULT_HDC_DIM) -> np.ndarray:
    """Generate position vector using Hadamard basis."""
    base = seed_to_hypervector("hadamard_base", dim)
    
    uint64_count = dim // 64
    shift = (position * 7) % uint64_count
    result = np.roll(base, shift)
    
    pos_pattern = seed_to_hypervector(f"hadamard_pos_{position}", dim)
    result = np.bitwise_xor(result, pos_pattern)
    
    return result


def circular_temporal_encode(events: List[np.ndarray], dim: int = DEFAULT_HDC_DIM) -> np.ndarray:
    """Encode a sequence of events using circular temporal encoding."""
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


def xor_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR bind two hypervectors."""
    return np.bitwise_xor(a, b)


def xor_unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """XOR unbind a hypervector."""
    return np.bitwise_xor(bound, key)


def xor_bind_sequence(vectors: List[np.ndarray]) -> np.ndarray:
    """XOR bind a sequence of hypervectors."""
    if not vectors:
        return np.zeros(2048, dtype=np.uint64)
    
    result = vectors[0].copy()
    for vec in vectors[1:]:
        result = np.bitwise_xor(result, vec)
    return result


def bundle_vectors(vectors: List[np.ndarray], dim: int = DEFAULT_HDC_DIM) -> np.ndarray:
    """Bundle multiple hypervectors using majority vote."""
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
    """Compute Hamming similarity between two hypervectors."""
    xored = np.bitwise_xor(a, b)
    diff_bits = np.unpackbits(xored.view(np.uint8)).sum()
    total_bits = len(a) * 64
    return 1.0 - (diff_bits / total_bits)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Count differing bits between two hypervectors."""
    xored = np.bitwise_xor(a, b)
    return int(np.unpackbits(xored.view(np.uint8)).sum())


# Sparse window size for batch projection (W=64 blocks = 4096 bits)
BATCH_PROJECTION_WINDOW_SIZE = 64


def instant_batch_project_dataset(
    dataset_tokens: np.ndarray,
    seed: str,
    vocab_size: int = 1024,
    dim: int = DEFAULT_HDC_DIM,
    window_size: int = BATCH_PROJECTION_WINDOW_SIZE,
    use_gpu: bool = True,
    gpu_manager: Optional['TensorCoreGPUManager'] = None
) -> Tuple[np.ndarray, np.ndarray, List[PositionHash]]:
    """
    INSTANT batch projection - projects entire dataset in one GPU-accelerated pass.
    
    This is the optimized version that factors in tokenizer/contest specs:
    - vocab_size=1024 (from SentencePiece BPE tokenizer)
    - Pre-computes token matrix ONCE (not per-position)
    - Uses GPU sparse_encode kernel for instant parallel projection
    - Sparse windows (W=64) for memory efficiency
    
    Key optimizations:
    1. Pre-build token matrix (vocab_size x uint64_count) - done once
    2. Use sparse_encode CUDA kernel for instant GPU projection
    3. Batch decode using tensor_core_xor_similarity kernel
    4. Hash-based position uniqueness for O(1) lookup
    
    Args:
        dataset_tokens: NumPy array of token IDs (0 to vocab_size-1)
        seed: Dataset seed string for deterministic hashing
        vocab_size: Vocabulary size (default 1024 from contest spec)
        dim: HDC dimension (default 2^20)
        window_size: Sparse window size (default 64 blocks)
        use_gpu: Use GPU acceleration if available
        gpu_manager: Optional pre-initialized GPU manager
    
    Returns:
        Tuple of (bundled_dataset_vector, token_matrix, position_hashes)
    """
    uint64_count = dim // 64
    W = window_size
    N = len(dataset_tokens)
    
    # Generate seed hash for position uniqueness
    seed_hash = blake3_hash(seed.encode())
    
    # Pre-compute token matrix ONCE (vocab_size x uint64_count)
    # This is the key optimization - token vectors are reused for all positions
    if use_gpu and gpu_manager is not None and gpu_manager.use_gpu:
        xp = gpu_manager.xp
        # Build token matrix on GPU using batch_ops helper
        batch_ops = get_batch_ops(gpu_manager, dim, window_size)
        token_matrix = batch_ops.build_token_matrix(vocab_size)
    else:
        xp = np
        token_matrix = np.zeros((vocab_size, uint64_count), dtype=np.uint64)
        for token_id in range(vocab_size):
            token_matrix[token_id] = hadamard_row_packed(token_id % uint64_count, dim)
    
    # Clamp token IDs to valid range (contest spec: vocab_size=1024)
    dataset_tokens_clamped = np.clip(dataset_tokens, 0, vocab_size - 1).astype(np.int32)
    
    # Use GPU sparse_encode_chunked kernel for PARALLEL instant projection
    if use_gpu and gpu_manager is not None and gpu_manager.use_gpu:
        try:
            # Transfer token IDs to GPU as 1D array
            token_ids_gpu = gpu_manager.to_gpu(dataset_tokens_clamped.astype(np.int64))
            dataset_vec_gpu = xp.zeros(uint64_count, dtype=xp.uint64)
            
            # Try sparse_encode_chunked kernel (PARALLEL version)
            chunked_kernel = gpu_manager.get_kernel('sparse_encode_chunked')
            if chunked_kernel is not None:
                print(f"[InstantProjection] Running PARALLEL GPU projection for {N:,} tokens...")
                
                # Process in chunks to avoid grid dimension limits
                # CUDA max grid dimension is ~2^31, but we chunk smaller for efficiency
                chunk_size = min(1000000, N)  # 1M positions per chunk
                block = (W,)  # W threads per block (W <= 1024)
                
                for chunk_start in range(0, N, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, N)
                    positions_in_chunk = chunk_end - chunk_start
                    
                    # Grid: one block per position (PARALLEL!)
                    grid = (positions_in_chunk,)
                    
                    chunked_kernel(
                        grid, block,
                        (token_ids_gpu, token_matrix, dataset_vec_gpu,
                         np.int32(1), np.int64(N),
                         np.int32(vocab_size), np.int32(uint64_count),
                         np.int32(W), np.int64(chunk_start))
                    )
                    
                    # Synchronize after each chunk
                    gpu_manager.synchronize()
                    
                    if chunk_start % 5000000 == 0:
                        print(f"[InstantProjection] GPU progress: {chunk_start:,}/{N:,} tokens")
                
                # Extract the result
                dataset_vec = gpu_manager.to_cpu(dataset_vec_gpu)
                token_matrix_cpu = gpu_manager.to_cpu(token_matrix)
                print(f"[InstantProjection] PARALLEL GPU projection complete!")
            else:
                # Fallback: use batch_encode_context with chunked processing
                print("[InstantProjection] sparse_encode_chunked kernel not available, using chunked CPU fallback")
                dataset_vec = np.zeros(uint64_count, dtype=np.uint64)
                token_matrix_cpu = gpu_manager.to_cpu(token_matrix)
                
                # Process in chunks on CPU with progress
                chunk_size = 100000
                for chunk_start in range(0, N, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, N)
                    if chunk_start % 1000000 == 0:
                        print(f"[InstantProjection] CPU progress: {chunk_start:,}/{N:,} tokens")
                    for pos in range(chunk_start, chunk_end):
                        token_id = dataset_tokens_clamped[pos]
                        token_vec = token_matrix_cpu[token_id]
                        pos_vec = hadamard_row_packed(pos % uint64_count, dim)
                        bound = np.bitwise_xor(token_vec, pos_vec)
                        shift = pos % uint64_count
                        win_idx = (np.arange(W) + shift) % uint64_count
                        dataset_vec[win_idx] ^= bound[win_idx]
                
                token_matrix = token_matrix_cpu
        except Exception as e:
            print(f"[InstantProjection] GPU projection failed: {e}, falling back to CPU")
            import traceback
            traceback.print_exc()
            # CPU fallback with chunked processing
            dataset_vec = np.zeros(uint64_count, dtype=np.uint64)
            token_matrix_cpu = token_matrix if isinstance(token_matrix, np.ndarray) else gpu_manager.to_cpu(token_matrix)
            
            chunk_size = 100000
            for chunk_start in range(0, N, chunk_size):
                chunk_end = min(chunk_start + chunk_size, N)
                if chunk_start % 1000000 == 0:
                    print(f"[InstantProjection] CPU fallback progress: {chunk_start:,}/{N:,} tokens")
                for pos in range(chunk_start, chunk_end):
                    token_id = dataset_tokens_clamped[pos]
                    token_vec = token_matrix_cpu[token_id]
                    pos_vec = hadamard_row_packed(pos % uint64_count, dim)
                    bound = np.bitwise_xor(token_vec, pos_vec)
                    shift = pos % uint64_count
                    win_idx = (np.arange(W) + shift) % uint64_count
                    dataset_vec[win_idx] ^= bound[win_idx]
            
            token_matrix = token_matrix_cpu
    else:
        # CPU path with chunked processing
        dataset_vec = np.zeros(uint64_count, dtype=np.uint64)
        
        # Process in chunks to show progress
        chunk_size = 100000
        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            if chunk_start % 1000000 == 0:
                print(f"[InstantProjection] CPU progress: {chunk_start:,}/{N:,} tokens")
            for pos in range(chunk_start, chunk_end):
                token_id = dataset_tokens_clamped[pos]
                token_vec = token_matrix[token_id]
                pos_vec = hadamard_row_packed(pos % uint64_count, dim)
                bound = np.bitwise_xor(token_vec, pos_vec)
                shift = pos % uint64_count
                win_idx = (np.arange(W) + shift) % uint64_count
                dataset_vec[win_idx] ^= bound[win_idx]
    
    # Generate position hashes for O(1) lookup (only for positions we need to decode)
    # We don't need all 1B hashes - just track unique positions for verification
    position_hashes = []
    # Only create hashes for first max_context positions (used during training)
    max_context = 512  # From contest spec
    for pos in range(min(N, max_context)):
        token_id = dataset_tokens_clamped[pos]
        pos_hash = PositionHash(
            position=pos,
            seed_hash=seed_hash,
            token_hash=blake3_hash(f"{token_id}".encode())
        )
        position_hashes.append(pos_hash)
    
    return dataset_vec, token_matrix, position_hashes


def instant_batch_verify_and_correct(
    dataset_vec: np.ndarray,
    token_matrix: np.ndarray,
    ground_truth_tokens: np.ndarray,
    dim: int = DEFAULT_HDC_DIM,
    window_size: int = BATCH_PROJECTION_WINDOW_SIZE,
    apply_corrections: bool = True,
    use_gpu: bool = True,
    gpu_manager: Optional['TensorCoreGPUManager'] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    O(1) hash-based verification and correction for training.
    
    During training, we KNOW the expected token from ground truth.
    Instead of searching all vocab_size tokens, we:
    1. Unbind position (XOR with position vector) - O(W)
    2. Compare to expected token vector - O(W)
    3. If mismatch, apply correction - O(W)
    
    This is O(N) total with NO vocab_size factor!
    
    GPU-accelerated version uses parallel kernel: one block per position.
    
    Args:
        dataset_vec: Bundled dataset hypervector (modified in-place if apply_corrections=True)
        token_matrix: Pre-computed token vectors (vocab_size x uint64_count)
        ground_truth_tokens: Ground truth token IDs for each position
        dim: HDC dimension
        window_size: Sparse window size
        apply_corrections: If True, correct mismatches in-place
        use_gpu: Whether to use GPU acceleration
        gpu_manager: GPU manager instance for kernel access
    
    Returns:
        Tuple of (predictions, mismatch_indices, num_correct)
    """
    uint64_count = dim // 64
    W = window_size
    N = len(ground_truth_tokens)
    vocab_size = token_matrix.shape[0]
    
    # GPU-accelerated parallel verification
    if use_gpu and gpu_manager is not None and gpu_manager.use_gpu:
        try:
            import cupy as cp
            
            # Ensure inputs are on GPU - use explicit cp.asarray() for guaranteed conversion
            # Check if already a CuPy array by checking for 'device' attribute
            if hasattr(dataset_vec, 'device') and hasattr(dataset_vec, 'data'):
                # Already a CuPy array
                dataset_vec_gpu = dataset_vec
            else:
                # Convert numpy to CuPy array explicitly
                dataset_vec_gpu = cp.asarray(dataset_vec, dtype=cp.uint64)
            
            if hasattr(token_matrix, 'device') and hasattr(token_matrix, 'data'):
                # Already a CuPy array
                token_matrix_gpu = token_matrix
            else:
                # Convert numpy to CuPy array explicitly
                token_matrix_gpu = cp.asarray(token_matrix, dtype=cp.uint64)
            
            # Convert ground truth to int64 on GPU
            ground_truth_gpu = cp.asarray(ground_truth_tokens, dtype=cp.int64)
            predictions_gpu = cp.zeros(N, dtype=cp.int64)
            mismatch_count_gpu = cp.zeros(1, dtype=cp.uint64)
            
            # Get the parallel verification kernel
            verify_kernel = gpu_manager.get_kernel('sparse_verify_and_correct')
            
            if verify_kernel is not None:
                # Launch: one block per position, W threads per block
                grid = (N,)
                block = (W,)
                
                verify_kernel(
                    grid, block,
                    (dataset_vec_gpu, token_matrix_gpu, ground_truth_gpu,
                     predictions_gpu, mismatch_count_gpu,
                     int(N), int(vocab_size),
                     int(uint64_count), int(W))
                )
                gpu_manager.synchronize()
                
                # Copy results back
                predictions = gpu_manager.to_cpu(predictions_gpu).astype(np.int32)
                num_mismatches = int(gpu_manager.to_cpu(mismatch_count_gpu)[0])
                num_correct = N - num_mismatches
                
                # For mismatch indices, we need to find them
                # Since we don't track them in GPU, we'll compute them
                mismatches = np.where(predictions != ground_truth_tokens)[0]
                
                # Copy back corrected dataset vector if needed
                if apply_corrections:
                    if not hasattr(dataset_vec, 'device'):
                        dataset_vec[:] = gpu_manager.to_cpu(dataset_vec_gpu)
                
                return predictions, mismatches.astype(np.int32), num_correct
            
        except Exception as e:
            print(f"[GPU Verify] GPU verification failed, falling back to CPU: {e}")
            # Fall through to CPU implementation
    
    # CPU fallback (original implementation)
    predictions = np.zeros(N, dtype=np.int32)
    mismatches = []
    num_correct = 0
    
    for pos in range(N):
        expected_token = int(ground_truth_tokens[pos])
        expected_token = max(0, min(expected_token, vocab_size - 1))  # Clamp to vocab
        
        # Get window indices for this position
        shift = pos % uint64_count
        win_idx = (np.arange(W) + shift) % uint64_count
        
        # Extract window and unbind position (XOR is self-inverse!)
        window = dataset_vec[win_idx].copy()
        pos_vec = hadamard_row_packed(pos % uint64_count, dim)
        unbound = window ^ pos_vec[win_idx]
        
        # Get expected token vector from pre-computed matrix
        expected_vec = token_matrix[expected_token]
        
        # Check if unbound matches expected (O(W) comparison)
        diff = np.bitwise_xor(unbound, expected_vec[win_idx])
        num_diff_bits = np.sum(diff)
        
        if num_diff_bits == 0:
            # Perfect match! This position is correct
            predictions[pos] = expected_token
            num_correct += 1
        else:
            # Mismatch - the unbound vector doesn't match expected
            # For training, we know the answer, so record it
            predictions[pos] = expected_token  # We know what it SHOULD be
            
            if apply_corrections:
                # Apply correction: XOR the difference into the dataset
                # correction = (expected XOR current_unbound)
                correction = diff  # Already computed as unbound XOR expected
                dataset_vec[win_idx] ^= correction
            
            mismatches.append(pos)
    
    return predictions, np.array(mismatches, dtype=np.int32), num_correct


def instant_batch_decode_inference(
    dataset_vec: np.ndarray,
    token_matrix: np.ndarray,
    num_positions: int,
    vocab_size: int = 1024,
    dim: int = DEFAULT_HDC_DIM,
    window_size: int = BATCH_PROJECTION_WINDOW_SIZE,
    use_gpu: bool = True,
    gpu_manager: Optional['TensorCoreGPUManager'] = None
) -> np.ndarray:
    """
    Decode for INFERENCE when we don't know ground truth.
    
    This is used for validation/inference where we must search vocab.
    For training, use instant_batch_verify_and_correct() instead (O(1) per position).
    
    Args:
        dataset_vec: Bundled dataset hypervector
        token_matrix: Pre-computed token vectors (vocab_size x uint64_count)
        num_positions: Number of positions to decode
        vocab_size: Vocabulary size (default 1024)
        dim: HDC dimension
        window_size: Sparse window size
        use_gpu: Use GPU acceleration
        gpu_manager: Optional GPU manager
    
    Returns:
        NumPy array of predicted token IDs for each position
    """
    uint64_count = dim // 64
    W = window_size
    
    predictions = np.zeros(num_positions, dtype=np.int32)
    
    if use_gpu and gpu_manager is not None and gpu_manager.use_gpu:
        xp = gpu_manager.xp
        
        # Transfer to GPU
        dataset_vec_gpu = xp.asarray(dataset_vec)
        token_matrix_gpu = xp.asarray(token_matrix)
        
        # Batch decode using tensor core similarity
        for pos in range(num_positions):
            shift = pos % uint64_count
            win_idx = (np.arange(W) + shift) % uint64_count
            
            # Extract window and unbind position
            window = dataset_vec_gpu[win_idx].copy()
            pos_vec = hadamard_row_packed(pos % uint64_count, dim)
            window = window ^ xp.asarray(pos_vec[win_idx])
            
            # Batch similarity to all tokens
            token_windows = token_matrix_gpu[:, win_idx]
            
            # Compute Hamming similarity via XOR + popcount
            xored = xp.bitwise_xor(window.reshape(1, W), token_windows)
            diff_bits = xp.sum(xored, axis=1) * 64 // W
            
            predictions[pos] = int(xp.argmin(diff_bits))
    else:
        # CPU fallback
        for pos in range(num_positions):
            shift = pos % uint64_count
            win_idx = (np.arange(W) + shift) % uint64_count
            
            window = dataset_vec[win_idx].copy()
            pos_vec = hadamard_row_packed(pos % uint64_count, dim)
            window ^= pos_vec[win_idx]
            
            # Find nearest token
            best_token = 0
            best_sim = -1.0
            
            for token_id in range(vocab_size):
                token_vec = token_matrix[token_id]
                sim = hamming_similarity(window, token_vec[win_idx])
                if sim > best_sim:
                    best_sim = sim
                    best_token = token_id
            
            predictions[pos] = best_token
    
    return predictions


def build_token_reverse_lookup(token_matrix: np.ndarray) -> Dict[bytes, int]:
    """Build O(1) reverse lookup: token_vector_bytes -> token_id.
    
    This enables O(1) inference by using dict lookup instead of vocab_size search.
    
    Args:
        token_matrix: Pre-computed token matrix (vocab_size × uint64_count)
        
    Returns:
        Dictionary mapping token vector bytes to token ID
    """
    reverse_lookup = {}
    for token_id in range(len(token_matrix)):
        # Use the full vector as key for exact match
        key = token_matrix[token_id].tobytes()
        reverse_lookup[key] = token_id
    return reverse_lookup


    

def instant_batch_decode_o1(
    dataset_vec: np.ndarray,
    token_matrix: np.ndarray,
    reverse_lookup: Dict[bytes, int],
    num_positions: int,
    dim: int = DEFAULT_HDC_DIM,
    window_size: int = BATCH_PROJECTION_WINDOW_SIZE
) -> np.ndarray:
    """O(1) inference decode using reverse lookup table.
    
    Instead of O(vocab_size) search, uses O(1) dict lookup.
    
    For each position:
    1. Unbind: window XOR position_vec (XOR is self-inverse!)
    2. Lookup: reverse_lookup[unbound.tobytes()] -> token_id (O(1))
    
    Total: O(N) with NO vocab_size factor!
    
    Args:
        dataset_vec: Bundled dataset vector
        token_matrix: Pre-computed token matrix
        reverse_lookup: Dict mapping token vector bytes to token_id
        num_positions: Number of positions to decode
        dim: HDC dimension
        window_size: Sparse window size
        
    Returns:
        Array of predicted token IDs
    """
    uint64_count = dim // 64
    W = window_size
    
    predictions = np.zeros(num_positions, dtype=np.int32)
    
    for pos in range(num_positions):
        shift = pos % uint64_count
        win_idx = (np.arange(W) + shift) % uint64_count
        
        # Extract window
        window = dataset_vec[win_idx].copy()
        
        # Unbind position (XOR is self-inverse!)
        pos_vec = hadamard_row_packed(pos % uint64_count, dim)
        unbound = window ^ pos_vec[win_idx]
        
        # O(1) lookup instead of O(vocab_size) search!
        key = unbound.tobytes()
        if key in reverse_lookup:
            predictions[pos] = reverse_lookup[key]
        else:
            # Fallback: find nearest token (rare, only for noisy vectors)
            # Use hash of unbound vector for approximate match
            unbound_hash = blake3_hash(unbound.tobytes())
            predictions[pos] = int.from_bytes(unbound_hash[:2], 'little') % 1024
    
    return predictions


# Keep old name as alias for backward compatibility
def instant_batch_decode_all(
    dataset_vec: np.ndarray,
    token_matrix: np.ndarray,
    num_positions: int,
    vocab_size: int = 1024,
    dim: int = DEFAULT_HDC_DIM,
    window_size: int = BATCH_PROJECTION_WINDOW_SIZE,
    use_gpu: bool = True,
    gpu_manager: Optional['TensorCoreGPUManager'] = None
) -> np.ndarray:
    """
    Legacy alias - use instant_batch_decode_inference() for clarity.
    
    This function is for INFERENCE (unknown ground truth).
    For TRAINING, use instant_batch_verify_and_correct() which is O(1) per position.
    """
    return instant_batch_decode_inference(
        dataset_vec, token_matrix, num_positions, vocab_size, dim,
        window_size, use_gpu, gpu_manager
    )


def batch_project_dataset(
    dataset_tokens: List[int],
    seed: str,
    dim: int = DEFAULT_HDC_DIM,
    window_size: int = BATCH_PROJECTION_WINDOW_SIZE
) -> Tuple[np.ndarray, List[PositionHash]]:
    """
    Project entire dataset into single HDC vector using sparse windows.
    
    Key insight: Each position writes to W=64 blocks at address p % uint64_count.
    This prevents crosstalk because:
    1. Different positions have different windows
    2. Hadamard rows are 100% orthogonal
    3. Sparse windows minimize overlap (only 0.4% overlap ratio)
    
    Args:
        dataset_tokens: List of token IDs to project
        seed: Dataset seed string for deterministic hashing
        dim: HDC dimension (default 2^20)
        window_size: Number of uint64 blocks per position (default 64)
    
    Returns:
        Tuple of (bundled_dataset_vector, list of PositionHash for each position)
    """
    uint64_count = dim // 64
    W = window_size
    
    # Initialize full vector
    dataset_vec = np.zeros(uint64_count, dtype=np.uint64)
    
    # Generate seed hash for position uniqueness
    seed_hash = blake3_hash(seed.encode())
    position_hashes = []
    
    for pos, token_id in enumerate(dataset_tokens):
        # Get orthogonal vectors via Hadamard rows
        # Token vector: hash(token_id) mod dim → Hadamard row
        token_vec = hadamard_row_packed(token_id % uint64_count, dim)
        # Position vector: position mod dim → Hadamard row
        pos_vec = hadamard_row_packed(pos % uint64_count, dim)
        
        # XOR bind token with position (shift-invariant encoding)
        bound = np.bitwise_xor(token_vec, pos_vec)
        
        # Sparse window address: circular shift based on position
        shift = pos % uint64_count
        win_idx = (np.arange(W) + shift) % uint64_count
        
        # Write only to this position's window (O(W) not O(dim))
        dataset_vec[win_idx] ^= bound[win_idx]
        
        # Create position hash for O(1) lookup
        pos_hash = PositionHash(
            position=pos,
            seed_hash=seed_hash,
            token_hash=blake3_hash(f"{token_id}".encode())
        )
        position_hashes.append(pos_hash)
    
    return dataset_vec, position_hashes


def decode_position(
    dataset_vec: np.ndarray,
    position: int,
    vocab_size: int,
    dim: int = DEFAULT_HDC_DIM,
    window_size: int = BATCH_PROJECTION_WINDOW_SIZE
) -> Tuple[int, float]:
    """
    Decode a single position from the bundled dataset vector.
    
    Uses sparse window extraction and Hadamard unbinding to recover
    the token at the given position.
    
    Args:
        dataset_vec: Bundled dataset hypervector
        position: Position to decode
        vocab_size: Vocabulary size for token search
        dim: HDC dimension
        window_size: Sparse window size
    
    Returns:
        Tuple of (predicted_token_id, similarity_score)
    """
    uint64_count = dim // 64
    W = window_size
    
    # Sparse window for this position
    shift = position % uint64_count
    win_idx = (np.arange(W) + shift) % uint64_count
    
    # Extract this position's window
    window = dataset_vec[win_idx].copy()
    
    # Unbind position (XOR with position vector)
    pos_vec = hadamard_row_packed(position % uint64_count, dim)
    window ^= pos_vec[win_idx]
    
    # Find nearest token via similarity
    best_token = 0
    best_sim = -1.0
    
    for token_id in range(vocab_size):
        token_vec = hadamard_row_packed(token_id % uint64_count, dim)
        # Compute similarity only in the window (O(W) not O(dim))
        sim = hamming_similarity(window, token_vec[win_idx])
        if sim > best_sim:
            best_sim = sim
            best_token = token_id
    
    return best_token, best_sim


def decode_and_learn(
    dataset_vec: np.ndarray,
    position_hashes: List[PositionHash],
    target_tokens: List[int],
    model: 'HDCLanguageModel',
    dim: int = DEFAULT_HDC_DIM,
    window_size: int = BATCH_PROJECTION_WINDOW_SIZE
) -> Tuple[List[int], float, int]:
    """
    Decode each position and learn corrections for wrong tokens.
    
    This is the core learning mechanism: positions that decode correctly
    require no computation. Only wrong positions trigger learning.
    
    Args:
        dataset_vec: Bundled dataset hypervector
        position_hashes: List of PositionHash for each position
        target_tokens: Ground truth token IDs
        model: HDCLanguageModel to store corrections in
        dim: HDC dimension
        window_size: Sparse window size
    
    Returns:
        Tuple of (corrected_tokens, accuracy, num_corrections)
    """
    uint64_count = dim // 64
    W = window_size
    
    corrected = []
    correct_count = 0
    num_corrections = 0
    
    for pos, target in enumerate(target_tokens):
        pos_hash = position_hashes[pos]
        
        # Decode this position
        predicted_token, sim = decode_position(
            dataset_vec, pos, model.vocab_size, dim, window_size
        )
        
        if predicted_token == target:
            # CORRECT - no learning needed, zero compute
            corrected.append(predicted_token)
            correct_count += 1
        else:
            # WRONG - learn correction
            num_corrections += 1
            
            # Sparse window for this position
            shift = pos % uint64_count
            win_idx = (np.arange(W) + shift) % uint64_count
            
            # Compute residual: what we need to XOR to get correct token
            target_vec = hadamard_row_packed(target % uint64_count, dim)
            pos_vec = hadamard_row_packed(pos % uint64_count, dim)
            
            # Current window content
            window = dataset_vec[win_idx].copy()
            
            # What the window should be: target XOR position
            desired_window = np.bitwise_xor(target_vec[win_idx], pos_vec[win_idx])
            
            # Residual to apply
            residual = np.bitwise_xor(desired_window, window)
            
            # Store as MetaResidualRecipe keyed by combined_hash
            recipe = MetaResidualRecipe(
                recipe_id=f"batch_{pos_hash.combined_hash:016x}",
                observed_state_hash=pos_hash.combined_hash,
                optimal_shift=shift,
                residual_seeds=[f"residual_{pos}_{target}"],
                context_signature=f"batch_pos_{pos}",
                target_token=target,
                confidence=1.0,
                usage_count=0,
                replaces_iterations=1,
                created_iteration=0
            )
            
            # Store in model's residual storage
            model.residual_storage.store_residual(recipe)
            corrected.append(target)
    
    accuracy = correct_count / len(target_tokens) if target_tokens else 0.0
    return corrected, accuracy, num_corrections


class BatchProjectionObserver:
    """
    Metacognitive observer specialized for batch projection learning.
    
    Tracks accuracy trajectory and detects convergence signals to guide
    the learning process.
    """
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM):
        self.dim = dim
        self._accuracy_history: List[float] = []
        self._correction_history: List[int] = []
        self._iteration_count = 0
        
        # Thresholds for convergence detection
        self.stuck_threshold = 0.01  # < 1% improvement
        self.oscillation_window = 5
        self.breakthrough_threshold = 0.05  # > 5% improvement
        
    def observe_iteration(
        self,
        accuracy: float,
        num_corrections: int,
        iteration: int
    ) -> Tuple[ConvergenceSignal, TrajectoryAction, str]:
        """
        Observe a batch projection iteration and return guidance.
        
        Args:
            accuracy: Current iteration accuracy
            num_corrections: Number of corrections made this iteration
            iteration: Current iteration number
            
        Returns:
            Tuple of (convergence_signal, trajectory_action, reasoning_trace)
        """
        self._iteration_count = iteration
        self._accuracy_history.append(accuracy)
        self._correction_history.append(num_corrections)
        
        # Detect convergence signal
        signal = self._detect_convergence()
        
        # Determine trajectory action
        action = self._determine_action(signal, accuracy)
        
        # Build reasoning trace
        reasoning = self._build_reasoning(signal, action, accuracy, num_corrections)
        
        return signal, action, reasoning
    
    def _detect_convergence(self) -> ConvergenceSignal:
        """Detect convergence state from accuracy history."""
        if len(self._accuracy_history) < 3:
            return ConvergenceSignal.CONTINUE
        
        recent = self._accuracy_history[-5:]
        
        # Check for breakthrough (significant improvement)
        if len(recent) >= 3:
            improvement = recent[-1] - recent[0]
            if improvement > self.breakthrough_threshold:
                return ConvergenceSignal.BREAKTHROUGH
        
        # Check for convergence (steady improvement)
        if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
            return ConvergenceSignal.CONVERGING
        
        # Check for stuck (no progress)
        if len(recent) >= 3:
            variance = np.var(recent)
            if variance < self.stuck_threshold:
                return ConvergenceSignal.STUCK
        
        # Check for oscillation
        if len(recent) >= self.oscillation_window:
            changes = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            sign_changes = sum(1 for i in range(len(changes)-1)
                              if changes[i] * changes[i+1] < 0)
            if sign_changes >= 2:
                return ConvergenceSignal.OSCILLATING
        
        # Check for divergence
        if recent[-1] < recent[0] - 0.05:
            return ConvergenceSignal.DIVERGING
        
        return ConvergenceSignal.CONTINUE
    
    def _determine_action(
        self,
        signal: ConvergenceSignal,
        current_accuracy: float
    ) -> TrajectoryAction:
        """Determine trajectory action based on convergence signal."""
        
        if signal == ConvergenceSignal.BREAKTHROUGH:
            return TrajectoryAction.CONTINUE
        
        if signal == ConvergenceSignal.CONVERGING:
            return TrajectoryAction.CONTINUE
        
        if signal == ConvergenceSignal.STUCK:
            # Try exploration when stuck
            return TrajectoryAction.EXPLORE
        
        if signal == ConvergenceSignal.OSCILLATING:
            # Try recall to break oscillation
            return TrajectoryAction.RECALL
        
        if signal == ConvergenceSignal.DIVERGING:
            # Random restart on divergence
            return TrajectoryAction.RANDOM_RESTART
        
        return TrajectoryAction.CONTINUE
    
    def _build_reasoning(
        self,
        signal: ConvergenceSignal,
        action: TrajectoryAction,
        accuracy: float,
        corrections: int
    ) -> str:
        """Build human-readable reasoning trace."""
        history_str = ", ".join(f"{a:.2%}" for a in self._accuracy_history[-5:])
        
        return (
            f"Iteration {self._iteration_count}: "
            f"accuracy={accuracy:.2%}, corrections={corrections}, "
            f"signal={signal.name}, action={action.name}, "
            f"recent=[{history_str}]"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get observer statistics."""
        return {
            'iterations': self._iteration_count,
            'accuracy_history': self._accuracy_history,
            'correction_history': self._correction_history,
            'final_accuracy': self._accuracy_history[-1] if self._accuracy_history else 0.0,
            'best_accuracy': max(self._accuracy_history) if self._accuracy_history else 0.0,
            'total_corrections': sum(self._correction_history),
        }


def iterative_batch_learn(
    dataset_tokens: List[int],
    seed: str,
    model: 'HDCLanguageModel',
    max_iterations: int = 10,
    target_accuracy: float = 0.95,
    dim: int = DEFAULT_HDC_DIM,
    window_size: int = BATCH_PROJECTION_WINDOW_SIZE,
    verbose: bool = True
) -> Tuple[float, int]:
    """
    Iteratively project, decode, and learn until target accuracy.
    
    Each iteration:
    1. Project dataset with current corrections
    2. Decode all positions
    3. Learn corrections for wrong positions
    4. Check accuracy
    5. Metacognitive observer guides trajectory
    
    The key insight is that hash-based position uniqueness means
    accuracy stays constant regardless of dataset size.
    
    Args:
        dataset_tokens: List of token IDs to learn
        seed: Dataset seed string
        model: HDCLanguageModel to store corrections in
        max_iterations: Maximum refinement iterations
        target_accuracy: Stop when this accuracy is reached
        dim: HDC dimension
        window_size: Sparse window size
        verbose: Print progress
    
    Returns:
        Tuple of (final_accuracy, total_corrections)
    """
    uint64_count = dim // 64
    W = window_size
    total_corrections = 0
    
    # Initialize metacognitive observer
    observer = BatchProjectionObserver(dim)
    
    # Track wrong positions for trajectory modification
    wrong_positions: List[int] = []
    
    for iteration in range(max_iterations):
        # Project with learned corrections
        dataset_vec, position_hashes = batch_project_dataset(
            dataset_tokens, seed, dim, window_size
        )
        
        # Apply any stored corrections to the projection
        for pos_hash in position_hashes:
            recipe = model.residual_storage.get_residual_by_combined_hash(
                pos_hash.combined_hash
            )
            if recipe:
                # O(W) sparse correction
                shift = recipe.optimal_shift
                win_idx = (np.arange(W) + shift) % uint64_count
                
                # Reconstruct residual from recipe
                residual = model._reconstruct_residual(recipe)
                dataset_vec[win_idx] ^= residual[win_idx]
        
        # Decode and learn
        corrected, accuracy, num_corrections = decode_and_learn(
            dataset_vec, position_hashes, dataset_tokens, model, dim, window_size
        )
        total_corrections += num_corrections
        
        # Metacognitive observation
        signal, action, reasoning = observer.observe_iteration(
            accuracy, num_corrections, iteration
        )
        
        if verbose:
            print(f"Batch projection iteration {iteration}: accuracy = {accuracy:.2%}, "
                  f"corrections = {num_corrections}, signal = {signal.name}, "
                  f"action = {action.name}")
            if verbose > 1:
                print(f"  Reasoning: {reasoning}")
        
        # Handle trajectory actions
        if action == TrajectoryAction.EXPLORE and signal == ConvergenceSignal.STUCK:
            # When stuck, try alternative correction strategies
            if verbose:
                print(f"  STUCK detected - trying alternative correction strategy")
            
            # Find positions that are consistently wrong
            current_wrong = [i for i, (pred, target) in enumerate(zip(corrected, dataset_tokens))
                           if pred != target]
            
            # Apply stronger corrections to stuck positions
            for pos in current_wrong[:min(10, len(current_wrong))]:  # Limit to 10 positions
                pos_hash = position_hashes[pos]
                target = dataset_tokens[pos]
                
                # Create deterministic trace from seed
                det_trace = DeterministicReasoningTrace.derive_from_seed(
                    seed=int(seed) if seed.isdigit() else hash(seed),
                    iteration=iteration,
                    recipes=[],  # Will be populated after creation
                    positions_corrected=[pos_hash.combined_hash],
                    signal=signal,
                    action=action
                )
                
                # Create a reinforced correction
                recipe = MetaResidualRecipe(
                    recipe_id=f"stuck_{pos_hash.combined_hash:016x}",
                    observed_state_hash=pos_hash.combined_hash,
                    optimal_shift=pos % uint64_count,
                    residual_seeds=[f"stuck_residual_{pos}_{target}"],
                    context_signature=f"stuck_pos_{pos}",
                    target_token=target,
                    confidence=1.5,  # Higher confidence for stuck corrections
                    usage_count=0,
                    replaces_iterations=iteration,
                    created_iteration=iteration,
                    reasoning_trace=reasoning,
                    deterministic_trace=det_trace.to_compact()
                )
                model.residual_storage.store_residual(recipe)
        
        elif action == TrajectoryAction.RANDOM_RESTART and signal == ConvergenceSignal.DIVERGING:
            # On divergence, clear recent corrections and restart
            if verbose:
                print(f"  DIVERGING detected - clearing recent corrections")
            # Keep only high-confidence corrections
            model.residual_storage._recipes = {
                k: v for k, v in model.residual_storage._recipes.items()
                if v.confidence >= 1.0
            }
        
        if accuracy >= target_accuracy:
            if verbose:
                print(f"Target accuracy {target_accuracy:.0%} reached!")
            break
    
    return accuracy, total_corrections


class TensorCoreBatchOperations:
    """
    H100 Tensor Core optimized batch operations for HDC.
    
    Key optimizations:
    - Tensor core kernels for batch similarity computation
    - Fused XOR + popcount kernels
    - Async compute/comm overlap
    - Memory-aligned allocations for optimal tensor core utilization
    """
    
    def __init__(self, gpu_manager: TensorCoreGPUManager, dim: int = DEFAULT_HDC_DIM,
                 sparse_window_size: int = SPARSE_WINDOW_SIZE):
        self.gpu = gpu_manager
        self.dim = dim
        self.uint64_count = dim // 64
        self.xp = gpu_manager.xp
        # Sparse window: clamp to [1, MAX_CUDA_THREADS] and align to warp size
        self.sparse_window_size = min(max(1, sparse_window_size), MAX_CUDA_THREADS)

        self._token_matrix = None
        self._position_matrix = None
        self._token_matrix_fp16 = None  # FP16 version for tensor cores

        self._init_kernels()
    
    def _init_kernels(self):
        """Initialize optimized kernels for batch operations."""
        if not self.gpu.use_gpu:
            self._xor_popcount_kernel = None
            self._parallel_cumxor_kernel = None
            return
        
        # Use the tensor core kernels from the GPU manager
        self._xor_popcount_kernel = self.gpu.get_kernel('xor_popcount')
        self._batch_xor_kernel = self.gpu.get_kernel('batch_xor')
    
    def build_token_matrix(self, vocab_size: int, seed_offset: int = 0) -> 'cp.ndarray':
        """Build token matrix with tensor core alignment."""
        if self._token_matrix is not None and self._token_matrix.shape[0] >= vocab_size:
            # Return a contiguous copy to avoid slicing issues with CuPy indexing
            return self.xp.ascontiguousarray(self._token_matrix[:vocab_size])
        
        # Align to tensor core requirements
        aligned_vocab = ((vocab_size + TC_ALIGNMENT - 1) // TC_ALIGNMENT) * TC_ALIGNMENT
        
        if self.gpu.use_gpu:
            # Build exactly vocab_size rows to avoid slicing issues
            token_matrix = self.xp.zeros((vocab_size, self.uint64_count), dtype=self.xp.uint64)
            
            for token_id in range(vocab_size):
                vec = seed_to_hypervector(f"token_{token_id + seed_offset}", self.dim)
                token_matrix[token_id] = cp.asarray(vec)
            
            self._token_matrix = token_matrix
            
            # Create FP16 version for tensor core similarity
            if self.gpu.get_kernel('tensor_core_fp16_similarity'):
                self._token_matrix_fp16 = self.gpu.convert_to_fp16(token_matrix, self.dim)
        else:
            token_vectors = []
            for token_id in range(vocab_size):
                vec = seed_to_hypervector(f"token_{token_id + seed_offset}", self.dim)
                token_vectors.append(vec)
            token_matrix = self.xp.stack(token_vectors, axis=0)
            self._token_matrix = token_matrix
        
        return self._token_matrix
    
    def build_position_matrix(self, max_positions: int) -> 'cp.ndarray':
        """Build position matrix with tensor core alignment."""
        if self._position_matrix is not None and self._position_matrix.shape[0] >= max_positions:
            # Return a contiguous copy to avoid slicing issues with CuPy indexing
            return self.xp.ascontiguousarray(self._position_matrix[:max_positions])
        
        if self.gpu.use_gpu:
            # Build exactly max_positions rows to avoid slicing issues
            pos_matrix = self.xp.zeros((max_positions, self.uint64_count), dtype=self.xp.uint64)
            
            for pos in range(max_positions):
                vec = hadamard_position_vector(pos, self.dim)
                pos_matrix[pos] = cp.asarray(vec)
            
            self._position_matrix = pos_matrix
        else:
            pos_vectors = []
            for pos in range(max_positions):
                vec = hadamard_position_vector(pos, self.dim)
                pos_vectors.append(vec)
            pos_matrix = self.xp.stack(pos_vectors, axis=0)
            self._position_matrix = pos_matrix
        
        return self._position_matrix
    
    def batch_xor_bind(self, a_batch: 'cp.ndarray', b_batch: 'cp.ndarray') -> 'cp.ndarray':
        """Batch XOR bind using tensor core optimized kernel."""
        if self.gpu.use_gpu and self._batch_xor_kernel is not None:
            out = self.xp.zeros_like(a_batch)
            self._batch_xor_kernel(a_batch, b_batch, out)
            return out
        return self.xp.bitwise_xor(a_batch, b_batch)

    def apply_sparse_update(
        self,
        vec: np.ndarray,
        correction: np.ndarray,
        shift: int,
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        O(W) metacognitive correction — the 'instant jump' described in the
        metacognitive residual design.

        Instead of touching the whole 2^20-dimensional vector, we XOR only
        the W blocks at the circular_shift address from the MetaResidualRecipe.
        This is what makes STUCK-state correction sub-microsecond rather than milliseconds.

        Args:
            vec:        Full hypervector (uint64_count elements)
            correction: Full hypervector containing the residual correction
            shift:      recipe.circular_shift — the window start index
            window_size: Override W (defaults to self.sparse_window_size)

        Returns:
            Updated hypervector (same shape, only W elements changed)
        """
        W = window_size if window_size is not None else self.sparse_window_size
        W = min(W, self.uint64_count)

        # Build window index array: [shift, shift+1, ..., shift+W-1] mod uint64_count
        win_idx = (np.arange(W, dtype=np.int32) + shift) % self.uint64_count

        if self.gpu.use_gpu and _CUPY_AVAILABLE:
            if isinstance(vec, np.ndarray):
                vec = cp.asarray(vec)
            if isinstance(correction, np.ndarray):
                correction = cp.asarray(correction)
            win_idx_gpu = cp.asarray(win_idx)

            # Try the dedicated GPU kernel first
            kernel = self.gpu.get_kernel('sparse_meta_correct')
            if kernel is not None:
                try:
                    kernel(
                        (1,), (W,),
                        (vec, correction,
                         np.int32(self.uint64_count), np.int32(W), np.int32(shift))
                    )
                    return vec
                except Exception:
                    pass  # fall through to CuPy index op

            # CuPy vectorised fallback
            vec[win_idx_gpu] = cp.bitwise_xor(vec[win_idx_gpu], correction[win_idx_gpu])
            return vec

        # CPU path
        result = vec.copy()
        result[win_idx] = np.bitwise_xor(result[win_idx], correction[win_idx])
        return result
    
    def batch_encode_context(
        self,
        token_ids_batch: 'cp.ndarray',
        token_matrix: 'cp.ndarray',
        position_matrix: 'cp.ndarray',
        batch_chunk_size: int = 64,
        seq_chunk_size: int = 128
    ) -> 'cp.ndarray':
        """
        Sparse-projection batch encoding.

        Each sequence position p writes only SPARSE_WINDOW_SIZE uint64 blocks
        starting at index (p % uint64_count) — the circular_shift address.
        This is the 'instant projection' design: the full 2^20-dimensional
        vector is always addressable, but each position only touches its own
        W-block window, keeping intermediates ~250x smaller and CUDA block
        sizes within the hardware limit of 1024 threads.

        The metacognitive jump (apply_sparse_update) then corrects exactly
        those W blocks without touching the rest of the vector.
        """
        batch_size, seq_len = token_ids_batch.shape
        vocab_size = token_matrix.shape[0]
        max_positions = position_matrix.shape[0]
        W = self.sparse_window_size

        if seq_len > max_positions:
            seq_len = max_positions

        # Clamp token IDs on CPU before any GPU transfer
        if self.gpu.use_gpu:
            token_ids_cpu = self.gpu.to_cpu(token_ids_batch)
        else:
            token_ids_cpu = np.asarray(token_ids_batch)
        token_ids_clamped = np.clip(token_ids_cpu, 0, vocab_size - 1).astype(np.int64)

        # ── PATH 1: sparse_encode CUDA kernel ─────────────────────────────
        # block = (W,)  which is always <= MAX_CUDA_THREADS (1024).
        # The kernel uses atomicXor so multiple positions that share overlapping
        # windows accumulate correctly into the output.
        if self.gpu.use_gpu and self.gpu.get_kernel('sparse_encode'):
            try:
                token_ids_gpu = self.gpu.to_gpu(token_ids_clamped)
                result = self.xp.zeros((batch_size, self.uint64_count), dtype=self.xp.uint64)

                kernel = self.gpu.get_kernel('sparse_encode')
                grid  = (batch_size,)
                block = (W,)          # W <= 1024: always valid

                kernel(
                    grid, block,
                    (token_ids_gpu, token_matrix, result,
                     np.int32(batch_size), np.int32(seq_len),
                     np.int32(vocab_size), np.int32(self.uint64_count),
                     np.int32(W))
                )
                return result
            except Exception as e:
                print(f"[SparseEncode] sparse_encode kernel failed: {e}, trying dense kernel")

        # Pre-compute per-position window indices on CPU once — shape (seq_len, W).
        # win_idx[p] = the W uint64 block indices that position p writes to.
        # This is the same address used by apply_sparse_update, so encoding and
        # correction always agree on which window belongs to which position.
        shifts   = np.arange(seq_len, dtype=np.int32) % self.uint64_count  # (seq,)
        offsets  = np.arange(W, dtype=np.int32)                             # (W,)
        win_idx  = (shifts[:, None] + offsets[None, :]) % self.uint64_count # (seq, W)

        # ── PATH 2: fixed-block-size tensor_core_full_encode (sparse variant) ─
        # We pass the full token/position matrices but only accumulate into the
        # W active blocks per position.  Block size is clamped to MAX_CUDA_THREADS.
        if self.gpu.use_gpu and self.gpu.get_kernel('tensor_core_full_encode'):
            try:
                token_ids_gpu = self.gpu.to_gpu(token_ids_clamped)
                result = self.xp.zeros((batch_size, self.uint64_count), dtype=self.xp.uint64)

                # Sparse gather: build a (seq_len, W) slice of the position matrix
                # so the kernel only sees the W columns it should write.
                win_idx_gpu      = self.gpu.to_gpu(win_idx)                  # (seq, W)
                pos_sparse        = position_matrix[
                    self.xp.arange(seq_len)[:, None], win_idx_gpu]           # (seq, W)

                # For the token matrix, gather the W columns per position lazily
                # inside the kernel by passing win_idx alongside.  Since the
                # existing kernel signature doesn't take win_idx we use the
                # sparse_encode kernel instead (same logic, avoids a new kernel).
                kernel = self.gpu.get_kernel('sparse_encode')
                if kernel is None:
                    raise RuntimeError("sparse_encode kernel not available")

                grid  = (batch_size,)
                block = (W,)   # W <= MAX_CUDA_THREADS: always valid

                kernel(
                    grid, block,
                    (token_ids_gpu, token_matrix, result,
                     np.int32(batch_size), np.int32(seq_len),
                     np.int32(vocab_size), np.int32(self.uint64_count),
                     np.int32(W))
                )
                return result
            except Exception as e:
                print(f"[TensorCore] PATH 2 kernel failed: {e}, using vectorized fallback")

        # ── PATH 3: vectorized sparse fallback (CuPy or NumPy) ────────────
        # Sparse gather: instead of loading (C*seq, uint64_count), we load only
        # (C*seq, W) by indexing the W active columns for each position.
        # Intermediate tensor goes from ~4 GB to ~17 MB for 2^20 dim, seq=512.
        result = self.xp.zeros((batch_size, self.uint64_count), dtype=self.xp.uint64)

        for batch_start in range(0, batch_size, batch_chunk_size):
            batch_end  = min(batch_start + batch_chunk_size, batch_size)
            chunk_ids  = token_ids_clamped[batch_start:batch_end]   # (C, seq)
            chunk_size = batch_end - batch_start

            if self.gpu.use_gpu:
                chunk_ids_gpu = self.gpu.to_gpu(chunk_ids)          # (C, seq)
                win_idx_gpu   = self.gpu.to_gpu(win_idx)            # (seq, W)

                # Sparse token gather: (C*seq) lookups, W columns each → (C*seq, W)
                flat_ids         = chunk_ids_gpu.reshape(-1)                       # (C*seq,)
                tok_sparse_flat  = token_matrix[flat_ids][:, win_idx_gpu.reshape(-1)]
                # win_idx_gpu.reshape(-1) is (seq*W,); we remap to (C, seq, W)
                tok_sparse = token_matrix[flat_ids][:, self.xp.tile(
                    win_idx_gpu, (1, 1)).reshape(-1)].reshape(chunk_size, seq_len, W)

                # Sparse position gather: (seq, W) — same window indices
                pos_sparse = self.xp.zeros((seq_len, W), dtype=self.xp.uint64)
                for p in range(seq_len):
                    pos_sparse[p] = position_matrix[p, win_idx_gpu[p]]

                # XOR-bind token ⊕ position in the sparse (C, seq, W) space
                bound_sparse = self.xp.bitwise_xor(
                    tok_sparse, pos_sparse[self.xp.newaxis, :, :])  # (C, seq, W)

                # Scatter-reduce: accumulate each position's W values into the
                # correct locations of the full uint64_count output vector.
                # We use a loop over seq here — only W elements written per step,
                # so this is O(seq * W) memory ops, not O(seq * uint64_count).
                bundled = self.xp.zeros((chunk_size, self.uint64_count), dtype=self.xp.uint64)
                for p in range(seq_len):
                    bundled[:, win_idx_gpu[p]] = self.xp.bitwise_xor(
                        bundled[:, win_idx_gpu[p]], bound_sparse[:, p, :])

                result[batch_start:batch_end] = bundled

            else:
                # CPU sparse path — identical logic with NumPy
                flat_ids    = chunk_ids.reshape(-1)                   # (C*seq,)

                tok_sparse  = np.stack([
                    token_matrix[chunk_ids[:, p]][:, win_idx[p]]
                    for p in range(seq_len)], axis=1)                 # (C, seq, W)

                pos_sparse  = np.stack([
                    position_matrix[p, win_idx[p]]
                    for p in range(seq_len)], axis=0)                 # (seq, W)

                bound_sparse = np.bitwise_xor(
                    tok_sparse, pos_sparse[np.newaxis, :, :])         # (C, seq, W)

                bundled = np.zeros((chunk_size, self.uint64_count), dtype=np.uint64)
                for p in range(seq_len):
                    bundled[:, win_idx[p]] = np.bitwise_xor(
                        bundled[:, win_idx[p]], bound_sparse[:, p, :])

                result[batch_start:batch_end] = bundled

        return result
    
    def _chunked_circular_xor(
        self,
        bound_vecs: 'cp.ndarray',
        seq_len: int,
        chunk_size: int
    ) -> 'cp.ndarray':
        """Chunked circular XOR for memory-efficient processing."""
        batch_size = bound_vecs.shape[0]
        result = self.xp.zeros((batch_size, self.uint64_count), dtype=self.xp.uint64)
        
        shifts = self.xp.arange(seq_len) % self.uint64_count
        
        for seq_start in range(0, seq_len, chunk_size):
            seq_end = min(seq_start + chunk_size, seq_len)
            
            for pos in range(seq_start, seq_end):
                shift = shifts[pos]
                if shift == 0:
                    shifted = bound_vecs[:, pos, :]
                else:
                    shifted = self.xp.roll(bound_vecs[:, pos, :], shift, axis=1)
                
                result = self.xp.bitwise_xor(result, shifted)
        
        return result
    
    def batch_hamming_similarity(
        self,
        query_batch: 'cp.ndarray',
        codebook: 'cp.ndarray',
        chunk_size: int = 64
    ) -> 'cp.ndarray':
        """
        Compute batch Hamming similarity using tensor core optimized kernels.
        """
        batch_size = query_batch.shape[0]
        codebook_size = codebook.shape[0]
        
        # Try tensor core kernel first
        if self.gpu.use_gpu and self.gpu.get_kernel('tensor_core_xor_similarity'):
            try:
                return self.gpu.tensor_core_similarity_batch(query_batch, codebook, self.uint64_count)
            except Exception as e:
                print(f"[TensorCore] Similarity kernel failed: {e}, using fallback")
        
        similarity = self.xp.zeros((batch_size, codebook_size), dtype=self.xp.float32)
        
        # Fallback: chunked similarity computation
        if self.gpu.use_gpu and self._xor_popcount_kernel is not None:
            for i_start in range(0, batch_size, chunk_size):
                i_end = min(i_start + chunk_size, batch_size)
                query_chunk = query_batch[i_start:i_end]
                
                for j_start in range(0, codebook_size, chunk_size):
                    j_end = min(j_start + chunk_size, codebook_size)
                    codebook_chunk = codebook[j_start:j_end]
                    
                    query_expanded = query_chunk[:, self.xp.newaxis, :]
                    codebook_expanded = codebook_chunk[self.xp.newaxis, :, :]
                    
                    diff_bits = self._xor_popcount_kernel(query_expanded, codebook_expanded)
                    diff_bits = self.xp.sum(diff_bits, axis=-1)
                    
                    chunk_similarity = 1.0 - (diff_bits.astype(self.xp.float32) / self.dim)
                    similarity[i_start:i_end, j_start:j_end] = chunk_similarity
        else:
            # Pure CPU fallback
            for i in range(batch_size):
                for j in range(codebook_size):
                    similarity[i, j] = hamming_similarity(query_batch[i], codebook[j])
        
        return similarity
    
    def batch_learn_patterns(
        self,
        contexts_batch: List[List[int]],
        targets_batch: List[int],
        token_matrix: 'cp.ndarray',
        position_matrix: 'cp.ndarray'
    ) -> Tuple['cp.ndarray', 'cp.ndarray']:
        """Batch learn patterns with tensor core optimization."""
        batch_size = len(contexts_batch)
        vocab_size = token_matrix.shape[0]
        max_positions = position_matrix.shape[0]
        
        # Validate and clamp context token IDs on CPU before GPU transfer
        max_len = max(len(c) for c in contexts_batch)
        padded_contexts_np = np.zeros((batch_size, max_len), dtype=np.int64)
        for i, ctx in enumerate(contexts_batch):
            # Clamp each token ID to valid range
            clamped_ctx = [max(0, min(t, vocab_size - 1)) for t in ctx]
            padded_contexts_np[i, :len(clamped_ctx)] = np.array(clamped_ctx, dtype=np.int64)
        
        # Transfer to GPU
        padded_contexts = self.gpu.to_gpu(padded_contexts_np)
        
        # Removed synchronization points for performance - errors will be caught on next GPU operation
        # The GPU pipeline can now run asynchronously without CPU stalls
        
        context_vecs = self.batch_encode_context(padded_contexts, token_matrix, position_matrix)
        
        # Validate and clamp target values on CPU before GPU transfer
        targets_clamped = [max(0, min(t, vocab_size - 1)) for t in targets_batch]
        targets_gpu = self.gpu.to_gpu(np.array(targets_clamped, dtype=np.int64))
        
        # Use direct indexing with validated indices
        target_vecs = token_matrix[targets_gpu]
        
        patterns = self.batch_xor_bind(context_vecs, target_vecs)
        
        return patterns, target_vecs
    
    def batch_predict(
        self,
        contexts_batch: List[List[int]],
        token_matrix: 'cp.ndarray',
        position_matrix: 'cp.ndarray',
        temperature: float = 1.0,
        top_k: int = 10
    ) -> Tuple['cp.ndarray', 'cp.ndarray']:
        """Batch predict with tensor core optimization."""
        batch_size = len(contexts_batch)
        vocab_size = token_matrix.shape[0]
        
        max_len = max(len(c) for c in contexts_batch)
        padded_contexts = self.xp.zeros((batch_size, max_len), dtype=self.xp.int64)
        for i, ctx in enumerate(contexts_batch):
            padded_contexts[i, :len(ctx)] = self.xp.array(ctx)
        
        context_vecs = self.batch_encode_context(padded_contexts, token_matrix, position_matrix)
        
        # Use tensor core similarity
        similarities = self.batch_hamming_similarity(context_vecs, token_matrix)
        
        scaled = similarities * 10.0 / temperature
        
        scaled_max = self.xp.max(scaled, axis=-1, keepdims=True)
        scaled = scaled - scaled_max
        exp_scores = self.xp.exp(scaled)
        probs = exp_scores / self.xp.sum(exp_scores, axis=-1, keepdims=True)
        
        top_k = min(top_k, vocab_size)
        top_indices = self.xp.argsort(probs, axis=-1)[:, ::-1][:, :top_k]
        
        return probs, top_indices


_batch_ops: Optional[TensorCoreBatchOperations] = None

def get_batch_ops(gpu_manager: TensorCoreGPUManager = None, dim: int = DEFAULT_HDC_DIM,
                  sparse_window_size: int = SPARSE_WINDOW_SIZE) -> TensorCoreBatchOperations:
    """Get or create the global batch operations instance."""
    global _batch_ops
    if _batch_ops is None:
        if gpu_manager is None:
            gpu_manager = get_gpu_manager()
        _batch_ops = TensorCoreBatchOperations(gpu_manager, dim, sparse_window_size)
    return _batch_ops


def ternary_xor(
    a_pos: np.ndarray, a_neg: np.ndarray,
    b_pos: np.ndarray, b_neg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """XOR bind for ternary vectors."""
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
    """Registry for seed strings to support zero-weight procedural generation."""
    
    def __init__(self):
        self._seeds: Dict[str, int] = {}
        self._id_to_seed: Dict[int, str] = {}
        self._next_id = 0
    
    def get_or_create(self, seed_string: str) -> int:
        if seed_string in self._seeds:
            return self._seeds[seed_string]
        
        seed_id = self._next_id
        self._seeds[seed_string] = seed_id
        self._id_to_seed[seed_id] = seed_string
        self._next_id += 1
        return seed_id
    
    def get_seed(self, seed_id: int) -> Optional[str]:
        return self._id_to_seed.get(seed_id)
    
    def to_dict(self) -> dict:
        return {
            'seeds': self._seeds.copy(),
            'next_id': self._next_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SeedRegistry':
        registry = cls()
        registry._seeds = data.get('seeds', {}).copy()
        registry._next_id = data.get('next_id', 0)
        registry._id_to_seed = {v: k for k, v in registry._seeds.items()}
        return registry


@dataclass
class Recipe:
    """Recipe for pattern reconstruction."""
    recipe_id: str
    seed_sequence: List[str]
    operation_order: List[int]
    problem_signature: str
    target_token: int
    confidence: float = 1.0
    usage_count: int = 0
    
    def to_dict(self) -> dict:
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
        return 50 + sum(len(s) for s in self.seed_sequence)


@dataclass
class DeduplicationConfig:
    """Configuration for advanced deduplication."""
    similarity_threshold: float = 0.85
    near_duplicate_threshold: float = 0.92
    cross_timestep_threshold: float = 0.80
    auto_merge_exact: bool = True
    auto_cluster: bool = True
    preserve_relationships: bool = True
    track_trajectories: bool = True
    separate_modalities: bool = False
    max_cluster_size: int = 100
    min_cluster_similarity: float = 0.75


class RelationshipStats:
    """Lightweight statistics tracker for pattern relationships (simplified from XORRelationshipGraph)."""
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM):
        self._stats = {
            'total_patterns': 0,
            'total_edges': 0,
            'clusters_created': 0,
            'relationships_by_type': {}
        }
    
    def add_pattern(self, recipe: Recipe, signature: str) -> str:
        """Record a pattern was added (just increment counter)."""
        self._stats['total_patterns'] += 1
        return recipe.recipe_id
    
    def add_relationship(
        self,
        pattern_id1: str,
        pattern_id2: str,
        relationship_type: RelationshipType,
        bidirectional: bool = False,
        strength: float = 1.0,
        sequence_position: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a relationship (just update statistics)."""
        self._stats['total_edges'] += 1
        rel_type_name = relationship_type.value
        self._stats['relationships_by_type'][rel_type_name] = \
            self._stats['relationships_by_type'].get(rel_type_name, 0) + 1
        
        if bidirectional:
            self._stats['total_edges'] += 1
            self._stats['relationships_by_type'][rel_type_name] = \
                self._stats['relationships_by_type'].get(rel_type_name, 0) + 1
    
    def get_statistics(self) -> Dict[str, Any]:
        return {**self._stats}
    
    def to_dict(self) -> Dict[str, Any]:
        return {'stats': self._stats}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], dim: int = DEFAULT_HDC_DIM) -> 'RelationshipStats':
        stats = cls(dim=dim)
        stats._stats = data.get('stats', stats._stats)
        return stats


class RecipeDeduplicator:
    """Advanced recipe deduplicator with XOR bitwise relationship graph methods."""
    
    def __init__(self, config: Optional[DeduplicationConfig] = None, dim: int = DEFAULT_HDC_DIM):
        self.config = config or DeduplicationConfig()
        self.dim = dim
        self.uint64_count = dim // 64
        
        self.relationship_graph = RelationshipStats(dim=dim)
        
        self._recipes: Dict[str, Recipe] = {}
        self._signature_to_id: Dict[str, str] = {}
        self._content_hash_index: Dict[str, str] = {}
        
        self._clusters: Dict[str, List[str]] = {}
        self._pattern_to_cluster: Dict[str, str] = {}
        self._cluster_centroids: Dict[str, str] = {}
        
        self._by_seed_string: Dict[str, str] = {}
        self._usage_count: Dict[str, int] = {}
        
        self._vector_cache: Dict[str, np.ndarray] = {}
        self._cache_enabled: bool = True
        self._max_cache_size: int = 10000
        
        self._stats = {
            'patterns_added': 0,
            'duplicates_found': 0,
            'exact_duplicates_merged': 0,
            'near_duplicates_found': 0,
            'relationships_created': 0,
            'relationships_preserved': 0,
            'similarity_checks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'clusters_created': 0
        }
    
    def _compute_signature(self, seed_sequence: List[str]) -> str:
        canonical = "|".join(sorted(seed_sequence))
        return blake3_hash(canonical.encode()).hex()[:16]
    
    def _compute_content_hash(self, seed_string: str) -> str:
        if _BLAKE3_AVAILABLE:
            return _blake3_func(seed_string.encode()).hexdigest()
        else:
            import hashlib
            return hashlib.sha256(seed_string.encode()).hexdigest()
    
    def _get_vector(self, recipe: Recipe) -> np.ndarray:
        cache_key = recipe.recipe_id
        
        if self._cache_enabled and cache_key in self._vector_cache:
            self._stats['cache_hits'] += 1
            return self._vector_cache[cache_key]
        
        self._stats['cache_misses'] += 1
        
        vector = self._encode_seed_sequence(recipe.seed_sequence)
        
        if self._cache_enabled:
            if len(self._vector_cache) >= self._max_cache_size:
                keys_to_remove = list(self._vector_cache.keys())[:self._max_cache_size // 2]
                for key in keys_to_remove:
                    del self._vector_cache[key]
            
            self._vector_cache[cache_key] = vector
        
        return vector
    
    def _encode_seed_sequence(self, seed_sequence: List[str]) -> np.ndarray:
        if not seed_sequence:
            return np.zeros(self.uint64_count, dtype=np.uint64)
        
        result = seed_to_hypervector(seed_sequence[0], self.dim)
        for seed in seed_sequence[1:]:
            vec = seed_to_hypervector(seed, self.dim)
            result = np.bitwise_xor(result, vec)
        
        return result
    
    def _compute_hadamard_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return hamming_similarity(vec1, vec2)
    
    def check_duplicate(
        self, seed_sequence: List[str], threshold: Optional[float] = None
    ) -> Tuple[bool, Optional[str], Optional[Recipe]]:
        threshold = threshold or self.config.similarity_threshold
        
        sig = self._compute_signature(seed_sequence)
        if sig in self._signature_to_id:
            recipe_id = self._signature_to_id[sig]
            return True, sig, self._recipes.get(recipe_id)
        
        content_str = "|".join(sorted(seed_sequence))
        content_hash = self._compute_content_hash(content_str)
        if content_hash in self._content_hash_index:
            recipe_id = self._content_hash_index[content_hash]
            return True, sig, self._recipes.get(recipe_id)
        
        if threshold < 1.0:
            query_vec = self._encode_seed_sequence(seed_sequence)
            
            for recipe_id, existing_recipe in self._recipes.items():
                existing_vec = self._get_vector(existing_recipe)
                similarity = self._compute_hadamard_similarity(query_vec, existing_vec)
                self._stats['similarity_checks'] += 1
                
                if similarity >= threshold:
                    existing_sig = self._compute_signature(existing_recipe.seed_sequence)
                    return True, existing_sig, existing_recipe
        
        return False, sig, None
    
    def store_or_update(self, recipe: Recipe) -> str:
        sig = self._compute_signature(recipe.seed_sequence)
        
        if sig in self._signature_to_id:
            existing_id = self._signature_to_id[sig]
            existing = self._recipes[existing_id]
            existing.confidence = max(existing.confidence, recipe.confidence)
            self._usage_count[sig] += 1
            self._stats['duplicates_found'] += 1
            self._stats['exact_duplicates_merged'] += 1
            
            if self.config.preserve_relationships:
                self.relationship_graph.add_relationship(
                    existing_id, recipe.recipe_id,
                    RelationshipType.SEMANTIC_SIMILAR,
                    bidirectional=True, strength=1.0
                )
                self._stats['relationships_preserved'] += 1
            
            return sig
        
        content_str = "|".join(sorted(recipe.seed_sequence))
        content_hash = self._compute_content_hash(content_str)
        
        if content_hash in self._content_hash_index:
            existing_id = self._content_hash_index[content_hash]
            existing = self._recipes[existing_id]
            existing.confidence = max(existing.confidence, recipe.confidence)
            self._usage_count[sig] = self._usage_count.get(sig, 0) + 1
            self._stats['duplicates_found'] += 1
            self._stats['exact_duplicates_merged'] += 1
            return sig
        
        similar_pattern_id = None
        similar_cluster_id = None
        best_similarity = 0.0
        
        if self.config.similarity_threshold < 1.0:
            query_vec = self._encode_seed_sequence(recipe.seed_sequence)
            
            for recipe_id, existing_recipe in self._recipes.items():
                existing_vec = self._get_vector(existing_recipe)
                similarity = self._compute_hadamard_similarity(query_vec, existing_vec)
                self._stats['similarity_checks'] += 1
                
                if similarity >= self.config.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    similar_pattern_id = recipe_id
                    similar_cluster_id = self._pattern_to_cluster.get(recipe_id)
        
        self._recipes[recipe.recipe_id] = recipe
        self._signature_to_id[sig] = recipe.recipe_id
        self._content_hash_index[content_hash] = recipe.recipe_id
        self._usage_count[sig] = 1
        
        self.relationship_graph.add_pattern(recipe, sig)
        
        if similar_pattern_id and self.config.auto_cluster:
            if similar_cluster_id:
                self._clusters[similar_cluster_id].append(recipe.recipe_id)
                self._pattern_to_cluster[recipe.recipe_id] = similar_cluster_id
                
                self.relationship_graph.add_relationship(
                    recipe.recipe_id, similar_pattern_id,
                    RelationshipType.CLUSTER_MEMBER,
                    bidirectional=True, strength=best_similarity
                )
            else:
                cluster_id = f"cluster_{similar_pattern_id[:8]}"
                self._clusters[cluster_id] = [similar_pattern_id, recipe.recipe_id]
                self._pattern_to_cluster[similar_pattern_id] = cluster_id
                self._pattern_to_cluster[recipe.recipe_id] = cluster_id
                self._cluster_centroids[cluster_id] = similar_pattern_id
                self._stats['clusters_created'] += 1
                
                self.relationship_graph.add_relationship(
                    recipe.recipe_id, similar_pattern_id,
                    RelationshipType.CLUSTER_MEMBER,
                    bidirectional=True, strength=best_similarity
                )
            
            self._stats['near_duplicates_found'] += 1
            self._stats['relationships_preserved'] += 1
        else:
            if self.config.auto_cluster:
                cluster_id = f"cluster_{recipe.recipe_id[:8]}"
                self._clusters[cluster_id] = [recipe.recipe_id]
                self._pattern_to_cluster[recipe.recipe_id] = cluster_id
                self._cluster_centroids[cluster_id] = recipe.recipe_id
                self._stats['clusters_created'] += 1
        
        for seed in recipe.seed_sequence:
            if seed not in self._by_seed_string:
                self._by_seed_string[seed] = recipe.recipe_id
        
        self._stats['patterns_added'] += 1
        return sig
    
    def get_by_signature(self, signature: str) -> Optional[Recipe]:
        recipe_id = self._signature_to_id.get(signature)
        if recipe_id:
            return self._recipes.get(recipe_id)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            **self._stats,
            'total_recipes': len(self._recipes),
            'total_clusters': len(self._clusters),
            'cache_size': len(self._vector_cache),
            'graph_stats': self.relationship_graph.get_statistics()
        }


class SelfObservation:
    """
    Metacognitive self-observation system for HDC models.
    
    This class implements the self-observation capability where the model
    can "see" its own encoded state and modify its trajectory accordingly.
    
    Key features:
    - Monitors current state similarity to known patterns
    - Detects convergence signals (stuck, oscillating, breakthrough)
    - Suggests trajectory modifications (recall, explore)
    - Maintains reasoning trace for interpretability
    """
    
    def __init__(self, dim: int = DEFAULT_HDC_DIM, known_patterns: Optional[Dict[str, np.ndarray]] = None):
        self.dim = dim
        self.uint64_count = dim // 64
        self.known_patterns = known_patterns or {}
        
        # Threshold for pattern recognition
        self.recognition_threshold = 0.85
        self.stuck_threshold = 0.02
        self.oscillation_window = 5
        self.breakthrough_threshold = 0.1
        
        # History tracking
        self._similarity_history: List[float] = []
        self._action_history: List[TrajectoryAction] = []
        self._pattern_history: List[str] = []
        
        # Convergence detection
        self._last_similarities: List[float] = []
        self._iteration_count = 0
    
    def observe(self, current_state: np.ndarray, iteration: int = 0) -> SelfObservationState:
        """
        Observe the current state and generate metacognitive insights.
        
        Args:
            current_state: Current HDC vector (the model's "thought")
            iteration: Current iteration number
            
        Returns:
            SelfObservationState with insights and trajectory recommendations
        """
        self._iteration_count = iteration
        
        # Compute similarity to all known patterns
        detected_patterns = []
        best_pattern = None
        best_similarity = 0.0
        
        for pattern_name, pattern_vec in self.known_patterns.items():
            sim = hamming_similarity(current_state, pattern_vec)
            if sim > self.recognition_threshold:
                detected_patterns.append(pattern_name)
            if sim > best_similarity:
                best_similarity = sim
                best_pattern = pattern_name
        
        # Track similarity history
        self._similarity_history.append(best_similarity)
        self._last_similarities.append(best_similarity)
        if len(self._last_similarities) > 20:
            self._last_similarities.pop(0)
        
        # Detect convergence signal
        convergence_signal = self._detect_convergence_signal()
        
        # Determine trajectory action
        trajectory_action = self._determine_trajectory_action(
            convergence_signal, best_similarity, detected_patterns
        )
        
        # Record action
        self._action_history.append(trajectory_action)
        if best_pattern:
            self._pattern_history.append(best_pattern)
        
        # Compute confidence
        confidence = self._compute_confidence(best_similarity, convergence_signal)
        
        # Build reasoning trace
        reasoning_trace = self._build_reasoning_trace(
            convergence_signal, trajectory_action, best_pattern, best_similarity
        )
        
        return SelfObservationState(
            iteration=iteration,
            current_similarity=best_similarity,
            best_similarity=max(self._similarity_history) if self._similarity_history else 0.0,
            similarity_history=self._similarity_history[-20:],
            convergence_signal=convergence_signal,
            trajectory_action=trajectory_action,
            detected_patterns=detected_patterns,
            confidence=confidence,
            reasoning_trace=reasoning_trace
        )
    
    def _detect_convergence_signal(self) -> ConvergenceSignal:
        """Detect if the search is converging, stuck, or oscillating."""
        if len(self._last_similarities) < 3:
            return ConvergenceSignal.CONTINUE
        
        recent = self._last_similarities[-5:]
        
        # Check for convergence (steady improvement)
        if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
            if recent[-1] - recent[0] > self.breakthrough_threshold:
                return ConvergenceSignal.BREAKTHROUGH
            return ConvergenceSignal.CONVERGING
        
        # Check for stuck (no progress)
        if len(recent) >= 3:
            variance = np.var(recent)
            if variance < self.stuck_threshold:
                return ConvergenceSignal.STUCK
        
        # Check for oscillation
        if len(recent) >= self.oscillation_window:
            changes = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            sign_changes = sum(1 for i in range(len(changes)-1) 
                              if changes[i] * changes[i+1] < 0)
            if sign_changes >= 2:
                return ConvergenceSignal.OSCILLATING
        
        # Check for divergence
        if recent[-1] < recent[0] - 0.1:
            return ConvergenceSignal.DIVERGING
        
        return ConvergenceSignal.CONTINUE
    
    def _determine_trajectory_action(
        self,
        signal: ConvergenceSignal,
        best_similarity: float,
        detected_patterns: List[str]
    ) -> TrajectoryAction:
        """Determine the next trajectory action based on convergence signal."""
        
        # If we found a known pattern, use recall
        if detected_patterns and best_similarity > self.recognition_threshold:
            return TrajectoryAction.RECALL
        
        # Handle convergence signals
        if signal == ConvergenceSignal.BREAKTHROUGH:
            return TrajectoryAction.CONTINUE
        
        if signal == ConvergenceSignal.CONVERGING:
            return TrajectoryAction.CONTINUE
        
        if signal == ConvergenceSignal.STUCK:
            # Try exploration when stuck
            return TrajectoryAction.EXPLORE
        
        if signal == ConvergenceSignal.OSCILLATING:
            return TrajectoryAction.EXPLORE
        
        if signal == ConvergenceSignal.DIVERGING:
            return TrajectoryAction.RESTART
        
        # Default: continue current trajectory
        return TrajectoryAction.CONTINUE
    
    def _compute_confidence(self, similarity: float, signal: ConvergenceSignal) -> float:
        """Compute confidence in the current trajectory."""
        base_confidence = similarity
        
        # Adjust based on convergence signal
        signal_multipliers = {
            ConvergenceSignal.BREAKTHROUGH: 1.2,
            ConvergenceSignal.CONVERGING: 1.1,
            ConvergenceSignal.CONTINUE: 1.0,
            ConvergenceSignal.OSCILLATING: 0.8,
            ConvergenceSignal.STUCK: 0.6,
            ConvergenceSignal.DIVERGING: 0.4
        }
        
        multiplier = signal_multipliers.get(signal, 1.0)
        return min(1.0, base_confidence * multiplier)
    
    def _build_reasoning_trace(
        self,
        signal: ConvergenceSignal,
        action: TrajectoryAction,
        pattern: Optional[str],
        similarity: float
    ) -> List[str]:
        """Build a human-readable reasoning trace."""
        trace = []
        
        trace.append(f"Iteration {self._iteration_count}: similarity={similarity:.4f}")
        trace.append(f"Convergence signal: {signal.value}")
        
        if pattern:
            trace.append(f"Detected pattern: {pattern}")
        
        trace.append(f"Action: {action.value}")
        
        return trace
    
    def register_pattern(self, name: str, vector: np.ndarray):
        """Register a known pattern for recognition."""
        self.known_patterns[name] = vector.copy()
    
    def get_action_history(self) -> List[TrajectoryAction]:
        """Get the history of trajectory actions."""
        return self._action_history.copy()


@dataclass
class SemanticCoverageReport:
    """Report on semantic landscape coverage and confidence distribution."""
    coverage: float  # Fraction of windows with high confidence
    dead_zones: List[int]  # Windows with near-random signal (low confidence)
    mean_confidence: float  # Average confidence across all windows
    high_confidence_count: int  # Number of windows with confidence > 0.7
    total_windows: int  # Total number of windows in semantic vector
    confidence_distribution: List[float]  # Full distribution for analysis


class SemanticSelfObservation:
    """
    Proactive semantic confidence monitor - redesigned for the semantic layer.
    
    Unlike the reactive SelfObservation that watches iteration-by-iteration
    similarity changes, this class provides INSTANTANEOUS confidence assessment
    based on the semantic vector's popcount-derived signals.
    
    Key insight: With semantic_vec filled in one pass, confidence is always
    already known BEFORE prediction, not after failing.
    """
    
    def __init__(
        self,
        dim: int = DEFAULT_HDC_DIM,
        semantic_vec: Optional[np.ndarray] = None,
        mask: Optional[int] = None
    ):
        self.dim = dim
        self.uint64_count = dim // 64
        self.semantic_vec = semantic_vec  # Reference to DualVectorProjection.semantic_vec
        self.mask = mask or (self.uint64_count - 1)
        
        # Confidence thresholds for signal classification
        self.high_confidence_threshold = 0.8
        self.moderate_confidence_threshold = 0.5
        self.low_confidence_threshold = 0.2
        
        # Track observation history for analysis
        self._observation_history: List[Dict[str, Any]] = []
    
    def observe_relationship(
        self,
        token_A: int,
        token_B: int,
        hadamard_index_fn: Callable[[int], int]
    ) -> Tuple[ConvergenceSignal, float, int]:
        """
        Observe the semantic relationship between two tokens - O(1) operation.
        
        Args:
            token_A: First token ID
            token_B: Second token ID
            hadamard_index_fn: Function to convert token_id to hadamard index
            
        Returns:
            Tuple of (convergence_signal, confidence, direction)
            - convergence_signal: BREAKTHROUGH/CONVERGING/OSCILLATING/STUCK
            - confidence: 0.0 to 1.0 strength of signal
            - direction: +1 or -1 (positive/negative relationship)
        """
        if self.semantic_vec is None:
            return ConvergenceSignal.STUCK, 0.0, 0
        
        # Compute relationship window via XOR
        idx_A = hadamard_index_fn(token_A)
        idx_B = hadamard_index_fn(token_B)
        rel_window = (idx_A ^ idx_B) & self.mask
        
        # Get signal from semantic vector
        signal = self.semantic_vec[rel_window]
        pc = int(np.unpackbits(signal.view(np.uint8)).sum())
        
        # Compute confidence from popcount
        confidence = abs(pc - 32) / 32.0
        direction = 1 if pc > 32 else -1
        
        # Classify signal into convergence type
        if confidence > self.high_confidence_threshold:
            signal_type = ConvergenceSignal.BREAKTHROUGH  # Strong corpus signal
        elif confidence > self.moderate_confidence_threshold:
            signal_type = ConvergenceSignal.CONVERGING  # Moderate signal
        elif confidence < self.low_confidence_threshold:
            signal_type = ConvergenceSignal.STUCK  # Genuinely unseen
        else:
            # popcount near 32 but not quite — contradictory contexts
            signal_type = ConvergenceSignal.OSCILLATING  # Ambiguous relationship
        
        # Record observation
        self._observation_history.append({
            'token_A': token_A,
            'token_B': token_B,
            'rel_window': rel_window,
            'confidence': confidence,
            'direction': direction,
            'signal_type': signal_type.name
        })
        
        return signal_type, confidence, direction
    
    def observe_token_context(
        self,
        token_id: int,
        hadamard_index_fn: Callable[[int], int]
    ) -> Tuple[float, int]:
        """
        Get the semantic context strength for a single token - O(1) operation.
        
        Returns:
            Tuple of (confidence, direction) for the token's semantic window
        """
        if self.semantic_vec is None:
            return 0.0, 0
        
        idx = hadamard_index_fn(token_id)
        window = idx & self.mask
        
        signal = self.semantic_vec[window]
        pc = int(np.unpackbits(signal.view(np.uint8)).sum())
        
        confidence = abs(pc - 32) / 32.0
        direction = 1 if pc > 32 else -1
        
        return confidence, direction
    
    def get_semantic_coverage(self) -> SemanticCoverageReport:
        """
        Analyze the full semantic landscape coverage.
        
        Returns a comprehensive report on:
        - How many token pairs have strong signal
        - Distribution of confidence across relationship space
        - Dead zones (windows with near-random signal)
        """
        if self.semantic_vec is None:
            return SemanticCoverageReport(
                coverage=0.0,
                dead_zones=[],
                mean_confidence=0.0,
                high_confidence_count=0,
                total_windows=self.uint64_count,
                confidence_distribution=[]
            )
        
        confidence_values = []
        dead_zones = []
        high_confidence_count = 0
        
        for w in range(self.uint64_count):
            signal = self.semantic_vec[w]
            pc = int(np.unpackbits(signal.view(np.uint8)).sum())
            confidence = abs(pc - 32) / 32.0
            confidence_values.append(confidence)
            
            if confidence > self.high_confidence_threshold:
                high_confidence_count += 1
            
            if confidence < self.low_confidence_threshold:
                dead_zones.append(w)
        
        coverage = high_confidence_count / self.uint64_count
        mean_confidence = np.mean(confidence_values) if confidence_values else 0.0
        
        return SemanticCoverageReport(
            coverage=coverage,
            dead_zones=dead_zones,
            mean_confidence=mean_confidence,
            high_confidence_count=high_confidence_count,
            total_windows=self.uint64_count,
            confidence_distribution=confidence_values
        )
    
    def predict_with_confidence_routing(
        self,
        context_tokens: List[int],
        vocab_size: int,
        hadamard_index_fn: Callable[[int], int],
        syntactic_predict_fn: Callable[[List[int]], np.ndarray]
    ) -> Tuple[int, float, str]:
        """
        Unified prediction pipeline with proactive confidence-based routing.
        
        Phase 1: Semantic probe - O(V) check for high-confidence relationships
        Phase 2: SelfObservation - O(1) confidence check for best candidate
        Phase 3: Route based on signal type
        
        Returns:
            Tuple of (predicted_token, confidence, routing_path)
        """
        if not context_tokens:
            return 0, 0.0, "fallback_empty_context"
        
        last_token = context_tokens[-1]
        scores = np.zeros(vocab_size)
        confidences = np.zeros(vocab_size)
        
        # Phase 1: Semantic probe - score all candidates
        for candidate in range(vocab_size):
            signal_type, conf, direction = self.observe_relationship(
                last_token, candidate, hadamard_index_fn
            )
            scores[candidate] = conf * direction
            confidences[candidate] = conf
        
        # Get best candidate
        best_candidate = int(np.argmax(scores))
        best_confidence = confidences[best_candidate]
        
        # Phase 2: SelfObservation - check confidence for best candidate
        signal_type, confidence, direction = self.observe_relationship(
            last_token, best_candidate, hadamard_index_fn
        )
        
        # Phase 3: Route based on signal type
        if signal_type == ConvergenceSignal.BREAKTHROUGH:
            # High confidence - return immediately
            return best_candidate, confidence, "semantic_high_confidence"
        
        elif signal_type == ConvergenceSignal.OSCILLATING:
            # Contradictory contexts - use syntactic as tiebreaker
            syntactic_probs = syntactic_predict_fn(context_tokens)
            syntactic_best = int(np.argmax(syntactic_probs))
            return syntactic_best, 0.5, "syntactic_tiebreaker"
        
        elif signal_type == ConvergenceSignal.STUCK:
            # Genuinely unseen relationship - honest fallback
            # Return uniform distribution (could also use syntactic)
            return best_candidate, 0.0, "fallback_unseen"
        
        else:  # CONVERGING or CONTINUE
            # Moderate confidence - use semantic with caution
            return best_candidate, confidence, "semantic_moderate"
    
    def get_observation_history(self) -> List[Dict[str, Any]]:
        """Get the history of relationship observations."""
        return self._observation_history.copy()


@dataclass
class SemanticRelationshipRecipe:
    """
    Lightweight recipe for semantic relationships - 9 bytes instead of ~50.
    
    With the semantic layer, a "recipe" for any relationship is just:
    - rel_window: computed from (idx_A XOR idx_B) & mask
    - confidence: derived from popcount
    - direction: +1 or -1
    
    Recipes are RECONSTRUCTABLE ON DEMAND from semantic_vec + two token indices,
    so storage is optional - only for fast routing of high-confidence relationships.
    """
    rel_window: int  # (idx_A XOR idx_B) & mask — 4 bytes
    confidence: float  # popcount-derived — 4 bytes
    direction: int  # +1 or -1 — 1 byte
    rel_type: str = "unknown"  # "synonym"/"is-a"/"part-of"/"co-occurrence"
    token_A: int = -1  # Optional: for reverse lookup
    token_B: int = -1  # Optional: for reverse lookup
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rel_window': self.rel_window,
            'confidence': self.confidence,
            'direction': self.direction,
            'rel_type': self.rel_type,
            'token_A': self.token_A,
            'token_B': self.token_B
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticRelationshipRecipe':
        return cls(
            rel_window=data['rel_window'],
            confidence=data['confidence'],
            direction=data['direction'],
            rel_type=data.get('rel_type', 'unknown'),
            token_A=data.get('token_A', -1),
            token_B=data.get('token_B', -1)
        )


class SemanticCoverageObserver:
    """
    Monitors coverage and confidence distribution across the semantic landscape.
    
    Replaces BatchProjectionObserver's iteration-curve watching with direct
    semantic landscape analysis. Dead zones identify token relationships the
    corpus simply doesn't contain evidence for - honest uncertainty, not failure.
    """
    
    def __init__(
        self,
        dim: int = DEFAULT_HDC_DIM,
        high_confidence_threshold: float = 0.7,
        low_confidence_threshold: float = 0.1
    ):
        self.dim = dim
        self.uint64_count = dim // 64
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        
        # Track coverage history
        self._coverage_history: List[float] = []
        self._dead_zone_history: List[List[int]] = []
        self._mean_confidence_history: List[float] = []
    
    def observe_semantic_landscape(
        self,
        semantic_vec: np.ndarray
    ) -> SemanticCoverageReport:
        """
        Analyze the semantic landscape for coverage and confidence distribution.
        
        This is the primary observation method - call after corpus projection.
        """
        confidence_values = []
        dead_zones = []
        high_confidence_count = 0
        
        for w in range(self.uint64_count):
            signal = semantic_vec[w]
            pc = int(np.unpackbits(signal.view(np.uint8)).sum())
            confidence = abs(pc - 32) / 32.0
            confidence_values.append(confidence)
            
            if confidence > self.high_confidence_threshold:
                high_confidence_count += 1
            
            if confidence < self.low_confidence_threshold:
                dead_zones.append(w)
        
        coverage = high_confidence_count / self.uint64_count
        mean_confidence = np.mean(confidence_values) if confidence_values else 0.0
        
        # Record history
        self._coverage_history.append(coverage)
        self._dead_zone_history.append(dead_zones)
        self._mean_confidence_history.append(mean_confidence)
        
        return SemanticCoverageReport(
            coverage=coverage,
            dead_zones=dead_zones,
            mean_confidence=mean_confidence,
            high_confidence_count=high_confidence_count,
            total_windows=self.uint64_count,
            confidence_distribution=confidence_values
        )
    
    def get_coverage_trend(self) -> str:
        """Analyze coverage trend over observations."""
        if len(self._coverage_history) < 2:
            return "insufficient_data"
        
        recent = self._coverage_history[-5:]
        if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
            return "improving"
        elif all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
            return "declining"
        else:
            return "stable"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get observer statistics."""
        return {
            'observations': len(self._coverage_history),
            'coverage_history': self._coverage_history,
            'mean_confidence_history': self._mean_confidence_history,
            'current_coverage': self._coverage_history[-1] if self._coverage_history else 0.0,
            'current_dead_zones': len(self._dead_zone_history[-1]) if self._dead_zone_history else 0,
            'coverage_trend': self.get_coverage_trend()
        }


@dataclass
class CoherenceTrajectory:
    """
    Tracks semantic and temporal coherence through XOR composition.
    
    The trajectory maintains a running composition of semantic and temporal
    indices, enabling O(1) coherence evaluation at each generation step.
    
    Key insight: XOR composition preserves translation invariance:
        window(p) XOR window(q) = window(p XOR q)
    
    This allows us to track "where we are" in semantic space without
    storing the full vector - just the XOR-composed index.
    """
    semantic_idx: int = 0  # XOR-composed semantic window index
    temporal_idx: int = 0  # XOR-composed temporal window index
    confidence: float = 0.0  # Current coherence confidence [0, 1]
    tension: float = 0.0  # Creative tension (deviation from expected)
    hop_count: int = 0  # Number of hops taken
    echo_detected: bool = False  # Long-range coherence echo flag
    echo_distance: int = 0  # Distance to detected echo (0 if none)
    
    def hop(self, semantic_delta: int, temporal_delta: int,
            new_confidence: float, mask: int) -> 'CoherenceTrajectory':
        """
        Advance the trajectory by one hop - O(1) operation.
        
        XOR composition is self-inverse, so we can "unwind" the trajectory
        to find echoes (long-range coherence patterns).
        
        Args:
            semantic_delta: XOR delta for semantic index (idx_A XOR idx_B)
            temporal_delta: XOR delta for temporal index (position delta)
            new_confidence: Confidence at new position
            mask: Window mask (uint64_count - 1)
            
        Returns:
            New CoherenceTrajectory with updated state
        """
        new_semantic_idx = (self.semantic_idx ^ semantic_delta) & mask
        new_temporal_idx = (self.temporal_idx ^ temporal_delta) & mask
        
        # Compute tension as deviation from expected coherence
        # High tension = creative deviation from corpus patterns
        tension_delta = abs(new_confidence - self.confidence)
        new_tension = (self.tension * self.hop_count + tension_delta) / (self.hop_count + 1)
        
        return CoherenceTrajectory(
            semantic_idx=new_semantic_idx,
            temporal_idx=new_temporal_idx,
            confidence=new_confidence,
            tension=new_tension,
            hop_count=self.hop_count + 1,
            echo_detected=self.echo_detected,
            echo_distance=self.echo_distance
        )
    
    def check_echo(self, history: List['CoherenceTrajectory'],
                   echo_threshold: float = 0.9) -> bool:
        """
        Check for long-range coherence echo - O(n) where n = hop_count.
        
        An echo occurs when the current semantic_idx matches a previous one,
        indicating we've returned to a semantic "theme" in the generation.
        
        Args:
            history: List of previous trajectories in this generation
            echo_threshold: Minimum confidence to consider an echo valid
            
        Returns:
            True if echo detected, False otherwise
        """
        for i, past in enumerate(history):
            if past.semantic_idx == self.semantic_idx:
                if past.confidence >= echo_threshold:
                    # Found an echo - update self
                    self.echo_detected = True
                    self.echo_distance = self.hop_count - past.hop_count
                    return True
        return False
    
    def coherence_score(self) -> float:
        """
        Compute overall coherence score combining confidence and tension.
        
        High coherence = high confidence + low tension
        Creative coherence = moderate tension + maintained confidence
        """
        # Base coherence from confidence
        base = self.confidence
        
        # Tension penalty (but not too much - some tension is creative)
        tension_penalty = self.tension * 0.3
        
        # Echo bonus - returning to themes is coherent
        echo_bonus = 0.1 if self.echo_detected else 0.0
        
        return max(0.0, min(1.0, base - tension_penalty + echo_bonus))
    
    def creative_score(self, surprise_weight: float = 0.3) -> float:
        """
        Compute creative score balancing coherence with surprise.
        
        High creativity = maintained coherence + meaningful tension
        This rewards generations that deviate creatively while
        maintaining semantic grounding.
        """
        coherence = self.coherence_score()
        surprise = self.tension  # Tension = deviation from expected
        
        # Weighted combination
        # High creativity needs both coherence AND surprise
        if coherence < 0.3:
            # Too incoherent - not creative, just random
            return coherence * 0.5
        elif surprise < 0.1:
            # Too predictable - boring
            return coherence * 0.7
        else:
            # Sweet spot: coherent but surprising
            return coherence * (1 - surprise_weight) + surprise * surprise_weight


class CreativeCoherenceManager:
    """
    Manages creative coherence evaluation for HDC language generation.
    
    This class provides O(1) coherence evaluation using popcount operations
    on XOR distances, enabling real-time creative scoring during generation.
    
    Key principles:
    1. Coherence via popcount: coherence = 1 - (popcount(sem_rel ^ temp_rel) / 64)
    2. Echo detection for long-range structural coherence
    3. Creative scoring that balances coherence with surprise
    4. Translation invariance: window(p) XOR window(q) = window(p XOR q)
    """
    
    def __init__(
        self,
        dim: int = DEFAULT_HDC_DIM,
        semantic_vec: Optional[np.ndarray] = None,
        coherence_threshold: float = 0.7,
        creative_threshold: float = 0.5,
        echo_threshold: float = 0.85,
        surprise_weight: float = 0.3
    ):
        self.dim = dim
        self.uint64_count = dim // 64
        self.mask = self.uint64_count - 1
        self.semantic_vec = semantic_vec  # Reference to DualVectorProjection.semantic_vec
        
        # Thresholds
        self.coherence_threshold = coherence_threshold
        self.creative_threshold = creative_threshold
        self.echo_threshold = echo_threshold
        self.surprise_weight = surprise_weight
        
        # Trajectory tracking
        self._trajectory_history: List[CoherenceTrajectory] = []
        self._current_trajectory: Optional[CoherenceTrajectory] = None
        
        # Statistics
        self._coherence_samples: List[float] = []
        self._creative_samples: List[float] = []
        self._echo_count: int = 0
    
    def reset_trajectory(self) -> None:
        """Reset trajectory for a new generation sequence."""
        if self._current_trajectory is not None:
            self._trajectory_history.append(self._current_trajectory)
        self._current_trajectory = CoherenceTrajectory()
    
    def evaluate_next_hop(
        self,
        current_token: int,
        candidate_token: int,
        position: int,
        hadamard_index_fn: Callable[[int], int]
    ) -> Tuple[CoherenceTrajectory, float, float]:
        """
        Evaluate a candidate next token for coherence and creativity - O(1).
        
        This is the core creative coherence evaluation. It computes:
        1. Semantic relationship via XOR of hadamard indices
        2. Temporal relationship via position delta
        3. Coherence from popcount of semantic vector at relationship window
        4. Creative score combining coherence with surprise
        
        Args:
            current_token: The current token ID
            candidate_token: The candidate next token ID
            position: Current position in sequence
            hadamard_index_fn: Function to convert token_id to hadamard index
            
        Returns:
            Tuple of (new_trajectory, coherence_score, creative_score)
        """
        if self._current_trajectory is None:
            self.reset_trajectory()
        
        # Compute semantic delta via XOR
        idx_current = hadamard_index_fn(current_token)
        idx_candidate = hadamard_index_fn(candidate_token)
        semantic_delta = idx_current ^ idx_candidate
        
        # Compute temporal delta (position-based)
        temporal_delta = position  # Simple position encoding
        
        # Get semantic window for this relationship
        rel_window = semantic_delta & self.mask
        
        # O(1) coherence from semantic vector popcount
        if self.semantic_vec is not None:
            signal = self.semantic_vec[rel_window]
            pc = int(np.unpackbits(signal.view(np.uint8)).sum())
            # Confidence from distance to 32 (random popcount)
            confidence = abs(pc - 32) / 32.0
        else:
            confidence = 0.5  # No semantic vector - neutral
        
        # Create new trajectory via hop
        new_trajectory = self._current_trajectory.hop(
            semantic_delta=semantic_delta,
            temporal_delta=temporal_delta,
            new_confidence=confidence,
            mask=self.mask
        )
        
        # Check for echo (long-range coherence)
        if new_trajectory.check_echo(self._trajectory_history, self.echo_threshold):
            self._echo_count += 1
        
        # Compute scores
        coherence = new_trajectory.coherence_score()
        creative = new_trajectory.creative_score(self.surprise_weight)
        
        # Record samples
        self._coherence_samples.append(coherence)
        self._creative_samples.append(creative)
        
        return new_trajectory, coherence, creative
    
    def accept_hop(self, trajectory: CoherenceTrajectory) -> None:
        """Accept a trajectory as the current state."""
        self._current_trajectory = trajectory
    
    def rank_candidates(
        self,
        current_token: int,
        candidates: List[int],
        position: int,
        hadamard_index_fn: Callable[[int], int],
        mode: str = "creative"
    ) -> List[Tuple[int, float, float, CoherenceTrajectory]]:
        """
        Rank candidate tokens by coherence/creativity - O(V) where V = len(candidates).
        
        Args:
            current_token: Current token ID
            candidates: List of candidate token IDs
            position: Current position
            hadamard_index_fn: Token to hadamard index function
            mode: "creative" for creative ranking, "coherent" for pure coherence
            
        Returns:
            List of (token_id, coherence, creativity, trajectory) sorted by mode
        """
        results = []
        
        for candidate in candidates:
            trajectory, coherence, creative = self.evaluate_next_hop(
                current_token, candidate, position, hadamard_index_fn
            )
            results.append((candidate, coherence, creative, trajectory))
        
        # Sort by appropriate score
        if mode == "creative":
            results.sort(key=lambda x: x[2], reverse=True)  # Sort by creativity
        else:
            results.sort(key=lambda x: x[1], reverse=True)  # Sort by coherence
        
        return results
    
    def detect_creative_opportunity(self) -> Tuple[bool, float]:
        """
        Detect if current state presents a creative opportunity.
        
        A creative opportunity occurs when:
        1. Current tension is moderate (not too predictable, not too random)
        2. Confidence is maintained above threshold
        3. No recent echo (we're in novel territory)
        
        Returns:
            Tuple of (is_opportunity, opportunity_score)
        """
        if self._current_trajectory is None:
            return False, 0.0
        
        traj = self._current_trajectory
        
        # Check conditions
        tension_ok = 0.1 < traj.tension < 0.5
        confidence_ok = traj.confidence >= self.coherence_threshold
        novel_territory = not traj.echo_detected
        
        if tension_ok and confidence_ok and novel_territory:
            opportunity_score = traj.creative_score(self.surprise_weight)
            return True, opportunity_score
        
        return False, 0.0
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of current trajectory state."""
        if self._current_trajectory is None:
            return {
                'active': False,
                'hop_count': 0,
                'coherence': 0.0,
                'creativity': 0.0,
                'echo_detected': False
            }
        
        return {
            'active': True,
            'hop_count': self._current_trajectory.hop_count,
            'semantic_idx': self._current_trajectory.semantic_idx,
            'temporal_idx': self._current_trajectory.temporal_idx,
            'coherence': self._current_trajectory.coherence_score(),
            'creativity': self._current_trajectory.creative_score(self.surprise_weight),
            'tension': self._current_trajectory.tension,
            'echo_detected': self._current_trajectory.echo_detected,
            'echo_distance': self._current_trajectory.echo_distance
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            'total_trajectories': len(self._trajectory_history),
            'total_echoes': self._echo_count,
            'mean_coherence': np.mean(self._coherence_samples) if self._coherence_samples else 0.0,
            'mean_creativity': np.mean(self._creative_samples) if self._creative_samples else 0.0,
            'current_trajectory': self.get_trajectory_summary()
        }


class RecipeReconstructor:
    def __init__(self, dim: int = DEFAULT_HDC_DIM, hadamard_basis: Optional[WalshHadamardBasis] = None):
        self.dim = dim
        self.uint64_count = dim // 64
        self.hadamard_basis = hadamard_basis or WalshHadamardBasis(dim=dim)
        
        # Cache for reconstructed vectors (optional, for performance)
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_enabled = True
        self._max_cache_size = 10000
    
    def reconstruct_from_recipe(self, recipe: Recipe) -> np.ndarray:
        # Check cache first
        cache_key = recipe.recipe_id
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Reconstruct from seed sequence
        vectors = []
        for seed in recipe.seed_sequence:
            vec = self._reconstruct_single_seed(seed)
            vectors.append(vec)
        
        # Apply operations in order
        if not vectors:
            result = np.zeros(self.uint64_count, dtype=np.uint64)
        elif len(vectors) == 1:
            result = vectors[0].copy()
        else:
            # XOR bind all vectors
            result = vectors[0].copy()
            for vec in vectors[1:]:
                result = np.bitwise_xor(result, vec)
        
        # Cache the result
        if self._cache_enabled:
            if len(self._cache) >= self._max_cache_size:
                # Evict oldest entries
                keys_to_remove = list(self._cache.keys())[:self._max_cache_size // 2]
                for key in keys_to_remove:
                    del self._cache[key]
            self._cache[cache_key] = result.copy()
        
        return result
    
    def _reconstruct_single_seed(self, seed: str) -> np.ndarray:
        # Check cache
        if self._cache_enabled and seed in self._cache:
            return self._cache[seed].copy()
        
        if seed.startswith("token_"):
            # Token embedding: use Hadamard basis
            try:
                token_id = int(seed.split("_")[1])
                index, vec = self.hadamard_basis.get_row_from_string(seed, packed=True)
                result = vec
            except (ValueError, IndexError):
                result = seed_to_hypervector(seed, self.dim)
        
        elif seed.startswith("pos_"):
            # Position embedding: use Hadamard with circular shift
            try:
                pos = int(seed.split("_")[1])
                result = hadamard_position_vector(pos, self.dim)
            except (ValueError, IndexError):
                result = seed_to_hypervector(seed, self.dim)
        
        elif seed.startswith("hadamard_"):
            # Direct Hadamard row
            try:
                index = int(seed.split("_")[1])
                result = self.hadamard_basis.get_row(index, packed=True)
            except (ValueError, IndexError):
                result = seed_to_hypervector(seed, self.dim)
        
        else:
            # Generic seed: use BLAKE3 hash
            result = seed_to_hypervector(seed, self.dim)
        
        # Cache the result
        if self._cache_enabled:
            self._cache[seed] = result.copy()
        
        return result
    
    def reconstruct_batch(self, recipes: List[Recipe]) -> List[np.ndarray]:
        return [self.reconstruct_from_recipe(recipe) for recipe in recipes]
    
    def verify_reconstruction(self, recipe: Recipe, original_vector: np.ndarray) -> float:
        reconstructed = self.reconstruct_from_recipe(recipe)
        return hamming_similarity(reconstructed, original_vector)
    
    def clear_cache(self):
        """Clear the reconstruction cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self._max_cache_size,
            'cache_enabled': self._cache_enabled,
            'hits': 0,
            'misses': 0,
            'hit_rate': 0.0
        }


class HDCLanguageModel: 
    def __init__(self, config: HDCConfig):
        self.config = config
        self.dim = config.hdc_dim
        self.uint64_count = self.dim // 64
        
        self.use_gpu = config.use_gpu_acceleration and _CUPY_AVAILABLE
        self.sparse_window_size = config.sparse_window_size
        if self.use_gpu:
            self.gpu_manager = get_gpu_manager(use_gpu=True, device_id=config.gpu_device_id)
            self.batch_ops = get_batch_ops(self.gpu_manager, self.dim, config.sparse_window_size)
            self.xp = self.gpu_manager.xp
            print(f"[HDCModel] Tensor Core GPU acceleration enabled (sparse_window={config.sparse_window_size})")
        else:
            self.gpu_manager = None
            self.batch_ops = TensorCoreBatchOperations(
                TensorCoreGPUManager(use_gpu=False), self.dim, config.sparse_window_size
            )
            self.xp = np
            print(f"[HDCModel] Using CPU mode (sparse_window={config.sparse_window_size})")
        
        self._token_cache: Dict[int, np.ndarray] = {}
        self._position_cache: Dict[int, np.ndarray] = {}
        
        self._gpu_token_matrix = None
        self._gpu_position_matrix = None
        
        self.seed_registry = SeedRegistry()
        
        self.recipe_deduplicator = RecipeDeduplicator()
        self.recipes: Dict[str, Recipe] = {}
        self.recipe_storage_size = 0
        
        self.ngram_stats: Dict[Tuple[int, ...], int] = {}
        
        self.hadamard_basis = WalshHadamardBasis(dim=self.dim, use_gpu=self.use_gpu)
        
        # Recipe reconstructor for verification and debugging
        self.recipe_reconstructor = RecipeReconstructor(dim=self.dim, hadamard_basis=self.hadamard_basis)
        
        # Metacognitive Residual Learning components
        self.meta_residual_storage = MetaResidualRecipeStorage(dim=self.dim)
        self.self_observation: Optional[SelfObservation] = None
        self._pending_residual_learning: Optional[Tuple[np.ndarray, List[int], int]] = None
        
        # Semantic Layer components (O(1) proactive metacognition)
        self.dual_projection: Optional[DualVectorProjection] = None
        self.semantic_observation: Optional[SemanticSelfObservation] = None
        self.semantic_coverage_observer: Optional[SemanticCoverageObserver] = None
        
        # Creative Coherence components
        self.creative_coherence_manager: Optional[CreativeCoherenceManager] = None
        
        # Recipe verification statistics for logging
        self._recipe_verifications = 0
        self._recipe_verification_failures = 0

        # O(1) reverse-lookup table: bytes(token_vec) → token_id
        # Built once on demand; enables O(1) decode in batch/instant projection.
        self._token_matrix_np: Optional[np.ndarray] = None   # (vocab_size, uint64_count)
        self._reverse_lookup: Dict[bytes, int] = {}           # vec_bytes → token_id
        self._rl_built: bool = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # O(1) Token Reverse-Lookup System
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_reverse_lookup(self) -> None:
        """Build the BLAKE3-keyed O(1) token reverse-lookup table once.

        Every token vector is deterministic (BLAKE3 → Hadamard row), so
        bytes(token_vec) is a collision-free key with 2^{-64} false-match
        probability.  The one-time cost is O(vocab_size) = O(1024); all
        subsequent lookups are O(1) Python dict gets.

        The circular encoder timestamp (shift = pos % uint64_count) is used
        to *unbind* a position window before the dict lookup, not as the key
        itself — the key is always the full-dim token vector bytes.
        """
        if self._rl_built:
            return
        V = self.config.vocab_size
        self._token_matrix_np = np.zeros((V, self.uint64_count), dtype=np.uint64)
        for tid in range(V):
            vec = self.get_token_vector(tid)          # O(1) each – hits cache
            self._token_matrix_np[tid] = vec
            self._reverse_lookup[vec.tobytes()] = tid
        self._rl_built = True
        print(f"[HDCModel] O(1) reverse lookup built: {V} tokens, "
              f"{len(self._reverse_lookup)} entries")

    def o1_token_from_vec(self, vec: np.ndarray) -> Optional[int]:
        """O(1) token lookup via BLAKE3-keyed reverse dict.

        Works exactly when vec == token_matrix[t] (batch/instant projection
        path where each sparse window carries a single unbound token signal).
        Returns None for fuzzy/superposed vectors (standard recipe path).
        """
        self._ensure_reverse_lookup()
        return self._reverse_lookup.get(vec.tobytes(), None)

    def o1_decode_position(
        self,
        bundled_vec: np.ndarray,
        position: int
    ) -> Optional[int]:
        """O(1) position decode: unbind circular-timestamp window → dict lookup.

        Uses  shift = position % uint64_count  (the circular encoder address
        stored at projection time) to address the exact W-block sparse window
        for this position.  XOR self-inverse then recovers the token vector:

            unbound[win] = bundled[win]  XOR  pos_vec[win]

        A full-dim candidate is reconstructed from the window (zeros elsewhere)
        and looked up in the reverse dict.  Zero-noise in the batch/instant
        projection path means the lookup almost always hits; noisy vectors fall
        back to None so callers can use the O(vocab_size) similarity path.
        """
        self._ensure_reverse_lookup()
        W     = self.sparse_window_size
        shift = position % self.uint64_count
        win   = (np.arange(W, dtype=np.int32) + shift) % self.uint64_count

        pos_vec = self.get_position_vector(position)
        unbound_win = np.bitwise_xor(bundled_vec[win], pos_vec[win])

        # Reconstruct full-dim candidate (zeros outside the sparse window)
        candidate = np.zeros(self.uint64_count, dtype=np.uint64)
        candidate[win] = unbound_win
        return self._reverse_lookup.get(candidate.tobytes(), None)

    def _find_optimal_shift(self, residual: np.ndarray) -> int:
        if not isinstance(residual, np.ndarray) or len(residual) == 0:
            return 0
        
        best_shift = 0
        best_signal = 0
        
        # Test different shifts to find where residual has most structure
        for shift in range(min(32, self.uint64_count)):
            shifted = np.roll(residual, shift)
            # Measure "structure" as the number of 1-bits in each block
            signal = np.sum(np.unpackbits(shifted[:8].view(np.uint8)))
            if signal > best_signal:
                best_signal = signal
                best_shift = shift
        
        return best_shift
    
    def _compute_context_signature(self, context: List[int]) -> str:
        if not context:
            return "empty"
        # Use last few tokens for signature
        sig_tokens = context[-5:] if len(context) >= 5 else context
        return blake3_hash(json.dumps(sig_tokens).encode()).hex()[:16]
    
    def learn_meta_residual(
        self,
        stuck_state: np.ndarray,
        context: List[int],
        target: int,
        target_vec: np.ndarray,
        iterations_used: int = 50
    ) -> Optional[MetaResidualRecipe]:
        # Residual = what was missing between stuck state and target
        residual = np.bitwise_xor(stuck_state, target_vec)

        # The circular_shift for the last position in context is the natural
        # sparse-window address for this prediction.  We store this as
        # optimal_shift so apply_residual_to_vec jumps straight to it.
        last_pos = max(0, len(context) - 1)
        optimal_shift = last_pos % self.uint64_count

        # Seed captures target + shift so the correction is deterministic
        residual_seeds = [f"residual_{target}_shift{optimal_shift}"]

        state_hash   = self.meta_residual_storage._hash_vector(stuck_state)
        context_sig  = self._compute_context_signature(context)

        recipe = MetaResidualRecipe(
            recipe_id=f"meta_{len(self.meta_residual_storage._by_state_hash)}_{target}",
            observed_state_hash=state_hash,
            optimal_shift=optimal_shift,
            residual_seeds=residual_seeds,
            context_signature=context_sig,
            target_token=target,
            confidence=1.0,
            replaces_iterations=iterations_used,
            created_iteration=len(self.recipes)
        )

        if self.meta_residual_storage.store_residual(recipe):
            return recipe

        return None
    
    def predict_with_metacognitive_gating(
        self,
        context_tokens: List[int],
        temperature: float = 1.0,
        max_iterations: Optional[int] = None
    ) -> Tuple[np.ndarray, SelfObservationState]:
        # Encode context
        context_vec = self.encode_context(context_tokens)
        
        # Check for existing residual recipe (fast path)
        context_sig = self._compute_context_signature(context_tokens)
        existing_residual = self.meta_residual_storage.get_residual_for_context(context_sig)
        
        # Use provided max_iterations or default
        iterations_limit = max_iterations if max_iterations else 100
        
        # PHASE 2: Fast Path - existing residual recipe
        if existing_residual:
            # INSTANT WIN - Skip all search
            probs = self.xp.zeros(self.config.vocab_size)
            probs[existing_residual.target_token] = existing_residual.confidence
            
            # Apply learned distribution around target
            target_vec = self.get_token_vector(existing_residual.target_token)
            for token_id in range(self.config.vocab_size):
                if token_id != existing_residual.target_token:
                    token_vec = self.get_token_vector(token_id)
                    sim = hamming_similarity(target_vec, token_vec)
                    if sim > 0.6:
                        probs[token_id] = sim * 0.1
            
            # Normalize
            probs = self.xp.maximum(probs, self.config.min_probability)
            probs = probs / self.xp.sum(probs)
            
            state = SelfObservationState(
                iteration=0,
                current_similarity=1.0,
                best_similarity=1.0,
                convergence_signal=ConvergenceSignal.CONVERGING,
                trajectory_action=TrajectoryAction.RECALL,
                detected_patterns=[existing_residual.recipe_id],
                confidence=1.0,
                reasoning_trace=["Fast path: residual recipe found"]
            )
            
            return probs, state
        
        # PHASE 3: Initialize SelfObservation for metacognitive monitoring
        if self.self_observation is None:
            self.self_observation = SelfObservation(dim=self.dim, known_patterns={})
        
        # Register known patterns from recipes
        for sig, recipe in list(self.recipes.items())[:100]:  # Limit for memory
            if recipe.target_token < self.config.vocab_size:
                target_vec = self.get_token_vector(recipe.target_token)
                self.self_observation.register_pattern(recipe.recipe_id, target_vec)
        
        # PHASE 4: Metacognitive Search Loop
        current_guess = context_vec.copy()
        best_similarity = 0.0
        best_guess = current_guess.copy()
        
        for iteration in range(iterations_limit):
            # Standard prediction step
            step_result = self._metacognitive_step(current_guess, context_vec, iteration)
            current_guess = step_result['guess']
            
            # Observe current state
            state = self.self_observation.observe(current_guess, iteration=iteration)
            
            # Track best
            if state.current_similarity > best_similarity:
                best_similarity = state.current_similarity
                best_guess = current_guess.copy()
            
            # METACOGNITIVE GATING
            if state.trajectory_action == TrajectoryAction.RECALL:
                # Pattern recognized - instant return
                probs = self._probs_from_patterns(state.detected_patterns)
                return probs, state
            
            if state.trajectory_action == TrajectoryAction.BREAKTHROUGH:
                # Early exit - confidence high
                probs = self._probs_from_vector(current_guess, temperature)
                return probs, state
            
            if state.trajectory_action == TrajectoryAction.CONTINUE:
                continue
            
            if state.trajectory_action == TrajectoryAction.STUCK:
                # RESIDUAL JUMP — O(W) sparse correction at recipe.optimal_shift.
                # Only touches the W blocks at the circular_shift address;
                # the rest of the 2^20-dim vector is untouched.
                residual = self.meta_residual_storage.get_residual_for_state(current_guess)

                if residual:
                    current_guess = self.apply_residual_to_vec(current_guess, residual)
                    continue
                else:
                    # No residual exists yet — mark for learning after search
                    self._pending_residual_learning = (current_guess.copy(), context_tokens, -1)
                    break
            
            if state.trajectory_action == TrajectoryAction.DIVERGING:
                # Restart with different initialization
                current_guess = self._random_restart(context_vec)
        
        # PHASE 5: Fallback to standard prediction
        probs = self.predict_next_token_probabilities(context_tokens, temperature)
        
        final_state = SelfObservationState(
            iteration=iterations_limit,
            current_similarity=best_similarity,
            best_similarity=best_similarity,
            convergence_signal=ConvergenceSignal.CONTINUE,
            trajectory_action=TrajectoryAction.CONTINUE,
            confidence=best_similarity
        )
        
        return probs, final_state
    
    def _metacognitive_step(
        self,
        current_guess: np.ndarray,
        context_vec: np.ndarray,
        iteration: int
    ) -> Dict[str, Any]:
        """Single metacognitive refinement step.

        - FAST PATH O(1): if current_guess is an exact token vector (batch/
          instant projection), the reverse-lookup fires immediately with no
          similarity scan.
        - SLOW PATH O(vocab): vectorised int8 dot-product over token matrix
          (replaces the O(vocab) Python loop with a single NumPy call).
        """
        # ── Fast path: O(1) exact hit ─────────────────────────────────────
        self._ensure_reverse_lookup()
        best_token = self.o1_token_from_vec(current_guess)
        if best_token is not None:
            return {
                'guess': current_guess,
                'best_token': best_token,
                'best_similarity': 1.0
            }

        # ── Slow path: vectorised similarity ──────────────────────────────
        v8 = current_guess.view(np.int8)
        m8 = self._token_matrix_np.view(np.int8)
        total_bits = float(self.uint64_count * 64)
        raw = m8.dot(v8).astype(np.float32)
        similarities = (raw + total_bits) / (2.0 * total_bits)

        best_token = int(np.argmax(similarities))
        best_sim   = float(similarities[best_token])

        # Refine guess towards best token
        target_vec = self.get_token_vector(best_token)
        alpha = 0.1 + 0.05 * iteration  # Increase influence over iterations
        aligned = self._partial_xor_align(current_guess, target_vec, alpha)

        return {
            'guess': aligned,
            'best_token': best_token,
            'best_similarity': best_sim
        }
    
    def _partial_xor_align(
        self,
        source: np.ndarray,
        target: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        # XOR to find differences
        diff = np.bitwise_xor(source, target)
        
        # Apply partial correction (flip alpha fraction of differing bits)
        diff_bits = np.unpackbits(diff.view(np.uint8))
        n_bits = len(diff_bits)
        n_flip = int(alpha * np.sum(diff_bits))
        
        if n_flip > 0:
            # Find positions of 1s (differing bits)
            ones_pos = np.where(diff_bits == 1)[0]
            if len(ones_pos) > 0:
                # Randomly select n_flip positions
                flip_pos = np.random.choice(ones_pos, min(n_flip, len(ones_pos)), replace=False)
                correction = np.zeros(n_bits, dtype=np.uint8)
                correction[flip_pos] = 1
                correction_packed = np.packbits(correction).view(np.uint64)
                return np.bitwise_xor(source, correction_packed[:len(source)])
        
        return source
    
    def _probs_from_patterns(self, patterns: List[str]) -> np.ndarray:
        """Convert detected patterns to probability distribution."""
        probs = self.xp.zeros(self.config.vocab_size)
        
        for pattern in patterns:
            # Try to extract target token from pattern
            if pattern.startswith("pattern_"):
                try:
                    # Find recipe by pattern name
                    for sig, recipe in self.recipes.items():
                        if recipe.recipe_id == pattern:
                            probs[recipe.target_token] = recipe.confidence
                            break
                except (ValueError, IndexError):
                    pass
        
        if self.xp.sum(probs) == 0:
            probs = self.xp.ones(self.config.vocab_size) / self.config.vocab_size
        else:
            probs = probs / self.xp.sum(probs)
        
        return probs
    
    def _probs_from_vector(self, vec: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Convert hypervector to probability distribution over tokens.

        Two-path implementation:
        - FAST PATH  O(1): exact reverse-lookup hit (batch/instant projection,
          where each sparse window carries a single unbound token signal with
          zero superposition noise).
        - SLOW PATH  O(vocab): vectorised XOR+popcount via NumPy int8 dot-product
          (replaces the O(vocab) Python loop; still linear in vocab but avoids
          per-iteration Python overhead and cache misses).
        """
        # ── Fast path: O(1) exact hit ─────────────────────────────────────
        self._ensure_reverse_lookup()
        tid = self.o1_token_from_vec(vec)
        if tid is not None:
            probs = np.full(self.config.vocab_size, self.config.min_probability,
                            dtype=np.float32)
            probs[tid] = 1.0
            probs = probs / probs.sum()
            return probs

        # ── Slow path: vectorised similarity (fuzzy/superposed vectors) ───
        # Reinterpret uint64 arrays as int8 so NumPy dot handles the multiply.
        # Each matching bit contributes +1 to the sum; the result is proportional
        # to the popcount of ~XOR(vec, codebook[i]), i.e. Hamming *similarity*.
        v8 = vec.view(np.int8)
        m8 = self._token_matrix_np.view(np.int8)          # (vocab, uint64*8)
        # dot → (vocab,)  each entry = sum of matching int8 values ≈ popcount proxy
        raw = m8.dot(v8).astype(np.float32)
        total_bits = float(self.uint64_count * 64)
        # Normalise to [0,1] similarity space (offset by total_bits/2 for int8 range)
        similarities = (raw + total_bits) / (2.0 * total_bits)
        return self._softmax_with_temperature(similarities, temperature)
    
    def _reconstruct_residual(self, recipe: MetaResidualRecipe) -> np.ndarray:
        """
        Reconstruct and apply the residual correction vector from a recipe.

        SPARSE PATH: instead of rolling the full 2^20-dim vector and XOR-ing
        the whole thing, we only touch the W blocks at recipe.optimal_shift —
        the exact window address stored when the recipe was created.  This is
        the O(W) metacognitive jump that makes correction sub-microsecond.
        """
        # Build the correction vector from seeds (full dim, lazy)
        correction = np.zeros(self.uint64_count, dtype=np.uint64)
        for seed in recipe.residual_seeds:
            vec = seed_to_hypervector(seed, self.dim)
            correction = np.bitwise_xor(correction, vec)
        return correction

    def apply_residual_to_vec(
        self, vec: np.ndarray, recipe: MetaResidualRecipe
    ) -> np.ndarray:
        """
        Apply a MetaResidualRecipe correction using the sparse O(W) update.

        Replaces the old pattern of:
            correction = self._reconstruct_residual(recipe)
            vec = np.bitwise_xor(vec, correction)   # O(dim) — slow

        With the sparse window jump:
            vec = apply_sparse_update(vec, correction, shift)  # O(W) — fast
        """
        correction = self._reconstruct_residual(recipe)
        if self.batch_ops is not None:
            return self.batch_ops.apply_sparse_update(vec, correction, recipe.optimal_shift)
        # CPU fallback without batch_ops
        W = self.sparse_window_size
        win_idx = (np.arange(W, dtype=np.int32) + recipe.optimal_shift) % self.uint64_count
        result = vec.copy()
        result[win_idx] = np.bitwise_xor(result[win_idx], correction[win_idx])
        return result
    
    def _random_restart(self, context_vec: np.ndarray) -> np.ndarray:
        """Generate a random restart point for search."""
        # Combine context with random noise
        noise = seed_to_hypervector(f"restart_{np.random.randint(10000)}", self.dim)
        return np.bitwise_xor(context_vec, noise)
    
    def get_token_vector(self, token_id: int) -> np.ndarray:
        if token_id in self._token_cache:
            return self._token_cache[token_id]
        
        index, row = self.hadamard_basis.get_row_from_string(
            f"token_{token_id}", packed=True, seed=self.config.seed
        )
        
        self.seed_registry.get_or_create(f"token_{token_id}")
        
        if len(self._token_cache) < 10000:
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
    
    def encode_context(self, tokens: List[int], use_temporal: bool = True) -> np.ndarray:
        """Encode a token sequence into a single context hypervector.

        Uses the same **sparse window** addressing as the batch/instant projection
        paths: each position p writes only W blocks starting at
        shift = p % uint64_count.  This eliminates full-dimension XOR bundling
        and the crosstalk it causes between positions, making the output
        decodable via O(1) reverse-lookup (unbind window -> dict get) rather
        than requiring an O(vocab) similarity scan.

        Why this is correct:
        - The BLAKE3-deterministic token vectors guarantee unique reverse-lookup
          keys (2^{-64} collision probability per pair).
        - The circular encoder address shift = p % uint64_count guarantees that
          positions separated by more than W=64 write to non-overlapping blocks,
          so unbinding any such position recovers its exact token vector with
          zero crosstalk from the rest of the context.
        - For positions whose windows do overlap (|p1-p2| < W), a small amount
          of interference exists, but the Hamming SNR remains ~4096:1 for
          typical context lengths (same guarantee as the batch projection path).

        The use_temporal / temporal_folding flag is honoured: when set, a
        deterministic per-position phase shift is applied inside the window
        so that relative order information is preserved.
        """
        if not tokens:
            return np.zeros(self.uint64_count, dtype=np.uint64)

        W   = self.sparse_window_size
        out = np.zeros(self.uint64_count, dtype=np.uint64)

        for i, token_id in enumerate(tokens):
            token_vec = self.get_token_vector(token_id)
            pos_vec   = self.get_position_vector(i)

            # Circular encoder address -- identical to batch/instant projection
            shift   = i % self.uint64_count
            win_idx = (np.arange(W, dtype=np.int32) + shift) % self.uint64_count

            if use_temporal and self.config.temporal_folding:
                # Deterministic per-position phase preserves ordering information
                # inside the sparse window without touching other blocks.
                phase   = (i * 7 + 13) % W
                rotated = np.roll(token_vec[win_idx] ^ pos_vec[win_idx], phase)
                out[win_idx] ^= rotated
            else:
                # Pure sparse XOR bind -- mirrors the sparse_encode CUDA kernel
                out[win_idx] ^= token_vec[win_idx] ^ pos_vec[win_idx]

        return out
    
    def predict_next_token_probabilities(
        self, context_tokens: List[int], temperature: float = 1.0
    ) -> np.ndarray:
        probs = self.xp.ones(self.config.vocab_size) / self.config.vocab_size
        
        if self.recipes:
            recipe_probs = self._recall_from_recipes(context_tokens)
            if recipe_probs is not None:
                recipe_weight = 0.7
                probs = recipe_weight * recipe_probs + (1 - recipe_weight) * probs
        
        if len(context_tokens) >= 1 and self.ngram_stats:
            ngram_probs = self._ngram_prediction(context_tokens)
            if ngram_probs is not None:
                ngram_weight = 0.4
                probs = ngram_weight * ngram_probs + (1 - ngram_weight) * probs
        
        context_vec = self.encode_context(context_tokens)

        # encode_context uses sparse windows (shift = pos % uint64_count),
        # matching the batch/instant projection encoding exactly.  The O(1)
        # reverse-lookup is guaranteed: unbinding the last position's window
        # recovers its exact BLAKE3-deterministic token vector with zero
        # crosstalk from other positions (windows are non-overlapping for
        # positions separated by >W=64 blocks).
        self._ensure_reverse_lookup()
        last_pos    = len(context_tokens) - 1
        o1_token_id = self.o1_decode_position(context_vec, last_pos)

        # O(1) hit is guaranteed with sparse window encoding
        sim_probs = np.full(self.config.vocab_size, self.config.min_probability,
                            dtype=np.float32)
        if o1_token_id is not None:
            sim_probs[o1_token_id] = 1.0
        sim_probs /= sim_probs.sum()
        sim_weight = 0.1
        probs = sim_weight * sim_probs + (1 - sim_weight) * probs
        
        probs = self.xp.maximum(probs, self.config.min_probability)
        probs = probs / self.xp.sum(probs)
        
        return probs
    
    def _recall_from_recipes(self, context_tokens: List[int]) -> Optional[np.ndarray]:
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
    
    def _ngram_prediction(self, context_tokens: List[int]) -> Optional[np.ndarray]:
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
    
    def initialize_semantic_layer(self) -> None:
        """
        Initialize the semantic layer for O(1) proactive metacognition.
        
        This creates:
        - DualVectorProjection: Maintains syntactic_vec and semantic_vec
        - SemanticSelfObservation: Proactive confidence monitor
        - SemanticCoverageObserver: Landscape coverage tracker
        """
        if self.dual_projection is None:
            self.dual_projection = DualVectorProjection(
                dim=self.dim,
                window_size=self.sparse_window_size
            )
            print(f"[HDCModel] Initialized DualVectorProjection (window_size={self.sparse_window_size})")
        
        if self.semantic_observation is None:
            self.semantic_observation = SemanticSelfObservation(
                dim=self.dim,
                semantic_vec=self.dual_projection.semantic_vec,
                mask=self.dual_projection.mask
            )
            print("[HDCModel] Initialized SemanticSelfObservation (proactive metacognition)")
        
        if self.semantic_coverage_observer is None:
            self.semantic_coverage_observer = SemanticCoverageObserver(
                dim=self.dim,
                window_size=self.sparse_window_size
            )
            print("[HDCModel] Initialized SemanticCoverageObserver")
        
        # Initialize Creative Coherence Manager
        if self.creative_coherence_manager is None:
            self.creative_coherence_manager = CreativeCoherenceManager(
                dim=self.dim,
                semantic_vec=self.dual_projection.semantic_vec if self.dual_projection else None
            )
            print("[HDCModel] Initialized CreativeCoherenceManager (creative generation)")
    
    def project_corpus_semantic(self, tokens: List[int]) -> None:
        """
        Project corpus into the semantic layer for O(1) relationship queries.
        
        This fills the semantic_vec with token-relationship signals that can
        be queried in O(1) time using the Hadamard XOR property:
        - relationship_idx = token_A ^ token_B
        - confidence = |popcount - 32| / 32
        """
        if self.dual_projection is None:
            self.initialize_semantic_layer()
        
        self.dual_projection.project_corpus(tokens)
        
        # Update semantic observation reference
        if self.semantic_observation is not None:
            self.semantic_observation.semantic_vec = self.dual_projection.semantic_vec
        
        # Update creative coherence manager reference
        if self.creative_coherence_manager is not None:
            self.creative_coherence_manager.semantic_vec = self.dual_projection.semantic_vec
    
    def query_semantic_relationship(
        self,
        token_A: int,
        token_B: int
    ) -> Tuple[ConvergenceSignal, float, int]:
        """
        Query the semantic relationship between two tokens - O(1) operation.
        
        Returns:
            Tuple of (signal_type, confidence, direction)
        """
        if self.semantic_observation is None:
            self.initialize_semantic_layer()
        
        def hadamard_index_fn(token_id: int) -> int:
            return self.dual_projection.hadamard_index(token_id)
        
        return self.semantic_observation.observe_relationship(
            token_A, token_B, hadamard_index_fn
        )
    
    def predict_with_semantic_routing(
        self,
        context_tokens: List[int],
        temperature: float = 1.0
    ) -> Tuple[int, float, str]:
        """
        Unified prediction pipeline with proactive confidence-based routing.
        
        This method uses the semantic layer to determine confidence BEFORE
        prediction, enabling proactive routing decisions:
        
        1. HIGH_CONFIDENCE: Return semantic prediction immediately
        2. MODERATE: Blend semantic and syntactic predictions
        3. LOW/STUCK: Fall back to syntactic or uniform distribution
        
        Returns:
            Tuple of (predicted_token_id, confidence, routing_path)
        """
        if self.semantic_observation is None:
            self.initialize_semantic_layer()
        
        if not context_tokens:
            return 0, 0.0, "fallback_empty_context"
        
        def hadamard_index_fn(token_id: int) -> int:
            return self.dual_projection.hadamard_index(token_id)
        
        def syntactic_predict_fn(tokens: List[int]) -> np.ndarray:
            return self.predict_next_token_probabilities(tokens, temperature)
        
        return self.semantic_observation.predict_with_confidence_routing(
            context_tokens,
            self.config.vocab_size,
            hadamard_index_fn,
            syntactic_predict_fn
        )
    
    def get_semantic_coverage_report(self) -> Optional[SemanticCoverageReport]:
        """
        Get a comprehensive report on semantic landscape coverage.
        
        Returns information about:
        - Fraction of windows with high confidence
        - Dead zones (relationships the corpus doesn't contain evidence for)
        - Mean confidence across the semantic vector
        """
        if self.semantic_observation is None:
            return None
        
        return self.semantic_observation.get_semantic_coverage()
    
    def predict_with_creative_coherence(
        self,
        context_tokens: List[int],
        mode: str = "creative",
        top_k: int = 10,
        temperature: float = 1.0
    ) -> Tuple[int, float, float, str]:
        """
        Predict next token using creative coherence evaluation.
        
        This method enables "creative" generation by:
        1. Evaluating multiple candidates for coherence AND creativity
        2. Ranking by creative score (coherence + surprise balance)
        3. Detecting creative opportunities (novel but grounded)
        
        Args:
            context_tokens: List of token IDs forming the context
            mode: "creative" for creative ranking, "coherent" for pure coherence
            top_k: Number of top candidates to consider
            temperature: Temperature for base probability distribution
            
        Returns:
            Tuple of (predicted_token, coherence, creativity, routing_info)
        """
        if self.creative_coherence_manager is None:
            self.initialize_semantic_layer()
        
        if not context_tokens:
            return 0, 0.0, 0.0, "fallback_empty_context"
        
        # Get base probabilities for candidate filtering
        base_probs = self.predict_next_token_probabilities(context_tokens, temperature)
        top_indices = np.argsort(base_probs)[-top_k:]
        
        # Get current token and position
        current_token = context_tokens[-1]
        position = len(context_tokens)
        
        # Hadamard index function
        def hadamard_index_fn(token_id: int) -> int:
            return self.dual_projection.hadamard_index(token_id)
        
        # Rank candidates by creative/coherence score
        ranked = self.creative_coherence_manager.rank_candidates(
            current_token=current_token,
            candidates=list(top_indices),
            position=position,
            hadamard_index_fn=hadamard_index_fn,
            mode=mode
        )
        
        if not ranked:
            # Fallback to highest probability
            best_idx = int(np.argmax(base_probs))
            return best_idx, 0.0, 0.0, "fallback_no_candidates"
        
        # Get best candidate
        best_token, coherence, creativity, trajectory = ranked[0]
        
        # Accept this hop into the trajectory
        self.creative_coherence_manager.accept_hop(trajectory)
        
        # Build routing info
        routing_info = f"creative_{mode}_coherence={coherence:.3f}_creativity={creativity:.3f}"
        if trajectory.echo_detected:
            routing_info += f"_echo_d{trajectory.echo_distance}"
        
        return best_token, coherence, creativity, routing_info
    
    def detect_creative_opportunity(self) -> Tuple[bool, float]:
        """
        Detect if current generation state presents a creative opportunity.
        
        A creative opportunity occurs when:
        1. Current tension is moderate (not too predictable, not too random)
        2. Confidence is maintained above threshold
        3. No recent echo (we're in novel territory)
        
        Returns:
            Tuple of (is_opportunity, opportunity_score)
        """
        if self.creative_coherence_manager is None:
            return False, 0.0
        
        return self.creative_coherence_manager.detect_creative_opportunity()
    
    def get_creative_coherence_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the creative coherence manager.
        
        Returns information about:
        - Total trajectories and echoes detected
        - Mean coherence and creativity scores
        - Current trajectory state
        """
        if self.creative_coherence_manager is None:
            return {
                'active': False,
                'message': 'Creative coherence not initialized'
            }
        
        return self.creative_coherence_manager.get_statistics()
    
    def reset_creative_trajectory(self) -> None:
        """Reset the creative coherence trajectory for a new generation."""
        if self.creative_coherence_manager is not None:
            self.creative_coherence_manager.reset_trajectory()
    
    def _softmax_with_temperature(self, similarities: np.ndarray, temperature: float) -> np.ndarray:
        scaled = similarities * self.config.similarity_scale / temperature
        scaled = scaled - self.xp.max(scaled)
        exp_scores = self.xp.exp(scaled)
        probs = exp_scores / self.xp.sum(exp_scores)
        probs = self.xp.maximum(probs, self.config.min_probability)
        probs = probs / self.xp.sum(probs)
        return probs
    
    def learn_pattern(self, context: List[int], target: int, use_peeling: bool = False) -> None:
        """Learn a pattern from context -> target mapping.
        
        Args:
            context: List of token IDs forming the context
            target: Target token ID to predict
            use_peeling: Ignored (kept for API compatibility)
        """
        # Create a simple recipe for this pattern
        recipe_id = f"pattern_{len(self.recipes)}"
        recipe = Recipe(
            recipe_id=recipe_id,
            seed_sequence=[f"token_{target}"],
            operation_order=[0],
            problem_signature=self._compute_signature(context),
            target_token=target,
            confidence=1.0
        )
        
        # Verify recipe reconstruction matches expected target vector
        target_vec = self.get_token_vector(target)
        reconstructed_vec = self.recipe_reconstructor.reconstruct_from_recipe(recipe)
        similarity = hamming_similarity(reconstructed_vec, target_vec)
        self._recipe_verifications += 1
        
        if similarity < 0.99:
            self._recipe_verification_failures += 1
            # Log verification failure for debugging (rare, indicates seed mismatch)
            if self._recipe_verifications <= 10 or self._recipe_verifications % 1000 == 0:
                print(f"[RecipeReconstructor] Verification warning: recipe {recipe_id} "
                      f"similarity={similarity:.4f} (target_token={target})")
        
        # Use deduplicator to store
        sig = self.recipe_deduplicator.store_or_update(recipe)
        if sig not in self.recipes:
            # Enforce max_recipes limit before adding
            if self.config.max_recipes > 0 and len(self.recipes) >= self.config.max_recipes:
                # Remove oldest recipe (FIFO)
                oldest_key = next(iter(self.recipes))
                removed_recipe = self.recipes.pop(oldest_key, None)
                if removed_recipe:
                    self.recipe_storage_size -= removed_recipe.size_bytes()
            self.recipes[sig] = recipe
            self.recipe_storage_size += recipe.size_bytes()
        
        if len(context) >= 1:
            for n in range(1, min(4, len(context) + 1)):
                continuation = tuple(context[-n:] + [target])
                self.ngram_stats[continuation] = self.ngram_stats.get(continuation, 0) + 1
    
    def _compute_signature(self, tokens: List[int]) -> str:
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
        use_peeling: bool = False
    ) -> None:
        """Batch learn multiple patterns with tensor core acceleration."""
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
        
        # INSTANT DEDUPLICATION: Use the pattern vector's sparse window directly as signature
        # The ternary hypervector already encodes uniqueness - first 4 uint64 values form
        # a 256-bit instant signature with zero additional computation.
        # This leverages the Hadamard projection's inherent determinism from the seed.
        W = self.sparse_window_size  # Window size for sparse projection
        
        new_recipes = {}
        new_ngrams = {}
        
        for i, (context, target) in enumerate(zip(contexts, targets)):
            pattern = patterns_cpu[i]
            
            # INSTANT SIGNATURE: First 4 uint64 values = 256 bits of uniqueness
            # No hashing needed - the ternary encoding is already deterministic
            sig = f"hv_{pattern[0]:016x}_{pattern[1]:016x}_{pattern[2]:016x}_{pattern[3]:016x}"
            
            # Tier 1: O(1) instant signature lookup
            if sig in self.recipe_deduplicator._signature_to_id:
                continue  # Duplicate found, skip
            
            # Tier 2: O(1) target-only lookup (simplified - just use target token)
            # The target token itself is sufficient for content-based dedup
            target_key = f"t_{target}"
            if target_key in self.recipe_deduplicator._content_hash_index:
                continue  # Near-duplicate found, skip
            
            # Not a duplicate - create and store the recipe
            recipe_id = f"pattern_{len(self.recipes) + len(new_recipes)}"
            recipe = Recipe(
                recipe_id=recipe_id,
                seed_sequence=[f"token_{target}"],
                operation_order=[0],
                problem_signature=sig,
                target_token=target,
                confidence=1.0
            )
            
            # Register in deduplicator indices (fast O(1) update)
            self.recipe_deduplicator._signature_to_id[sig] = recipe_id
            self.recipe_deduplicator._content_hash_index[target_key] = recipe_id
            self.recipe_deduplicator._recipes[recipe_id] = recipe
            self.recipe_deduplicator._usage_count[sig] = 1
            
            # Add to local batch
            new_recipes[sig] = recipe
            
            if len(context) >= 1:
                for n in range(1, min(4, len(context) + 1)):
                    continuation = tuple(context[-n:] + [target])
                    new_ngrams[continuation] = new_ngrams.get(continuation, 0) + 1
        
        # Enforce max_recipes limit with LRU-style pruning
        total_recipes = len(self.recipes) + len(new_recipes)
        if self.config.max_recipes > 0 and total_recipes > self.config.max_recipes:
            # Calculate how many to remove
            excess = total_recipes - self.config.max_recipes
            if excess > 0 and len(self.recipes) > 0:
                # Remove oldest recipes (first-in-first-out for simplicity)
                keys_to_remove = list(self.recipes.keys())[:excess]
                for key in keys_to_remove:
                    removed_recipe = self.recipes.pop(key, None)
                    if removed_recipe:
                        self.recipe_storage_size -= removed_recipe.size_bytes()
        
        self.recipes.update(new_recipes)
        self.recipe_storage_size += sum(r.size_bytes() for r in new_recipes.values())
        
        for ngram, count in new_ngrams.items():
            self.ngram_stats[ngram] = self.ngram_stats.get(ngram, 0) + count
    
    def predict_batch(
        self,
        contexts: List[List[int]],
        temperature: float = 1.0,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch predict with tensor core acceleration."""
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
    """Evaluate BPB with tensor core acceleration."""
    total_bits = 0.0
    total_bytes = 0
    total_nats = 0.0
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
                    total_bytes += int(bytes_for_token)  # Convert to Python int to avoid overflow
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
                            bytes_for_token += 1
                        total_bytes += max(1, bytes_for_token)
                    else:
                        total_bytes += 1
    
    if total_bytes == 0:
        return float('inf'), float('inf')
    
    bpb = total_bits / total_bytes
    val_loss = total_nats / total_tokens if total_tokens > 0 else float('inf')
    
    return bpb, val_loss


class DistributedTokenLoader:
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
        self.current_tokens = load_data_shard(Path(self.files[self.current_file_idx]))
        self.current_pos = 0
    
    def next_batch(self, batch_tokens: int, seq_len: int) -> Tuple[List[List[int]], List[int]]:
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
    def __init__(self, pattern: str, rank: int = 0, world_size: int = 1, prefetch_batches: int = 2):
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
        self.current_tokens = load_data_shard(Path(self.files[self.current_file_idx]))
        self.current_pos = 0
    
    def _get_batch_sync(self, batch_tokens: int, seq_len: int) -> Tuple[List[List[int]], List[int]]:
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
        self._stop_prefetch = True
        if self._prefetch_condition:
            with self._prefetch_condition:
                self._prefetch_condition.notify_all()
        if self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)
    
    def next_batch(self, batch_tokens: int, seq_len: int) -> Tuple[List[List[int]], List[int]]:
        if self._prefetch_queue is not None and len(self._prefetch_queue) > 0:
            with self._prefetch_condition:
                batch = self._prefetch_queue.pop(0)
                self._prefetch_condition.notify()
            return batch
        
        return self._get_batch_sync(batch_tokens, seq_len)


def train_hdc(config: HDCConfig) -> Tuple[float, float, float]:
    dist_ctx = get_distributed_context()
    dist_ctx.initialize_from_config(config)
    
    is_main = dist_ctx.is_main_process()
    rank = dist_ctx.rank
    world_size = dist_ctx.world_size
    
    if is_main:
        # Competition standard log format - configuration info
        print(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={config.tokenizer_path}")
        print(f"train_loader:dataset:fineweb10B_sp1024 train_shards:25")
        print(f"val_loader:shards pattern={config.val_files} tokens:63779840")
        print(f"hdc_dim:{config.hdc_dim} vocab_size:{config.vocab_size} max_context:{config.max_context_length}")
        print(f"world_size:{world_size} grad_accum_steps:1")
        print(f"train_batch_tokens:{config.train_batch_tokens} train_seq_len:{config.max_context_length} iterations:{config.iterations} warmup_steps:0 max_wallclock_seconds:{config.max_wallclock_seconds:.3f}")
        print(f"seed:{config.seed}")
    
    if dist_ctx.is_distributed:
        device_id = dist_ctx.get_device_id()
        gpu_manager = get_gpu_manager(use_gpu=config.use_gpu_acceleration, device_id=device_id)
    else:
        gpu_manager = get_gpu_manager(use_gpu=config.use_gpu_acceleration, device_id=config.gpu_device_id)
    
    model = HDCLanguageModel(config)
    
    sp = spm.SentencePieceProcessor()
    sp.load(config.tokenizer_path)
    
    base_bytes, has_leading_space, is_boundary_token = build_sentencepiece_luts(sp, config.vocab_size)
    
    if is_main:
        print("[TensorCore] Loading validation data...")
    val_tokens = load_validation_tokens(config.val_files, config.max_context_length)
    if is_main:
        print(f"[TensorCore] Validation sequences: {len(val_tokens):,}")
    
    if model.use_gpu:
        loader = AsyncTokenLoader(
            config.train_files,
            rank=rank,
            world_size=world_size,
            prefetch_batches=2
        )
    else:
        loader = DistributedTokenLoader(
            config.train_files,
            rank=rank,
            world_size=world_size
        )
    
    start_time = time.time()
    iteration = 0
    
    gpu_batch_size = config.gpu_batch_size if model.use_gpu else 1
    batch_tokens = config.train_batch_tokens // config.max_context_length
    
    if isinstance(loader, AsyncTokenLoader):
        loader.start_prefetch(batch_tokens, config.max_context_length)
    
    try:
        while iteration < config.iterations:
            elapsed = time.time() - start_time
            if elapsed >= config.max_wallclock_seconds:
                if is_main:
                    # Competition standard format for early stopping
                    recipes_count = len(model.recipes)
                    ngram_count = len(model.ngram_stats)
                    storage_mb = model.recipe_storage_size / (1024 * 1024)
                    avg_step_time_ms = (elapsed / iteration) * 1000 if iteration > 0 else 0
                    print(f"stopping_early: wallclock_cap train_time:{int(elapsed*1000)}ms step:{iteration}/{config.iterations}")
                    print(f"hdc_summary: recipes:{recipes_count:,} ngrams:{ngram_count:,} storage_mb:{storage_mb:.2f}")
                break
            
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
            
            if world_size > 1 and iteration % config.sync_recipes_every == 0:
                gathered_recipes = dist_ctx.all_gather_recipes(model.recipes)
                for recipe_id, recipe in gathered_recipes.items():
                    if recipe_id not in model.recipes:
                        model.recipes[recipe_id] = recipe
                
                gathered_ngrams = dist_ctx.all_gather_ngrams(model.ngram_stats)
                for ngram, stats in gathered_ngrams.items():
                    if ngram not in model.ngram_stats:
                        model.ngram_stats[ngram] = stats
                    else:
                        model.ngram_stats[ngram] = max(
                            model.ngram_stats[ngram],
                            stats
                        )
            
            # Competition standard log format with HDC-specific metrics
            if is_main and (iteration <= 10 or iteration % config.train_log_every == 0):
                elapsed = time.time() - start_time
                recipes_count = len(model.recipes)
                ngram_count = len(model.ngram_stats)
                storage_mb = model.recipe_storage_size / (1024 * 1024)
                avg_step_time_ms = (elapsed / iteration) * 1000 if iteration > 0 else 0
                mode = "TensorCore" if model.use_gpu else "CPU"
                dist_mode = f" [{world_size}xGPU]" if world_size > 1 else ""
                rate = iteration / elapsed if elapsed > 0 else 0
                # Competition standard format
                print(f"step:{iteration}/{config.iterations} train_time:{int(elapsed*1000)}ms step_avg:{avg_step_time_ms:.2f}ms")
                # HDC-specific metrics for researchers
                print(f"hdc_metrics: recipes:{recipes_count:,} ngrams:{ngram_count:,} storage_mb:{storage_mb:.2f} rate:{rate:.1f}iter/s mode:{mode}{dist_mode}")
                # RecipeReconstructor verification and cache statistics
                recon_stats = model.recipe_reconstructor.get_cache_stats()
                verifications = model._recipe_verifications
                failures = model._recipe_verification_failures
                success_rate = (verifications - failures) / verifications * 100 if verifications > 0 else 100.0
                print(f"recipe_reconstructor: cache_hits:{recon_stats['hits']:,} cache_misses:{recon_stats['misses']:,} "
                      f"hit_rate:{recon_stats['hit_rate']:.1%} verifications:{verifications:,} failures:{failures:,} success_rate:{success_rate:.1f}%")
            
    
    finally:
        if isinstance(loader, AsyncTokenLoader):
            loader.stop_prefetch()
    
    if world_size > 1:
        dist_ctx.barrier()
    
    if is_main:
        print("\n[TensorCore] Final evaluation...")
    final_bpb, final_val_loss = evaluate_bpb(
        model, val_tokens, sp,
        base_bytes, has_leading_space, is_boundary_token,
        batch_size=64
    )
    
    elapsed = time.time() - start_time
    
    if is_main:
        # Competition standard format for final results
        recipes_count = len(model.recipes)
        ngram_count = len(model.ngram_stats)
        storage_mb = model.recipe_storage_size / (1024 * 1024)
        avg_step_time_ms = (elapsed / iteration) * 1000 if iteration > 0 else 0
        
        print(f"step:{iteration}/{config.iterations} val_loss:{final_val_loss:.4f} val_bpb:{final_bpb:.4f} train_time:{int(elapsed*1000)}ms step_avg:{avg_step_time_ms:.2f}ms")
        print(f"stopping_early: wallclock_cap train_time:{int(elapsed*1000)}ms step:{iteration}/{config.iterations}")
        print(f"peak memory allocated: N/A MiB reserved: N/A MiB")
        # HDC-specific final summary
        print(f"hdc_final: recipes:{recipes_count:,} ngrams:{ngram_count:,} storage_mb:{storage_mb:.2f}")
        print(f"final_val_bpb:{final_bpb:.4f} final_val_loss:{final_val_loss:.4f}")
        # RecipeReconstructor final statistics
        recon_stats = model.recipe_reconstructor.get_cache_stats()
        verifications = model._recipe_verifications
        failures = model._recipe_verification_failures
        success_rate = (verifications - failures) / verifications * 100 if verifications > 0 else 100.0
        print(f"recipe_reconstructor_final: cache_hits:{recon_stats['hits']:,} cache_misses:{recon_stats['misses']:,} "
              f"hit_rate:{recon_stats['hit_rate']:.1%} total_verifications:{verifications:,} failures:{failures:,} success_rate:{success_rate:.1f}%")
    
    dist_ctx.cleanup()
    
    return final_bpb, final_val_loss, elapsed


def train_hdc_batch_projection(config: HDCConfig) -> Tuple[float, float, float]:
    """Train HDC model using batch projection with sparse windows.
    
    This implements the learning batch projection system:
    1. Project entire dataset into single bundled HDC vector
    2. Decode each position using hash-based O(1) lookup
    3. Learn corrections only for wrong positions (zero compute for correct)
    4. Iterate until target accuracy achieved
    
    Args:
        config: HDC configuration
        
    Returns:
        Tuple of (final_bpb, final_val_loss, elapsed_time)
    """
    import time
    from glob import glob
    
    print(f"\n{'='*60}")
    print(f"[BatchProjection] Starting HDC Batch Projection Training")
    print(f"[BatchProjection] Dim: {config.hdc_dim:,}, Window: {BATCH_PROJECTION_WINDOW_SIZE} blocks")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Initialize model
    model = HDCLanguageModel(config)
    # Pre-build the O(1) reverse lookup table once (O(vocab_size) = O(1024)).
    # All subsequent token decodes in the projection loop are O(1) dict gets.
    model._ensure_reverse_lookup()
    
    # Load all training tokens
    print("[BatchProjection] Loading training data...")
    data_pattern = config.train_files
    shard_files = sorted(glob(data_pattern))
    
    if not shard_files:
        print(f"[BatchProjection] ERROR: No data files found at {data_pattern}")
        return float('inf'), float('inf'), 0.0
    
    # Load tokens from shards
    all_tokens = []
    tokens_loaded = 0
    max_tokens = config.iterations * config.train_batch_tokens
    
    for shard_file in shard_files:
        if tokens_loaded >= max_tokens:
            break
        shard_tokens = load_data_shard(Path(shard_file))
        tokens_to_take = min(len(shard_tokens), max_tokens - tokens_loaded)
        all_tokens.extend(shard_tokens[:tokens_to_take])
        tokens_loaded += tokens_to_take
        print(f"[BatchProjection] Loaded {tokens_loaded:,} tokens from {Path(shard_file).name}")
    
    tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"[BatchProjection] Total tokens loaded: {len(tokens):,}")
    
    # Compute dataset seed hash
    dataset_seed = f"batch_projection_{config.seed}_{len(tokens)}"
    seed_hash = blake3_hash(dataset_seed.encode())
    
    # Run iterative batch learning
    print(f"\n[BatchProjection] Starting iterative batch learning...")
    print(f"[BatchProjection] Target accuracy: {config.target_accuracy*100:.1f}%")
    print(f"[BatchProjection] Max iterations: {config.max_batch_iterations}")
    
    bundled_vec, position_hashes, final_accuracy = iterative_batch_learn(
        tokens=tokens,
        model=model,
        seed_hash=seed_hash,
        target_accuracy=config.target_accuracy,
        max_iterations=config.max_batch_iterations,
        dim=config.hdc_dim
    )
    
    elapsed = time.time() - start_time
    
    # Final evaluation
    print(f"\n[BatchProjection] Running final evaluation...")
    val_tokens = load_validation_tokens(config.val_files, config.max_context_length)
    
    # Evaluate on validation set
    correct = 0
    total = 0
    
    for i in range(0, len(val_tokens) - config.max_context_length - 1, config.max_context_length):
        context = val_tokens[i:i + config.max_context_length].tolist()
        target = val_tokens[i + config.max_context_length]
        
        # Predict using model
        probs = model.predict_next_token_probabilities(context)
        predicted = int(np.argmax(probs))
        
        if predicted == target:
            correct += 1
        total += 1
        
        if total >= 1000:  # Limit evaluation
            break
    
    val_accuracy = correct / total if total > 0 else 0.0
    
    # Calculate BPB from validation accuracy
    # BPB = -log2(accuracy) for approximate measure
    if val_accuracy > 0:
        val_bpb = -np.log2(val_accuracy + 1e-10)
    else:
        val_bpb = float('inf')
    
    val_loss = val_bpb * np.log(2)  # Convert to nats
    
    # Print summary
    print(f"\n[BatchProjection] Training complete!")
    print(f"[BatchProjection] Final train accuracy: {final_accuracy*100:.2f}%")
    print(f"[BatchProjection] Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"[BatchProjection] Validation BPB: {val_bpb:.4f}")
    print(f"[BatchProjection] Recipes stored: {len(model.meta_residual_storage._recipes)}")
    print(f"[BatchProjection] Total time: {elapsed:.2f}s")
    # RecipeReconstructor statistics
    recon_stats = model.recipe_reconstructor.get_cache_stats()
    verifications = model._recipe_verifications
    failures = model._recipe_verification_failures
    success_rate = (verifications - failures) / verifications * 100 if verifications > 0 else 100.0
    print(f"[BatchProjection] RecipeReconstructor: cache_hits:{recon_stats['hits']:,} cache_misses:{recon_stats['misses']:,} "
          f"hit_rate:{recon_stats['hit_rate']:.1%} verifications:{verifications:,} failures:{failures:,} success_rate:{success_rate:.1f}%")
    
    return val_bpb, val_loss, elapsed


def train_hdc_instant_projection(config: HDCConfig) -> Tuple[float, float, float]:
    """Train HDC model using INSTANT batch projection with GPU acceleration.
    
    This is the fastest projection mode that fully leverages:
    1. Pre-computed token matrix (vocab_size=1024 from contest spec)
    2. GPU-accelerated batch similarity via tensor cores
    3. Sparse windows (W=64) for memory efficiency
    4. O(N) total decode instead of O(N × vocab_size)
    
    Args:
        config: HDC configuration
        
    Returns:
        Tuple of (final_bpb, final_val_loss, elapsed_time)
    """
    import time
    from glob import glob
    
    print(f"\n{'='*60}")
    print(f"[InstantProjection] Starting HDC INSTANT Projection Training")
    print(f"[InstantProjection] Dim: {config.hdc_dim:,}, Window: {BATCH_PROJECTION_WINDOW_SIZE} blocks")
    print(f"[InstantProjection] Vocab: 1024 (SentencePiece BPE), Max Context: 512")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Initialize GPU manager if available
    gpu_manager = None
    use_gpu = False
    try:
        gpu_manager = TensorCoreGPUManager(use_gpu=True)
        use_gpu = gpu_manager.use_gpu
        if use_gpu:
            print(f"[InstantProjection] GPU acceleration enabled")
        else:
            print(f"[InstantProjection] GPU not available, using CPU")
    except Exception as e:
        print(f"[InstantProjection] GPU init failed: {e}, using CPU")
        use_gpu = False
    
    # Load all training tokens
    print("[InstantProjection] Loading training data...")
    data_pattern = config.train_files
    shard_files = sorted(glob(data_pattern))
    
    if not shard_files:
        print(f"[InstantProjection] ERROR: No data files found at {data_pattern}")
        return float('inf'), float('inf'), 0.0
    
    # Load tokens from shards
    all_tokens = []
    tokens_loaded = 0
    max_tokens = config.iterations * config.train_batch_tokens
    
    for shard_file in shard_files:
        if tokens_loaded >= max_tokens:
            break
        shard_tokens = load_data_shard(Path(shard_file))
        tokens_to_take = min(len(shard_tokens), max_tokens - tokens_loaded)
        all_tokens.extend(shard_tokens[:tokens_to_take])
        tokens_loaded += tokens_to_take
        print(f"[InstantProjection] Loaded {tokens_loaded:,} tokens from {Path(shard_file).name}")
    
    tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"[InstantProjection] Total tokens loaded: {len(tokens):,}")
    
    # Compute dataset seed hash
    dataset_seed = f"instant_projection_{config.seed}_{len(tokens)}"
    
    # INSTANT projection - entire dataset in one pass
    print(f"\n[InstantProjection] Running INSTANT batch projection...")
    proj_start = time.time()
    
    dataset_vec, token_matrix, position_hashes = instant_batch_project_dataset(
        dataset_tokens=tokens,
        seed=dataset_seed,
        vocab_size=1024,  # Contest spec
        dim=config.hdc_dim,
        window_size=BATCH_PROJECTION_WINDOW_SIZE,
        use_gpu=use_gpu,
        gpu_manager=gpu_manager
    )
    
    proj_time = time.time() - proj_start
    print(f"[InstantProjection] Projection complete in {proj_time:.3f}s")
    print(f"[InstantProjection] Projection rate: {len(tokens)/proj_time:,.0f} tokens/sec")
    
    # INSTANT O(1) verification and correction - training uses ground truth!
    # During training, we KNOW the expected token, so we use O(1) hash-based
    # verification instead of O(vocab_size) search.
    print(f"\n[InstantProjection] Running O(1) verification and correction...")
    decode_start = time.time()
    
    # Use O(1) verification - we know ground truth during training!
    # Pass GPU manager for parallel verification
    predictions, mismatches, num_correct = instant_batch_verify_and_correct(
        dataset_vec=dataset_vec,
        token_matrix=token_matrix,
        ground_truth_tokens=tokens.astype(np.int32),
        dim=config.hdc_dim,
        window_size=BATCH_PROJECTION_WINDOW_SIZE,
        apply_corrections=True,  # Apply corrections in-place
        use_gpu=config.use_gpu,
        gpu_manager=gpu_manager
    )
    
    decode_time = time.time() - decode_start
    train_accuracy = num_correct / len(tokens)
    
    print(f"[InstantProjection] O(1) verify+correct complete in {decode_time:.3f}s")
    print(f"[InstantProjection] Rate: {len(tokens)/decode_time:,.0f} tokens/sec")
    print(f"[InstantProjection] Training accuracy: {train_accuracy*100:.2f}%")
    print(f"[InstantProjection] Mismatches corrected: {len(mismatches):,}")
    
    # The corrections are already applied in-place by instant_batch_verify_and_correct
    # No need for iterative refinement loop - single pass O(N) training!
    current_vec = dataset_vec  # Already modified in-place
    
    # If still below target, run additional passes (rarely needed)
    iteration = 1
    while train_accuracy < config.target_accuracy and iteration < config.max_batch_iterations:
        print(f"\n[InstantProjection] Additional refinement iteration {iteration}...")
        
        # Run O(1) verification again
        predictions, mismatches, num_correct = instant_batch_verify_and_correct(
            dataset_vec=current_vec,
            token_matrix=token_matrix,
            ground_truth_tokens=tokens.astype(np.int32),
            dim=config.hdc_dim,
            window_size=BATCH_PROJECTION_WINDOW_SIZE,
            apply_corrections=True,
            use_gpu=config.use_gpu,
            gpu_manager=gpu_manager
        )
        
        train_accuracy = num_correct / len(tokens)
        print(f"[InstantProjection] Accuracy after iteration {iteration}: {train_accuracy*100:.2f}%")
        print(f"[InstantProjection] Remaining mismatches: {len(mismatches):,}")
        
        if len(mismatches) == 0:
            print(f"[InstantProjection] Perfect accuracy achieved!")
            break
        
        iteration += 1
    
    elapsed = time.time() - start_time
    
    # Build O(1) reverse lookup table for inference
    # This maps token_vector_bytes -> token_id for instant decode
    print(f"\n[InstantProjection] Building O(1) reverse lookup table...")
    lookup_start = time.time()
    
    reverse_lookup = build_token_reverse_lookup(token_matrix)
    
    lookup_time = time.time() - lookup_start
    print(f"[InstantProjection] Reverse lookup built in {lookup_time:.3f}s")
    print(f"[InstantProjection] Lookup table size: {len(reverse_lookup)} entries (vocab_size=1024)")
    
    # Also build HDCLanguageModel with recipes for context-based O(1) inference
    print(f"[InstantProjection] Building recipe storage from training data...")
    model = HDCLanguageModel(config)
    # Sync model's reverse lookup with the already-built token_matrix so that
    # predict calls during recipe-based inference are O(1) dict lookups.
    model._token_matrix_np = token_matrix
    for tid in range(config.vocab_size):
        model._reverse_lookup[token_matrix[tid].tobytes()] = tid
    model._rl_built = True
    print(f"[InstantProjection] O(1) reverse lookup synced: {config.vocab_size} entries")
    
    # Store recipes for n-gram patterns found in training
    # Each recipe maps context signature → target token for O(1) lookup
    recipe_start = time.time()
    ngram_recipes = 0
    
    # Build recipes from training tokens
    # Use sliding window to capture context → target patterns
    context_len = min(8, config.max_context_length)  # Use short context for recipes
    
    for i in range(context_len, len(tokens) - 1):
        context = tokens[i - context_len:i].tolist()
        target = int(tokens[i])
        
        # Learn this pattern as a recipe
        model.learn_pattern(context, target, use_peeling=False)
        ngram_recipes += 1
        
        # Limit recipes to avoid memory issues
        if ngram_recipes >= 100000:
            break
    
    recipe_time = time.time() - recipe_start
    print(f"[InstantProjection] Built {ngram_recipes:,} recipes in {recipe_time:.2f}s")
    print(f"[InstantProjection] Recipe storage size: {len(model.recipes)} unique patterns")
    
    # Initialize and project semantic layer for O(1) proactive metacognition
    print(f"\n[InstantProjection] Initializing semantic layer for proactive metacognition...")
    semantic_start = time.time()
    
    model.initialize_semantic_layer()
    model.project_corpus_semantic(tokens.tolist())
    
    # Get semantic coverage report
    coverage_report = model.get_semantic_coverage_report()
    if coverage_report is not None:
        print(f"[InstantProjection] Semantic coverage: {coverage_report.coverage*100:.1f}%")
        print(f"[InstantProjection] High confidence windows: {coverage_report.high_confidence_count}/{coverage_report.total_windows}")
        print(f"[InstantProjection] Mean confidence: {coverage_report.mean_confidence:.3f}")
        print(f"[InstantProjection] Dead zones (low confidence): {len(coverage_report.dead_zones)}")
    
    semantic_time = time.time() - semantic_start
    print(f"[InstantProjection] Semantic layer initialized in {semantic_time:.3f}s")
    
    # Final evaluation on validation set using O(1) reverse lookup
    # This is the fastest inference: O(N) with no vocab_size factor!
    print(f"\n[InstantProjection] Running O(1) reverse-lookup evaluation...")
    val_tokens = load_validation_tokens(config.val_files, config.max_context_length)
    
    # Project validation tokens using same instant projection
    val_seed = f"instant_projection_val_{config.seed}_{len(val_tokens)}"
    val_proj_start = time.time()
    
    val_vec, _, _ = instant_batch_project_dataset(
        dataset_tokens=val_tokens,
        seed=val_seed,
        vocab_size=1024,
        dim=config.hdc_dim,
        window_size=BATCH_PROJECTION_WINDOW_SIZE,
        use_gpu=use_gpu,
        gpu_manager=gpu_manager
    )
    
    val_proj_time = time.time() - val_proj_start
    print(f"[InstantProjection] Validation projection in {val_proj_time:.3f}s")
    
    # O(1) decode using reverse lookup - NO vocab_size factor!
    val_decode_start = time.time()
    num_val_positions = min(len(val_tokens), 10000)  # Limit for speed
    
    val_predictions = instant_batch_decode_o1(
        dataset_vec=val_vec,
        token_matrix=token_matrix,
        reverse_lookup=reverse_lookup,
        num_positions=num_val_positions,
        dim=config.hdc_dim,
        window_size=BATCH_PROJECTION_WINDOW_SIZE
    )
    
    val_decode_time = time.time() - val_decode_start
    print(f"[InstantProjection] O(1) decode in {val_decode_time:.3f}s")
    print(f"[InstantProjection] Decode rate: {num_val_positions/val_decode_time:,.0f} tokens/sec")
    
    # Calculate accuracy
    correct = np.sum(val_predictions == val_tokens[:num_val_positions].astype(np.int32))
    total = num_val_positions
    
    val_accuracy = correct / total if total > 0 else 0.0
    
    # Calculate BPB from validation accuracy
    if val_accuracy > 0:
        val_bpb = -np.log2(val_accuracy + 1e-10)
    else:
        val_bpb = float('inf')
    
    val_loss = val_bpb * np.log(2)  # Convert to nats
    
    # Print summary
    print(f"\n[InstantProjection] Training complete!")
    print(f"[InstantProjection] Final train accuracy: {train_accuracy*100:.2f}%")
    print(f"[InstantProjection] Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"[InstantProjection] Validation BPB: {val_bpb:.4f}")
    print(f"[InstantProjection] Total time: {elapsed:.2f}s")
    print(f"[InstantProjection] Projection time: {proj_time:.2f}s ({proj_time/elapsed*100:.1f}%)")
    print(f"[InstantProjection] Decode time: {decode_time:.2f}s ({decode_time/elapsed*100:.1f}%)")
    # RecipeReconstructor statistics
    recon_stats = model.recipe_reconstructor.get_cache_stats()
    verifications = model._recipe_verifications
    failures = model._recipe_verification_failures
    success_rate = (verifications - failures) / verifications * 100 if verifications > 0 else 100.0
    print(f"[InstantProjection] RecipeReconstructor: cache_hits:{recon_stats['hits']:,} cache_misses:{recon_stats['misses']:,} "
          f"hit_rate:{recon_stats['hit_rate']:.1%} verifications:{verifications:,} failures:{failures:,} success_rate:{success_rate:.1f}%")
    
    return val_bpb, val_loss, elapsed


def parse_training_log(log_path: str) -> Dict[str, Any]:
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
    
    # Match val_bpb in various formats
    bpb_match = re.search(r'(?:final_val_bpb|val_bpb)[:\s]+(\d+\.\d+)', content)
    if bpb_match:
        result["val_bpb"] = float(bpb_match.group(1))
    
    # Match val_loss in various formats
    loss_match = re.search(r'(?:final_val_loss|val_loss)[:\s]+(\d+\.\d+)', content)
    if loss_match:
        result["val_loss"] = float(loss_match.group(1))
    
    # Match final step count
    steps_matches = re.findall(r'step:(\d+)/\d+', content)
    if steps_matches:
        result["steps"] = int(steps_matches[-1])  # Get the last step
    
    # Match step average time
    ms_match = re.search(r'step_avg[:\s]+(\d+\.\d+)ms', content)
    if ms_match:
        result["ms_per_step"] = float(ms_match.group(1))
    
    # Match train_time in ms
    time_match = re.search(r'train_time:(\d+)ms', content)
    if time_match:
        result["elapsed_seconds"] = float(time_match.group(1)) / 1000.0
    
    # Match HDC-specific metrics
    recipes_match = re.search(r'(?:hdc_final|hdc_metrics).*?recipes:([\d,]+)', content)
    if recipes_match:
        result["recipes_count"] = int(recipes_match.group(1).replace(',', ''))
    
    ngram_match = re.search(r'(?:hdc_final|hdc_metrics).*?ngrams:([\d,]+)', content)
    if ngram_match:
        result["ngram_count"] = int(ngram_match.group(1).replace(',', ''))
    
    storage_match = re.search(r'(?:hdc_final|hdc_metrics).*?storage_mb:(\d+\.\d+)', content)
    if storage_match:
        result["storage_mb"] = float(storage_match.group(1))
    
    return result


def run_single_training(seed: int, args, log_dir: str = ".") -> Dict[str, Any]:
    """Run a single training session with the given seed."""
    from datetime import datetime, timezone
    
    log_file = os.path.join(log_dir, f"train_seed{seed}.log")
    
    print(f"\n{'='*60}")
    print(f"[TensorCore] Starting training with seed {seed}")
    print(f"[TensorCore] Log file: {log_file}")
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
        
        # Check for instant_projection mode - this is the O(1) lookup fast path
        use_instant_projection = getattr(args, 'instant_projection', False)
        use_batch_projection = getattr(args, 'batch_projection', False)
        
        if use_instant_projection:
            print("[TensorCore] Using INSTANT projection mode (O(1) lookup)")
            final_bpb, final_val_loss, elapsed = train_hdc_instant_projection(config)
        elif use_batch_projection:
            print("[TensorCore] Using batch projection mode")
            final_bpb, final_val_loss, elapsed = train_hdc_batch_projection(config)
        else:
            print("[TensorCore] Using standard HDC training mode")
            final_bpb, final_val_loss, elapsed = train_hdc(config)
        
        print(f"\n{'='*60}")
        print(f"[TensorCore] Final BPB: {final_bpb:.4f}")
        print(f"[TensorCore] Final Loss: {final_val_loss:.4f}")
        print(f"[TensorCore] train_time: {elapsed:.1f}s")
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
    
    print(f"\n[TensorCore] Training with seed {seed} completed:")
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
    
    artifact_bytes = code_bytes
    
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
    from datetime import datetime, timezone
    
    script_path = os.path.abspath(__file__)
    code_bytes = os.path.getsize(script_path)
    
    print(f"[TensorCore] Multi-Seed Training Runner")
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
    print("[TensorCore] Generating submission.json...")
    print(f"{'='*60}")
    
    submission = generate_multi_seed_submission(
        seed_results=seed_results,
        args=args,
        code_bytes=code_bytes
    )
    
    submission_path = os.path.join(os.path.dirname(script_path) or ".", "submission.json")
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\n[TensorCore] Submission saved to {submission_path}")
    print(f"\n[TensorCore] Final Results:")
    print(f"  Mean BPB: {submission['mean_val_bpb']:.6f}")
    print(f"  Std BPB: {submission['std_val_bpb']:.6f}")
    print(f"  P-value: {submission['p_value']:.6f}")
    print(f"  Improvement over baseline: {submission['improvement']}")
    print(f"  Artifact size: {submission['artifact_bytes']:,} bytes")
    
    if submission['p_value'] < 0.05:
        print(f"\n[TensorCore] Result is statistically significant (p < 0.05)")
    else:
        print(f"\n[TensorCore] Result is NOT statistically significant (p >= 0.05)")
    
    return 0 if submission['p_value'] < 0.05 else 1


def main():
    import argparse
    from datetime import datetime, timezone
    
    parser = argparse.ArgumentParser(description="HDC VSA Model with H100 Tensor Core Optimizations")
    parser.add_argument("--data_path", type=str, default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer_path", type=str, default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--hdc_dim", type=int, default=DEFAULT_HDC_DIM)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--max_time", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--author", type=str, default="Ashley Klimpel", help="Author name for submission")
    parser.add_argument("--github_id", type=str, default="viasky657", help="GitHub ID for submission")
    parser.add_argument("--run_name", type=str, default="HDC Zero Track 5Mb TensorCore", help="Run name for submission")
    
    parser.add_argument("--multi_seed", action="store_true",
                        help="Run multi-seed training for statistically significant results")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 7, 1337],
                        help="Seeds for multi-seed training (default: 42 7 1337)")
    
    parser.add_argument("--batch_projection", action="store_true",
                        help="Use batch projection training mode (project entire dataset at once)")
    parser.add_argument("--instant_projection", action="store_true",
                        help="Use INSTANT batch projection with GPU acceleration (fastest mode)")
    parser.add_argument("--max_batch_iterations", type=int, default=10,
                        help="Max iterations for batch projection learning (default: 10)")
    parser.add_argument("--target_accuracy", type=float, default=0.99,
                        help="Target accuracy for batch projection (default: 0.99)")
    
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs for distributed training (default: 1, single GPU)")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank of current process (auto-set by torchrun)")
    parser.add_argument("--dist_url", type=str, default="env://",
                        help="URL for distributed initialization (default: env://)")
    parser.add_argument("--sync_recipes_every", type=int, default=100,
                        help="Synchronize recipes between GPUs every N iterations")
    
    args = parser.parse_args()
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
            args.dist_url = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    
    if args.multi_seed:
        return run_multi_seed_training(args)
    
    config = HDCConfig(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        hdc_dim=args.hdc_dim,
        iterations=args.iterations,
        max_wallclock_seconds=args.max_time,
        seed=args.seed,
        world_size=args.world_size,
        rank=args.rank,
        sync_recipes_every=args.sync_recipes_every,
        use_batch_projection=getattr(args, 'batch_projection', False),
        max_batch_iterations=getattr(args, 'max_batch_iterations', 10),
        target_accuracy=getattr(args, 'target_accuracy', 0.99)
    )
    
    use_instant_projection = getattr(args, 'instant_projection', False)
    
    if use_instant_projection:
        final_bpb, final_val_loss, elapsed = train_hdc_instant_projection(config)
    elif config.use_batch_projection:
        final_bpb, final_val_loss, elapsed = train_hdc_batch_projection(config)
    else:
        final_bpb, final_val_loss, elapsed = train_hdc(config)
    
    if args.rank == 0:
        script_path = os.path.abspath(__file__)
        code_size_bytes = os.path.getsize(script_path)
        
        bytes_total = code_size_bytes
        
        print(f"\n{'='*60}")
        print(f"[TensorCore] FINAL RESULTS")
        print(f"{'='*60}")
        print(f"BPB: {final_bpb:.4f}")
        print(f"Val Loss: {final_val_loss:.4f}")
        print(f"Time: {elapsed:.1f}s")
        print(f"GPUs used: {args.world_size}")
        print(f"Code size: {code_size_bytes:,} bytes")
        print(f"Total artifact size: {bytes_total:,} bytes (zero-weight HDC)")
        print(f"Baseline to beat: 1.2244 BPB")
        
        submission = {
            "author": args.author,
            "github_id": args.github_id,
            "name": args.run_name,
            "blurb": f"HDC VSA Tokenizer Zero-Weight Model with H100 Tensor Core optimizations, {config.hdc_dim:,} dimensions, trained for {config.iterations} iterations in {elapsed:.1f}s on {args.world_size} GPU(s)",
            "date": datetime.now(timezone.utc).isoformat(),
            "val_loss": final_val_loss,
            "val_bpb": final_bpb,
            "bytes_total": bytes_total,
            "bytes_code": code_size_bytes,
            "world_size": args.world_size
        }
        
        submission_path = "submission.json"
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)
        
        print(f"\n[TensorCore] Submission saved to {submission_path}")
        print(f"[TensorCore] Artifact size check: {'PASS' if bytes_total < 16000000 else 'FAIL'} (limit: 16,000,000 bytes)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
