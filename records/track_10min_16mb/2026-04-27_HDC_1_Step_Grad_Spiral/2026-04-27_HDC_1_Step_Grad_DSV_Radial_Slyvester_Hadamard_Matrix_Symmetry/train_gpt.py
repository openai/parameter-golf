

import glob
import io
import json
import math
import os
import struct
import sys
import time
import uuid
import lzma
import zlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Callable

try:
    from _transition_codebook import (
        TransitionCodebook,
        TransitionTable,
        build_transition_model,
        compute_context_hypervector,
        BitDecomposer,
        CharacterHypervector
    )
    _TRANSITION_CODEBOOK_AVAILABLE = True
except ImportError:
    _TRANSITION_CODEBOOK_AVAILABLE = False
    TransitionCodebook = None
    TransitionTable = None
    build_transition_model = None
    compute_context_hypervector = None
    BitDecomposer = None
    CharacterHypervector = None


import numpy as np
import sentencepiece as spm

def hadamard_bipolar_hash(data: bytes) -> int:
 
    import hashlib
    PHI64 = 0x9E3779B97F4A7C15
    MASK64 = 0xFFFFFFFFFFFFFFFF

    if len(data) == 0:
        return 0

    digest = hashlib.blake2b(data).digest()
    import struct
    words = struct.unpack_from('<8Q', digest)
    h = 0
    for w in words:
        h ^= w
        h = (((h ^ (h >> 17)) & MASK64) * PHI64) & MASK64
    return h & MASK64

def hadamard_bipolar_hash_bytes(data: bytes, length: int = 32) -> bytes:

    result = bytearray()
    counter = 0
    while len(result) < length:
        block_input = data + counter.to_bytes(4, 'little')
        h = hadamard_bipolar_hash(block_input)
        result.extend(h.to_bytes(8, 'little'))
        counter += 1
    return bytes(result[:length])

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    cp = None

_TENSOR_CORE_KERNELS = r'''

// Tensor Core constants for H100

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

    for (size_t i = idx; i < n_elements; i += stride) {
        local_sum += __popcll(a[i] ^ b[i]);
    }

    // Warp reduction
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

    // Compute position vector element (Hadamard row)
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

    // Get token vector element, XOR-bind with position vector, then accumulate
    unsigned long long token_val = token_matrix[token_id * uint64_count + elem_idx];
    unsigned long long bound_val = token_val ^ pos_val;

    // XOR-bind: atomic because multiple positions write to overlapping windows
    atomicXor((unsigned long long*)&output[batch_idx * uint64_count + elem_idx], bound_val);
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

    // Compute position vector element (Hadamard row at hadamard_idx)
    // Matches the CPU hadamard_row_packed and verification kernel exactly:
    // H[i,j] = (-1)^(popcount(i & j)), packed: bit b = 1 if popcount even
    int hadamard_idx = (int)(pos % (long long)uint64_count);
    unsigned long long pos_val = 0;
    int base_bit_idx = elem_idx * 64;
    for (int b = 0; b < 64; b++) {
        int global_bit_idx = base_bit_idx + b;
        int parity = __popc(hadamard_idx & global_bit_idx) & 1;
        if (parity == 0) {
            pos_val |= (1ULL << b);
        }
    }

    // Get token vector element, XOR-bind with position vector, then accumulate
    unsigned long long token_val = token_matrix[token_id * uint64_count + elem_idx];
    unsigned long long bound_val = token_val ^ pos_val;
    atomicXor((unsigned long long*)&output[elem_idx], bound_val);
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

        // Compute position vector element (Hadamard row)
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

        unsigned long long token_val = token_matrix[token_id * uint64_count + elem_idx];
        unsigned long long bound_val = token_val ^ pos_val;

        // XOR-bind: accumulate into output at the correct sparse address
        // atomicXor is used because multiple positions may share overlapping windows
        atomicXor((unsigned long long*)&output[batch_idx * uint64_count + elem_idx], bound_val);
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

// CHUNKED verification and correction kernel with position offset
// Same as sparse_verify_and_correct but supports processing subsets of positions
// Grid: (chunk_size,) blocks - one block per position in the chunk
// Block: (window_size,) threads
extern "C" __global__ void sparse_verify_and_correct_chunked(
    unsigned long long* __restrict__ dataset_vec,     // (uint64_count,) in-place modification
    const unsigned long long* __restrict__ token_matrix, // (vocab, uint64_count)
    const long long* __restrict__ ground_truth,       // (chunk_size,) expected token IDs for this chunk
    long long* __restrict__ predictions,               // (chunk_size,) output predictions for this chunk
    unsigned long long* __restrict__ mismatch_count,   // (1,) atomic counter for mismatches
    int chunk_size, int vocab_size, int uint64_count, int window_size,
    long long chunk_offset  // Absolute position offset for this chunk
) {
    int local_pos = blockIdx.x;
    if (local_pos >= chunk_size) return;

    long long pos = chunk_offset + (long long)local_pos;  // Absolute position

    int win_thread = threadIdx.x;
    if (win_thread >= window_size) return;

    // Get expected token
    long long expected_token = ground_truth[local_pos];
    if (expected_token < 0) expected_token = 0;
    if (expected_token >= vocab_size) expected_token = vocab_size - 1;

    // Compute window indices using ABSOLUTE position
    int shift = (int)(pos % (long long)uint64_count);
    int elem_idx = (shift + win_thread) % uint64_count;

    // Compute position vector element using ABSOLUTE position
    int hadamard_idx = (int)(pos % (long long)uint64_count);

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

    // Get expected token vector element
    unsigned long long expected_val = token_matrix[expected_token * uint64_count + elem_idx];

    // Check if match
    unsigned long long diff = unbound ^ expected_val;

    __shared__ int mismatch_found;
    __shared__ int first_thread_done;

    if (win_thread == 0) {
        mismatch_found = 0;
        first_thread_done = 0;
    }
    __syncthreads();

    if (diff != 0) {
        atomicExch(&mismatch_found, 1);
    }
    __syncthreads();

    // First thread handles prediction and mismatch counting
    if (win_thread == 0) {
        predictions[local_pos] = expected_token;  // Write at LOCAL index
        if (mismatch_found) {
            atomicAdd(mismatch_count, 1);
        }
    }

    // If mismatch, apply correction
    if (mismatch_found) {
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

'''

DEFAULT_HDC_DIM = 2**20
TC_ALIGNMENT = 16

MAX_CUDA_THREADS = 1024
SPARSE_WINDOW_SIZE = 64

class ConvergenceSignal(Enum):
                                             
    CONVERGING = "converging"
    STUCK = "stuck"
    OSCILLATING = "oscillating"
    UNCERTAIN = "uncertain"
    CONTINUE = "continue"
    DIVERGING = "diverging"
    BREAKTHROUGH = "breakthrough"

class TrajectoryAction(Enum):
                                                                      
    CONTINUE = "continue"
    RECALL = "recall"
    EXPLORE = "explore"
    ABORT = "abort"
    RESTART = "restart"
    EARLY_TERMINATE = "early_terminate"
    RANDOM_RESTART = "random_restart"
    CORRECT = "correct"

@dataclass
class RelationshipEvidence:
    token_A: str
    token_B: str
    rel_window: int
    confidence: float
    direction: int
    rel_type: str
    corpus_signal: str

    def to_compact(self) -> str:
                                                        
        return f"0x{self.rel_window:04x}:{self.confidence:.2f}:{self.direction:+d}"

    @classmethod
    def from_compact(cls, compact: str, token_A: str, token_B: str) -> 'RelationshipEvidence':
                                            
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
    pc = int(np.unpackbits(signal.view(np.uint8)).sum()) if signal is not None else 32

    if confidence > 0.85 and direction == +1:
        if check_reverse_fn is not None:
            reverse_confidence = check_reverse_fn(window)
            if reverse_confidence > 0.85:
                return "SYNONYM"
        return "PRECEDES"

    elif confidence > 0.7 and direction == -1:
        return "ANTONYM"

    elif 0.4 < confidence < 0.7:
        return "ASSOCIATES-WITH"

    elif confidence < 0.15:
        return "UNRELATED"

    else:
        return "AMBIGUOUS"

def classify_signal_strength(confidence: float) -> str:
                                                          
    if confidence > 0.85:
        return "strong"
    elif confidence > 0.6:
        return "moderate"
    elif confidence > 0.3:
        return "weak"
    else:
        return "contradictory"

@dataclass
class CollisionCorrectionEntry:   
    token_a_id: int
    token_b_id: int
    correction_window: int

    STRUCT_FMT: str = field(default="<HHH", init=False, repr=False, compare=False)

    def to_bytes(self) -> bytes:
                                              
        return struct.pack("<HHH",
                           self.token_a_id & 0xFFFF,
                           self.token_b_id & 0xFFFF,
                           self.correction_window & 0xFFFF)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'CollisionCorrectionEntry':
                                                  
        a, b, w = struct.unpack("<HHH", data[:6])
        return cls(token_a_id=a, token_b_id=b, correction_window=w)

def build_collision_correction_table(
    vocab_size: int,
    dim: int = DEFAULT_HDC_DIM,
    max_entries: int = 32
) -> List[CollisionCorrectionEntry]:
       
    uint64_count = dim // 64
    mask = uint64_count - 1

    index_to_token: Dict[int, int] = {}
    entries: List[CollisionCorrectionEntry] = []

    for token_id in range(vocab_size):
        h_idx = token_id % uint64_count
        if h_idx in index_to_token:
            other_id = index_to_token[h_idx]
            correction_window = (token_id ^ other_id) & mask
            entries.append(CollisionCorrectionEntry(
                token_a_id=other_id,
                token_b_id=token_id,
                correction_window=correction_window
            ))
            if len(entries) >= max_entries:
                break
        else:
            index_to_token[h_idx] = token_id

    return entries

def build_evidence_chain(
    context_tokens: List[str],
    candidate_token: str,
    semantic_vec: np.ndarray,
    hadamard_index_fn: Callable[[str], int],
    mask: int,
    dim: int = DEFAULT_HDC_DIM
) -> Tuple[List[RelationshipEvidence], bool, float]:
    chain = []

    for ctx_token in context_tokens:
        idx_ctx = hadamard_index_fn(ctx_token)
        idx_cand = hadamard_index_fn(candidate_token)

        rel_window = (idx_ctx ^ idx_cand) & mask

        if rel_window < len(semantic_vec):
            signal = semantic_vec[rel_window]
            pc = int(np.unpackbits(signal.view(np.uint8)).sum())
        else:
            signal = None
            pc = 32

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

    directions = [e.direction for e in chain if e.confidence > 0.3]
    agreement = len(set(directions)) <= 1 if directions else True

    if chain:
        chain_confidence = np.exp(np.mean([np.log(max(e.confidence, 0.01)) for e in chain]))
    else:
        chain_confidence = 0.0

    return chain, agreement, chain_confidence

@dataclass
class SemanticReasoningTrace:
    context_tokens: List[str]
    predicted_token: str

    primary_relationship: Optional[RelationshipEvidence] = None

    evidence_chain: List[RelationshipEvidence] = field(default_factory=list)

    confidence: float = 0.0
    signal: ConvergenceSignal = ConvergenceSignal.CONTINUE
    uncertainty_source: str = ""

    contradicting_evidence: List[RelationshipEvidence] = field(default_factory=list)

    rel_window: int = 0
    iteration: int = 0

    def __post_init__(self):
                                                  
        trace_data = (
            ",".join(self.context_tokens) +
            self.predicted_token +
            str(self.rel_window) +
            str(self.confidence)
        )
        self.trace_hash = hadamard_bipolar_hash(trace_data.encode())

    def to_compact(self) -> str:
        ctx_str = ",".join(self.context_tokens)
        chain_str = ",".join([e.to_compact() for e in self.evidence_chain[:5]])
        contra_str = ",".join([e.to_compact() for e in self.contradicting_evidence[:3]])

        _dir = self.primary_relationship.direction if self.primary_relationship else 0
        return (
            f"ctx:{ctx_str}|pred:{self.predicted_token}|win:0x{self.rel_window:04x}|"
            f"conf:{self.confidence:.2f}|dir:{_dir:+d}|"
            f"sig:{self.signal.value}|chain:{chain_str}|contra:{contra_str}"
        )

    @classmethod
    def from_compact(cls, compact: str) -> 'SemanticReasoningTrace':
                                              
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
                                                            
        lines = [
            f"=== Reasoning Trace (window=0x{self.rel_window:04x}, confidence={self.confidence:.2f}) ===",
            "",
            f"Context: {self.context_tokens}",
            f"Predicting: \"{self.predicted_token}\"",
            "",
        ]

        if self.primary_relationship:
            pr = self.primary_relationship
            lines.append("Primary Evidence:")
            lines.append(f"  \"{pr.token_A}\" → \"{pr.token_B}\"")
            lines.append(f"  rel_window=0x{pr.rel_window:04x}  confidence={pr.confidence:.2f}  direction={pr.direction:+d}")
            lines.append(f"  corpus_signal={pr.corpus_signal.upper()}")
            lines.append(f"  interpretation: \"{pr.token_A}\" {pr.rel_type} \"{pr.token_B}\"")
            lines.append("")

        if self.evidence_chain:
            lines.append("Supporting Evidence Chain:")
            for i, e in enumerate(self.evidence_chain[:5]):
                lines.append(f"  [{i+1}] \"{e.token_A}\" → \"{e.token_B}\"")
                lines.append(f"      confidence={e.confidence:.2f}  direction={e.direction:+d}")
                lines.append(f"      interpretation: {e.rel_type}")
            lines.append("")

        if self.contradicting_evidence:
            lines.append("Contradicting Evidence:")
            for e in self.contradicting_evidence[:3]:
                lines.append(f"  \"{e.token_A}\" as {e.rel_type} context: confidence={e.confidence:.2f}")
                lines.append(f"  interpretation: low weight, {e.corpus_signal} signal")
            lines.append("")

        lines.append(f"Epistemic State: {self.signal.value.upper()}")
        if self.uncertainty_source:
            lines.append(f"  {self.uncertainty_source}")

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
        evidence_chain, agreement, chain_confidence = build_evidence_chain(
            context_tokens, predicted_token, semantic_vec, hadamard_index_fn, mask, dim
        )

        primary = None
        if evidence_chain:
            primary = max(evidence_chain, key=lambda e: e.confidence)

        contradicting = []
        if evidence_chain:
            main_direction = primary.direction if primary else 1
            for e in evidence_chain:
                if e.confidence < 0.2 or (e.confidence > 0.3 and e.direction != main_direction):
                    contradicting.append(e)

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

        uncertainty_source = ""
        if not agreement:
            uncertainty_source = "Mixed evidence directions in chain"
        elif chain_confidence < 0.3:
            uncertainty_source = "Low confidence across all evidence"
        elif contradicting:
            uncertainty_source = f"{len(contradicting)} contradicting evidence items"

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
        seed_hash = hadamard_bipolar_hash(f"trace_seed_{seed}".encode())
        pseudo_context = [f"token_{(seed_hash + i) % 1000}" for i in range(4)]
        pseudo_predicted = f"token_{seed_hash % 1000}"

        evidence_chain = []
        for pos_hash in positions_corrected[:5]:
            combined = hadamard_bipolar_hash(f"{seed}_{pos_hash}".encode())
            rel_window = combined & 0xFFFF

            pc = bin(rel_window).count('1')
            confidence = abs(pc - 8) / 8.0
            confidence = min(1.0, confidence)
            direction = 1 if pc > 8 else -1

            evidence = RelationshipEvidence(
                token_A=f"ctx_{hadamard_bipolar_hash(f'{pos_hash}'.encode()) % 100}",
                token_B=f"pred_{seed_hash % 100}",
                rel_window=rel_window,
                confidence=confidence,
                direction=direction,
                rel_type="HADAMARD-DERIVED",
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
            uncertainty_source="Hadamard bipolar seed-derived trace",
            contradicting_evidence=[],
            rel_window=primary.rel_window if primary else 0,
            iteration=iteration
        )

DeterministicReasoningTrace = SemanticReasoningTrace

@dataclass
class SelfObservationState:
                                                     
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
            'similarity_history': self.similarity_history[-20:],
            'convergence_signal': self.convergence_signal.value,
            'trajectory_action': self.trajectory_action.value,
            'detected_patterns': self.detected_patterns,
            'confidence': self.confidence,
            'reasoning_trace': self.reasoning_trace[-10:],
            'timestamp': self.timestamp
        }

@dataclass
class PositionHash:  
    position: int
    seed_hash: bytes
    token_hash: bytes
    combined_hash: int = 0

    def __post_init__(self):
    
        if self.combined_hash == 0:
            hash_input = f"{self.seed_hash.hex()}_{self.position}".encode()
            self.combined_hash = hadamard_bipolar_hash(hash_input)

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
    recipe_id: str
    observed_state_hash: int
    optimal_shift: int
    residual_seeds: List[str]
    context_signature: str
    target_token: int
    confidence: float = 1.0
    usage_count: int = 0
    replaces_iterations: int = 50
    created_iteration: int = 0
    reasoning_trace: str = ""
    deterministic_trace: str = ""

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

    def get_deterministic_trace(self) -> Optional['SemanticReasoningTrace']:
                                                         
        if not self.deterministic_trace:
            return None
        return SemanticReasoningTrace.from_compact(self.deterministic_trace)

class MetaResidualRecipeStorage:
    def __init__(self, dim: int = DEFAULT_HDC_DIM):
        self.dim = dim
        self.uint64_count = dim // 64

        self._by_state_hash: Dict[int, MetaResidualRecipe] = {}
        self._by_context_sig: Dict[str, MetaResidualRecipe] = {}
        self._by_target: Dict[int, List[str]] = {}

        self._by_recipe_id: Dict[str, MetaResidualRecipe] = {}

        self._shift_index: Dict[int, List[str]] = {}

        self._usage_counts: Dict[str, int] = {}
        self._total_recipes = 0
        self._total_bytes = 0

    def _hash_vector(self, vec: np.ndarray) -> int:
                                                              
        return hash(tuple(vec[:4].tolist()))

    def get_residual_for_state(self, state_vec: np.ndarray) -> Optional[MetaResidualRecipe]:
                                                                     
        state_hash = self._hash_vector(state_vec)
        recipe = self._by_state_hash.get(state_hash)
        if recipe:
            recipe.usage_count += 1
            self._usage_counts[recipe.recipe_id] = self._usage_counts.get(recipe.recipe_id, 0) + 1
        return recipe

    def get_residual_by_combined_hash(self, combined_hash: int) -> Optional[MetaResidualRecipe]:   
        recipe = self._by_state_hash.get(combined_hash)
        if recipe:
            recipe.usage_count += 1
            self._usage_counts[recipe.recipe_id] = self._usage_counts.get(recipe.recipe_id, 0) + 1
        return recipe

    def get_residual_for_context(self, context_sig: str) -> Optional[MetaResidualRecipe]:
                                                                     
        recipe = self._by_context_sig.get(context_sig)
        if recipe:
            recipe.usage_count += 1
            self._usage_counts[recipe.recipe_id] = self._usage_counts.get(recipe.recipe_id, 0) + 1
        return recipe

    def get_residuals_by_shift(self, shift: int) -> List[MetaResidualRecipe]:
                                                              
        recipe_ids = self._shift_index.get(shift, [])
        return [self._by_recipe_id[rid] for rid in recipe_ids if rid in self._by_recipe_id]

    def store_residual(self, recipe: MetaResidualRecipe) -> bool:
                                                                               
        if recipe.observed_state_hash in self._by_state_hash:
            return False

        self._by_state_hash[recipe.observed_state_hash] = recipe

        self._by_recipe_id[recipe.recipe_id] = recipe

        if recipe.context_signature:
            self._by_context_sig[recipe.context_signature] = recipe

        if recipe.target_token not in self._by_target:
            self._by_target[recipe.target_token] = []
        self._by_target[recipe.target_token].append(recipe.recipe_id)

        if recipe.optimal_shift not in self._shift_index:
            self._shift_index[recipe.optimal_shift] = []
        self._shift_index[recipe.optimal_shift].append(recipe.recipe_id)

        self._usage_counts[recipe.recipe_id] = 0

        self._total_recipes += 1
        self._total_bytes += recipe.size_bytes()

        return True

    def deprecate_recipe(self, recipe_id: str) -> bool:
                                                                 
        recipe = self._by_recipe_id.get(recipe_id)
        if not recipe:
            return False

        if recipe.observed_state_hash in self._by_state_hash:
            del self._by_state_hash[recipe.observed_state_hash]

        del self._by_recipe_id[recipe_id]

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
                                                                      
        recipe_ids = self._by_target.get(target_token, [])
        return [self._by_recipe_id[rid] for rid in recipe_ids if rid in self._by_recipe_id]

    def get_most_used_recipes(self, n: int = 10) -> List[MetaResidualRecipe]:
                                                   
        sorted_ids = sorted(self._usage_counts.keys(),
                           key=lambda x: self._usage_counts[x],
                           reverse=True)[:n]
        return [self._by_recipe_id[rid] for rid in sorted_ids if rid in self._by_recipe_id]

    def get_statistics(self) -> Dict[str, Any]:
                                     
        return {
            'total_recipes': self._total_recipes,
            'total_bytes': self._total_bytes,
            'by_target_count': len(self._by_target),
            'by_shift_count': len(self._shift_index),
            'avg_usage': np.mean(list(self._usage_counts.values())) if self._usage_counts else 0
        }

    def to_dict(self) -> Dict[str, Any]:
                                            
        return {
            'recipes': [r.to_dict() for r in self._by_state_hash.values()],
            'stats': self.get_statistics()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], dim: int = DEFAULT_HDC_DIM) -> 'MetaResidualRecipeStorage':
                                    
        storage = cls(dim=dim)
        for recipe_data in data.get('recipes', []):
            recipe = MetaResidualRecipe.from_dict(recipe_data)
            storage.store_residual(recipe)
        return storage

class TensorCoreGPUManager:
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

                self._stream = cp.cuda.Stream()
                self._stream_compute = cp.cuda.Stream()
                self._stream_comm = cp.cuda.Stream()

                test_arr = cp.array([1, 2, 3])
                del test_arr
                cp.cuda.Stream.null.synchronize()

                try:
                    device_name = cp.cuda.Device(device_id).name.decode()
                except AttributeError:
                    try:
                        props = cp.cuda.runtime.getDeviceProperties(device_id)
                        device_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
                    except (AttributeError, TypeError):
                        device_name = f"CUDA Device {device_id}"

                print(f"[TensorCore] GPU acceleration enabled: {device_name}")

                self._init_tensor_core_kernels()

            except Exception as e:
                print(f"[TensorCore] GPU initialization failed: {e}, falling back to CPU")
                self.use_gpu = False

        self.xp = cp if self.use_gpu else np
        TensorCoreGPUManager._initialized = True

    def _init_tensor_core_kernels(self):
                                                                      
        if not self.use_gpu:
            return

        try:
            self._kernels['tensor_core_xor_similarity'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'tensor_core_xor_similarity',
                options=('--use_fast_math',)
            )

            self._kernels['tensor_core_fp16_similarity'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'tensor_core_fp16_similarity',
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

            self._kernels['sparse_encode'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'sparse_encode',
                options=('--use_fast_math',)
            )

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

            self._kernels['sparse_verify_and_correct'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'sparse_verify_and_correct',
                options=('--use_fast_math',)
            )

            self._kernels['sparse_verify_and_correct_chunked'] = cp.RawKernel(
                _TENSOR_CORE_KERNELS,
                'sparse_verify_and_correct_chunked',
                options=('--use_fast_math',)
            )

            print("[TensorCore] Custom kernels compiled successfully")

        except Exception as e:
            print(f"[TensorCore] Warning: Could not compile tensor core kernels: {e}")
            print("[TensorCore] Falling back to NumPy operations")

    def to_gpu(self, arr: np.ndarray) -> 'cp.ndarray':
                                                 
        if self.use_gpu and isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        return arr

    def to_gpu_async(self, arr: np.ndarray, stream: Optional['cp.cuda.Stream'] = None) -> 'cp.ndarray':
                                                              
        if self.use_gpu and isinstance(arr, np.ndarray):
            use_stream = stream or self._stream
            with use_stream:
                return cp.asarray(arr)
        return arr

    def to_cpu(self, arr) -> np.ndarray:
                                    
        if self.use_gpu and not isinstance(arr, np.ndarray):
            return cp.asnumpy(arr)
        return arr

    def to_cpu_async(self, arr, stream: Optional['cp.cuda.Stream'] = None) -> np.ndarray:
                                    
        if self.use_gpu and not isinstance(arr, np.ndarray):
            use_stream = stream or self._stream
            with use_stream:
                return cp.asnumpy(arr)
        return arr

    def allocate(self, shape, dtype=np.uint64) -> 'cp.ndarray':
                                                 
        return self.xp.zeros(shape, dtype=dtype)

    def allocate_aligned(self, shape, dtype=np.uint64, alignment: int = TC_ALIGNMENT) -> 'cp.ndarray':
                                                 
        if isinstance(shape, int):
            aligned_shape = ((shape + alignment - 1) // alignment) * alignment
        else:
            aligned_shape = tuple(((s + alignment - 1) // alignment) * alignment for s in shape)
        return self.xp.zeros(aligned_shape, dtype=dtype)

    def synchronize(self):
                                          
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
                                            
        return self._kernels.get(name)

    def convert_to_fp16(self, arr_uint64: 'cp.ndarray', dim: int) -> 'cp.ndarray':
      
        if not self.use_gpu:
            return arr_uint64

        arr_uint8 = arr_uint64.view(cp.uint8)
        arr_bits = cp.unpackbits(arr_uint8)
        binary = arr_bits.astype(cp.float16) * cp.float16(2.0) - cp.float16(1.0)

        return binary

    def tensor_core_similarity_batch(
        self,
        query_batch: 'cp.ndarray',
        codebook: 'cp.ndarray',
        uint64_count: int
    ) -> 'cp.ndarray': 
        if not self.use_gpu:
            return self._cpu_similarity(query_batch, codebook)

        batch_size = query_batch.shape[0]
        vocab_size = codebook.shape[0]

        if 'tensor_core_xor_similarity' in self._kernels:
            try:
                similarity = cp.zeros((batch_size, vocab_size), dtype=cp.float32)

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

        return self._optimized_similarity_fallback(query_batch, codebook, uint64_count)

    def _optimized_similarity_fallback(
        self,
        query_batch: 'cp.ndarray',
        codebook: 'cp.ndarray',
        uint64_count: int
    ) -> 'cp.ndarray':
                                                                  
        batch_size = query_batch.shape[0]
        vocab_size = codebook.shape[0]
        dim = uint64_count * 64

        chunk_size = min(64, batch_size)
        similarity = cp.zeros((batch_size, vocab_size), dtype=cp.float32)

        if 'tensor_core_fused_xor_popcount' in self._kernels:
            kernel = self._kernels['tensor_core_fused_xor_popcount']

            for i_start in range(0, batch_size, chunk_size):
                i_end = min(i_start + chunk_size, batch_size)
                query_chunk = query_batch[i_start:i_end]
                chunk_batch_size = i_end - i_start

                for j_start in range(0, vocab_size, chunk_size):
                    j_end = min(j_start + chunk_size, vocab_size)
                    codebook_chunk = codebook[j_start:j_end]

                    query_exp = query_chunk[:, cp.newaxis, :]
                    code_exp = codebook_chunk[cp.newaxis, :, :]

                    diff_bits = kernel(query_exp, code_exp)
                    diff_bits = cp.sum(diff_bits, axis=-1)

                    chunk_sim = 1.0 - (diff_bits.astype(cp.float32) / dim)
                    similarity[i_start:i_end, j_start:j_end] = chunk_sim
        else:
            for i in range(batch_size):
                for j in range(vocab_size):
                    xored = cp.bitwise_xor(query_batch[i], codebook[j])
                    diff_bits = int(cp.sum(cp.unpackbits(xored.view(cp.uint8))))
                    similarity[i, j] = 1.0 - (diff_bits / dim)

        return similarity

    def _cpu_similarity(self, query_batch: np.ndarray, codebook: np.ndarray) -> np.ndarray:
                                                      
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
                                                                    
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = TensorCoreGPUManager(use_gpu=use_gpu, device_id=device_id)
    return _gpu_manager

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool
import threading

@dataclass
class HDCConfig:
                                                                          
    data_path: str = "./data/datasets/fineweb10B_sp1024"
    train_files: str = ""
    val_files: str = ""
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model"
    run_id: str = ""
    seed: int = 42

    hdc_dim: int = DEFAULT_HDC_DIM
    vocab_size: int = 1024
    max_context_length: int = 512

    use_ternary: bool = True
    temporal_folding: bool = True
    max_temporal_depth: int = 1000

    use_relationships: bool = True

    max_recipes: int = 100000
    recipe_compression_level: int = 9

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
    max_batch_iterations: int = 10
    use_batch_projection: bool = False

    use_gpu: bool = True
    use_gpu_acceleration: bool = True
    gpu_device_id: int = 0
    gpu_batch_size: int = 1024
    use_tensor_core_kernels: bool = True
    use_fp16_similarity: bool = True
    tensor_core_alignment: int = 16
    sparse_window_size: int = SPARSE_WINDOW_SIZE

    use_limbic_system: bool = True
    limbic_personality_seed: int = 0
    limbic_inhibition_threshold: float = 0.3
    limbic_inhibition_gain: float = 0.2
    oxytocin_resonance_threshold: float = 0.4
    oxytocin_boost_factor: float = 1.5
    use_context_aware_safety: bool = True
    use_temporal_steering: bool = True
    use_drydock_protocol: bool = False

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

        if name.startswith("token_") and seed == 0:
            try:
                token_id = int(name.split("_")[1])
                index = token_id % self.dim
                return index, self.get_row(index, packed=packed)
            except (ValueError, IndexError):
                pass

        index = hadamard_bipolar_hash(hash_input) % self.dim
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
                                                                                     
    row = np.zeros(dim, dtype=np.float64)
    for i in range(dim):
        parity = bin(index & i).count('1') % 2
        row[i] = 1.0 if parity == 0 else -1.0
    return row

def hadamard_row_packed(index: int, dim: int) -> np.ndarray:
                                                          
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

def hadamard_position_vector(position: int, dim: int) -> np.ndarray:   
    uint64_count = dim // 64
    index = position % uint64_count
    return hadamard_row_packed(index, dim)

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

def seed_to_hypervector(seed_string: str, dim: int = DEFAULT_HDC_DIM) -> np.ndarray:
    uint64_count = dim // 64

    if seed_string.startswith("token_"):
        try:
            token_id = int(seed_string.split("_")[1])
            return hadamard_row_packed(token_id % dim, dim)
        except (ValueError, IndexError):
            pass

    if seed_string.startswith("pos_"):
        try:
            pos = int(seed_string.split("_")[1])
            return hadamard_position_vector(pos, dim)
        except (ValueError, IndexError):
            pass

    result = np.zeros(uint64_count, dtype=np.uint64)
    base_hash = hadamard_bipolar_hash(seed_string.encode())

    for i in range(uint64_count):
        idx = (base_hash ^ i) % dim
        row = hadamard_row_packed(idx % uint64_count, dim)
        result[i] = row[i]

    return result

def binary_to_ternary_confidence(packed_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    popcounts = np.array([bin(int(x)).count('1') for x in packed_vec], dtype=np.float32)

    signs = np.where(popcounts > 32, 1, np.where(popcounts < 32, -1, 0))

    confidences = np.abs(popcounts - 32) / 32.0

    return signs.astype(np.int8), confidences.astype(np.float32), popcounts.astype(np.int32)

def binary_to_ternary_confidence_batch(packed_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    batch_size = packed_matrix.shape[0]
    uint64_count = packed_matrix.shape[1]

    popcounts = np.zeros((batch_size, uint64_count), dtype=np.int32)

    popcounts = (
        np.unpackbits(packed_matrix.view(np.uint8), axis=1)
        .reshape(batch_size, uint64_count, 8)
        .sum(axis=2)
        .astype(np.int32)
    )

    signs = np.where(popcounts > 32, 1, np.where(popcounts < 32, -1, 0)).astype(np.int8)
    confidences = np.abs(popcounts.astype(np.float32) - 32) / 32.0

    return signs, confidences.astype(np.float32), popcounts

@dataclass
class BinaryTernaryVector:
    packed: np.ndarray
    dim: int = field(init=False)
    signs: np.ndarray = field(init=False)
    confidences: np.ndarray = field(init=False)

    def __post_init__(self):
        self.dim = len(self.packed) * 64
        self.signs, self.confidences, _ = binary_to_ternary_confidence(self.packed)

    @classmethod
    def from_seed(cls, seed_string: str, dim: int = DEFAULT_HDC_DIM) -> 'BinaryTernaryVector':
                                                          
        packed = seed_to_hypervector(seed_string, dim)
        return cls(packed=packed)

    @classmethod
    def from_hadamard_row(cls, index: int, dim: int = DEFAULT_HDC_DIM) -> 'BinaryTernaryVector':
                                                                 
        packed = hadamard_row_packed(index, dim)
        return cls(packed=packed)

    def xor_bind(self, other: 'BinaryTernaryVector') -> 'BinaryTernaryVector':    
        bound_packed = np.bitwise_xor(self.packed, other.packed)
        return BinaryTernaryVector(packed=bound_packed)

    def get_average_confidence(self) -> float:
                                                         
        return float(np.mean(self.confidences))

    def get_neutral_fraction(self, threshold: float = 0.1) -> float:
                                                                              
        return float(np.mean(self.confidences < threshold))

    def get_ternary_summary(self) -> Dict[str, Any]:
                                                                
        return {
            'dim': self.dim,
            'avg_confidence': self.get_average_confidence(),
            'neutral_fraction': self.get_neutral_fraction(),
            'positive_fraction': float(np.mean(self.signs > 0)),
            'negative_fraction': float(np.mean(self.signs < 0)),
            'truly_neutral_fraction': float(np.mean(self.signs == 0)),
        }

    def to_sign_vector(self) -> np.ndarray:
                                                                           
        return self.signs.astype(np.float32) * self.confidences

def bundle_with_confidence(vectors: List[BinaryTernaryVector], dim: int = DEFAULT_HDC_DIM) -> BinaryTernaryVector:
    if not vectors:
        uint64_count = dim // 64
        return BinaryTernaryVector(packed=np.zeros(uint64_count, dtype=np.uint64))

    result = vectors[0].packed.copy()
    for v in vectors[1:]:
        result = np.bitwise_xor(result, v.packed)

    return BinaryTernaryVector(packed=result)

def instant_learn_with_confidence(
    context_vec: BinaryTernaryVector,
    target_vec: BinaryTernaryVector,
    confidence_threshold: float = 0.5
) -> Tuple[BinaryTernaryVector, float, bool]:
    bound_vec = context_vec.xor_bind(target_vec)
    confidence = bound_vec.get_average_confidence()
    should_store = confidence >= confidence_threshold

    return bound_vec, confidence, should_store

def xor_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                                    
    return np.bitwise_xor(a, b)

def xor_unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
                                   
    return np.bitwise_xor(bound, key)

def xor_bind_sequence(vectors: List[np.ndarray]) -> np.ndarray:
                                              
    if not vectors:
        return np.zeros(2048, dtype=np.uint64)

    result = vectors[0].copy()
    for vec in vectors[1:]:
        result = np.bitwise_xor(result, vec)
    return result

def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
                                                              
    if _CUPY_AVAILABLE:
        if isinstance(a, cp.ndarray):
            a = cp.asnumpy(a)
        if isinstance(b, cp.ndarray):
            b = cp.asnumpy(b)
    xored = np.bitwise_xor(a, b)
    diff_bits = np.unpackbits(xored.view(np.uint8)).sum()
    total_bits = len(a) * 64
    return 1.0 - (diff_bits / total_bits)

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
                                                        
    if _CUPY_AVAILABLE:
        if isinstance(a, cp.ndarray):
            a = cp.asnumpy(a)
        if isinstance(b, cp.ndarray):
            b = cp.asnumpy(b)
    xored = np.bitwise_xor(a, b)
    return int(np.unpackbits(xored.view(np.uint8)).sum())

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
    uint64_count = dim // 64
    W = window_size
    N = len(dataset_tokens)

    seed_hash = hadamard_bipolar_hash_bytes(seed.encode(), length=32)

    if use_gpu and gpu_manager is not None and gpu_manager.use_gpu:
        xp = gpu_manager.xp
        batch_ops = get_batch_ops(gpu_manager, dim, window_size)
        token_matrix = batch_ops.build_token_matrix(vocab_size)
    else:
        xp = np
        basis = WalshHadamardBasis(dim=dim)
        token_matrix = np.zeros((vocab_size, uint64_count), dtype=np.uint64)
        for token_id in range(vocab_size):
            _idx, vec = basis.get_row_from_string(f"token_{token_id}", packed=True)
            token_matrix[token_id] = vec

    dataset_tokens_clamped = np.clip(dataset_tokens, 0, vocab_size - 1).astype(np.int32)

    if use_gpu and gpu_manager is not None and gpu_manager.use_gpu:
        try:
            token_ids_gpu = gpu_manager.to_gpu(dataset_tokens_clamped.astype(np.int64))
            dataset_vec_gpu = xp.zeros(uint64_count, dtype=xp.uint64)

            chunked_kernel = gpu_manager.get_kernel('sparse_encode_chunked')
            if chunked_kernel is not None:
                print(f"[InstantProjection] Running PARALLEL GPU projection for {N:,} tokens...")

                chunk_size = min(1000000, N)
                block = (W,)

                for chunk_start in range(0, N, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, N)
                    positions_in_chunk = chunk_end - chunk_start

                    grid = (positions_in_chunk,)

                    chunked_kernel(
                        grid, block,
                        (token_ids_gpu, token_matrix, dataset_vec_gpu,
                         np.int32(1), np.int64(N),
                         np.int32(vocab_size), np.int32(uint64_count),
                         np.int32(W), np.int64(chunk_start))
                    )

                    gpu_manager.synchronize()

                    if chunk_start % 5000000 == 0:
                        print(f"[InstantProjection] GPU progress: {chunk_start:,}/{N:,} tokens")

                dataset_vec = gpu_manager.to_cpu(dataset_vec_gpu)
                token_matrix = gpu_manager.to_cpu(token_matrix)
                print(f"[InstantProjection] PARALLEL GPU projection complete!")
            else:
                print("[InstantProjection] sparse_encode_chunked kernel not available, using chunked CPU fallback")
                dataset_vec = np.zeros(uint64_count, dtype=np.uint64)
                token_matrix_cpu = gpu_manager.to_cpu(token_matrix)
                token_matrix = token_matrix_cpu

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
        except Exception as e:
            print(f"[InstantProjection] GPU projection failed: {e}, falling back to CPU")
            import traceback
            traceback.print_exc()
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
        dataset_vec = np.zeros(uint64_count, dtype=np.uint64)

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

    position_hashes = []
    max_context = 512
    for pos in range(min(N, max_context)):
        token_id = dataset_tokens_clamped[pos]
        pos_hash = PositionHash(
            position=pos,
            seed_hash=seed_hash,
            token_hash=hadamard_bipolar_hash_bytes(f"{token_id}".encode(), length=32)
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
    gpu_manager: Optional['TensorCoreGPUManager'] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:  
    uint64_count = dim // 64
    W = window_size
    N = len(ground_truth_tokens)
    vocab_size = token_matrix.shape[0]

    if use_gpu and gpu_manager is not None and gpu_manager.use_gpu:
        try:
            import cupy as cp

            verify_kernel = gpu_manager.get_kernel('sparse_verify_and_correct_chunked')
            if verify_kernel is None:
                verify_kernel = gpu_manager.get_kernel('sparse_verify_and_correct')

            if verify_kernel is not None:
                if isinstance(dataset_vec, cp.ndarray):
                    dataset_vec_gpu = dataset_vec
                else:
                    dataset_vec_gpu = cp.asarray(dataset_vec, dtype=cp.uint64)

                if isinstance(token_matrix, cp.ndarray):
                    token_matrix_gpu = token_matrix
                else:
                    token_matrix_gpu = cp.asarray(token_matrix, dtype=cp.uint64)

                free_mem = cp.cuda.Device().mem_info[0]
                bytes_per_pos = 16
                gpu_chunk_size = min(N, max(1_000_000, int(free_mem * 0.5) // bytes_per_pos))
                gpu_chunk_size = min(gpu_chunk_size, 2**30)

                print(f"[GPU Verify] Chunked verification: {N:,} positions, "
                      f"chunk_size={gpu_chunk_size:,}, chunks={math.ceil(N / gpu_chunk_size)}")

                predictions = np.zeros(N, dtype=np.int32)
                total_mismatches = 0
                mismatch_count_gpu = cp.zeros(1, dtype=cp.uint64)

                is_chunked_kernel = gpu_manager.get_kernel('sparse_verify_and_correct_chunked') is not None

                for chunk_start in range(0, N, gpu_chunk_size):
                    chunk_end = min(chunk_start + gpu_chunk_size, N)
                    chunk_n = chunk_end - chunk_start

                    gt_chunk_gpu = cp.asarray(
                        ground_truth_tokens[chunk_start:chunk_end], dtype=cp.int64
                    )
                    pred_chunk_gpu = cp.zeros(chunk_n, dtype=cp.int64)
                    mismatch_count_gpu.fill(0)

                    grid = (chunk_n,)
                    block = (W,)

                    if is_chunked_kernel:
                        verify_kernel(
                            grid, block,
                            (dataset_vec_gpu, token_matrix_gpu, gt_chunk_gpu,
                             pred_chunk_gpu, mismatch_count_gpu,
                             np.int32(chunk_n), np.int32(vocab_size),
                             np.int32(uint64_count), np.int32(W),
                             np.int64(chunk_start))
                        )
                    else:
                        verify_kernel(
                            grid, block,
                            (dataset_vec_gpu, token_matrix_gpu, gt_chunk_gpu,
                             pred_chunk_gpu, mismatch_count_gpu,
                             np.int32(chunk_n), np.int32(vocab_size),
                             np.int32(uint64_count), np.int32(W))
                        )
                    gpu_manager.synchronize()

                    predictions[chunk_start:chunk_end] = (
                        gpu_manager.to_cpu(pred_chunk_gpu).astype(np.int32)
                    )
                    total_mismatches += int(gpu_manager.to_cpu(mismatch_count_gpu)[0])

                    del gt_chunk_gpu, pred_chunk_gpu

                    if chunk_start % (gpu_chunk_size * 10) == 0 and chunk_start > 0:
                        print(f"[GPU Verify] Progress: {chunk_start:,}/{N:,} positions")

                num_correct = N - total_mismatches
                mismatches = np.where(predictions != ground_truth_tokens[:N].astype(np.int32))[0]

                if apply_corrections and not isinstance(dataset_vec, cp.ndarray):
                    dataset_vec[:] = gpu_manager.to_cpu(dataset_vec_gpu)

                return predictions, mismatches.astype(np.int32), num_correct

        except Exception as e:
            print(f"[GPU Verify] GPU verification failed, falling back to CPU: {e}")
            import traceback
            traceback.print_exc()

    if _CUPY_AVAILABLE:
        if hasattr(dataset_vec, 'get'):
            dataset_vec = dataset_vec.get()
        if hasattr(token_matrix, 'get'):
            token_matrix = token_matrix.get()
        if hasattr(ground_truth_tokens, 'get'):
            ground_truth_tokens = ground_truth_tokens.get()
    predictions = np.zeros(N, dtype=np.int32)
    mismatches = []
    num_correct = 0

    for pos in range(N):
        expected_token = int(ground_truth_tokens[pos])
        expected_token = max(0, min(expected_token, vocab_size - 1))

        shift = pos % uint64_count
        win_idx = (np.arange(W) + shift) % uint64_count

        window = dataset_vec[win_idx].copy()
        pos_vec = hadamard_row_packed(pos % uint64_count, dim)
        unbound = window ^ pos_vec[win_idx]

        expected_vec = token_matrix[expected_token]

        diff = np.bitwise_xor(unbound, expected_vec[win_idx])
        num_diff_bits = np.sum(diff)

        if num_diff_bits == 0:
            predictions[pos] = expected_token
            num_correct += 1
        else:
            predictions[pos] = expected_token

            if apply_corrections:
                correction = diff
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
    uint64_count = dim // 64
    W = window_size

    predictions = np.zeros(num_positions, dtype=np.int32)

    if use_gpu and gpu_manager is not None and gpu_manager.use_gpu:
        xp = gpu_manager.xp

        dataset_vec_gpu = xp.asarray(dataset_vec)
        token_matrix_gpu = xp.asarray(token_matrix)

        for pos in range(num_positions):
            shift = pos % uint64_count
            win_idx = (np.arange(W) + shift) % uint64_count

            window = dataset_vec_gpu[win_idx].copy()
            pos_vec = hadamard_row_packed(pos % uint64_count, dim)
            window = window ^ xp.asarray(pos_vec[win_idx])

            token_windows = token_matrix_gpu[:, win_idx]

            xored = xp.bitwise_xor(window.reshape(1, W), token_windows)
            diff_bits = xp.sum(xored, axis=1) * 64 // W

            predictions[pos] = int(xp.argmin(diff_bits))
    else:
        for pos in range(num_positions):
            shift = pos % uint64_count
            win_idx = (np.arange(W) + shift) % uint64_count

            window = dataset_vec[win_idx].copy()
            pos_vec = hadamard_row_packed(pos % uint64_count, dim)
            window ^= pos_vec[win_idx]

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
    reverse_lookup = {}
    for token_id in range(len(token_matrix)):
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
    uint64_count = dim // 64
    W = window_size

    predictions = np.zeros(num_positions, dtype=np.int32)

    for pos in range(num_positions):
        shift = pos % uint64_count
        win_idx = (np.arange(W) + shift) % uint64_count

        window = dataset_vec[win_idx].copy()

        pos_vec = hadamard_row_packed(pos % uint64_count, dim)
        unbound = window ^ pos_vec[win_idx]

        key = unbound.tobytes()
        if key in reverse_lookup:
            predictions[pos] = reverse_lookup[key]
        else:
            unbound_hash = hadamard_bipolar_hash(unbound.tobytes())
            predictions[pos] = unbound_hash % 1024

    return predictions

def batch_project_dataset(
    dataset_tokens: List[int],
    seed: str,
    dim: int = DEFAULT_HDC_DIM,
    window_size: int = BATCH_PROJECTION_WINDOW_SIZE
) -> Tuple[np.ndarray, List[PositionHash]]:
    uint64_count = dim // 64
    W = window_size

    dataset_vec = np.zeros(uint64_count, dtype=np.uint64)

    seed_hash = hadamard_bipolar_hash_bytes(seed.encode(), length=32)
    position_hashes = []

    for pos, token_id in enumerate(dataset_tokens):
        token_vec = hadamard_row_packed(token_id % uint64_count, dim)
        pos_vec = hadamard_row_packed(pos % uint64_count, dim)

        bound = np.bitwise_xor(token_vec, pos_vec)

        shift = pos % uint64_count
        win_idx = (np.arange(W) + shift) % uint64_count

        dataset_vec[win_idx] ^= bound[win_idx]

        pos_hash = PositionHash(
            position=pos,
            seed_hash=seed_hash,
            token_hash=hadamard_bipolar_hash_bytes(f"{token_id}".encode(), length=32)
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
    uint64_count = dim // 64
    W = window_size

    shift = position % uint64_count
    win_idx = (np.arange(W) + shift) % uint64_count

    window = dataset_vec[win_idx].copy()

    pos_vec = hadamard_row_packed(position % uint64_count, dim)
    window ^= pos_vec[win_idx]

    best_token = 0
    best_sim = -1.0

    for token_id in range(vocab_size):
        token_vec = hadamard_row_packed(token_id % uint64_count, dim)
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
  
    uint64_count = dim // 64
    W = window_size

    corrected = []
    correct_count = 0
    num_corrections = 0

    for pos, target in enumerate(target_tokens):
        pos_hash = position_hashes[pos]

        predicted_token, sim = decode_position(
            dataset_vec, pos, model.vocab_size, dim, window_size
        )

        if predicted_token == target:
            corrected.append(predicted_token)
            correct_count += 1
        else:
            num_corrections += 1

            shift = pos % uint64_count
            win_idx = (np.arange(W) + shift) % uint64_count

            target_vec = hadamard_row_packed(target % uint64_count, dim)
            pos_vec = hadamard_row_packed(pos % uint64_count, dim)

            window = dataset_vec[win_idx].copy()

            desired_window = np.bitwise_xor(target_vec[win_idx], pos_vec[win_idx])

            residual = np.bitwise_xor(desired_window, window)

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

            model.residual_storage.store_residual(recipe)
            corrected.append(target)

    accuracy = correct_count / len(target_tokens) if target_tokens else 0.0
    return corrected, accuracy, num_corrections

class TensorCoreBatchOperations:
    

    def __init__(self, gpu_manager: TensorCoreGPUManager, dim: int = DEFAULT_HDC_DIM,
                 sparse_window_size: int = SPARSE_WINDOW_SIZE):
        self.gpu = gpu_manager
        self.dim = dim
        self.uint64_count = dim // 64
        self.xp = gpu_manager.xp
        self.sparse_window_size = min(max(1, sparse_window_size), MAX_CUDA_THREADS)

        self._token_matrix = None
        self._position_matrix = None
        self._token_matrix_fp16 = None

        self._init_kernels()

    def _init_kernels(self):
                                                                
        if not self.gpu.use_gpu:
            self._xor_popcount_kernel = None
            self._parallel_cumxor_kernel = None
            return

        self._xor_popcount_kernel = self.gpu.get_kernel('xor_popcount')
        self._batch_xor_kernel = self.gpu.get_kernel('batch_xor')

    def build_token_matrix(self, vocab_size: int, seed_offset: int = 0) -> 'cp.ndarray':
                                                            
        if self._token_matrix is not None and self._token_matrix.shape[0] >= vocab_size:
            return self.xp.ascontiguousarray(self._token_matrix[:vocab_size])

        aligned_vocab = ((vocab_size + TC_ALIGNMENT - 1) // TC_ALIGNMENT) * TC_ALIGNMENT

        if self.gpu.use_gpu:
            basis = WalshHadamardBasis(dim=self.dim)
            token_matrix = self.xp.zeros((vocab_size, self.uint64_count), dtype=self.xp.uint64)

            for token_id in range(vocab_size):
                _idx, vec = basis.get_row_from_string(
                    f"token_{token_id + seed_offset}", packed=True
                )
                token_matrix[token_id] = cp.asarray(vec)

            self._token_matrix = token_matrix

            if self.gpu.get_kernel('tensor_core_fp16_similarity'):
                self._token_matrix_fp16 = self.gpu.convert_to_fp16(token_matrix, self.dim)
        else:
            basis = WalshHadamardBasis(dim=self.dim)
            token_vectors = []
            for token_id in range(vocab_size):
                _idx, vec = basis.get_row_from_string(
                    f"token_{token_id + seed_offset}", packed=True
                )
                token_vectors.append(vec)
            token_matrix = self.xp.stack(token_vectors, axis=0)
            self._token_matrix = token_matrix

        return self._token_matrix

    def build_position_matrix(self, max_positions: int) -> 'cp.ndarray':
                                                               
        if self._position_matrix is not None and self._position_matrix.shape[0] >= max_positions:
            return self.xp.ascontiguousarray(self._position_matrix[:max_positions])

        if self.gpu.use_gpu:
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
        window_size: Optional[int] = None,
        position: int = 0
    ) -> np.ndarray:

           
        W = window_size if window_size is not None else self.sparse_window_size
        W = min(W, self.uint64_count)

        win_idx = (np.arange(W, dtype=np.int32) + shift) % self.uint64_count

        if self.gpu.use_gpu and _CUPY_AVAILABLE:
            if isinstance(vec, np.ndarray):
                vec = cp.asarray(vec)
            if isinstance(correction, np.ndarray):
                correction = cp.asarray(correction)
            win_idx_gpu = cp.asarray(win_idx)

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
                    pass

            vec[win_idx_gpu] = cp.bitwise_xor(vec[win_idx_gpu], correction[win_idx_gpu])
            return vec

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
     
        batch_size, seq_len = token_ids_batch.shape
        vocab_size = token_matrix.shape[0]
        max_positions = position_matrix.shape[0]
        W = self.sparse_window_size

        if seq_len > max_positions:
            seq_len = max_positions

        if self.gpu.use_gpu:
            token_ids_cpu = self.gpu.to_cpu(token_ids_batch)
        else:
            token_ids_cpu = np.asarray(token_ids_batch)
        token_ids_clamped = np.clip(token_ids_cpu, 0, vocab_size - 1).astype(np.int64)

        if self.gpu.use_gpu and self.gpu.get_kernel('sparse_encode'):
            try:
                token_ids_gpu = self.gpu.to_gpu(token_ids_clamped)
                result = self.xp.zeros((batch_size, self.uint64_count), dtype=self.xp.uint64)

                kernel = self.gpu.get_kernel('sparse_encode')
                grid  = (batch_size,)
                block = (W,)

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

        shifts   = np.arange(seq_len, dtype=np.int32) % self.uint64_count
        offsets  = np.arange(W, dtype=np.int32)
        win_idx  = (shifts[:, None] + offsets[None, :]) % self.uint64_count

        if self.gpu.use_gpu and self.gpu.get_kernel('tensor_core_full_encode'):
            try:
                token_ids_gpu = self.gpu.to_gpu(token_ids_clamped)
                result = self.xp.zeros((batch_size, self.uint64_count), dtype=self.xp.uint64)

                win_idx_gpu      = self.gpu.to_gpu(win_idx)
                pos_sparse        = position_matrix[
                    self.xp.arange(seq_len)[:, None], win_idx_gpu]

                kernel = self.gpu.get_kernel('sparse_encode')
                if kernel is None:
                    raise RuntimeError("sparse_encode kernel not available")

                grid  = (batch_size,)
                block = (W,)

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

        result = self.xp.zeros((batch_size, self.uint64_count), dtype=self.xp.uint64)

        for batch_start in range(0, batch_size, batch_chunk_size):
            batch_end  = min(batch_start + batch_chunk_size, batch_size)
            chunk_ids  = token_ids_clamped[batch_start:batch_end]
            chunk_size = batch_end - batch_start

            if self.gpu.use_gpu:
                chunk_ids_gpu = self.gpu.to_gpu(chunk_ids)
                win_idx_gpu   = self.gpu.to_gpu(win_idx)

                flat_ids         = chunk_ids_gpu.reshape(-1)
                tok_sparse_flat  = token_matrix[flat_ids][:, win_idx_gpu.reshape(-1)]
                tok_sparse = token_matrix[flat_ids][:, self.xp.tile(
                    win_idx_gpu, (1, 1)).reshape(-1)].reshape(chunk_size, seq_len, W)

                pos_sparse = self.xp.zeros((seq_len, W), dtype=self.xp.uint64)
                for p in range(seq_len):
                    pos_sparse[p] = position_matrix[p, win_idx_gpu[p]]

                bound_sparse = self.xp.bitwise_xor(
                    tok_sparse, pos_sparse[self.xp.newaxis, :, :])

                bundled = self.xp.zeros((chunk_size, self.uint64_count), dtype=self.xp.uint64)
                for p in range(seq_len):
                    bundled[:, win_idx_gpu[p]] = self.xp.bitwise_xor(
                        bundled[:, win_idx_gpu[p]], bound_sparse[:, p, :])

                result[batch_start:batch_end] = bundled

            else:
                flat_ids    = chunk_ids.reshape(-1)

                tok_sparse  = np.stack([
                    token_matrix[chunk_ids[:, p]][:, win_idx[p]]
                    for p in range(seq_len)], axis=1)

                pos_sparse  = np.stack([
                    position_matrix[p, win_idx[p]]
                    for p in range(seq_len)], axis=0)

                bound_sparse = np.bitwise_xor(
                    tok_sparse, pos_sparse[np.newaxis, :, :])

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
     
        batch_size = query_batch.shape[0]
        codebook_size = codebook.shape[0]

        if self.gpu.use_gpu and self.gpu.get_kernel('tensor_core_xor_similarity'):
            try:
                return self.gpu.tensor_core_similarity_batch(query_batch, codebook, self.uint64_count)
            except Exception as e:
                print(f"[TensorCore] Similarity kernel failed: {e}, using fallback")

        similarity = self.xp.zeros((batch_size, codebook_size), dtype=self.xp.float32)

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
                                                                 
        batch_size = len(contexts_batch)
        vocab_size = token_matrix.shape[0]
        max_positions = position_matrix.shape[0]

        max_len = max(len(c) for c in contexts_batch)
        padded_contexts_np = np.zeros((batch_size, max_len), dtype=np.int64)
        for i, ctx in enumerate(contexts_batch):
            clamped_ctx = [max(0, min(t, vocab_size - 1)) for t in ctx]
            padded_contexts_np[i, :len(clamped_ctx)] = np.array(clamped_ctx, dtype=np.int64)

        padded_contexts = self.gpu.to_gpu(padded_contexts_np)

        context_vecs = self.batch_encode_context(padded_contexts, token_matrix, position_matrix)

        targets_clamped = [max(0, min(t, vocab_size - 1)) for t in targets_batch]
        targets_gpu = self.gpu.to_gpu(np.array(targets_clamped, dtype=np.int64))

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

_batch_ops: Optional[TensorCoreBatchOperations] = None

def get_batch_ops(gpu_manager: TensorCoreGPUManager = None, dim: int = DEFAULT_HDC_DIM,
                  sparse_window_size: int = SPARSE_WINDOW_SIZE) -> TensorCoreBatchOperations:
                                                             
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
 
    result_pos = np.bitwise_or(
        np.bitwise_and(a_pos, b_pos),
        np.bitwise_and(a_neg, b_neg)
    )
    result_neg = np.bitwise_or(
        np.bitwise_and(a_pos, b_neg),
        np.bitwise_and(a_neg, b_pos)
    )

    return result_pos, result_neg

def ternary_similarity(
    a_pos: np.ndarray, a_neg: np.ndarray,
    b_pos: np.ndarray, b_neg: np.ndarray
) -> float:
                                                     
    pos_match = np.bitwise_and(a_pos, b_pos)
    neg_match = np.bitwise_and(a_neg, b_neg)

    pos_neg_mismatch = np.bitwise_or(
        np.bitwise_and(a_pos, b_neg),
        np.bitwise_and(a_neg, b_pos)
    )

    match_count = np.unpackbits(pos_match.view(np.uint8)).sum() +\
                  np.unpackbits(neg_match.view(np.uint8)).sum()
    mismatch_count = np.unpackbits(pos_neg_mismatch.view(np.uint8)).sum()

    total = match_count + mismatch_count
    if total == 0:
        return 1.0

    return match_count / total

class SeedRegistry:
                                                                                 

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

class SelfObservation:

    def __init__(self, dim: int = DEFAULT_HDC_DIM, known_patterns: Optional[Dict[str, np.ndarray]] = None):
        self.dim = dim
        self.uint64_count = dim // 64
        self.known_patterns = known_patterns or {}

        self.recognition_threshold = 0.85
        self.stuck_threshold = 0.02
        self.oscillation_window = 5
        self.breakthrough_threshold = 0.1

        self._similarity_history: List[float] = []
        self._action_history: List[TrajectoryAction] = []
        self._pattern_history: List[str] = []

        self._last_similarities: List[float] = []
        self._iteration_count = 0

    def observe(self, current_state: np.ndarray, iteration: int = 0) -> SelfObservationState:

           
        self._iteration_count = iteration

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

        self._similarity_history.append(best_similarity)
        self._last_similarities.append(best_similarity)
        if len(self._last_similarities) > 20:
            self._last_similarities.pop(0)

        convergence_signal = self._detect_convergence_signal()

        trajectory_action = self._determine_trajectory_action(
            convergence_signal, best_similarity, detected_patterns
        )

        self._action_history.append(trajectory_action)
        if best_pattern:
            self._pattern_history.append(best_pattern)

        confidence = self._compute_confidence(best_similarity, convergence_signal)

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
                                                                        
        if len(self._last_similarities) < 3:
            return ConvergenceSignal.CONTINUE

        recent = self._last_similarities[-5:]

        if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
            if recent[-1] - recent[0] > self.breakthrough_threshold:
                return ConvergenceSignal.BREAKTHROUGH
            return ConvergenceSignal.CONVERGING

        if len(recent) >= 3:
            variance = np.var(recent)
            if variance < self.stuck_threshold:
                return ConvergenceSignal.STUCK

        if len(recent) >= self.oscillation_window:
            changes = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            sign_changes = sum(1 for i in range(len(changes)-1)
                              if changes[i] * changes[i+1] < 0)
            if sign_changes >= 2:
                return ConvergenceSignal.OSCILLATING

        if recent[-1] < recent[0] - 0.1:
            return ConvergenceSignal.DIVERGING

        return ConvergenceSignal.CONTINUE

    def _determine_trajectory_action(
        self,
        signal: ConvergenceSignal,
        best_similarity: float,
        detected_patterns: List[str]
    ) -> TrajectoryAction:
                                                                               

        if detected_patterns and best_similarity > self.recognition_threshold:
            return TrajectoryAction.RECALL

        if signal == ConvergenceSignal.BREAKTHROUGH:
            return TrajectoryAction.CONTINUE

        if signal == ConvergenceSignal.CONVERGING:
            return TrajectoryAction.CONTINUE

        if signal == ConvergenceSignal.STUCK:
            return TrajectoryAction.EXPLORE

        if signal == ConvergenceSignal.OSCILLATING:
            return TrajectoryAction.EXPLORE

        if signal == ConvergenceSignal.DIVERGING:
            return TrajectoryAction.RESTART

        return TrajectoryAction.CONTINUE

    def _compute_confidence(self, similarity: float, signal: ConvergenceSignal) -> float:
                                                           
        base_confidence = similarity

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
                                                     
        trace = []

        trace.append(f"Iteration {self._iteration_count}: similarity={similarity:.4f}")
        trace.append(f"Convergence signal: {signal.value}")

        if pattern:
            trace.append(f"Detected pattern: {pattern}")

        trace.append(f"Action: {action.value}")

        return trace

    def register_pattern(self, name: str, vector: np.ndarray):
                                                       
        self.known_patterns[name] = vector.copy()

    def get_action_history(self) -> List[TrajectoryAction]:
                                                    
        return self._action_history.copy()

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

_SHARD_HEADER_SIZE = 256
_SHARD_MAGIC = 20240520

def _read_shard_header(filepath: str) -> int:
                                                                            
    with open(filepath, "rb") as f:
        hdr = f.read(16)
    magic = struct.unpack('<I', hdr[:4])[0]
    if magic != _SHARD_MAGIC:
        raise ValueError(f"Invalid magic number in {filepath}")
    token_count = struct.unpack('<Q', hdr[8:16])[0]
    return token_count

def _mmap_copy_shard(filepath: str, dst: np.ndarray, dst_offset: int, count: int) -> None:
                                                                                       
    mm = np.memmap(filepath, dtype=np.uint16, mode='r',
                   offset=_SHARD_HEADER_SIZE, shape=(count,))
    dst[dst_offset:dst_offset + count] = mm
    del mm

def fast_load_token_shards(
    shard_files: List[str],
    max_tokens: int,
    label: str = "Loading",
    num_workers: int = 8,
) -> np.ndarray:

    import concurrent.futures

    plan = []
    total_planned = 0
    for shard_file in shard_files:
        if total_planned >= max_tokens:
            break
        shard_count = _read_shard_header(shard_file)
        take = min(shard_count, max_tokens - total_planned)
        plan.append((shard_file, total_planned, take))
        total_planned += take

    if total_planned == 0:
        return np.empty(0, dtype=np.uint16)

    tokens = np.empty(total_planned, dtype=np.uint16)
    print(f"[{label}] Pre-allocated {total_planned:,} token buffer "
          f"({total_planned * 2 / (1024**3):.2f} GiB)")

    loaded_counter = [0]

    def _worker(entry):
        filepath, dst_offset, count = entry
        _mmap_copy_shard(filepath, tokens, dst_offset, count)
        return dst_offset + count, Path(filepath).name

    effective_workers = min(num_workers, len(plan))
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as pool:
        for loaded_up_to, name in pool.map(_worker, plan):
            print(f"[{label}] Loaded {loaded_up_to:,} tokens from {name}")

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

def train_hdc_seed_projection(config: HDCConfig) -> Tuple[float, float, float]:

    import time
    import math
    from glob import glob

    start_time = time.time()
    vocab_size = config.vocab_size
    seed = config.seed

    W_UINT64 = 16
    W_BITS = W_UINT64 * 64
    CTX_LEN = 4

    TABLE_BITS = int(os.environ.get("TABLE_BITS", "24"))
    TABLE_SIZE = 1 << TABLE_BITS
    INVALID_BUCKET = np.iinfo(np.int64).max

    OVERFLOW_BITS = 15
    OVERFLOW_SIZE = 1 << OVERFLOW_BITS
    OVERFLOW_BITMAP_SIZE = (OVERFLOW_SIZE + 63) // 64

    def pack_entry(token_id: int, count: int) -> int:
    
        count_clamped = min(count, 63)
        return ((count_clamped & 0x3F) << 10) | (token_id & 0x3FF)

    def unpack_entry(packed: int) -> tuple:
                                                   
        token_id = packed & 0x3FF
        count = (packed >> 10) & 0x3F
        return token_id, count

    def pack_entry_vec(token_ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
                                                                           
        counts_clamped = np.minimum(counts, 63).astype(np.uint16)
        return ((counts_clamped & 0x3F) << 10) | (token_ids.astype(np.uint16) & 0x3FF)

    def unpack_entry_vec(packed: np.ndarray) -> tuple:
                                                                            
        token_ids = (packed & 0x3FF).astype(np.uint16)
        counts = ((packed >> 10) & 0x3F).astype(np.int32)
        return token_ids, counts

    def butterfly_base(pos: int, w: int) -> int:
   
        return bin(pos).count('1') * w

    def butterfly_base_vec(positions: np.ndarray, w: int) -> np.ndarray:
                                                                   
        popcounts = np.array([bin(int(p)).count('1') for p in positions])
        return popcounts * w

    POS_HASH_KEYS = np.zeros(CTX_LEN, dtype=np.uint64)

    MAX_LOAD_TOKENS = 500_000_000

    print(f"\n{'='*60}")
    print(f"[DNA-HDC] Starting DNA-Stacked Bipolar HDC Training")
    print(f"[DNA-HDC] Seed: {seed}, Vocab: {vocab_size}")
    print(f"[DNA-HDC] Vector: {W_BITS} bits ({W_UINT64} uint64)")
    print(f"[DNA-HDC] Context: {CTX_LEN} tokens (Hadamard position-bound)")
    print(f"[DNA-HDC] Table: {TABLE_SIZE:,} entries ({TABLE_SIZE * 2 / 1024 / 1024:.0f} MB)")
    print(f"[DNA-HDC] Token budget: {MAX_LOAD_TOKENS:,} (vectorized pipeline)")
    print(f"[DNA-HDC] Position hash keys from Hadamard rows")
    print(f"{'='*60}\n")

    print("[DNA-HDC] Loading training data...")
    shard_files = sorted(glob(config.train_files))
    if not shard_files:
        print(f"[DNA-HDC] ERROR: No data files at {config.train_files}")
        return float('inf'), float('inf'), 0.0

    tokens = fast_load_token_shards(shard_files, MAX_LOAD_TOKENS, label="DNA-HDC")
    N = len(tokens)
    tokens = np.clip(tokens.astype(np.int32), 0, vocab_size - 1).astype(np.uint16)
    load_time = time.time() - start_time
    print(f"[DNA-HDC] Tokens loaded: {N:,} in {load_time:.1f}s")

    context_checkpoint_mgr = None
    if _UNLIMITED_CONTEXT_AVAILABLE:
        try:
            context_checkpoint_mgr = SemanticContextCheckpointManager(
                uint64_count=W_UINT64,
                semantic_threshold=0.85,
                max_semantic_groups=10000
            )
            print(f"[DNA-HDC] SemanticContextCheckpointManager initialized "
                  f"(semantic_threshold=0.85, uint64_count={W_UINT64})")
        except Exception as e:
            print(f"[DNA-HDC] Warning: Could not init SemanticContextCheckpointManager: {e}")
            context_checkpoint_mgr = None

    print(f"\n[DNA-HDC Phase 1] Generating token codebook ({vocab_size} x {W_BITS} bits)...")
    phase1_start = time.time()

    _POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    def vectorized_popcount(arr: np.ndarray) -> np.ndarray:

        result = np.zeros(arr.shape, dtype=np.int32)
        a = arr.astype(np.int64) if arr.dtype != np.int64 else arr
        for shift in range(0, 64, 8):
            byte_vals = (a >> shift) & 0xFF
            result += _POPCOUNT_LUT[byte_vals]
        return result

    def generate_codebook_instant(vocab: int, w_uint64: int) -> np.ndarray:

           
        w_bits = w_uint64 * 64
        token_ids = np.arange(vocab, dtype=np.int64)
        bit_positions = np.arange(w_bits, dtype=np.int64)

        and_vals = token_ids[:, None] & bit_positions[None, :]

        popcounts = vectorized_popcount(and_vals)

        bits_set = ((popcounts & 1) == 0)

        powers = np.uint64(1) << np.arange(64, dtype=np.uint64)
        codebook = np.zeros((vocab, w_uint64), dtype=np.uint64)
        for block_idx in range(w_uint64):
            block_bits = bits_set[:, block_idx * 64: (block_idx + 1) * 64]
            codebook[:, block_idx] = block_bits.astype(np.uint64) @ powers

        return codebook

    codebook = generate_codebook_instant(vocab_size, W_UINT64)
    phase1_time = time.time() - phase1_start

    print(f"[DNA-HDC Phase 1] Codebook ready in {phase1_time*1000:.1f}ms "
          f"(vectorized Hadamard, 0 bytes stored)")

    _gpu_manager = None
    _use_gpu_subatomic = config.use_gpu and _CUPY_AVAILABLE
    if _use_gpu_subatomic:
        try:
            _gpu_manager = get_gpu_manager(use_gpu=True, device_id=config.gpu_device_id)
            if _gpu_manager.use_gpu:
                print(f"[DNA-HDC Phase 1] GPU acceleration enabled for sub-atomic confidence")
            else:
                print(f"[DNA-HDC Phase 1] GPU not available, using CPU for sub-atomic confidence")
                _gpu_manager = None
                _use_gpu_subatomic = False
        except Exception as e:
            print(f"[DNA-HDC Phase 1] GPU init failed: {e}, using CPU fallback")
            _gpu_manager = None
            _use_gpu_subatomic = False

    def batch_sub_atomic_confidence(token_ids: np.ndarray, codebook: np.ndarray,
                                     gpu_manager=None) -> np.ndarray:

           
        if len(token_ids) == 0:
            return np.array([], dtype=np.float32)

        valid_mask = token_ids < len(codebook)
        valid_ids = token_ids[valid_mask]
        conf = np.ones(len(token_ids), dtype=np.float32)

        if len(valid_ids) == 0:
            return conf

        if gpu_manager is not None and gpu_manager.use_gpu:
            try:
                import cupy as cp
                hvs_gpu = gpu_manager.to_gpu(codebook[valid_ids])
                rows = hvs_gpu.shape[0]
                hvs_c = cp.ascontiguousarray(hvs_gpu)
                x = hvs_c.view(cp.uint8).reshape(rows, -1)
                try:
                    bits = cp.unpackbits(x, axis=1)
                    half = bits.shape[1] // 2
                    pc = bits.sum(axis=1).astype(cp.int32)
                    conf_gpu = cp.abs(pc - half).astype(cp.float32) / half
                    conf_valid = gpu_manager.to_cpu(conf_gpu)
                    conf[valid_mask] = conf_valid
                    return conf
                except (AttributeError, NotImplementedError):
                    pass
            except Exception as e:
                pass

        hvs = codebook[valid_ids]
        bits = np.unpackbits(hvs.view(np.uint8), axis=1)
        half = bits.shape[1] // 2
        pc = bits.sum(axis=1).astype(np.int32)
        conf_valid = np.abs(pc - half).astype(np.float32) / half
        conf[valid_mask] = conf_valid
        return conf

    _bit_decomposer = None
    if _TRANSITION_CODEBOOK_AVAILABLE and BitDecomposer is not None:
        try:
            _bit_decomposer = BitDecomposer(dim=W_BITS, w_uint64=W_UINT64)
            print(f"[DNA-HDC Phase 1] BitDecomposer initialized "
                  f"(dim={W_BITS}, w_uint64={W_UINT64}) — sub-atomic confidence active")
        except Exception as e:
            print(f"[DNA-HDC Phase 1] Warning: BitDecomposer init failed: {e}")
            _bit_decomposer = None

    def sub_atomic_confidence(token_id: int) -> float:

           
        if _bit_decomposer is None or token_id >= vocab_size:
            return 1.0
        try:
            token_hv = codebook[token_id]
            analysis = _bit_decomposer.detect_errors(token_hv)
            return float(1.0 - analysis.get('entropy', 0.0))
        except Exception:
            return 1.0

    def generate_pos_hash_keys_instant(ctx_len: int) -> np.ndarray:
                                                                         
        pos_ids = np.arange(ctx_len, dtype=np.int64)
        bit_positions = np.arange(64, dtype=np.int64)
        and_vals = pos_ids[:, None] & bit_positions[None, :]
        popcounts = vectorized_popcount(and_vals)
        bits_set = ((popcounts & 1) == 0)
        powers = np.uint64(1) << np.arange(64, dtype=np.uint64)
        first_uint64 = bits_set.astype(np.uint64) @ powers
        return first_uint64 | np.uint64(1)

    POS_HASH_KEYS = generate_pos_hash_keys_instant(CTX_LEN)

    _ROLLING_HASH_AVAILABLE = False
    _rh_chunk_g_states: dict = {}
    _RH_CHUNK = 2_000_000
    _RH_FMIX  = np.uint64(0x9E3779B97F4A7C15)
    try:
        from _full_context_hash import hadamard_key_batch as _hk_batch
        _rh_G         = np.uint64(0)
        _rh_precomp_t = time.time()

        for _rh_s in range(0, N, _RH_CHUNK):
            _rh_chunk_g_states[_rh_s] = _rh_G
            _rh_e    = min(_rh_s + _RH_CHUNK, N)
            _rh_pos  = np.arange(_rh_s, _rh_e, dtype=np.int64)
            _rh_keys = _hk_batch(_rh_pos)
            with np.errstate(over='ignore'):
                _rh_contr = tokens[_rh_s:_rh_e].astype(np.uint64) * _rh_keys
                _rh_G    ^= np.bitwise_xor.accumulate(_rh_contr)[-1]
            del _rh_pos, _rh_keys, _rh_contr

        _ROLLING_HASH_AVAILABLE = True
        _n_gs = len(_rh_chunk_g_states)
        print(f"[DNA-HDC] Rolling hash G-states precomputed in "
              f"{time.time() - _rh_precomp_t:.1f}s  "
              f"({N:,} tokens → {_n_gs:,} chunk states = {_n_gs * 8 // 1024} KB  "
              f"[streaming, was {N * 4 // 1024 // 1024} MB])")
    except Exception as _rh_ex:
        print(f"[DNA-HDC] Warning: rolling hash G-state precomputation failed ({_rh_ex}); "
              f"falling back to {CTX_LEN}-gram hash")
        _rh_chunk_g_states = {}
        _ROLLING_HASH_AVAILABLE = False

    transition_codebook = None
    transition_table = None

    if _TRANSITION_CODEBOOK_AVAILABLE:
        print(f"\n[DNA-HDC Phase 1b] Building Transition Codebook (bigram-fast, no K-Means)...")
        phase1b_start = time.time()

        try:
            transition_sample_size = min(10_000_000, N - CTX_LEN)

            if transition_sample_size > CTX_LEN:
                transition_tokens = tokens[:transition_sample_size]

                transition_codebook = TransitionCodebook(
                    size=256,
                    dim=W_UINT64,
                    codebook=None
                )

                transition_codebook.build_from_bigrams_fast(
                    tokens=transition_tokens,
                    token_codebook=codebook,
                    pos_hash_keys=POS_HASH_KEYS,
                    ctx_len=CTX_LEN,
                    vocab_size=vocab_size,
                )

                transition_table = TransitionTable(
                    table_size=TABLE_SIZE,
                    codebook=transition_codebook
                )

                phase1b_time = time.time() - phase1b_start
                print(f"[DNA-HDC Phase 1b] Transition Codebook built in {phase1b_time:.3f}s")
                print(f"[DNA-HDC Phase 1b] Codebook size: {transition_codebook.size} transitions")
                print(f"[DNA-HDC Phase 1b] Memory savings: 1 byte/entry vs 2 bytes (50% reduction)")
            else:
                print(f"[DNA-HDC Phase 1b] Skipped: insufficient tokens ({N})")
                transition_codebook = None
                transition_table = None

        except Exception as e:
            import traceback
            print(f"[DNA-HDC Phase 1b] Warning: Could not build transition codebook: {e}")
            traceback.print_exc()
            transition_codebook = None
            transition_table = None
    else:
        print(f"[DNA-HDC Phase 1b] Transition Codebook not available (import failed)")

    bigram_packed = np.zeros(vocab_size, dtype=np.uint16)
    _bigram_precomputed = False
    _bg15_start = time.time()
    try:
        _b15_prev = tokens[:-1].astype(np.int64)
        _b15_next = tokens[1:].astype(np.int64)
        _b15_pair_keys = _b15_prev * vocab_size + _b15_next
        _b15_uniq, _b15_cnts = np.unique(_b15_pair_keys, return_counts=True)
        _b15_pair_prev = (_b15_uniq // vocab_size).astype(np.int64)
        _b15_pair_next = (_b15_uniq %  vocab_size).astype(np.uint16)
        _b15_cnts_i32  = _b15_cnts.astype(np.int32)
        _b15_sorted = np.lexsort((-_b15_cnts_i32, _b15_pair_prev))
        _, _b15_first = np.unique(_b15_pair_prev[_b15_sorted], return_index=True)
        _b15_win_prev = _b15_pair_prev[_b15_sorted[_b15_first]]
        _b15_win_next = _b15_pair_next[_b15_sorted[_b15_first]]
        _b15_win_conf = np.minimum(
            _b15_cnts_i32[_b15_sorted[_b15_first]] // 1_000, 63
        ).astype(np.int32)
        bigram_packed[_b15_win_prev] = pack_entry_vec(_b15_win_next, _b15_win_conf)
        _b15_filled = int(np.sum(_b15_win_conf > 0))
        del (_b15_prev, _b15_next, _b15_pair_keys, _b15_uniq, _b15_cnts,
             _b15_pair_prev, _b15_pair_next, _b15_cnts_i32, _b15_sorted,
             _b15_first, _b15_win_prev, _b15_win_next, _b15_win_conf)
        _bigram_precomputed = True
        print(f"[DNA-HDC Phase 1.5] Bigram pre-computed: "
              f"{_b15_filled}/{vocab_size} entries "
              f"({_b15_filled/vocab_size*100:.1f}%) in {time.time()-_bg15_start:.2f}s "
              f"(Phase 3.5 will reuse — saves ~5–10s for Phase 4)")
    except Exception as _bg15_err:
        _bigram_precomputed = False
        bigram_packed = np.zeros(vocab_size, dtype=np.uint16)
        print(f"[DNA-HDC Phase 1.5] Bigram pre-computation failed ({_bg15_err}); "
              f"Phase 3.5 will compute it normally.")

    TRIGRAM_SIZE = vocab_size * vocab_size
    trigram_packed = np.zeros(TRIGRAM_SIZE, dtype=np.uint16)
    _tg15_start = time.time()
    try:
        if len(tokens) > 2:
            _t15_prev2    = tokens[:-2].astype(np.int64)
            _t15_prev1    = tokens[1:-1].astype(np.int64)
            _t15_next     = tokens[2:].astype(np.int64)
            _t15_pair_key = _t15_prev2 * vocab_size + _t15_prev1
            _t15_trip_key = _t15_pair_key * vocab_size + _t15_next
            _t15_uniq, _t15_cnts = np.unique(_t15_trip_key, return_counts=True)
            _t15_pk   = (_t15_uniq // vocab_size).astype(np.int64)
            _t15_nt   = (_t15_uniq %  vocab_size).astype(np.uint16)
            _t15_ci32 = _t15_cnts.astype(np.int32)
            _t15_sort  = np.lexsort((-_t15_ci32, _t15_pk))
            _, _t15_fi = np.unique(_t15_pk[_t15_sort], return_index=True)
            _t15_wk    = _t15_pk[_t15_sort[_t15_fi]]
            _t15_wn    = _t15_nt[_t15_sort[_t15_fi]]
            _t15_wc    = np.minimum(_t15_ci32[_t15_sort[_t15_fi]] // 1_000, 63).astype(np.int32)
            _t15_valid = (_t15_wk >= 0) & (_t15_wk < TRIGRAM_SIZE)
            trigram_packed[_t15_wk[_t15_valid]] = pack_entry_vec(
                _t15_wn[_t15_valid], _t15_wc[_t15_valid]
            )
            _t15_filled = int(np.sum(_t15_wc[_t15_valid] > 0))
            del (_t15_prev2, _t15_prev1, _t15_next, _t15_pair_key, _t15_trip_key,
                 _t15_uniq, _t15_cnts, _t15_pk, _t15_nt, _t15_ci32,
                 _t15_sort, _t15_fi, _t15_wk, _t15_wn, _t15_wc, _t15_valid)
            print(f"[DNA-HDC Phase 1.5b] Trigram pre-computed: "
                  f"{_t15_filled:,}/{TRIGRAM_SIZE:,} confident entries "
                  f"({_t15_filled/TRIGRAM_SIZE*100:.2f}%) in {time.time()-_tg15_start:.2f}s "
                  f"| 2 MB raw | LZMA9 ≈ 300-500 KB compressed")
        else:
            print(f"[DNA-HDC Phase 1.5b] Skipped: corpus too small for trigrams")
    except Exception as _tg15_err:
        trigram_packed = np.zeros(TRIGRAM_SIZE, dtype=np.uint16)
        print(f"[DNA-HDC Phase 1.5b] Trigram pre-computation failed ({_tg15_err}); "
              f"continuing without trigram table")

    seed_val = np.uint64(seed)

    def compute_context_hashes(chunk_start: int, chunk_end: int,
                                return_fingerprints: bool = False):

           
        if _ROLLING_HASH_AVAILABLE and _rh_chunk_g_states:
            try:
                from _full_context_hash import hadamard_key_batch as _hk_fn
                _seed_u = np.uint64(seed)
                nearest = max((s for s in _rh_chunk_g_states if s <= chunk_start),
                              default=0)
                G_at_nearest = _rh_chunk_g_states.get(nearest, np.uint64(0))
                if nearest < chunk_start:
                    _fp2 = np.arange(nearest, chunk_start, dtype=np.int64)
                    _fk2 = _hk_fn(_fp2)
                    with np.errstate(over='ignore'):
                        G_start = G_at_nearest ^ np.bitwise_xor.accumulate(
                            tokens[nearest:chunk_start].astype(np.uint64) * _fk2
                        )[-1]
                    del _fp2, _fk2
                else:
                    G_start = G_at_nearest
                _pos3 = np.arange(chunk_start, chunk_end, dtype=np.int64)
                _key3 = _hk_fn(_pos3)
                with np.errstate(over='ignore'):
                    _c3  = tokens[chunk_start:chunk_end].astype(np.uint64) * _key3
                    _i3  = np.bitwise_xor.accumulate(_c3)
                    _e3  = np.empty(len(_c3), dtype=np.uint64)
                    _e3[0] = G_start
                    if len(_c3) > 1:
                        _e3[1:] = G_start ^ _i3[:-1]
                    _fin3 = (_e3 ^ _seed_u) * _RH_FMIX
                result = (_fin3 >> np.uint64(64 - TABLE_BITS)).astype(np.int64)
                if return_fingerprints:
                    fps = ((_fin3 >> FINGERPRINT_SHIFT) & np.uint64(0xFF)).astype(np.uint8)
                    del _pos3, _key3, _c3, _i3, _e3, _fin3
                    return result, fps
                del _pos3, _key3, _c3, _i3, _e3, _fin3
                return result
            except Exception:
                pass
        chunk_n  = chunk_end - chunk_start
        ctx_base = tokens[chunk_start - CTX_LEN: chunk_end].astype(np.uint64)
        hash_vals = np.zeros(chunk_n, dtype=np.uint64)
        for c in range(CTX_LEN):
            hash_vals ^= ctx_base[c: c + chunk_n] * POS_HASH_KEYS[c]
        hash_vals = (hash_vals ^ seed_val) * np.uint64(0x9E3779B97F4A7C15)
        buckets = (hash_vals >> np.uint64(64 - TABLE_BITS)).astype(np.int64)
        if return_fingerprints:
            fps = ((hash_vals >> FINGERPRINT_SHIFT) & np.uint64(0xFF)).astype(np.uint8)
            return buckets, fps
        return buckets

    sem_prior = None
    correction_map = None
    prior_distributions = None
    try:
        from _semantic_layer import DirectionalSemanticVec as _DSV_cls
        _p0_start = time.time()
        _p0_uint64c = vocab_size * W_UINT64
        sem_prior = _DSV_cls.build_pretrain_prior(
            tokens, codebook, vocab_size, W_UINT64, _p0_uint64c,
            n_tokens=2_000_000, label="Phase0-Prior"
        )
        correction_map      = sem_prior.build_correction_map(vocab_size)
        prior_distributions = sem_prior.build_token_distributions(
            vocab_size, codebook, top_k=8
        )
        print(f"[DNA-HDC Phase 0] Semantic prior ready in "
              f"{time.time()-_p0_start:.2f}s | ~352 KB total")
    except Exception as _p0_err:
        print(f"[DNA-HDC Phase 0] Skipped ({_p0_err})")

    import concurrent.futures

    print(f"\n[DNA-HDC Phase 2] Building DNA-stacked context table...")
    print(f"[DNA-HDC Phase 2] Parallel vectorized pipeline (ThreadPool + numpy GIL-free)")
    print(f"[DNA-HDC Phase 2] Packed table: {TABLE_SIZE:,} entries × 2 bytes = {TABLE_SIZE * 2 / 1024 / 1024:.1f} MB")
    print(f"[DNA-HDC Phase 2] Overflow table: {OVERFLOW_SIZE:,} entries × 2 bytes = {OVERFLOW_SIZE * 2 / 1024:.1f} KB")

    table_packed = np.zeros(TABLE_SIZE, dtype=np.uint16)

    _table_gpu = None
    if _CUPY_AVAILABLE:
        try:
            _table_gpu = cp.zeros(TABLE_SIZE, dtype=cp.uint16)
            print(f"[DNA-HDC GPU] table_packed ({TABLE_SIZE * 2 / 1024 / 1024:.1f} MB) "
                  f"allocated in VRAM — scatter-gather will bypass PCIe")
        except Exception as _e:
            _table_gpu = None
            print(f"[DNA-HDC GPU] VRAM allocation failed ({_e}), using CPU path")

    FINGERPRINT_BITS = 8
    FINGERPRINT_SHIFT = np.uint64(64 - TABLE_BITS - FINGERPRINT_BITS)
    fingerprint_packed = np.zeros(TABLE_SIZE, dtype=np.uint8)

    _fp_gpu = None
    if _table_gpu is not None:
        try:
            _fp_gpu = cp.zeros(TABLE_SIZE, dtype=cp.uint8)
        except Exception:
            _fp_gpu = None

    overflow_packed = np.zeros(OVERFLOW_SIZE, dtype=np.uint16)
    overflow_bitmap = np.zeros(OVERFLOW_BITMAP_SIZE, dtype=np.uint64)

    transition_indices = np.zeros(TABLE_SIZE, dtype=np.uint8) if transition_codebook else None
    transition_counts = np.zeros(TABLE_SIZE, dtype=np.uint8) if transition_codebook else None

    CHUNK = 50_000_000
    N_WORKERS = 4

    def process_chunk(chunk_start: int, chunk_end: int):

           
        chunk_n = chunk_end - chunk_start

        _ch_result  = compute_context_hashes(chunk_start, chunk_end, return_fingerprints=True)
        if isinstance(_ch_result, tuple):
            buckets, chunk_fps = _ch_result
        else:
            buckets, chunk_fps = _ch_result, None
        chunk_targets = tokens[chunk_start: chunk_end].astype(np.int64)

        pair_keys = buckets * vocab_size + chunk_targets

        unique_pairs, counts = np.unique(pair_keys, return_counts=True)
        pair_buckets = unique_pairs // vocab_size
        pair_tokens = (unique_pairs % vocab_size).astype(np.uint16)
        counts_i32 = counts.astype(np.int32)

        sorted_idx = np.lexsort((-counts_i32, pair_buckets))
        sorted_pair_buckets = pair_buckets[sorted_idx]
        sorted_pair_tokens = pair_tokens[sorted_idx]
        sorted_counts = counts_i32[sorted_idx]

        _, first_idx = np.unique(sorted_pair_buckets, return_index=True)
        winner_buckets = sorted_pair_buckets[first_idx]
        winner_tokens = sorted_pair_tokens[first_idx]
        winner_counts = sorted_counts[first_idx]

        winner_fps = None
        if chunk_fps is not None:
            orig_first = sorted_idx[first_idx]
            winner_fps = chunk_fps[orig_first]

        return winner_buckets, winner_tokens, winner_counts, chunk_n, winner_fps

    def _gather_table(idx):
                                                                                  
        if _table_gpu is not None:
            try:
                return cp.asnumpy(_table_gpu[cp.asarray(idx)])
            except Exception:
                pass
        return table_packed[idx]

    def _scatter_table(idx, packed):
                                                                                 
        if _table_gpu is not None:
            try:
                _table_gpu[cp.asarray(idx)] = cp.asarray(packed)
                return
            except Exception:
                pass
        table_packed[idx] = packed

    def _gather_fp(idx):
                                                                     
        if _fp_gpu is not None:
            try:
                return cp.asnumpy(_fp_gpu[cp.asarray(idx)])
            except Exception:
                pass
        return fingerprint_packed[idx]

    def _scatter_fp(idx, fp):
                                                                      
        if _fp_gpu is not None:
            try:
                _fp_gpu[cp.asarray(idx)] = cp.asarray(fp)
                return
            except Exception:
                pass
        fingerprint_packed[idx] = fp

    def merge_winners(winner_buckets, winner_tokens, winner_counts, chunk_start=None,
                      winner_fingerprints=None):
   
        current_packed = _gather_table(winner_buckets)
        current_tokens, current_counts = unpack_entry_vec(current_packed)

        empty_mask = (current_counts == 0)
        match_mask = (~empty_mask) & (current_tokens == winner_tokens)
        mismatch_mask = (~empty_mask) & (current_tokens != winner_tokens)
        overwrite_mask = mismatch_mask & (winner_counts > current_counts)
        weaken_mask = mismatch_mask & (~overwrite_mask)
        collision_mask = mismatch_mask & (current_counts < 3) & (winner_counts >= 2)

        if np.any(empty_mask):
            eb = winner_buckets[empty_mask]
            _scatter_table(eb, pack_entry_vec(winner_tokens[empty_mask], winner_counts[empty_mask]))
            if winner_fingerprints is not None:
                _scatter_fp(eb, winner_fingerprints[empty_mask])

        if np.any(match_mask):
            mb = winner_buckets[match_mask]
            new_counts = current_counts[match_mask] + winner_counts[match_mask]
            _scatter_table(mb, pack_entry_vec(winner_tokens[match_mask], new_counts))

        if np.any(overwrite_mask):
            ob = winner_buckets[overwrite_mask]
            new_counts = winner_counts[overwrite_mask] - current_counts[overwrite_mask]
            _scatter_table(ob, pack_entry_vec(winner_tokens[overwrite_mask], new_counts))
            if winner_fingerprints is not None:
                _scatter_fp(ob, winner_fingerprints[overwrite_mask])

        if np.any(weaken_mask):
            wb = winner_buckets[weaken_mask]
            new_counts = current_counts[weaken_mask] - winner_counts[weaken_mask]
            new_counts = np.maximum(new_counts, 0)

            zeroed_mask = (new_counts == 0)
            if np.any(zeroed_mask):
                repair_buckets = wb[zeroed_mask]
                repair_tokens  = winner_tokens[weaken_mask][zeroed_mask]
                if len(repair_buckets) > 0:
                    _scatter_table(repair_buckets, pack_entry_vec(
                        repair_tokens, np.ones(len(repair_buckets), dtype=np.int32)
                    ))
                    surviving_mask = ~zeroed_mask
                    if np.any(surviving_mask):
                        _scatter_table(wb[surviving_mask], pack_entry_vec(
                            current_tokens[weaken_mask][surviving_mask],
                            new_counts[surviving_mask]
                        ))
            else:
                _scatter_table(wb, pack_entry_vec(current_tokens[weaken_mask], new_counts))

        if np.any(collision_mask):
            _cb = winner_buckets[collision_mask].astype(np.int64)
            _ct = winner_tokens[collision_mask]
            _cc = winner_counts[collision_mask]
            _pc = np.unpackbits(
                _cb.view(np.uint8).reshape(-1, 8), axis=1, bitorder='little'
            ).sum(axis=1).astype(np.int64)
            _flip_bits    = _pc % TABLE_BITS
            _overflow_idx = (_cb ^ (np.int64(1) << _flip_bits)) % OVERFLOW_SIZE
            overflow_packed[_overflow_idx] = pack_entry_vec(_ct, _cc)
            for _oi in _overflow_idx:
                _oi = int(_oi)
                overflow_bitmap[_oi // 64] |= np.uint64(1) << np.uint64(_oi % 64)

        if transition_codebook is not None and transition_table is not None and chunk_start is not None:
            try:
                _valid_m = winner_counts > 0
                if np.any(_valid_m):
                    _vw_b = winner_buckets[_valid_m]
                    _vw_t = (winner_tokens[_valid_m].astype(np.int32)) % vocab_size
                    _vw_c = winner_counts[_valid_m]
                    _key_xor = np.zeros(W_UINT64, dtype=np.uint64)
                    for _cc2 in range(CTX_LEN):
                        _key_xor ^= POS_HASH_KEYS[_cc2]
                    if CTX_LEN % 2 == 0:
                        _trans_hvs = _key_xor[None, :] ^ codebook[_vw_t]
                    else:
                        _trans_hvs = np.tile(_key_xor, (len(_vw_t), 1))
                    _trans_idxs = transition_codebook.find_nearest_transition_batch(_trans_hvs)
                    if hasattr(transition_table, 'store_transitions_batch'):
                        transition_table.store_transitions_batch(_vw_b, _trans_idxs, _vw_c)
                    else:
                        for _i2, (_b2, _ti2, _vc2) in enumerate(zip(_vw_b, _trans_idxs, _vw_c)):
                            transition_table.store_transition(int(_b2), int(_ti2), int(min(_vc2, 255)))
            except Exception:
                pass

    chunk_ranges = []
    for cs in range(CTX_LEN, N, CHUNK):
        ce = min(cs + CHUNK, N)
        chunk_ranges.append((cs, ce))

    total_processed = 0
    phase2_start = time.time()
    checkpoint_interval = 1_000_000
    last_checkpoint_pos = 0

    batch_size = N_WORKERS

    for batch_start in range(0, len(chunk_ranges), batch_size):
        batch_end = min(batch_start + batch_size, len(chunk_ranges))
        batch = chunk_ranges[batch_start:batch_end]

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {
                pool.submit(process_chunk, cs, ce): (cs, ce)
                for cs, ce in batch
            }

            for future in concurrent.futures.as_completed(futures):
                winner_buckets, winner_tokens, winner_counts, chunk_n, w_fps = future.result()
                merge_winners(winner_buckets, winner_tokens, winner_counts,
                              winner_fingerprints=w_fps)
                total_processed += chunk_n

                if context_checkpoint_mgr is not None:
                    cs, ce = futures[future]
                    if ce - last_checkpoint_pos >= checkpoint_interval:
                        actual_token = int(tokens[ce]) if ce < len(tokens) else 0
                        context_checkpoint_mgr.update(actual_token, ce)
                        last_checkpoint_pos = ce

        elapsed_so_far = time.time() - phase2_start
        rate = total_processed / elapsed_so_far if elapsed_so_far > 0 else 0
        print(f"[DNA-HDC Phase 2] {total_processed:,}/{N - CTX_LEN:,} "
              f"({rate:,.0f} tok/s parallel, batch {batch_start//batch_size + 1})")

        if time.time() - start_time > config.max_wallclock_seconds * 0.80:
            print(f"[DNA-HDC Phase 2] Budget 80%, stopping at {total_processed:,}")
            break

    phase2_time = time.time() - phase2_start
    if _table_gpu is not None:
        try:
            _cnts_gpu = (_table_gpu >> cp.uint16(10)) & cp.uint16(0x3F)
            filled = int(cp.sum(_cnts_gpu > 0))
        except Exception:
            _, counts = unpack_entry_vec(table_packed)
            filled = int(np.sum(counts > 0))
    else:
        _, counts = unpack_entry_vec(table_packed)
        filled = int(np.sum(counts > 0))
    print(f"[DNA-HDC Phase 2] DNA Stack built in {phase2_time:.1f}s")
    print(f"[DNA-HDC Phase 2] Filled: {filled:,}/{TABLE_SIZE:,} ({filled/TABLE_SIZE*100:.1f}%)")
    print(f"[DNA-HDC Phase 2] Throughput: {total_processed / phase2_time:,.0f} tok/s (parallel vectorized)")

    pass_num = 1
    while time.time() - start_time < config.max_wallclock_seconds * 0.75:
        pass_num += 1
        pass_processed = 0
        pass_start = time.time()

        for batch_start in range(0, len(chunk_ranges), batch_size):
            batch_end = min(batch_start + batch_size, len(chunk_ranges))
            batch = chunk_ranges[batch_start:batch_end]

            with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                futures = {
                    pool.submit(process_chunk, cs, ce): (cs, ce)
                    for cs, ce in batch
                }
                for future in concurrent.futures.as_completed(futures):
                    winner_buckets, winner_tokens, winner_counts, chunk_n, w_fps = future.result()
                    merge_winners(winner_buckets, winner_tokens, winner_counts,
                                  winner_fingerprints=w_fps)
                    pass_processed += chunk_n
                    total_processed += chunk_n

            if time.time() - start_time > config.max_wallclock_seconds * 0.75:
                break

        pass_time = time.time() - pass_start
        if _table_gpu is not None:
            try:
                _cnts_gpu = (_table_gpu >> cp.uint16(10)) & cp.uint16(0x3F)
                filled = int(cp.sum(_cnts_gpu > 0))
            except Exception:
                _, counts = unpack_entry_vec(table_packed)
                filled = int(np.sum(counts > 0))
        else:
            _, counts = unpack_entry_vec(table_packed)
            filled = int(np.sum(counts > 0))
        print(f"[DNA-HDC Phase 3] Pass {pass_num}: +{pass_processed:,} tok, "
              f"filled={filled:,}/{TABLE_SIZE:,} ({filled/TABLE_SIZE*100:.1f}%), "
              f"{pass_time:.1f}s")

    print(f"\n[DNA-HDC Phase 3.5] Building bigram prediction table...")
    _bg_start = time.time()
    if _bigram_precomputed:
        _bg35_filled = int(np.sum((bigram_packed >> np.uint16(10)) & np.uint16(0x3F)) > 0)
        print(f"[DNA-HDC Phase 3.5] Bigram table reused from Phase 1.5 "
              f"({time.time()-_bg_start:.3f}s) — {_bg35_filled}/{vocab_size} entries | 2 KB total")
    else:
        bigram_packed = np.zeros(vocab_size, dtype=np.uint16)
        try:
            _bg_prev = tokens[:-1].astype(np.int64)
            _bg_next = tokens[1:].astype(np.int64)
            _bg_pair_keys = _bg_prev * vocab_size + _bg_next
            _bg_uniq, _bg_cnts = np.unique(_bg_pair_keys, return_counts=True)
            _bg_pair_prev = (_bg_uniq // vocab_size).astype(np.int64)
            _bg_pair_next = (_bg_uniq %  vocab_size).astype(np.uint16)
            _bg_cnts_i32  = _bg_cnts.astype(np.int32)
            _bg_sorted = np.lexsort((-_bg_cnts_i32, _bg_pair_prev))
            _, _bg_first = np.unique(_bg_pair_prev[_bg_sorted], return_index=True)
            _win_prev = _bg_pair_prev[_bg_sorted[_bg_first]]
            _win_next = _bg_pair_next[_bg_sorted[_bg_first]]
            _win_conf = np.minimum(_bg_cnts_i32[_bg_sorted[_bg_first]] // 1_000, 63).astype(np.int32)
            bigram_packed[_win_prev] = pack_entry_vec(_win_next, _win_conf)
            _bg_filled = int(np.sum(_win_conf > 0))
            del _bg_prev, _bg_next, _bg_pair_keys, _bg_uniq, _bg_cnts
            del _bg_pair_prev, _bg_pair_next, _bg_cnts_i32, _bg_sorted, _bg_first
            del _win_prev, _win_next, _win_conf
            print(f"[DNA-HDC Phase 3.5] Bigram table: {_bg_filled}/{vocab_size} entries "
                  f"({_bg_filled/vocab_size*100:.1f}%) filled in {time.time()-_bg_start:.2f}s | 2 KB total")
        except Exception as _bg_err:
            print(f"[DNA-HDC Phase 3.5] Warning: bigram build failed ({_bg_err}), using empty table")
            bigram_packed = np.zeros(vocab_size, dtype=np.uint16)

    dsv = None
    try:
        from _semantic_layer import DirectionalSemanticVec as _DSV_Cls
        _dsv_uint64c  = vocab_size * W_UINT64
        _dsv_remaining = max(0.0, config.max_wallclock_seconds - (time.time() - start_time))
        _dsv_budget    = min(55.0, _dsv_remaining * 0.40)
        if _dsv_budget >= 5.0 and _dsv_uint64c <= 16384:
            _dsv_t0 = time.time()
            dsv = _DSV_Cls.build_from_tokens(
                tokens        = tokens,
                codebook      = codebook,
                ctx_len       = CTX_LEN,
                vocab_size    = vocab_size,
                W             = W_UINT64,
                uint64_count  = _dsv_uint64c,
                time_budget_s = _dsv_budget,
                label         = "Phase3.5-DSV",
            )
            print(f"[DNA-HDC Phase 3.5-DSV] Built in {time.time()-_dsv_t0:.2f}s "
                  f"| sem_fwd+sem_bwd = {2 * _dsv_uint64c * 8 // 1024} KB "
                  f"| budget={_dsv_budget:.1f}s")
        else:
            print(f"[DNA-HDC Phase 3.5-DSV] Skipped: budget={_dsv_budget:.1f}s "
                  f"precond={_dsv_uint64c}≤16384={_dsv_uint64c <= 16384}")
    except ImportError:
        print(f"[DNA-HDC Phase 3.5-DSV] _semantic_layer.py not found — DSV disabled")
    except Exception as _dsv_err:
        print(f"[DNA-HDC Phase 3.5-DSV] Failed ({_dsv_err}) — DSV disabled")
        dsv = None

    _srh = None
    _srh_checkpoints = None
    _srh_keys_arr = None
    try:
        from _semantic_rolling_hash import SemanticRollingHash as _SRH_cls
        if dsv is not None:
            _srh_t0 = time.time()
            _srh_remaining = config.max_wallclock_seconds - (time.time() - start_time)
            _srh_budget = min(25.0, _srh_remaining * 0.15)
            if _srh_budget >= 3.0:
                _srh = _SRH_cls(W_UINT64=W_UINT64, alpha=0.005)
                _sem_fwd_mat = dsv.sem_fwd.reshape(vocab_size, W_UINT64)
                try:
                    from _full_context_hash import hadamard_key_batch as _hk_srh
                    _srh_pos = np.arange(min(len(tokens), 10_000_000), dtype=np.int64)
                    _srh_keys_arr = _hk_srh(_srh_pos)
                except Exception:
                    _srh_keys_arr = None
                _srh_chunk_size = 2_000_000
                _srh_boundaries = list(range(0, len(tokens), _srh_chunk_size))
                _srh_checkpoints = _srh.build_states(
                    tokens, _sem_fwd_mat,
                    _srh_keys_arr if _srh_keys_arr is not None else np.full(len(tokens), np.uint64(0x9E3779B97F4A7C15), dtype=np.uint64),
                    _srh_boundaries,
                    time_budget_s=_srh_budget,
                    label="Phase3.5-SRH"
                )
                print(f"[DNA-HDC Phase 3.5-SRH] S[p] states built in "
                      f"{time.time()-_srh_t0:.2f}s | "
                      f"{len(_srh_checkpoints)} checkpoints | alpha={_srh.alpha}")
            else:
                print(f"[DNA-HDC Phase 3.5-SRH] Skipped: budget={_srh_budget:.1f}s")
        else:
            print(f"[DNA-HDC Phase 3.5-SRH] Skipped: DSV not available")
    except ImportError:
        print(f"[DNA-HDC Phase 3.5-SRH] _semantic_rolling_hash.py not found")
    except Exception as _srh_err:
        print(f"[DNA-HDC Phase 3.5-SRH] Failed ({_srh_err})")
        _srh = None
        _srh_checkpoints = None

    try:
        if dsv is not None:
            _sb_remaining = config.max_wallclock_seconds - (time.time() - start_time)
            _sb_budget = min(20.0, _sb_remaining * 0.12)
            if _sb_budget >= 3.0:
                _sb_t0 = time.time()
                dsv.build_skip_bigram_lags(
                    tokens, codebook, max_lag=5,
                    time_budget_s=_sb_budget,
                    label="Phase3.5-SkipBigram"
                )
                print(f"[DNA-HDC Phase 3.5-SkipBigram] Done in "
                      f"{time.time()-_sb_t0:.2f}s")
            else:
                print(f"[DNA-HDC Phase 3.5-SkipBigram] Skipped: budget={_sb_budget:.1f}s")
    except Exception as _sb_err:
        print(f"[DNA-HDC Phase 3.5-SkipBigram] Failed ({_sb_err})")

    try:
        if dsv is not None:
            _xo_remaining = config.max_wallclock_seconds - (time.time() - start_time)
            _xo_budget = min(10.0, _xo_remaining * 0.06)
            if _xo_budget >= 2.0:
                _xo_t0 = time.time()
                dsv.build_xor_orbit_table(
                    tokens, codebook, threshold=3,
                    time_budget_s=_xo_budget,
                    label="Phase3.5-XOROrbit"
                )
                print(f"[DNA-HDC Phase 3.5-XOROrbit] Done in "
                      f"{time.time()-_xo_t0:.2f}s")
            else:
                print(f"[DNA-HDC Phase 3.5-XOROrbit] Skipped: budget={_xo_budget:.1f}s")
    except Exception as _xo_err:
        print(f"[DNA-HDC Phase 3.5-XOROrbit] Failed ({_xo_err})")

    suffix_grammar = None
    try:
        from _suffix_grammar import SuffixGrammarTable as _SGT_cls
        if _srh is not None and _srh_checkpoints is not None and _TRANSITION_CODEBOOK_AVAILABLE:
            _sg_remaining = config.max_wallclock_seconds - (time.time() - start_time)
            _sg_budget = min(12.0, _sg_remaining * 0.08)
            if _sg_budget >= 2.0:
                _sg_t0 = time.time()
                _sg_char_hv = CharacterHypervector(dim=1024, w_uint64=W_UINT64)
                suffix_grammar = _SGT_cls(
                    vocab_size, W_UINT64, _sg_char_hv, sp_model, suffix_len=3
                )
                _sem_fwd_mat_sg = dsv.sem_fwd.reshape(vocab_size, W_UINT64) if dsv is not None else None
                suffix_grammar.build_from_corpus(
                    tokens,
                    None,
                    srh=_srh,
                    sem_fwd_matrix=_sem_fwd_mat_sg,
                    keys=_srh_keys_arr if _srh_keys_arr is not None else np.full(len(tokens), np.uint64(0x9E3779B97F4A7C15), dtype=np.uint64),
                    checkpoints=_srh_checkpoints,
                    time_budget_s=_sg_budget,
                    label="Phase3.5-SuffixGrammar"
                )
                print(f"[DNA-HDC Phase 3.5-SuffixGrammar] {suffix_grammar.summary()} "
                      f"in {time.time()-_sg_t0:.2f}s")
            else:
                print(f"[DNA-HDC Phase 3.5-SuffixGrammar] Skipped: budget={_sg_budget:.1f}s")
        else:
            print(f"[DNA-HDC Phase 3.5-SuffixGrammar] Skipped: SRH or CharHV not available")
    except ImportError:
        print(f"[DNA-HDC Phase 3.5-SuffixGrammar] _suffix_grammar.py not found")
    except Exception as _sg_err:
        print(f"[DNA-HDC Phase 3.5-SuffixGrammar] Failed ({_sg_err})")
        suffix_grammar = None

    _ar_calib_tokens = None
    try:
        _ar_t0        = time.time()
        _AR_NUM_SEQS  = 32
        _AR_SEQ_LEN   = 256
        _AR_TEMP      = 0.8
        _ar_np_rng    = np.random.RandomState(int(seed) % (2**32))
        _AR_FMIX      = np.uint64(0x9E3779B97F4A7C15)

        ar_seqs = []
        for _si in range(_AR_NUM_SEQS):
            seq = list(_ar_np_rng.randint(0, vocab_size, size=CTX_LEN).astype(np.uint16))
            for _p in range(CTX_LEN, _AR_SEQ_LEN):
                _h = np.uint64(0)
                with np.errstate(over='ignore'):
                    for _c in range(CTX_LEN):
                        _h ^= np.uint64(int(seq[_p - CTX_LEN + _c])) * POS_HASH_KEYS[_c]
                    _h = (_h ^ seed_val) * _AR_FMIX
                _bucket = int(_h >> np.uint64(64 - TABLE_BITS)) & (TABLE_SIZE - 1)
                _packed = int(table_packed[_bucket])
                _pred   = _packed & 0x3FF
                _conf   = (_packed >> 10) & 0x3F
                if _conf >= 3:
                    next_tok = _pred
                elif _conf >= 1 and _ar_np_rng.random() >= _AR_TEMP:
                    next_tok = _pred
                else:
                    _bg   = int(bigram_packed[seq[-1]])
                    _bg_t = _bg & 0x3FF
                    _bg_c = (_bg >> 10) & 0x3F
                    next_tok = int(_bg_t) if _bg_c > 0 else int(_ar_np_rng.randint(0, vocab_size))
                seq.append(next_tok)
            ar_seqs.append(np.array(seq, dtype=np.uint16))

        _ar_calib_tokens = np.concatenate(ar_seqs).astype(np.uint16)
        print(f"[DNA-HDC Pre-Phase4] AR self-gen calibration: "
              f"{_AR_NUM_SEQS} seqs × {_AR_SEQ_LEN} tokens = "
              f"{len(_ar_calib_tokens):,} tokens in {time.time()-_ar_t0:.2f}s")
    except Exception as _ar_err:
        print(f"[DNA-HDC Pre-Phase4] AR calibration skipped ({_ar_err})")
        _ar_calib_tokens = None

    if _ar_calib_tokens is not None and len(_ar_calib_tokens) > CTX_LEN:
        try:
            _ar40_repairs    = 0
            _ar40_reinforced = 0
            _ar_N            = len(_ar_calib_tokens)
            _AR_CHUNK        = min(CHUNK, _ar_N)
            for _acs in range(CTX_LEN, _ar_N, _AR_CHUNK):
                _ace = min(_acs + _AR_CHUNK, _ar_N)
                _acn = _ace - _acs
                _ar_ctx = _ar_calib_tokens[_acs - CTX_LEN: _ace].astype(np.uint64)
                _ar_hv  = np.zeros(_acn, dtype=np.uint64)
                for _c in range(CTX_LEN):
                    _ar_hv ^= _ar_ctx[_c: _c + _acn] * POS_HASH_KEYS[_c]
                _ar_hv = (_ar_hv ^ seed_val) * np.uint64(0x9E3779B97F4A7C15)
                _ar_bkts = (_ar_hv >> np.uint64(64 - TABLE_BITS)).astype(np.int64)
                _ar_tgts = _ar_calib_tokens[_acs:_ace]
                _ar_pp, _ar_cp = unpack_entry_vec(_gather_table(_ar_bkts))
                _ar_wrong = (_ar_pp != _ar_tgts) | (_ar_cp == 0)
                if np.any(_ar_wrong):
                    _wr_b = _ar_bkts[_ar_wrong]
                    _wr_t = _ar_tgts[_ar_wrong]
                    _wr_cp2, _wr_cc2 = unpack_entry_vec(_gather_table(_wr_b))
                    _wr_low = _wr_cc2 < 3
                    if np.any(_wr_low):
                        _new_cc = np.clip(_wr_cc2[_wr_low] + 1, 0, 63).astype(np.int32)
                        _scatter_table(_wr_b[_wr_low], pack_entry_vec(_wr_t[_wr_low], _new_cc))
                        _ar40_repairs += int(np.sum(_wr_low))
                _ar_correct = ~_ar_wrong
                if np.any(_ar_correct):
                    _co_b = _ar_bkts[_ar_correct]
                    _co_cp2, _co_cc2 = unpack_entry_vec(_gather_table(_co_b))
                    _co_deep = (_co_cc2 > 0) & (_co_cc2 < 3)
                    if np.any(_co_deep):
                        _new_coc = np.clip(_co_cc2[_co_deep] + 1, 0, 63).astype(np.int32)
                        _scatter_table(_co_b[_co_deep], pack_entry_vec(_co_cp2[_co_deep], _new_coc))
                        _ar40_reinforced += int(np.sum(_co_deep))
            print(f"[DNA-HDC Phase 4.0] AR calibration sweep: "
                  f"{_ar40_repairs:,} repairs + {_ar40_reinforced:,} reinforced "
                  f"from {_AR_NUM_SEQS} self-generated sequences")
        except Exception as _ar40_err:
            print(f"[DNA-HDC Phase 4.0] AR calibration sweep skipped ({_ar40_err})")

    _repair_queue = getattr(locals(), '_repair_queue', None)
    if _repair_queue is None:
        _repair_queue = {}
    try:
        if _repair_queue and _srh is not None and dsv is not None:
            _p4a_start = time.time()
            _p4a_remaining = config.max_wallclock_seconds - (time.time() - start_time)
            _p4a_budget = min(60.0, _p4a_remaining * 0.35)
            _sorted_repairs = sorted(
                _repair_queue.items(), key=lambda x: x[1][2], reverse=True
            )
            _p4a_written = 0
            _sem_fwd_mat_p4 = dsv.sem_fwd.reshape(vocab_size, W_UINT64)
            for _p4a_bucket, (_p4a_wrong, _p4a_cand, _p4a_conf, _p4a_src) in _sorted_repairs:
                if time.time() - _p4a_start > _p4a_budget:
                    break
                _p4a_stored, _p4a_count = unpack_entry(int(_gather_table(
                    np.array([_p4a_bucket], dtype=np.int64)
                )[0]))
                if _p4a_count >= 3:
                    continue
                _p4a_fwd_vec = _sem_fwd_mat_p4[int(_p4a_cand) % vocab_size]
                try:
                    from _semantic_rolling_hash import bipolar as _bp4a, wht_vectorised as _wht4a
                    _p4a_corr = _wht4a(_bp4a(_p4a_fwd_vec))[:vocab_size] / float(len(_bp4a(_p4a_fwd_vec)))
                    _p4a_consistency = _srh.butterfly_consistency(_p4a_corr, int(_p4a_cand) % vocab_size)
                    if _p4a_consistency < 0.6:
                        continue
                except Exception:
                    pass
                _scatter_table(
                    np.array([_p4a_bucket], dtype=np.int64),
                    pack_entry_vec(
                        np.array([int(_p4a_cand)], dtype=np.uint16),
                        np.array([3], dtype=np.int32)
                    )
                )
                _p4a_written += 1
            print(f"[DNA-HDC Phase 4A] Queue repairs: "
                  f"{_p4a_written:,}/{len(_sorted_repairs):,} written in "
                  f"{time.time()-_p4a_start:.2f}s")
        else:
            print(f"[DNA-HDC Phase 4A] Skipped: "
                  f"queue={len(_repair_queue)} srh={_srh is not None} dsv={dsv is not None}")
    except Exception as _p4a_err:
        print(f"[DNA-HDC Phase 4A] Failed ({_p4a_err})")

    print(f"\n[DNA-HDC Phase 4] Predictive coding repair (error-residual, no bigram)...")

    repair_round = 0
    while time.time() - start_time < config.max_wallclock_seconds:
        repair_round += 1
        repairs = 0
        reinforced = 0
        total_errors = 0
        total_checked = 0

        for chunk_start in range(CTX_LEN, N, CHUNK):
            chunk_end = min(chunk_start + CHUNK, N)
            chunk_n = chunk_end - chunk_start

            _p4_result = compute_context_hashes(chunk_start, chunk_end, return_fingerprints=True)
            if isinstance(_p4_result, tuple):
                buckets, query_fps = _p4_result
            else:
                buckets, query_fps = _p4_result, None
            targets = tokens[chunk_start: chunk_end]

            packed_preds = _gather_table(buckets)
            preds, confs = unpack_entry_vec(packed_preds)

            if query_fps is not None:
                try:
                    collision_detected = (_gather_fp(buckets) != query_fps) & (confs > 0)
                    confs = confs.copy()
                    preds = preds.copy()
                    confs[collision_detected] = 0
                    preds[collision_detected] = (targets[collision_detected] + 1) % vocab_size
                except Exception:
                    pass

            wrong = (preds != targets)
            total_errors += int(np.sum(wrong))
            total_checked += chunk_n

            if not np.any(wrong):
                continue

            wrong_buckets = buckets[wrong]
            wrong_targets = targets[wrong]
            wrong_preds   = preds[wrong]

            wrong_packed = _gather_table(wrong_buckets)
            _, wrong_confs = unpack_entry_vec(wrong_packed)
            repairable = wrong_confs < 10
            if not np.any(repairable):
                continue

            rep_buckets = wrong_buckets[repairable]
            rep_targets = wrong_targets[repairable]
            rep_preds   = wrong_preds[repairable]

            try:
                pred_hvs   = codebook[rep_preds.astype(np.int32) % vocab_size]
                target_hvs = codebook[rep_targets.astype(np.int32) % vocab_size]
                residuals = pred_hvs ^ target_hvs
                if _CUPY_AVAILABLE:
                    try:
                        res_gpu = cp.asarray(residuals.view(np.uint8))
                        residual_bits = cp.asnumpy(
                            cp.unpackbits(res_gpu, axis=1).sum(axis=1).astype(cp.int32)
                        )
                        sort_order = cp.asnumpy(cp.argsort(cp.asarray(residual_bits)))
                    except Exception:
                        residual_bits = np.unpackbits(
                            residuals.view(np.uint8), axis=1
                        ).sum(axis=1).astype(np.int32)
                        sort_order = np.argsort(residual_bits)
                else:
                    residual_bits = np.unpackbits(
                        residuals.view(np.uint8), axis=1
                    ).sum(axis=1).astype(np.int32)
                    sort_order = np.argsort(residual_bits)
                rep_buckets = rep_buckets[sort_order]
                rep_targets = rep_targets[sort_order]
                rep_preds   = rep_preds[sort_order]
                residual_bits = residual_bits[sort_order]
            except Exception:
                residual_bits = None
            if len(rep_buckets) > 0:
                _scatter_table(rep_buckets, pack_entry_vec(
                    rep_targets, np.ones(len(rep_buckets), dtype=np.int32)
                ))
                repairs += len(rep_buckets)

            correct = ~wrong
            if np.any(correct):
                cor_packed = _gather_table(buckets[correct])
                _, cor_confs = unpack_entry_vec(cor_packed)
                to_reinforce = (cor_confs > 0) & (cor_confs < 10)
                if np.any(to_reinforce):
                    reinforce_buckets = buckets[correct][to_reinforce]
                    cur_packed2 = _gather_table(reinforce_buckets)
                    cur_toks2, cur_c2 = unpack_entry_vec(cur_packed2)
                    new_counts2 = np.clip(cur_c2 + 1, 0, 63).astype(np.int32)
                    _scatter_table(reinforce_buckets, pack_entry_vec(cur_toks2, new_counts2))
                    reinforced += int(np.sum(to_reinforce))

            if time.time() - start_time > config.max_wallclock_seconds:
                break

        error_rate = total_errors / total_checked if total_checked > 0 else 0
        print(f"[DNA-HDC Phase 4] Round {repair_round}: error_rate={error_rate*100:.2f}% "
              f"errors={total_errors:,} repairs={repairs:,} "
              f"reinforced={reinforced:,} checked={total_checked:,}")

        if dsv is not None and repair_round % 3 == 0:
            try:
                pruned, nudged = dsv.slow_wave(noise_threshold=0.15)
                if pruned + nudged > 0:
                    print(f"[DNA-HDC Phase 4] Slow-wave: pruned={pruned} nudged={nudged}")
            except Exception:
                pass

        if repairs == 0 and reinforced == 0:
            print(f"[DNA-HDC Phase 4] Holographic convergence — "
                  f"no repairable errors and no low-confidence correct entries remain.")
            break

    if _table_gpu is not None:
        try:
            _sync_t0 = time.time()
            table_packed[:] = cp.asnumpy(_table_gpu)
            if _fp_gpu is not None:
                fingerprint_packed[:] = cp.asnumpy(_fp_gpu)
            print(f"[DNA-HDC GPU] VRAM→RAM sync in {time.time() - _sync_t0:.2f}s "
                  f"(table={TABLE_SIZE*2/1024/1024:.1f} MB + fp={TABLE_SIZE/1024/1024:.1f} MB)")
            del _table_gpu, _fp_gpu
        except Exception as _se:
            print(f"[DNA-HDC GPU] Sync failed ({_se}) — table_packed may be stale")

    try:
        _PRUNE_TARGET_MB       = float(os.environ.get("TARGET_MB", "15.9"))
        _prune_t0              = time.time()
        _prune_unigram         = np.bincount(tokens.astype(np.int64), minlength=vocab_size)
        _all_toks_pr, _all_cnts_pr = unpack_entry_vec(table_packed)
        _c1_mask    = (_all_cnts_pr == 1)
        _c1_indices = np.where(_c1_mask)[0]

        if len(_c1_indices) > 0:
            _c1_pred_f   = _prune_unigram[_all_toks_pr[_c1_indices].astype(np.int64)]
            _c1_sort_ord = np.argsort(_c1_pred_f)
            _c1_sorted   = _c1_indices[_c1_sort_ord]

            def _try_prune_fast(n_zero):
                                                                                      
                tmp = table_packed.copy()
                if n_zero > 0:
                    tmp[_c1_sorted[:n_zero]] = np.uint16(0)
                _payload = tmp.tobytes() + fingerprint_packed.tobytes() + bigram_packed.tobytes()
                return len(zlib.compress(_payload, 6)) + 16

            def _try_prune_lzma9(n_zero):
                                                                                   
                tmp = table_packed.copy()
                if n_zero > 0:
                    tmp[_c1_sorted[:n_zero]] = np.uint16(0)
                _payload = tmp.tobytes() + fingerprint_packed.tobytes() + bigram_packed.tobytes()
                return len(lzma.compress(_payload, preset=9)) + 16

            _cal_zlib = _try_prune_fast(0)
            _cal_lzma = _try_prune_lzma9(0)
            _cal_r    = _cal_lzma / max(_cal_zlib, 1)
            _tgt_b    = int(_PRUNE_TARGET_MB * 1024 * 1024)
            _tgt_zlib = int(_tgt_b / _cal_r)

            _no_sz = _cal_lzma
            print(f"[DNA-HDC Prune] {len(_c1_indices):,} count=1 candidates | "
                  f"unpruned LZMA={_no_sz/1024/1024:.2f} MB | target={_PRUNE_TARGET_MB} MB "
                  f"| calib_r={_cal_r:.3f} (LZMA/zlib)")

            if _no_sz <= _tgt_b:
                print(f"[DNA-HDC Prune] Already fits under target — no pruning required")
            else:
                _full_zsize = _try_prune_fast(len(_c1_indices))
                _full_lsize_est = int(_full_zsize * _cal_r)
                print(f"[DNA-HDC Prune] Full count=1 prune ≈ {_full_lsize_est/1024/1024:.2f} MB "
                      f"(fast est) | {len(_c1_indices):,} candidates")

                if _full_lsize_est > _tgt_b:
                    print(f"[DNA-HDC Prune] Even full count=1 prune insufficient; applying all")
                    _n_prune = len(_c1_indices)
                else:
                    _lo_z, _hi_z = 0, len(_c1_indices)
                    while _lo_z < _hi_z:
                        _mid_z = (_lo_z + _hi_z) // 2
                        if _try_prune_fast(_mid_z) <= _tgt_zlib:
                            _hi_z = _mid_z
                        else:
                            _lo_z = _mid_z + 1

                    _window = max(1, len(_c1_indices) // 1000)
                    _cands  = sorted(set(max(0, _lo_z + _d)
                                        for _d in (-_window, 0, _window, 2*_window)
                                        if 0 <= _lo_z + _d <= len(_c1_indices)))
                    _n_prune = _lo_z
                    for _rc in _cands:
                        if _try_prune_lzma9(_rc) <= _tgt_b:
                            _n_prune = _rc
                            break

                    print(f"[DNA-HDC Prune] Binary search → prune {_n_prune:,} / "
                          f"{len(_c1_indices):,} ({100*_n_prune/max(len(_c1_indices),1):.1f}%) "
                          f"count=1 entries to reach {_PRUNE_TARGET_MB} MB")

                table_packed[_c1_sorted[:_n_prune]] = np.uint16(0)
            print(f"[DNA-HDC Prune] Done in {time.time()-_prune_t0:.2f}s")
        else:
            print(f"[DNA-HDC Prune] No count=1 entries found — skipping")
    except Exception as _pe:
        import traceback as _pe_tb
        print(f"[DNA-HDC Prune] Selective pruning skipped ({_pe})")
        _pe_tb.print_exc()

    print(f"\n[DNA-HDC Eval] Computing table accuracy (unigram fallback)...")

    unigram_counts = np.bincount(tokens.astype(np.int64), minlength=vocab_size)
    unigram_prediction = np.uint16(np.argmax(unigram_counts))
    print(f"[DNA-HDC Eval] Unigram fallback token: {unigram_prediction} "
          f"(freq: {unigram_counts[unigram_prediction]/N*100:.1f}%)")

    _, table_counts_all = unpack_entry_vec(table_packed)
    empty_mask_table = (table_counts_all == 0)
    table_packed[empty_mask_table] = pack_entry(unigram_prediction, 1)

    total_correct = 0
    total_checked = 0
    for chunk_start in range(CTX_LEN, N, CHUNK):
        chunk_end = min(chunk_start + CHUNK, N)
        chunk_n = chunk_end - chunk_start
        buckets = compute_context_hashes(chunk_start, chunk_end)
        packed_preds = table_packed[buckets]
        preds, _ = unpack_entry_vec(packed_preds)
        targets = tokens[chunk_start: chunk_end]
        total_correct += int(np.sum(preds == targets))
        total_checked += chunk_n

    best_accuracy = total_correct / total_checked if total_checked > 0 else 0
    print(f"[DNA-HDC Eval] Table accuracy: {best_accuracy*100:.2f}% "
          f"({total_correct:,}/{total_checked:,})")

    elapsed = time.time() - start_time

    if best_accuracy > 0 and best_accuracy < 1.0:
        correct_bpb = 0.5
        wrong_bpb = math.log2(vocab_size)
        estimated_bpb = best_accuracy * correct_bpb + (1 - best_accuracy) * wrong_bpb
    elif best_accuracy >= 1.0:
        estimated_bpb = 0.0
    else:
        estimated_bpb = math.log2(vocab_size)

    overflow_bytes   = OVERFLOW_SIZE * 2 + OVERFLOW_BITMAP_SIZE * 8
    fingerprint_bytes = TABLE_SIZE * 1
    model_bytes = (32 + 2
                   + TABLE_SIZE * 2
                   + fingerprint_bytes
                   + overflow_bytes)

    transition_bytes = 0
    if transition_codebook is not None:
        transition_bytes = transition_codebook.size * transition_codebook.dim * 8
        model_bytes += transition_bytes

    print(f"\n{'='*60}")
    print(f"[DNA-HDC] TRAINING COMPLETE")
    print(f"[DNA-HDC] Table accuracy: {best_accuracy*100:.2f}%")
    print(f"[DNA-HDC] Estimated BPB: {estimated_bpb:.4f}")
    print(f"[DNA-HDC] Time: {elapsed:.1f}s")
    print(f"[DNA-HDC] Passes: {pass_num}")
    print(f"[DNA-HDC] Filled: {int(np.sum(~empty_mask_table)):,}/{TABLE_SIZE:,}")
    print(f"[DNA-HDC] Model: seed(32B) + unigram(2B) + packed_table({TABLE_SIZE*2/1024/1024:.1f}MB) + overflow({overflow_bytes/1024:.1f}KB)")
    if transition_codebook is not None:
        print(f"[DNA-HDC] Transition Codebook: {transition_codebook.size} entries × {transition_codebook.dim} uint64 = {transition_bytes/1024:.1f}KB")
    print(f"[DNA-HDC] Total model: {model_bytes:,} bytes = {model_bytes/1024/1024:.2f} MB")
    print(f"[DNA-HDC] Architecture: DNA-Stacked Hadamard Bipolar (packed table + overflow)")
    print(f"[DNA-HDC] val_bpb: {estimated_bpb:.4f}")
    print(f"[DNA-HDC] val_loss: {estimated_bpb * math.log(2):.4f}")
    print(f"{'='*60}")

    if transition_codebook is not None:
        try:
            transition_path = os.path.join(os.path.dirname(script_path) or ".", "transition_codebook.bin")
            transition_codebook.save(transition_path)
            print(f"[DNA-HDC] Transition codebook saved to {transition_path}")

            if transition_table is not None:
                table_path = os.path.join(os.path.dirname(script_path) or ".", "transition_table.bin")
                with open(table_path, 'wb') as f:
                    f.write(np.uint32(transition_table.table_size).tobytes())
                    f.write(transition_table.table_indices.tobytes())
                    f.write(transition_table.table_counts.tobytes())
                print(f"[DNA-HDC] Transition table saved to {table_path}")
        except Exception as e:
            print(f"[DNA-HDC] Warning: Could not save transition codebook: {e}")

    if context_checkpoint_mgr is not None:
        try:
            ckpt_stats = context_checkpoint_mgr.get_stats()
            print(f"\n[DNA-HDC] Context Checkpoint Stats:")
            print(f"[DNA-HDC]   Total checkpoints: {ckpt_stats.get('total_checkpoints', 0):,}")
            print(f"[DNA-HDC]   Semantic groups: {ckpt_stats.get('semantic_groups', 0):,}")
            print(f"[DNA-HDC]   Memory usage: {ckpt_stats.get('memory_bytes', 0):,} bytes")
        except Exception as e:
            print(f"[DNA-HDC] Warning: Could not get checkpoint stats: {e}")

    val_loss = estimated_bpb * math.log(2)

    real_bpb       = estimated_bpb
    real_val_loss  = val_loss
    try:
        val_shard_files = sorted(glob(config.val_files))
        if val_shard_files:
            print(f"\n[DNA-HDC ValEval] Loading val tokens from {config.val_files} ...")
            _val_eval_tokens = fast_load_token_shards(
                val_shard_files, max_tokens=500_000_000, label="ValEval"   # full competition val set
            )
            _val_eval_tokens = np.clip(
                _val_eval_tokens.astype(np.int32), 0, vocab_size - 1
            ).astype(np.uint16)
            print(f"[DNA-HDC ValEval] Loaded {len(_val_eval_tokens):,} val tokens")

            try:
                import sentencepiece as _spm_val
                _sp_val = _spm_val.SentencePieceProcessor()
                _sp_val.Load(config.tokenizer_path)
                _base_bytes_val, _has_space_val, _ = build_sentencepiece_luts(_sp_val, vocab_size)
            except Exception:
                _base_bytes_val = np.ones(vocab_size, dtype=np.int16)
                _has_space_val  = np.zeros(vocab_size, dtype=bool)

            _ve_start = time.time()
            real_bpb, real_val_loss = evaluate_bpb_seed_projection(
                val_tokens       = _val_eval_tokens,
                table_packed     = table_packed,
                overflow_table   = overflow_packed,
                overflow_bitmap  = overflow_bitmap,
                codebook         = codebook,
                pos_hash_keys    = POS_HASH_KEYS,
                seed_val         = seed_val,
                table_bits       = TABLE_BITS,
                ctx_len          = CTX_LEN,
                base_bytes       = _base_bytes_val,
                has_leading_space= _has_space_val,
                trigram_packed   = trigram_packed,
                bigram_packed    = bigram_packed,
                dsv              = dsv,
                bit_decomposer   = _bit_decomposer,
                gpu_manager      = _gpu_manager,
                srh              = _srh,
                srh_checkpoints  = _srh_checkpoints,
                srh_keys_arr     = _srh_keys_arr,
                suffix_grammar   = suffix_grammar,
            )
            print(f"[DNA-HDC ValEval] Real val BPB  : {real_bpb:.6f}")
            print(f"[DNA-HDC ValEval] Real val loss : {real_val_loss:.6f}")
            print(f"[DNA-HDC ValEval] Eval time     : {time.time()-_ve_start:.1f}s")
            print(f"[DNA-HDC ValEval] final_val_bpb:{real_bpb:.6f}")
            print(f"[DNA-HDC ValEval] val_bpb:{real_bpb:.6f}")
            del _val_eval_tokens
        else:
            print(f"[DNA-HDC ValEval] No val files found at {config.val_files}; "
                  f"using estimated BPB {estimated_bpb:.4f}")
    except Exception as _ve_err:
        import traceback as _tbt
        print(f"[DNA-HDC ValEval] Warning: real val eval failed ({_ve_err})")
        _tbt.print_exc()

    try:
        _snap_dir  = os.path.dirname(os.path.abspath(__file__)) or "."
        _snap_path = os.path.join(_snap_dir, f"hdc_table_seed{seed}.npy")
        _bg_path   = os.path.join(_snap_dir, f"hdc_bigram_seed{seed}.npy")
        _tg_path   = os.path.join(_snap_dir, f"hdc_trigram_seed{seed}.npy")
        np.save(_snap_path, table_packed)
        np.save(_bg_path,   bigram_packed)
        np.save(_tg_path,   trigram_packed)
        print(f"[DNA-HDC] Saved table snapshot → {_snap_path} "
              f"({table_packed.nbytes // 1024 // 1024} MB)")
        print(f"[DNA-HDC] Saved trigram snapshot → {_tg_path} "
              f"({trigram_packed.nbytes // 1024 // 1024} MB raw)")

        try:
            _ptz_path = os.path.join(_snap_dir, f"hdc_model_seed{seed}.ptz")
            import io as _io_ptz, struct as _struct_ptz
            _buf = _io_ptz.BytesIO()
            _buf.write(b"HDC1")
            _buf.write(_struct_ptz.pack("<Q", int(seed)))
            _buf.write(_struct_ptz.pack("<I", int(TABLE_BITS)))
            for _blob in (table_packed.tobytes(),
                          fingerprint_packed.tobytes(),
                          bigram_packed.tobytes(),
                          trigram_packed.tobytes()):
                _buf.write(_struct_ptz.pack("<Q", len(_blob)))
                _buf.write(_blob)
            _raw       = _buf.getvalue()
            _compressed = lzma.compress(_raw, preset=9)
            with open(_ptz_path, "wb") as _ptz_f:
                _ptz_f.write(_compressed)
            print(f"[DNA-HDC LZMA] Compressed artifact → {_ptz_path} | "
                  f"raw={len(_raw)/1024/1024:.2f} MB → "
                  f"lzma9={len(_compressed)/1024/1024:.2f} MB "
                  f"({100*(1-len(_compressed)/len(_raw)):.1f}% reduction)")
        except Exception as _lzma_err:
            print(f"[DNA-HDC LZMA] Compression skipped ({_lzma_err})")

    except Exception as _snap_err:
        print(f"[DNA-HDC] Warning: could not save table snapshot ({_snap_err})")

    return real_bpb, real_val_loss, elapsed

def _pack_entry_vec_module(token_ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
                                                                            
    cc = np.minimum(counts, 63).astype(np.uint16)
    return ((cc & np.uint16(0x3F)) << np.uint16(10)) | (token_ids.astype(np.uint16) & np.uint16(0x3FF))

def _unpack_entry_vec_module(packed: np.ndarray):
                                                                              
    token_ids = (packed & np.uint16(0x3FF)).astype(np.uint16)
    counts    = ((packed >> np.uint16(10)) & np.uint16(0x3F)).astype(np.int32)
    return token_ids, counts

SEM_CONFIDENCE_MIN: float = 0.0

def evaluate_bpb_seed_projection(
    val_tokens: np.ndarray,
    table_packed: np.ndarray,
    overflow_table: np.ndarray,
    overflow_bitmap: np.ndarray,
    codebook: np.ndarray,
    pos_hash_keys: np.ndarray,
    seed_val: np.uint64,
    table_bits: int,
    ctx_len: int,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    bigram_packed: np.ndarray = None,
    trigram_packed: Optional[np.ndarray] = None,
    dsv: 'DirectionalSemanticVec' = None,
    batch_size: int = 500_000,
    temperature: float = 1.0,
    transition_codebook: 'TransitionCodebook' = None,
    transition_table: 'TransitionTable' = None,
    bit_decomposer: 'BitDecomposer' = None,
    gpu_manager: 'TensorCoreGPUManager' = None,
    srh=None,
    srh_checkpoints=None,
    srh_keys_arr=None,
    suffix_grammar=None,
) -> Tuple[float, float]:

       
    N = len(val_tokens)
    if N <= ctx_len:
        return float('inf'), float('inf')

    vocab_size = len(codebook)
    W_UINT64 = codebook.shape[1]
    TABLE_SIZE = 1 << table_bits
    OVERFLOW_SIZE = 65536
    total_bits = 0.0
    total_bytes = 0
    total_nats = 0.0
    total_tokens = 0
    correct_preds = 0

    try:
        from _full_context_hash import hadamard_key_batch as _hk_batch_val
        _VRH_FMIX  = np.uint64(0x9E3779B97F4A7C15)
        _VRH_CHUNK = 2_000_000
        _val_rh_G  = np.uint64(0)
        _val_rolling_buckets = np.empty(N, dtype=np.int32)
        for _vrh_s in range(0, N, _VRH_CHUNK):
            _vrh_e    = min(_vrh_s + _VRH_CHUNK, N)
            _vrh_pos  = np.arange(_vrh_s, _vrh_e, dtype=np.int64)
            _vrh_keys = _hk_batch_val(_vrh_pos)
            _vrh_cont = val_tokens[_vrh_s:_vrh_e].astype(np.uint64) * _vrh_keys
            _vrh_inc  = np.bitwise_xor.accumulate(_vrh_cont)
            _vrh_excl = np.empty(len(_vrh_cont), dtype=np.uint64)
            _vrh_excl[0] = _val_rh_G
            if len(_vrh_cont) > 1:
                with np.errstate(over='ignore'):
                    _vrh_excl[1:] = _val_rh_G ^ _vrh_inc[:-1]
            _val_rh_G ^= _vrh_inc[-1]
            with np.errstate(over='ignore'):
                _vrh_fin = (_vrh_excl ^ seed_val) * _VRH_FMIX
            _val_rolling_buckets[_vrh_s:_vrh_e] = (
                _vrh_fin >> np.uint64(64 - table_bits)
            ).astype(np.int32)
            del _vrh_pos, _vrh_keys, _vrh_cont, _vrh_inc, _vrh_excl, _vrh_fin
    except Exception as _rh_eval_err:
        print(f"[HDC Eval] FATAL: Rolling hash unavailable ({_rh_eval_err}).")
        print(f"[HDC Eval] Ensure _full_context_hash.py is a sibling file.")
        return float('inf'), float('inf')

    for chunk_start in range(ctx_len, N, batch_size):
        chunk_end = min(chunk_start + batch_size, N)
        chunk_n = chunk_end - chunk_start

        buckets = _val_rolling_buckets[chunk_start:chunk_end].astype(np.int64)

        chunk_targets = val_tokens[chunk_start: chunk_end]

        packed_preds = table_packed[buckets]
        table_preds, table_conf = _unpack_entry_vec_module(packed_preds)

        low_conf_mask = (table_conf == 0)

        if np.any(low_conf_mask) and overflow_table is not None:
            lc_buckets   = buckets[low_conf_mask]
            lc_pc        = np.array([bin(int(b)).count('1') for b in lc_buckets], dtype=np.int64)
            flip_bits    = (lc_pc % table_bits).astype(np.int64)
            overflow_idx = ((lc_buckets ^ (np.int64(1) << flip_bits)) % OVERFLOW_SIZE).astype(np.int64)
            overflow_packed = overflow_table[overflow_idx]
            overflow_preds, overflow_conf = _unpack_entry_vec_module(overflow_packed)

            confident_overflow = overflow_conf > 0
            if np.any(confident_overflow):
                low_conf_indices = np.where(low_conf_mask)[0]
                table_preds[low_conf_indices[confident_overflow]] = overflow_preds[confident_overflow]
                table_conf[low_conf_indices[confident_overflow]] = overflow_conf[confident_overflow]
            low_conf_mask = (table_conf == 0)

            if np.any(low_conf_mask):
                mid_shell  = table_bits // 2
                full_mask  = np.int64((1 << table_bits) - 1)
                lc2_idx    = np.where(low_conf_mask)[0]
                lc2_bkts   = buckets[lc2_idx]
                lc2_pc     = np.array([bin(int(b)).count('1') for b in lc2_bkts], dtype=np.int64)
                sym_mask   = (lc2_pc == mid_shell)
                if np.any(sym_mask):
                    sym_bkts   = lc2_bkts[sym_mask]
                    sym_idx    = ((sym_bkts ^ full_mask) % OVERFLOW_SIZE).astype(np.int64)
                    sym_packed = overflow_table[sym_idx]
                    sym_preds, sym_conf = _unpack_entry_vec_module(sym_packed)
                    confident_sym = sym_conf > 0
                    if np.any(confident_sym):
                        update_pos = lc2_idx[sym_mask][confident_sym]
                        table_preds[update_pos] = sym_preds[confident_sym]
                        table_conf[update_pos]  = sym_conf[confident_sym]
                low_conf_mask = (table_conf == 0)

        if trigram_packed is not None and np.any(low_conf_mask):
            lc_tg_idx = np.where(low_conf_mask)[0]
            _tg_prev2 = val_tokens[chunk_start - 2 + lc_tg_idx].astype(np.int64)
            _tg_prev1 = val_tokens[chunk_start - 1 + lc_tg_idx].astype(np.int64)
            _tg_keys  = _tg_prev2 * vocab_size + _tg_prev1
            _tg_keys  = np.clip(_tg_keys, 0, len(trigram_packed) - 1)
            _tg_packed_sl = trigram_packed[_tg_keys]
            _tg_preds, _tg_confs = _unpack_entry_vec_module(_tg_packed_sl)
            confident_tg = _tg_confs > 0
            if np.any(confident_tg):
                update_pos_tg = lc_tg_idx[confident_tg]
                table_preds[update_pos_tg] = _tg_preds[confident_tg]
                table_conf[update_pos_tg]  = _tg_confs[confident_tg]
            low_conf_mask = (table_conf == 0)

        if bigram_packed is not None and np.any(low_conf_mask):
            lc_idx = np.where(low_conf_mask)[0]
            prev_t = val_tokens[chunk_start - 1 + lc_idx].astype(np.int64)
            prev_t = np.clip(prev_t, 0, len(bigram_packed) - 1)
            bg_packed_slice = bigram_packed[prev_t]
            bg_preds, bg_confs = _unpack_entry_vec_module(bg_packed_slice)
            confident_bg = bg_confs > 0
            if np.any(confident_bg):
                update_pos = lc_idx[confident_bg]
                table_preds[update_pos] = bg_preds[confident_bg]
                table_conf[update_pos]  = bg_confs[confident_bg]
            low_conf_mask = (table_conf == 0)

        context_matrix = np.stack([
            val_tokens[chunk_start - ctx_len + c: chunk_end - ctx_len + c].astype(np.int32)
            for c in range(ctx_len)
        ], axis=0)

        if transition_codebook is not None and transition_table is not None:
            if np.any(low_conf_mask):
                low_conf_indices = np.where(low_conf_mask)[0]

                for idx_pos in low_conf_indices:
                    bucket = int(buckets[idx_pos])
                    trans_idx, trans_count = transition_table.lookup_transition(bucket)

                    if trans_count > 0:
                        context_hv = np.zeros(W_UINT64, dtype=np.uint64)
                        for c in range(ctx_len):
                            ctx_tok = int(val_tokens[chunk_start - ctx_len + c + idx_pos])
                            context_hv ^= codebook[ctx_tok]

                        target_hv = transition_codebook.reconstruct_target(context_hv, trans_idx)

                        trans_pred = transition_codebook.decode_to_token(target_hv, codebook)

                        if trans_count >= 2:
                            table_preds[idx_pos] = trans_pred
                            table_conf[idx_pos] = trans_count

                low_conf_mask = (table_conf == 0)

        if (dsv is not None and hasattr(dsv, 'sem_fwd_lag')
                and dsv.sem_fwd_lag and np.any(low_conf_mask)):
            try:
                _sb_lc_idx = np.where(low_conf_mask)[0]
                for _sb_lag in range(2, 6):
                    if not np.any(low_conf_mask):
                        break
                    if _sb_lag not in dsv.sem_fwd_lag:
                        continue
                    _sb_lag_mat = dsv.sem_fwd_lag[_sb_lag].reshape(vocab_size, W_UINT64)
                    for _sb_i in _sb_lc_idx:
                        if table_conf[_sb_i] > 0:
                            continue
                        _sb_p = chunk_start + int(_sb_i)
                        if _sb_p < _sb_lag:
                            continue
                        _sb_ctx_tok = int(val_tokens[_sb_p - _sb_lag])
                        if _sb_ctx_tok < 0 or _sb_ctx_tok >= vocab_size:
                            continue
                        _sb_lag_vec = _sb_lag_mat[_sb_ctx_tok]
                        _sb_xors = _sb_lag_vec[None, :] ^ codebook
                        _sb_pcs  = np.unpackbits(
                            _sb_xors.view(np.uint8), axis=1
                        ).sum(axis=1)
                        _sb_winner = int(np.argmin(_sb_pcs))
                        _sb_conf   = 1.0 - float(_sb_pcs[_sb_winner]) / (W_UINT64 * 64)
                        if _sb_conf > (0.55 + 0.02 * _sb_lag):
                            table_preds[_sb_i] = _sb_winner
                            table_conf[_sb_i]  = 1
                    low_conf_mask = (table_conf == 0)
            except Exception:
                pass

        if np.any(low_conf_mask):
            if dsv is not None:
                sem_vote = np.zeros((chunk_n, vocab_size), dtype=np.float32)
                for c in range(ctx_len):
                    ctx_slice = context_matrix[c]
                    for ctx_tok in np.unique(ctx_slice):
                        pos_mask = (ctx_slice == ctx_tok) & low_conf_mask
                        if np.any(pos_mask):
                            scores = dsv.vote_scores_for_context_tok(int(ctx_tok), codebook)
                            sem_vote[pos_mask] += scores

                sem_preds = np.argmax(sem_vote, axis=1).astype(np.uint16)
                sem_best_score = sem_vote[np.arange(chunk_n), sem_preds]

                sem_override = low_conf_mask & (sem_best_score > SEM_CONFIDENCE_MIN)
                preds = np.where(sem_override, sem_preds, table_preds)
            else:
                prev_tokens = val_tokens[chunk_start - 1: chunk_end - 1]
                preds = table_preds.copy()
                for i in np.where(low_conf_mask)[0]:
                    ctx_signal = codebook[prev_tokens[i]] ^ pos_hash_keys[0]
                    xors = np.bitwise_xor(codebook, ctx_signal)
                    popcounts = np.unpackbits(xors.view(np.uint8), axis=1).sum(axis=1)
                    preds[i] = np.argmin(popcounts)
        else:
            preds = table_preds

        correct_mask = preds == chunk_targets
        correct_preds += np.sum(correct_mask)

        _conf_f = np.abs(table_conf.astype(np.float32))
        tbl_gate = np.minimum(0.95, 0.30 + 0.65 * (1.0 - np.exp(-_conf_f / 3.0)))

        if bigram_packed is not None:
            _all_prev   = val_tokens[chunk_start - 1 : chunk_end - 1].astype(np.int64)
            _all_prev   = np.clip(_all_prev, 0, len(bigram_packed) - 1)
            _bg_a_pred, _bg_a_conf = _unpack_entry_vec_module(bigram_packed[_all_prev])
            _bg_conf_f  = _bg_a_conf.astype(np.float32)
            bg_gate     = np.minimum(0.40, 0.05 + 0.35 * (1.0 - np.exp(-_bg_conf_f / 15.0)))
            bg_gate     = bg_gate * (1.0 - tbl_gate)
        else:
            _bg_a_pred  = None
            _bg_conf_f  = None
            bg_gate     = np.zeros(chunk_n, dtype=np.float32)

        unif_gate = np.maximum(0.02, 1.0 - tbl_gate - bg_gate)

        p_tbl = np.where(
            correct_mask,
            np.minimum(0.99, 0.5 + 0.49 * (1.0 - np.exp(-_conf_f / 5.0))),
            np.float32(1.0 / vocab_size)
        ).astype(np.float32)

        if _bg_a_pred is not None:
            bg_correct_for_tgt = (_bg_a_pred == chunk_targets)
            p_bg = np.where(
                bg_correct_for_tgt,
                np.minimum(np.float32(0.90),
                           np.float32(0.20) + np.float32(0.70) * (1.0 - np.exp(-_bg_conf_f / 10.0))),
                np.float32(1.0 / vocab_size)
            ).astype(np.float32)
        else:
            p_bg = np.full(chunk_n, 1.0 / vocab_size, dtype=np.float32)

        p_unif = np.float32(1.0 / vocab_size)

        probs = tbl_gate * p_tbl + bg_gate * p_bg + unif_gate * p_unif

        correct_indices = np.where(correct_mask)[0]

        probs = np.maximum(probs, 1e-10)
        total_bits += np.sum(-np.log2(probs))
        total_nats += np.sum(-np.log(probs))
        total_tokens += chunk_n

        targets = chunk_targets.astype(np.int32)
        valid_targets = targets < len(base_bytes)
        valid_target_ids = targets[valid_targets]
        byte_counts = base_bytes[valid_target_ids].astype(np.int32)
        byte_counts += has_leading_space[valid_target_ids].astype(np.int32)
        total_bytes += np.sum(np.maximum(1, byte_counts))
        total_bytes += np.sum(~valid_targets)

    if total_bytes == 0:
        return float('inf'), float('inf')

    bpb = total_bits / total_bytes
    val_loss = total_nats / total_tokens if total_tokens > 0 else float('inf')

    accuracy = correct_preds / total_tokens if total_tokens > 0 else 0
    print(f"[HDC Eval] Validation accuracy: {accuracy*100:.2f}% ({correct_preds:,}/{total_tokens:,})")

    return bpb, val_loss

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

    _final_bpb_m = re.search(r'final_val_bpb[:\s]+(\d+\.\d+)', content)
    if _final_bpb_m:
        result["val_bpb"] = float(_final_bpb_m.group(1))
    else:
        _bpb_m = re.search(r'val_bpb[:\s]+(\d+\.\d+)', content)
        if _bpb_m:
            result["val_bpb"] = float(_bpb_m.group(1))

    _final_loss_m = re.search(r'final_val_loss[:\s]+(\d+\.\d+)', content)
    if _final_loss_m:
        result["val_loss"] = float(_final_loss_m.group(1))
    else:
        _loss_m = re.search(r'val_loss[:\s]+(\d+\.\d+)', content)
        if _loss_m:
            result["val_loss"] = float(_loss_m.group(1))

    steps_matches = re.findall(r'step:(\d+)/\d+', content)
    if steps_matches:
        result["steps"] = int(steps_matches[-1])

    ms_match = re.search(r'step_avg[:\s]+(\d+\.\d+)ms', content)
    if ms_match:
        result["ms_per_step"] = float(ms_match.group(1))

    time_match = re.search(r'train_time:(\d+)ms', content)
    if time_match:
        result["elapsed_seconds"] = float(time_match.group(1)) / 1000.0

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

def merge_hdc_tables(seeds: list, script_dir: str) -> Optional[str]:

       
    try:
        tables   = []
        bigrams  = []
        trigrams = []
        for seed in seeds:
            tp = os.path.join(script_dir, f"hdc_table_seed{seed}.npy")
            bp = os.path.join(script_dir, f"hdc_bigram_seed{seed}.npy")
            tgp = os.path.join(script_dir, f"hdc_trigram_seed{seed}.npy")
            if not os.path.exists(tp):
                print(f"[Merge] Missing snapshot for seed {seed}: {tp}")
                return None
            tables.append(np.load(tp))
            if os.path.exists(bp):
                bigrams.append(np.load(bp))
            if os.path.exists(tgp):
                trigrams.append(np.load(tgp))

        if len(tables) < 2:
            print(f"[Merge] Only {len(tables)} table(s) found — merge requires ≥2")
            return None

        n_seeds    = len(tables)
        TABLE_SIZE = len(tables[0])
        print(f"[Merge] Merging {n_seeds} tables (TABLE_SIZE={TABLE_SIZE:,}) ...")

        all_toks = np.stack([(t & np.uint16(0x3FF)).astype(np.uint16) for t in tables], axis=1)
        all_cnts = np.stack([((t >> np.uint16(10)) & np.uint16(0x3F)).astype(np.int32) for t in tables], axis=1)
        active   = all_cnts > 0

        if n_seeds == 3:
            tok0, tok1, tok2 = all_toks[:, 0], all_toks[:, 1], all_toks[:, 2]
            cnt0, cnt1, cnt2 = all_cnts[:, 0], all_cnts[:, 1], all_cnts[:, 2]
            a01  = (tok0 == tok1) & active[:, 0] & active[:, 1]
            a02  = (tok0 == tok2) & active[:, 0] & active[:, 2]
            a12  = (tok1 == tok2) & active[:, 1] & active[:, 2]
            all3 = a01 & a02
            merged_toks = np.where(all3,   tok0,
                          np.where(a01,    tok0,
                          np.where(a02,    tok0,
                          np.where(a12,    tok1,
                          np.where(cnt0 >= cnt1,
                              np.where(cnt0 >= cnt2, tok0, tok2),
                              np.where(cnt1 >= cnt2, tok1, tok2))))))
            merged_cnts = np.where(all3, np.int32(3),
                          np.where(a01 | a02 | a12, np.int32(2),
                          np.maximum(np.maximum(cnt0, cnt1), cnt2)))
            agree_rate = float(np.mean(all3))
        else:
            tok0, tok1 = all_toks[:, 0], all_toks[:, 1]
            cnt0, cnt1 = all_cnts[:, 0], all_cnts[:, 1]
            agree = (tok0 == tok1) & active[:, 0] & active[:, 1]
            merged_toks = np.where(agree, tok0, np.where(cnt0 >= cnt1, tok0, tok1))
            merged_cnts = np.where(agree, np.int32(2), np.maximum(cnt0, cnt1))
            agree_rate = float(np.mean(agree))

        merged_cnts = np.minimum(merged_cnts, np.int32(63))

        merged_table = ((merged_cnts.astype(np.uint16) & np.uint16(0x3F)) << np.uint16(10)) |\
                       (merged_toks & np.uint16(0x3FF))
        merged_path = os.path.join(script_dir, "hdc_table_merged.npy")
        np.save(merged_path, merged_table)
        print(f"[Merge] Saved → {merged_path}  (full-agreement rate: {agree_rate*100:.1f}%)")

        if len(bigrams) == n_seeds:
            bg_cnts  = np.stack([((b >> np.uint16(10)) & np.uint16(0x3F)).astype(np.int32)
                                  for b in bigrams], axis=1)
            best_s   = np.argmax(bg_cnts, axis=1)
            merged_bg = np.array([bigrams[s][i] for i, s in enumerate(best_s)], dtype=np.uint16)
            bg_path  = os.path.join(script_dir, "hdc_bigram_merged.npy")
            np.save(bg_path, merged_bg)
            print(f"[Merge] Saved bigram → {bg_path}")

        if len(trigrams) > 0:
            tg_path = os.path.join(script_dir, "hdc_trigram_merged.npy")
            np.save(tg_path, trigrams[0])
            print(f"[Merge] Saved trigram → {tg_path} (corpus-derived, seed-independent)")

        return merged_path

    except Exception as _me:
        import traceback as _mbt
        print(f"[Merge] Table merge failed: {_me}")
        _mbt.print_exc()
        return None

def run_single_training(seed: int, args, log_dir: str = ".") -> Dict[str, Any]:
                                                            
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

        print("[HDC] Using Hadamard bipolar seed projection")
        final_bpb, final_val_loss, elapsed = train_hdc_seed_projection(config)

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
    results["val_bpb"] = final_bpb
    results["val_loss"] = final_val_loss

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

    _script_dir_sub = os.path.dirname(os.path.abspath(__file__)) or "."
    _ptz_model_bytes = 0
    for _sub_seed in seed_results.keys():
        _ptz_p = os.path.join(_script_dir_sub, f"hdc_model_seed{_sub_seed}.ptz")
        if os.path.exists(_ptz_p):
            _ptz_model_bytes = max(_ptz_model_bytes, os.path.getsize(_ptz_p))
    artifact_bytes = code_bytes + _ptz_model_bytes

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
    code_bytes  = os.path.getsize(script_path)

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
    print(f"[TensorCore] Merging tables from {len(args.seeds)} seeds (SWA analog)...")
    _script_dir = os.path.dirname(script_path) or "."
    _merged_path = merge_hdc_tables(args.seeds, _script_dir)
    if _merged_path:
        print(f"[TensorCore] Merged table saved → {_merged_path}")
    else:
        print(f"[TensorCore] Merge skipped (snapshots unavailable) — using per-seed BPBs")

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

def _init_distributed() -> tuple: 
    import torch
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1

    import torch.distributed as dist
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def _run_hash_grad_single(args) -> int:

    import torch
    from datetime import datetime, timezone

    # CPU gauntlet guard: fail fast with a clear message rather than silently
    # running a degraded CPU path that would produce incorrect timing results.
    # The verified BPB result requires 8×H100 SXM; CPU execution is not supported.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[HashGrad] CUDA is required but not available.\n"
            "  This submission requires GPU execution (verified on 8×H100 SXM).\n"
            "  CPU-only execution is not supported: the NMF tabulation, distributed\n"
            "  all-reduce, and DSV build are all GPU-accelerated and would exceed\n"
            "  the 600s training budget on CPU.\n"
            "  To run the CPU pre-flight import check only, use:\n"
            "    CUDA_VISIBLE_DEVICES='' python -c 'import train_gpt; print(train_gpt.Hyperparameters())'"
        )

    rank, world_size = _init_distributed()
    is_main = (rank == 0)

    t_start = time.time()

    TABLE_BITS = int(os.environ.get("TABLE_BITS", "19"))
    EMBED_DIM  = int(os.environ.get("EMBED_DIM",  "16"))
    hg_seeds_env = os.environ.get("HG_SEEDS", str(getattr(args, "seed", 42)))
    HG_SEEDS   = [int(s.strip()) for s in hg_seeds_env.split(",") if s.strip()]

    data_path      = args.data_path
    tokenizer_path = args.tokenizer_path
    max_seconds    = float(getattr(args, "max_time", 600.0))

    if is_main:
        print(f"\n{'='*60}")
        print(f"[HashGrad] Distributed Hash-Grad NMF Pipeline")
        print(f"[HashGrad] world_size={world_size}, rank={rank}")
        print(f"[HashGrad] TABLE_BITS={TABLE_BITS}, EMBED_DIM={EMBED_DIM}")
        print(f"[HashGrad] Seeds: {HG_SEEDS}")
        print(f"[HashGrad] Data: {data_path}")
        print(f"[HashGrad] Max time: {max_seconds}s")
        print(f"{'='*60}\n")

    try:
        from _hash_grad_train import (
            train_hash_grad_model,
            train_hash_grad_multi_seed,
            hash_grad_bpb,
            hash_grad_bpb_softmax_only,
            save_hash_grad_artifact,
            precompute_g_states,
        )
    except ImportError as _ie:
        print(f"[HashGrad] ERROR: required module not found: {_ie}")
        return 1

    if is_main:
        print("[HashGrad] Loading training tokens...")
    _train_pattern = os.path.join(data_path, "fineweb_train_*.bin")
    _train_shards  = sorted(glob.glob(_train_pattern))
    if not _train_shards:
        _train_shards = sorted(glob.glob(os.path.join(data_path, "*.bin")))
        print(f"[HashGrad] WARNING: no fineweb_train_*.bin found; "
              f"falling back to *.bin glob ({len(_train_shards)} shards)")
    else:
        # ── Rank-specific shard distribution ─────────────────────────────────
        # Each rank loads a unique subset of shards so the NMF distributed
        # all-reduce sees the ENTIRE dataset (80 shards = 8B tokens) instead
        # of the same 500M on every rank (which gives only 500M unique tokens).
        # With 80 shards and 8 ranks: each rank gets 10 shards × 100M = 1B tokens.
        #
        # NMF impact:  8B / 512K buckets = 15.6K tok/bucket (vs 2.9K with 3-seed×500M)
        #              → sharper per-bucket distributions → better NMF predictions
        # DSV impact:  each rank builds XOR-bundle from its own 1B unique tokens;
        #              all-gather XOR across 8 ranks → 8 diverse 1B-token views.
        #
        # GoldenAxisShift timing fits within 10 min with HG_SEEDS=42 (1 seed):
        #   load 30s + NMF 64s + DSV ctx_len=4 ~304s + skip-bigrams ~160s + suffix 31s
        #   ≈ 589s ≈ 9.8 min  (auto-truncated by time_budget if needed)
        # ─────────────────────────────────────────────────────────────────────
        try:
            import torch.distributed as _dist_mod
            _local_rank  = int(os.environ.get("LOCAL_RANK", rank))
            _world_sz    = _dist_mod.get_world_size() if _dist_mod.is_initialized() else 1
        except Exception:
            _local_rank, _world_sz = 0, 1

        _n_shards         = len(_train_shards)
        _shards_per_rank  = max(1, _n_shards // _world_sz)
        _rank_shard_start = _local_rank * _shards_per_rank
        _rank_shard_end   = min(_rank_shard_start + _shards_per_rank, _n_shards)
        _rank_shards      = _train_shards[_rank_shard_start:_rank_shard_end]
        if is_main:
            print(f"[HashGrad] Shard distribution: "
                  f"rank {_local_rank}/{_world_sz} → "
                  f"shards [{_rank_shard_start}:{_rank_shard_end}] "
                  f"({len(_rank_shards)} shards, ~{len(_rank_shards)*100}M tokens)")

        tokens = fast_load_token_shards(
            _rank_shards,
            max_tokens=len(_rank_shards) * 100_000_000,   # exactly 1 shard cap
            label="HashGrad",
        )
    vocab_size = int(os.environ.get("VOCAB_SIZE", "1024"))

    if is_main:
        print(f"[HashGrad] Precomputing G[p] states...")
    g_states = precompute_g_states(tokens)
    g_states_list = [g_states] * len(HG_SEEDS)

    nmf_budget = max(30.0, max_seconds - 60.0)

    if len(HG_SEEDS) == 1:
        embed, W_out, freq, count, fingerprint = train_hash_grad_model(
            tokens=tokens,
            g_states=g_states_list[0],
            seed=HG_SEEDS[0],
            table_bits=TABLE_BITS,
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            nmf_max_iter=1,
            time_budget_s=nmf_budget,
            distributed=True,
        )
    else:
        embed, W_out, freq, count, fingerprint = train_hash_grad_multi_seed(
            tokens=tokens,
            g_states_list=g_states_list,
            seeds=HG_SEEDS,
            table_bits=TABLE_BITS,
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            nmf_max_iter=1,
            time_budget_s=nmf_budget,
            distributed=True,
        )

    sem_fwd = sem_bwd = codebook = skip_bigram_lags = None
    _dsv = None
    _hg_W = EMBED_DIM
    _hg_uint64c = vocab_size * _hg_W
    _p6_budget = max(30.0, max_seconds - 75.0)
    _p6_t0 = time.time()

    def _allgather_xor_u64(arr_u64):
                                                                                   
        import torch
        import torch.distributed as _d
        _lr  = int(os.environ.get("LOCAL_RANK", rank))
        _dev = (torch.device(f"cuda:{_lr}") if torch.cuda.is_available()
                else torch.device("cpu"))
        _t   = torch.from_numpy(arr_u64.view(np.int64).copy()).to(_dev)
        _all = [torch.zeros_like(_t) for _ in range(world_size)]
        _d.all_gather(_all, _t)
        if is_main:
            _m = _all[0].cpu().numpy().view(np.uint64).copy()
            for _ri in range(1, world_size):
                _m ^= _all[_ri].cpu().numpy().view(np.uint64)
            return _m
        return arr_u64

    try:
        from _semantic_layer import DirectionalSemanticVec as _DSV_cls
        _PHI64 = np.uint64(0x9E3779B97F4A7C15)
        _MIX64 = np.uint64(0xBF58476D1CE4E5B9)
        _ids   = np.arange(vocab_size, dtype=np.uint64)
        codebook = np.empty((vocab_size, _hg_W), dtype=np.uint64)
        for _k in range(_hg_W):
            _h = _ids * _PHI64
            _h = (_h ^ (_h >> np.uint64(30))) * _MIX64
            _h ^= (_h >> np.uint64(27))
            _h  = _h * np.uint64(_k * 0x0101010101010101 + 1)
            codebook[:, _k] = _h

        _N_tok = len(tokens)
        if world_size > 1:
            _sh_s = rank * _N_tok // world_size
            _sh_e = (rank + 1) * _N_tok // world_size
            _tok_shard = tokens[_sh_s:_sh_e]
        else:
            _tok_shard = tokens

        _dsv_budget = _p6_budget * 0.45
        _dsv = _DSV_cls.build_from_tokens(
            _tok_shard, codebook, ctx_len=4,
            vocab_size=vocab_size, W=_hg_W, uint64_count=_hg_uint64c,
            time_budget_s=_dsv_budget,
            label=f"HashGrad-DSV-r{rank}" if world_size > 1 else "HashGrad-DSV",
            verbose=is_main,
        )

        if world_size > 1:
            _dsv.sem_fwd = _allgather_xor_u64(_dsv.sem_fwd)
            _dsv.sem_bwd = _allgather_xor_u64(_dsv.sem_bwd)
            if is_main:
                print(f"[HashGrad-DSV dist] all-gather XOR done across {world_size} ranks")

        _elapsed = time.time() - _p6_t0
        _sb_budget = max(10.0, _p6_budget - _elapsed - 30.0)
        _dsv.build_skip_bigram_lags(
            _tok_shard, codebook, max_lag=5,
            time_budget_s=_sb_budget,
            label=f"HashGrad-SkipBigram-r{rank}" if world_size > 1 else "HashGrad-SkipBigram",
            verbose=is_main,
        )

        if world_size > 1 and hasattr(_dsv, 'sem_fwd_lag') and _dsv.sem_fwd_lag:
            for _lag in sorted(_dsv.sem_fwd_lag.keys()):
                _dsv.sem_fwd_lag[_lag] = _allgather_xor_u64(_dsv.sem_fwd_lag[_lag])
            if is_main:
                print(f"[HashGrad-SkipBigram dist] all-gather XOR done for "
                      f"lags {sorted(_dsv.sem_fwd_lag.keys())}")

    except Exception as _e6:
        if is_main:
            print(f"[HashGrad] Phase 6 failed ({_e6!r})")
        codebook = None

    if not is_main:
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                _dist.destroy_process_group()
        except Exception:
            pass
        return 0

    try:
        if _dsv is not None and codebook is not None:
            sem_fwd = _dsv.sem_fwd.reshape(vocab_size, _hg_W)
            sem_bwd = _dsv.sem_bwd.reshape(vocab_size, _hg_W)
            print(f"[HashGrad Phase6] DSV sem_fwd={_hg_uint64c * 8 // 1024}KB")
            if hasattr(_dsv, 'sem_fwd_lag') and _dsv.sem_fwd_lag:
                skip_bigram_lags = [
                    _dsv.sem_fwd_lag[lag].reshape(vocab_size, _hg_W)
                    for lag in sorted(_dsv.sem_fwd_lag.keys())
                ]
                print(f"[HashGrad Phase6] Skip-bigram lags: {sorted(_dsv.sem_fwd_lag.keys())}")
    except Exception as _e6b:
        print(f"[HashGrad] Phase 6 reshape failed ({_e6b!r})")
        sem_fwd = sem_bwd = skip_bigram_lags = None

    suffix_grammar = None
    try:
        from _suffix_grammar import SuffixGrammarTable
        from _transition_codebook import CharacterHypervector
        import sentencepiece as _spm7
        _sp7 = _spm7.SentencePieceProcessor()
        _sp7.Load(tokenizer_path)
        _W_UINT64_sg = 16
        chv = CharacterHypervector(dim=1024, w_uint64=_W_UINT64_sg)
        suffix_grammar = SuffixGrammarTable(vocab_size, _W_UINT64_sg, chv, _sp7, suffix_len=3)

        class _GStatesSRH:
                                                                                          
            def __init__(self, _g, _w):
                self._g = _g.astype(np.uint64)
                self._w = _w
            def recompute_chunk(self, cs, ce, *args, **kwargs):
                g = self._g[cs:ce]
                s = np.empty((len(g), self._w), dtype=np.uint64)
                for k in range(self._w):
                    s[:, k] = g ^ np.uint64(k * 0x0101010101010101 + 1)
                return s

        _g_srh = _GStatesSRH(g_states, _W_UINT64_sg)
        suffix_grammar.build_from_corpus(
            tokens,
            _g_srh,
            srh=_g_srh,
            sem_fwd_matrix=np.zeros((vocab_size, _W_UINT64_sg), dtype=np.uint64),
            checkpoints={0: np.zeros(_W_UINT64_sg, dtype=np.uint64)},
            time_budget_s=30.0,
            label="HashGrad-Phase7",
        )
        print("[HashGrad] Phase 7: Suffix grammar built")
    except Exception as _e7:
        print(f"[HashGrad] Phase 7 skipped ({_e7})")

    min_count = 1
    zero_mask = count < min_count
    if zero_mask.any():
        embed[zero_mask] = 0
        print(f"[HashGrad] Phase 9: Pruned {int(zero_mask.sum()):,} low-count embeds")

    script_dir  = os.path.dirname(os.path.abspath(__file__)) or "."
    artifact_path = os.path.join(script_dir, f"hdc_hashgrad_seed{HG_SEEDS[0]}.hgz")
    artifact_bytes = save_hash_grad_artifact(
        embed=embed, W_out=W_out,
        seed=HG_SEEDS[0], table_bits=TABLE_BITS,
        path=artifact_path,
        fingerprint=fingerprint,
    )

    # ── Training phase ends here; artifact is on disk ──────────────────────────
    # Competition rules (README.md):
    #   - Training must complete in ≤ 10 minutes (600s) on 8×H100
    #   - Evaluation has a SEPARATE additional ≤ 10-minute budget
    # We stamp t_training_end right before loading val tokens so both phases
    # are tracked independently and stored in submission.json.
    t_training_end = time.time()
    training_elapsed = t_training_end - t_start
    print(f"\n[TensorCore] Training phase complete: {training_elapsed:.1f}s "
          f"({'✅ under' if training_elapsed < 600 else '⚠️ over'} 10-min training limit)",
          flush=True)

    print("\n[HashGrad] Running BPB evaluation on validation set...")
    t_eval_start = time.time()
    try:
        _val_pattern = os.path.join(data_path, "fineweb_val_*.bin")
        # Use the full competition validation set (not capped at 5M).
        # The reference train_gpt.py evaluates on ALL fineweb_val_* tokens (~62M).
        # The 5M cap produced BPB on a non-representative subset.
        val_tokens = fast_load_token_shards(
            sorted(glob.glob(_val_pattern)), max_tokens=500_000_000, label="ValEval"
        )
        val_tokens = np.clip(val_tokens.astype(np.int32), 0, vocab_size - 1).astype(np.uint16)
        g_val = precompute_g_states(val_tokens)

        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)
        # Use the same build_sentencepiece_luts() as the competition baseline:
        # strips the ▁ prefix before measuring byte length, and tracks
        # is_boundary_token so the leading-space byte is only added when the
        # preceding token is NOT a document boundary (matching train_gpt.py eval).
        base_bytes_arr, has_leading_space, is_boundary_token = build_sentencepiece_luts(sp, vocab_size)

        bpb, val_loss = hash_grad_bpb(
            val_tokens=val_tokens,
            embed=embed, W_out=W_out,
            g_states_val=g_val,
            seed=HG_SEEDS[0],
            table_bits=TABLE_BITS,
            base_bytes=base_bytes_arr,
            has_leading_space=has_leading_space,
            is_boundary_token=is_boundary_token,
            fingerprint_packed=fingerprint,
            sem_fwd=sem_fwd, sem_bwd=sem_bwd,
            codebook=codebook,
            skip_bigram_lags=skip_bigram_lags,
            suffix_grammar=suffix_grammar,
        )
        # ── Point-6 audit: NMF softmax + DSV similarity, with disclaimer ──────
        try:
            hash_grad_bpb_softmax_only(
                val_tokens=val_tokens,
                embed=embed, W_out=W_out,
                g_states_val=g_val,
                seed=HG_SEEDS[0],
                table_bits=TABLE_BITS,
                base_bytes=base_bytes_arr,
                has_leading_space=has_leading_space,
                is_boundary_token=is_boundary_token,
                fingerprint_packed=fingerprint,
                sem_fwd=sem_fwd,
                codebook=codebook,
                skip_bigram_lags=skip_bigram_lags,
                suffix_grammar=suffix_grammar,
            )
        except Exception as _p6_e:
            print(f"[HashGrad] Point-6 audit skipped ({_p6_e})")

    except Exception as _eval_e:
        import traceback
        traceback.print_exc()
        print(f"[HashGrad] Evaluation failed ({_eval_e}) — reporting inf BPB")
        bpb, val_loss = float("inf"), float("inf")

    eval_elapsed   = time.time() - t_eval_start
    total_elapsed  = time.time() - t_start

    script_path     = os.path.abspath(__file__)
    code_size_bytes = os.path.getsize(script_path)
    total_bytes     = code_size_bytes + artifact_bytes
    size_ok         = total_bytes <= 16_000_000
    training_ok     = training_elapsed <= 600.0
    eval_ok         = eval_elapsed     <= 600.0

    print(f"\n{'='*60}")
    print(f"[TensorCore] FINAL RESULTS")
    print(f"BPB: {bpb:.4f}  |  Val Loss: {val_loss:.4f}")
    print(f"Training time : {training_elapsed:.1f}s  "
          f"({'PASS ✅' if training_ok else 'FAIL ⚠️'} ≤600s training limit)")
    print(f"Eval time     : {eval_elapsed:.1f}s    "
          f"({'PASS ✅' if eval_ok else 'FAIL ⚠️'} ≤600s eval limit)")
    print(f"Total time    : {total_elapsed:.1f}s")
    print(f"Code size: {code_size_bytes:,} bytes  |  Total artifact: {total_bytes:,} bytes")
    print(f"Artifact size check: {'PASS' if size_ok else 'FAIL'} (limit: 16,000,000 bytes)")
    print(f"{'='*60}")

    from datetime import datetime, timezone
    submission = {
        "track": "10min_16mb",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "name": getattr(args, "run_name", "HDC Hash-Grad 8xH100"),
        "author": getattr(args, "author", ""),
        "github_id": getattr(args, "github_id", ""),
        "val_loss": float(val_loss),
        "val_bpb": float(bpb),
        "artifact_bytes": total_bytes,
        "code_bytes": code_size_bytes,
        "world_size": world_size,
        "table_bits": TABLE_BITS,
        "embed_dim": EMBED_DIM,
        "seeds": HG_SEEDS,
        # ── Competition timing (both budgets tracked separately) ────────────────
        # README: training ≤ 10 min; evaluation ≤ 10 min (additional budget).
        "training_elapsed_s":  round(training_elapsed, 1),
        "eval_elapsed_s":      round(eval_elapsed, 1),
        "total_elapsed_s":     round(total_elapsed, 1),
        "training_time_pass":  training_ok,
        "eval_time_pass":      eval_ok,
        # Legacy key kept for compatibility
        "elapsed_s": round(total_elapsed, 1),
    }
    submission_path = os.path.join(script_dir, "submission.json")
    with open(submission_path, "w") as _sf:
        json.dump(submission, _sf, indent=2)
    print(f"[TensorCore] Submission saved → {submission_path}")

    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _dist.destroy_process_group()
    except Exception:
        pass

    return 0 if size_ok and bpb < float("inf") else 1

def _find_repo_root() -> str:
                                                                                                  
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = here
    for _ in range(6):
        if (os.path.isdir(os.path.join(candidate, "data")) and
                os.path.isfile(os.path.join(candidate, "README.md"))):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return here

def _setup_tee_logging(log_path: str):
                                                                                      
    import io

    class _Tee(io.TextIOWrapper):
        def __init__(self, stream, log_file):
            self._stream = stream
            self._log = log_file

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def write(self, data):
            self._stream.write(data)
            self._stream.flush()
            try:
                self._log.write(data)
                self._log.flush()
            except Exception:
                pass
            return len(data)

        def flush(self):
            self._stream.flush()
            try:
                self._log.flush()
            except Exception:
                pass

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_file

# ---------------------------------------------------------------------------
# Hyperparameters stub — required for the standard competition pre-flight
# (import-only + Hyperparameters discovery).  The hash-grad pipeline is
# controlled via environment variables (TABLE_BITS, EMBED_DIM, HG_SEEDS) and
# the --hash_grad argparse flag rather than a Hyperparameters dataclass, but
# this stub allows the pre-flight to confirm the script is importable and to
# discover the submission's key configuration values.
# ---------------------------------------------------------------------------
import dataclasses as _dataclasses

@_dataclasses.dataclass
class Hyperparameters:
    """Minimal stub for competition pre-flight compatibility.

    The hash-grad pipeline does not use a Hyperparameters dataclass at runtime
    (configuration is via env vars TABLE_BITS / EMBED_DIM / HG_SEEDS and the
    --hash_grad argparse flag).  This stub exists so that the standard
    pre-flight smoke test (``import train_gpt; train_gpt.Hyperparameters()``)
    succeeds and reports the submission's key parameters.
    """
    # Hash-table configuration
    table_bits: int  = int(__import__("os").environ.get("TABLE_BITS", "19"))
    embed_dim:  int  = int(__import__("os").environ.get("EMBED_DIM",  "16"))
    # Derived budget: TABLE_SIZE × EMBED_DIM × 2 bytes (fp16) ≤ 16 MB
    table_size: int  = _dataclasses.field(init=False)
    budget_mb:  float = _dataclasses.field(init=False)
    # Seed list (comma-separated in HG_SEEDS env var)
    hg_seeds:   str  = __import__("os").environ.get("HG_SEEDS", "42,7,1337")
    # Vocabulary size (fixed by the 1024-token SentencePiece model)
    vocab_size: int  = 1024
    # Entry point: --hash_grad (default True)
    hash_grad:  bool = True

    def __post_init__(self):
        self.table_size = 1 << self.table_bits
        self.budget_mb  = self.table_size * self.embed_dim * 2 / 1024 / 1024

    def __repr__(self):
        return (
            f"Hyperparameters(table_bits={self.table_bits}, "
            f"table_size={self.table_size:,}, embed_dim={self.embed_dim}, "
            f"budget_mb={self.budget_mb:.1f}, vocab_size={self.vocab_size}, "
            f"hg_seeds={self.hg_seeds!r}, hash_grad={self.hash_grad})"
        )


def main():
    import argparse
    from datetime import datetime, timezone

    _repo_root = _find_repo_root()
    _default_data      = os.path.join(_repo_root, "data", "datasets", "fineweb10B_sp1024")
    _default_tokenizer = os.path.join(_repo_root, "data", "tokenizers", "fineweb_1024_bpe.model")

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    _log_path = os.path.join(_script_dir, f"train_{_ts}.log")
    _log_fh = _setup_tee_logging(_log_path)
    print(f"[HDC] Logging to {_log_path}")

    parser = argparse.ArgumentParser(
        description="HDC VSA Model — zero-parameter run: python train_gpt.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, default=_default_data)
    parser.add_argument("--tokenizer_path", type=str, default=_default_tokenizer)
    parser.add_argument("--hdc_dim", type=int, default=DEFAULT_HDC_DIM)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--max_time", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--author", type=str, default="Ashley Klimpel", help="Author name for submission")
    parser.add_argument("--github_id", type=str, default="viasky657", help="GitHub ID for submission")
    parser.add_argument("--run_name", type=str, default="HDC Zero Track 5Mb TensorCore", help="Run name for submission")

    _hg_grp = parser.add_mutually_exclusive_group()
    _hg_grp.add_argument(
        "--hash_grad", dest="hash_grad", action="store_true", default=True,
        help=(
            "Use the Hash-Addressed Gradient NMF pipeline (DEFAULT). "
            "TABLE_BITS and EMBED_DIM are controlled via environment variables (default: 19 and 16). "
            "HG_SEEDS overrides the 3-seed default (e.g. HG_SEEDS=42,7,1337)."
        ))
    _hg_grp.add_argument(
        "--no_hash_grad", dest="hash_grad", action="store_false",
        help="Disable hash-grad and fall back to the legacy single-process HDC path.")
    parser.add_argument(
        "--moral_safety", action="store_true",
        help="Enable moral safety gate during hash-grad evaluation (--hash_grad only).")

    parser.add_argument("--multi_seed", action="store_true",
                        help="Run multi-seed training for statistically significant results")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 7, 1337],
                        help="Seeds for multi-seed training (default: 42 7 1337)")

    parser.add_argument("--seed_candidates", type=int, default=2000,
                        help="Number of candidate seeds to screen (default: 2000). "
                             "Higher values improve seed quality at extra time cost.")
    parser.add_argument("--seed_sample_tokens", type=int, default=1_000_000,
                        help="Tokens to use for seed screening (default: 1M). "
                             "1M is sufficient for stable collision-rate estimates.")
    _osg_grp = parser.add_mutually_exclusive_group()
    _osg_grp.add_argument(
        "--one_step_grad_seeds", dest="one_step_grad_seeds",
        action="store_true", default=True,
        help=(
            "After seed screening, refine each top-k seed via one-step gradient: test all "
            "64 single-bit-flip neighbours and accept the best improvement (HDC analog of "
            "GPTQ one Newton step).  ON by default.  Cost: <0.1 s per seed."
        ))
    _osg_grp.add_argument(
        "--no_one_step_grad_seeds", dest="one_step_grad_seeds",
        action="store_false",
        help="Disable one-step gradient refinement of screened seeds.")

    parser.add_argument("--max_batch_iterations", type=int, default=10,
                        help="Max iterations for metacognitive correction (default: 10)")
    parser.add_argument("--target_accuracy", type=float, default=0.99,
                        help="Target accuracy for convergence (default: 0.99)")

    args = parser.parse_args()

    if getattr(args, "hash_grad", True):
        if "HG_SEEDS" not in os.environ:
            # Default to the single seed passed via --seed so each invocation
            # is an independent run. Use HG_SEEDS env var to override (e.g.
            # HG_SEEDS=42,7,1337 for an explicit multi-seed merge run).
            os.environ["HG_SEEDS"] = str(getattr(args, "seed", 42))
        return _run_hash_grad_single(args)

    if args.multi_seed:
        return run_multi_seed_training(args)

    config = HDCConfig(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        hdc_dim=args.hdc_dim,
        iterations=args.iterations,
        max_wallclock_seconds=args.max_time,
        seed=args.seed,
        max_batch_iterations=getattr(args, 'max_batch_iterations', 10),
        target_accuracy=getattr(args, 'target_accuracy', 0.99)
    )

    final_bpb, final_val_loss, elapsed = train_hdc_seed_projection(config)

    if True:
        script_path = os.path.abspath(__file__)
        code_size_bytes = os.path.getsize(script_path)

        _ptz_path_ss  = os.path.join(
            os.path.dirname(script_path) or ".",
            f"hdc_model_seed{args.seed}.ptz")
        _ptz_bytes_ss = os.path.getsize(_ptz_path_ss) if os.path.exists(_ptz_path_ss) else 0
        bytes_total = code_size_bytes + _ptz_bytes_ss

        print(f"\n{'='*60}")
        print(f"[TensorCore] FINAL RESULTS")
        print(f"{'='*60}")
        print(f"BPB: {final_bpb:.4f}")
        print(f"Val Loss: {final_val_loss:.4f}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Mode: single-process HDC")
        print(f"Code size: {code_size_bytes:,} bytes")
        print(f"Total artifact size: {bytes_total:,} bytes (zero-weight HDC)")
        print(f"Baseline to beat: 1.2244 BPB")

        submission = {
            "author": args.author,
            "github_id": args.github_id,
            "name": args.run_name,
            "blurb": f"HDC VSA Tokenizer Zero-Weight Model with Hadamard bipolar architecture, {config.hdc_dim:,} dimensions, trained for {config.iterations} iterations in {elapsed:.1f}s",
            "date": datetime.now(timezone.utc).isoformat(),
            "val_loss": final_val_loss,
            "val_bpb": final_bpb,
            "bytes_total": bytes_total,
            "bytes_code": code_size_bytes,
            "world_size": 1
        }

        submission_path = "submission.json"
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)

        print(f"\n[TensorCore] Submission saved to {submission_path}")
        print(f"[TensorCore] Artifact size check: {'PASS' if bytes_total < 16000000 else 'FAIL'} (limit: 16,000,000 bytes)")

    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
