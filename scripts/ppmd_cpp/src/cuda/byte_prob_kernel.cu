// byte_prob_kernel.cu — CUDA implementation of the PPM-D backoff byte-prob
// computation.  Mirrors backoff_byte_probs<Provider> from backoff.hpp exactly
// in double precision.
//
// Constraints:
//   - No atomics, no warp shuffles, no fast-math (__fmaf_rn etc.).
//   - --fmad=false enforced at Makefile level; arithmetic written defensively
//     to avoid implicit fused-multiply-add opportunities.
//   - One thread per window (correctness-first; Phase 3).
//   - Iteration orders mirror CPU: bytes b=0..255 ascending,
//     contexts k=order..0 descending.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "cuda/flat_ctx.hpp"

namespace ppmd_cuda {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Pack context bytes into the uint64_t key format used by PPMDState.
// Bit layout: [63:61] = 3-bit length, [39:0] = payload (5 bytes, slot i at bits (4-i)*8).
// Mirrors ppmd::pack_ctx from ppmd.hpp exactly.
__device__ __forceinline__ uint64_t device_pack_ctx(const uint8_t* bytes, int len) {
    uint64_t key = static_cast<uint64_t>(len) << 61;
    for (int i = 0; i < len; ++i) {
        key |= static_cast<uint64_t>(bytes[i]) << ((4 - i) * 8);
    }
    return key;
}

// 256-bit bitset helpers (4 × uint64_t words, bit b in word b/64, position b%64).
__device__ __forceinline__ void set_bit(uint64_t* bits4, int b) {
    bits4[b >> 6] |= (uint64_t(1) << (b & 63));
}

__device__ __forceinline__ bool test_bit(const uint64_t* bits4, int b) {
    return (bits4[b >> 6] >> (b & 63)) & uint64_t(1);
}

__device__ __forceinline__ int count_bits(const uint64_t* bits4) {
    return __popcll(bits4[0]) + __popcll(bits4[1]) +
           __popcll(bits4[2]) + __popcll(bits4[3]);
}

// Look up key in the flat device hash table.
// Returns true and sets *out_counts to the 256-uint32_t row if found.
__device__ bool device_lookup(const FlatCtxTable& tbl, uint64_t key,
                              const uint32_t** out_counts) {
    if (tbl.num_unique_ctxs == 0) return false;
    uint64_t h = key ^ tbl.hash_seed;
    h = h * uint64_t(0xBF58476D1CE4E5B9ULL);
    h ^= h >> 31;
    int bucket = static_cast<int>(h & static_cast<uint64_t>(tbl.hash_size - 1));
    for (int probe = 0; probe < tbl.hash_size; ++probe) {
        int b = (bucket + probe) & (tbl.hash_size - 1);
        uint64_t stored = tbl.keys[b];
        if (stored == ~uint64_t(0)) return false;   // empty sentinel
        if (stored == key) {
            int idx    = tbl.bucket_to_idx[b];
            *out_counts = tbl.counts + static_cast<ptrdiff_t>(idx) * 256;
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Overlay computation
// Replay fork_and_update for `update_bytes` starting from `base_window` and
// compute how many times each byte was inserted at context key `query_key`.
// overlay_delta[256] is zeroed and filled by this function.
// ---------------------------------------------------------------------------
__device__ void compute_overlay_for_key(
    const uint8_t* base_window, int base_wlen,
    const uint8_t* update_bytes, int update_len,
    int order,
    uint64_t query_key,
    uint32_t* overlay_delta)   // length 256, written
{
    for (int b = 0; b < 256; ++b) overlay_delta[b] = 0u;
    if (update_len == 0) return;

    // Virtual window starts at the tail of base_window (capped to order).
    const int MAX_VWIN = 12;
    uint8_t vwindow[MAX_VWIN];
    int vwlen = (base_wlen > order) ? order : base_wlen;
    int base_start = base_wlen - vwlen;
    for (int i = 0; i < vwlen; ++i) vwindow[i] = base_window[base_start + i];

    for (int step = 0; step < update_len; ++step) {
        uint8_t b_step = update_bytes[step];
        int kmax_now = (vwlen < order) ? vwlen : order;
        for (int k = 0; k <= kmax_now; ++k) {
            uint64_t key = device_pack_ctx(vwindow + vwlen - k, k);
            if (key == query_key) {
                overlay_delta[b_step] += 1u;
            }
        }
        // Advance virtual window (fork_and_update logic).
        if (vwlen < order) {
            vwindow[vwlen++] = b_step;
        } else {
            for (int i = 0; i < order - 1; ++i) vwindow[i] = vwindow[i + 1];
            vwindow[order - 1] = b_step;
        }
    }
}

// ---------------------------------------------------------------------------
// Core device function — mirrors backoff_byte_probs exactly.
// update_bytes: bytes to apply via fork_and_update before computing probs.
// probs: output array of 256 doubles.
// ---------------------------------------------------------------------------
__device__ void device_compute_byte_probs(
    const FlatCtxTable& tbl,
    const uint8_t* update_bytes,
    int update_len,
    int order,
    double* probs)
{
    const int MAX_VWIN = 12;

    // --- Compute effective window after applying update_bytes --------------
    uint8_t eff_window[MAX_VWIN];
    int eff_wlen = (tbl.base_window_len > order) ? order : tbl.base_window_len;
    int base_start = tbl.base_window_len - eff_wlen;
    for (int i = 0; i < eff_wlen; ++i)
        eff_window[i] = tbl.base_window[base_start + i];
    for (int step = 0; step < update_len; ++step) {
        uint8_t b = update_bytes[step];
        if (eff_wlen < order) {
            eff_window[eff_wlen++] = b;
        } else {
            for (int i = 0; i < order - 1; ++i) eff_window[i] = eff_window[i + 1];
            eff_window[order - 1] = b;
        }
    }

    // --- Initialize output and state variables ----------------------------
    for (int b = 0; b < 256; ++b) probs[b] = 0.0;
    uint64_t assigned[4] = {0ULL, 0ULL, 0ULL, 0ULL};
    double escape_mass   = 1.0;

    // Overlay delta: reused for each context level.
    uint32_t overlay_delta[256];
    // Active-byte buffer: indices of bytes with non-zero unassigned count.
    int active_bytes[256];

    int kmax = (order < eff_wlen) ? order : eff_wlen;

    // --- Main backoff loop (k = kmax..0) ----------------------------------
    for (int k = kmax; k >= 0; --k) {
        uint64_t key = device_pack_ctx(eff_window + eff_wlen - k, k);

        // Look up base counts.
        const uint32_t* base_row = nullptr;
        bool found_base = device_lookup(tbl, key, &base_row);

        // Compute overlay for this context key.
        compute_overlay_for_key(
            tbl.base_window, tbl.base_window_len,
            update_bytes, update_len,
            order, key, overlay_delta);

        // Collect active (unassigned, non-zero) bytes.
        int active_unique = 0;
        uint64_t active_total = 0ULL;
        for (int b = 0; b < 256; ++b) {
            uint32_t base_c = found_base ? base_row[b] : 0u;
            uint32_t combined = base_c + overlay_delta[b];
            if (combined == 0u) continue;
            if (test_bit(assigned, b)) continue;
            active_bytes[active_unique++] = b;
            active_total += static_cast<uint64_t>(combined);
        }
        if (active_unique == 0) continue;

        int active_alphabet_size = 256 - count_bits(assigned);

        // Fully-assigned shortcut (no escape needed at this level).
        if (active_unique == active_alphabet_size) {
            if (active_total == 0ULL) continue;
            double total_d = static_cast<double>(active_total);
            for (int i = 0; i < active_unique; ++i) {
                int b      = active_bytes[i];
                uint32_t base_c   = found_base ? base_row[b] : 0u;
                uint32_t combined = base_c + overlay_delta[b];
                double c  = static_cast<double>(combined);
                probs[b]  = escape_mass * (c / total_d);
                set_bit(assigned, b);
            }
            escape_mass = 0.0;
            break;
        }

        // Normal (partial) assignment with PPM-D escape.
        double total_d  = static_cast<double>(active_total);
        double unique_d = static_cast<double>(active_unique);
        double denom    = total_d + unique_d;
        for (int i = 0; i < active_unique; ++i) {
            int b      = active_bytes[i];
            uint32_t base_c   = found_base ? base_row[b] : 0u;
            uint32_t combined = base_c + overlay_delta[b];
            double c  = static_cast<double>(combined);
            probs[b]  = escape_mass * (c / denom);
            set_bit(assigned, b);
        }
        // escape_mass *= unique_d / denom  — written as two ops to match CPU.
        escape_mass = escape_mass * (unique_d / denom);
    }

    // --- Uniform spread for unassigned bytes ------------------------------
    int remaining = 256 - count_bits(assigned);
    if (remaining > 0) {
        double per_byte = escape_mass / static_cast<double>(remaining);
        for (int b = 0; b < 256; ++b) {
            if (!test_bit(assigned, b)) probs[b] = per_byte;
        }
    }
}

// ---------------------------------------------------------------------------
// Global kernel — one thread per window.
// ---------------------------------------------------------------------------
__global__ void byte_prob_kernel(
    FlatCtxTable tbl,
    const uint8_t*  windows,
    const int32_t*  window_offsets,
    const int32_t*  window_lens,
    int order,
    int n_windows,
    double* out_probs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_windows) return;

    int32_t off = window_offsets[idx];
    int32_t len = window_lens[idx];
    device_compute_byte_probs(
        tbl,
        windows + off,
        static_cast<int>(len),
        order,
        out_probs + static_cast<ptrdiff_t>(idx) * 256);
}

// ---------------------------------------------------------------------------
// Host-side launcher (public, declared in flat_ctx.hpp).
// ---------------------------------------------------------------------------
void launch_byte_prob_kernel(
    const DeviceCtxTableHandle& table,
    const uint8_t*  d_windows,
    const int32_t*  d_window_offsets,
    const int32_t*  d_window_lens,
    int n_windows,
    int order,
    double* d_out_probs)
{
    if (n_windows <= 0) return;
    constexpr int BLOCK = 128;
    int grid = (n_windows + BLOCK - 1) / BLOCK;
    byte_prob_kernel<<<grid, BLOCK>>>(
        table.host_view,
        d_windows, d_window_offsets, d_window_lens,
        order, n_windows, d_out_probs);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("byte_prob_kernel launch: ") + cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(err));
    }
}

}  // namespace ppmd_cuda
