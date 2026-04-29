// flat_ctx.hpp — Host declarations for the flat open-addressing device context
// table used by the CUDA byte-prob kernel.
//
// Uses only standard C++ types so that cuda_module.cpp can include this header
// when compiled by g++ (no cuda_runtime.h here).  All CUDA API calls live in
// flat_ctx.cu and byte_prob_kernel.cu, compiled by nvcc.

#pragma once

#include <cstdint>
#include <functional>

namespace ppmd {
// Forward declarations — full definitions in ppmd.hpp / virtual_ppmd.hpp.
struct CtxCounts;
class PPMDState;
class VirtualPPMDState;
}  // namespace ppmd

namespace ppmd_cuda {

// Maximum PPM order supported by the device table (mirrors ppmd::kMaxOrder).
static constexpr int kDevMaxOrder = 5;

// ---------------------------------------------------------------------------
// FlatCtxTable
// Sorted-by-key open-addressing hash table for deterministic GPU lookup.
//   hash_size  = next power-of-two >= 2 * num_unique_ctxs  (load factor <= 0.5)
//   keys[]     = device array, length hash_size; UINT64_MAX = empty sentinel
//   counts[]   = device array, num_unique_ctxs * 256 uint32_t  (row-major)
//   bucket_to_idx[] = device array, length hash_size; -1 if empty
//   hash_seed  = 0x9E3779B97F4A7C15ULL  (fixed for reproducibility)
// The base PPMDState window is embedded so the kernel can advance it.
// ---------------------------------------------------------------------------
struct FlatCtxTable {
    uint64_t* keys;
    uint32_t* counts;
    int32_t*  bucket_to_idx;
    int       num_unique_ctxs;
    int       hash_size;
    uint64_t  hash_seed;
    uint8_t   base_window[kDevMaxOrder];
    int       base_window_len;
    int       order;
};

// RAII wrapper — holds device pointers for FlatCtxTable and tracks validity.
struct DeviceCtxTableHandle {
    FlatCtxTable host_view;   // struct whose pointer fields point to device memory
    void* d_keys          = nullptr;
    void* d_counts        = nullptr;
    void* d_bucket_to_idx = nullptr;
    int   byte_count      = 0;
    bool  valid           = false;
};

// ---------------------------------------------------------------------------
// Host-callable CUDA memory helpers (implemented in flat_ctx.cu).
// These are the only CUDA API entry points that cuda_module.cpp (compiled by
// CXX) needs; they avoid dragging cuda_runtime.h into the CXX compile unit.
// ---------------------------------------------------------------------------
void device_alloc(void** ptr, size_t bytes);
void device_free(void* ptr);
void device_memcpy_h2d(void* dst, const void* src, size_t bytes);
void device_memcpy_d2h(void* dst, const void* src, size_t bytes);
void device_memset_zero(void* ptr, size_t bytes);

// ---------------------------------------------------------------------------
// Table builders (implemented in flat_ctx.cu).
// ---------------------------------------------------------------------------

// Build from a PPMDState — uses its public ctx_counts() and window().
DeviceCtxTableHandle build_device_ctx_table(const ppmd::PPMDState& state);

// Build from a VirtualPPMDState — iterates the merged (base+overlay) counts
// via VirtualPPMDState::each_context().
DeviceCtxTableHandle build_device_ctx_table(const ppmd::VirtualPPMDState& state);

// Free all device memory in handle h and mark it invalid.
void free_device_ctx_table(DeviceCtxTableHandle& h);

// ---------------------------------------------------------------------------
// Kernel launcher (implemented in byte_prob_kernel.cu).
// d_windows       : flat buffer of update bytes (all windows concatenated)
// d_window_offsets: start offset in d_windows for window i
// d_window_lens   : byte length of window i
// n_windows       : number of windows
// order           : PPM order
// d_out_probs     : device buffer, shape [n_windows * 256] doubles (caller-alloc)
// ---------------------------------------------------------------------------
void launch_byte_prob_kernel(
    const DeviceCtxTableHandle& table,
    const uint8_t* d_windows,
    const int32_t* d_window_offsets,
    const int32_t* d_window_lens,
    int n_windows,
    int order,
    double* d_out_probs
);

}  // namespace ppmd_cuda
