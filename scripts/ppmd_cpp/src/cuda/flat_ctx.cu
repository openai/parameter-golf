// flat_ctx.cu — Build and manage the device-side flat context table used by
// the CUDA byte-prob kernel.
//
// Compiled by nvcc.  Provides:
//   - CUDA memory helpers (device_alloc / device_free / memcpy wrappers)
//   - build_device_ctx_table(PPMDState&)
//   - build_device_ctx_table(VirtualPPMDState&)
//   - free_device_ctx_table(DeviceCtxTableHandle&)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ppmd.hpp"
#include "virtual_ppmd.hpp"
#include "cuda/flat_ctx.hpp"

namespace ppmd_cuda {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Next power of two >= n, minimum 2.
static int next_pow2_ge(int n) {
    if (n <= 2) return 2;
    int p = 2;
    while (p < n) p <<= 1;
    return p;
}

// Hash function for the open-addressing table.
static uint32_t hash_key(uint64_t key, uint64_t seed, int hash_size) {
    uint64_t h = key ^ seed;
    h = h * 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 31;
    return static_cast<uint32_t>(h & static_cast<uint64_t>(hash_size - 1));
}

// Insert one (key, idx) pair into a host-side open-addressing hash table.
static void ht_insert(
    std::vector<uint64_t>& keys,
    std::vector<int32_t>&  bucket_to_idx,
    uint64_t key,
    int32_t  idx,
    uint64_t seed,
    int      hash_size) {
    uint32_t bucket = hash_key(key, seed, hash_size);
    for (int probe = 0; probe < hash_size; ++probe) {
        uint32_t b = (bucket + probe) & static_cast<uint32_t>(hash_size - 1);
        if (keys[b] == ~uint64_t(0)) {
            keys[b]          = key;
            bucket_to_idx[b] = idx;
            return;
        }
    }
    throw std::runtime_error("flat_ctx: hash table is full (should not happen)");
}

// Build a DeviceCtxTableHandle from an arbitrary (key → CtxCounts) collection.
// `entries` must be (sorted-ascending-by-key, CtxCounts*) pairs.
static DeviceCtxTableHandle build_from_entries(
    const std::vector<std::pair<uint64_t, const ppmd::CtxCounts*>>& entries,
    const std::vector<uint8_t>& base_window,
    int order) {
    const uint64_t HASH_SEED = 0x9E3779B97F4A7C15ULL;
    const uint64_t SENTINEL  = ~uint64_t(0);

    int n = static_cast<int>(entries.size());
    int hash_size = next_pow2_ge(2 * std::max(n, 1));

    // --- Host-side table -------------------------------------------------
    std::vector<uint64_t> h_keys(hash_size, SENTINEL);
    std::vector<int32_t>  h_bkt(hash_size, -1);
    // counts: n * 256 uint32_t (row-major: row i = entries[i].second->counts)
    std::vector<uint32_t> h_counts(static_cast<size_t>(n) * 256, 0u);

    for (int i = 0; i < n; ++i) {
        uint64_t key           = entries[i].first;
        const ppmd::CtxCounts* cc = entries[i].second;
        for (int b = 0; b < 256; ++b) {
            h_counts[static_cast<size_t>(i) * 256 + b] = cc->counts[b];
        }
        ht_insert(h_keys, h_bkt, key, i, HASH_SEED, hash_size);
    }

    // --- Device allocation -----------------------------------------------
    size_t keys_bytes   = static_cast<size_t>(hash_size) * sizeof(uint64_t);
    size_t counts_bytes = static_cast<size_t>(n) * 256 * sizeof(uint32_t);
    size_t bkt_bytes    = static_cast<size_t>(hash_size) * sizeof(int32_t);

    void* d_keys   = nullptr;
    void* d_counts = nullptr;
    void* d_bkt    = nullptr;

    // Use try-block so we free on partial failure.
    try {
        check_cuda(cudaMalloc(&d_keys,   keys_bytes   ? keys_bytes   : 1),
                   "cudaMalloc d_keys");
        check_cuda(cudaMalloc(&d_counts, counts_bytes ? counts_bytes : 1),
                   "cudaMalloc d_counts");
        check_cuda(cudaMalloc(&d_bkt,    bkt_bytes    ? bkt_bytes    : 1),
                   "cudaMalloc d_bucket_to_idx");
    } catch (...) {
        if (d_keys)   cudaFree(d_keys);
        if (d_counts) cudaFree(d_counts);
        if (d_bkt)    cudaFree(d_bkt);
        throw;
    }

    check_cuda(cudaMemcpy(d_keys, h_keys.data(), keys_bytes ? keys_bytes : 1,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_keys");
    if (counts_bytes) {
        check_cuda(cudaMemcpy(d_counts, h_counts.data(), counts_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy d_counts");
    }
    check_cuda(cudaMemcpy(d_bkt, h_bkt.data(), bkt_bytes ? bkt_bytes : 1,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy d_bucket_to_idx");

    // --- Assemble handle -------------------------------------------------
    DeviceCtxTableHandle h;
    h.d_keys          = d_keys;
    h.d_counts        = d_counts;
    h.d_bucket_to_idx = d_bkt;
    h.byte_count      = static_cast<int>(keys_bytes + counts_bytes + bkt_bytes);
    h.valid           = true;

    h.host_view.keys          = static_cast<uint64_t*>(d_keys);
    h.host_view.counts        = static_cast<uint32_t*>(d_counts);
    h.host_view.bucket_to_idx = static_cast<int32_t*>(d_bkt);
    h.host_view.num_unique_ctxs = n;
    h.host_view.hash_size       = hash_size;
    h.host_view.hash_seed       = HASH_SEED;

    // Base window (last `order` bytes, or fewer if shorter).
    int wlen = static_cast<int>(base_window.size());
    int keep = std::min(wlen, order);
    h.host_view.base_window_len = keep;
    h.host_view.order           = order;
    for (int i = 0; i < keep; ++i) {
        h.host_view.base_window[i] = base_window[wlen - keep + i];
    }
    return h;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void device_alloc(void** ptr, size_t bytes) {
    check_cuda(cudaMalloc(ptr, bytes ? bytes : 1), "device_alloc");
}

void device_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void device_memset_zero(void* ptr, size_t bytes) {
    if (bytes == 0) return;
    check_cuda(cudaMemset(ptr, 0, bytes), "device_memset_zero");
}

void device_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice),
               "device_memcpy_h2d");
}

void device_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost),
               "device_memcpy_d2h");
}

DeviceCtxTableHandle build_device_ctx_table(const ppmd::PPMDState& state) {
    const auto& map = state.ctx_counts();

    // Collect and sort entries by key for deterministic insertion order.
    std::vector<std::pair<uint64_t, const ppmd::CtxCounts*>> entries;
    entries.reserve(map.size());
    for (const auto& kv : map) {
        entries.emplace_back(kv.first, &kv.second);
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    return build_from_entries(entries, state.window(), state.order());
}

DeviceCtxTableHandle build_device_ctx_table(const ppmd::VirtualPPMDState& vstate) {
    // Collect merged (base+overlay) entries.
    std::vector<std::pair<uint64_t, ppmd::CtxCounts>> owned;
    vstate.each_context([&](uint64_t key, const ppmd::CtxCounts& cc) {
        owned.emplace_back(key, cc);
    });

    // Sort by key.
    std::sort(owned.begin(), owned.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::pair<uint64_t, const ppmd::CtxCounts*>> entries;
    entries.reserve(owned.size());
    for (const auto& kv : owned) {
        entries.emplace_back(kv.first, &kv.second);
    }

    return build_from_entries(entries, vstate.window(), vstate.order());
}

void free_device_ctx_table(DeviceCtxTableHandle& h) {
    if (!h.valid) return;
    if (h.d_keys)          { cudaFree(h.d_keys);          h.d_keys          = nullptr; }
    if (h.d_counts)        { cudaFree(h.d_counts);        h.d_counts        = nullptr; }
    if (h.d_bucket_to_idx) { cudaFree(h.d_bucket_to_idx); h.d_bucket_to_idx = nullptr; }
    h.valid = false;
}

}  // namespace ppmd_cuda
