// trie_kernel.cu — CUDA trie DFS scoring kernel for Path A PPM-D.
//
// Mirrors C++ trie_partial_z_and_target from scorer.cpp bit-for-bit:
//   - Sequential LIFO DFS (one thread, one block).
//   - Children pushed in insertion order → popped in reverse (LIFO).
//   - Root terminals processed before root children.
//   - Reduction order matches CPU single-thread path.
//
// Arithmetic contract:
//   - All fp arithmetic is double precision.
//   - --fmad=false enforced by Makefile.
//   - No warp shuffles, no atomicAdd, no fast-math.
//
// Phase 4: port of trie_partial_z_and_target to device.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "cuda/flat_ctx.hpp"
#include "cuda/trie_kernel.hpp"
#include "trie.hpp"

namespace ppmd_cuda {

// ============================================================================
// Device helpers (mirrors byte_prob_kernel.cu helpers; static linkage).
// ============================================================================

__device__ __forceinline__ static uint64_t tk_pack_ctx(const uint8_t* bytes, int len) {
    uint64_t key = static_cast<uint64_t>(len) << 61;
    for (int i = 0; i < len; ++i)
        key |= static_cast<uint64_t>(bytes[i]) << ((4 - i) * 8);
    return key;
}

__device__ __forceinline__ static void tk_set_bit(uint64_t* bits4, int b) {
    bits4[b >> 6] |= (uint64_t(1) << (b & 63));
}

__device__ __forceinline__ static bool tk_test_bit(const uint64_t* bits4, int b) {
    return (bits4[b >> 6] >> (b & 63)) & uint64_t(1);
}

__device__ __forceinline__ static int tk_count_bits(const uint64_t* bits4) {
    return __popcll(bits4[0]) + __popcll(bits4[1]) +
           __popcll(bits4[2]) + __popcll(bits4[3]);
}

// Look up key in device hash table.  Returns true and sets *out_counts on hit.
__device__ static bool tk_lookup(const FlatCtxTable& tbl, uint64_t key,
                                  const uint32_t** out_counts) {
    if (tbl.num_unique_ctxs == 0) return false;
    uint64_t h = key ^ tbl.hash_seed;
    h = h * uint64_t(0xBF58476D1CE4E5B9ULL);
    h ^= h >> 31;
    int bucket = static_cast<int>(h & static_cast<uint64_t>(tbl.hash_size - 1));
    for (int probe = 0; probe < tbl.hash_size; ++probe) {
        int b = (bucket + probe) & (tbl.hash_size - 1);
        uint64_t stored = tbl.keys[b];
        if (stored == ~uint64_t(0)) return false;
        if (stored == key) {
            int idx     = tbl.bucket_to_idx[b];
            *out_counts = tbl.counts + static_cast<ptrdiff_t>(idx) * 256;
            return true;
        }
    }
    return false;
}

// Compute overlay counts for one context key by replaying path_bytes updates.
// overlay_delta[256] is zeroed then filled.
__device__ static void tk_overlay_for_key(
    const uint8_t* base_window, int base_wlen,
    const uint8_t* path_bytes,  int path_len,
    int order,
    uint64_t query_key,
    uint32_t* overlay_delta)
{
    for (int b = 0; b < 256; ++b) overlay_delta[b] = 0u;
    if (path_len == 0) return;

    const int MAX_VWIN = 12;
    uint8_t vwindow[MAX_VWIN];
    int vwlen = (base_wlen > order) ? order : base_wlen;
    int base_start = base_wlen - vwlen;
    for (int i = 0; i < vwlen; ++i) vwindow[i] = base_window[base_start + i];

    for (int step = 0; step < path_len; ++step) {
        uint8_t b_step = path_bytes[step];
        int kmax_now = (vwlen < order) ? vwlen : order;
        for (int k = 0; k <= kmax_now; ++k) {
            uint64_t key = tk_pack_ctx(vwindow + vwlen - k, k);
            if (key == query_key) overlay_delta[b_step] += 1u;
        }
        if (vwlen < order) {
            vwindow[vwlen++] = b_step;
        } else {
            for (int i = 0; i < order - 1; ++i) vwindow[i] = vwindow[i + 1];
            vwindow[order - 1] = b_step;
        }
    }
}

// ============================================================================
// device_byte_prob_single
//
// Compute P(target_b | base_state + fork_and_update(path_bytes)).
// Mirrors backoff_byte_prob from backoff.hpp with overlay replay.
// ============================================================================
__device__ static double device_byte_prob_single(
    const FlatCtxTable& tbl,
    const uint8_t* path_bytes, int path_len,
    int order,
    int target_b)
{
    const int MAX_VWIN = 12;

    // Compute effective window: base_window + path_bytes (capped to order).
    uint8_t eff_window[MAX_VWIN];
    int eff_wlen = (tbl.base_window_len > order) ? order : tbl.base_window_len;
    int base_start = tbl.base_window_len - eff_wlen;
    for (int i = 0; i < eff_wlen; ++i) eff_window[i] = tbl.base_window[base_start + i];
    for (int step = 0; step < path_len; ++step) {
        uint8_t b = path_bytes[step];
        if (eff_wlen < order) {
            eff_window[eff_wlen++] = b;
        } else {
            for (int i = 0; i < order - 1; ++i) eff_window[i] = eff_window[i + 1];
            eff_window[order - 1] = b;
        }
    }

    // PPM-D backoff loop (mirrors backoff_byte_prob exactly).
    uint64_t assigned[4] = {0ULL, 0ULL, 0ULL, 0ULL};
    double   escape_mass = 1.0;
    int      kmax        = (order < eff_wlen) ? order : eff_wlen;

    uint32_t overlay_delta[256];
    int      active_bytes[256];

    for (int k = kmax; k >= 0; --k) {
        uint64_t key = tk_pack_ctx(eff_window + eff_wlen - k, k);

        const uint32_t* base_row = nullptr;
        bool found_base = tk_lookup(tbl, key, &base_row);

        tk_overlay_for_key(
            tbl.base_window, tbl.base_window_len,
            path_bytes, path_len,
            order, key, overlay_delta);

        int      active_unique = 0;
        uint64_t active_total  = 0ULL;

        for (int b = 0; b < 256; ++b) {
            uint32_t base_c   = found_base ? base_row[b] : 0u;
            uint32_t combined = base_c + overlay_delta[b];
            if (combined == 0u) continue;
            if (tk_test_bit(assigned, b)) continue;
            active_bytes[active_unique++] = b;
            active_total += static_cast<uint64_t>(combined);
        }
        if (active_unique == 0) continue;

        int active_alphabet_size = 256 - tk_count_bits(assigned);

        if (active_unique == active_alphabet_size) {
            // No escape — fully-assigned shortcut.
            uint32_t base_tc   = found_base ? base_row[target_b] : 0u;
            uint32_t tc        = base_tc + overlay_delta[target_b];
            if (tc > 0u && !tk_test_bit(assigned, target_b)) {
                return escape_mass * (static_cast<double>(tc) /
                                      static_cast<double>(active_total));
            }
            return 0.0;
        }

        // Normal PPM-D escape.
        double total_d  = static_cast<double>(active_total);
        double unique_d = static_cast<double>(active_unique);
        double denom    = total_d + unique_d;

        uint32_t base_tc = found_base ? base_row[target_b] : 0u;
        uint32_t tc      = base_tc + overlay_delta[target_b];
        if (tc > 0u && !tk_test_bit(assigned, target_b)) {
            return escape_mass * (static_cast<double>(tc) / denom);
        }

        for (int i = 0; i < active_unique; ++i)
            tk_set_bit(assigned, active_bytes[i]);

        // escape_mass *= unique_d / denom  — two ops to match CPU.
        escape_mass = escape_mass * (unique_d / denom);
    }

    if (tk_test_bit(assigned, target_b)) return 0.0;
    int remaining = 256 - tk_count_bits(assigned);
    if (remaining <= 0) return 0.0;
    return escape_mass / static_cast<double>(remaining);
}

// ============================================================================
// trie_score_kernel
//
// Single-threaded sequential LIFO DFS (1 block, 1 thread).
// Matches C++ trie_partial_z_and_target with OMP_NUM_THREADS=1.
//
// Accumulation order:
//   1. Root terminals (prefix_prob = 1.0).
//   2. Root children pushed in insertion order → popped LIFO.
//   3. Within each subtree: same LIFO ordering.
// This matches the CPU reverse-reduction of slot_z at the root level.
//
// Output:
//   d_out_z[0]            = sum of all terminal prefix_probs in shard.
//   d_token_probs[tid]   += prefix_prob for each terminal with id == tid.
//                           (Caller must zero this array before calling.)
// ============================================================================
__global__ void trie_score_kernel(
    FlatCtxTable tbl,
    // Trie arrays.
    const int32_t* __restrict__ first_child,
    const int32_t* __restrict__ first_terminal,
    const int32_t* __restrict__ next_sibling,
    const int32_t* __restrict__ child_node_arr,
    const uint8_t* __restrict__ child_byte_arr,
    const int32_t* __restrict__ next_terminal,
    const int32_t* __restrict__ terminal_token_id,
    int shard_start,
    int shard_end,
    int order,
    TrieFrame* __restrict__ d_stack,
    int stack_capacity,
    double* __restrict__ d_token_probs,
    double* __restrict__ d_out_z)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    double z = 0.0;

    // ---- 1. Process root terminals (prefix_prob = 1.0) --------------------
    for (int32_t term = first_terminal[0]; term != -1; term = next_terminal[term]) {
        int32_t tid = terminal_token_id[term];
        if (tid >= shard_start && tid < shard_end) {
            z += 1.0;
            d_token_probs[tid] += 1.0;
        }
    }

    // ---- 2. Push root children in insertion order (LIFO → last first) ----
    int stack_top = 0;
    for (int32_t edge = first_child[0]; edge != -1; edge = next_sibling[edge]) {
        uint8_t b = child_byte_arr[edge];
        // Byte prob from empty path (base state directly).
        double p = device_byte_prob_single(tbl, nullptr, 0, order, static_cast<int>(b));
        if (p > 0.0) {
            TrieFrame fr;
            fr.node       = child_node_arr[edge];
            fr.path[0]    = b;
            fr.path_len   = 1;
            fr.prefix_prob = p;
            if (stack_top < stack_capacity)
                d_stack[stack_top++] = fr;
        }
    }

    // ---- 3. LIFO DFS -------------------------------------------------------
    while (stack_top > 0) {
        TrieFrame fr = d_stack[--stack_top];

        // Process terminals at this node.
        for (int32_t term = first_terminal[fr.node]; term != -1; term = next_terminal[term]) {
            int32_t tid = terminal_token_id[term];
            if (tid >= shard_start && tid < shard_end) {
                z += fr.prefix_prob;
                d_token_probs[tid] += fr.prefix_prob;
            }
        }

        // Push children in insertion order.
        for (int32_t edge = first_child[fr.node]; edge != -1; edge = next_sibling[edge]) {
            uint8_t b = child_byte_arr[edge];
            double p = device_byte_prob_single(tbl, fr.path, fr.path_len, order, static_cast<int>(b));
            if (p > 0.0) {
                TrieFrame child_fr;
                child_fr.node = child_node_arr[edge];
                // Copy parent path and append b.
                for (int i = 0; i < fr.path_len && i < 15; ++i)
                    child_fr.path[i] = fr.path[i];
                child_fr.path[fr.path_len] = b;
                child_fr.path_len  = fr.path_len + 1;
                child_fr.prefix_prob = fr.prefix_prob * p;
                if (stack_top < stack_capacity)
                    d_stack[stack_top++] = child_fr;
            }
        }
    }

    *d_out_z = z;
}

// ============================================================================
// Host-side build/free for DeviceTrieHandle.
// ============================================================================

DeviceTrieHandle build_device_trie(const ppmd::Trie& trie) {
    DeviceTrieHandle h;

    int nn = trie.num_nodes();
    int ne = trie.num_edges();
    int nt = trie.num_terminals();

    // Copy trie arrays to host flat buffers.
    std::vector<int32_t> h_first_child(nn);
    std::vector<int32_t> h_first_terminal(nn);
    std::vector<int32_t> h_next_sibling(ne);
    std::vector<int32_t> h_child_node(ne);
    std::vector<uint8_t> h_child_byte(ne);
    std::vector<int32_t> h_next_terminal(nt);
    std::vector<int32_t> h_terminal_token_id(nt);

    for (int i = 0; i < nn; ++i) {
        h_first_child[i]    = trie.first_child(i);
        h_first_terminal[i] = trie.first_terminal(i);
    }
    for (int i = 0; i < ne; ++i) {
        h_next_sibling[i] = trie.next_sibling(i);
        h_child_node[i]   = trie.child_node(i);
        h_child_byte[i]   = trie.child_byte(i);
    }
    for (int i = 0; i < nt; ++i) {
        h_next_terminal[i]     = trie.next_terminal(i);
        h_terminal_token_id[i] = trie.terminal_token_id(i);
    }

    // Allocate + copy each array to device.
    auto alloc_copy = [&](const void* src, size_t bytes, void** dst) {
        device_alloc(dst, bytes > 0 ? bytes : 1);
        if (bytes > 0) device_memcpy_h2d(*dst, src, bytes);
    };

    alloc_copy(h_first_child.data(),       static_cast<size_t>(nn) * 4, &h.d_first_child);
    alloc_copy(h_first_terminal.data(),    static_cast<size_t>(nn) * 4, &h.d_first_terminal);
    alloc_copy(h_next_sibling.data(),      static_cast<size_t>(ne) * 4, &h.d_next_sibling);
    alloc_copy(h_child_node.data(),        static_cast<size_t>(ne) * 4, &h.d_child_node);
    alloc_copy(h_child_byte.data(),        static_cast<size_t>(ne) * 1, &h.d_child_byte);
    alloc_copy(h_next_terminal.data(),     static_cast<size_t>(nt) * 4, &h.d_next_terminal);
    alloc_copy(h_terminal_token_id.data(), static_cast<size_t>(nt) * 4, &h.d_terminal_token_id);

    // Set device view.
    h.dev_view.first_child        = static_cast<int32_t*>(h.d_first_child);
    h.dev_view.first_terminal     = static_cast<int32_t*>(h.d_first_terminal);
    h.dev_view.next_sibling       = static_cast<int32_t*>(h.d_next_sibling);
    h.dev_view.child_node         = static_cast<int32_t*>(h.d_child_node);
    h.dev_view.child_byte         = static_cast<uint8_t*>(h.d_child_byte);
    h.dev_view.next_terminal      = static_cast<int32_t*>(h.d_next_terminal);
    h.dev_view.terminal_token_id  = static_cast<int32_t*>(h.d_terminal_token_id);
    h.dev_view.n_nodes            = nn;
    h.dev_view.n_edges            = ne;
    h.dev_view.n_terms            = nt;
    h.valid                       = true;

    return h;
}

void free_device_trie(DeviceTrieHandle& h) {
    if (!h.valid) return;
    auto safe_free = [](void*& p) { if (p) { device_free(p); p = nullptr; } };
    safe_free(h.d_first_child);
    safe_free(h.d_first_terminal);
    safe_free(h.d_next_sibling);
    safe_free(h.d_child_node);
    safe_free(h.d_child_byte);
    safe_free(h.d_next_terminal);
    safe_free(h.d_terminal_token_id);
    h.valid = false;
}

// ============================================================================
// launch_trie_score_kernel
// ============================================================================

void launch_trie_score_kernel(
    const DeviceCtxTableHandle& ctx,
    const DeviceTrieHandle&     dtrie,
    int vocab_size,
    int shard_start,
    int shard_end,
    int order,
    void*   d_stack,
    int     stack_capacity,
    double* d_token_probs,
    double* d_out_z)
{
    const DeviceTrie& dv = dtrie.dev_view;
    trie_score_kernel<<<1, 1>>>(
        ctx.host_view,
        dv.first_child,
        dv.first_terminal,
        dv.next_sibling,
        dv.child_node,
        dv.child_byte,
        dv.next_terminal,
        dv.terminal_token_id,
        shard_start,
        shard_end,
        order,
        static_cast<TrieFrame*>(d_stack),
        stack_capacity,
        d_token_probs,
        d_out_z);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("trie_score_kernel launch: ") + cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("cudaDeviceSynchronize (trie): ") + cudaGetErrorString(err));
}

}  // namespace ppmd_cuda
