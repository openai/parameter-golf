// trie_kernel.hpp — Device trie data structures and launcher declarations.
//
// Used by cuda_module.cpp (compiled by g++) so must not include cuda_runtime.h.
// All CUDA API calls live in trie_kernel.cu (compiled by nvcc).
//
// Phase 4: port trie_partial_z_and_target to device.

#pragma once

#include <cstdint>
#include <string>

namespace ppmd {
class Trie;
}  // namespace ppmd

namespace ppmd_cuda {

// Forward declaration (defined in flat_ctx.hpp).
struct DeviceCtxTableHandle;

// ---------------------------------------------------------------------------
// Device-side flat trie view — pointer fields point to device memory.
// Mirrors ppmd::Trie flat arrays (insertion-ordered).
// ---------------------------------------------------------------------------
struct DeviceTrie {
    int32_t* first_child;       // [n_nodes]
    int32_t* first_terminal;    // [n_nodes]
    int32_t* next_sibling;      // [n_edges]
    int32_t* child_node;        // [n_edges]
    uint8_t*  child_byte;       // [n_edges]
    int32_t* next_terminal;     // [n_terms]
    int32_t* terminal_token_id; // [n_terms]
    int n_nodes;
    int n_edges;
    int n_terms;
};

// RAII owner of device trie allocations.
struct DeviceTrieHandle {
    DeviceTrie dev_view{};
    void* d_first_child       = nullptr;
    void* d_first_terminal    = nullptr;
    void* d_next_sibling      = nullptr;
    void* d_child_node        = nullptr;
    void* d_child_byte        = nullptr;
    void* d_next_terminal     = nullptr;
    void* d_terminal_token_id = nullptr;
    bool  valid               = false;
};

// Build device trie from host ppmd::Trie.
DeviceTrieHandle build_device_trie(const ppmd::Trie& trie);

// Free all device memory in h and mark it invalid.
void free_device_trie(DeviceTrieHandle& h);

// ---------------------------------------------------------------------------
// DFS stack frame — 32 bytes, 8-byte aligned for double.
// path[16] holds the bytes applied via fork_and_update from root to this node
// (maximum depth 16; vocab tokens are at most ~5 bytes long in practice).
// ---------------------------------------------------------------------------
struct TrieFrame {
    int32_t node;
    uint8_t path[16];
    int32_t path_len;
    double  prefix_prob;  // at offset 24 — aligned
};
static_assert(sizeof(TrieFrame) == 32, "TrieFrame must be 32 bytes");

// ---------------------------------------------------------------------------
// Kernel launcher.
//
// Computes per-token prefix_probs (sum of all path-prefix-prob * terminal
// indicator for each token) and the total partition function z.
//
// d_token_probs : device buffer of vocab_size doubles — MUST BE ZEROED by
//                 caller before each invocation (use cudaMemset).
// d_out_z       : device buffer of 1 double — written by kernel.
// d_stack       : device workspace for DFS stack (stack_capacity TrieFrames).
// ---------------------------------------------------------------------------
void launch_trie_score_kernel(
    const DeviceCtxTableHandle& ctx,
    const DeviceTrieHandle&     dtrie,
    int vocab_size,
    int shard_start,
    int shard_end,
    int order,
    void*   d_stack,          // TrieFrame*, length stack_capacity
    int     stack_capacity,
    double* d_token_probs,    // [vocab_size], caller zeroes
    double* d_out_z           // [1]
);

}  // namespace ppmd_cuda
