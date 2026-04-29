// Implementation of Path A scoring kernels.

#include "scorer.hpp"

#include "ppmd.hpp"
#include "virtual_ppmd.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ppmd {

namespace {

// One frame of the explicit-stack DFS. Owning storage: the VirtualPPMDState is
// move-only-friendly (deep copies its overlay map on copy-construct).
struct Frame {
    int32_t node;
    VirtualPPMDState virt;
    double prefix_prob;
};

// DFS a single subtree rooted at `start_node` reached via a `start_virt` and
// `start_prefix`. Accumulates into out_z / out_q / out_count.
inline void dfs_subtree(
    int32_t start_node,
    VirtualPPMDState start_virt,
    double start_prefix,
    const Trie& trie,
    int32_t target_id,
    int32_t shard_start,
    int32_t shard_end,
    double& out_z,
    double& out_q,
    int64_t& out_count) {
    std::vector<Frame> stack;
    stack.reserve(64);
    stack.push_back({start_node, std::move(start_virt), start_prefix});

    while (!stack.empty()) {
        // Move out the top frame so we own its VirtualPPMDState. The
        // pop_back() destroys the (now moved-from) slot. We then explore the
        // node's children using `fr.virt` as the parent state.
        Frame fr = std::move(stack.back());
        stack.pop_back();

        // Process terminals at this node in insertion order.
        int32_t term = trie.first_terminal(fr.node);
        while (term != -1) {
            int32_t tid = trie.terminal_token_id(term);
            if (tid >= shard_start && tid < shard_end) {
                out_z += fr.prefix_prob;
                out_count += 1;
                if (tid == target_id) {
                    out_q += fr.prefix_prob;
                }
            }
            term = trie.next_terminal(term);
        }

        // Push children in insertion order. The last-pushed child is popped
        // first, mirroring Python's `stack.pop()` LIFO semantics.
        int32_t edge = trie.first_child(fr.node);
        while (edge != -1) {
            int b = trie.child_byte(edge);
            // byte_prob writes to the thread_local merged scratch in
            // VirtualPPMDState::combined_counts and consumes it within the
            // same backoff loop. fork_and_update below does NOT call
            // combined_counts. No nested clobber. Each OMP thread has its
            // own thread_local. Verified safe for Phase 3.
            double p = fr.virt.byte_prob(b);
            if (p > 0.0) {
                Frame child_frame{trie.child_node(edge),
                                  fr.virt.fork_and_update(b),
                                  fr.prefix_prob * p};
                stack.push_back(std::move(child_frame));
            }
            edge = trie.next_sibling(edge);
        }
    }
}

}  // namespace

PartialResult trie_partial_z_and_target(
    const VirtualPPMDState& root_state,
    const Trie& trie,
    int32_t target_id,
    int32_t shard_start,
    int32_t shard_end) {
    PartialResult result;

    // Process root terminals first (matches Python: terminals at popped node
    // before children are processed, and the root is the first node popped).
    int32_t root_term = trie.first_terminal(0);
    while (root_term != -1) {
        int32_t tid = trie.terminal_token_id(root_term);
        if (tid >= shard_start && tid < shard_end) {
            result.z += 1.0;
            result.terminal_count += 1;
            if (tid == target_id) {
                result.target_q += 1.0;
            }
        }
        root_term = trie.next_terminal(root_term);
    }

    // Collect root children edge indices in insertion order.
    std::vector<int32_t> root_edges;
    root_edges.reserve(16);
    for (int32_t e = trie.first_child(0); e != -1; e = trie.next_sibling(e)) {
        root_edges.push_back(e);
    }
    int32_t n_children = static_cast<int32_t>(root_edges.size());
    if (n_children == 0) {
        return result;
    }

    // Per-root-child accumulator slots. Filled in parallel when threads > 1;
    // reduction is sequential in REVERSE child order to mirror Python's LIFO
    // pop ordering at the root level. This makes the result deterministic
    // across thread counts and tight to the Python reference.
    std::vector<double> slot_z(n_children, 0.0);
    std::vector<double> slot_q(n_children, 0.0);
    std::vector<int64_t> slot_count(n_children, 0);

#pragma omp parallel for schedule(dynamic, 1)
    for (int32_t i = 0; i < n_children; ++i) {
        int32_t edge = root_edges[i];
        int b = trie.child_byte(edge);
        // Each iteration starts from a fresh clone of root_state; cheap because
        // VirtualPPMDState only holds an overlay map (empty here) plus a
        // window vector.
        double p = root_state.byte_prob(b);
        if (p > 0.0) {
            VirtualPPMDState child_virt = root_state.fork_and_update(b);
            double z_acc = 0.0;
            double q_acc = 0.0;
            int64_t c_acc = 0;
            dfs_subtree(trie.child_node(edge), std::move(child_virt), p, trie,
                        target_id, shard_start, shard_end,
                        z_acc, q_acc, c_acc);
            slot_z[i] = z_acc;
            slot_q[i] = q_acc;
            slot_count[i] = c_acc;
        }
    }

    // Reduce in REVERSE order so that with one thread the running sum order
    // matches Python's LIFO pop walk over root children.
    for (int32_t i = n_children - 1; i >= 0; --i) {
        result.z += slot_z[i];
        result.target_q += slot_q[i];
        result.terminal_count += slot_count[i];
    }
    return result;
}

namespace {

// Decide whether the prev token's emitted byte string includes a leading
// space. Mirrors `_build_py_candidates` in the Python reference: prev_id < 0
// (BOS) is treated as "previous is boundary"; otherwise consult is_boundary
// LUT (out-of-range -> treat as boundary, matching Python's defensive default).
inline bool prev_is_boundary(int32_t prev_id, const uint8_t* is_boundary, int32_t vocab_size) {
    if (prev_id < 0) return true;
    if (prev_id >= vocab_size) return true;
    return is_boundary[prev_id] != 0;
}

}  // namespace

ScoreArraysResult score_path_a_arrays(
    const int32_t* target_ids,
    const int32_t* prev_ids,
    const double* nll_nats,
    int64_t n_positions,
    const ScoreArraysVocab& vocab,
    const ScoreHyperparams& hyper) {
    // Build the two candidate tries from the CSR-packed vocab tables.
    Trie boundary_trie;
    Trie non_boundary_trie;
    for (int32_t tid = 0; tid < vocab.vocab_size; ++tid) {
        if (vocab.emittable[tid] == 0) continue;
        {
            int32_t off = vocab.boundary_offsets[tid];
            int32_t end = vocab.boundary_offsets[tid + 1];
            boundary_trie.insert(tid, vocab.boundary_bytes + off,
                                 static_cast<std::size_t>(end - off));
        }
        {
            int32_t off = vocab.nonboundary_offsets[tid];
            int32_t end = vocab.nonboundary_offsets[tid + 1];
            non_boundary_trie.insert(tid, vocab.nonboundary_bytes + off,
                                     static_cast<std::size_t>(end - off));
        }
    }

    PPMDState state(hyper.order);
    ScoreArraysResult res;
    res.start_state_digest = state.state_digest();

    for (int64_t pos = 0; pos < n_positions; ++pos) {
        int32_t target_id = target_ids[pos];
        int32_t prev_id = prev_ids[pos];
        bool boundary = prev_is_boundary(prev_id, vocab.is_boundary, vocab.vocab_size);
        const Trie& trie = boundary ? boundary_trie : non_boundary_trie;

        VirtualPPMDState virt = state.clone_virtual();
        PartialResult part = trie_partial_z_and_target(
            virt, trie, target_id, /*shard_start=*/0, /*shard_end=*/INT32_MAX);

        if (part.z <= 0.0) {
            throw std::runtime_error("Path A normalization constant Z is non-positive");
        }
        double p_ppm = part.target_q / part.z;
        if (p_ppm < 0.0) {
            throw std::runtime_error("Negative target PPM probability");
        }
        double p_nn = std::exp(-nll_nats[pos]);
        double conf = state.confidence();
        double lam = (conf >= hyper.conf_threshold) ? hyper.lambda_lo : hyper.lambda_hi;
        double p_mix = lam * p_nn + (1.0 - lam) * p_ppm;
        if (p_mix <= 0.0) {
            throw std::runtime_error("Mixture assigned zero probability to target");
        }
        double bits = -std::log(p_mix) / std::log(2.0);
        res.total_bits += bits;

        // Determine actual target bytes for normalization.
        int32_t bo = (boundary ? vocab.boundary_offsets[target_id]
                               : vocab.nonboundary_offsets[target_id]);
        int32_t be = (boundary ? vocab.boundary_offsets[target_id + 1]
                               : vocab.nonboundary_offsets[target_id + 1]);
        int32_t blen = be - bo;
        const uint8_t* bbytes =
            (boundary ? vocab.boundary_bytes : vocab.nonboundary_bytes) + bo;
        res.total_bytes += blen;

        if (hyper.update_after_score) {
            state.update_bytes(bbytes, static_cast<std::size_t>(blen));
        }
        ++res.positions;
    }

    res.bpb = (res.total_bytes > 0)
        ? (res.total_bits / static_cast<double>(res.total_bytes))
        : 0.0;
    res.end_state_digest = state.state_digest();
    return res;
}

void set_num_threads(int n) {
#ifdef _OPENMP
    if (n < 1) n = 1;
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace ppmd
