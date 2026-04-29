// scorer.hpp — Path A scoring functions.
//
// trie_partial_z_and_target: explicit-stack DFS over candidate trie shard,
// summing prefix probabilities at terminals whose token_id falls in
// [shard_start, shard_end). Mirrors Python `trie_partial_z_and_target` exactly
// (eval_path_a_ppmd.py L342-373).
//
// score_path_a_arrays: end-to-end scoring loop over positions. Per position:
//   - choose boundary or non-boundary trie based on prev token's emit pattern;
//   - DFS to compute (z, target_q);
//   - p_ppm = q/z;
//   - p_nn  = exp(-nll);
//   - lambda from confidence gate;
//   - p_mix = lambda * p_nn + (1-lambda) * p_ppm;
//   - bits += -log2(p_mix);
//   - bytes += len(actual_target_bytes);
//   - state.update_bytes(actual_target_bytes) [iff update_after_score].
//
// THREADING NOTE on Phase-2 thread_local scratch (combined_counts merged):
// Within trie DFS each call sequence is:
//   p = virtual.byte_prob(b);          // overwrites thread_local merged once
//                                      // per backoff level, but consumes it
//                                      // before next provider() call within
//                                      // the same backoff loop (verified in
//                                      // backoff_byte_prob).
//   child_virtual = virtual.fork_and_update(b);  // does NOT call
//                                                // combined_counts (writes
//                                                // overlay directly).
// No nested call clobbers the scratch. Each OpenMP thread has its own
// thread_local. Safe.

#pragma once

#include "trie.hpp"
#include "virtual_ppmd.hpp"

#include <cstdint>
#include <vector>

namespace ppmd {

struct PartialResult {
    double z = 0.0;
    double target_q = 0.0;
    int64_t terminal_count = 0;
};

// Single-shard DFS. shard_end == INT32_MAX disables the upper bound (matches
// Python's sys.maxsize default).
PartialResult trie_partial_z_and_target(
    const VirtualPPMDState& root_state,
    const Trie& trie,
    int32_t target_id,
    int32_t shard_start,
    int32_t shard_end);

struct ScoreHyperparams {
    int order = 5;
    double lambda_hi = 0.9;
    double lambda_lo = 0.05;
    double conf_threshold = 0.9;
    bool update_after_score = true;
};

struct ScoreArraysVocab {
    // CSR-style packed bytes per token, after-boundary table.
    const uint8_t* boundary_bytes;
    const int32_t* boundary_offsets;  // length V+1
    const uint8_t* nonboundary_bytes;
    const int32_t* nonboundary_offsets;  // length V+1
    const uint8_t* emittable;       // length V (0/1)
    const uint8_t* is_boundary;     // length V (0/1)
    int32_t vocab_size;
};

struct ScoreArraysResult {
    int64_t positions = 0;
    double total_bits = 0.0;
    int64_t total_bytes = 0;
    double bpb = 0.0;  // total_bits / total_bytes (0 if total_bytes==0)
    std::string start_state_digest;
    std::string end_state_digest;
};

// End-to-end scorer. target_ids/prev_ids/nll_nats are length n_positions.
ScoreArraysResult score_path_a_arrays(
    const int32_t* target_ids,
    const int32_t* prev_ids,
    const double* nll_nats,
    int64_t n_positions,
    const ScoreArraysVocab& vocab,
    const ScoreHyperparams& hyper);

// OpenMP wrappers exposed to Python.
void set_num_threads(int n);
int get_max_threads();

}  // namespace ppmd
