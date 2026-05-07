#pragma once

// Rule-30 stepping primitives. Bit-parallel evolution + double-CA XOR.
// See docs/architecture.md "In-Kernel Rule 30".
//
// Update rule (verified Wolfram NKS p.869, MathWorld):
//     L = (s >> 1) | (carry_l << 63);   // p (left neighbor)
//     R = (s << 1) | (carry_r >> 63);   // r (right neighbor)
//     s_next = L ^ (s | R);              // q' = p XOR (q OR r)
//
// Bit packing convention (matches docs/source-design.md):
//   bit 63 of `s` = leftmost cell of the 64-cell tile
//   bit 0  of `s` = rightmost cell of the 64-cell tile
// `carry_l` carries the rightmost cell of the word to the left (useful bits in bit 0;
// the formulation `(carry_l << 63)` shifts that bit into position 63 regardless
// of any other bits in carry_l, so a full word may be passed for convenience).
// `carry_r` carries the leftmost cell of the word to the right (useful bits in
// bit 63; `(carry_r >> 63)` extracts it).

#include <cstdint>

#ifdef __CUDACC__
  #define GH_HOST_DEVICE __host__ __device__ __forceinline__
#else
  #define GH_HOST_DEVICE inline
#endif

namespace gemm_hopper {

// XOR mixing constant for the double-CA scheme (docs/source-design.md §3 "Bias mitigation").
// Random 64-bit prime-like constant; value is a contract — changing it changes the
// regenerated weights of every existing route.
inline constexpr uint64_t kRule30MixConstant = 0xD2B74407B1CE6E93ULL;

// Rotation amount for the paired seed (docs/source-design.md §3).
inline constexpr unsigned kRule30MixRotate = 17;

// Number of CA generations to discard from the seeded singleton state before sampling.
// Rule 30 has a left-to-right-permutive triangle that is structured for the first
// O(few hundred) steps; 256 generations is comfortably past that.
inline constexpr unsigned kRule30Warmup = 256;

GH_HOST_DEVICE uint64_t rotl64(uint64_t x, unsigned n) {
  // Use unsigned-modular shift to keep the function defined for n in {0..63}.
  // n must be in [0, 64); callers are expected to honor that.
  return (x << n) | (x >> ((64u - n) & 63u));
}

GH_HOST_DEVICE uint64_t rule30_step(uint64_t s, uint64_t carry_l, uint64_t carry_r) {
  // p (left neighbor) = (s >> 1) with carry_l's LSB stitched into position 63.
  // r (right neighbor) = (s << 1) with carry_r's MSB stitched into position 0.
  uint64_t L = (s >> 1) | (carry_l << 63);
  uint64_t R = (s << 1) | (carry_r >> 63);
  return L ^ (s | R);
}

// Derive the second-CA seed from the user-supplied seed (per source-design.md §3).
GH_HOST_DEVICE uint64_t derive_paired_seed(uint64_t seed) {
  return rotl64(seed, kRule30MixRotate) ^ kRule30MixConstant;
}

// Combine two independent CA lanes by XOR. The mixing step in the double-CA scheme.
GH_HOST_DEVICE uint64_t rule30_double_ca_xor(uint64_t a, uint64_t b) {
  return a ^ b;
}

// Ensure a non-zero singleton-style starting state (Rule 30 from all-zero stays zero).
// If `seed` is 0, returns 1 (a single cell at the rightmost position). Otherwise returns
// `seed` unchanged. Callers should still run kRule30Warmup steps before sampling.
GH_HOST_DEVICE uint64_t make_nonzero_state(uint64_t seed) {
  return seed == 0ULL ? 1ULL : seed;
}

// SplitMix64 (Steele/Lea 2014) — high-quality 64-bit-to-64-bit mixer.
// Used to derive per-word seed entropy when spreading a single 64-bit seed
// across a multi-word CA state.
GH_HOST_DEVICE uint64_t splitmix64(uint64_t z) {
  z = (z + 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Initialize a multi-word CA state (n_words >= 1) from a single 64-bit seed.
// Each word is filled with an independent SplitMix64 hash of (seed, word_index).
// All-zero words are forced to a non-zero singleton (Rule 30 absorbs all-zero).
//
// Why multi-word: a single isolated 64-cell Rule-30 lane with zero boundaries
// collapses to the attractor 0xAAAAAAAAAAAAAAAB after O(64) iterations. The
// source design (docs/source-design.md §3) uses multi-word state because a route's
// CA panel spans ~2048 words; the test/reference path must match that contract.
GH_HOST_DEVICE void init_multi_word_state(uint64_t seed,
                                          uint64_t* state,
                                          unsigned n_words) {
  for (unsigned w = 0; w < n_words; ++w) {
    state[w] = make_nonzero_state(splitmix64(seed ^ (static_cast<uint64_t>(w) + 1ULL)));
  }
}

// Device-backed local validation helpers implemented in src/generator_kernels.cu.
// They exercise the same rule30_step primitive on CUDA and are intended for
// sm_89 local proof before any Hopper-only generator work.
enum class Rule30Variant : uint32_t {
  U64Baseline = 0,
  U64Lop3 = 1,
  U32Halves = 2,
  U32Lop3 = 3,
};

inline const char* rule30_variant_name(Rule30Variant variant) {
  switch (variant) {
    case Rule30Variant::U64Baseline: return "u64_baseline";
    case Rule30Variant::U64Lop3: return "u64_lop3";
    case Rule30Variant::U32Halves: return "u32_halves";
    case Rule30Variant::U32Lop3: return "u32_lop3";
    default: return "unknown";
  }
}

bool cuda_rule30_warp_evolve(uint64_t seed, uint32_t steps, uint64_t out_host[32]);

bool cuda_rule30_warp_evolve_words_per_lane(uint64_t seed,
                                            uint32_t steps,
                                            uint32_t words_per_lane,
                                            uint64_t* out_host);

bool cuda_rule30_warp_evolve_variant(uint64_t seed,
                                     uint32_t steps,
                                     uint32_t words_per_lane,
                                     Rule30Variant variant,
                                     uint64_t* out_host);

struct Rule30BenchmarkResult {
  Rule30Variant variant;
  uint32_t words_per_lane;
  uint32_t latency_steps;
  uint32_t occupancy_steps;
  uint32_t occupancy_warps;
  uint64_t latency_cycles;
  uint64_t latency_checksum;
  uint64_t occupancy_checksum;
  float occupancy_ms;
  double latency_cells_per_cycle_per_warp;
  double occupancy_cells_per_second;
};

bool cuda_benchmark_rule30_warp(uint32_t steps,
                                uint64_t* cycles_out,
                                uint64_t* checksum_out,
                                double* cells_per_cycle_per_warp_out);

bool cuda_benchmark_rule30_variants(uint32_t latency_steps,
                                    uint32_t occupancy_steps,
                                    uint32_t occupancy_warps,
                                    Rule30BenchmarkResult* results_host,
                                    unsigned max_results,
                                    unsigned* n_results_out,
                                    Rule30Variant* best_variant_out,
                                    uint32_t* best_words_per_lane_out);

} // namespace gemm_hopper
