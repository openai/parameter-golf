#pragma once

// CPU reference: ground truth for parity checks.
// See docs/test-plan.md Tier-1 "B2B mathematical correctness".
//
// Computes: Y = scale * (X @ W0_fp8.T) + alpha * (X @ A_ternary.T) @ B_ternary.T
//   - W0 in FP8 (e4m3); scale is a per-tile scalar
//   - A and B regenerated from `seed` via the same Rule-30 schedule the device uses
//   - Output Y in BF16
//
// Tolerances per docs/test-plan.md Tier-1:
//   atol = 1e-2, rtol = 1e-2 vs eager PyTorch reference
//   atol = 1e-3 (BF16) vs FP32 ground truth for component checks

#include <cstddef>
#include <cstdint>

namespace gemm_hopper {

// -- Phase 1: Rule-30 + ternary generation -----------------------------------

// Multi-word Rule-30 evolution. Performs `steps` Rule-30 steps over an array of
// `n_words` 64-bit words representing a contiguous CA state, modifying `state`
// in place. Boundary cells outside `state` are treated as zero.
//
// O(n_words * steps) work; pure host code.
void rule30_evolve_multiword(uint64_t* state, std::size_t n_words, std::size_t steps);

// Generate a row-major matrix of `rows * cols` ternary values in {-1, 0, +1}
// (stored as int8_t) from a single 64-bit seed. The protocol is:
//   1. Initialize CA lane A from `seed` (forced non-zero); lane B from
//      derive_paired_seed(seed).
//   2. Discard kRule30Warmup steps from each lane.
//   3. Per output pair-of-32: sample low 32 bits of (A^B) -> b0; step both
//      lanes once; sample low 32 bits of (A^B) -> b1; pack 32 ternary values
//      via pack_ternary_2bit(b0, b1); step both lanes once more before the
//      next batch (so adjacent ternary words come from different CA states).
//   4. Decode and write into `out`.
//
// `out` must hold at least `rows * cols` int8_t values. Excess past the last
// pair is unused (no padding); cols not a multiple of 32 still works.
void generate_ternary_matrix(uint64_t seed, std::size_t rows, std::size_t cols, std::int8_t* out);

// -- Phase 2: B2B GEMM reference --------------------------------------------

// B2B GEMM CPU reference. Ground truth for the device path's parity check.
//
// Computes:  Y = scale * (X @ W0^T) + alpha * (X @ A^T) @ B^T
//
//   X:   row-major (M, K) FP32
//   W0:  row-major (N, K) FP32  -- caller may pre-quantize/dequantize through
//                                  FP8 if testing the FP8 base path; the
//                                  reference treats W0 as plain FP32.
//   A:   row-major (r, K) ternary {-1, 0, +1}, regenerated internally from
//        seed_A via generate_ternary_matrix.
//   B:   row-major (N, r) ternary {-1, 0, +1}, regenerated internally from
//        seed_B via generate_ternary_matrix.
//   Y:   row-major (M, N) FP32 (caller casts to BF16 if needed).
//
// All accumulation is in FP32. Loop order is (m outer, n outer, k inner)
// for both the base and adapter passes, matching the device epilogue's
// FP32-accumulator contract (docs/design-corrections.md correction #1).
//
// Two seeds, not one: the route descriptor in P3 will define the
// route-seed -> (seed_A, seed_B) derivation. Keeping them explicit here
// lets the test harness drive the operands directly.
//
// O(M*N*K + M*r*K + M*r*N) work; pure host code.
void reference_b2b_gemm(std::size_t M, std::size_t N, std::size_t K, std::size_t r,
                        const float* X, const float* W0,
                        std::uint64_t seed_A, std::uint64_t seed_B,
                        float scale, float alpha,
                        float* Y_out);

} // namespace gemm_hopper
