// CPU reference. See docs/test-plan.md Tier-1 "B2B mathematical correctness".
// Phase 1: multi-word Rule-30 evolution + ternary matrix generation.
// Phase 2: B2B GEMM reference (Y = scale * X @ W0^T + alpha * X @ A^T @ B^T).

#include "gemm_hopper/reference.hpp"
#include "gemm_hopper/generator.hpp"
#include "gemm_hopper/ternary.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gemm_hopper {

void rule30_evolve_multiword(std::uint64_t* state, std::size_t n_words, std::size_t steps) {
  if (n_words == 0 || steps == 0) return;
  std::vector<std::uint64_t> next(n_words);
  for (std::size_t step = 0; step < steps; ++step) {
    for (std::size_t w = 0; w < n_words; ++w) {
      // carry_l = the word to the *left* (lower index). The Rule-30 step extracts
      // its useful bit (bit 0, the rightmost cell) via `(carry_l << 63)` and
      // discards the rest, so passing the full word is fine.
      std::uint64_t carry_l = (w == 0)             ? 0ULL : state[w - 1];
      std::uint64_t carry_r = (w == n_words - 1)   ? 0ULL : state[w + 1];
      next[w] = rule30_step(state[w], carry_l, carry_r);
    }
    for (std::size_t w = 0; w < n_words; ++w) state[w] = next[w];
  }
}

void generate_ternary_matrix(std::uint64_t seed,
                             std::size_t rows,
                             std::size_t cols,
                             std::int8_t* out) {
  const std::size_t total = rows * cols;
  if (total == 0) return;

  // Multi-word CA state. Two independent lanes (A, B) for the double-CA XOR mix.
  // n_words = 256 -> 16,384 cells per lane. Boundary collapse on this size lane
  // reaches the central word at ~step 8000; we sample at most ~total / (n_words*16)
  // steps below, which keeps a comfortable margin for matrices up to ~4M ternary
  // values. Using n_words=64 introduces a measurable ~3% bias toward zero in the
  // sampled bits because the propagating-edge structure reaches the inner region
  // at this scale (verified by test_diagnose_source_bias in tests/test_reference.cpp).
  constexpr unsigned n_words = 256;
  std::uint64_t state_a[n_words];
  std::uint64_t state_b[n_words];
  init_multi_word_state(seed,                       state_a, n_words);
  init_multi_word_state(derive_paired_seed(seed),   state_b, n_words);

  // Warmup discard.
  rule30_evolve_multiword(state_a, n_words, kRule30Warmup);
  rule30_evolve_multiword(state_b, n_words, kRule30Warmup);

  // Per step we pair words spatially to decorrelate b0 and b1: word w supplies
  // b0 (low 32 bits of mixed word), word w + n_words/2 supplies b1. Halving the
  // word stride avoids the within-word correlation that Rule 30's locally-causal
  // dynamics introduce when b0 and b1 are taken from the same uint64. Each pair
  // yields 32 ternary values; n_words/2 pairs per step => n_words*16 ternary/step.
  static_assert((n_words & 1u) == 0u, "n_words must be even for the half-stride pairing.");
  constexpr unsigned half = n_words / 2;
  std::size_t out_idx = 0;
  while (out_idx < total) {
    for (unsigned w = 0; w < half && out_idx < total; ++w) {
      std::uint64_t mixed_b0 = state_a[w]        ^ state_b[w];
      std::uint64_t mixed_b1 = state_a[w + half] ^ state_b[w + half];
      std::uint32_t b0 = static_cast<std::uint32_t>(mixed_b0 & 0xFFFFFFFFULL);
      std::uint32_t b1 = static_cast<std::uint32_t>(mixed_b1 & 0xFFFFFFFFULL);
      std::uint64_t packed = pack_ternary_2bit(b0, b1);
      for (int i = 0; i < 32 && out_idx < total; ++i) {
        std::uint32_t code = static_cast<std::uint32_t>((packed >> (2 * i)) & 0x3ULL);
        out[out_idx++] = static_cast<std::int8_t>(decode_ternary_2bit(code));
      }
    }
    if (out_idx < total) {
      rule30_evolve_multiword(state_a, n_words, 1);
      rule30_evolve_multiword(state_b, n_words, 1);
    }
  }
}

void reference_b2b_gemm(std::size_t M, std::size_t N, std::size_t K, std::size_t r,
                        const float* X, const float* W0,
                        std::uint64_t seed_A, std::uint64_t seed_B,
                        float scale, float alpha,
                        float* Y_out) {
  if (M == 0 || N == 0) return;

  // Regenerate A (r, K) and B (N, r) from the route seeds.
  std::vector<std::int8_t> A(r * K);
  std::vector<std::int8_t> B(N * r);
  if (r != 0 && K != 0) generate_ternary_matrix(seed_A, r, K, A.data());
  if (N != 0 && r != 0) generate_ternary_matrix(seed_B, N, r, B.data());

  // Z = X @ A^T : (M, r). FP32 accumulation.
  std::vector<float> Z(M * r, 0.0f);
  for (std::size_t m = 0; m < M; ++m) {
    const float* x_row = X + m * K;
    float* z_row = Z.data() + m * r;
    for (std::size_t j = 0; j < r; ++j) {
      const std::int8_t* a_row = A.data() + j * K;
      float acc = 0.0f;
      for (std::size_t k = 0; k < K; ++k) {
        acc += x_row[k] * static_cast<float>(a_row[k]);
      }
      z_row[j] = acc;
    }
  }

  // Y = scale * (X @ W0^T) + alpha * (Z @ B^T). One outer loop over (m, n);
  // base and adapter share the (m, n) slot but accumulate in independent FP32
  // partials to match docs/design-corrections.md correction #1 (FP32-only acc).
  for (std::size_t m = 0; m < M; ++m) {
    const float* x_row = X + m * K;
    const float* z_row = Z.data() + m * r;
    float* y_row = Y_out + m * N;
    for (std::size_t n = 0; n < N; ++n) {
      const float* w0_row = W0 + n * K;
      float base = 0.0f;
      for (std::size_t k = 0; k < K; ++k) {
        base += x_row[k] * w0_row[k];
      }
      const std::int8_t* b_row = B.data() + n * r;
      float adapter = 0.0f;
      for (std::size_t j = 0; j < r; ++j) {
        adapter += z_row[j] * static_cast<float>(b_row[j]);
      }
      y_row[n] = scale * base + alpha * adapter;
    }
  }
}

} // namespace gemm_hopper
