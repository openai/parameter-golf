#pragma once

// 2-bit-packed -> ternary {-1, 0, +1} expander helpers.
// See docs/architecture.md "In-Kernel Rule 30" / "Bit -> ternary mapping".
//
// Mapping (Achlioptas {1/4, 1/2, 1/4}):
//   b0 == 0           -> 0  (encoded 00)
//   b0 == 1, b1 == 0  -> -1 (encoded 10)
//   b0 == 1, b1 == 1  -> +1 (encoded 11)
//
// 2-bit code layout per ternary value, packed LSB-first into a uint64_t:
//   bit 2*i     = "is non-zero" (= b0_i)
//   bit 2*i + 1 = "is positive when non-zero" (= b0_i & b1_i)

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
  #define GH_HOST_DEVICE __host__ __device__ __forceinline__
#else
  #define GH_HOST_DEVICE inline
#endif

namespace gemm_hopper {

// Scatter the 32 bits of `x` into the even positions of a 64-bit word
// (bit i of x -> bit 2i of result). Standard Morton-encode bit-twiddle.
GH_HOST_DEVICE uint64_t expand_to_even_positions(uint32_t x) {
  uint64_t r = static_cast<uint64_t>(x);
  r = (r | (r << 16)) & 0x0000FFFF0000FFFFULL;
  r = (r | (r << 8))  & 0x00FF00FF00FF00FFULL;
  r = (r | (r << 4))  & 0x0F0F0F0F0F0F0F0FULL;
  r = (r | (r << 2))  & 0x3333333333333333ULL;
  r = (r | (r << 1))  & 0x5555555555555555ULL;
  return r;
}

// Pack 32 ternary values from two CA-bit lanes into a 64-bit word (2 bits per ternary).
// `bits_b0` and `bits_b1` hold 32 independent CA bits each.
GH_HOST_DEVICE uint64_t pack_ternary_2bit(uint32_t bits_b0, uint32_t bits_b1) {
  uint64_t evens = expand_to_even_positions(bits_b0);
  uint64_t odds  = expand_to_even_positions(bits_b0 & bits_b1) << 1;
  return evens | odds;
}

// Decode one 2-bit code (low 2 bits of `code`) to a ternary value in {-1, 0, +1}.
GH_HOST_DEVICE int decode_ternary_2bit(uint32_t code) {
  // bit 0 = "is non-zero", bit 1 = sign-when-non-zero.
  if ((code & 1u) == 0u) return 0;
  return (code & 2u) ? 1 : -1;
}

// BF16 representation of {-1.0, 0.0, +1.0}.
//   IEEE-754 bfloat16: sign(1) | exponent(8) | mantissa(7).
//   +0.0  = 0x0000
//   -1.0  = 0xBF80   (sign=1, exponent=01111111, mantissa=0000000)
//   +1.0  = 0x3F80   (sign=0, exponent=01111111, mantissa=0000000)
GH_HOST_DEVICE uint16_t ternary_to_bf16(int ternary) {
  switch (ternary) {
    case  0: return 0x0000;
    case -1: return 0xBF80;
    case +1: return 0x3F80;
    default: return 0x0000; // unreachable; defensive.
  }
}

// Expand 32 ternary values from a packed-2bit word to a BF16 buffer.
// `out_bf16` must be writable for 32 elements (each elem is uint16_t / bf16-bitpattern).
GH_HOST_DEVICE void expand_ternary_2bit_to_bf16(uint64_t packed, uint16_t* out_bf16) {
  // Per-element decode + lookup. The kernel-side fast path will use __byte_perm
  // + sign-flip to vectorize; this is the correctness reference both paths
  // must match bit-for-bit.
  for (int i = 0; i < 32; ++i) {
    uint32_t code = static_cast<uint32_t>((packed >> (2 * i)) & 0x3ULL);
    out_bf16[i] = ternary_to_bf16(decode_ternary_2bit(code));
  }
}

// Byte-perm selector for one ternary code. Source bytes are:
//   x = 0x3F80BF80 -> [0x80, 0xBF, 0x80, 0x3F]
//   y = 0x00000000 -> [0x00, 0x00, 0x00, 0x00]
// Therefore selectors are:
//   00 -> 0x44 => 0x0000
//   10 -> 0x10 => 0xBF80
//   11 -> 0x32 => 0x3F80
// Invalid 01 follows decode_ternary_2bit and maps to -1.
GH_HOST_DEVICE uint32_t ternary_code_to_bf16_byte_perm_selector(uint32_t code) {
  if ((code & 1u) == 0u) return 0x44u;
  return (code & 2u) ? 0x32u : 0x10u;
}

// Expand two adjacent 2-bit ternary codes into a packed pair of BF16 bitpatterns.
// Low 16 bits are element 0; high 16 bits are element 1.
GH_HOST_DEVICE uint32_t expand_ternary_2bit_pair_to_bf16_u32(uint32_t packed_4bit) {
  const uint32_t code0 = packed_4bit & 0x3u;
  const uint32_t code1 = (packed_4bit >> 2) & 0x3u;
#if defined(__CUDA_ARCH__)
  const uint32_t selector =
      ternary_code_to_bf16_byte_perm_selector(code0) |
      (ternary_code_to_bf16_byte_perm_selector(code1) << 8);
  return __byte_perm(0x3F80BF80u, 0x00000000u, selector);
#else
  const uint32_t lo = static_cast<uint32_t>(ternary_to_bf16(decode_ternary_2bit(code0)));
  const uint32_t hi = static_cast<uint32_t>(ternary_to_bf16(decode_ternary_2bit(code1)));
  return lo | (hi << 16);
#endif
}

// Device-backed parity helper implemented in src/generator_kernels.cu.
// Expands `n_words` packed ternary words into `n_words * 32` BF16 bitpatterns.
bool cuda_expand_ternary_2bit_to_bf16(const uint64_t* packed_host,
                                      uint16_t* out_host,
                                      std::size_t n_words);

} // namespace gemm_hopper
