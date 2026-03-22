---
name: triton-quantized-block-scaled-gemm
description: Teach an AI agent how to implement block-scaled (microscaling) quantized matmul kernels in Triton.
---

# Quantized & Block-Scaled Matmul Kernels in Triton

> **Targets:** Triton >= 3.0; `tl.dot_scaled` requires SM100+/CDNA4; dequantize fallback works on SM70+/CDNA2+

Overview
This guide explains how to implement low-precision block-scaled matrix multiplication in Triton for mxfp4/mxfp8/nvfp4 formats. It covers scale tensor layouts (OCP microscaling 5D), hardware-accelerated tl.dot_scaled, dequantize fallbacks, mixed-format support, and unpacking INT4/FP4 weight encodings. Use FP32/FP16 accumulators for numerical stability.

Key principles / step-by-step
1. Quant format & scales:
   - Block scaling: one floating scale per contiguous block (e.g., 32 elements → 1 scale). Granularities: per-tensor, per-channel, per-group, per-block.
   - OCP microscaling: store scales in a packed 5D layout for contiguous access (batch, head, row_block, col_block, scale_elems). Follow vendor layout (NVIDIA vs AMD differ in minor stride).
2. Hardware path (SM100+/CDNA4):
   - Use tl.dot_scaled(a_ptr, scale_a_ptr, b_ptr, scale_b_ptr) which performs scaleddot with device TCs. Load tiles and corresponding scale tiles alongside data in the K-loop.
3. Dequantize path (fallback hardware):
   - Load quantized tile (packed bits if INT4/FP4). Depack/unpack into FP16/FP32, multiply by scale tile: a_dec = a_unpacked.to(tl.float16) * scale_a.
   - Compute acc with FP32: acc += tl.dot(a_dec, b_dec).to(tl.float32).
4. Mixed formats:
   - Support A in FP8 and B in FP4: load each tile and its scale, dequantize separately or call tl.dot_scaled with both scales if hardware supports mixed types.
5. INT4/FP4 unpacking:
   - For mxfp4/unpacked INT4: load bytes, extract low/high nibble, sign-extend if needed, cast to float and multiply scale.

Practical examples
Hardware-accelerated scaled dot (conceptual):
```python
# hardware TCs
a = tl.load(a_ptr + a_offs)         # packed mxfp8 tile pointer
scale_a = tl.load(scale_a_ptr + s_offs)
b = tl.load(b_ptr + b_offs)
scale_b = tl.load(scale_b_ptr + s_offs_b)
acc += tl.dot_scaled(a, scale_a, b, scale_b)   # returns FP32 accumulatation
```

Dequantize fallback:
```python
a_packed = tl.load(a_ptr + ...)
a_unp = unpack_4bit(a_packed)                  # produce FP16 tensor
a_dec = a_unp.to(tl.float16) * tl.load(scale_a_ptr + ...)
b_dec = ...
acc += tl.dot(a_dec, b_dec).to(tl.float32)
```

Unpack nibble example:
```python
def unpack_4bit(x_byte):
    lo = (x_byte & 0xF).astype(tl.int8)
    hi = ((x_byte >> 4) & 0xF).astype(tl.int8)
    # sign-extend if signed format, then cast to float
    return tl.where(lo>7, lo-16, lo).to(tl.float16), tl.where(hi>7, hi-16, hi).to(tl.float16)
```

Best practices & common pitfalls
- Prefer tl.dot_scaled on supported hardware for best perf and lower register pressure.
- Align block shapes so scales and data tiles have contiguous memory access; conform to vendor OCP scale layout (shuffle indices if necessary).
- Use FP16 for dequantized values and FP32 accumulation to reduce numerical error.
- Avoid atomics on scales; load scale tiles once per K-iteration.
- Benchmark against FP16/cuBLAS and tune block sizes and scale block granularity for memory bandwidth vs compute trade-offs.
- Validate symmetric vs asymmetric quantization behavior (handle zero-point offsets in dequant path).
- Test correctness across edge tails (feature blocks not divisible by block size) and ensure sign-extension for signed 4-bit formats.