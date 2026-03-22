---
name: triton-kernels
description: Write optimized Triton GPU kernels for deep learning operations. Covers the full spectrum from basic vector ops to Flash Attention, persistent matmul, fused normalization, quantized GEMM, and memory-efficient patterns.
---

# Writing Optimized Triton GPU Kernels

> **Targets:** Triton >= 2.1, any GPU with `tl.dot` support (SM70+/CDNA2+)

## Core Patterns (always apply)

**Kernel structure:** Use `@triton.jit` decorator. Get block ID with `tl.program_id(axis)`. Compute element offsets with `tl.arange(0, BLOCK_SIZE)`. Build `mask = offsets < n_elements` for all loads/stores.

**Block sizes:** Strongly prefer powers of two (required for `tl.arange`; non-power-of-two may work but can reduce performance). Declare as `tl.constexpr` parameters. Use `@triton.autotune` to sweep `BLOCK_SIZE_M/N/K` configs per hardware.

**Memory hierarchy:** Keep intermediates in SRAM via block-level reductions (`tl.sum`, `tl.max`) before writing to global memory. Fuse multiple pointwise ops into one kernel to avoid DRAM round-trips.

**Matmul:** Use `tl.dot(a, b)` for tensor core operations. Always accumulate in `tl.float32` when inputs are FP16. For L2 cache locality, use grouped tile ordering via `group_id = pid // GROUP_SIZE`.

**Grid launching:** Size grid dynamically: `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)`.

**Masking:** ALWAYS mask boundary loads/stores: `tl.load(ptr + offs, mask=offs < dim, other=0.0)`. Missing masks corrupt memory silently.

**Benchmarking:** Use `triton.testing.Benchmark` with `x_names`, `x_vals`, `line_arg`, `line_vals` to compare against PyTorch baselines.

## Quick Reference Examples

Fused row-wise softmax — verified, based on official Triton tutorial:
```python
@triton.jit
def fused_softmax(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=-1e9)
    x_max = tl.max(x, axis=0)
    ex = tl.exp(x - x_max)
    out = ex / tl.sum(ex, axis=0)
    tl.store(out_ptr + row * cols + offs, out, mask=mask)
```

Seed-based dropout — verified, based on official Triton tutorial:
```python
@triton.jit
def dropout(x_ptr, out_ptr, seed, p, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    r = tl.rand(seed, offs)  # Philox PRNG, deterministic
    keep = r > p
    tl.store(out_ptr + offs, x * keep / (1.0 - p), mask=mask)
```

## Performance Bottleneck Quick-Reference

When optimizing an existing kernel, classify the bottleneck first (profile with `ncu`):

| Bottleneck | Diagnosis | Fix |
|------------|-----------|-----|
| **Memory-bound** | DRAM throughput > 60% of peak, compute < 30% | PID swizzle, TMA, fuse ops to reduce loads |
| **Compute-bound** | Tensor core utilization > 60%, DRAM < 40% | Persistent kernels, increase `num_stages`, warp specialization |
| **Underutilized** | Both < 60%, high stall metrics | Reduce register pressure, increase `num_warps`, autotune |

See `triton-gpu-kernel-optimization.md` for specific NCU metric names and detailed strategies.

## Specialized Topics

Read these files for detailed guidance when the task involves these areas:

| Task | File to read |
|------|-------------|
| Flash Attention / fused self-attention | `triton-flash-attention-v2.md` |
| Persistent kernels, warp specialization, TMA | `triton-persistent-warp-matmul.md` |
| LayerNorm, RMSNorm, GroupNorm (fwd + bwd) | `triton-fused-normalizations.md` |
| FP4/FP8 quantized matmul, block scaling | `triton-quantized-block-scaled-gemm.md` |
| Kernel fusion, Philox dropout, recomputation | `triton-memory-efficient-patterns.md` |
| General tiled GEMM, autotune, benchmarking | `triton-gpu-kernel-optimization.md` |
| Fusing normalization/gating/residual into attention or matmul epilogue | `triton-fused-epilogue-kernels.md` |
| Sequential stateful processing (LRU routing, mutable register state) | `triton-sequential-stateful-blocks.md` |
| Launcher tile selection, num_stages/num_warps heuristics | `triton-dynamic-launcher-tiling.md` |

**When to read specialized files:** Only read the relevant file when the user's task specifically involves that topic. The core patterns above are sufficient for basic kernels (vector ops, elementwise fusion, simple reductions).
