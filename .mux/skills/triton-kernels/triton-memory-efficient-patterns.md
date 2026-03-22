---
name: triton-memory-efficient-patterns
description: Teach an AI agent to minimize GPU memory via seed-based PRNG, fusion, and recomputation in Triton kernels.
---

# Memory-efficient Triton kernels: seed PRNG, fusion, and recomputation

> **Targets:** Triton >= 2.1, SM70+/CDNA2+

Overview
This guide describes patterns for minimizing GPU memory footprint in Triton kernels: Philox seed-based PRNG (generate dropout masks on-the-fly), activation checkpointing via recomputation, fused elementwise/residual kernels, safe in-place updates, and using tl.extra.libdevice for math functions. These techniques trade a bit of compute for large memory savings and fewer global-memory round-trips.

Key principles / step-by-step
1. Seed-based Philox PRNG:
   - Use a single seed and per-element offsets to generate deterministic random numbers: r = tl.rand(seed, offset). Create mask = r > p. Forward and backward regenerate identical mask from the same (seed, offsets) so no mask tensor is stored.
   - Keep seed + base_offset per kernel launch; offset = base_offset + linear_index.
2. Activation checkpointing / recomputation:
   - Don’t store intermediates: recompute cheap intermediates in backward kernels (e.g., activations, linear inputs). Balance compute vs saved memory.
3. Kernel fusion:
   - Fuse chains of pointwise ops into one kernel (bias + activation + dropout + residual) to avoid extra reads/writes.
   - Use in-place writes when input can be safely overwritten.
4. Use tl.extra.libdevice for transcendental functions to keep computations on-device and avoid library calls.
5. Grid design:
   - Map one program per element / row; loop over feature chunks if needed. Ensure offsets for PRNG are computed consistently.

Practical examples
Fused bias + GELU + seed-dropout + residual (simplified):
```python
@triton.jit
def fused_bias_gelu_dropout_res(x_ptr, bias_ptr, res_ptr, out_ptr, seed, p, M, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + idx)
    b = tl.load(bias_ptr + (idx % bias_len))
    y = x + b
    # GELU via libdevice erf
    y_act = 0.5 * y * (1.0 + tl.erf(y / 2**0.5))
    # PRNG per-element offset
    offsets = idx.astype(tl.int32)
    r = tl.rand(seed, offsets)
    mask = r > p
    y_drop = (y_act * mask) * (1.0 / (1.0 - p))
    res = tl.load(res_ptr + idx)
    out = y_drop + res
    tl.store(out_ptr + idx, out)
```

Seed-based dropout regeneration in backward:
```python
# backward: regenerate r = tl.rand(seed, offsets) to get same mask, compute dx without stored mask
```

Best practices & pitfalls
- Use fp32 accumulators where needed; tl.rand returns uniform [0,1).
- Keep seed and offset computation consistent between forward and backward; use a per-layer seed and contiguous offsets (e.g., linear element index).
- Recompute only cheap intermediates—expensive recompute may outweigh memory savings.
- Avoid atomic updates in fused kernels when possible; prefer per-thread outputs or staged reductions.
- Measure memory vs compute trade-offs and benchmark GB/s: fusion often yields 2–4× speedups vs unfused chains.
- Be careful with in-place: ensure no other consumer needs original values. Validate numerical parity with unfused baseline.