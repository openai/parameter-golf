---
name: triton-gpu-kernel-optimization
description: Write high-performance tiled Triton GPU kernels with autotune, grouped tile ordering, stride-based addressing, and proper benchmarking.
---

# Tiled GEMM & General Kernel Optimization in Triton

> **Targets:** Triton >= 2.1, SM70+/CDNA2+

## Overview

High-performance Triton kernels follow a consistent structure: block-tiled work distribution, stride-based pointer arithmetic, FP32 accumulation, boundary masking, and autotune sweeps. This file covers the general tiled GEMM pattern, L2-friendly tile ordering, stride-based addressing (verified from production kernels), and benchmarking.

## Stride-based pointer arithmetic (verified pattern)

Always pass strides via `.stride()` rather than assuming contiguous layout. This is the pattern used in all production kernels:

```python
# Launcher — pass all strides explicitly
kernel[grid](
    q, k, v, out,
    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    ...
)

# Kernel — build pointers from batch/head offsets and strides
@triton.jit
def _kernel(Q, K, V, OUT,
            stride_qb, stride_qh, stride_qm, stride_qd,
            stride_kb, stride_kh, stride_kn, stride_kd,
            ...):
    off_b = pid_bh // H
    off_h = pid_bh % H
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh \
             + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
```

**Why:** Tensors from `.transpose()`, `.permute()`, or GQA expansion are often non-contiguous. Stride-based addressing handles all layouts correctly. Call `.contiguous()` in the launcher only when profiling shows it helps.

## Tiled GEMM with autotune and grouped ordering

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=1),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr = 8,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)

    # Grouped tile ordering for L2 cache locality
    num_tiles_in_group = GROUP_SIZE * num_n
    group_id = pid // num_tiles_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + ((pid % num_tiles_in_group) % group_size_m)
    pid_n = (pid % num_tiles_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator — always accumulate in FP32 for FP16/BF16 inputs
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    # Cast to output dtype only at store time
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

**Grouped tile ordering** processes `GROUP_SIZE` adjacent M-tiles before advancing N, keeping A-tile data in L2 across consecutive programs.

## Grid launching

Size grids dynamically with lambda:
```python
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
gemm_kernel[grid](A, B, C, M, N, K, ...)
```

For attention-style 2D grids:
```python
grid = (triton.cdiv(Q_LEN, BLOCK_M), B * H)
```

## Elementwise fusion

Fuse pointwise ops into a single kernel to avoid HBM round-trips:
```python
@triton.jit
def fused_add_relu(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    out = tl.maximum(x + y, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)
```

## Benchmarking

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(12, 25)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        ylabel='GB/s',
        plot_name='fused-add-relu',
        args={},
    )
)
def benchmark(N, provider):
    x = torch.randn(N, device='cuda', dtype=torch.float16)
    y = torch.randn(N, device='cuda', dtype=torch.float16)
    if provider == 'triton':
        out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
        return triton.testing.do_bench(lambda: fused_add_relu[grid](x, y, out, N, BLOCK=1024))
    else:
        return triton.testing.do_bench(lambda: torch.relu(x + y))
```

## Bottleneck diagnosis with NCU metrics

Before optimizing, profile with `ncu` (NVIDIA Nsight Compute) and classify the kernel into one of three categories:

| Category | Symptom | Key NCU metrics |
|----------|---------|----------------|
| **Memory-bound** | DRAM throughput near peak, compute underutilized | `dram__throughput.avg.pct_of_peak_sustained_elapsed` > 60%, tensor core % < 30% |
| **Compute-bound** | Tensor core / SM utilization high, memory idle | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` > 60%, DRAM < 40% |
| **Underutilized** | Neither saturated (<60% both) — stalls or low occupancy | High `smsp__warp_issue_stalled_*` percentages, `launch__occupancy_limit_*` flags |

**Key NCU metrics to check:**



**Fix strategies by category:**

- **Memory-bound** → PID swizzle for L2 locality, TMA descriptors (Hopper+), reduce loads via fusion. See `triton-persistent-warp-matmul.md`.
- **Compute-bound** → Persistent programming (loop over tiles), increase `num_stages`, enable warp specialization.
- **Underutilized** → Reduce register pressure (smaller BLOCK sizes), increase `num_warps`, sweep autotune configs.

## Best practices

- **Always mask:** `mask = offs < dim` on every `tl.load`/`tl.store`. Missing masks corrupt memory silently.
- **BLOCK sizes:** Strongly prefer powers of two (required for `tl.arange`; non-power-of-two may work but can reduce performance). Declare as `tl.constexpr`.
- **FP32 accumulation:** Always use `tl.float32` accumulators for FP16/BF16 inputs. Cast with `.to(OUT.dtype.element_ty)` only at `tl.store`.
- **Stride-based addressing:** Pass strides via `.stride()` — never assume contiguous. See `triton-dynamic-launcher-tiling.md` for launcher patterns.
- **Autotune configs:** Include at least one small config (32x32) for small problem sizes. Use `key=['M', 'N', 'K']` so Triton re-tunes when shapes change.
- **Recompute over materialize:** Prefer recomputing PRNG masks (Philox `tl.rand`) in backward over storing large boolean masks. See `triton-memory-efficient-patterns.md`.
- **`tl.max_contiguous` / `tl.multiple_of`:** Hint the compiler for better codegen on aligned accesses: `offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)`.
- **Fallback:** Provide a PyTorch reference for CPU/non-Triton environments; check `_HAS_TRITON` and `tensor.is_cuda` before launching.
