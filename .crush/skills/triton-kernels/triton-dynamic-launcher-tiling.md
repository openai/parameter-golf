---
name: triton-dynamic-launcher-tiling
description: Build Triton kernel launchers that pick tile sizes, warps, and stages at runtime based on problem shape, dtype, and hardware.
---

# Dynamic Tile & Pipeline Launcher for Triton

> **Targets:** Triton >= 2.1, SM70+/CDNA2+; shared memory heuristics tuned for A100/H100

## Overview

For real-time inference, `@triton.autotune` warmup is unacceptable. Write a Python launcher that selects BLOCK sizes, `num_warps`, and `num_stages` heuristically from input shapes and dtype. The launcher passes these as `tl.constexpr` kernel params so the compiler optimizes without runtime branching.

## Verified launcher (from production differential FlashAttention)

This exact pattern runs in production. It handles decode (Lq=1), short prefill, long prefill, large HEAD_DIM, and FP32 inputs:

```python
def _diff_flash_fwd(q, q_noise, k, v, lam, out, *, rms_weight, eps, sm_scale, is_causal, APPLY_RMS):
    B, H, Q_LEN, HEAD_DIM = q.shape
    _, H_KV, KV_LEN, _ = k.shape

    # ---- Tile selection based on sequence lengths ----
    if Q_LEN <= 16:
        BLOCK_M = 16          # decode path
    elif Q_LEN <= 64:
        BLOCK_M = 64          # short prefill
    else:
        BLOCK_M = 128         # long prefill

    if KV_LEN <= 64:
        BLOCK_N = 32
    elif KV_LEN <= 256:
        BLOCK_N = 64
    else:
        BLOCK_N = 128

    # ---- Cap for register pressure ----
    if HEAD_DIM > 128:
        BLOCK_M = min(BLOCK_M, 64)
        BLOCK_N = min(BLOCK_N, 64)

    # ---- Dtype-aware reduction (FP32 = 2x shared memory pressure) ----
    dtype_bytes = q.element_size()
    if dtype_bytes >= 4:
        BLOCK_M = min(BLOCK_M, 32)
        BLOCK_N = min(BLOCK_N, 64)

    # ---- Pipeline depth ----
    tile_bytes = (BLOCK_M + BLOCK_N) * HEAD_DIM * dtype_bytes * 2
    num_stages = 1 if tile_bytes > 64 * 1024 else 2

    # ---- Dummy pointer for optional features ----
    if rms_weight is None:
        rms_weight = torch.empty(0, device=q.device, dtype=q.dtype)

    # ---- Grid: 2D (query_blocks, batch*heads) ----
    grid = (triton.cdiv(Q_LEN, BLOCK_M), B * H)

    _kernel[grid](
        q, q_noise, k, v, lam, out, rms_weight,
        B, H, H_KV, Q_LEN, KV_LEN, HEAD_DIM,
        # all strides passed explicitly via .stride()
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # ... k, v, lam, out strides ...
        sm_scale, eps,
        IS_CAUSAL=is_causal,
        APPLY_RMS=APPLY_RMS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=num_stages,
        num_warps=4,
    )
```

## Decision table summary

| Parameter | Condition | Value | Rationale |
|-----------|-----------|-------|-----------|
| BLOCK_M | Lq <= 16 | 16 | Decode: 1 query row, don't waste compute |
| BLOCK_M | 16 < Lq <= 64 | 64 | Short prefill |
| BLOCK_M | Lq > 64 | 128 | Long prefill: maximize throughput |
| BLOCK_N | Lk <= 64 | 32 | Small KV cache |
| BLOCK_N | 64 < Lk <= 256 | 64 | Medium KV |
| BLOCK_N | Lk > 256 | 128 | Large KV: amortize loop overhead |
| BLOCK_M/N | HEAD_DIM > 128 | min(current, 64) | Cap register pressure |
| BLOCK_M | dtype_bytes >= 4 | min(current, 32) | FP32 doubles shared memory |
| BLOCK_N | dtype_bytes >= 4 | min(current, 64) | FP32 doubles shared memory |
| num_stages | tile_bytes > 64KB | 1 | No room for double buffering |
| num_stages | tile_bytes <= 64KB | 2 | Latency hiding via pipelining |
| num_stages | tile_bytes < 16KB | 3-4 | Triple/quad buffer for tiny tiles |
| num_warps | BLOCK_M >= 128, BLOCK_N >= 128 | 8 | Fill large tiles |
| num_warps | BLOCK_M <= 16 | 2 | Decode path: few rows |
| num_warps | otherwise | 4 | Default |

## Grid patterns

**Attention (2D):** One program per (query_block, batch×head).
```python
grid = (triton.cdiv(Q_LEN, BLOCK_M), B * H)
```

**Matmul (1D):** One program per output tile, use `//` and `%` for 2D mapping.
```python
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
# in-kernel: m_block = pid // num_tiles_n; n_block = pid % num_tiles_n
```

**Sequential stateful (1D):** One program per batch element.
```python
grid = (B,)
```

## GQA head mapping

Pass `H` and `H_KV` as kernel args; compute the mapping in-kernel:
```python
groups: tl.constexpr = H // H_KV
off_h_kv = off_h // groups    # which KV head serves this Q head
```

## Optional feature handling

When a feature is disabled (e.g., no RMSNorm), pass a dummy empty tensor and use `tl.constexpr` to skip the code path entirely:
```python
# Launcher
rms_weight = torch.empty(0, device=q.device, dtype=q.dtype) if rms_weight is None else rms_weight
APPLY_RMS = rms_weight.numel() > 0

# Kernel
if APPLY_RMS:        # tl.constexpr — compiled out when False
    rms_w = tl.load(RMS_W + offs_d)
    diff = diff * rstd[:, None] * rms_w[None, :]
```

## GPU hardware reference (from KernelAgent/Meta specs database)

| GPU | Arch | Peak FP16 (TFLOPS) | Peak BW (GB/s) | SMs | L1/SM (KB) | L2 (MB) | VRAM |
|-----|------|--------------------|----------------|-----|------------|---------|------|
| H100 SXM5 | Hopper | 1979 | 3350 | 132 | 256 | 50 | 80 GB HBM3 |
| H100 PCIe | Hopper | 1513 | 2000 | 114 | 256 | 50 | 80 GB HBM2e |
| A100 SXM4 80GB | Ampere | 312 | 2039 | 108 | 192 | 40 | 80 GB HBM2e |
| A100 SXM4 40GB | Ampere | 312 | 1555 | 108 | 192 | 40 | 40 GB HBM2e |
| A100 PCIe 80GB | Ampere | 312 | 1935 | 108 | 192 | 40 | 80 GB HBM2e |
| RTX 4090 | Ada | 82.6 | 1008 | 128 | 128 | 72 | 24 GB GDDR6X |
| RTX 5080 | Blackwell | 56.3 | 960 | 84 | 128 | 64 | 16 GB GDDR7 |

**Shared memory per SM:** H100 = 228 KB configurable, A100 = 164 KB, Ada/Turing = 64-128 KB.

**Tile budget estimate:** `tile_bytes = (BLOCK_M + BLOCK_N) * HEAD_DIM * dtype_bytes * 2`

**Arithmetic intensity threshold:** A kernel is memory-bound when `FLOPs / bytes_transferred < peak_TFLOPS / peak_BW_TB`. For H100 SXM: `1979 / 3.35 ≈ 591 FLOP/byte`. For A100 SXM: `312 / 2.04 ≈ 153 FLOP/byte`.

## Best practices

- **Conservative tiles:** prefer undersized tiles over oversized — register/shared memory spills silently kill performance.
- **Stride-based addressing:** always pass strides via `.stride()` rather than assuming contiguous layout. Call `.contiguous()` in the launcher if needed.
- **Validate hardware:** A100 and H100 have different shared memory budgets. Test on target device.
- **Fallback:** provide a PyTorch reference for CPU/non-Triton environments; check `_HAS_TRITON` and `tensor.is_cuda` before launching.
