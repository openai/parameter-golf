---
name: triton-fused-normalizations
description: Implement fused LayerNorm, RMSNorm, and GroupNorm kernels (forward & backward) in Triton with single-pass statistics and two-stage reductions.
---

# Fused Normalization Kernels in Triton (LayerNorm, RMSNorm, GroupNorm)

> **Targets:** Triton >= 2.1, SM70+/CDNA2+

## Overview

Fused normalization kernels compute statistics and apply normalization in a single pass, avoiding extra HBM round-trips. Map one program per normalization "row" (per token for LayerNorm/RMSNorm, per group for GroupNorm). Always use FP32 accumulators.

## Forward formulas

- **LayerNorm:** `x_hat = (x - mean) * rstd; y = x_hat * gamma + beta`
  - `mean = sum(x) / F; var = sum(x*x)/F - mean*mean; rstd = 1/sqrt(var + eps)`
- **RMSNorm:** `y = x * rstd * gamma` where `rstd = 1/sqrt(mean(x^2) + eps)`
  - No mean subtraction — simpler and 2-3x faster than PyTorch for bandwidth-bound shapes
- **GroupNorm:** treat each group as a LayerNorm row

## RMSNorm — standalone kernel

```python
@triton.jit
def rmsnorm_fwd(x_ptr, gamma_ptr, y_ptr, F, eps, BLOCK_F: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_F)
    mask = offs < F
    x = tl.load(x_ptr + row * F + offs, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    rstd = tl.math.rsqrt(ss / F + eps)
    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    y = x * rstd * gamma
    tl.store(y_ptr + row * F + offs, y.to(x_ptr.dtype.element_ty), mask=mask)
```

## RMSNorm — fused into attention epilogue (verified)

From production differential-attention kernel — applies RMSNorm in-register right before the final store, eliminating an extra kernel launch and HBM read/write:

```python
# After online-softmax finalization:
diff = acc_s - lam[:, None] * acc_n    # (BLOCK_M, HEAD_DIM), already FP32

if APPLY_RMS:    # tl.constexpr — compiled out when False
    var = tl.sum(diff * diff, axis=1) / HEAD_DIM           # (BLOCK_M,)
    rstd = tl.math.rsqrt(var + eps)                         # (BLOCK_M,)
    diff = diff * rstd[:, None]                              # normalize
    rms_w = tl.load(RMS_W + offs_d)                         # (HEAD_DIM,) — loaded once
    diff = diff * rms_w[None, :]                             # apply weight

tl.store(out_ptrs, diff.to(OUT.dtype.element_ty), mask=mask_m[:, None])
```

**Key:** `tl.math.rsqrt(var + eps)` is the preferred API for reciprocal square root.

## LayerNorm — forward with feature chunking

When `F > BLOCK_F`, loop over chunks to accumulate partial sums:

```python
@triton.jit
def layernorm_fwd(x_ptr, gamma_ptr, beta_ptr, mean_ptr, rstd_ptr, y_ptr,
                  F, eps, BLOCK_F: tl.constexpr):
    row = tl.program_id(0)
    # Single-pass accumulation
    s = tl.zeros([], dtype=tl.float32)
    ss = tl.zeros([], dtype=tl.float32)
    for chunk_start in range(0, F, BLOCK_F):
        offs = chunk_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(x_ptr + row * F + offs, mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)

    mean = s / F
    rstd = 1.0 / tl.sqrt(ss / F - mean * mean + eps)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Second pass: normalize and store
    for chunk_start in range(0, F, BLOCK_F):
        offs = chunk_start + tl.arange(0, BLOCK_F)
        mask = offs < F
        x = tl.load(x_ptr + row * F + offs, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        g = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x_hat * g + b
        tl.store(y_ptr + row * F + offs, y.to(x_ptr.dtype.element_ty), mask=mask)
```

## Backward — two-stage reduction for dgamma/dbeta

**Kernel A (per-row):** compute dx and partial dgamma/dbeta per block:
```python
@triton.jit
def layernorm_bwd(x_ptr, dy_ptr, gamma_ptr, mean_ptr, rstd_ptr,
                  dx_ptr, dgamma_partial_ptr, dbeta_partial_ptr,
                  F, BLOCK_F: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_F)
    mask = offs < F
    x = tl.load(x_ptr + row * F + offs, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + row * F + offs, mask=mask).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    gamma = tl.load(gamma_ptr + offs, mask=mask).to(tl.float32)

    x_hat = (x - mean) * rstd
    s_dy = tl.sum(dy * gamma, axis=0)
    s_dyx = tl.sum(dy * gamma * x_hat, axis=0)

    # dx = rstd * (dy*gamma - (s_dy + x_hat*s_dyx)/F)
    dx = rstd * (dy * gamma - (s_dy + x_hat * s_dyx) / F)
    tl.store(dx_ptr + row * F + offs, dx.to(x_ptr.dtype.element_ty), mask=mask)

    # Write partial dgamma/dbeta for this row
    tl.store(dgamma_partial_ptr + row * F + offs, dy * x_hat, mask=mask)
    tl.store(dbeta_partial_ptr + row * F + offs, dy, mask=mask)
```

**Kernel B (reduction):** sum partials across rows to get final dgamma/dbeta per feature.

## Weight handling: may be None

Some models use `elementwise_affine=False`. Handle both cases:
```python
has_weight = gamma is not None
if not has_weight:
    gamma = torch.ones(F, device=x.device, dtype=x.dtype)
```

## Best practices

- **FP32 accumulators always** — fp16 sum/sumsq leads to large numerical errors.
- **Save mean and rstd** per row for backward reuse.
- **Two-stage reduction** for dgamma/dbeta avoids atomic contention; use `tl.atomic_add` only when contention is low.
- **Boundary masking:** always mask tail elements when F is not divisible by BLOCK_F.
- **Fuse activation** (GELU, SiLU) into the same kernel after normalization to save bandwidth.
- **Fuse into attention epilogue** when possible — see `triton-fused-epilogue-kernels.md`.
- **Test numerics** vs PyTorch reference: bf16 inputs with fp32 accumulators should give max diff < 1e-3 for standalone normalization.
