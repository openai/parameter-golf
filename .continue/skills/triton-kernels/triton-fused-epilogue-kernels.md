---
name: triton-fused-epilogue-kernels
description: Fuse epilogue ops (normalization, gating, residual, activation, dropout) into Triton attention/matmul kernels to eliminate HBM round-trips.
---

# Fused Epilogue Kernels in Triton

> **Targets:** Triton >= 2.1, SM70+/CDNA2+

## Overview

Fusing epilogue work directly into attention or GEMM kernels avoids extra HBM writes/reads and kernel launches. Perform all final math in-register immediately before the final `tl.store`. Use `tl.constexpr` bool flags so the compiler emits branch-free specialized variants.

## Pattern 1: Fused differential attention + RMSNorm epilogue

Verified pattern from production kernel — two online-softmax accumulators sharing K/V loads, with RMSNorm fused before the final store:

```python
@triton.jit
def _diff_flash_attn_fwd_kernel(
    Q, Q_NOISE, K, V, LAM, OUT, RMS_W,
    # ... strides, dimensions ...
    IS_CAUSAL: tl.constexpr,
    APPLY_RMS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # ... setup offsets, load q_tile and qn_tile ...

    # Two independent online-softmax accumulators (both in FP32)
    m_s = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_s = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_s = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_n = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_n = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_n = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for block_n in range(n_blocks):
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)  # loaded ONCE
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # loaded ONCE

        qk_s = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        qk_n = tl.dot(qn_tile, tl.trans(k_tile)) * sm_scale
        # ... apply causal/OOB masks to both ...

        # Update signal stream
        m_s_new = tl.maximum(m_s, tl.max(qk_s, axis=1))
        alpha_s = tl.exp(m_s - m_s_new)
        p_s = tl.exp(qk_s - m_s_new[:, None])
        l_s = l_s * alpha_s + tl.sum(p_s, axis=1)
        acc_s = acc_s * alpha_s[:, None] + tl.dot(p_s.to(v_tile.dtype), v_tile)
        m_s = m_s_new

        # Update noise stream (identical structure)
        m_n_new = tl.maximum(m_n, tl.max(qk_n, axis=1))
        alpha_n = tl.exp(m_n - m_n_new)
        p_n = tl.exp(qk_n - m_n_new[:, None])
        l_n = l_n * alpha_n + tl.sum(p_n, axis=1)
        acc_n = acc_n * alpha_n[:, None] + tl.dot(p_n.to(v_tile.dtype), v_tile)
        m_n = m_n_new

    # ---- Epilogue: differential + optional RMSNorm ----
    acc_s = acc_s / (l_s[:, None] + 1e-10)
    acc_n = acc_n / (l_n[:, None] + 1e-10)
    diff = acc_s - lam_tile[:, None] * acc_n     # all in-register

    if APPLY_RMS:
        var = tl.sum(diff * diff, axis=1) / HEAD_DIM
        rstd = tl.math.rsqrt(var + eps)
        diff = diff * rstd[:, None]
        rms_w = tl.load(RMS_W + offs_d)          # load HEAD_DIM weights once
        diff = diff * rms_w[None, :]

    tl.store(out_ptrs, diff.to(OUT.dtype.element_ty), mask=mask_m[:, None])
```

**Key insight:** K/V tiles are loaded once, used by both streams. This halves HBM bandwidth vs. two separate attention calls.

## Pattern 2: Fused GEMM + bias + activation + dropout

```python
@triton.jit
def gemm_kernel(..., APPLY_BIAS: tl.constexpr, APPLY_LEAKY: tl.constexpr,
                APPLY_DROPOUT: tl.constexpr):
    # ... K-loop accumulating acc in FP32 ...

    if APPLY_BIAS:
        b = tl.load(bias_ptr + col_offsets)
        acc = acc + b[None, :]
    if APPLY_LEAKY:
        acc = tl.where(acc > 0, acc, acc * 0.01)
    if APPLY_DROPOUT:
        # Seed-based Philox dropout — no mask tensor stored
        r = tl.rand(dropout_seed, offs)
        keep = r > dropout_p
        acc = acc * keep / (1.0 - dropout_p)
    tl.store(C_ptr + offs, acc.to(C.dtype.element_ty))
```

## Pattern 3: Gating + residual fusion

```python
if APPLY_GATE:
    g = tl.load(gate_ptr + row_idx)        # (M,) — per-token gate
    res = tl.load(residual_ptr + offs)      # (M, D)
    out = g[:, None] * attn_out + res       # fused: 1 store instead of 3 kernels
    tl.store(OUT_ptr + offs, out.to(OUT.dtype.element_ty))
```

## Launcher: constexpr flags and dummy pointers

```python
def launch_kernel(q, k, v, *, rms_weight=None, is_causal=False):
    APPLY_RMS = rms_weight is not None
    # Pass dummy empty tensor when feature is disabled
    if rms_weight is None:
        rms_weight = torch.empty(0, device=q.device, dtype=q.dtype)

    kernel[grid](q, k, v, rms_weight, ...,
                 IS_CAUSAL=is_causal, APPLY_RMS=APPLY_RMS,
                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
```

## Best practices

- **constexpr flags** eliminate dead code at compile time — no runtime branch overhead.
- **Load small vectors outside K-loops:** bias, norm weights, gate values are loaded once, not per-K-block.
- **FP32 accumulation throughout:** apply epilogue in FP32, cast only at `tl.store` with `.to(OUT.dtype.element_ty)`.
- **RMSNorm formula:** `var = tl.sum(x*x, axis=1) / dim; rstd = tl.math.rsqrt(var + eps)`. Always add eps.
- **Dropout in epilogues:** prefer seed-based `tl.rand(seed, offs)` over loading mask from HBM. Forward and backward regenerate the same mask from `(seed, offs)`. See `triton-memory-efficient-patterns.md`.
- **Register pressure:** multi-stream fusions (2+ accumulators) increase register usage. Monitor occupancy; reduce BLOCK sizes if needed.
- **Verification:** test fused kernel numerics against unfused PyTorch reference. Expect bf16 max diff ~6e-2 for attention-based epilogues.
