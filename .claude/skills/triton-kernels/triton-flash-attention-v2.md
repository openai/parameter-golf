---
name: triton-flash-attention-v2
description: Implement FlashAttention v2 kernels in Triton with online softmax, causal masking, GQA head routing, multi-stream accumulators, and fused epilogues.
---

# FlashAttention v2 kernels in Triton

> **Targets:** Triton >= 2.1, SM70+/CDNA2+

## Overview

FlashAttention v2 computes `O = softmax(QK^T / sqrt(d_k)) V` without materializing the N×N attention matrix. The kernel iterates over K/V blocks, maintains running softmax statistics `(m, l, acc)` in registers, and recomputes attention weights in the backward pass.

## Grid and program mapping

Use a 2D grid: `(cdiv(Q_LEN, BLOCK_M), B * H)` — one program per (query_block, batch×head) pair.

```python
pid_m = tl.program_id(0)     # query block index
pid_bh = tl.program_id(1)    # batch * head index
off_b = pid_bh // H
off_h = pid_bh % H
```

For GQA (grouped-query attention), map Q heads to K/V heads in-kernel:
```python
groups: tl.constexpr = H // H_KV
off_h_kv = off_h // groups   # which KV head serves this Q head
```

## Online softmax — the core loop

Initialize FP32 accumulators before the KV loop:
```python
m = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
l = tl.zeros([BLOCK_M], dtype=tl.float32)
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
```

The KV loop uses unconditional `tl.maximum` — never branch on tensor values:
```python
# Verified pattern (from production differential-attention kernel)
for block_n in range(0, n_blocks):
    offs_kv = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_kv < KV_LEN

    k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale

    # Causal + OOB mask
    if IS_CAUSAL:
        causal_mask = (offs_m[:, None] + prefix_len) >= offs_kv[None, :]
        qk = tl.where(causal_mask & mask_n[None, :], qk, float("-inf"))
    else:
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

    # Online softmax update (unconditional — no tensor `if`)
    m_new = tl.maximum(m, tl.max(qk, axis=1))
    alpha = tl.exp(m - m_new)
    p = tl.exp(qk - m_new[:, None])
    l = l * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
    m = m_new
```

Finalize and store:
```python
acc = acc / (l[:, None] + 1e-10)            # guard against div-by-zero
tl.store(out_ptrs, acc.to(OUT.dtype.element_ty), mask=mask_m[:, None])
```

## Causal masking — lower-right triangle

For causal attention where KV_LEN >= Q_LEN (e.g., prefill with KV cache):
```python
prefix_len = KV_LEN - Q_LEN
# Query at position q_idx attends to k_idx where: q_idx + prefix_len >= k_idx
causal_mask = (offs_m[:, None] + prefix_len) >= offs_kv[None, :]
```

## Multi-stream accumulators (differential / mixture attention)

For N parallel attention streams sharing K/V tile loads, maintain separate `(m, l, acc)` per stream:
```python
# Two streams: signal and noise (differential attention)
m_s = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
l_s = tl.zeros([BLOCK_M], dtype=tl.float32)
acc_s = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

m_n = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
l_n = tl.zeros([BLOCK_M], dtype=tl.float32)
acc_n = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

for block_n in range(n_blocks):
    k_tile = tl.load(...)   # loaded ONCE
    v_tile = tl.load(...)   # loaded ONCE

    qk_s = tl.dot(q_signal, tl.trans(k_tile)) * sm_scale
    qk_n = tl.dot(q_noise,  tl.trans(k_tile)) * sm_scale
    # ... apply masks to both, update both accumulators independently ...

# Combine in-register after loop (no extra HBM round-trip)
acc_s = acc_s / (l_s[:, None] + 1e-10)
acc_n = acc_n / (l_n[:, None] + 1e-10)
diff = acc_s - lam[:, None] * acc_n
```

## Verification harness pattern

Always test against a PyTorch SDPA reference:
```python
def reference(q, q_noise, k, v, lam, is_causal=False):
    # GQA expansion: k[:, :, None, :, :].expand(...).reshape(B, H, Lk, Dh)
    out_sig = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)
    out_noise = F.scaled_dot_product_attention(q_noise, k_exp, v_exp, is_causal=is_causal)
    return out_sig - lam * out_noise

# Tolerances: bf16 → max 6e-2, mean 1e-2; fp16 → max 2e-2, mean 5e-3
```

## TMA tensor descriptors (Hopper+ / SM90+)

On Hopper GPUs, replace pointer arithmetic with hardware TMA for higher bandwidth:

```python
from triton.tools.tensor_descriptor import TensorDescriptor

# Launcher — create descriptors
y_dim = B * H * SEQ_LEN
desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[1, 1])
desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[1, 1])
desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[1, 1])
desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[1, 1])

# Kernel — reconstruct with real block shapes, then load/store
@triton.jit
def _attn_fwd(desc_q, desc_k, desc_v, desc_o, ...,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    desc_q = tl.make_tensor_descriptor(desc_q, shape=[y_dim, HEAD_DIM],
                                        strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(desc_k, shape=[y_dim, HEAD_DIM],
                                        strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM])
    # Load Q once — stays in registers for entire KV loop
    q = desc_q.load([qo_offset_y, 0])

    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=True):
        k = desc_k.load([offset_kv, 0]).T     # .T for K^T in QK^T
        v = desc_v.load([offset_kv, 0])
        qk = tl.dot(q, k) * qk_scale
        # ... online softmax update as before ...

    desc_o.store([qo_offset_y, 0], acc.to(dtype))
```

Key differences from pointer-based path:
- No manual stride computation — TMA handles address generation in hardware
- `warp_specialize=True` in `tl.range` enables producer/consumer warp roles automatically
- `desc.load().T` transposes during load (free on hardware)
- Pass `TensorDescriptor` objects instead of raw pointers + strides

## Best practices

- Apply `sm_scale` after `tl.dot`, not by pre-scaling Q — avoids promoting Q from bf16 to fp32 which causes dtype mismatch in `tl.dot`.
- Use `tl.trans(k)`, not the deprecated `trans_b` kwarg.
- Cast `p.to(v_tile.dtype)` before `tl.dot(p, v)` — Triton requires matching dtypes.
- Add `+ 1e-10` to the denominator when dividing by `l` to guard against all-masked rows.
- For causal decode (Lq=1), use small BLOCK_M (16) to avoid wasted compute.
- Use `1.44269504 * sm_scale` (= `sm_scale / ln(2)`) with `tl.math.exp2` instead of `tl.exp` for slightly faster softmax on NVIDIA hardware.
- Backward pass: recompute S blocks using saved Q/K and `logsumexp = m + tl.log(l)` per query row.
