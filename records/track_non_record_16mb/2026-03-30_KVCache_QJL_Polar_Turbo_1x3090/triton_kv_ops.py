from __future__ import annotations

import math

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised on hosts without triton
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc


def triton_is_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _require_triton() -> None:
    if triton_is_available():
        return
    if _TRITON_IMPORT_ERROR is not None:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR}") from _TRITON_IMPORT_ERROR
    raise RuntimeError("Triton is unavailable on this host")


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


if triton is not None:

    @triton.jit
    def _qjl_score_kernel(
        q_ptr,
        sign_ptr,
        norm_ptr,
        out_ptr,
        q_stride_b,
        q_stride_h,
        q_stride_d,
        sign_stride_b,
        sign_stride_kh,
        sign_stride_t,
        sign_stride_d,
        norm_stride_b,
        norm_stride_kh,
        norm_stride_t,
        out_stride_b,
        out_stride_h,
        out_stride_t,
        seq_len,
        head_dim,
        num_heads,
        kv_head_repeat,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // num_heads
        h = pid_bh - b * num_heads
        kv_h = h // kv_head_repeat

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)
        mask_t = offs_t < seq_len
        mask_d = offs_d < head_dim

        q = tl.load(
            q_ptr + b * q_stride_b + h * q_stride_h + offs_d * q_stride_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        signs = tl.load(
            sign_ptr
            + b * sign_stride_b
            + kv_h * sign_stride_kh
            + offs_t[:, None] * sign_stride_t
            + offs_d[None, :] * sign_stride_d,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0,
        ).to(tl.float32)
        signs = signs * 2.0 - 1.0
        norms = tl.load(
            norm_ptr + b * norm_stride_b + kv_h * norm_stride_kh + offs_t * norm_stride_t,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)
        dots = tl.sum(signs * q[None, :], axis=1)
        scores = dots * (norms / head_dim)
        tl.store(
            out_ptr + b * out_stride_b + h * out_stride_h + offs_t * out_stride_t,
            scores,
            mask=mask_t,
        )


    @triton.jit
    def _grouped_score_kernel(
        q_ptr,
        code_ptr,
        scale_ptr,
        out_ptr,
        q_stride_b,
        q_stride_h,
        q_stride_d,
        code_stride_b,
        code_stride_kh,
        code_stride_t,
        code_stride_d,
        scale_stride_b,
        scale_stride_kh,
        scale_stride_t,
        scale_stride_g,
        out_stride_b,
        out_stride_h,
        out_stride_t,
        seq_len,
        head_dim,
        group_count,
        num_heads,
        kv_head_repeat,
        inv_sqrt_head_dim,
        GROUP_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // num_heads
        h = pid_bh - b * num_heads
        kv_h = h // kv_head_repeat

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)
        group_offs = offs_d // GROUP_SIZE
        mask_t = offs_t < seq_len
        mask_d = offs_d < head_dim
        mask_scale = mask_t[:, None] & (group_offs[None, :] < group_count)

        q = tl.load(
            q_ptr + b * q_stride_b + h * q_stride_h + offs_d * q_stride_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        codes = tl.load(
            code_ptr
            + b * code_stride_b
            + kv_h * code_stride_kh
            + offs_t[:, None] * code_stride_t
            + offs_d[None, :] * code_stride_d,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0,
        ).to(tl.float32)
        scales = tl.load(
            scale_ptr
            + b * scale_stride_b
            + kv_h * scale_stride_kh
            + offs_t[:, None] * scale_stride_t
            + group_offs[None, :] * scale_stride_g,
            mask=mask_scale,
            other=0.0,
        ).to(tl.float32)
        dots = tl.sum(codes * scales * q[None, :], axis=1)
        scores = dots * inv_sqrt_head_dim
        tl.store(
            out_ptr + b * out_stride_b + h * out_stride_h + offs_t * out_stride_t,
            scores,
            mask=mask_t,
        )


    @triton.jit
    def _grouped_apply_kernel(
        attn_ptr,
        code_ptr,
        scale_ptr,
        out_ptr,
        attn_stride_b,
        attn_stride_h,
        attn_stride_t,
        code_stride_b,
        code_stride_kh,
        code_stride_t,
        code_stride_d,
        scale_stride_b,
        scale_stride_kh,
        scale_stride_t,
        scale_stride_g,
        out_stride_b,
        out_stride_h,
        out_stride_d,
        seq_len,
        head_dim,
        group_count,
        num_heads,
        kv_head_repeat,
        GROUP_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_d = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // num_heads
        h = pid_bh - b * num_heads
        kv_h = h // kv_head_repeat

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < head_dim
        group_offs = offs_d // GROUP_SIZE
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for start_t in tl.range(0, seq_len, BLOCK_T):
            offs_t = start_t + tl.arange(0, BLOCK_T)
            mask_t = offs_t < seq_len
            attn = tl.load(
                attn_ptr + b * attn_stride_b + h * attn_stride_h + offs_t * attn_stride_t,
                mask=mask_t,
                other=0.0,
            ).to(tl.float32)
            codes = tl.load(
                code_ptr
                + b * code_stride_b
                + kv_h * code_stride_kh
                + offs_t[:, None] * code_stride_t
                + offs_d[None, :] * code_stride_d,
                mask=mask_t[:, None] & mask_d[None, :],
                other=0,
            ).to(tl.float32)
            scales = tl.load(
                scale_ptr
                + b * scale_stride_b
                + kv_h * scale_stride_kh
                + offs_t[:, None] * scale_stride_t
                + group_offs[None, :] * scale_stride_g,
                mask=mask_t[:, None] & (group_offs[None, :] < group_count),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(attn[:, None] * codes * scales, axis=0)

        tl.store(
            out_ptr + b * out_stride_b + h * out_stride_h + offs_d * out_stride_d,
            acc,
            mask=mask_d,
        )


def qjl_score(q_rot: Tensor, signs: Tensor, norms: Tensor, *, num_heads: int) -> Tensor:
    _require_triton()
    if q_rot.ndim != 4 or q_rot.size(2) != 1:
        raise ValueError(f"Expected q_rot shape [batch, heads, 1, dim], got {tuple(q_rot.shape)}")
    if signs.ndim != 4 or norms.ndim != 4:
        raise ValueError("Encoded QJL tensors must have shape [batch, kv_heads, seq, dim]")
    q = q_rot.squeeze(2).contiguous().float()
    sign = signs.contiguous()
    norm = norms.squeeze(-1).contiguous()
    batch, heads, head_dim = q.shape
    kv_heads = sign.size(1)
    seq_len = sign.size(2)
    if heads % kv_heads != 0:
        raise ValueError(f"Cannot expand {kv_heads} KV heads to {heads} attention heads")
    out = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
    block_d = min(128, max(16, _next_power_of_two(head_dim)))
    _qjl_score_kernel[(triton.cdiv(seq_len, 128), batch * heads)](
        q,
        sign,
        norm,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sign.stride(0),
        sign.stride(1),
        sign.stride(2),
        sign.stride(3),
        norm.stride(0),
        norm.stride(1),
        norm.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        seq_len,
        head_dim,
        heads,
        heads // kv_heads,
        BLOCK_T=128,
        BLOCK_D=block_d,
        num_warps=4 if block_d <= 64 else 8,
    )
    return out.unsqueeze(2)


def grouped_score(q: Tensor, codes: Tensor, scales: Tensor, *, num_heads: int, group_size: int) -> Tensor:
    _require_triton()
    if q.ndim != 4 or q.size(2) != 1:
        raise ValueError(f"Expected q shape [batch, heads, 1, dim], got {tuple(q.shape)}")
    q_flat = q.squeeze(2).contiguous().float()
    code = codes.contiguous()
    scale = scales.contiguous()
    batch, heads, head_dim = q_flat.shape
    kv_heads = code.size(1)
    seq_len = code.size(2)
    if heads % kv_heads != 0:
        raise ValueError(f"Cannot expand {kv_heads} KV heads to {heads} attention heads")
    out = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
    block_d = min(128, max(16, _next_power_of_two(head_dim)))
    _grouped_score_kernel[(triton.cdiv(seq_len, 128), batch * heads)](
        q_flat,
        code,
        scale,
        out,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        code.stride(0),
        code.stride(1),
        code.stride(2),
        code.stride(3),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        scale.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        seq_len,
        head_dim,
        scale.size(-1),
        heads,
        heads // kv_heads,
        1.0 / math.sqrt(head_dim),
        GROUP_SIZE=group_size,
        BLOCK_T=128,
        BLOCK_D=block_d,
        num_warps=4 if block_d <= 64 else 8,
    )
    return out.unsqueeze(2)


def grouped_apply(attn_probs: Tensor, codes: Tensor, scales: Tensor, *, num_heads: int, group_size: int) -> Tensor:
    _require_triton()
    if attn_probs.ndim != 4 or attn_probs.size(2) != 1:
        raise ValueError(
            f"Expected attn_probs shape [batch, heads, 1, seq_len], got {tuple(attn_probs.shape)}"
        )
    attn = attn_probs.squeeze(2).contiguous().float()
    code = codes.contiguous()
    scale = scales.contiguous()
    batch, heads, seq_len = attn.shape
    kv_heads = code.size(1)
    head_dim = code.size(3)
    if heads % kv_heads != 0:
        raise ValueError(f"Cannot expand {kv_heads} KV heads to {heads} attention heads")
    out = torch.empty((batch, heads, head_dim), device=attn.device, dtype=torch.float32)
    block_d = 32 if head_dim > 32 else max(16, _next_power_of_two(head_dim))
    _grouped_apply_kernel[(triton.cdiv(head_dim, block_d), batch * heads)](
        attn,
        code,
        scale,
        out,
        attn.stride(0),
        attn.stride(1),
        attn.stride(2),
        code.stride(0),
        code.stride(1),
        code.stride(2),
        code.stride(3),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        scale.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        seq_len,
        head_dim,
        scale.size(-1),
        heads,
        heads // kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_T=128,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out.unsqueeze(2)
