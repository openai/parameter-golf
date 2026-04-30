"""XSA (vector-rejection) Triton kernels for GQA attention outputs.

Replaces the torch decomposition:
    y_g = y.reshape(B, T, Hkv, group, D)
    vn  = F.normalize(v, dim=-1).unsqueeze(-2)
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, T, H, D)

with a pair of Triton kernels (fwd + bwd), each launching one program per
(B, T, Hkv) and loading v exactly once.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
@triton.jit
def _xsa_fwd_kernel(
    y_ptr, v_ptr, y_out_ptr, s_ptr,
    sy_b, sy_t, sy_h, sy_d,
    sv_b, sv_t, sv_h, sv_d,
    so_b, so_t, so_h, so_d,
    ss_b, ss_t, ss_h,
    T, Hkv,
    BLOCK_D: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (T * Hkv)
    r = pid - b * (T * Hkv)
    t = r // Hkv
    hk = r - t * Hkv

    d_offs = tl.arange(0, BLOCK_D)

    v = tl.load(v_ptr + b * sv_b + t * sv_t + hk * sv_h + d_offs * sv_d).to(tl.float32)
    vnorm = tl.maximum(tl.sqrt(tl.sum(v * v)), 1e-12)
    vhat = v / vnorm

    for g in tl.static_range(GROUP):
        h = hk * GROUP + g
        y = tl.load(y_ptr + b * sy_b + t * sy_t + h * sy_h + d_offs * sy_d).to(tl.float32)
        s = tl.sum(y * vhat)
        y_out = y - s * vhat
        tl.store(y_out_ptr + b * so_b + t * so_t + h * so_h + d_offs * so_d, y_out)
        tl.store(s_ptr + b * ss_b + t * ss_t + h * ss_h, s)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------
# Let S_g = <y_g, v̂>, U_g = <grad_f_g, v̂>.
# grad_y_g = grad_f_g - U_g · v̂        (same projection as forward)
# grad_v   = -( Σ_g U_g y_g + Σ_g S_g grad_f_g - 2·(Σ_g U_g S_g) · v̂ ) / ||v||
# Note:   Σ_g U_g S_g = < Σ_g U_g y_g, v̂ >   (since S_g = <y_g, v̂> and v̂ is unit)
@triton.jit
def _xsa_bwd_kernel(
    y_ptr, v_ptr, s_ptr, gf_ptr,
    gy_ptr, gv_ptr,
    sy_b, sy_t, sy_h, sy_d,
    sv_b, sv_t, sv_h, sv_d,
    ss_b, ss_t, ss_h,
    sgf_b, sgf_t, sgf_h, sgf_d,
    sgy_b, sgy_t, sgy_h, sgy_d,
    sgv_b, sgv_t, sgv_h, sgv_d,
    T, Hkv,
    BLOCK_D: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (T * Hkv)
    r = pid - b * (T * Hkv)
    t = r // Hkv
    hk = r - t * Hkv

    d_offs = tl.arange(0, BLOCK_D)

    v = tl.load(v_ptr + b * sv_b + t * sv_t + hk * sv_h + d_offs * sv_d).to(tl.float32)
    vnorm = tl.maximum(tl.sqrt(tl.sum(v * v)), 1e-12)
    inv_vnorm = 1.0 / vnorm
    vhat = v * inv_vnorm

    sum_Uy = tl.zeros((BLOCK_D,), dtype=tl.float32)
    sum_SGf = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for g in tl.static_range(GROUP):
        h = hk * GROUP + g
        y = tl.load(y_ptr + b * sy_b + t * sy_t + h * sy_h + d_offs * sy_d).to(tl.float32)
        gf = tl.load(gf_ptr + b * sgf_b + t * sgf_t + h * sgf_h + d_offs * sgf_d).to(tl.float32)
        s_g = tl.load(s_ptr + b * ss_b + t * ss_t + h * ss_h).to(tl.float32)

        U = tl.sum(gf * vhat)
        grad_y_vec = gf - U * vhat
        tl.store(gy_ptr + b * sgy_b + t * sgy_t + h * sgy_h + d_offs * sgy_d, grad_y_vec)

        sum_Uy = sum_Uy + U * y
        sum_SGf = sum_SGf + s_g * gf

    # Σ_g U_g S_g = < Σ_g U_g y_g, v̂ >   (using v̂ unit)
    sum_US = tl.sum(sum_Uy * vhat)
    grad_v = -(sum_Uy + sum_SGf - 2.0 * sum_US * vhat) * inv_vnorm
    tl.store(gv_ptr + b * sgv_b + t * sgv_t + hk * sgv_h + d_offs * sgv_d, grad_v)


# ---------------------------------------------------------------------------
# autograd wrapper
# ---------------------------------------------------------------------------
class XSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert y.dim() == 4 and v.dim() == 4
        B, T, H, D = y.shape
        Bv, Tv, Hkv, Dv = v.shape
        assert B == Bv and T == Tv and D == Dv, f"shape mismatch y={y.shape} v={v.shape}"
        assert H % Hkv == 0, f"H={H} must be divisible by Hkv={Hkv}"
        G = H // Hkv
        assert D in (16, 32, 64, 128, 256), f"D={D} not supported"

        y_out = torch.empty_like(y)
        s = torch.empty(B, T, H, device=y.device, dtype=torch.float32)

        grid = (B * T * Hkv,)
        _xsa_fwd_kernel[grid](
            y, v, y_out, s,
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3),
            s.stride(0), s.stride(1), s.stride(2),
            T, Hkv,
            BLOCK_D=D, GROUP=G,
        )
        ctx.save_for_backward(y, v, s)
        return y_out

    @staticmethod
    def backward(ctx, grad_f: torch.Tensor):
        y, v, s = ctx.saved_tensors
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        G = H // Hkv

        grad_y = torch.empty_like(y)
        grad_v = torch.empty_like(v)

        grid = (B * T * Hkv,)
        _xsa_bwd_kernel[grid](
            y, v, s, grad_f,
            grad_y, grad_v,
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            s.stride(0), s.stride(1), s.stride(2),
            grad_f.stride(0), grad_f.stride(1), grad_f.stride(2), grad_f.stride(3),
            grad_y.stride(0), grad_y.stride(1), grad_y.stride(2), grad_y.stride(3),
            grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
            T, Hkv,
            BLOCK_D=D, GROUP=G,
        )
        return grad_y, grad_v


def xsa_triton(y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return XSAFunction.apply(y, v)


def xsa_torch(y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference implementation matching bigbag's `_xsa_efficient`."""
    B, T, H, D = y.shape
    Hkv = v.size(-2)
    group = H // Hkv
    y_g = y.reshape(B, T, Hkv, group, D)
    vn = F.normalize(v, dim=-1).unsqueeze(-2)
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, T, H, D)
