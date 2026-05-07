"""Triton-backed regenerated-adapter wrapper.

This is the first native path for the live trainer. It accelerates the adapter
term, keeps A/B frozen, and uses PyTorch autograd for the backward formula.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

def _identity_disable(fn):
    return fn


_dynamo_disable = getattr(getattr(torch, "compiler", None), "disable", None)
if _dynamo_disable is None:
    _dynamo_disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
if _dynamo_disable is None:
    _dynamo_disable = _identity_disable

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - local Windows env has no Triton wheel.
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and tl is not None


if triton_available():
    @triton.jit
    def _matmul_kernel(a, b, c, m:tl.constexpr, n:tl.constexpr, k:tl.constexpr,
                       sa0:tl.constexpr, sa1:tl.constexpr, sb0:tl.constexpr, sb1:tl.constexpr,
                       sc0:tl.constexpr, sc1:tl.constexpr, bm:tl.constexpr, bn:tl.constexpr,
                       bk:tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * bm + tl.arange(0, bm)
        offs_n = pid_n * bn + tl.arange(0, bn)
        offs_k = tl.arange(0, bk)
        acc = tl.zeros((bm, bn), tl.float32)
        for k0 in range(0, k, bk):
            kk = k0 + offs_k
            av = tl.load(a + offs_m[:, None] * sa0 + kk[None, :] * sa1,
                         mask=(offs_m[:, None] < m) & (kk[None, :] < k), other=0.0)
            bv = tl.load(b + kk[:, None] * sb0 + offs_n[None, :] * sb1,
                         mask=(kk[:, None] < k) & (offs_n[None, :] < n), other=0.0)
            acc += tl.dot(av, bv, out_dtype=tl.float32)
        tl.store(c + offs_m[:, None] * sc0 + offs_n[None, :] * sc1, acc,
                 mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def _triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not triton_available():
        raise RuntimeError("Triton is unavailable; install/use a Linux CUDA PyTorch environment")
    if not (a.is_cuda and b.is_cuda):
        raise RuntimeError("Triton path requires CUDA tensors")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"expected 2D matmul shapes (M,K)@(K,N), got {tuple(a.shape)} and {tuple(b.shape)}")
    a = a.contiguous()
    b = b.contiguous()
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.float32)
    bm = 16 if m <= 16 else 32
    bn = 32 if n <= 32 else 64
    bk = 32
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _matmul_kernel[grid](a, b, out, m, n, k, a.stride(0), a.stride(1),
                         b.stride(0), b.stride(1), out.stride(0), out.stride(1),
                         bm, bn, bk, num_warps=4)
    return out


class _TritonAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x_shape = tuple(x.shape)
        x2 = x.reshape(-1, x_shape[-1]).contiguous()
        z = _triton_matmul(x2, a.t().contiguous())
        out = _triton_matmul(z.to(x.dtype), b.t().contiguous())
        ctx.save_for_backward(a.float(), b.float())
        ctx.x_shape = x_shape
        return out.reshape(*x_shape[:-1], b.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
        go = grad_out.reshape(-1, b.shape[0]).float()
        grad_x = (go @ b) @ a
        return grad_x.reshape(ctx.x_shape).to(grad_out.dtype), None, None


def _backend_from_env(use_triton: bool | None = None, backend: str | None = None) -> str:
    if backend is not None:
        value = backend
    elif use_triton is not None:
        value = "triton" if use_triton else "torch"
    else:
        value = os.environ.get("GEMM_HOPPER_BACKEND", "triton")
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "triton"}:
        return "triton"
    if value in {"0", "false", "no", "torch", "pytorch"}:
        return "torch"
    if value == "native":
        return "native"
    raise ValueError(f"unknown GEMM_HOPPER_BACKEND/USE_HOPPER_REGEN_GEMM backend: {value!r}")


@lru_cache(maxsize=1)
def _native_module():
    root = Path(__file__).resolve().parents[1]
    build_dir = root / "build-torch-ext"
    build_dir.mkdir(exist_ok=True)
    from torch.utils.cpp_extension import load

    return load(
        name="gemm_hopper_native_ext",
        sources=[
            str(root / "python" / "gemm_hopper_native.cpp"),
            str(root / "python" / "gemm_hopper_native_kernel.cu"),
            str(root / "src" / "reference.cpp"),
        ],
        extra_include_paths=[str(root / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90"],
        build_directory=str(build_dir),
        verbose=bool(int(os.environ.get("GEMM_HOPPER_NATIVE_VERBOSE", "0"))),
    )


def _require_native_sm90(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise RuntimeError("GEMM Hopper native backend requires a CUDA tensor on an SM90/Hopper device")
    if not torch.cuda.is_available():
        raise RuntimeError("GEMM Hopper native backend requires CUDA")
    major, minor = torch.cuda.get_device_capability(x.device)
    if major < 9:
        raise RuntimeError(
            "GEMM Hopper native backend requires an SM90/Hopper CUDA device; "
            f"current device capability is {major}.{minor}"
        )


@_dynamo_disable
def _native_adapter_gemm(x: torch.Tensor, seed_a: int, seed_b: int,
                         out_features: int, rank: int) -> torch.Tensor:
    _require_native_sm90(x)
    x_shape = tuple(x.shape)
    if len(x_shape) == 0:
        raise ValueError("native adapter expects at least 1D input")
    k = int(x_shape[-1])
    x2 = x.reshape(-1, k).contiguous()
    mod = _native_module()
    out = mod.adapter_forward(
        x2,
        int(seed_a) & 0xFFFFFFFFFFFFFFFF,
        int(seed_b) & 0xFFFFFFFFFFFFFFFF,
        int(out_features),
        k,
        int(rank),
    )
    return out.reshape(*x_shape[:-1], int(out_features)).to(x.dtype)


class _NativeSeedAdapter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, seed_a: int, seed_b: int,
                out_features: int, rank: int,
                a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = _native_adapter_gemm(x, seed_a, seed_b, out_features, rank)
        ctx.save_for_backward(a.float(), b.float())
        ctx.x_shape = tuple(x.shape)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
        go = grad_out.reshape(-1, b.shape[0]).float()
        grad_x = (go @ b) @ a
        return grad_x.reshape(ctx.x_shape).to(grad_out.dtype), None, None, None, None, None, None


def regen_adapter_gemm(x: torch.Tensor, regen_a: torch.Tensor, regen_b: torch.Tensor,
                       use_triton: bool | None = None,
                       backend: str | None = None,
                       seed_a: int | None = None,
                       seed_b: int | None = None) -> torch.Tensor:
    """Compute only the frozen regenerated adapter term: (X @ A.T) @ B.T."""
    selected = _backend_from_env(use_triton=use_triton, backend=backend)
    a = regen_a.to(device=x.device, dtype=x.dtype)
    b = regen_b.to(device=x.device, dtype=x.dtype)
    if selected == "native":
        if seed_a is None or seed_b is None:
            raise RuntimeError("GEMM Hopper native backend requires seed_a and seed_b")
        return _NativeSeedAdapter.apply(x, int(seed_a), int(seed_b), b.shape[0], a.shape[0], a, b)
    if selected == "triton":
        return _TritonAdapter.apply(x, a, b)
    return F.linear(F.linear(x, a), b)


def regen_b2b_gemm(x: torch.Tensor, weight: torch.Tensor, regen_a: torch.Tensor,
                   regen_b: torch.Tensor, alpha: torch.Tensor,
                   bias: torch.Tensor | None = None,
                   use_triton: bool | None = None,
                   backend: str | None = None,
                   seed_a: int | None = None,
                   seed_b: int | None = None) -> torch.Tensor:
    """Compute W*x plus alpha times the regenerated B2B adapter term."""
    bias = bias.to(x.dtype) if bias is not None else None
    base = F.linear(x, weight.to(x.dtype), bias)
    adapter = regen_adapter_gemm(
        x,
        regen_a,
        regen_b,
        use_triton=use_triton,
        backend=backend,
        seed_a=seed_a,
        seed_b=seed_b,
    )
    return base + alpha.to(x.dtype) * adapter
