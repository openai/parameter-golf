"""Fake quantization and export helpers for recurrent core.

Provides STE-based fake quantization for training and matching export
quantization. Supports symmetric quantization with configurable bits (5, 6, 8),
per-tensor and per-row modes, and selective application to recurrent core.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class _FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: Tensor, scale: Tensor, qmin: int, qmax: int) -> Tensor:
        return torch.clamp(torch.round(w / scale), qmin, qmax) * scale

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None, None, None


def _compute_scale(w: Tensor, bits: int, per_row: bool) -> Tensor:
    qmax = (1 << (bits - 1)) - 1
    if per_row and w.ndim == 2:
        amax = w.detach().abs().amax(dim=1, keepdim=True)
    else:
        amax = w.detach().abs().amax()
    return (amax / qmax).clamp_min(1.0 / qmax)


def fake_quantize_weight(w: Tensor, bits: int = 6, per_row: bool = True) -> Tensor:
    """Apply symmetric fake quantization with STE."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax - 1
    scale = _compute_scale(w, bits, per_row)
    return _FakeQuantSTE.apply(w, scale, qmin, qmax)


def fake_quantize_bank(bank: Tensor, bits: int = 6, per_row: bool = True,
                       indices: list[int] | None = None) -> Tensor:
    """Apply fake quantization to selected slices of a 3-D bank tensor.

    If *indices* is None every slice is quantized; otherwise only the
    listed indices are touched and the rest pass through unchanged.
    """
    if indices is None:
        indices = list(range(bank.shape[0]))
    out = bank
    for i in indices:
        q_slice = fake_quantize_weight(bank[i], bits, per_row)
        if out is bank:
            out = bank.clone()
        out[i] = q_slice
    return out


# --------------- export helpers ------------------------------------------------

def quantize_for_export(w: Tensor, bits: int = 6) -> tuple[Tensor, Tensor]:
    """True integer quantization for model export."""
    qmax = (1 << (bits - 1)) - 1
    w32 = w.float()
    if w32.ndim == 2:
        amax = w32.abs().amax(dim=1)
        scale = (amax / qmax).clamp_min(1.0 / qmax).to(torch.float16)
        q = torch.clamp(torch.round(w32 / scale.float()[:, None]),
                        -qmax - 1, qmax).to(torch.int8)
        return q, scale
    amax = w32.abs().max().item()
    scale = torch.tensor(amax / qmax if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(w32 / scale.float()),
                    -qmax - 1, qmax).to(torch.int8)
    return q, scale


def dequantize_exported(q: Tensor, scale: Tensor, dtype: torch.dtype = torch.bfloat16) -> Tensor:
    if scale.ndim > 0:
        return (q.float() * scale.float().view(q.shape[0],
                *([1] * (q.ndim - 1)))).to(dtype)
    return (q.float() * float(scale.item())).to(dtype)
