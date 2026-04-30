"""BitNet b1.58 ternary weight Linear layer + packed ternary serialization.

Reference: Ma et al. 2024, "The Era of 1-bit LLMs", arxiv:2402.17764, eq (1)-(2).

Drop-in replacement for nn.Linear / CastedLinear. During training the weight is
constrained to {-γ, 0, +γ} via per-tensor absmean quantization with STE for
gradient pass-through. Activations are kept bf16 (no activation quant in v1 —
isolate the weight-quant axis).

v2 (0086) adds pack_ternary/unpack_ternary helpers for 2-bit packed serialization.
4 ternary values per byte. Encoding offset (+1): -1→0b00, 0→0b01, +1→0b10. Bits 0-1
hold val[0], bits 2-3 hold val[1], bits 4-5 hold val[2], bits 6-7 hold val[3].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Linear):
    """BitNet b1.58 ternary weight Linear layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # Per-tensor absmean scale γ (recomputed each forward; not a param).
        gamma = w.abs().mean().clamp(min=1e-8)
        # Ternarize: round(W/γ) clipped to {-1, 0, +1}.
        w_q = (w / gamma).round().clamp_(-1.0, 1.0)
        # Straight-through estimator: forward uses w_q, backward identity.
        w_ste = w + (w_q - w).detach()
        # Apply scale at matmul time.
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, (gamma * w_ste).to(x.dtype), bias)


def pack_ternary(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Ternarize w via absmean γ, pack into uint8 (4 vals/byte).

    Returns (packed_uint8, gamma_float). Packed shape: ⌈N/4⌉ bytes where
    N = w.numel(). Original shape recovered via shape arg in unpack_ternary.
    """
    w_flat = w.detach().contiguous().view(-1).float()
    gamma = w_flat.abs().mean().clamp(min=1e-8).item()
    w_q = (w_flat / gamma).round().clamp_(-1.0, 1.0).to(torch.int8)
    # Map -1 → 0, 0 → 1, +1 → 2.
    codes = (w_q + 1).to(torch.uint8)
    n = codes.numel()
    pad = (4 - n % 4) % 4
    if pad:
        codes = torch.cat([codes, torch.zeros(pad, dtype=torch.uint8)])
    quad = codes.view(-1, 4)
    packed = (
        quad[:, 0] | (quad[:, 1] << 2) | (quad[:, 2] << 4) | (quad[:, 3] << 6)
    ).contiguous()
    return packed, gamma


def unpack_ternary(packed: torch.Tensor, gamma: float, shape) -> torch.Tensor:
    """Inverse of pack_ternary. Returns fp32 tensor of `shape` such that
    BitLinear forward recovers the original effective weight γ·W_q.

    Subtle: BitLinear recomputes γ' = abs.mean(W) every forward. If we returned
    W_q · γ directly, then γ' = γ · frac_nonzero ≠ γ → forward output shrinks.
    Fix: scale by 1/frac_nonzero so abs.mean comes out to γ. Then BitLinear's
    re-ternarize maps each element to W_q again (rescaled values are ±1/frac_nonzero
    or 0, which round to ±1 or 0, then clamp to ternary), and the recovered γ' = γ.
    Effective output γ' · W_q' = γ · W_q exactly matches training.
    """
    n_total = 1
    for s in shape:
        n_total *= s
    p = packed.to(torch.uint8)
    c0 = p & 0b11
    c1 = (p >> 2) & 0b11
    c2 = (p >> 4) & 0b11
    c3 = (p >> 6) & 0b11
    codes = torch.stack([c0, c1, c2, c3], dim=1).view(-1)
    codes = codes[:n_total].to(torch.int8)
    w_q = codes.to(torch.float32) - 1.0  # back to {-1, 0, +1}
    n_nonzero = (w_q != 0).sum().item()
    frac_nonzero = max(n_nonzero / max(n_total, 1), 1e-8)
    # Scale so abs.mean(out) = gamma. abs.mean(α·W_q) = α·frac_nonzero, set = γ → α = γ/frac_nonzero.
    alpha = gamma / frac_nonzero
    return (w_q * alpha).view(*shape)
