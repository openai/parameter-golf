"""
TernaryLinear: BitLinear drop-in replacement for CastedLinear.

Stores weights in fp32 (for optimizer quality), quantizes to ternary {-1, 0, +1}
in the forward pass via absmean scaling + STE (straight-through estimator).

At inference/serialization time, weights are truly ternary — each weight is one of
three values, packed at 5 trits per byte for 1.6 bits/param density.

During training, bf16 shadow weights are maintained (same as every other QAT approach).
The density advantage is purely at artifact time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary quantization-aware training (QAT).

    Forward pass: quantize weight to {-1, 0, +1} * scale via absmean,
    then do standard F.linear. Gradient flows through via STE.

    Supports per-row or per-tensor scaling:
      - per_row=True:  scale_i = mean(|W[i,:]|) for each output row
      - per_row=False: scale = mean(|W|) for entire weight matrix
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        per_row: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.per_row = per_row
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        # Mark for zero-init detection (same convention as CastedLinear)
        self._zero_init = False

    def _quantize_ternary(self, w: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize weight to ternary via absmean scaling.

        Returns (w_quantized_scaled, scale) where w_quantized_scaled = ternary * scale.
        """
        if self.per_row:
            # Per-row scale: each output channel gets its own magnitude
            scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            # Per-tensor scale
            scale = w.abs().mean().clamp_min(1e-8)
        # Quantize: round(w / scale) clamped to {-1, 0, +1}
        w_ternary = torch.clamp(torch.round(w / scale), -1, 1)
        return w_ternary * scale, scale

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        # Quantize to ternary
        w_q, _ = self._quantize_ternary(w)
        # STE: forward uses quantized weights, backward sees continuous weights
        # w + (w_q - w).detach() has the value of w_q but the gradient of w
        w_ste = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_ste, bias)

    def get_ternary_weights(self) -> tuple[Tensor, Tensor]:
        """Extract the discrete ternary weights and scales (for serialization).

        Returns:
            ternary: int8 tensor of {-1, 0, +1}, shape (out_features, in_features)
            scale: fp16 tensor, shape (out_features, 1) if per_row else scalar
        """
        with torch.no_grad():
            w = self.weight.float()
            if self.per_row:
                scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                scale = w.abs().mean().clamp_min(1e-8)
            ternary = torch.clamp(torch.round(w / scale), -1, 1).to(torch.int8)
            return ternary, scale.to(torch.float16)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, per_row={self.per_row}"
        )


def replace_linear_with_ternary(
    module: nn.Module,
    per_row: bool = True,
    skip_patterns: tuple[str, ...] = ("tok_emb", "lm_head", "bigram"),
) -> nn.Module:
    """Replace CastedLinear/nn.Linear layers with TernaryLinear, except those
    matching skip_patterns (embeddings, head, etc. stay fp16)."""
    for name, child in list(module.named_children()):
        full_name = name
        if any(pat in full_name for pat in skip_patterns):
            continue
        if isinstance(child, nn.Linear):
            ternary = TernaryLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                per_row=per_row,
            )
            # Copy weights
            ternary.weight.data.copy_(child.weight.data)
            if child.bias is not None and ternary.bias is not None:
                ternary.bias.data.copy_(child.bias.data)
            # Preserve zero-init flag
            ternary._zero_init = getattr(child, "_zero_init", False)
            setattr(module, name, ternary)
        else:
            replace_linear_with_ternary(child, per_row=per_row, skip_patterns=skip_patterns)
    return module
