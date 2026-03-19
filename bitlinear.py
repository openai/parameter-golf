"""
BitNet-style ternary linear layer for MLX.

Weights are stored in full precision during training. The forward pass quantizes
them to {-1, 0, +1} with a learned per-tensor scale factor, using a
straight-through estimator (STE) so gradients flow through the quantization.

Reference: Ma et al., "The Era of 1-bit LLMs" (BitNet b1.58)
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class BitLinear(nn.Module):
    """Drop-in replacement for nn.Linear with ternary weight quantization.

    During training, full-precision weights receive gradients via STE.
    During the forward pass, the effective weight is:
        alpha * sign(W) * (|W| > alpha * threshold_ratio)
    where alpha = mean(|W|).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        threshold_ratio: float = 0.5,
    ):
        super().__init__()
        scale = math.sqrt(1.0 / in_dim)
        self.weight = mx.random.normal((out_dim, in_dim)).astype(mx.float32) * scale
        self.bias = mx.zeros((out_dim,), dtype=mx.float32) if bias else None
        self.threshold_ratio = threshold_ratio

    def _quantize_ternary(self, w: mx.array) -> mx.array:
        alpha = mx.mean(mx.abs(w))
        threshold = alpha * self.threshold_ratio
        w_ternary = mx.sign(w) * (mx.abs(w) > threshold)
        w_q = alpha * w_ternary
        # STE: evaluates to w_q in forward, but gradient flows through w.
        return w + mx.stop_gradient(w_q - w)

    def __call__(self, x: mx.array) -> mx.array:
        w_q = self._quantize_ternary(self.weight)
        out = x @ w_q.astype(x.dtype).T
        if self.bias is not None:
            out = out + self.bias.astype(x.dtype)
        return out
