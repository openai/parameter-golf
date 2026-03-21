from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _swiglu_hidden_dim(dim: int, hidden_mult: float) -> int:
    if hidden_mult <= 0:
        raise ValueError(f"SWIGLU_HIDDEN_MULT must be positive, got {hidden_mult}")
    return max(1, int(round(dim * hidden_mult)))


class ReluSquaredMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, linear_cls: type[nn.Linear]):
        super().__init__()
        self.fc = linear_cls(dim, hidden, bias=False)
        self.proj = linear_cls(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: float, linear_cls: type[nn.Linear]):
        super().__init__()
        hidden = _swiglu_hidden_dim(dim, hidden_mult)
        self.up = linear_cls(dim, hidden, bias=False)
        self.gate = linear_cls(dim, hidden, bias=False)
        self.proj = linear_cls(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.gate(x)) * self.up(x))


def build_mlp(
    dim: int,
    mlp_mult: int,
    mlp_hidden: int | None,
    mlp_kind: str,
    swiglu_hidden_mult: float,
    linear_cls: type[nn.Linear],
) -> nn.Module:
    kind = mlp_kind.strip().lower()
    if kind == "relu2":
        hidden = mlp_hidden if mlp_hidden is not None else mlp_mult * dim
        return ReluSquaredMLP(dim, hidden, linear_cls)
    if kind == "swiglu":
        return SwiGLUMLP(dim, swiglu_hidden_mult, linear_cls)
    raise ValueError(f"Unsupported MLP_KIND={mlp_kind!r}; expected 'relu2' or 'swiglu'")
