"""Error feedback modules for recurrent quantization correction.

Implements low-rank residual approximation and correction operators
to compensate for quantization error amplification in recurrent passes.

  e_k  = U (V^T h_k)                         -- low-rank residual approx.
  c_k  = D_k(e_k)                             -- correction operator
  h_{k+1} = f_{W_q}(h_k + c_k)               -- corrected recurrent update
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch import Tensor


class LowRankResidual(nn.Module):
    """e_k = U (V^T h_k)  with U, V in R^{d x r}."""

    def __init__(self, dim: int, rank: int = 2):
        super().__init__()
        self.V = nn.Parameter(torch.randn(dim, rank) * (1.0 / math.sqrt(dim)))
        self.U = nn.Parameter(torch.randn(dim, rank) * (1.0 / math.sqrt(rank)))

    def forward(self, h: Tensor) -> Tensor:
        return (h @ self.V) @ self.U.T


class DiagonalFeedback(nn.Module):
    """c_k = d odot e_k."""

    def __init__(self, dim: int, init_ones: bool = True):
        super().__init__()
        init_val = torch.ones(dim) if init_ones else torch.zeros(dim)
        self.d = nn.Parameter(init_val)

    def forward(self, e: Tensor) -> Tensor:
        return self.d.to(dtype=e.dtype) * e


class LowRankFeedback(nn.Module):
    """c_k = U_D (V_D^T e_k)  with U_D, V_D in R^{d x r}."""

    def __init__(self, dim: int, rank: int = 2):
        super().__init__()
        self.V_D = nn.Parameter(torch.randn(dim, rank) * (1.0 / math.sqrt(dim)))
        self.U_D = nn.Parameter(torch.randn(dim, rank) * (1.0 / math.sqrt(rank)))

    def forward(self, e: Tensor) -> Tensor:
        return (e @ self.V_D) @ self.U_D.T


class AffineJunction(nn.Module):
    """c_k^{aff} = gamma_k odot h_k + beta_k."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, h: Tensor) -> Tensor:
        return self.gamma.to(dtype=h.dtype) * h + self.beta.to(dtype=h.dtype)


class ErrorFeedbackModule(nn.Module):
    """Combined error-feedback path:  residual -> correction -> (optional junction).

    Supports shared or per-pass correction operators.  Correction is inactive
    on pass 0 (the first recurrence pass sees no prior quantization residual).

    Args:
        dim:              model hidden dimension
        rank:             rank for low-rank components
        feedback_mode:    'identity' | 'diagonal' | 'low_rank'
        per_pass:         separate correction per pass if True
        num_passes:       number of recurrence passes (K)
        affine_junction:  add an affine junction path
    """

    def __init__(
        self,
        dim: int,
        rank: int = 2,
        feedback_mode: str = "diagonal",
        per_pass: bool = False,
        num_passes: int = 3,
        affine_junction: bool = False,
    ):
        super().__init__()
        self.feedback_mode = feedback_mode
        self.per_pass = per_pass
        self.num_passes = num_passes

        self.residual = LowRankResidual(dim, rank)

        if feedback_mode == "identity":
            self.correction: nn.Module | nn.ModuleList | None = None
        elif feedback_mode == "diagonal":
            if per_pass:
                self.correction = nn.ModuleList(
                    [DiagonalFeedback(dim) for _ in range(num_passes)]
                )
            else:
                self.correction = DiagonalFeedback(dim)
        elif feedback_mode == "low_rank":
            if per_pass:
                self.correction = nn.ModuleList(
                    [LowRankFeedback(dim, rank) for _ in range(num_passes)]
                )
            else:
                self.correction = LowRankFeedback(dim, rank)
        else:
            raise ValueError(f"Unknown feedback_mode: {feedback_mode}")

        self.junction: AffineJunction | None = (
            AffineJunction(dim) if affine_junction else None
        )

    def forward(self, h: Tensor, pass_idx: int) -> Tensor | None:
        """Return correction tensor, or None for pass 0."""
        if pass_idx == 0:
            return None
        e = self.residual(h)
        if self.correction is None:
            c = e
        elif self.per_pass:
            c = self.correction[pass_idx](e)
        else:
            c = self.correction(e)
        if self.junction is not None:
            c = c + self.junction(h)
        return c

    def extra_repr(self) -> str:
        return (f"mode={self.feedback_mode}, per_pass={self.per_pass}, "
                f"passes={self.num_passes}")

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
