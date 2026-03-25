"""Stability monitoring and control for recurrent passes.

Provides per-pass diagnostics, hidden-state clipping, learnable residual
scaling, and a cheap Jacobian proxy regulariser.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field


@dataclass
class PassDiagnostics:
    h_norms: list[float] = field(default_factory=list)
    delta_norms: list[float] = field(default_factory=list)
    error_norms: list[float] = field(default_factory=list)
    correction_norms: list[float] = field(default_factory=list)
    growth_ratios: list[float] = field(default_factory=list)

    def reset(self):
        for lst in (self.h_norms, self.delta_norms, self.error_norms,
                    self.correction_norms, self.growth_ratios):
            lst.clear()

    def summary(self) -> dict[str, list[float]]:
        return {
            "h_norms": list(self.h_norms),
            "delta_norms": list(self.delta_norms),
            "error_norms": list(self.error_norms),
            "correction_norms": list(self.correction_norms),
            "growth_ratios": list(self.growth_ratios),
        }


class RecurrentStabilizer:
    """Manages stability diagnostics and optional controls for recurrence."""

    def __init__(
        self,
        clip_hidden: bool = False,
        clip_value: float = 10.0,
        clip_mode: str = "value",
        jacobian_proxy_weight: float = 0.0,
        eps: float = 1e-6,
    ):
        self.clip_hidden = clip_hidden
        self.clip_value = clip_value
        self.clip_mode = clip_mode
        self.jacobian_proxy_weight = jacobian_proxy_weight
        self.eps = eps
        self.diagnostics = PassDiagnostics()

    def clip(self, h: Tensor) -> Tensor:
        if not self.clip_hidden:
            return h
        if self.clip_mode == "value":
            return torch.clamp(h, -self.clip_value, self.clip_value)
        norm = h.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.clip_value / (norm + self.eps), max=1.0)
        return h * scale

    def record_pass(
        self,
        h_prev: Tensor,
        h_next: Tensor,
        error: Tensor | None = None,
        correction: Tensor | None = None,
    ):
        with torch.no_grad():
            h_pn = h_prev.float().norm().item()
            h_nn = h_next.float().norm().item()
            self.diagnostics.h_norms.append(h_nn)
            self.diagnostics.delta_norms.append(
                (h_next - h_prev).float().norm().item()
            )
            self.diagnostics.growth_ratios.append(h_nn / (h_pn + self.eps))
            if error is not None:
                self.diagnostics.error_norms.append(error.float().norm().item())
            if correction is not None:
                self.diagnostics.correction_norms.append(
                    correction.float().norm().item()
                )

    def jacobian_proxy_loss(self, h_in: Tensor, h_out: Tensor) -> Tensor:
        """Finite-difference proxy for Jacobian spectral norm."""
        if self.jacobian_proxy_weight <= 0:
            return h_in.new_zeros(())
        delta = h_out - h_in
        ratio = delta.norm() / (h_in.norm() + self.eps)
        return self.jacobian_proxy_weight * torch.relu(ratio - 1.0).square()

    def reset(self):
        self.diagnostics.reset()


class ResidualScale(nn.Module):
    """Learnable per-pass residual scaling:
       h_{k+1} = h_k + alpha_k * F(h_k + c_k)"""

    def __init__(self, num_passes: int, init_value: float = 1.0):
        super().__init__()
        self.scales = nn.Parameter(
            torch.full((num_passes,), init_value, dtype=torch.float32)
        )

    def forward(self, residual: Tensor, pass_idx: int) -> Tensor:
        return self.scales[pass_idx].to(dtype=residual.dtype) * residual
