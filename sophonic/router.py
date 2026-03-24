"""
Sophonic Router: lightweight per-token layer precision routing.

Maps mean-pooled hidden states to per-layer gate scores.
Top-k layers receive precision residual corrections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SophonicRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, k: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.k = k
        self.proj = nn.Linear(hidden_dim, num_layers)
        # Initialize near-uniform so early training doesn't commit to bad routes
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h_pooled: torch.Tensor, hard: bool = False):
        """
        Args:
            h_pooled: (batch,) or (batch, hidden_dim) mean-pooled hidden state
            hard: if True, return binary top-k gates; if False, return soft sigmoid scores
        Returns:
            gates: (batch, num_layers) — soft scores or hard 0/1 gates
        """
        scores = torch.sigmoid(self.proj(h_pooled))  # (batch, num_layers)

        if hard:
            # Top-k hard gating
            topk_vals, topk_idx = scores.topk(self.k, dim=-1)
            gates = torch.zeros_like(scores)
            gates.scatter_(-1, topk_idx, 1.0)
            # Straight-through: gradients flow through soft scores
            gates = scores + (gates - scores).detach()
        else:
            gates = scores

        return gates
