"""
Cosine-MSE Auxiliary Prediction Loss
=====================================

Next-position latent prediction head inspired by LeWM's prediction loss,
adapted for language models.

Key design choices:
  1. L2-normalize before MSE → effectively cosine distance, no magnitude pressure
  2. Stop-gradient on target → gradients only flow through the predictor MLP
  3. Dedicated projection head → isolates auxiliary gradients from the main trunk

This avoids the Neural Collapse conflict where MSE pushes embeddings toward
equal norms while cross-entropy pushes toward classifier-specific structure.

Formula:
    L_cosine_mse = || ĥ_{t+1}/||ĥ_{t+1}|| - sg(h_{t+1}/||h_{t+1}||) ||²

where ĥ_{t+1} = predictor_mlp(h_t) and sg = stop_gradient.
"""

import torch
import torch.nn as nn
from torch import Tensor


class LatentPredictor(nn.Module):
    """
    2-layer MLP projection head for next-position latent prediction.
    
    Architecture mirrors SimCLR/BYOL projection heads — keeps auxiliary
    loss gradients scoped to this head rather than leaking into the trunk.
    
    Input:  h_t  (batch, seq_len, dim)
    Output: ĥ_{t+1} (batch, seq_len-1, dim) — predicted next-position embedding
    """

    def __init__(self, dim: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim, bias=False),
        )
        # Initialize output layer to near-zero for stable training start
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, h: Tensor) -> Tensor:
        """
        Predict next-position latent from current-position hidden states.
        
        Args:
            h: (batch, seq_len, dim)
        Returns:
            (batch, seq_len - 1, dim) — predicted h_{t+1} for positions 0..T-2
        """
        return self.net(h[:, :-1, :])


def cosine_mse_loss(
    predicted: Tensor,
    target: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Cosine-MSE loss with stop-gradient on target.
    
    L2-normalizes both predicted and target, then computes MSE.
    This is equivalent to 2 * (1 - cosine_similarity) but with cleaner gradients.
    
    Args:
        predicted: (batch, seq_len-1, dim) — from LatentPredictor
        target: (batch, seq_len-1, dim) — actual h_{t+1}, will be detached
        eps: Small constant for numerical stability in normalization
        
    Returns:
        Scalar loss.
    """
    # Stop-gradient on target: gradients only flow through the predictor
    target = target.detach()

    # L2-normalize → cosine distance (no magnitude pressure)
    pred_norm = predicted / (predicted.norm(dim=-1, keepdim=True) + eps)
    tgt_norm = target / (target.norm(dim=-1, keepdim=True) + eps)

    # MSE on normalized vectors = 2 * (1 - cos_sim)
    loss = (pred_norm - tgt_norm).pow(2).mean()

    return loss
