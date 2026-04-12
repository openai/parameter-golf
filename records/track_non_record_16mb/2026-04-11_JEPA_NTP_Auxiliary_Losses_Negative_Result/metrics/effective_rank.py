"""
Effective Rank Metric
======================

Measures how many embedding dimensions are actually being used.

Effective rank = exp(entropy of normalized singular values)
  - Value of d → all d dimensions used equally (perfect)
  - Value of 1 → total collapse to a single axis
  - Values between → partial dimensional usage

This is the primary anti-collapse diagnostic. If spectral floor loss
is working correctly, effective rank should stay close to d (the model dim)
rather than collapsing over training.
"""

import torch
from torch import Tensor


@torch.no_grad()
def compute_effective_rank(
    hidden_states: Tensor,
    use_deltas: bool = True,
) -> float:
    """
    Compute effective rank of hidden state distribution.
    
    Args:
        hidden_states: (batch, seq_len, dim)
        use_deltas: If True, compute on Δh rather than h
        
    Returns:
        Effective rank (float between 1 and dim).
    """
    if use_deltas:
        h = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    else:
        h = hidden_states

    h_flat = h.reshape(-1, h.size(-1)).float()
    h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)

    s = torch.linalg.svdvals(h_centered)

    # Normalize to probability distribution
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]

    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())
