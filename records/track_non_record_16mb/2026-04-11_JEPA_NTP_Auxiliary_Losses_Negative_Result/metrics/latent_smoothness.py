"""
Latent Path Curvature / Smoothness
====================================

Measures how "straight" the latent trajectory is across sequence positions.

In LeWM, one of the key emergent properties is "Temporal Latent Path Straightening" —
latent trajectories become smoother/more linear over training without explicit
regularization. We measure whether this emerges in LLMs too.

Curvature is computed as the angular deviation of second-order differences:
  - Low curvature → smooth, predictable trajectories (good)
  - High curvature → erratic, unpredictable transitions (bad)

A steadily decreasing curvature over training confirms the path-straightening effect.
"""

import torch
from torch import Tensor


@torch.no_grad()
def compute_latent_curvature(
    hidden_states: Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute mean angular curvature of latent trajectories.
    
    Uses second-order finite differences:
        d_t = h_{t+1} - h_t     (first difference / velocity)
        dd_t = d_{t+1} - d_t    (second difference / acceleration)
        curvature ≈ mean(||dd_t|| / ||d_t||)
    
    Args:
        hidden_states: (batch, seq_len, dim) — must have seq_len >= 3
        eps: Small constant for numerical stability
        
    Returns:
        Mean curvature (float). Lower = smoother.
    """
    if hidden_states.size(1) < 3:
        return 0.0

    h = hidden_states.float()

    # First differences (velocity): (batch, seq_len-1, dim)
    d = h[:, 1:, :] - h[:, :-1, :]

    # Second differences (acceleration): (batch, seq_len-2, dim)
    dd = d[:, 1:, :] - d[:, :-1, :]

    # Curvature = ||acceleration|| / ||velocity|| for each position
    d_norm = d[:, :-1, :].norm(dim=-1) + eps  # align with dd
    dd_norm = dd.norm(dim=-1)

    curvature = (dd_norm / d_norm).mean()
    return float(curvature.item())


@torch.no_grad()
def compute_cosine_smoothness(
    hidden_states: Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Alternative smoothness metric: mean cosine similarity between
    consecutive velocity vectors.
    
    Values close to 1.0 mean nearly-straight trajectories.
    Values close to 0.0 mean random-walk trajectories.
    
    Args:
        hidden_states: (batch, seq_len, dim) — must have seq_len >= 3
        
    Returns:
        Mean cosine similarity of consecutive velocities (float in [-1, 1]).
    """
    if hidden_states.size(1) < 3:
        return 0.0

    h = hidden_states.float()
    d = h[:, 1:, :] - h[:, :-1, :]

    d1 = d[:, :-1, :]  # velocity at t
    d2 = d[:, 1:, :]   # velocity at t+1

    cos_sim = torch.nn.functional.cosine_similarity(d1, d2, dim=-1, eps=eps)
    return float(cos_sim.mean().item())
