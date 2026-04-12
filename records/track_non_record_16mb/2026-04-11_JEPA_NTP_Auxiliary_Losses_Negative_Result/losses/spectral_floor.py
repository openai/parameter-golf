"""
Spectral Variance Floor Loss
=============================

Anti-collapse regularizer adapted from LeWM's SIGReg for language models.

Key insight: We apply this to hidden-state *deltas* (Δh_t = h_{t+1} - h_t)
rather than absolute states, because language embeddings are naturally anisotropic
(this is load-bearing structure, not a bug). Transition deltas have less intrinsic
semantic structure and benefit more from anti-collapse pressure.

The loss penalizes only total dimensional collapse (singular values → 0), NOT
full isotropy. This is the "minimal sufficient intervention" — it ensures the
model uses all available embedding dimensions without destroying the natural
cone structure of language representations.

Formula:
    L_spec = Σ max(0, ε - σ_i(Cov(ΔH)))
    
where σ_i are eigenvalues of the covariance matrix of hidden-state deltas,
and ε is a small floor (default 0.01).
"""

import torch
from torch import Tensor


def spectral_variance_floor(
    hidden_states: Tensor,
    eps_floor: float = 0.01,
    use_deltas: bool = True,
) -> Tensor:
    """
    Compute the spectral variance floor loss on hidden states.

    Args:
        hidden_states: (batch, seq_len, dim) — hidden states from a target layer.
        eps_floor: Minimum eigenvalue threshold. Dimensions with eigenvalue below
                   this are penalized. Default 0.01.
        use_deltas: If True (default), compute on Δh_t = h_{t+1} - h_t rather
                    than raw h_t. Recommended for language models.

    Returns:
        Scalar loss — sum of max(0, eps - eigenvalue) over all dimensions.
    """
    if use_deltas:
        # Δh_t = h_{t+1} - h_t : captures transition dynamics
        h = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    else:
        h = hidden_states

    # Flatten batch and seq dims: (batch * seq, dim)
    h_flat = h.reshape(-1, h.size(-1)).float()

    # Center the embeddings
    h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)

    # Covariance matrix: (dim, dim)
    n = h_centered.size(0)
    cov = (h_centered.T @ h_centered) / max(n - 1, 1)

    # Eigenvalues of the covariance matrix (symmetric → use eigh)
    eigenvalues = torch.linalg.eigvalsh(cov)  # sorted ascending

    # Penalize any eigenvalue below the floor
    # This catches total dimensional collapse without enforcing isotropy
    violations = torch.clamp(eps_floor - eigenvalues, min=0.0)

    return violations.sum()


def effective_rank_from_cov(hidden_states: Tensor, use_deltas: bool = True) -> float:
    """
    Compute effective rank (entropy-based) of the hidden state covariance.
    Useful as a diagnostic metric — NOT a loss.

    Effective rank = exp(entropy of normalized singular values)
    A value of d means the representation uses all d dimensions equally.
    A value of 1 means all information is on a single axis (total collapse).
    """
    if use_deltas:
        h = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    else:
        h = hidden_states

    h_flat = h.reshape(-1, h.size(-1)).float()
    h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)

    # SVD for singular values
    s = torch.linalg.svdvals(h_centered)
    # Normalize to a probability distribution
    p = s / s.sum()
    p = p[p > 1e-12]  # avoid log(0)
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())
