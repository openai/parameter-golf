"""
Singular Value Spectrum
========================

Logs the full singular value spectrum of hidden state covariance.
Visualizing this as a line plot or histogram reveals:
  - Healthy spectrum: smooth decay with no sharp cliff
  - Collapsed spectrum: few large values, many near zero
  - Over-regularized: flat spectrum (too isotropic — rare in practice)

When plotted over training steps, you should see:
  - Baseline: progressive collapse (tail drops to zero)
  - With spectral floor: tail stays above the floor ε
"""

import torch
from torch import Tensor
import numpy as np


@torch.no_grad()
def compute_singular_spectrum(
    hidden_states: Tensor,
    use_deltas: bool = True,
    top_k: int = 64,
) -> np.ndarray:
    """
    Compute top-k singular values of hidden state distribution.
    
    Args:
        hidden_states: (batch, seq_len, dim)
        use_deltas: If True, compute on Δh
        top_k: Number of singular values to return (default 64).
               Set to dim for the full spectrum.
        
    Returns:
        numpy array of top_k singular values (descending order).
    """
    if use_deltas:
        h = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    else:
        h = hidden_states

    h_flat = h.reshape(-1, h.size(-1)).float()
    h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)

    s = torch.linalg.svdvals(h_centered)
    
    k = min(top_k, s.numel())
    # svdvals returns descending order
    return s[:k].cpu().numpy()
