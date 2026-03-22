import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Fused residual mix + RMSNorm.

    Each transformer block starts with:
      x = mix[0] * x + mix[1] * x0   (weighted residual blend)
      n = rms_norm(x)                  (normalization)

    This is non-standard architecture — torch.compile emits multiple
    small kernels. Fusing loads x, x0, mix once, computes blend,
    normalizes, writes result once. Called 11x per forward, 11x backward.
    """

    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim
        self.resid_mix = nn.Parameter(torch.stack([
            0.7 * torch.ones(dim),
            0.3 * torch.ones(dim)
        ]).to(torch.bfloat16))

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim] current residual stream (bfloat16)
            x0: [batch, seq_len, dim] initial embeddings (bfloat16)
        Returns:
            n: [batch, seq_len, dim] blended + normalized (bfloat16)
        """
        mix = self.resid_mix.to(dtype=x.dtype)
        blended = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        return F.rms_norm(blended, (self.dim,))


BATCH = 8
SEQ_LEN = 2048
DIM = 512


def get_inputs():
    x = torch.randn(BATCH, SEQ_LEN, DIM, dtype=torch.bfloat16)
    x0 = torch.randn(BATCH, SEQ_LEN, DIM, dtype=torch.bfloat16)
    return [x, x0]


def get_init_inputs():
    return [DIM]
