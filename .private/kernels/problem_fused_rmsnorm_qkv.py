import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Fused RMSNorm + Q/K/V linear projections for GQA attention.

    In each transformer block, we compute:
      n = rms_norm(x)
      q = n @ W_q^T    (dim -> dim, 8 heads)
      k = n @ W_k^T    (dim -> kv_dim, 4 KV heads)
      v = n @ W_v^T    (dim -> kv_dim, 4 KV heads)

    The normalized tensor 'n' is only used for these three projections,
    so fusing avoids writing it back to HBM. At dim=512 with GQA (8 heads,
    4 KV heads), these are small matmuls that are heavily memory-bound.
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super(Model, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.w_q = nn.Parameter(torch.randn(dim, dim, dtype=torch.bfloat16))
        self.w_k = nn.Parameter(torch.randn(kv_dim, dim, dtype=torch.bfloat16))
        self.w_v = nn.Parameter(torch.randn(kv_dim, dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim] input hidden states (bfloat16)

        Returns:
            qkv: [batch, seq_len, dim + 2*kv_dim] concatenated Q, K, V
        """
        n = F.rms_norm(x, (x.size(-1),))
        q = F.linear(n, self.w_q)
        k = F.linear(n, self.w_k)
        v = F.linear(n, self.w_v)
        return torch.cat([q, k, v], dim=-1)


# Dimensions matching parameter-golf 10-layer model
BATCH = 64  # TTT uses batch=64, training uses variable
SEQ_LEN = 1024
DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4


def get_inputs():
    x = torch.randn(BATCH, SEQ_LEN, DIM, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [DIM, NUM_HEADS, NUM_KV_HEADS]
