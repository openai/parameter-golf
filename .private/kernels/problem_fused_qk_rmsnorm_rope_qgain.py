import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Fused Q/K RMSNorm + RoPE + q_gain scaling.

    In each attention layer, after projecting Q and K, we:
    1. Reshape to [batch, heads, seq, head_dim]
    2. RMSNorm each head independently
    3. Apply Rotary Position Embeddings
    4. Scale Q by per-head q_gain

    This is 5-6 kernel launches fused into 1. Called 11x per forward,
    11x per backward. Public benchmarks show fused RoPE alone at 5.68x.
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, seq_len: int):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_gain = nn.Parameter(torch.ones(num_heads, dtype=torch.float32) * 1.5)
        # Precompute RoPE cos/sin
        inv_freq = 1.0 / (50000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos', freqs.cos()[None, None, :, :].to(torch.bfloat16))
        self.register_buffer('sin', freqs.sin()[None, None, :, :].to(torch.bfloat16))

    def forward(self, q_proj: torch.Tensor, k_proj: torch.Tensor) -> tuple:
        """
        Args:
            q_proj: [batch, seq, dim] raw Q projection output (bfloat16)
            k_proj: [batch, seq, kv_dim] raw K projection output (bfloat16)
        Returns:
            q: [batch, heads, seq, head_dim] normalized + rotated + scaled
            k: [batch, kv_heads, seq, head_dim] normalized + rotated
        """
        bsz, seqlen = q_proj.shape[0], q_proj.shape[1]
        q = q_proj.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_proj.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # RMSNorm per head
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # RoPE
        cos = self.cos[:, :, :seqlen, :]
        sin = self.sin[:, :, :seqlen, :]
        half = q.size(-1) // 2
        q1, q2 = q[..., :half], q[..., half:]
        q = torch.cat((q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos), dim=-1)
        k1, k2 = k[..., :half], k[..., half:]
        k = torch.cat((k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos), dim=-1)
        # q_gain
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        return torch.cat([q.reshape(bsz, -1), k.reshape(bsz, -1)], dim=-1)


BATCH = 8
SEQ_LEN = 2048
DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4


def get_inputs():
    q_proj = torch.randn(BATCH, SEQ_LEN, DIM, dtype=torch.bfloat16)
    k_proj = torch.randn(BATCH, SEQ_LEN, NUM_KV_HEADS * (DIM // NUM_HEADS), dtype=torch.bfloat16)
    return [q_proj, k_proj]


def get_init_inputs():
    return [DIM, NUM_HEADS, NUM_KV_HEADS, SEQ_LEN]
