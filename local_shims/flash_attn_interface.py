"""Drop-in shim for flash_attn_interface on non-Hopper GPUs.

Wraps torch.nn.functional.scaled_dot_product_attention to match the
flash_attn_3_func(q, k, v, causal=True) signature used by training scripts.

Handles GQA (different num_heads for q vs k/v) by repeating k/v.

Add this directory to PYTHONPATH for local DGX Spark runs:
  export PYTHONPATH=/path/to/local_shims:$PYTHONPATH
"""

import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v, causal=False):
    """Match flash_attn_3 signature: (B, S, H, D) -> (B, S, H, D).

    Handles GQA: if q has more heads than k/v, repeats k/v to match.
    """
    bsz, seqlen, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]

    # GQA expansion: repeat k/v heads to match q heads
    if q_heads != kv_heads:
        repeats = q_heads // kv_heads
        k = k.unsqueeze(3).expand(bsz, seqlen, kv_heads, repeats, head_dim).reshape(bsz, seqlen, q_heads, head_dim)
        v = v.unsqueeze(3).expand(bsz, seqlen, kv_heads, repeats, head_dim).reshape(bsz, seqlen, q_heads, head_dim)

    # flash_attn: (B, S, H, D) -> SDPA: (B, H, S, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    return out.transpose(1, 2)
