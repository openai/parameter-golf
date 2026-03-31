import math
import torch

def flash_attn_func(q, k, v, causal=True):
    # Expected shapes:
    # q: [B, T, H, D]
    # k: [B, T, Hkv, D]
    # v: [B, T, Hkv, D]
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"Unexpected shapes: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")

    bsz, seqlen, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]

    if q_heads != kv_heads:
        if q_heads % kv_heads != 0:
            raise ValueError(f"q_heads={q_heads} must be divisible by kv_heads={kv_heads}")
        repeat = q_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

    # Manual causal attention for maximum compatibility on Colab GPUs.
    # Move heads forward: [B, H, T, D]
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()

    # Do attention math in fp32 for stability, then cast back.
    qf = q.float()
    kf = k.float()
    vf = v.float()

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale

    if causal:
        mask = torch.ones((seqlen, seqlen), device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    y = torch.matmul(probs, vf).to(q.dtype)

    return y.permute(0, 2, 1, 3).contiguous()
