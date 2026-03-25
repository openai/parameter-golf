"""Diamond Fusion Crystal Language Model.

Runnable MLX reference implementation for the Diamond Fusion / Crystal
competition architecture.
"""

import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn

COMPUTE_DTYPE = mx.bfloat16


def rms_norm(x, eps=1e-6):
    """RMS normalize without learned scale."""
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


class CastedLinear(nn.Module):
    """Linear layer with float32 weights cast on forward."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        scale = 1.0 / math.sqrt(in_dim)
        self.weight = mx.random.normal((out_dim, in_dim), dtype=mx.float32) * scale
        self.bias = mx.zeros((out_dim,), dtype=mx.float32)

    def __call__(self, x):
        w = self.weight.astype(x.dtype)
        b = self.bias.astype(x.dtype)
        return mx.matmul(x, w.T) + b


class PartialRoPE:
    """Apply rotary position embeddings to the first rope_dims only."""

    def __init__(self, head_dim, rope_dims=16, base=10000):
        self.head_dim = head_dim
        self.rope_dims = min(rope_dims, head_dim)
        self.base = base
        half = self.rope_dims // 2
        inv_freq = 1.0 / (base ** (mx.arange(0, half, dtype=mx.float32) / half))
        self.inv_freq = inv_freq

    def __call__(self, x):
        if self.rope_dims < 2:
            return x
        t = mx.arange(x.shape[2], dtype=mx.float32)
        freqs = mx.einsum("t,d->td", t, self.inv_freq)
        sin = mx.sin(freqs)[None, None, :, :]
        cos = mx.cos(freqs)[None, None, :, :]
        rot = x[..., : self.rope_dims]
        rest = x[..., self.rope_dims :]
        rot1, rot2 = mx.split(rot, 2, axis=-1)
        rot = mx.concatenate([rot1 * cos - rot2 * sin, rot1 * sin + rot2 * cos], axis=-1)
        return mx.concatenate([rot, rest], axis=-1)


class CrystalAttention(nn.Module):
    """Grouped query attention with partial RoPE and QK normalization."""

    def __init__(self, dim=512, num_heads=8, num_kv_heads=4, rope_dims=16, rope_base=10000, qk_gain=1.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_proj = CastedLinear(dim, num_heads * self.head_dim)
        self.k_proj = CastedLinear(dim, num_kv_heads * self.head_dim)
        self.v_proj = CastedLinear(dim, num_kv_heads * self.head_dim)
        self.o_proj = CastedLinear(num_heads * self.head_dim, dim)
        self.rope = PartialRoPE(self.head_dim, rope_dims=rope_dims, base=rope_base)
        self.qk_gain = mx.array(float(qk_gain), dtype=mx.float32)

    def __call__(self, x, mask="causal"):
        if x.ndim == 4:
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q)
        k = self.rope(k)
        q = q * self.qk_gain
        k = k * self.qk_gain
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(b, t, self.dim)
        return self.o_proj(y)


class CrystalMLP(nn.Module):
    """Three-times expanded MLP with squared leaky activation."""

    def __init__(self, dim=512, mult=3):
        super().__init__()
        hidden = dim * mult
        self.fc1 = CastedLinear(dim, hidden)
        self.fc2 = CastedLinear(hidden, dim)

    def __call__(self, x):
        h = self.fc1(x)
        h = mx.where(h > 0, h, 0.5 * h)
        return self.fc2(h * h)


class CrystalBlock(nn.Module):
    """Shared recurrent transformer block."""

    def __init__(self, dim=512, num_heads=8, num_kv_heads=4, rope_dims=16, rope_base=10000, qk_gain=1.5):
        super().__init__()
        self.attn = CrystalAttention(dim, num_heads, num_kv_heads, rope_dims, rope_base, qk_gain)
        self.mlp = CrystalMLP(dim, mult=3)
        self.attn_scale = mx.array(1.0, dtype=mx.float32)
        self.mlp_scale = mx.array(1.0, dtype=mx.float32)
        self.resid_mix = mx.array(0.5, dtype=mx.float32)

    def __call__(self, x, x0, attn_gate, mlp_gate):
        h = rms_norm(x).astype(COMPUTE_DTYPE)
        attn_out = self.attn(h)
        x = x + attn_out * (attn_gate * self.attn_scale)
        h = rms_norm(x).astype(COMPUTE_DTYPE)
        mlp_out = self.mlp(h)
        x = x + mlp_out * (mlp_gate * self.mlp_scale)
        alpha = self.resid_mix
        return alpha * x + (1.0 - alpha) * x0


class DiamondFusion(nn.Module):
    """Fuse multiple expert states with consensus bonus."""

    def __init__(self, num_experts=8, dim=512):
        super().__init__()
        self.num_experts = num_experts
        self.score = CastedLinear(dim, 1)
        self.beta = mx.array(0.1, dtype=mx.float32)

    def __call__(self, states):
        # states: list of (B, T, D) tensors
        stacked = mx.stack(states, axis=0)          # (E, B, T, D)
        mean = mx.mean(stacked, axis=0)              # (B, T, D)
        var = mx.mean(mx.square(stacked - mean[None]), axis=0)  # (B, T, D)
        # Per-expert scoring: mean-pool over T, project to scalar
        logits = mx.stack([self.score(mx.mean(s, axis=1)) for s in states], axis=-1)  # (B, 1, E)
        logits = logits.squeeze(1)                   # (B, E)
        weights = mx.softmax(logits, axis=-1)        # (B, E)
        fused = sum(weights[:, i, None, None] * states[i] for i in range(len(states)))  # (B, T, D)
        agreement = mx.exp(-var * 10.0)              # (B, T, D) — high where experts agree
        return fused + self.beta * agreement * mean


class CrystalGPT(nn.Module):
    """Single-expert recurrent Crystal GPT with U-Net skips."""

    def __init__(self, vocab_size=1024, crystal_iters=12, dim=512, num_heads=8, num_kv_heads=4, rope_dims=16, rope_base=10000, qk_gain=1.5, softcap=30.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.crystal_iters = crystal_iters
        self.dim = dim
        self.softcap = softcap
        self.embed = nn.Embedding(vocab_size, dim)
        self.in_norm = rms_norm
        self.out_norm = rms_norm
        self.block = CrystalBlock(dim, num_heads, num_kv_heads, rope_dims, rope_base, qk_gain)
        self.attn_gates = [mx.array(1.0 / math.sqrt(crystal_iters), dtype=mx.float32) for _ in range(crystal_iters)]
        self.mlp_gates = [mx.array(1.0 / math.sqrt(crystal_iters), dtype=mx.float32) for _ in range(crystal_iters)]
        self.lm_head = self.embed.weight

    def __call__(self, tokens):
        x = self.embed(tokens)
        x = x[0] if x.ndim == 4 else x
        x = self.in_norm(x)
        skips = []
        half = self.crystal_iters // 2
        for i in range(half):
            x = self.block(x, x, self.attn_gates[i], self.mlp_gates[i])
            skips.append(x)
        for i in range(half, self.crystal_iters):
            skip = skips.pop()
            x = self.block(x + skip, x, self.attn_gates[i], self.mlp_gates[i])
        x = self.out_norm(x)
        logits = mx.matmul(x, self.lm_head.T)
        return mx.clip(logits, -self.softcap, self.softcap)


class CrystalMoE(nn.Module):
    """Multi-expert crystal model with periodic diamond fusion."""

    def __init__(self, vocab_size=1024, crystal_iters=12, dim=512, num_experts=8, fusion_every=3, num_heads=8, num_kv_heads=4, rope_dims=16, rope_base=10000, qk_gain=1.5, softcap=30.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.crystal_iters = crystal_iters
        self.dim = dim
        self.num_experts = num_experts
        self.fusion_every = fusion_every
        self.softcap = softcap
        self.embed = nn.Embedding(vocab_size, dim)
        self.block = CrystalBlock(dim, num_heads, num_kv_heads, rope_dims, rope_base, qk_gain)  # SHARED
        self.fusion = DiamondFusion(num_experts, dim)
        self.attn_gates = [[mx.array(1.0 / math.sqrt(crystal_iters), dtype=mx.float32) for _ in range(crystal_iters)] for _ in range(num_experts)]
        self.mlp_gates = [[mx.array(1.0 / math.sqrt(crystal_iters), dtype=mx.float32) for _ in range(crystal_iters)] for _ in range(num_experts)]
        self.lm_head = self.embed.weight

    def __call__(self, tokens):
        x = rms_norm(self.embed(tokens))
        x = x[0] if x.ndim == 4 else x
        states = [x for _ in range(self.num_experts)]
        for i in range(self.crystal_iters):
            states = [self.block(states[e], x, self.attn_gates[e][i], self.mlp_gates[e][i]) for e in range(self.num_experts)]
            x = states[0]
            if (i + 1) % self.fusion_every == 0:
                x = self.fusion(states)
                states = [x for _ in range(self.num_experts)]
        x = rms_norm(x)
        logits = mx.matmul(x, self.lm_head.T)
        return mx.clip(logits, -self.softcap, self.softcap)

def tree_items(d, prefix=""):
    """Yield a flattened parameter tree listing."""
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            items.extend(tree_items(v, f"{prefix}{k}/"))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            items.extend(tree_items(v, f"{prefix}{i}/"))
    else:
        items.append((prefix[:-1], d))
    return items


def count_params(model):
    total = 0
    for _, v in tree_items(model.parameters()) if hasattr(model, "parameters") else []:
        if hasattr(v, "shape"):
            total += int(np.prod(v.shape))
    return total


if __name__ == "__main__":
    mx.random.seed(7)
    gpt = CrystalGPT()
    moe = CrystalMoE()
    tokens = mx.array(np.random.randint(0, 1024, size=(2, 16)), dtype=mx.int32)
    y1 = gpt(tokens)
    y2 = moe(tokens)
    print("CrystalGPT logits:", y1.shape, y1.dtype)
    print("CrystalMoE logits:", y2.shape, y2.dtype)
    print("CrystalGPT params:", count_params(gpt))
    print("CrystalMoE params:", count_params(moe))
