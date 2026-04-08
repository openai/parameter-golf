from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


COMPUTE_DTYPE = mx.bfloat16


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_base: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.q_proj = CastedLinear(dim, dim)
        self.k_proj = CastedLinear(dim, dim)
        self.v_proj = CastedLinear(dim, dim)
        self.out_proj = CastedLinear(dim, dim)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.q_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, rope_base: float):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = BidirectionalSelfAttention(dim, num_heads, rope_base)
        self.mlp = MLP(dim, mlp_mult)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        mlp_mult: int,
        num_diffusion_steps: int,
        rope_base: float,
        tied_embed_init_std: float,
        logit_softcap: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError("LOGIT_SOFTCAP must be positive")
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.time_emb = nn.Embedding(num_diffusion_steps + 1, dim)
        self.blocks = [Block(dim, num_heads, mlp_mult, rope_base) for _ in range(num_layers)]
        self.final_norm = RMSNormNoWeight()
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)
        self.time_emb.weight = (
            mx.random.normal(self.time_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def hidden(self, input_ids: mx.array, timesteps: mx.array) -> mx.array:
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        t = self.time_emb(timesteps).astype(COMPUTE_DTYPE)[:, None, :]
        x = rms_norm(x + t)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def logits(self, input_ids: mx.array, timesteps: mx.array) -> mx.array:
        h = self.hidden(input_ids, timesteps)
        logits = h @ self.tok_emb.weight.astype(h.dtype).T
        return self.softcap(logits)

    def loss(
        self,
        corrupted_ids: mx.array,
        target_ids: mx.array,
        timesteps: mx.array,
        loss_mask: mx.array,
    ) -> mx.array:
        logits = self.logits(corrupted_ids, timesteps).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, target_ids, reduction="none").astype(mx.float32)
        weights = loss_mask.astype(mx.float32)
        return mx.sum(losses * weights) / mx.maximum(mx.sum(weights), mx.array(1.0, dtype=mx.float32))
