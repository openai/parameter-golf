from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .config import COMPUTE_DTYPE


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    # Functional RMSNorm is kept separate because several call sites want the
    # same normalization logic without carrying a learned scale parameter. This
    # baseline leans on "small explicit helpers" instead of a deeper module
    # hierarchy. If the architecture keeps growing, one likely improvement would
    # be to consolidate these numerics into a more conventional layers module.
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Keep master weights in fp32 for update stability, then cast them to the
        # incoming activation dtype at matmul time. On MLX this is a pragmatic
        # compromise between numeric robustness and activation-memory pressure.
        # A future tuning pass could revisit which matrices really need fp32
        # storage versus bfloat16 storage plus selective fp32 accumulators.
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    # - separate q/k/v projections
    # - RMSNorm on q and k before attention
    # - RoPE on q and k
    # - causal masked SDPA
    #
    # This is intentionally a compact, custom attention block rather than an
    # attempt to mirror every abstraction from a larger transformer framework.
    # The code is optimized for readability and low ceremony, but that also
    # means several design choices are encoded directly in this class instead of
    # being pluggable policies.
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        # RoPE rotates pairs of features, so each head dimension must be even.
        # The baseline treats this as a hard invariant instead of padding around
        # it because the rest of the stack assumes standard transformer-shaped
        # attention tensors.
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        # Queries use all attention heads, while keys/values may use fewer heads
        # (`num_kv_heads`) and are shared across groups of query heads. That
        # grouped-KV layout reduces memory/bandwidth pressure compared to full
        # multi-head K/V without changing the external block contract.
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # q/k RMSNorm before attention is one of the more opinionated choices in
        # this file. It keeps the attention statistics well-behaved and works
        # nicely with the learned `q_gain`, but it is also a place where future
        # experiments could cleanly branch into alternate normalization schemes.
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        # These learned scales and mixes are the "control tensors" that get
        # special treatment elsewhere. They are cheap, expressive knobs for
        # residual behavior, but because they are small and numerically sensitive
        # they are also handled differently by optimization and quantization.
        # A more formal architecture definition could expose them as named layer
        # features rather than ad-hoc tensors on each block.
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        # `resid_mix` lets each block interpolate between the running hidden
        # state and the original post-embedding state. It is a deliberately
        # simple mechanism for preserving access to the input representation
        # without introducing a larger routing subsystem.
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    # - token embedding + RMSNorm
    # - encoder half accumulates skip tensors
    # - decoder half consumes reversed skips with learned skip_weights
    # - tied embeddings for the LM head (the baseline default setup)
    #
    # The "encoder/decoder" naming here is structural, not about seq2seq
    # semantics. The first half of the stack stores skip activations and the
    # second half consumes them in reverse order. This gives the model a simple
    # U-Net-like shape while keeping the code close to a plain transformer block
    # stack. If this architecture family keeps evolving, one useful cleanup
    # would be to split the baseline transformer pieces from these additional
    # skip/mix experiments more explicitly.
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        tied_embed_init_std: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        # Zero-initializing the projection weights makes each residual branch
        # start life close to an identity map, which is a common stabilization
        # trick for deep residual stacks. It is not the only reasonable choice,
        # but it matches the "safe baseline first" philosophy of this script.
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        # The token embedding doubles as the LM head via weight tying. We seed it
        # with a narrow normal distribution and keep the actual logits path in
        # `loss()` so the model's forward pass remains "hidden states only".
        # A more feature-rich training stack might expose logits as a separate
        # method, but the current shape keeps the common training path concise.
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []

        # The first half of the network records intermediate activations for
        # later reuse. We store the post-block states directly because that is
        # the simplest thing compatible with the decoder-side additive skips.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            # Odd layer counts have one more decoder block than encoder block. The baseline only
            # applies a skip connection when one exists, then runs the remaining decoder block(s)
            # without an added skip.
            #
            # Possible improvement: make this asymmetry explicit in a higher-level
            # stack builder rather than letting the decoder loop silently handle
            # the "one extra decoder block" case.
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        # Cross-entropy over flattened tokens. We keep optional logit chunking because it is a useful
        # memory knob on Macs, but the common path is chunk_tokens=0 (single matmul + CE).
        #
        # Important: chunking the logits path is only a memory/performance trade.
        # It does not change the objective, only how much of the `[tokens, vocab]`
        # projection is materialized at once. This is another place where the
        # code optimizes for a broad range of Mac hardware instead of assuming a
        # large dedicated GPU memory budget.
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)
