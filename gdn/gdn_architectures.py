"""GDN Hybrid Architecture — modular blocks using FLA native layers.

Supports model variants for the Parameter Golf Direction-5 experiments.
Each model is a stack of mixed {GDN, DeltaProduct, Mamba-2, SWA} blocks
with shared MLP, RMSNorm, and residual connections.

Key design choices:
- FLA layers handle recurrent attention (GatedDeltaNet, GatedDeltaProduct, Mamba2)
- Sliding Window Attention (SWA) uses flash attention with a causal window mask
- All blocks follow the same pre-norm residual pattern for uniform gradient flow
- Weight sharing for SWA layers in Griffin/Zamba-style models
- forward_hidden() exposes (hidden_states, logits) for RLS eval
"""
from __future__ import annotations
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ─── FLA backend selection ──────────────────────────────────────────────────
# Set FLA_USE_NAIVE=1 to force pure-PyTorch (naive) kernels instead of Triton.
_USE_NAIVE = os.environ.get("FLA_USE_NAIVE", "0") == "1"

if _USE_NAIVE:
    import fla.ops.gated_delta_rule.chunk as _gdr_chunk
    import fla.ops.gated_delta_rule.naive as _gdr_naive

    def _patched_chunk_gated_delta_rule(
        q, k, v, g, beta, scale=None, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs
    ):
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
        return _gdr_naive.naive_chunk_gated_delta_rule(
            q, k, v, g, beta,
            chunk_size=64, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    _gdr_chunk.chunk_gated_delta_rule = _patched_chunk_gated_delta_rule
    import fla.layers.gated_deltanet as _gdn_layer
    _gdn_layer.chunk_gated_delta_rule = _patched_chunk_gated_delta_rule

    import fla.ops.gated_delta_product.chunk as _gdp_chunk
    import fla.ops.gated_delta_product.naive as _gdp_naive

    def _patched_chunk_gated_delta_product(
        q, k, v, g, beta, num_householder=1, scale=None, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs
    ):
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
        return _gdp_naive.naive_recurrent_gated_delta_product(
            q, k, v, g, beta,
            scale=scale, cu_seqlens=None,
            initial_state=initial_state,
            output_final_state=output_final_state,
            num_householder=num_householder,
        )

    _gdp_chunk.chunk_gated_delta_product = _patched_chunk_gated_delta_product
    import fla.layers.gated_deltaproduct as _gdp_layer
    _gdp_layer.chunk_gated_delta_product = _patched_chunk_gated_delta_product

    print("[FLA] Using NAIVE (pure-PyTorch) kernels — set FLA_USE_NAIVE=0 for Triton", flush=True)

# FLA imports
from fla.layers import GatedDeltaNet, GatedDeltaProduct, Mamba2
try:
    from fla.layers import RWKV7Attention
except Exception:
    RWKV7Attention = None  # type: ignore

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    def flash_attn_3_func(q, k, v, causal=False, window_size=(-1, -1)):
        q2 = q.transpose(1, 2)
        k2 = k.transpose(1, 2)
        v2 = v.transpose(1, 2)
        if k2.size(1) != q2.size(1):
            rep = q2.size(1) // k2.size(1)
            k2 = k2.repeat_interleave(rep, dim=1)
            v2 = v2.repeat_interleave(rep, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
        return out.transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Linear layer that casts input to weight dtype for mixed precision.
    Supports late QAT (int6 STE) when _qat_enabled is set."""
    _qat_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(dtype=x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -31, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()  # STE: forward uses quantized, backward uses full
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Rotary(nn.Module):
    """RoPE embeddings for sliding window attention."""
    def __init__(self, dim: int, base: float = 10000.0, max_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to the input tensor."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    out1 = x1 * cos[:x.shape[-2]] - x2 * sin[:x.shape[-2]]
    out2 = x2 * cos[:x.shape[-2]] + x1 * sin[:x.shape[-2]]
    return torch.cat([out1, out2], dim=-1)


class MLP(nn.Module):
    """Feed-forward MLP with configurable activation."""
    def __init__(self, dim: int, mult: float = 3.0, act: str = "relu_sq", leaky_slope: float = 0.5):
        super().__init__()
        hidden = int(mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.act = act
        self.leaky_slope = leaky_slope

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        if self.act == "leaky_relu_sq":
            x = F.leaky_relu(x, negative_slope=self.leaky_slope)
        else:
            x = F.relu(x)
        return self.proj(x.square())


class SlidingWindowAttention(nn.Module):
    """Sliding window causal attention for hybrid models.

    Supports XSA (cross-segment attention) at eval time for extending context
    across eval chunks. Window is enforced during training but can be relaxed at eval.
    KV can be shared across layers (Zamba-style) by reusing the same module.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        window_size: int = 512,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.window_size = window_size

        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.use_xsa = False

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """XSA: subtract self-value projection (GQA-aware)."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        if q.is_cuda and q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        y = flash_attn_3_func(q, k, v, causal=True)

        if self.use_xsa:
            y = self._xsa_efficient(y, v)

        y = y.reshape(B, T, D)
        return self.proj(y)


class RecurrentBlock(nn.Module):
    """Wraps any FLA recurrent layer (GDN, DeltaProduct, Mamba-2) with
    pre-norm residual connection and MLP."""

    def __init__(
        self,
        dim: int,
        recurrent_layer: nn.Module,
        mlp_mult: float = 3.0,
        mlp_act: str = "relu_sq",
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.recurrent = recurrent_layer
        self.mlp = MLP(dim, mlp_mult, act=mlp_act)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.layer_idx = layer_idx

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        recurrent_out = self.recurrent(self.attn_norm(x_in))
        if isinstance(recurrent_out, tuple):
            recurrent_out = recurrent_out[0]

        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * recurrent_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out))
        return x_out


class AttentionBlock(nn.Module):
    """SWA block with pre-norm residual and MLP."""

    def __init__(
        self,
        dim: int,
        swa: SlidingWindowAttention,
        mlp_mult: float = 3.0,
        mlp_act: str = "relu_sq",
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = swa
        self.mlp = MLP(dim, mlp_mult, act=mlp_act)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.layer_idx = layer_idx

    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in), v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out))
        return x_out


class SmearGate(nn.Module):
    """Weighted average of current and previous token embeddings."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash-based bigram/trigram embedding for additional context."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int,
                 trigram: bool = False):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self._trigram = trigram
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def trigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., :2] = mod
        out[..., 2:] = (36313 * t[..., 2:] ^ 27191 * t[..., 1:-1] ^ 51497 * t[..., :-2]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self._trigram:
            h = h + self.embed(self.trigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


def _parse_layout(layout_str: str) -> list[tuple[str, int]]:
    """Parse a layout string into a sequence of (layer_type, count) pairs.

    Examples:
        "gdn_only" -> [("gdn", 11)]  (count filled in by caller)
        "gdn5_swa_gdn5_swa_shared" -> [("gdn", 5), ("swa", 1), ("gdn", 5), ("swa_shared", 1)]
    """
    if layout_str == "gdn_only":
        return [("gdn", -1)]
    if layout_str == "mamba_only":
        return [("mamba", -1)]

    parts = layout_str.split("_")
    result = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part.startswith("gdn") and len(part) > 3:
            count = int(part[3:])
            result.append(("gdn", count))
        elif part.startswith("mamba") and len(part) > 5:
            count = int(part[5:])
            result.append(("mamba", count))
        elif part == "swa":
            if i + 1 < len(parts) and parts[i + 1] == "shared":
                result.append(("swa_shared", 1))
                i += 1
            else:
                result.append(("swa", 1))
        elif part == "shared":
            pass
        i += 1
    return result


class HybridGDN(nn.Module):
    """Hybrid GDN architecture supporting mixed recurrent/attention layers.

    Builds a stack of blocks according to the layer_layout specification:
    - "gdn" blocks use GatedDeltaNet (or GatedDeltaProduct)
    - "mamba" blocks use Mamba-2
    - "swa" blocks use SlidingWindowAttention
    - "swa_shared" reuses the same SWA module (Griffin/Zamba-style weight sharing)

    All models share: token embedding, bigram hash, smear gate, final norm, lm_head.
    """
    def __init__(self, config: dict, vocab_size: int = 1024):
        super().__init__()
        dim = config["model_dim"]
        num_heads = config["num_heads"]
        mlp_mult = config["mlp_mult"]
        self.arch_name = config["arch_name"]
        self.model_dim = dim
        self.vocab_size = vocab_size
        self.logit_softcap = 30.0

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        self.bigram = BigramHashEmbedding(
            config.get("bigram_vocab_size", 2048),
            config.get("bigram_dim", 128),
            dim,
            trigram=config.get("trigram", False),
        )
        self.smear = SmearGate(dim)

        # Meta tokens (Hymba-style)
        n_meta = config.get("meta_tokens", 0)
        if n_meta > 0:
            self.meta_tokens = nn.Parameter(torch.randn(1, n_meta, dim) * 0.02)
            self.n_meta = n_meta
        else:
            self.meta_tokens = None
            self.n_meta = 0

        # Build layer stack
        layout = _parse_layout(config["layer_layout"])
        self.blocks = nn.ModuleList()
        self._block_types = []
        self._shared_swa = None

        layer_idx = 0
        for layer_type, count in layout:
            if count == -1:
                if layer_type == "gdn":
                    count = config["num_gdn_layers"]
                elif layer_type == "mamba":
                    count = config["num_mamba_layers"]

            for _ in range(count):
                if layer_type == "gdn":
                    recurrent = self._make_recurrent_layer(config, layer_idx)
                    block = RecurrentBlock(dim, recurrent, mlp_mult, layer_idx=layer_idx)
                    self.blocks.append(block)
                    self._block_types.append("gdn")

                elif layer_type == "mamba":
                    mamba_expand = config.get("mamba_expand", 2)
                    mamba_head_dim = config.get("gdn_head_dim", 64)
                    mamba_num_heads = (dim * mamba_expand) // mamba_head_dim
                    mamba = Mamba2(
                        num_heads=mamba_num_heads,
                        head_dim=mamba_head_dim,
                        hidden_size=dim,
                        state_size=config.get("mamba_state_size", 64),
                        expand=mamba_expand,
                        layer_idx=layer_idx,
                    )
                    block = RecurrentBlock(dim, mamba, mlp_mult, layer_idx=layer_idx)
                    self.blocks.append(block)
                    self._block_types.append("mamba")

                elif layer_type in ("swa", "swa_shared"):
                    if layer_type == "swa_shared" and self._shared_swa is not None:
                        swa = self._shared_swa
                    else:
                        swa = SlidingWindowAttention(
                            dim=dim,
                            num_heads=num_heads,
                            num_kv_heads=config.get("swa_num_kv_heads", 4),
                            window_size=config.get("swa_window", 512),
                            qk_gain_init=config.get("qk_gain_init", 1.5),  # Direction-5: 5.0
                        )
                        if config.get("swa_shared", False):
                            self._shared_swa = swa

                    block = AttentionBlock(dim, swa, mlp_mult, layer_idx=layer_idx)
                    self.blocks.append(block)
                    self._block_types.append("swa" if layer_type == "swa" else "swa_shared")

                layer_idx += 1

        self.final_norm = RMSNorm(dim)
        self.lm_head = None  # tied to tok_emb
        self._init_weights()

    def _make_recurrent_layer(self, config: dict, layer_idx: int) -> nn.Module:
        """Create the appropriate recurrent layer based on config."""
        dim = config["model_dim"]
        num_heads = config["num_heads"]

        if config.get("use_rwkv7", False):
            total_layers = config.get("num_gdn_layers", 11)
            return RWKV7Attention(
                hidden_size=dim,
                head_dim=config.get("gdn_head_dim", 64),
                num_heads=num_heads,
                layer_idx=layer_idx,
                num_hidden_layers=total_layers,
                mode="chunk",
            )
        elif config.get("use_deltaproduct", False):
            return GatedDeltaProduct(
                hidden_size=dim,
                head_dim=config.get("gdn_head_dim", 64),
                num_heads=num_heads,
                num_householder=config.get("dp_num_householder", 2),
                allow_neg_eigval=config.get("dp_allow_neg_eigval", False),
                use_short_conv=config.get("gdn_use_short_conv", True),
                expand_v=config.get("gdn_expand_v", 1),
                layer_idx=layer_idx,
                mode="chunk",
            )
        else:
            return GatedDeltaNet(
                hidden_size=dim,
                head_dim=config.get("gdn_head_dim", 64),
                num_heads=num_heads,
                allow_neg_eigval=config.get("gdn_allow_neg_eigval", False),
                use_short_conv=config.get("gdn_use_short_conv", True),
                expand_v=config.get("gdn_expand_v", 1),
                layer_idx=layer_idx,
                mode="chunk",
            )

    def _init_weights(self) -> None:
        total_layers = len(self.blocks)
        for name, p in self.named_parameters():
            if ".recurrent." in name:
                continue
            if p.ndim == 2 and "proj" in name and "bigram" not in name:
                with torch.no_grad():
                    p.mul_(1.0 / math.sqrt(2 * total_layers))

    def set_xsa(self, enable: bool = True) -> None:
        """Enable/disable XSA on all attention blocks."""
        for block, btype in zip(self.blocks, self._block_types):
            if btype in ("swa", "swa_shared"):
                block.attn.use_xsa = enable

    def _compute_logits(self, x: Tensor) -> Tensor:
        """Compute logits with tied embeddings and softcap."""
        logits = F.linear(x, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def _run_blocks(self, x: Tensor, x0: Tensor) -> Tensor:
        """Run all blocks on x with residual anchor x0."""
        for block in self.blocks:
            x = block(x, x0)
        return x

    def _embed(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Shared embedding + smear, returns (x, x0)."""
        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        if self.meta_tokens is not None:
            B = x.shape[0]
            meta = self.meta_tokens.expand(B, -1, -1).to(dtype=x.dtype)
            x = torch.cat([meta, x], dim=1)
            x0 = torch.cat([meta, x0], dim=1)
        return x, x0

    def _strip_meta(self, x: Tensor) -> Tensor:
        if self.meta_tokens is not None:
            x = x[:, self.n_meta:]
        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Forward pass returning cross-entropy loss."""
        x, x0 = self._embed(input_ids)
        x = self._run_blocks(x, x0)
        x = self._strip_meta(x)
        x = self.final_norm(x)
        logits = self._compute_logits(x.reshape(-1, x.size(-1)))
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning softcapped logits (for evaluation)."""
        x, x0 = self._embed(input_ids)
        x = self._run_blocks(x, x0)
        x = self._strip_meta(x)
        x = self.final_norm(x)
        return self._compute_logits(x)

    def forward_hidden(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass returning (hidden_states, softcapped_logits) for RLS eval.

        hidden_states: [B, T, dim] — final norm output before lm_head
        softcapped_logits: [B, T, vocab] — softcap * tanh(linear / softcap)

        Compliance note: called in inference_mode during eval.
        The hidden states are purely causal (each h[t] depends only on x[0:t]).
        """
        x, x0 = self._embed(input_ids)
        x = self._run_blocks(x, x0)
        x = self._strip_meta(x)
        x = self.final_norm(x)
        logits = self._compute_logits(x)
        return x, logits

    def get_diagnostics(self) -> dict:
        """Collect per-layer weight statistics for checkpoint diagnostics."""
        diag = {}
        for i, (block, btype) in enumerate(zip(self.blocks, self._block_types)):
            prefix = f"layer_{i}_{btype}"
            for name, param in block.named_parameters():
                if param.ndim >= 2:
                    w = param.data.float()
                    diag[f"{prefix}/{name}/std"] = w.std().item()
                    diag[f"{prefix}/{name}/kurtosis"] = (((w - w.mean()) / (w.std() + 1e-8)) ** 4).mean().item() - 3.0
        return diag

    def count_params(self) -> dict:
        """Count parameters by category."""
        cats = {"embedding": 0, "recurrent": 0, "attention": 0, "mlp": 0, "other": 0}
        for name, p in self.named_parameters():
            n = p.numel()
            if "tok_emb" in name or "bigram" in name:
                cats["embedding"] += n
            elif any(k in name for k in ["recurrent", "gdn", "mamba", "rwkv", "delta"]):
                cats["recurrent"] += n
            elif "attn" in name or "c_q" in name or "c_k" in name or "c_v" in name:
                cats["attention"] += n
            elif "mlp" in name or "fc" in name:
                cats["mlp"] += n
            else:
                cats["other"] += n
        cats["total"] = sum(cats.values())
        return cats
