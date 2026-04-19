"""GDN Hybrid Architecture — modular blocks using FLA native layers.

Supports 8 model variants (A-H) for the Parameter Golf screening experiments.
Each model is a stack of mixed {GDN, DeltaProduct, RWKV-7, Mamba-2, SWA} blocks
with shared MLP, RMSNorm, and residual connections.

Key design choices:
- FLA layers handle recurrent attention (GatedDeltaNet, GatedDeltaProduct, RWKV7, Mamba2)
- Sliding Window Attention (SWA) uses flash attention with a causal window mask
- All blocks follow the same pre-norm residual pattern for uniform gradient flow
- Weight sharing for SWA layers in Zamba/Hymba-style models
- Score-first eval: XSA-all only extends attention layers (no future context leakage)
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
# This is needed when:
# - Running on V100 (sm_70) which doesn't support FLA's Triton kernels well
# - Triton cache is corrupted (FileNotFoundError on .json files)
# - Debugging without Triton dependency
#
# On A100 (sm_80+), the Triton kernels are ~3-10x faster and should be used.
_USE_NAIVE = os.environ.get("FLA_USE_NAIVE", "0") == "1"

if _USE_NAIVE:
    # 1. Patch GatedDeltaNet's chunk op
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

    # 2. Patch GatedDeltaProduct's chunk op
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
        nn.init.zeros_(self.proj.weight)  # zero-init output for residual
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
        self.use_xsa = False  # enabled at eval time for XSA-all

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

        # Use window during training, full causal at eval if XSA enabled
        y = flash_attn_3_func(q, k, v, causal=True)

        if self.use_xsa:
            y = self._xsa_efficient(y, v)

        y = y.reshape(B, T, D)
        return self.proj(y)


class RecurrentBlock(nn.Module):
    """Wraps any FLA recurrent layer (GDN, DeltaProduct, RWKV-7, Mamba-2) with
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

        # FLA layers return (output, state) or just output depending on mode
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
        "gdn3_swa_gdn3_swa_shared_gdn3" -> [("gdn", 3), ("swa", 1), ("gdn", 3), ("swa_shared", 1), ("gdn", 3)]
        "mamba_only" -> [("mamba", 11)]
        "gdn3_mamba2_swa_gdn3_mamba2" -> [("gdn", 3), ("mamba", 2), ("swa", 1), ("gdn", 3), ("mamba", 2)]
    """
    if layout_str == "gdn_only":
        return [("gdn", -1)]  # -1 = use num_gdn_layers
    if layout_str == "mamba_only":
        return [("mamba", -1)]  # -1 = use num_mamba_layers
    if layout_str == "swa_only":
        return [("swa", -1)]  # -1 = use num_swa_layers

    # Parse custom layouts like "gdn5_swa_gdn5_swa_shared"
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
            # Check if next token is "shared"
            if i + 1 < len(parts) and parts[i + 1] == "shared":
                result.append(("swa_shared", 1))
                i += 1
            else:
                result.append(("swa", 1))
        elif part == "shared":
            # Already consumed by swa check above
            pass
        i += 1
    return result


class HybridGDN(nn.Module):
    """Hybrid GDN architecture supporting mixed recurrent/attention layers.

    Builds a stack of blocks according to the layer_layout specification:
    - "gdn" blocks use GatedDeltaNet (or GatedDeltaProduct, or RWKV-7)
    - "mamba" blocks use Mamba-2
    - "swa" blocks use SlidingWindowAttention
    - "swa_shared" reuses the same SWA module (Zamba-style weight sharing)

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

        # Meta tokens (Hymba-style, for Model E)
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
        self._block_types = []  # track type for XSA/diagnostics
        self._shared_swa = None  # shared SWA module for Zamba/Hymba models

        layer_idx = 0
        for layer_type, count in layout:
            if count == -1:
                # Fill with the specified layer type
                if layer_type == "gdn":
                    count = config["num_gdn_layers"]
                elif layer_type == "mamba":
                    count = config["num_mamba_layers"]
                elif layer_type == "swa":
                    count = config["num_swa_layers"]

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
                        swa = self._shared_swa  # reuse same SWA module
                    else:
                        swa = SlidingWindowAttention(
                            dim=dim,
                            num_heads=num_heads,
                            num_kv_heads=config.get("swa_num_kv_heads", 4),
                            window_size=config.get("swa_window", 512),
                        )
                        if config.get("swa_shared", False):
                            self._shared_swa = swa

                    # Each SWA position gets its own MLP even if SWA weights are shared
                    block = AttentionBlock(dim, swa, mlp_mult, layer_idx=layer_idx)
                    self.blocks.append(block)
                    self._block_types.append("swa" if layer_type == "swa" else "swa_shared")

                layer_idx += 1

        # KV sharing: share k/v projections between adjacent layers
        kv_stride = config.get("kv_sharing_stride", 0)
        if kv_stride > 0:
            self._apply_kv_sharing(kv_stride)

        self.final_norm = RMSNorm(dim)
        # Tied embeddings (standard for parameter golf)
        self.lm_head = None  # use tok_emb.weight
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
            # Default: GatedDeltaNet
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

    def _apply_kv_sharing(self, stride: int) -> None:
        """Share KV projection modules between adjacent layer groups.

        For GDN layers: shares k_proj, v_proj, k_conv1d, v_conv1d.
        For SWA layers: shares c_k, c_v.
        The first layer in each group is the anchor; subsequent layers in the
        group become followers that reference the anchor's modules.
        """
        # Collect indices by block type
        gdn_indices = [i for i, t in enumerate(self._block_types) if t == "gdn"]
        swa_indices = [i for i, t in enumerate(self._block_types)
                       if t in ("swa", "swa_shared")]

        # Share GDN KV projections within each stride-group
        for group_start in range(0, len(gdn_indices), stride):
            anchor_idx = gdn_indices[group_start]
            anchor = self.blocks[anchor_idx].recurrent
            for j in range(1, stride):
                if group_start + j >= len(gdn_indices):
                    break
                follower_idx = gdn_indices[group_start + j]
                follower = self.blocks[follower_idx].recurrent
                follower.k_proj = anchor.k_proj
                follower.v_proj = anchor.v_proj
                follower.k_conv1d = anchor.k_conv1d
                follower.v_conv1d = anchor.v_conv1d

        # Share SWA KV projections within each stride-group
        for group_start in range(0, len(swa_indices), stride):
            anchor_idx = swa_indices[group_start]
            anchor = self.blocks[anchor_idx].attn
            for j in range(1, stride):
                if group_start + j >= len(swa_indices):
                    break
                follower_idx = swa_indices[group_start + j]
                follower = self.blocks[follower_idx].attn
                follower.c_k = anchor.c_k
                follower.c_v = anchor.c_v

    def _init_weights(self) -> None:
        """Weight initialization.

        Each sub-module handles its own init (MLP zeros proj, SWA zeros proj,
        FLA layers do own init). We just do the residual scaling for output
        projections on our own CastedLinear layers.
        """
        total_layers = len(self.blocks)
        for name, p in self.named_parameters():
            # Skip FLA-internal parameters
            if ".recurrent." in name:
                continue
            # Scale down output projections for residual stream
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

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Forward pass returning cross-entropy loss."""
        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x

        # Prepend meta tokens if Hymba-style
        if self.meta_tokens is not None:
            B = x.shape[0]
            meta = self.meta_tokens.expand(B, -1, -1).to(dtype=x.dtype)
            x = torch.cat([meta, x], dim=1)
            x0 = torch.cat([meta, x0], dim=1)

        for block, btype in zip(self.blocks, self._block_types):
            if btype in ("swa", "swa_shared"):
                x = block(x, x0)
            else:
                x = block(x, x0)

        # Remove meta tokens before computing logits
        if self.meta_tokens is not None:
            x = x[:, self.n_meta:]

        x = self.final_norm(x)
        logits = self._compute_logits(x.reshape(-1, x.size(-1)))
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning logits (for evaluation)."""
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

        for block, btype in zip(self.blocks, self._block_types):
            if btype in ("swa", "swa_shared"):
                x = block(x, x0)
            else:
                x = block(x, x0)

        if self.meta_tokens is not None:
            x = x[:, self.n_meta:]

        x = self.final_norm(x)
        return self._compute_logits(x)

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
