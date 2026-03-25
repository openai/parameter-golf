"""Recurrent GPT model derived from the current best record.

Preserves: LeakyReLU(0.5)^2, BigramHash, XSA, partial RoPE, VE128,
LayerNorm scaling, Parameter Banking layout, EMA/SWA hooks.

Adds: stem / recurrent-core / tail partitioning with optional
fake-quantization inside the shared core and correction injection.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    flash_attn_3_func = None  # allow import on CPU for tests

from quant import fake_quantize_weight

# ---------------------------------------------------------------------------
# Base components (mirrored from the current-best record)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0,
                 train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2,
                                                  dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device,
                dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2,
                                  dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor,
                     rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin,
                             x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin,
                       x1 * (-sin) + x2 * cos), dim=-1)


class CastedLinear(nn.Linear):
    _qat_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]),
                                   -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float,
                 gated_attention: bool = False, value_residual: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init,
                                              dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False
        self.gated_attention = gated_attention
        if gated_attention:
            self.attn_gate = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)
        self.value_residual = value_residual
        if value_residual:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5],
                                                       dtype=torch.float32))

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor,
                v_w: Tensor, out_w: Tensor,
                v_embed: Tensor | None = None,
                v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(
            bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(
            bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v if self.value_residual else None
        if self.value_residual and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.gated_attention:
            gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
            y = y * gate
        y = y.reshape(bsz, seqlen, dim)
        return F.linear(y, out_w.to(x.dtype)), raw_v


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int,
                 model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = (CastedLinear(bigram_dim, model_dim, bias=False)
                     if bigram_dim != model_dim else None)
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = (torch.bitwise_xor(
            36313 * t[..., 1:], 27191 * t[..., :-1]) % mod)
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = (CastedLinear(ve_dim, model_dim, bias=False)
                     if ve_dim != model_dim else None)
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()

    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)),
                         negative_slope=0.5)
        return F.linear(x.square(), down_w.to(x.dtype))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float, qk_gain_init: float,
                 layer_idx: int = 0, ln_scale: bool = False,
                 dtg: bool = False, gated_attention: bool = False,
                 value_residual: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            gated_attention=gated_attention, value_residual=value_residual)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = (1.0 / math.sqrt(layer_idx + 1)
                                if ln_scale else 1.0)
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None

    def forward(self, x: Tensor, x0: Tensor,
                q_w: Tensor, k_w: Tensor, v_w: Tensor,
                out_w: Tensor, up_w: Tensor, down_w: Tensor,
                v_embed: Tensor | None = None,
                v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, raw_v = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor,
            q_w, k_w, v_w, out_w, v_embed=v_embed, v0=v0)
        x_out = x_in + (self.attn_scale.to(dtype=x_in.dtype)[None, None, :]
                         * attn_out)
        x_out = x_out + (self.mlp_scale.to(dtype=x_out.dtype)[None, None, :]
                          * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor,
                                     up_w, down_w))
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out, raw_v


# ---------------------------------------------------------------------------
# Recurrent GPT
# ---------------------------------------------------------------------------

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix",
    "resid_mixes", "q_gain", "skip_weight", "skip_weights", "smear",
    "dtg_gate", "ve_layer_scales", "ve_shared.scale", "attn_gate",
    "vr_lambda",
)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if ((param.ndim < 2
                 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS))
                    and param.dtype != torch.float32):
                param.data = param.data.float()


class RecurrentGPT(nn.Module):
    """Recurrent GPT: stem → shared core (×K) → tail.

    All architectural defaults match the current best record.  The recurrent
    core replaces a contiguous range of layers with a small set of shared
    blocks repeated *num_passes* times.

    The bank layout stores weights for num_unique layers (stem + core + tail),
    and the core bank entries are reused on each recurrence pass — optionally
    with fake quantization applied via STE.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        model_dim: int = 512,
        num_heads: int = 8,
        num_kv_heads: int = 4,
        mlp_mult: float = 3.0,
        tie_embeddings: bool = True,
        tied_embed_init_std: float = 0.005,
        logit_softcap: float = 30.0,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        bigram_vocab_size: int = 1536,
        bigram_dim: int = 128,
        rope_dims: int = 16,
        ln_scale: bool = True,
        ve_enabled: bool = True,
        ve_dim: int = 128,
        ve_layers: str = "",
        xsa_last_n: int = 4,
        gated_attention: bool = False,
        value_residual: bool = False,
        # recurrence
        num_stem_layers: int = 3,
        num_core_layers: int = 2,
        num_tail_layers: int = 3,
        num_passes: int = 3,
        # fake quant for core
        core_quant_bits: int = 6,
        core_quant_enabled: bool = True,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")

        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.value_residual = value_residual

        self.num_stem = num_stem_layers
        self.num_core = num_core_layers
        self.num_tail = num_tail_layers
        self.num_unique = num_stem_layers + num_core_layers + num_tail_layers
        self.num_passes = num_passes
        self.core_quant_bits = core_quant_bits
        self.core_quant_enabled = core_quant_enabled

        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)
        self._kv_dim = kv_dim

        # --- embeddings ---
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = (BigramHashEmbedding(bigram_vocab_size, bigram_dim,
                                           model_dim)
                       if bigram_vocab_size > 0 else None)
        self.smear = SmearGate(model_dim)

        # --- skip connections (stem ↔ tail) ---
        self.num_skip_weights = min(num_stem_layers, num_tail_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # --- parameter banks (sized for unique layers only) ---
        n = self.num_unique
        self.qo_bank = nn.Parameter(
            torch.empty(2 * n, model_dim, model_dim))
        self.kv_bank = nn.Parameter(
            torch.empty(2 * n, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(
            torch.empty(n, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(
            torch.empty(n, model_dim, mlp_dim))

        # core bank indices for quick lookup
        self._core_bank_start = num_stem_layers
        self._core_bank_end = num_stem_layers + num_core_layers

        # --- blocks ---
        def _make_block(layer_idx: int) -> Block:
            return Block(
                model_dim, num_heads, num_kv_heads, int(mlp_mult),
                rope_base, qk_gain_init, layer_idx=layer_idx,
                ln_scale=ln_scale, gated_attention=gated_attention,
                value_residual=value_residual)

        self.stem_blocks = nn.ModuleList(
            [_make_block(i) for i in range(num_stem_layers)])
        self.core_blocks = nn.ModuleList(
            [_make_block(num_stem_layers + j)
             for j in range(num_core_layers)])
        self.tail_blocks = nn.ModuleList(
            [_make_block(num_stem_layers + num_core_layers + i)
             for i in range(num_tail_layers)])

        # partial RoPE
        if rope_dims > 0:
            for blk in list(self.stem_blocks) + list(self.core_blocks) + list(self.tail_blocks):
                blk.attn.rope_dims = rope_dims
                blk.attn.rotary = Rotary(head_dim, base=rope_base,
                                         train_seq_len=1024, rope_dims=rope_dims)

        # XSA on last N unique layers
        all_blocks = (list(self.stem_blocks) + list(self.core_blocks)
                      + list(self.tail_blocks))
        if xsa_last_n > 0:
            for blk in all_blocks[max(0, len(all_blocks) - xsa_last_n):]:
                blk.attn.use_xsa = True

        # Value Embedding
        self.ve_layer_indices = ([int(x) for x in ve_layers.split(",") if x.strip()]
                                 if ve_enabled and ve_layers else [])
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32))
                 for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()

        # --- output ---
        self.final_norm = RMSNorm()
        self.lm_head = (None if tie_embeddings
                        else CastedLinear(model_dim, vocab_size, bias=False))
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        self._init_weights()

    # ---- weight init (mirrors base) ----

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0,
                            std=self.tied_embed_init_std)
        n = self.num_unique
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[n + i])
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.qo_bank.data[n + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (module.weight.ndim == 2 and module.weight.shape[0] >= 64
                      and module.weight.shape[1] >= 64):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    # ---- helpers ----

    def _get_ve(self, unique_layer_idx: int, input_ids: Tensor,
                ve_cache: dict) -> Tensor | None:
        if self.ve_shared is None or unique_layer_idx not in self.ve_layer_indices:
            return None
        if "ve" not in ve_cache:
            ve_cache["ve"] = self.ve_shared(input_ids)
        ve_base = ve_cache["ve"]
        ve_idx = self.ve_layer_indices.index(unique_layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    def _maybe_fq(self, w: Tensor, bank_idx: int) -> Tensor:
        """Apply fake quantization if bank_idx belongs to the core."""
        if (self.core_quant_enabled and self.training
                and self._core_bank_start <= bank_idx < self._core_bank_end):
            return fake_quantize_weight(w, self.core_quant_bits, per_row=True)
        return w

    def _bank_weights(self, bank_idx: int) -> tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return (q_w, k_w, v_w, out_w, up_w, down_w) for a unique layer."""
        n = self.num_unique
        q_w = self._maybe_fq(self.qo_bank[bank_idx], bank_idx)
        out_w = self._maybe_fq(self.qo_bank[n + bank_idx], bank_idx)
        k_w = self._maybe_fq(self.kv_bank[bank_idx], bank_idx)
        v_w = self._maybe_fq(self.kv_bank[n + bank_idx], bank_idx)
        up_w = self._maybe_fq(self.mlp_up_bank[bank_idx], bank_idx)
        down_w = self._maybe_fq(self.mlp_down_bank[bank_idx], bank_idx)
        return q_w, k_w, v_w, out_w, up_w, down_w

    # ---- forward (training) ----

    def forward(self, input_ids: Tensor, target_ids: Tensor,
                feedback_fn=None,
                stabilizer=None) -> Tensor:
        """Full forward with loss.

        Args:
            feedback_fn: callable(h, pass_idx) -> correction | None
            stabilizer:  RecurrentStabilizer instance (or None)
        """
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        ve_cache: dict = {}
        skips: list[Tensor] = []

        # --- STEM ---
        for i, blk in enumerate(self.stem_blocks):
            bi = i
            ve = self._get_ve(bi, input_ids, ve_cache)
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(bi)
            x, raw_v = blk(x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                           v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)

        # --- RECURRENT CORE ---
        for k in range(self.num_passes):
            for j, blk in enumerate(self.core_blocks):
                bi = self.num_stem + j
                h_prev = x

                # correction injection (inactive on pass 0)
                correction = None
                if feedback_fn is not None:
                    correction = feedback_fn(x, k)
                if correction is not None:
                    x = x + correction

                if stabilizer is not None:
                    x = stabilizer.clip(x)

                ve = self._get_ve(bi, input_ids, ve_cache)
                q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(bi)
                x, _ = blk(x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                           v_embed=ve, v0=v0)

                if stabilizer is not None:
                    stabilizer.record_pass(h_prev, x, correction=correction)

        # --- TAIL ---
        for i, blk in enumerate(self.tail_blocks):
            bi = self.num_stem + self.num_core + i
            if skips:
                x = x + (self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
                          * skips.pop())
            ve = self._get_ve(bi, input_ids, ve_cache)
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(bi)
            x, _ = blk(x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                       v_embed=ve, v0=v0)

        # --- OUTPUT ---
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(
            logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    # ---- forward_logits (eval) ----

    def forward_logits(self, input_ids: Tensor,
                       feedback_fn=None,
                       stabilizer=None) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without loss."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        ve_cache: dict = {}
        skips: list[Tensor] = []

        for i, blk in enumerate(self.stem_blocks):
            bi = i
            ve = self._get_ve(bi, input_ids, ve_cache)
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(bi)
            x, raw_v = blk(x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                           v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)

        for k in range(self.num_passes):
            for j, blk in enumerate(self.core_blocks):
                bi = self.num_stem + j
                correction = None
                if feedback_fn is not None:
                    correction = feedback_fn(x, k)
                if correction is not None:
                    x = x + correction
                if stabilizer is not None:
                    x = stabilizer.clip(x)
                ve = self._get_ve(bi, input_ids, ve_cache)
                q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(bi)
                x, _ = blk(x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                           v_embed=ve, v0=v0)

        for i, blk in enumerate(self.tail_blocks):
            bi = self.num_stem + self.num_core + i
            if skips:
                x = x + (self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
                          * skips.pop())
            ve = self._get_ve(bi, input_ids, ve_cache)
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(bi)
            x, _ = blk(x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                       v_embed=ve, v0=v0)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(
            logits_proj / self.logit_softcap)

    # ---- bank utility (for export) ----

    def core_bank_indices(self) -> list[int]:
        return list(range(self._core_bank_start, self._core_bank_end))

    def stem_bank_indices(self) -> list[int]:
        return list(range(self.num_stem))

    def tail_bank_indices(self) -> list[int]:
        start = self.num_stem + self.num_core
        return list(range(start, start + self.num_tail))

    def all_blocks(self) -> list[Block]:
        return (list(self.stem_blocks) + list(self.core_blocks)
                + list(self.tail_blocks))
