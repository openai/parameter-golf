from __future__ import annotations

import sys
import weakref
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import CONTROL_TENSOR_NAME_PATTERNS
from .flash_attention import get_flash_attn_fn
from .ternary import ternary_ste
from .initialization import apply_initialization

def _activation_frobenius_norm(t: torch.Tensor) -> float:
    """L2 norm of flattened activations (same for (B,H,S,D) vs (B,S,H,D) up to reshape)."""
    with torch.no_grad():
        return float(torch.linalg.vector_norm(t.detach().float().reshape(-1), ord=2))


def _maybe_log_activation(gpt: Any, layer_idx: int, suffix: str, t: torch.Tensor) -> None:
    if not getattr(gpt, "_activation_norm_collect", False):
        return
    buf = getattr(gpt, "_activation_norms", None)
    if buf is None:
        return
    buf[f"activations/layer{layer_idx}/{suffix}"] = _activation_frobenius_norm(t)


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    _qat_bits = 0
    _ternary = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self._ternary:
            w = ternary_ste(w)
        elif self._qat_bits > 0:
            # Generic intN QAT
            clip_range = (1 << (self._qat_bits - 1)) - 1
            # Per-row max abs
            amax = w.abs().max(dim=1, keepdim=True).values
            scale = (amax / clip_range).clamp_min(1e-8)
            w_q = (w / scale).round().clamp(-(clip_range + 1), clip_range)
            w_dq = w_q * scale
            w = w + (w_dq - w).detach()

        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        flash_attn_version: int = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        # 0 = torch SDPA, 2 = FlashAttention-2, 3 = FlashAttention-3
        self.flash_attn_version = flash_attn_version
        self._flash_fn = get_flash_attn_fn(flash_attn_version) if flash_attn_version in (2, 3) else None

    def _act_norm_ctx(self) -> tuple[Any, int] | tuple[None, None]:
        wr = getattr(self, "_norm_sink_block", None)
        if wr is None:
            return None, None
        block = wr()
        if block is None:
            return None, None
        gpt = getattr(block, "_gpt_parent", None)
        if gpt is None:
            return None, None
        return gpt, int(block.layer_idx)

    def _log_act(self, suffix: str, t: Tensor) -> None:
        gpt, layer_idx = self._act_norm_ctx()
        if gpt is not None:
            _maybe_log_activation(gpt, layer_idx, suffix, t)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        if self._flash_fn is not None:
            # FlashAttention path: keep (B, S, H, D) layout throughout to avoid
            # extra transposes.  cos/sin are permuted to (1, S, 1, D//2).
            q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
            k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            # cos/sin: (1, 1, S, D//2) → (1, S, 1, D//2)
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            cos = cos.permute(0, 2, 1, 3)
            sin = sin.permute(0, 2, 1, 3)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            # q_gain: (num_heads,) → (1, 1, H, 1)
            q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
            self._log_act("q", q)
            self._log_act("k", k)
            self._log_act("v", v)
            y = self._flash_fn(q, k, v, causal=True)  # (B, S, H, D)
            y = y.reshape(bsz, seqlen, dim)
            self._log_act("attn_post_mha", y)
        else:
            # Standard PyTorch SDPA path: (B, H, S, D) layout.
            q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
            self._log_act("q", q)
            self._log_act("k", k)
            self._log_act("v", v)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
            y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
            self._log_act("attn_post_mha", y)

        y = self.proj(y)
        self._log_act("attn_out", y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, proj_init: str = "zero"):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        if proj_init == "zero":
            self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


def _maybe_hyper_conn_init(
    hyper_conn_type: str, hyper_conn_n: int
) -> tuple[Callable | None, nn.Module | None, nn.Module | None]:
    if hyper_conn_type == "none" or hyper_conn_n <= 1:
        return None, None, None
    mhc_root = Path(__file__).resolve().parents[1] / "mhc-lite"
    mhc_root_str = str(mhc_root)
    if mhc_root_str not in sys.path:
        sys.path.insert(0, mhc_root_str)
    from hyper_conn import hyper_conn_init_func  # type: ignore

    init_hc, expand_stream, reduce_stream = hyper_conn_init_func(hyper_conn_type, hyper_conn_n)
    return init_hc, expand_stream, reduce_stream


class Block(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        init_hc: Callable | None = None,
        flash_attn_version: int = 0,
        mlp_proj_init: str = "zero",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, flash_attn_version)
        self.attn._norm_sink_block = weakref.ref(self)
        self.mlp = MLP(dim, mlp_mult, proj_init=mlp_proj_init)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.hc_attn = None if init_hc is None else init_hc(dim=dim, branch=nn.Sequential(self.attn_norm, self.attn))
        self.hc_mlp  = None if init_hc is None else init_hc(dim=dim, branch=nn.Sequential(self.mlp_norm, self.mlp))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.hc_attn is not None and self.hc_mlp is not None:
            x = self.hc_attn(x)
            x = self.hc_mlp(x)
        else:
            attn_out = self.attn(self.attn_norm(x))
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            gpt = getattr(self, "_gpt_parent", None)
            if gpt is not None and getattr(gpt, "_activation_norm_collect", False):
                _maybe_log_activation(gpt, self.layer_idx, "x_post_attn_residual", x)
            mlp_out = self.mlp(self.mlp_norm(x))
            if gpt is not None and getattr(gpt, "_activation_norm_collect", False):
                _maybe_log_activation(gpt, self.layer_idx, "mlp_out", mlp_out)
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
            if gpt is not None and getattr(gpt, "_activation_norm_collect", False):
                _maybe_log_activation(gpt, self.layer_idx, "x_post_mlp_residual", x)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        hyper_conn_type: str = "none",
        hyper_conn_n: int = 1,
        flash_attn_version: int = 0,
        mlp_proj_init: str = "zero",
        init_scheme: str = "default",
        mtp_num_heads: int = 1,
        mtp_loss_weight: float = 1.0,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        init_hc, expand_stream, reduce_stream = _maybe_hyper_conn_init(hyper_conn_type, hyper_conn_n)
        self.expand_stream = expand_stream
        self.reduce_stream = reduce_stream
        self.blocks = nn.ModuleList(
            [
                Block(
                    layer_idx,
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    init_hc,
                    flash_attn_version,
                    mlp_proj_init=mlp_proj_init,
                )
                for layer_idx in range(num_layers)
            ]
        )
        # Must not use plain assignment: nn.Module would register GPT as a child of Block
        # (cycle: GPT → blocks → Block → _gpt_parent → GPT), breaking .to() / named_children().
        for block in self.blocks:
            object.__setattr__(block, "_gpt_parent", self)
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if mtp_num_heads > 1:
            self.mtp_heads: nn.ModuleList | None = nn.ModuleList([
                CastedLinear(model_dim, vocab_size, bias=False)
                for _ in range(mtp_num_heads - 1)
            ])
            for head in self.mtp_heads:
                head._zero_init = True
        else:
            self.mtp_heads = None
        self._init_weights()
        apply_initialization(self, init_scheme, num_layers)

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        if getattr(self, "_activation_norm_collect", False):
            self._activation_norms = {}
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.expand_stream is not None:
            x = self.expand_stream(x)
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x)
        if self.reduce_stream is not None:
            x = self.reduce_stream(x)

        D = x.size(-1)
        x_flat = x.reshape(-1, D)
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        if self.training and self.mtp_heads is not None:
            S = x.size(1)
            mtp_aux_loss = torch.zeros_like(loss)
            for k, head in enumerate(self.mtp_heads, start=1):
                h_k = x[:, :S - k, :].reshape(-1, D)
                t_k = target_ids[:, k:].reshape(-1)
                logits_k = head(h_k)
                logits_k = self.logit_softcap * torch.tanh(logits_k / self.logit_softcap)
                mtp_aux_loss = mtp_aux_loss + F.cross_entropy(logits_k.float(), t_k, reduction="mean")
            loss = loss + self.mtp_loss_weight * mtp_aux_loss / len(self.mtp_heads)

        return loss
