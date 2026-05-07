"""
HybridQuantGPT v6.1 — Single-file Training + Evaluation Script

Mixed-precision quantization: Q/K:Int6, V/O:Int5, MLP-up:Pentanary, MLP-down:Int4, Embed:FP16
rANS entropy coding compression (15.07 MB artifact, 32.8M params)
Muon optimizer + SWA weight averaging + Sliding Window eval
Trained on 1×RTX 3090 for ~28 hours (10K steps)

Track: non-record-unlimited-compute-16mb
val_bpb: 1.2100 (sliding window stride=64, SWA weights)
val_bpb: 1.2420 (sequential, SWA weights)

Training:
    CUDA_VISIBLE_DEVICES=0 python train_gpt.py --train --iterations 10000 \\
        --v61 --ema 0.997 --ema-type hma --swa --lr-scale 1.0 \\
        --muon-momentum 0.95 --warmdown-ratio 0.175 \\
        --val-every 500 --save-every 2500 --micro-batch 16

Evaluation:
    python train_gpt.py --eval --checkpoint model.rans.ptz --stride 64
    python train_gpt.py --eval --checkpoint model.rans.ptz --ttt --stride 64
"""

from __future__ import annotations

import argparse
import copy
import glob
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# rANS Codec (Pure Python — no Rust FFI needed for eval)
# ============================================================

RANS_PRECISION = 16
RANS_NORM = 1 << RANS_PRECISION       # 65536
RANS_BYTE_L = 1 << 23


def _build_cdf(counts: np.ndarray, alphabet_size: int) -> list[int]:
    total = int(counts.sum())
    cdf = [0] * (alphabet_size + 1)
    cumulative = 0
    for i in range(alphabet_size):
        cdf[i] = (cumulative * RANS_NORM) // total
        cumulative += int(counts[i])
        if i > 0 and cdf[i] == cdf[i - 1]:
            cdf[i] = cdf[i - 1] + 1
    cdf[alphabet_size] = RANS_NORM
    return cdf


def rans_decode(compressed: bytes | np.ndarray, counts: np.ndarray,
                alphabet_size: int, num_symbols: int) -> np.ndarray:
    if isinstance(compressed, np.ndarray):
        data = compressed.tobytes()
    elif isinstance(compressed, (bytes, bytearray)):
        data = bytes(compressed)
    else:
        data = bytes(compressed)

    cdf = _build_cdf(counts, alphabet_size)
    sym_lut = np.zeros(RANS_NORM, dtype=np.uint8)
    for s in range(alphabet_size):
        sym_lut[cdf[s]:cdf[s + 1]] = s

    pos = 0
    state = 0
    for _ in range(4):
        state = (state << 8) | data[pos]
        pos += 1

    symbols = np.empty(num_symbols, dtype=np.uint8)
    mask = RANS_NORM - 1

    for i in range(num_symbols):
        slot = state & mask
        sym = sym_lut[slot]
        s = int(sym)
        freq = cdf[s + 1] - cdf[s]
        start = cdf[s]
        state = freq * (state >> RANS_PRECISION) + (state & mask) - start
        while state < RANS_BYTE_L and pos < len(data):
            state = (state << 8) | data[pos]
            pos += 1
        symbols[i] = sym

    return symbols


def deserialize_hybrid_rans(obj: dict) -> dict:
    """Pure Python rANS decoder: .rans.ptz artifact -> state_dict."""
    state_dict = {}

    for key in obj["rans_data"]:
        compressed = obj["rans_data"][key]
        counts = obj["rans_counts"][key]
        alpha = obj["rans_alphas"][key]
        shape = obj["rans_shapes"][key]
        scales = obj["rans_scales"][key].float()

        num_elements = 1
        for s in shape:
            num_elements *= s

        if hasattr(compressed, 'numpy'):
            comp_bytes = compressed.numpy()
        elif isinstance(compressed, torch.Tensor):
            comp_bytes = compressed.numpy()
        else:
            comp_bytes = np.frombuffer(compressed, dtype=np.uint8)

        if hasattr(counts, 'numpy'):
            count_array = counts.numpy().astype(np.uint32)
        elif isinstance(counts, torch.Tensor):
            count_array = counts.numpy().astype(np.uint32)
        else:
            count_array = np.ascontiguousarray(counts, dtype=np.uint32)

        decoded = rans_decode(comp_bytes, count_array, int(alpha), num_elements)
        symbols = torch.tensor(decoded, dtype=torch.float32).reshape(shape)
        half = alpha // 2
        w_q = symbols - half
        if alpha > 5:
            state_dict[key] = w_q * scales.unsqueeze(-1) / half
        else:
            state_dict[key] = w_q * scales.unsqueeze(-1)

    for key, val in obj["passthrough"].items():
        state_dict[key] = val.float()

    return state_dict


# ============================================================
# Quantization Layers
# ============================================================

class IntNLinear(nn.Module):
    """N-bit uniform quantization Linear."""
    _qat_enabled = True

    def __init__(self, in_features, out_features, n_bits, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits
        self.quant_type = f'int{n_bits}'
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._zero_init = False

    def _quantize(self, w):
        w_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        w_scaled = w / w_max
        half = self.n_levels // 2
        w_int = (w_scaled * half).round().clamp(-half, half - 1)
        w_q = w_int / half * w_max
        return w + (w_q - w).detach()

    def forward(self, x):
        if IntNLinear._qat_enabled and self.training:
            w_q = self._quantize(self.weight)
        else:
            w_q = self.weight
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_q.to(x.dtype), bias)

    def get_quantized_weights(self):
        """rANS 직렬화용: (symbols, alphabet_size, counts, scales)"""
        with torch.no_grad():
            w = self.weight
            clip = getattr(self, '_clip_ratio', None)
            if clip is not None:
                abs_w = w.abs()
                n = abs_w.shape[1]
                k = max(1, int(clip * n))
                w_max = abs_w.kthvalue(min(k, n), dim=1, keepdim=True).values.clamp(min=1e-5)
            else:
                w_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
            w_scaled = (w / w_max).clamp(-1, 1)
            half = self.n_levels // 2
            w_int = (w_scaled * half).round().clamp(-half, half - 1)
            symbols = (w_int + half).to(torch.uint8).cpu().numpy().flatten()
            alpha = self.n_levels
            counts = np.bincount(symbols, minlength=alpha).astype(np.uint32)
            scales = w_max.squeeze(-1).half().cpu()
        return symbols, alpha, counts, scales


class PentanaryLinear(nn.Module):
    """5-level quantization: {-2, -1, 0, +1, +2}."""
    _qat_enabled = True

    def __init__(self, in_features, out_features, bias=False, threshold_ratio=0.7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_ratio = threshold_ratio
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self._zero_init = False
        self.sparse_mask = None

    def _quantize_core(self, w, sparse_mask=None):
        abs_w = w.abs()
        mean_abs = abs_w.mean(dim=1, keepdim=True)
        t1 = self.threshold_ratio * mean_abs
        t2 = 2.0 * t1
        mask1 = abs_w > t1
        mask2 = abs_w > t2
        w_q = torch.sign(w) * (mask1.float() + mask2.float())
        if sparse_mask is not None:
            w_q = w_q * sparse_mask
        wq_sq = (w_q * w_q).sum(dim=1, keepdim=True).clamp(min=1e-8)
        w_wq = (w * w_q).sum(dim=1, keepdim=True)
        scale = w_wq / wq_sq
        return w_q, scale

    def forward(self, x):
        if not PentanaryLinear._qat_enabled and self.training:
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, self.weight.to(x.dtype), bias)
        w_q, scale = self._quantize_core(self.weight, self.sparse_mask)
        w_q_scaled = w_q * scale
        if self.sparse_mask is not None:
            w_active = self.weight * self.sparse_mask
            w_q_scaled = w_active + (w_q_scaled - w_active).detach()
        else:
            w_q_scaled = self.weight + (w_q_scaled - self.weight).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_q_scaled.to(x.dtype), bias)

    def get_quantized_weights(self):
        """rANS 직렬화용: (symbols, alphabet_size, counts, scales)"""
        w_q, scale = self._quantize_core(self.weight.detach(), self.sparse_mask)
        alpha = 5
        half = 2
        symbols = (w_q + half).to(torch.uint8).cpu().numpy().flatten()
        counts = np.bincount(symbols, minlength=alpha).astype(np.uint32)
        scales = scale.squeeze(-1).half().cpu()
        return symbols, alpha, counts, scales


class BitLinear(nn.Module):
    """Ternary quantized linear (compatibility shim)."""
    def __init__(self, in_features, out_features, bias=False, threshold_ratio=0.7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_ratio = threshold_ratio
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self._zero_init = False

    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


# ============================================================
# GPTQ-lite Clip Search
# ============================================================

def gptq_clip_search(model, percentiles=None, verbose=True):
    if percentiles is None:
        percentiles = [0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 1.0]
    total_before = 0.0
    total_after = 0.0
    n_layers = 0
    for name, module in model.named_modules():
        if not isinstance(module, IntNLinear):
            continue
        w = module.weight.data
        half = module.n_levels // 2
        out_feat, in_feat = w.shape
        best_ratios = torch.ones(out_feat, device=w.device)
        best_mse = torch.full((out_feat,), float('inf'), device=w.device)
        abs_w = w.abs()
        w_max_default = abs_w.amax(dim=1, keepdim=True).clamp(min=1e-5)
        w_scaled = w / w_max_default
        w_int = (w_scaled * half).round().clamp(-half, half - 1)
        w_q = w_int / half * w_max_default
        mse_default = (w - w_q).pow(2).mean(dim=1)
        total_before += mse_default.sum().item()
        for p in percentiles:
            k = max(1, int(p * in_feat))
            w_max_p = abs_w.kthvalue(min(k, in_feat), dim=1, keepdim=True).values.clamp(min=1e-5)
            w_scaled_p = (w / w_max_p).clamp(-1, 1)
            w_int_p = (w_scaled_p * half).round().clamp(-half, half - 1)
            w_q_p = w_int_p / half * w_max_p
            mse_p = (w - w_q_p).pow(2).mean(dim=1)
            improved = mse_p < best_mse
            best_mse[improved] = mse_p[improved]
            best_ratios[improved] = p
        module._clip_ratio = best_ratios.mean().item()
        total_after += best_mse.sum().item()
        n_layers += 1
        if verbose:
            improv = (1 - best_mse.sum().item() / mse_default.sum().item()) * 100
            print(f"  {name}: avg_clip={best_ratios.mean().item():.4f}, MSE improvement={improv:.2f}%")
    if verbose and total_before > 0:
        print(f"  Total: {n_layers} layers, MSE improvement={(1 - total_after / total_before) * 100:.2f}%")
    return total_before - total_after


# ============================================================
# Model Architecture
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class PartialRotary(nn.Module):
    def __init__(self, head_dim: int, rope_dims: int = 0, base: float = 10000.0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (
            self._cos_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        if bigram_dim != model_dim:
            self.proj = nn.Linear(bigram_dim, model_dim, bias=False)
            nn.init.zeros_(self.proj.weight)
        else:
            self.proj = None
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
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
        if ve_dim != model_dim:
            self.proj = nn.Linear(ve_dim, model_dim, bias=False)
            nn.init.zeros_(self.proj.weight)
        else:
            self.proj = None
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class HybridAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base=10000.0,
                 qk_gain_init=1.5, use_xsa=False, value_residual=False, rope_dims=0):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = IntNLinear(dim, dim, n_bits=6, bias=False)
        self.c_k = IntNLinear(dim, kv_dim, n_bits=6, bias=False)
        self.c_v = IntNLinear(dim, kv_dim, n_bits=5, bias=False)
        self.proj = IntNLinear(dim, dim, n_bits=5, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = rope_dims
        self.rotary = PartialRotary(self.head_dim, rope_dims=rope_dims, base=rope_base)
        self.use_xsa = use_xsa
        self.value_residual = value_residual
        if value_residual:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x, v0=None, v_embed=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_raw = self.c_v(x)
        if v_embed is not None:
            v_raw = v_raw + v_embed
        v = v_raw.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        raw_v = v if self.value_residual else None
        if self.value_residual and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, rope_dims=self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, rope_dims=self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        if self.use_xsa:
            y = y.transpose(1, 2)
            v_for_xsa = v.transpose(1, 2)
            y = self._xsa_efficient(y, v_for_xsa)
            y = y.contiguous().reshape(bsz, seqlen, dim)
        else:
            y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y), raw_v


class HybridMLP(nn.Module):
    def __init__(self, dim, hidden_mult=3.0):
        super().__init__()
        hidden = int(hidden_mult * dim)
        hidden = ((hidden + 63) // 64) * 64
        self.up = PentanaryLinear(dim, hidden, bias=False)
        self.down = IntNLinear(hidden, dim, n_bits=4, bias=False)
        self.down._zero_init = True

    def forward(self, x):
        x = F.leaky_relu(self.up(x), negative_slope=0.5)
        return self.down(x.square())


class HybridBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base=10000.0,
                 qk_gain_init=1.5, hidden_mult=3.0, use_xsa=False,
                 value_residual=False, rope_dims=0, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = HybridAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            use_xsa=use_xsa, value_residual=value_residual, rope_dims=rope_dims,
        )
        self.mlp = HybridMLP(dim, hidden_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x, x0, v0=None, v_embed=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = self.attn_norm(x) * self.ln_scale_factor
        attn_out, raw_v = self.attn(n, v0=v0, v_embed=v_embed)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * self.ln_scale_factor)
        return x, raw_v


class HybridQuantGPT(nn.Module):
    def __init__(self, vocab_size=1024, num_layers=11, model_dim=512,
                 num_heads=8, num_kv_heads=4, logit_softcap=30.0,
                 rope_base=10000.0, qk_gain_init=1.5, hidden_mult=3.0,
                 tie_embeddings=True, xsa_last_n=11, value_residual=True,
                 use_smear=True, bigram_vocab=2048, bigram_dim=128,
                 ve_enabled=True, ve_dim=128, ve_layers="9,10",
                 rope_dims=0, ln_scale=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.logit_softcap = logit_softcap
        self.tie_embeddings = tie_embeddings
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.smear = SmearGate(model_dim) if use_smear else None
        self.bigram = BigramHashEmbedding(bigram_vocab, bigram_dim, model_dim) if bigram_vocab > 0 else None

        self.blocks = nn.ModuleList([
            HybridBlock(
                model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init, hidden_mult,
                use_xsa=(i >= num_layers - xsa_last_n),
                value_residual=value_residual,
                rope_dims=rope_dims, layer_idx=i, ln_scale=ln_scale,
            )
            for i in range(num_layers)
        ])

        self.skip_weights = nn.Parameter(
            0.1 * torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )

        kv_dim = num_kv_heads * (model_dim // num_heads)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()

        self.final_norm = RMSNorm()
        if not tie_embeddings:
            self.lm_head = IntNLinear(model_dim, vocab_size, n_bits=8, bias=False)
        else:
            self.lm_head = None

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for name, module in self.named_modules():
            if isinstance(module, (IntNLinear, PentanaryLinear)):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and min(module.weight.shape) >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(proj_scale)

    def _get_ve(self, layer_idx, input_ids, ve_cache):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve']
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    def _forward_body(self, input_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        skips = []
        v0 = None
        ve_cache = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](x, x0, v0=v0, v_embed=ve)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            eff_idx = self.num_encoder_layers + i
            if skips:
                skip_w = self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
                x = x + skip_w * skips.pop()
            ve = self._get_ve(eff_idx, input_ids, ve_cache)
            x, _ = self.blocks[eff_idx](x, x0, v0=v0, v_embed=ve)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids, target_ids, z_loss_weight=0.0):
        logits = self._forward_body(input_ids)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            target_ids.reshape(-1), reduction="mean",
        )
        if z_loss_weight > 0:
            loss = loss + z_loss_weight * logits.float().logsumexp(-1).pow(2).mean()
        return loss

    def forward_logits(self, input_ids):
        return self._forward_body(input_ids)

    def param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        int6_params = int5_params = int4_params = penta_params = 0
        for m in self.modules():
            if isinstance(m, IntNLinear):
                n = sum(p.numel() for p in m.parameters() if p.ndim == 2)
                if m.n_bits == 6: int6_params += n
                elif m.n_bits == 5: int5_params += n
                elif m.n_bits == 4: int4_params += n
            elif isinstance(m, PentanaryLinear):
                penta_params += sum(p.numel() for p in m.parameters() if p.ndim == 2)
        quantized = int6_params + int5_params + int4_params + penta_params
        rans_est = (int6_params * 6 / 8 * 0.87 + int5_params * 5 / 8 * 0.90
                    + int4_params * 4 / 8 * 0.95 + penta_params * 2.32 / 8 * 0.89
                    + (total - quantized) * 2)
        return {"total_params": total, "ternary_params": quantized,
                "non_ternary_params": total - quantized,
                "effective_layers": self.num_layers,
                "estimated_artifact_mb": rans_est / 1_000_000,
                "under_16mb": rans_est < 16_000_000}


def make_model(qk_gain_init=2.0, logit_softcap=15.0):
    """v6.1: XSA-all + VE128(9,10) + PartialRoPE(16) + LN Scale."""
    return HybridQuantGPT(
        vocab_size=1024, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, hidden_mult=4.0, xsa_last_n=11,
        ve_enabled=True, ve_dim=128, ve_layers="9,10",
        rope_dims=16, ln_scale=True,
        qk_gain_init=qk_gain_init, logit_softcap=logit_softcap,
    )


# ============================================================
# rANS Serialization (training artifact — requires rans_codec_rs)
# ============================================================

def serialize_hybrid_rans(model: nn.Module) -> dict:
    """HybridQuantGPT -> rANS compressed artifact (requires rans_codec_rs Rust FFI)."""
    try:
        import rans_codec_rs
    except ImportError:
        raise ImportError("rans_codec_rs not available. Install from ngram_rs/ or use pre-built artifact.")

    rans_data = {}
    rans_counts = {}
    rans_alphas = {}
    rans_shapes = {}
    rans_scales = {}
    passthrough = {}

    for name, module in model.named_modules():
        if isinstance(module, (IntNLinear, PentanaryLinear)):
            key = name + ".weight"
            symbols, alpha, counts, scales = module.get_quantized_weights()
            counts = np.maximum(counts, 1).astype(np.uint32)
            compressed = rans_codec_rs.rans_encode(
                np.ascontiguousarray(symbols, dtype=np.uint8),
                np.ascontiguousarray(counts, dtype=np.uint32),
                int(alpha),
            )
            rans_data[key] = torch.frombuffer(bytearray(compressed), dtype=torch.uint8)
            rans_counts[key] = torch.from_numpy(counts.copy()).to(torch.int32)
            rans_alphas[key] = int(alpha)
            rans_shapes[key] = list(module.weight.shape)
            rans_scales[key] = scales
            if hasattr(module, 'bias') and module.bias is not None:
                passthrough[name + ".bias"] = module.bias.detach().half().cpu()

    quantized_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, (IntNLinear, PentanaryLinear)):
            quantized_modules.add(name)
    for name, param in model.named_parameters():
        base_name = name.rsplit(".", 1)[0] if "." in name else ""
        if base_name in quantized_modules:
            continue
        passthrough[name] = param.detach().half().cpu()

    return {
        "__format__": "hybrid_rans_v1",
        "rans_data": rans_data,
        "rans_counts": rans_counts,
        "rans_alphas": rans_alphas,
        "rans_shapes": rans_shapes,
        "rans_scales": rans_scales,
        "passthrough": passthrough,
    }


# ============================================================
# Data Loading Utilities
# ============================================================

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[:usable + 1]


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


# ============================================================
# Model Loading
# ============================================================

def load_model(checkpoint_path, device):
    model = make_model()
    if checkpoint_path.endswith(".rans.ptz"):
        print(f"[Load] rANS artifact: {checkpoint_path}")
        t0 = time.time()
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = deserialize_hybrid_rans(obj)
        print(f"  rANS decode: {time.time()-t0:.1f}s")
    elif checkpoint_path.endswith(".pt"):
        print(f"[Load] raw checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in ckpt and "step" in ckpt:
            if "ema_shadow" in ckpt:
                ema_state = ckpt["ema_shadow"]
                if "fast" in ema_state:
                    state_dict = ema_state["smoother"]
                else:
                    state_dict = ema_state
            else:
                state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unsupported format: {checkpoint_path}")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    summary = model.param_summary()
    print(f"  Parameters: {summary['total_params']:,}")
    return model


# ============================================================
# Muon Optimizer (from parameter-golf/train_gpt.py)
# ============================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        import torch.distributed as dist
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()


# ============================================================
# EMA / HMA — Weight Averaging
# ============================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self._backup = {}

    def update(self, model: nn.Module):
        d = self.decay
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].lerp_(p.data, 1.0 - d)

    def apply(self, model: nn.Module):
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}

    def state_dict(self):
        return {n: v.clone() for n, v in self.shadow.items()}

    def load_state_dict(self, state):
        for n, v in state.items():
            if n in self.shadow:
                self.shadow[n].copy_(v)


class HMA:
    """Hull Moving Average: 2 EMA (fast + slow) + sqrt(n) smoothing."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        decay_fast = 1.0 - 2.0 * (1.0 - decay)
        self.fast = EMA(model, decay=decay_fast)
        self.slow = EMA(model, decay=decay)
        n = 1.0 / (1.0 - decay)
        smooth_decay = 1.0 - 1.0 / max(n ** 0.5, 1.0)
        self.smoother = EMA(model, decay=smooth_decay)
        self._backup = {}

    def update(self, model: nn.Module):
        self.fast.update(model)
        self.slow.update(model)
        with torch.no_grad():
            for n in self.smoother.shadow:
                hull = 2.0 * self.fast.shadow[n] - self.slow.shadow[n]
                self.smoother.shadow[n].lerp_(hull, 1.0 - self.smoother.decay)

    def apply(self, model: nn.Module):
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.smoother.shadow:
                self._backup[n] = p.data.clone()
                p.data.copy_(self.smoother.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}

    def state_dict(self):
        return {"fast": self.fast.state_dict(), "slow": self.slow.state_dict(),
                "smoother": self.smoother.state_dict()}

    def load_state_dict(self, state):
        self.fast.load_state_dict(state["fast"])
        self.slow.load_state_dict(state["slow"])
        self.smoother.load_state_dict(state["smoother"])


# ============================================================
# Data Loader (simplified from parameter-golf)
# ============================================================

class SimpleTokenLoader:
    """Single-process token loader from binary shards."""
    def __init__(self, train_pattern: str, device: torch.device):
        self.files = sorted(glob.glob(train_pattern))
        assert self.files, f"No train files found: {train_pattern}"
        self.device = device
        self._shard_idx = 0
        self._pos = 0
        self._tokens = None
        self._load_shard()

    def _load_shard(self):
        self._tokens = load_data_shard(Path(self.files[self._shard_idx]))
        self._pos = 0

    def next_batch(self, batch_tokens: int, seq_len: int, grad_accum_steps: int):
        micro_batch_seqs = batch_tokens // seq_len // grad_accum_steps
        total = micro_batch_seqs * seq_len + 1
        if self._pos + total > self._tokens.numel():
            self._shard_idx = (self._shard_idx + 1) % len(self.files)
            self._load_shard()
        buf = self._tokens[self._pos:self._pos + total].to(dtype=torch.int64, device=self.device)
        self._pos += micro_batch_seqs * seq_len
        x = buf[:-1].reshape(micro_batch_seqs, seq_len)
        y = buf[1:].reshape(micro_batch_seqs, seq_len)
        return x, y


# ============================================================
# Training Loop
# ============================================================

def get_lr_scale(step: int, warmup_steps: int, iterations: int, warmdown_iters: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    elif step >= iterations - warmdown_iters:
        progress = (iterations - step) / warmdown_iters
        return max(0.0, progress)
    return 1.0


def train_main(args):
    """Training entry point."""
    device = torch.device(args.device if hasattr(args, 'device') else "cuda:0")

    # Hyperparameters
    lr_s = args.lr_scale
    matrix_lr = 0.01 * lr_s
    tied_embed_lr = 0.0125 * lr_s
    scalar_lr = 0.01 * lr_s
    iterations = args.iterations
    seq_len = args.seq_len
    batch_tokens = args.batch_tokens
    warmup_steps = max(10, min(200, iterations // 50))
    warmdown_iters = max(50, int(iterations * args.warmdown_ratio))

    # Model
    model = make_model(qk_gain_init=args.qk_gain, logit_softcap=args.softcap)
    model = model.to(device)
    summary = model.param_summary()

    print(f"{'=' * 60}")
    print(f"HybridQuantGPT v6.1 Training")
    print(f"{'=' * 60}")
    print(f"Total params:     {summary['total_params']:>12,}")
    print(f"Est. artifact:    {summary['estimated_artifact_mb']:.2f} MB")
    print(f"Iterations:       {iterations}")
    print(f"Batch tokens:     {batch_tokens}")
    print(f"Seq len:          {seq_len}")

    # Data
    data_dir = args.data_dir
    tokenizer_path = args.tokenizer
    train_pattern = os.path.join(data_dir, "fineweb_train_*.bin")
    val_pattern = os.path.join(data_dir, "fineweb_val_*.bin")
    loader = SimpleTokenLoader(train_pattern, device)

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    base_bytes_lut, has_space_lut, is_boundary_lut = build_sentencepiece_luts(sp, 1024, device)
    val_tokens = load_validation_tokens(val_pattern, seq_len)

    # Grad accumulation
    micro_batch_seqs = batch_tokens // seq_len
    max_micro = args.micro_batch if args.micro_batch > 0 else 64
    if micro_batch_seqs > max_micro:
        grad_accum_steps = math.ceil(micro_batch_seqs / max_micro)
        micro_batch_seqs = micro_batch_seqs // grad_accum_steps
    else:
        grad_accum_steps = 1

    # Optimizers
    embed_params, matrix_params, scalar_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "tok_emb" in name:
            embed_params.append(param)
        elif param.ndim >= 2:
            matrix_params.append(param)
        else:
            scalar_params.append(param)

    optimizer_adam = torch.optim.AdamW(
        [{"params": embed_params, "lr": tied_embed_lr, "base_lr": tied_embed_lr},
         {"params": scalar_params, "lr": scalar_lr, "base_lr": scalar_lr}],
        betas=(0.9, 0.95), eps=1e-8, weight_decay=args.wd, fused=True,
    )
    optimizer_muon = Muon(
        [{"params": matrix_params, "lr": matrix_lr, "base_lr": matrix_lr}],
        lr=matrix_lr, momentum=args.muon_momentum, backend_steps=5,
    )
    optimizers = [optimizer_adam, optimizer_muon]

    # EMA / HMA
    ema = None
    if args.ema > 0:
        if args.ema_type == "hma":
            ema = HMA(model, decay=args.ema)
            print(f"EMA: type=hma, decay={args.ema}")
        else:
            ema = EMA(model, decay=args.ema)
            print(f"EMA: type=ema, decay={args.ema}")

    # Compile
    global zeropower_via_newtonschulz5
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # SWA state
    swa_state = None
    swa_count = 0
    swa_interval = 50
    swa_enabled = args.swa

    # Training
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    run_name = args.run_name or "v61_run"
    save_dir = f"runs/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    t0 = time.perf_counter()
    step = 0

    print(f"\nTraining started...")
    while step < iterations:
        scale = get_lr_scale(step, warmup_steps, iterations, warmdown_iters)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        # Late QAT
        if args.late_qat > 0:
            qat_start = int(iterations * (1 - args.late_qat))
            if step >= qat_start:
                IntNLinear._qat_enabled = True
                PentanaryLinear._qat_enabled = True
            else:
                IntNLinear._qat_enabled = False
                PentanaryLinear._qat_enabled = False

        # Forward + backward
        train_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = loader.next_batch(batch_tokens, seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, z_loss_weight=args.z_loss)
            train_loss += loss.item()
            (loss / grad_accum_steps).backward()
        train_loss /= grad_accum_steps

        # Momentum warmup
        frac = min(step / max(args.momentum_warmup_steps, 1), 1.0)
        muon_mom = (1 - frac) * args.momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_mom

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        if args.wd > 0:
            with torch.no_grad():
                for group in optimizer_muon.param_groups:
                    for p in group["params"]:
                        p.mul_(1.0 - group["lr"] * args.wd)

        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update(model)

        # SWA collection
        if swa_enabled and scale < 0.2 and (step + 1) % swa_interval == 0:
            with torch.no_grad():
                sd = model.state_dict()
                if swa_state is None:
                    swa_state = {k: v.float().cpu() for k, v in sd.items()}
                    swa_count = 1
                else:
                    swa_count += 1
                    for k in swa_state:
                        swa_state[k] += (sd[k].float().cpu() - swa_state[k]) / swa_count
                print(f"  SWA snapshot #{swa_count} at step {step + 1}")

        step += 1
        training_time_ms = 1000.0 * (time.perf_counter() - t0)

        if step <= 10 or step % args.log_every == 0:
            print(f"step:{step}/{iterations} train_loss:{train_loss:.4f} "
                  f"step_avg:{training_time_ms / step:.2f}ms")

        # Validation
        if args.val_every > 0 and step % args.val_every == 0:
            if ema is not None:
                ema.apply(model)
            model.eval()
            # Simple sequential eval
            val_loss_sum = 0.0
            val_count = 0
            with torch.inference_mode():
                for i in range(0, min(val_tokens.numel() - 1, 524288), seq_len):
                    end = min(i + seq_len, val_tokens.numel() - 1)
                    if end - i < seq_len:
                        break
                    xv = val_tokens[i:i + seq_len].unsqueeze(0).to(dtype=torch.int64, device=device)
                    yv = val_tokens[i + 1:i + seq_len + 1].unsqueeze(0).to(dtype=torch.int64, device=device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        vloss = model(xv, yv)
                    val_loss_sum += vloss.item()
                    val_count += 1
            if val_count > 0:
                vl = val_loss_sum / val_count
                vbpb = vl / math.log(2.0)
                print(f"  -> val_loss:{vl:.4f} val_bpb:{vbpb:.4f}")
            if ema is not None:
                ema.restore(model)
            model.train()

        # Checkpoint
        if args.save_every > 0 and step % args.save_every == 0:
            ckpt_path = f"{save_dir}/step{step}.pt"
            ckpt_data = {"model": model.state_dict(), "step": step, "train_loss": train_loss}
            if ema is not None:
                ckpt_data["ema_shadow"] = ema.state_dict()
            torch.save(ckpt_data, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            print(f"  checkpoint: {ckpt_path}")

    total_time = time.perf_counter() - t0
    print(f"\nTraining done: {step} steps, {total_time:.1f}s")

    # Final eval
    if ema is not None:
        ema.apply(model)

    # SWA comparison
    if swa_enabled and swa_state is not None and swa_count > 1:
        print(f"\nSWA collected {swa_count} snapshots")
        swa_sd = {k: v.to(device).to(model.state_dict()[k].dtype) for k, v in swa_state.items()}
        model.load_state_dict(swa_sd)

    # Save
    model_path = f"{save_dir}/model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")

    try:
        rans_path = f"{save_dir}/model.rans.ptz"
        obj = serialize_hybrid_rans(model)
        torch.save(obj, rans_path)
        ptz_size = os.path.getsize(rans_path)
        print(f"Saved: {rans_path} ({ptz_size:,} bytes)")
        print(f"Under 16MB: {'YES' if ptz_size < 16_000_000 else 'NO'}")
    except ImportError as e:
        print(f"rANS serialization skipped: {e}")


# ============================================================
# Sliding Window Eval
# ============================================================

def eval_sliding_window(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                        is_boundary_token_lut, device, seq_len=1024, stride=64,
                        batch_seqs=32, temperature=1.0):
    total_tokens = val_tokens.numel() - 1
    window_starts = [
        ws for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0
    ]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    t0 = time.perf_counter()
    dev_type = "cuda" if device.type == "cuda" else "cpu"

    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_seqs):
            batch_ws = window_starts[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda")):
                logits = model.forward_logits(x_batch)

            scaled_logits = logits.float() / temperature if temperature != 1.0 else logits.float()
            nll = F.cross_entropy(
                scaled_logits.reshape(-1, logits.size(-1)),
                y_batch.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            if bi % (batch_seqs * 50) == 0:
                done = min(bi + batch_seqs, len(window_starts))
                pct = done / len(window_starts) * 100
                if token_count.item() > 0:
                    rl = (loss_sum / token_count).item()
                    rbpb = rl / math.log(2.0) * (token_count.item() / byte_count.item())
                    print(f"  [{pct:5.1f}%] {done}/{len(window_starts)} windows bpb={rbpb:.6f}")

    elapsed = time.perf_counter() - t0
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return {"val_loss": val_loss, "val_bpb": val_bpb,
            "total_tokens": int(token_count.item()),
            "total_bytes": int(byte_count.item()), "elapsed": elapsed}


# ============================================================
# Legal TTT: Score-First Recipe
# ============================================================

def eval_sliding_ttt(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                     is_boundary_token_lut, device, seq_len=1024, stride=64,
                     batch_seqs=32, ttt_lr=0.002, ttt_epochs=3, ttt_momentum=0.9,
                     ttt_grad_clip=1.0, ttt_chunk_tokens=32768,
                     ttt_freeze_blocks=0, ttt_batch_seqs=32, temperature=1.0):
    saved_qat_intn = IntNLinear._qat_enabled
    saved_qat_penta = PentanaryLinear._qat_enabled
    IntNLinear._qat_enabled = False
    PentanaryLinear._qat_enabled = False

    total_tokens = val_tokens.numel() - 1
    window_starts = [
        ws for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0
    ]
    num_chunks = (total_tokens + ttt_chunk_tokens - 1) // ttt_chunk_tokens
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)

    print(f"[TTT] chunks={num_chunks} lr={ttt_lr} epochs={ttt_epochs}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    frozen_block_ids = set(range(min(ttt_freeze_blocks, len(model.blocks))))
    ttt_params = []
    for name, p in model.named_parameters():
        freeze = any(f"blocks.{bi}." in name for bi in frozen_block_ids)
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    optimizer = torch.optim.SGD(ttt_params, lr=ttt_lr, momentum=ttt_momentum)
    t0 = time.perf_counter()
    dev_type = "cuda" if device.type == "cuda" else "cpu"

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue

        # Score phase
        model.eval()
        with torch.inference_mode():
            for bi in range(0, len(windows), batch_seqs):
                batch_ws = windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]
                with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda")):
                    logits = model.forward_logits(x_batch)
                scaled_logits = logits.float() / temperature if temperature != 1.0 else logits.float()
                nll = F.cross_entropy(
                    scaled_logits.reshape(-1, logits.size(-1)),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # Train phase
        chunk_start = ci * ttt_chunk_tokens
        chunk_end = min((ci + 1) * ttt_chunk_tokens, total_tokens)
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and ttt_epochs > 0:
            model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                for _ep in range(ttt_epochs):
                    for bs in range(0, chunk_seqs, ttt_batch_seqs):
                        be = min(bs + ttt_batch_seqs, chunk_seqs)
                        start_tok = chunk_start + bs * seq_len
                        end_tok = chunk_start + be * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type == "cuda")):
                            loss = model(x, y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(ttt_params, ttt_grad_clip)
                        optimizer.step()

        if ci % 10 == 0 or ci == num_chunks - 1:
            elapsed = time.perf_counter() - t0
            if token_count.item() > 0:
                rl = loss_sum.item() / max(token_count.item(), 1)
                rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1))
                print(f"  [TTT chunk {ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    IntNLinear._qat_enabled = saved_qat_intn
    PentanaryLinear._qat_enabled = saved_qat_penta

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    elapsed = time.perf_counter() - t0
    print(f"[TTT] Done: val_bpb={val_bpb:.6f} elapsed={elapsed:.1f}s")
    return {"val_loss": val_loss, "val_bpb": val_bpb,
            "total_tokens": int(token_count.item()),
            "total_bytes": int(byte_count.item()), "elapsed": elapsed}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HybridQuantGPT v6.1 Train + Eval")

    # Mode
    parser.add_argument("--train", action="store_true", help="Training mode")
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")

    # Common
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seq-len", type=int, default=1024)

    # Eval args
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-seqs", type=int, default=32)
    parser.add_argument("--ttt", action="store_true")
    parser.add_argument("--ttt-lr", type=float, default=0.002)
    parser.add_argument("--ttt-epochs", type=int, default=3)
    parser.add_argument("--ttt-momentum", type=float, default=0.9)
    parser.add_argument("--ttt-grad-clip", type=float, default=1.0)
    parser.add_argument("--ttt-chunk-tokens", type=int, default=32768)
    parser.add_argument("--ttt-freeze-blocks", type=int, default=0)
    parser.add_argument("--ttt-batch-seqs", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gptq-clip", action="store_true")
    parser.add_argument("--compile", action="store_true")

    # Training args
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch-tokens", type=int, default=524288)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--micro-batch", type=int, default=0)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--ema", type=float, default=0.0)
    parser.add_argument("--ema-type", choices=["ema", "hma"], default="ema")
    parser.add_argument("--v61", action="store_true")
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--warmdown-ratio", type=float, default=0.175)
    parser.add_argument("--late-qat", type=float, default=0.0)
    parser.add_argument("--z-loss", type=float, default=0.0)
    parser.add_argument("--qk-gain", type=float, default=2.0)
    parser.add_argument("--softcap", type=float, default=15.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--momentum-warmup-steps", type=int, default=500)
    parser.add_argument("--momentum-warmup-start", type=float, default=0.85)
    parser.add_argument("--run-name", type=str, default="")

    # Data paths
    script_dir = Path(__file__).resolve().parent
    default_pg = script_dir
    for candidate in [script_dir / "parameter-golf", script_dir.parent / "parameter-golf",
                      script_dir.parent.parent / "parameter-golf"]:
        if candidate.exists():
            default_pg = candidate
            break
    parser.add_argument("--data-dir", default=str(default_pg / "data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--tokenizer", default=str(default_pg / "data/tokenizers/fineweb_1024_bpe.model"))
    args = parser.parse_args()

    if args.train:
        train_main(args)
    elif args.eval or args.checkpoint:
        device = torch.device(args.device)
        print("=" * 60)
        print("HybridQuantGPT v6.1 Eval")
        print("=" * 60)
        model = load_model(args.checkpoint, device)

        if args.compile and not args.ttt:
            model = torch.compile(model, dynamic=False, fullgraph=True)

        if args.gptq_clip:
            gptq_clip_search(model, verbose=True)

        val_pattern = os.path.join(args.data_dir, "fineweb_val_*.bin")
        val_tokens = load_validation_tokens(val_pattern, args.seq_len)
        print(f"  val tokens: {val_tokens.numel():,}")

        sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
            build_sentencepiece_luts(sp, 1024, device)

        print(f"\n{'=' * 60}")
        print(f"[1] Sliding Window (stride={args.stride})")
        print(f"{'=' * 60}")
        sw_result = eval_sliding_window(
            model, val_tokens, base_bytes_lut, has_leading_space_lut,
            is_boundary_token_lut, device, args.seq_len, args.stride,
            args.batch_seqs, temperature=args.temperature,
        )
        print(f"\n  val_bpb: {sw_result['val_bpb']:.6f}")
        print(f"  Time: {sw_result['elapsed']:.1f}s")

        if args.ttt:
            print(f"\n{'=' * 60}")
            print(f"[2] Legal TTT (score-first)")
            print(f"{'=' * 60}")
            ttt_model = copy.deepcopy(model)
            ttt_result = eval_sliding_ttt(
                ttt_model, val_tokens, base_bytes_lut, has_leading_space_lut,
                is_boundary_token_lut, device, args.seq_len, args.stride,
                args.batch_seqs, ttt_lr=args.ttt_lr, ttt_epochs=args.ttt_epochs,
                ttt_momentum=args.ttt_momentum, ttt_grad_clip=args.ttt_grad_clip,
                ttt_chunk_tokens=args.ttt_chunk_tokens,
                ttt_freeze_blocks=args.ttt_freeze_blocks,
                ttt_batch_seqs=args.ttt_batch_seqs, temperature=args.temperature,
            )
            print(f"\n  TTT val_bpb: {ttt_result['val_bpb']:.6f}")

        print(f"\n{'=' * 60}")
        print(f"Results")
        print(f"{'=' * 60}")
        print(f"  Sliding Window: {sw_result['val_bpb']:.6f} bpb")
        if args.ttt:
            print(f"  Legal TTT:      {ttt_result['val_bpb']:.6f} bpb")
        if os.path.exists(args.checkpoint):
            print(f"  Artifact: {os.path.getsize(args.checkpoint):,} bytes")
    else:
        parser.print_help()
        print("\nUse --train or --eval (with --checkpoint) to run.")


if __name__ == "__main__":
    main()
