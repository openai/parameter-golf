#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel as DDP

from spectral_flood_walk_v0 import (
    build_sentencepiece_luts,
    cleanup_distributed_if_needed,
    distributed_env,
    export_quantized_model_npz,
    get_cuda_memory_stats,
    init_distributed_if_needed,
    is_rank0,
    load_data_shard_prefix,
    load_vocab_size,
    maybe_all_reduce_mean,
    maybe_reset_cuda_peak_memory,
    maybe_sync_cuda,
    resolve_device,
    set_seed,
    setup_auto_logging,
)


try:
    _SDPA_SUPPORTS_ENABLE_GQA = "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
except (TypeError, ValueError):
    _SDPA_SUPPORTS_ENABLE_GQA = False


try:
    _IS_BATCHED_TENSOR = torch._C._functorch.is_batchedtensor
except AttributeError:
    _IS_BATCHED_TENSOR = None


def attention_backend_override(*tensors: torch.Tensor) -> SDPBackend | None:
    """Pick a vmap-safe SDPA backend when functorch batched tensors are active."""
    if _IS_BATCHED_TENSOR is None:
        return None
    if any(_IS_BATCHED_TENSOR(tensor) for tensor in tensors):
        return SDPBackend.MATH
    return None


def parse_int_csv(spec: str) -> tuple[int, ...]:
    if not spec.strip():
        return ()
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


def build_lm_starts(num_tokens: int, seq_len: int, stride: int) -> list[int]:
    stop = num_tokens - seq_len - 1
    if stop <= 0:
        return []
    return list(range(0, stop, stride))


def build_eval_score_mask(batch_starts: list[int], seq_len: int, stride: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(batch_starts), seq_len), dtype=torch.bool, device=device)
    tail = min(stride, seq_len)
    for row, start in enumerate(batch_starts):
        if start == 0:
            mask[row, :] = True
        else:
            mask[row, seq_len - tail :] = True
    return mask


def batch_from_starts(
    tokens: torch.Tensor,
    starts: list[int],
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    windows = [tokens[start : start + seq_len + 1] for start in starts]
    batch = torch.stack(windows).to(device=device, dtype=torch.long)
    return batch[:, :-1], batch[:, 1:]


def lr_scale_for_step(step: int, total_steps: int, warmup_steps: int, min_lr_scale: float, cooldown_start_frac: float) -> float:
    if total_steps <= 1:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    cooldown_start = max(warmup_steps, int(total_steps * cooldown_start_frac))
    if step < cooldown_start or cooldown_start >= total_steps - 1:
        return 1.0
    progress = (step - cooldown_start) / max(total_steps - cooldown_start - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_scale + (1.0 - min_lr_scale) * cosine


def summarize_history(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {}
    summary: dict[str, float] = {"num_records": float(len(history))}
    keys = sorted({key for item in history for key in item.keys() if key != "step"})
    for key in keys:
        values = [float(item[key]) for item in history if key in item]
        if values:
            summary[f"mean_{key}"] = float(sum(values) / len(values))
    return summary


@dataclass
class V2Config:
    data_path: str = "data/datasets/fineweb10B_sp1024"
    tokenizer_path: str = "data/tokenizers/fineweb_1024_bpe.model"
    output_json: str | None = None
    auto_log_path: str | None = None
    model_artifact_path: str | None = None
    residual_artifact_path: str | None = None
    device: str = "auto"
    train_tokens: int = 262_144
    val_tokens: int = 65_536
    seq_len: int = 256
    stride: int = 64
    batch_size: int = 8
    train_steps: int = 128
    calibration_steps: int = 96
    eval_batches: int = 128
    report_every: int = 8
    vocab_size_override: int = 0
    model_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 3
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    rope_base: float = 10_000.0
    logit_softcap: float = 30.0
    qk_gain_init: float = 1.5
    spine_variant: str = "plain"
    xsa_last_n: int = 0
    residual_rank: int = 16
    residual_orders: str = "1,2,3,4"
    residual_table_size: int = 65_536
    residual_update_lr: float = 0.05
    residual_decay: float = 0.999
    residual_init_scale: float = 0.01
    base_lr: float = 2e-3
    basis_lr: float = 1e-2
    min_lr_scale: float = 0.10
    warmup_steps: int = 8
    cooldown_start_frac: float = 0.75
    weight_decay: float = 1e-2
    seed: int = 1337

    @property
    def residual_order_ids(self) -> tuple[int, ...]:
        return parse_int_csv(self.residual_orders)


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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
        use_xsa: bool = False,
    ) -> None:
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
        self.use_xsa = use_xsa

    def _xsa_efficient(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch, seq, heads, dim = y.shape
        kv_heads = v.size(-2)
        group = heads // kv_heads
        y_grouped = y.reshape(batch, seq, kv_heads, group, dim)
        v_norm = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_grouped * v_norm).sum(dim=-1, keepdim=True) * v_norm
        return (y_grouped - proj).reshape(batch, seq, heads, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q = self.c_q(x).reshape(batch, seq, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(batch, seq, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seq, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        v = v.to(dtype=q.dtype)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        if self.num_heads != self.num_kv_heads and not _SDPA_SUPPORTS_ENABLE_GQA:
            repeat = self.num_heads // self.num_kv_heads
            k_t = k_t.repeat_interleave(repeat, dim=1)
            v_t = v_t.repeat_interleave(repeat, dim=1)
        sdpa_kwargs = {"is_causal": True}
        if self.num_heads != self.num_kv_heads and _SDPA_SUPPORTS_ENABLE_GQA:
            sdpa_kwargs["enable_gqa"] = True
        backend_override = attention_backend_override(q_t, k_t, v_t)
        if backend_override is None:
            y = F.scaled_dot_product_attention(q_t, k_t, v_t, **sdpa_kwargs).transpose(1, 2)
        else:
            with sdpa_kernel(backend_override):
                y = F.scaled_dot_product_attention(q_t, k_t, v_t, **sdpa_kwargs).transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(batch, seq, dim))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int) -> None:
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        use_xsa: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class StrongTransformerLM(nn.Module):
    def __init__(self, cfg: V2Config, vocab_size: int) -> None:
        super().__init__()
        if cfg.num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tie_embeddings = cfg.tie_embeddings
        self.logit_softcap = cfg.logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, cfg.model_dim)
        self.num_encoder_layers = cfg.num_layers // 2
        self.num_decoder_layers = cfg.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, cfg.model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList()
        xsa_start = max(0, cfg.num_layers - cfg.xsa_last_n) if cfg.spine_variant == "xsa" else cfg.num_layers
        for idx in range(cfg.num_layers):
            self.blocks.append(
                Block(
                    cfg.model_dim,
                    cfg.num_heads,
                    cfg.num_kv_heads,
                    cfg.mlp_mult,
                    cfg.rope_base,
                    cfg.qk_gain_init,
                    use_xsa=idx >= xsa_start,
                )
            )
        self.final_norm = RMSNorm()
        self.lm_head = None if cfg.tie_embeddings else CastedLinear(cfg.model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.cfg.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def compact_bytes(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[torch.Tensor] = []
        for idx in range(self.num_encoder_layers):
            x = self.blocks[idx](x, x0)
            skips.append(x)
        for idx in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + idx](x, x0)
        return self.final_norm(x)

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        flat = hidden.reshape(-1, hidden.size(-1))
        if self.tie_embeddings:
            logits_proj = F.linear(flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits.reshape(hidden.size(0), hidden.size(1), self.vocab_size)

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        hidden = self.encode(input_ids)
        logits = self.logits_from_hidden(hidden)
        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), target_ids.reshape(-1))
        return {"hidden": hidden, "logits": logits, "loss": loss}


class ResidualBasis(nn.Module):
    def __init__(self, vocab_size: int, rank: int, num_orders: int, init_scale: float) -> None:
        super().__init__()
        self.basis = nn.Parameter(torch.randn(vocab_size, rank) * init_scale)
        self.order_scale_logits = nn.Parameter(torch.zeros(num_orders, dtype=torch.float32))

    def order_scales(self) -> torch.Tensor:
        return torch.sigmoid(self.order_scale_logits)

    def project(self, coeffs: torch.Tensor) -> torch.Tensor:
        return coeffs @ self.basis.transpose(0, 1)

    def correction(self, order_coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scales = self.order_scales().to(dtype=order_coeffs.dtype)
        combined = (order_coeffs * scales.view(1, 1, -1, 1)).sum(dim=2)
        return self.project(combined), combined

    def projected_direction(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat_probs = probs.reshape(-1, probs.size(-1))
        expected = flat_probs @ self.basis
        target_basis = self.basis[targets.reshape(-1)]
        return (target_basis - expected).reshape(*targets.shape, -1)


class ResidualRouter:
    def __init__(self, orders: tuple[int, ...], table_size: int, device: torch.device) -> None:
        self.orders = orders
        self.table_size = table_size
        self.device = device
        self.offset_mul = torch.tensor(
            [6361, 10007, 16931, 26699, 39367, 58757, 75941, 100151],
            dtype=torch.int64,
            device=device,
        )
        self.mix_prime = 1_000_003

    def context_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        if not self.orders:
            return torch.empty((batch, seq, 0), dtype=torch.long, device=input_ids.device)
        token_ids = input_ids.to(torch.int64) + 1
        mod = max(self.table_size - 1, 1)
        outputs: list[torch.Tensor] = []
        for order in self.orders:
            ids = torch.zeros((batch, seq), dtype=torch.long, device=input_ids.device)
            if order <= seq:
                length = seq - order + 1
                acc = torch.zeros((batch, length), dtype=torch.int64, device=input_ids.device)
                for offset in range(order):
                    segment = token_ids[:, offset : offset + length]
                    acc = (acc * self.mix_prime + segment * self.offset_mul[offset]) % mod
                ids[:, order - 1 :] = (acc + 1).to(torch.long)
            outputs.append(ids)
        return torch.stack(outputs, dim=2)


class ResidualTables:
    def __init__(
        self,
        *,
        orders: tuple[int, ...],
        table_size: int,
        rank: int,
        device: torch.device,
        decay: float,
    ) -> None:
        self.orders = orders
        self.table_size = table_size
        self.rank = rank
        self.device = device
        self.decay = float(decay)
        self.coeffs = torch.zeros((len(orders), table_size, rank), dtype=torch.float32, device=device)
        self.counts = torch.zeros((len(orders), table_size), dtype=torch.float32, device=device)
        self.hits = torch.zeros((len(orders), table_size), dtype=torch.float32, device=device)

    def clone(self) -> "ResidualTables":
        cloned = ResidualTables(
            orders=self.orders,
            table_size=self.table_size,
            rank=self.rank,
            device=self.device,
            decay=self.decay,
        )
        cloned.coeffs = self.coeffs.clone()
        cloned.counts = self.counts.clone()
        cloned.hits = self.hits.clone()
        return cloned

    def resident_bytes(self) -> int:
        total = 0
        for tensor in (self.coeffs, self.counts, self.hits):
            total += int(tensor.numel() * tensor.element_size())
        return total

    def compact_bytes_estimate(self) -> int:
        coeff_bytes = int(self.coeffs.numel())
        scale_bytes = int(self.coeffs.size(0) * self.coeffs.size(1) * 2)
        stat_bytes = int((self.counts.numel() + self.hits.numel()) * 2)
        return coeff_bytes + scale_bytes + stat_bytes

    def lookup(self, context_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if context_ids.numel() == 0:
            empty = torch.zeros((*context_ids.shape[:2], 0, self.rank), dtype=torch.float32, device=self.device)
            return empty, {"active_slots": 0.0, "mean_coeff_norm": 0.0}
        gathered = []
        active_slots = 0.0
        coeff_norms: list[float] = []
        for order_idx in range(context_ids.size(2)):
            ids = context_ids[:, :, order_idx]
            coeff = self.coeffs[order_idx, ids]
            gathered.append(coeff)
            if ids.numel() > 0:
                active_slots += float((self.counts[order_idx, ids] > 0).float().mean().item())
                coeff_norms.append(float(coeff.norm(dim=-1).mean().item()))
        stats = {
            "active_slots": active_slots / max(context_ids.size(2), 1),
            "mean_coeff_norm": sum(coeff_norms) / max(len(coeff_norms), 1),
        }
        return torch.stack(gathered, dim=2), stats

    @torch.no_grad()
    def note_reads(self, context_ids: torch.Tensor) -> None:
        if context_ids.numel() == 0:
            return
        for order_idx in range(context_ids.size(2)):
            ids = context_ids[:, :, order_idx].reshape(-1)
            self.hits[order_idx].scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @torch.no_grad()
    def update(
        self,
        context_ids: torch.Tensor,
        projected_direction: torch.Tensor,
        order_scales: torch.Tensor,
        step_size: float,
        score_mask: torch.Tensor | None = None,
    ) -> None:
        if context_ids.numel() == 0:
            return
        scale_values = order_scales.to(device=self.device, dtype=torch.float32)
        direction = projected_direction.to(device=self.device, dtype=torch.float32)
        weights = None
        if score_mask is not None:
            weights = score_mask.to(device=self.device, dtype=torch.float32).reshape(-1)
        for order_idx in range(context_ids.size(2)):
            ids = context_ids[:, :, order_idx].reshape(-1)
            order_delta = (direction.reshape(-1, self.rank) * scale_values[order_idx]) * step_size
            if weights is not None:
                order_delta = order_delta * weights.unsqueeze(-1)
            self.coeffs[order_idx].mul_(self.decay)
            self.coeffs[order_idx].scatter_add_(0, ids.unsqueeze(-1).expand(-1, self.rank), order_delta)
            count_values = torch.ones_like(ids, dtype=torch.float32)
            if weights is not None:
                count_values = weights
            self.counts[order_idx].scatter_add_(0, ids, count_values)

    def stats(self) -> dict[str, float]:
        non_empty = (self.counts > 0).float()
        per_order_non_empty = non_empty.sum(dim=1)
        coeff_norm = self.coeffs.norm(dim=-1)
        return {
            "resident_mb": float(self.resident_bytes() / (1024.0 * 1024.0)),
            "compact_mb_estimate": float(self.compact_bytes_estimate() / (1024.0 * 1024.0)),
            "non_empty_fraction": float(non_empty.mean().item()),
            "mean_non_empty_slots": float(per_order_non_empty.mean().item()),
            "max_non_empty_slots": float(per_order_non_empty.max().item()),
            "mean_coeff_norm": float(coeff_norm.mean().item()),
            "mean_hits": float(self.hits.mean().item()),
        }

    def export_npz(self, basis: ResidualBasis, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        basis_arr = basis.basis.detach().cpu().numpy()
        basis_scale = max(float(np.max(np.abs(basis_arr))) / 127.0, 1e-8) if basis_arr.size else 1e-8
        basis_q = np.clip(np.round(basis_arr / basis_scale), -127, 127).astype(np.int8, copy=False)
        coeff_arr = self.coeffs.detach().cpu().numpy()
        coeff_scale = np.maximum(np.max(np.abs(coeff_arr), axis=-1, keepdims=True) / 127.0, 1e-8).astype(np.float32)
        coeff_q = np.clip(np.round(coeff_arr / coeff_scale), -127, 127).astype(np.int8, copy=False)
        np.savez_compressed(
            path,
            basis_q=basis_q,
            basis_scale=np.array([basis_scale], dtype=np.float32),
            coeff_q=coeff_q,
            coeff_scale=coeff_scale.astype(np.float16),
            order_scales=basis.order_scales().detach().cpu().numpy().astype(np.float16),
            counts=self.counts.detach().cpu().numpy().astype(np.float16),
        )
        return int(path.stat().st_size)


def train_base_model(
    model: StrongTransformerLM,
    cfg: V2Config,
    train_tokens: torch.Tensor,
    train_starts: list[int],
    device: torch.device,
    ddp_enabled: bool,
) -> tuple[list[dict[str, float]], float]:
    rank, local_rank, _ = distributed_env()
    core_model: StrongTransformerLM = model
    if ddp_enabled:
        ddp_model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        model_for_train: StrongTransformerLM | DDP = ddp_model
        core_model = ddp_model.module  # type: ignore[assignment]
    else:
        model_for_train = model

    optimizer = torch.optim.AdamW(core_model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed + rank + 17)
    history: list[dict[str, float]] = []
    maybe_sync_cuda(device)
    maybe_reset_cuda_peak_memory(device)
    start_time = time.perf_counter()

    for step in range(cfg.train_steps):
        lr_scale = lr_scale_for_step(step, cfg.train_steps, cfg.warmup_steps, cfg.min_lr_scale, cfg.cooldown_start_frac)
        for group in optimizer.param_groups:
            group["lr"] = cfg.base_lr * lr_scale
        step_start = time.perf_counter()
        batch_starts = rng.sample(train_starts, cfg.batch_size)
        inputs, targets = batch_from_starts(train_tokens, batch_starts, cfg.seq_len, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model_for_train(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss = out["loss"]
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core_model.parameters(), 1.0)
        optimizer.step()

        if is_rank0() and (cfg.report_every <= 1 or step % cfg.report_every == 0 or step == cfg.train_steps - 1):
            loss_value = maybe_all_reduce_mean(float(loss.item()), device, ddp_enabled)
            metrics = {
                "step": float(step),
                "loss": loss_value,
                "lr_scale": float(lr_scale),
                "step_ms": float((time.perf_counter() - step_start) * 1000.0),
            }
            cuda_stats = get_cuda_memory_stats(device)
            if cuda_stats is not None:
                metrics["cuda_allocated_mb"] = float(cuda_stats["allocated_bytes"] / (1024.0 * 1024.0))
                metrics["cuda_peak_allocated_mb"] = float(cuda_stats["max_allocated_bytes"] / (1024.0 * 1024.0))
            history.append(metrics)
            cuda_fragment = ""
            if "cuda_allocated_mb" in metrics:
                cuda_fragment = f" vram={metrics['cuda_allocated_mb']:.1f}MB peak={metrics['cuda_peak_allocated_mb']:.1f}MB"
            print(
                "[train] "
                f"step={step}/{cfg.train_steps - 1} "
                f"loss={metrics['loss']:.4f} "
                f"lr_scale={metrics['lr_scale']:.3f} "
                f"step_ms={metrics['step_ms']:.1f}"
                f"{cuda_fragment}"
            )
        elif ddp_enabled:
            _ = maybe_all_reduce_mean(float(loss.item()), device, ddp_enabled)

    maybe_sync_cuda(device)
    return history, time.perf_counter() - start_time


def calibrate_residuals(
    model: StrongTransformerLM,
    basis: ResidualBasis,
    tables: ResidualTables,
    router: ResidualRouter,
    cfg: V2Config,
    train_tokens: torch.Tensor,
    train_starts: list[int],
    device: torch.device,
) -> tuple[list[dict[str, float]], float]:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    basis.train()
    optimizer = torch.optim.AdamW(basis.parameters(), lr=cfg.basis_lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed + 901)
    history: list[dict[str, float]] = []
    maybe_sync_cuda(device)
    maybe_reset_cuda_peak_memory(device)
    start_time = time.perf_counter()

    for step in range(cfg.calibration_steps):
        lr_scale = lr_scale_for_step(step, cfg.calibration_steps, max(1, cfg.warmup_steps // 2), cfg.min_lr_scale, cfg.cooldown_start_frac)
        for group in optimizer.param_groups:
            group["lr"] = cfg.basis_lr * lr_scale
        step_start = time.perf_counter()
        batch_starts = rng.sample(train_starts, cfg.batch_size)
        inputs, targets = batch_from_starts(train_tokens, batch_starts, cfg.seq_len, device)
        context_ids = router.context_ids(inputs)
        with torch.no_grad():
            base_logits = model(inputs)["logits"]
            assert base_logits is not None
        order_coeffs, table_stats = tables.lookup(context_ids)
        delta_logits, _ = basis.correction(order_coeffs)
        logits = base_logits.detach() + delta_logits
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(basis.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=-1)
            direction = basis.projected_direction(probs, targets)
            tables.update(context_ids, direction, basis.order_scales().detach(), cfg.residual_update_lr)
            tables.note_reads(context_ids)

        if cfg.report_every <= 1 or step % cfg.report_every == 0 or step == cfg.calibration_steps - 1:
            stats = tables.stats()
            metrics = {
                "step": float(step),
                "loss": float(loss.item()),
                "lr_scale": float(lr_scale),
                "step_ms": float((time.perf_counter() - step_start) * 1000.0),
                "active_slots": float(table_stats["active_slots"]),
                "mean_coeff_norm": float(table_stats["mean_coeff_norm"]),
                "table_resident_mb": float(stats["resident_mb"]),
                "table_non_empty_fraction": float(stats["non_empty_fraction"]),
            }
            history.append(metrics)
            print(
                "[calib] "
                f"step={step}/{cfg.calibration_steps - 1} "
                f"loss={metrics['loss']:.4f} "
                f"lr_scale={metrics['lr_scale']:.3f} "
                f"active={metrics['active_slots']:.3f} "
                f"table_mb={metrics['table_resident_mb']:.2f} "
                f"step_ms={metrics['step_ms']:.1f}"
            )

    maybe_sync_cuda(device)
    basis.eval()
    return history, time.perf_counter() - start_time


def evaluate_mode(
    *,
    mode: str,
    model: StrongTransformerLM,
    basis: ResidualBasis | None,
    tables: ResidualTables | None,
    router: ResidualRouter | None,
    tokens: torch.Tensor,
    starts: list[int],
    cfg: V2Config,
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> dict[str, object]:
    model.eval()
    active_tables = tables.clone() if mode == "online_residual" and tables is not None else tables
    maybe_sync_cuda(device)
    maybe_reset_cuda_peak_memory(device)
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    total_correct = 0
    active_slot_means: list[float] = []
    coeff_norm_means: list[float] = []
    eval_starts = starts[: cfg.eval_batches * cfg.batch_size]
    start_time = time.perf_counter()
    with torch.no_grad():
        for idx in range(0, len(eval_starts), cfg.batch_size):
            batch_starts = eval_starts[idx : idx + cfg.batch_size]
            inputs, targets = batch_from_starts(tokens, batch_starts, cfg.seq_len, device)
            score_mask = build_eval_score_mask(batch_starts, cfg.seq_len, cfg.stride, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(inputs)
            logits = out["logits"]
            assert logits is not None
            if mode != "context" and basis is not None and active_tables is not None and router is not None:
                context_ids = router.context_ids(inputs)
                order_coeffs, table_stats = active_tables.lookup(context_ids)
                delta_logits, _ = basis.correction(order_coeffs)
                logits = logits + delta_logits.to(dtype=logits.dtype)
                active_slot_means.append(float(table_stats["active_slots"]))
                coeff_norm_means.append(float(table_stats["mean_coeff_norm"]))
            flat_loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            total_loss += float(flat_loss[score_mask].sum().item())
            total_tokens += int(score_mask.sum().item())
            total_correct += int(((logits.argmax(dim=-1) == targets) & score_mask).sum().item())
            prev_ids = inputs.reshape(-1)
            tgt_ids = targets.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            total_bytes += float(token_bytes.reshape_as(targets)[score_mask].to(torch.float64).sum().item())

            if mode == "online_residual" and basis is not None and active_tables is not None and router is not None:
                probs = torch.softmax(logits.float(), dim=-1)
                direction = basis.projected_direction(probs, targets)
                active_tables.update(
                    context_ids,
                    direction,
                    basis.order_scales().detach(),
                    cfg.residual_update_lr,
                    score_mask=score_mask,
                )
                active_tables.note_reads(context_ids)

    maybe_sync_cuda(device)
    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / max(total_tokens, 1)
    bits_per_token = avg_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / max(total_bytes, 1.0))
    result: dict[str, object] = {
        "loss": float(avg_loss),
        "val_bpb": float(val_bpb),
        "accuracy": float(total_correct / max(total_tokens, 1)),
        "tokens": float(total_tokens),
        "bytes": float(total_bytes),
        "elapsed_s": float(elapsed),
        "tokens_per_s": float(total_tokens / max(elapsed, 1e-6)),
    }
    if active_slot_means:
        result["active_slots_mean"] = float(sum(active_slot_means) / len(active_slot_means))
        result["coeff_norm_mean"] = float(sum(coeff_norm_means) / len(coeff_norm_means))
    if active_tables is not None:
        result["residual_tables"] = active_tables.stats()
    cuda_stats = get_cuda_memory_stats(device)
    if cuda_stats is not None:
        result["cuda_memory"] = cuda_stats
    return result


def run_v2a(cfg: V2Config) -> dict[str, object]:
    set_seed(cfg.seed)
    rank, local_rank, world_size = distributed_env()
    device = resolve_device(cfg.device, local_rank)
    ddp_enabled = init_distributed_if_needed(device)
    tokenizer_path = Path(cfg.tokenizer_path)
    vocab_size = cfg.vocab_size_override or load_vocab_size(tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        tokenizer_path, vocab_size, device
    )

    train_path = Path(cfg.data_path) / "fineweb_train_000000.bin"
    val_path = Path(cfg.data_path) / "fineweb_val_000000.bin"
    train_tokens = load_data_shard_prefix(train_path, cfg.train_tokens)
    val_tokens = load_data_shard_prefix(val_path, cfg.val_tokens)
    train_starts = build_lm_starts(int(train_tokens.numel()), cfg.seq_len, cfg.stride)
    val_starts = build_lm_starts(int(val_tokens.numel()), cfg.seq_len, cfg.stride)
    if len(train_starts) < cfg.batch_size or len(val_starts) < cfg.batch_size:
        raise ValueError("not enough tokens for the requested V2a config")

    model = StrongTransformerLM(cfg, vocab_size=vocab_size).to(device)
    train_history, train_elapsed = train_base_model(model, cfg, train_tokens, train_starts, device, ddp_enabled)

    residual_basis = ResidualBasis(vocab_size, cfg.residual_rank, len(cfg.residual_order_ids), cfg.residual_init_scale).to(device)
    router = ResidualRouter(cfg.residual_order_ids, cfg.residual_table_size, device)
    residual_tables = ResidualTables(
        orders=cfg.residual_order_ids,
        table_size=cfg.residual_table_size,
        rank=cfg.residual_rank,
        device=device,
        decay=cfg.residual_decay,
    )
    calibration_history, calibration_elapsed = calibrate_residuals(
        model,
        residual_basis,
        residual_tables,
        router,
        cfg,
        train_tokens,
        train_starts,
        device,
    )

    result: dict[str, object] = {}
    if is_rank0():
        train_cuda_memory = get_cuda_memory_stats(device)
        model_artifact_path = Path(cfg.model_artifact_path) if cfg.model_artifact_path is not None else None
        if model_artifact_path is None and cfg.output_json is not None:
            model_artifact_path = Path(cfg.output_json).with_name("model_int8.npz")
        residual_artifact_path = Path(cfg.residual_artifact_path) if cfg.residual_artifact_path is not None else None
        if residual_artifact_path is None and cfg.output_json is not None:
            residual_artifact_path = Path(cfg.output_json).with_name("residual_tables.npz")
        model_artifact_bytes = None
        residual_artifact_bytes = None
        if model_artifact_path is not None:
            model_artifact_bytes = export_quantized_model_npz(model, model_artifact_path)
        if residual_artifact_path is not None:
            residual_artifact_bytes = residual_tables.export_npz(residual_basis, residual_artifact_path)

        eval_context = evaluate_mode(
            mode="context",
            model=model,
            basis=None,
            tables=None,
            router=None,
            tokens=val_tokens,
            starts=val_starts,
            cfg=cfg,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        eval_static = evaluate_mode(
            mode="static_residual",
            model=model,
            basis=residual_basis,
            tables=residual_tables,
            router=router,
            tokens=val_tokens,
            starts=val_starts,
            cfg=cfg,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        eval_online = evaluate_mode(
            mode="online_residual",
            model=model,
            basis=residual_basis,
            tables=residual_tables,
            router=router,
            tokens=val_tokens,
            starts=val_starts,
            cfg=cfg,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )

        result = {
            "config": {**asdict(cfg), "resolved_device": str(device), "world_size": world_size, "vocab_size": vocab_size},
            "train_history": train_history,
            "train_summary": summarize_history(train_history),
            "calibration_history": calibration_history,
            "calibration_summary": summarize_history(calibration_history),
            "train_elapsed_s": float(train_elapsed),
            "calibration_elapsed_s": float(calibration_elapsed),
            "train_tokens_per_s": float((cfg.train_steps * cfg.batch_size * cfg.seq_len * world_size) / max(train_elapsed, 1e-6)),
            "calibration_tokens_per_s": float((cfg.calibration_steps * cfg.batch_size * cfg.seq_len) / max(calibration_elapsed, 1e-6)),
            "train_cuda_memory": train_cuda_memory,
            "base_model_param_mb_estimate": float(model.compact_bytes() / (1024.0 * 1024.0)),
            "residual_table_resident_mb_estimate": float(residual_tables.resident_bytes() / (1024.0 * 1024.0)),
            "model_artifact_path": str(model_artifact_path) if model_artifact_path is not None else None,
            "residual_artifact_path": str(residual_artifact_path) if residual_artifact_path is not None else None,
            "model_artifact_bytes": float(model_artifact_bytes) if model_artifact_bytes is not None else None,
            "residual_artifact_bytes": float(residual_artifact_bytes) if residual_artifact_bytes is not None else None,
            "total_artifact_bytes": float((model_artifact_bytes or 0) + (residual_artifact_bytes or 0)),
            "eval_context": eval_context,
            "eval_static_residual": eval_static,
            "eval_online_residual": eval_online,
            "eval_delta_static_bpb": float(eval_static["val_bpb"] - eval_context["val_bpb"]),
            "eval_delta_online_bpb": float(eval_online["val_bpb"] - eval_context["val_bpb"]),
        }
        if cfg.output_json is not None:
            output_path = Path(cfg.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2) + "\n")

    if ddp_enabled:
        torch.distributed.barrier()
    cleanup_distributed_if_needed()
    return result


def parse_args() -> V2Config:
    parser = argparse.ArgumentParser(description="Spectral Flood Walk V2a runner")
    parser.add_argument("--data-path", default=V2Config.data_path)
    parser.add_argument("--tokenizer-path", default=V2Config.tokenizer_path)
    parser.add_argument("--output-json", default=V2Config.output_json)
    parser.add_argument("--auto-log-path", default=V2Config.auto_log_path)
    parser.add_argument("--model-artifact-path", default=V2Config.model_artifact_path)
    parser.add_argument("--residual-artifact-path", default=V2Config.residual_artifact_path)
    parser.add_argument("--device", default=V2Config.device)
    parser.add_argument("--train-tokens", type=int, default=V2Config.train_tokens)
    parser.add_argument("--val-tokens", type=int, default=V2Config.val_tokens)
    parser.add_argument("--seq-len", type=int, default=V2Config.seq_len)
    parser.add_argument("--stride", type=int, default=V2Config.stride)
    parser.add_argument("--batch-size", type=int, default=V2Config.batch_size)
    parser.add_argument("--train-steps", type=int, default=V2Config.train_steps)
    parser.add_argument("--calibration-steps", type=int, default=V2Config.calibration_steps)
    parser.add_argument("--eval-batches", type=int, default=V2Config.eval_batches)
    parser.add_argument("--report-every", type=int, default=V2Config.report_every)
    parser.add_argument("--vocab-size-override", type=int, default=V2Config.vocab_size_override)
    parser.add_argument("--model-dim", type=int, default=V2Config.model_dim)
    parser.add_argument("--num-layers", type=int, default=V2Config.num_layers)
    parser.add_argument("--num-heads", type=int, default=V2Config.num_heads)
    parser.add_argument("--num-kv-heads", type=int, default=V2Config.num_kv_heads)
    parser.add_argument("--mlp-mult", type=int, default=V2Config.mlp_mult)
    parser.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction, default=V2Config.tie_embeddings)
    parser.add_argument("--tied-embed-init-std", type=float, default=V2Config.tied_embed_init_std)
    parser.add_argument("--rope-base", type=float, default=V2Config.rope_base)
    parser.add_argument("--logit-softcap", type=float, default=V2Config.logit_softcap)
    parser.add_argument("--qk-gain-init", type=float, default=V2Config.qk_gain_init)
    parser.add_argument("--spine-variant", choices=("plain", "xsa"), default=V2Config.spine_variant)
    parser.add_argument("--xsa-last-n", type=int, default=V2Config.xsa_last_n)
    parser.add_argument("--residual-rank", type=int, default=V2Config.residual_rank)
    parser.add_argument("--residual-orders", default=V2Config.residual_orders)
    parser.add_argument("--residual-table-size", type=int, default=V2Config.residual_table_size)
    parser.add_argument("--residual-update-lr", type=float, default=V2Config.residual_update_lr)
    parser.add_argument("--residual-decay", type=float, default=V2Config.residual_decay)
    parser.add_argument("--residual-init-scale", type=float, default=V2Config.residual_init_scale)
    parser.add_argument("--base-lr", type=float, default=V2Config.base_lr)
    parser.add_argument("--basis-lr", type=float, default=V2Config.basis_lr)
    parser.add_argument("--min-lr-scale", type=float, default=V2Config.min_lr_scale)
    parser.add_argument("--warmup-steps", type=int, default=V2Config.warmup_steps)
    parser.add_argument("--cooldown-start-frac", type=float, default=V2Config.cooldown_start_frac)
    parser.add_argument("--weight-decay", type=float, default=V2Config.weight_decay)
    parser.add_argument("--seed", type=int, default=V2Config.seed)
    args = parser.parse_args()
    return V2Config(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        output_json=args.output_json,
        auto_log_path=args.auto_log_path,
        model_artifact_path=args.model_artifact_path,
        residual_artifact_path=args.residual_artifact_path,
        device=args.device,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        calibration_steps=args.calibration_steps,
        eval_batches=args.eval_batches,
        report_every=args.report_every,
        vocab_size_override=args.vocab_size_override,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        rope_base=args.rope_base,
        logit_softcap=args.logit_softcap,
        qk_gain_init=args.qk_gain_init,
        spine_variant=args.spine_variant,
        xsa_last_n=args.xsa_last_n,
        residual_rank=args.residual_rank,
        residual_orders=args.residual_orders,
        residual_table_size=args.residual_table_size,
        residual_update_lr=args.residual_update_lr,
        residual_decay=args.residual_decay,
        residual_init_scale=args.residual_init_scale,
        base_lr=args.base_lr,
        basis_lr=args.basis_lr,
        min_lr_scale=args.min_lr_scale,
        warmup_steps=args.warmup_steps,
        cooldown_start_frac=args.cooldown_start_frac,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    setup_auto_logging(cfg.auto_log_path)
    result = run_v2a(cfg)
    if is_rank0():
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
