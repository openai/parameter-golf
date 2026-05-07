from __future__ import annotations

import glob
import io
import json
import math
import os
import random
import copy
import subprocess
import sys
import time
from datetime import timedelta
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

BYTE260_PAD_ID = 0
BYTE260_BOS_ID = 1
BYTE260_EOS_ID = 2
BYTE260_UNK_ID = 3
BYTE260_OFFSET = 4
BYTE260_VOCAB_SIZE = 260
BYTE_VOCAB_SIZE = 256


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
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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

        return loss


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


def parse_positive_ints(value: str) -> tuple[int, ...]:
    ints = sorted({int(part) for part in value.split(",") if part.strip()})
    if not ints or any(item <= 0 for item in ints):
        raise ValueError(f"Expected a comma-separated list of positive ints, got {value!r}")
    return tuple(ints)


def parse_checkpoint_bytes(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    ints = tuple(int(part) for part in value.split(",") if part.strip())
    if any(item <= 0 for item in ints):
        raise ValueError(f"Expected positive checkpoint byte counts, got {value!r}")
    return ints


def parse_multiscale_groups(value: str, patch_size: int) -> tuple[int, ...]:
    groups = parse_positive_ints(value)
    if patch_size not in groups:
        groups = (patch_size, *groups)
    if any(group % patch_size != 0 for group in groups):
        raise ValueError(f"All MULTISCALE_GROUPS must be multiples of PATCH_SIZE={patch_size}, got {groups}")
    return tuple(sorted(set(groups)))


def rank0_only() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(value: int | float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_reduce_any(value: bool, device: torch.device) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor(1 if value else 0, device=device, dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(int(tensor.item()))


@dataclass
class Hyperparameters:
    run_mode: str = os.environ.get("RUN_MODE", "backbone")
    run_id: str = os.environ.get("RUN_ID", "pure_jepa")
    run_phase: str = os.environ.get("RUN_PHASE", "smoke")
    output_root: str = os.environ.get("OUTPUT_ROOT", ".")
    data_path: str = os.environ.get("DATA_PATH", "data/datasets/fineweb10B_byte260")
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", str(BYTE260_VOCAB_SIZE)))
    pad_id: int = int(os.environ.get("PAD_ID", str(BYTE260_PAD_ID)))
    bos_id: int = int(os.environ.get("BOS_ID", str(BYTE260_BOS_ID)))
    eos_id: int = int(os.environ.get("EOS_ID", str(BYTE260_EOS_ID)))
    unk_id: int = int(os.environ.get("UNK_ID", str(BYTE260_UNK_ID)))
    backbone_kind: str = os.environ.get("BACKBONE_KIND", "transformer_rope_gqa_base")
    objective_kind: str = os.environ.get("OBJECTIVE_KIND", "slot_l2")
    patch_encoder_kind: str = os.environ.get("PATCH_ENCODER_KIND", "mlp_baseline")
    patch_size: int = int(os.environ.get("PATCH_SIZE", "8"))
    num_slots: int = int(os.environ.get("NUM_SLOTS", "4"))
    slot_bytes: int = int(os.environ.get("SLOT_BYTES", "2"))
    byte_embed_dim: int = int(os.environ.get("BYTE_EMBED_DIM", "64"))
    model_dim: int = int(os.environ.get("MODEL_DIM", "512"))
    num_layers: int = int(os.environ.get("NUM_LAYERS", "4"))
    num_heads: int = int(os.environ.get("NUM_HEADS", "8"))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", "0"))
    ff_mult: int = int(os.environ.get("FF_MULT", "3"))
    patch_encoder_layers: int = int(os.environ.get("PATCH_ENCODER_LAYERS", "2"))
    patch_encoder_heads: int = int(os.environ.get("PATCH_ENCODER_HEADS", "4"))
    patch_encoder_ff_mult: int = int(os.environ.get("PATCH_ENCODER_FF_MULT", "2"))
    rope_base: float = float(os.environ.get("ROPE_BASE", "10000"))
    local_window_size: int = int(os.environ.get("LOCAL_WINDOW_SIZE", "64"))
    conv_kernel_size: int = int(os.environ.get("CONV_KERNEL_SIZE", "5"))
    decoder_hidden: int = int(os.environ.get("DECODER_HIDDEN", "512"))
    decoder_layers: int = int(os.environ.get("DECODER_LAYERS", "2"))
    decoder_num_heads: int = int(os.environ.get("DECODER_HEADS", "8"))
    decoder_num_kv_heads: int = int(os.environ.get("DECODER_NUM_KV_HEADS", "4"))
    decoder_ff_mult: int = int(os.environ.get("DECODER_FF_MULT", "2"))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", "4096"))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", "131072"))
    train_shards: int = int(os.environ.get("TRAIN_SHARDS", "10"))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", "131072"))
    val_max_seqs: int = int(os.environ.get("VAL_MAX_SEQS", "256"))
    final_val_max_seqs: int = int(os.environ.get("FINAL_VAL_MAX_SEQS", os.environ.get("VAL_MAX_SEQS", "256")))
    iterations: int = int(os.environ.get("ITERATIONS", "2000"))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "0"))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", "250"))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", "50"))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", "0"))
    lr: float = float(os.environ.get("LR", "3e-4"))
    min_lr_ratio: float = float(os.environ.get("MIN_LR_RATIO", "0.1"))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", "0.01"))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", "1.0"))
    seed: int = int(os.environ.get("SEED", "42"))
    predict_horizons: tuple[int, ...] = field(
        default_factory=lambda: parse_positive_ints(os.environ.get("PREDICT_HORIZONS", "1"))
    )
    jepa_weight: float = float(os.environ.get("JEPA_WEIGHT", "1.0"))
    sigreg_weight: float = float(os.environ.get("SIGREG_WEIGHT", "0.01"))
    patch_summary_weight: float = float(os.environ.get("PATCH_SUMMARY_WEIGHT", "0.1"))
    masked_context_prob: float = float(os.environ.get("MASKED_CONTEXT_PROB", "0.15"))
    ema_decay: float = float(os.environ.get("EMA_DECAY", "0.99"))
    vicreg_var_weight: float = float(os.environ.get("VICREG_VAR_WEIGHT", "1.0"))
    vicreg_cov_weight: float = float(os.environ.get("VICREG_COV_WEIGHT", "0.04"))
    multiscale_groups: tuple[int, ...] = field(init=False)
    checkpoint_bytes: tuple[int, ...] = field(
        default_factory=lambda: parse_checkpoint_bytes(os.environ.get("CHECKPOINT_BYTES", ""))
    )
    stop_after_last_checkpoint: bool = env_flag("STOP_AFTER_LAST_CHECKPOINT", False)
    probe_kind: str = os.environ.get("PROBE_KIND", "cheap")
    probe_checkpoint: str = os.environ.get("PROBE_CHECKPOINT", "")
    probe_detach_backbone: bool = env_flag("PROBE_DETACH_BACKBONE", True)
    probe_val_mode: str = os.environ.get("PROBE_VAL_MODE", "proxy")
    probe_train_batch_tokens: int = int(os.environ.get("PROBE_TRAIN_BATCH_TOKENS", "131072"))
    probe_train_shards: int = int(os.environ.get("PROBE_TRAIN_SHARDS", os.environ.get("TRAIN_SHARDS", "10")))
    probe_iterations: int = int(os.environ.get("PROBE_ITERATIONS", "1000"))
    probe_max_wallclock_seconds: float = float(os.environ.get("PROBE_MAX_WALLCLOCK_SECONDS", "0"))
    probe_val_loss_every: int = int(os.environ.get("PROBE_VAL_LOSS_EVERY", "100"))
    probe_train_log_every: int = int(os.environ.get("PROBE_TRAIN_LOG_EVERY", "50"))
    probe_lr: float = float(os.environ.get("PROBE_LR", "5e-4"))
    probe_weight_decay: float = float(os.environ.get("PROBE_WEIGHT_DECAY", "0.01"))
    probe_grad_clip_norm: float = float(os.environ.get("PROBE_GRAD_CLIP_NORM", "1.0"))
    probe_warmup_steps: int = int(os.environ.get("PROBE_WARMUP_STEPS", "0"))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", os.environ.get("LR", "3e-4")))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", "0.95"))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", "5"))
    self_test: bool = env_flag("SELF_TEST", False)

    def __post_init__(self) -> None:
        if self.vocab_size != BYTE260_VOCAB_SIZE:
            raise ValueError(f"Expected VOCAB_SIZE=260, got {self.vocab_size}")
        if self.patch_size <= 0:
            raise ValueError("PATCH_SIZE must be positive")
        if self.train_batch_tokens % self.train_seq_len != 0:
            raise ValueError("TRAIN_BATCH_TOKENS must be divisible by TRAIN_SEQ_LEN")
        if self.val_batch_size % self.train_seq_len != 0:
            raise ValueError("VAL_BATCH_SIZE must be divisible by TRAIN_SEQ_LEN")
        if self.probe_train_batch_tokens % self.train_seq_len != 0:
            raise ValueError("PROBE_TRAIN_BATCH_TOKENS must be divisible by TRAIN_SEQ_LEN")
        if self.byte_embed_dim <= 0 or self.model_dim <= 0 or self.decoder_hidden <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.backbone_kind not in {
            "transformer_rope_gqa_base",
            "transformer_rope_gqa_convstem",
            "transformer_rope_gqa_localglobal",
        }:
            raise ValueError(f"Unsupported BACKBONE_KIND={self.backbone_kind}")
        if self.objective_kind not in {
            "slot_l2",
            "slot_cosine",
            "slot_vicreg",
            "slot_ema_teacher",
            "masked_slot_jepa",
        }:
            raise ValueError(f"Unsupported OBJECTIVE_KIND={self.objective_kind}")
        if self.patch_encoder_kind not in {
            "mlp_baseline",
            "patch_transformer",
            "latent_queries",
            "conv_patch",
        }:
            raise ValueError(f"Unsupported PATCH_ENCODER_KIND={self.patch_encoder_kind}")
        if self.probe_kind not in {"cheap", "strong"}:
            raise ValueError(f"Unsupported PROBE_KIND={self.probe_kind}")
        if self.run_mode not in {"backbone", "probe"} and not self.self_test:
            raise ValueError(f"Unsupported RUN_MODE={self.run_mode}")
        if not (0.0 < self.min_lr_ratio <= 1.0):
            raise ValueError("MIN_LR_RATIO must be in (0, 1]")
        if self.patch_size != self.num_slots * self.slot_bytes:
            raise ValueError("PATCH_SIZE must equal NUM_SLOTS * SLOT_BYTES")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        if self.model_dim % self.patch_encoder_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by PATCH_ENCODER_HEADS")
        if self.num_kv_heads <= 0:
            self.num_kv_heads = max(1, self.num_heads // 2)
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("NUM_HEADS must be divisible by NUM_KV_HEADS")
        if self.num_layers <= 0:
            raise ValueError("NUM_LAYERS must be positive")
        if self.decoder_hidden % self.decoder_num_heads != 0:
            raise ValueError("DECODER_HIDDEN must be divisible by DECODER_HEADS")
        if self.decoder_num_heads % self.decoder_num_kv_heads != 0:
            raise ValueError("DECODER_HEADS must be divisible by DECODER_NUM_KV_HEADS")
        if self.patch_encoder_layers <= 0:
            raise ValueError("PATCH_ENCODER_LAYERS must be positive")
        if 1 not in self.predict_horizons:
            self.predict_horizons = tuple(sorted({1, *self.predict_horizons}))
        self.multiscale_groups = parse_multiscale_groups(os.environ.get("MULTISCALE_GROUPS", str(self.patch_size)), self.patch_size)

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def max_patches(self) -> int:
        return math.ceil(self.train_seq_len / self.patch_size)


def select_data_files(pattern: str, max_shards: int, rank: int = 0, world_size: int = 1) -> list[Path]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if max_shards > 0:
        files = files[:max_shards]
    if not files:
        raise FileNotFoundError(f"No files found for pattern={pattern!r} max_shards={max_shards}")
    if world_size <= 1:
        return files
    rank_files = files[rank::world_size]
    if rank_files:
        return rank_files
    return [files[rank % len(files)]]


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, files: list[Path]):
        self.files = files
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            take = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + take])
            self.pos += take
            remaining -= take
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class TokenLoader:
    def __init__(self, files: list[Path]):
        self.files = files
        self.stream = TokenStream(files)

    def next_batch(self, global_tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
        if global_tokens % seq_len != 0:
            raise ValueError("global_tokens must be divisible by seq_len")
        tokens = self.stream.take(global_tokens + 1)
        x = tokens[:-1].reshape(-1, seq_len).to(dtype=torch.int64)
        y = tokens[1:].reshape(-1, seq_len).to(dtype=torch.int64)
        return x, y


def load_validation_tokens(pattern: str, seq_len: int, max_seqs: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern={pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if max_seqs > 0:
        usable = min(usable, max_seqs * seq_len)
    if usable <= 0:
        raise ValueError(f"Validation split too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1].to(dtype=torch.int64)


def count_payload_bytes(tokens: Tensor) -> int:
    return int((tokens >= BYTE260_OFFSET).sum().item())


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


def bpb_from_nats(loss_nats: float) -> float:
    return loss_nats / math.log(2.0)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(dtype=x.dtype)
        return x * norm * self.weight


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class PatchSlotEncoder(nn.Module):
    def __init__(self, patch_size: int, num_slots: int, slot_bytes: int, byte_embed_dim: int, model_dim: int):
        super().__init__()
        flat_dim = patch_size * byte_embed_dim
        slot_dim = slot_bytes * byte_embed_dim
        self.num_slots = num_slots
        self.summary_in = nn.Linear(flat_dim, model_dim * 2)
        self.summary_out = nn.Linear(model_dim, model_dim)
        self.slot_in = nn.Linear(slot_dim, model_dim * 2)
        self.slot_out = nn.Linear(model_dim, model_dim)
        self.summary_norm = RMSNorm(model_dim)
        self.slot_norm = RMSNorm(model_dim)

    def _gated_proj(self, x: Tensor, in_proj: nn.Linear, out_proj: nn.Linear, norm: RMSNorm) -> Tensor:
        gate, value = in_proj(x).chunk(2, dim=-1)
        hidden = F.silu(gate) * value
        return norm(out_proj(hidden))

    def forward(self, patch_emb: Tensor) -> tuple[Tensor, Tensor]:
        batch, num_patches, patch_size, embed_dim = patch_emb.shape
        flat_patch = patch_emb.reshape(batch, num_patches, patch_size * embed_dim)
        summary = self._gated_proj(flat_patch.float(), self.summary_in, self.summary_out, self.summary_norm)
        slot_views = patch_emb.reshape(batch, num_patches, self.num_slots, -1)
        slots = self._gated_proj(slot_views.float(), self.slot_in, self.slot_out, self.slot_norm)
        return summary, slots


class PatchTokenBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_mult: int):
        super().__init__()
        self.attn_norm = RMSNorm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.ffn_norm = RMSNorm(model_dim)
        self.ffn = SwiGLU(model_dim, ff_mult)
        self.attn_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
        self.ffn_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.ffn_scale.to(dtype=x.dtype)[None, None, :] * self.ffn(self.ffn_norm(x))
        return x


class PatchTransformerEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_slots: int,
        byte_embed_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_mult: int,
    ):
        super().__init__()
        total_tokens = 1 + num_slots + patch_size
        self.num_slots = num_slots
        self.byte_proj = nn.Linear(byte_embed_dim, model_dim)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.slot_tokens = nn.Parameter(torch.zeros(1, num_slots, model_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, total_tokens, model_dim) * 0.02)
        self.blocks = nn.ModuleList([PatchTokenBlock(model_dim, num_heads, ff_mult) for _ in range(num_layers)])
        self.final_norm = RMSNorm(model_dim)

    def forward(self, patch_emb: Tensor) -> tuple[Tensor, Tensor]:
        batch, num_patches, patch_size, _ = patch_emb.shape
        byte_tokens = self.byte_proj(patch_emb.float())
        summary = self.summary_token.expand(batch * num_patches, -1, -1)
        slots = self.slot_tokens.expand(batch * num_patches, -1, -1)
        byte_tokens = byte_tokens.reshape(batch * num_patches, patch_size, -1)
        tokens = torch.cat([summary, slots, byte_tokens], dim=1)
        tokens = tokens + self.pos_emb[:, : tokens.size(1)].to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens).reshape(batch, num_patches, tokens.size(1), -1)
        return tokens[:, :, 0], tokens[:, :, 1 : 1 + self.num_slots]


class LatentQueryEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_slots: int,
        byte_embed_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_mult: int,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.byte_proj = nn.Linear(byte_embed_dim, model_dim)
        self.byte_pos = nn.Parameter(torch.randn(1, patch_size, model_dim) * 0.02)
        self.query_tokens = nn.Parameter(torch.randn(1, 1 + num_slots, model_dim) * 0.02)
        self.cross_q_norm = RMSNorm(model_dim)
        self.cross_kv_norm = RMSNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.cross_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([PatchTokenBlock(model_dim, num_heads, ff_mult) for _ in range(num_layers)])
        self.final_norm = RMSNorm(model_dim)

    def forward(self, patch_emb: Tensor) -> tuple[Tensor, Tensor]:
        batch, num_patches, patch_size, _ = patch_emb.shape
        byte_tokens = self.byte_proj(patch_emb.float()).reshape(batch * num_patches, patch_size, -1)
        byte_tokens = byte_tokens + self.byte_pos[:, :patch_size].to(dtype=byte_tokens.dtype)
        queries = self.query_tokens.expand(batch * num_patches, -1, -1)
        cross_out, _ = self.cross_attn(
            self.cross_q_norm(queries),
            self.cross_kv_norm(byte_tokens),
            self.cross_kv_norm(byte_tokens),
            need_weights=False,
        )
        tokens = queries + self.cross_scale.to(dtype=queries.dtype)[None, None, :] * cross_out
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens).reshape(batch, num_patches, 1 + self.num_slots, -1)
        return tokens[:, :, 0], tokens[:, :, 1:]


class ConvPatchEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_slots: int,
        slot_bytes: int,
        byte_embed_dim: int,
        model_dim: int,
        kernel_size: int,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_bytes = slot_bytes
        self.byte_proj = nn.Linear(byte_embed_dim, model_dim)
        self.blocks = nn.ModuleList(
            [CausalDepthwiseConvStem(model_dim, kernel_size) for _ in range(2)]
        )
        self.post_norm = RMSNorm(model_dim)
        self.summary_proj = nn.Linear(model_dim, model_dim)
        self.summary_norm = RMSNorm(model_dim)
        self.slot_proj = nn.Linear(model_dim, model_dim)
        self.slot_norm = RMSNorm(model_dim)

    def forward(self, patch_emb: Tensor) -> tuple[Tensor, Tensor]:
        batch, num_patches, patch_size, _ = patch_emb.shape
        h = self.byte_proj(patch_emb.float()).reshape(batch * num_patches, patch_size, -1)
        for block in self.blocks:
            h = block(h)
        h = self.post_norm(h).reshape(batch, num_patches, patch_size, -1)
        summary = self.summary_norm(self.summary_proj(h.mean(dim=2)))
        slots = h.reshape(batch, num_patches, self.num_slots, self.slot_bytes, -1).mean(dim=3)
        slots = self.slot_norm(self.slot_proj(slots))
        return summary, slots


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / max(knots - 1, 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, latents: Tensor) -> Tensor:
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        if latents.ndim != 3:
            raise ValueError(f"SIGReg expects (B, T, D) or (N, D), got {tuple(latents.shape)}")
        if latents.size(1) <= 1:
            return latents.new_zeros(())
        proj = torch.randn(latents.size(-1), self.num_proj, device=latents.device, dtype=latents.dtype)
        proj = proj / proj.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6)
        t = self.t.to(device=latents.device, dtype=latents.dtype)
        phi = self.phi.to(device=latents.device, dtype=latents.dtype)
        weights = self.weights.to(device=latents.device, dtype=latents.dtype)
        x_t = (latents @ proj).unsqueeze(-1) * t
        err = (x_t.cos().mean(dim=1) - phi).square()
        err = err + x_t.sin().mean(dim=1).square()
        statistic = (err @ weights) * latents.size(1)
        return statistic.mean()


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
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        local_window_size: int = 0,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        if num_heads % num_kv_heads != 0:
            raise ValueError("NUM_HEADS must be divisible by NUM_KV_HEADS")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = CastedLinear(model_dim, model_dim, bias=False)
        self.k_proj = CastedLinear(model_dim, kv_dim, bias=False)
        self.v_proj = CastedLinear(model_dim, kv_dim, bias=False)
        self.out_proj = CastedLinear(model_dim, model_dim, bias=False)
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.local_window_size = local_window_size
        self._mask_cache: dict[tuple[int, str], Tensor] = {}

    def _local_mask(self, seqlen: int, device: torch.device) -> Tensor | None:
        if self.local_window_size <= 0:
            return None
        key = (seqlen, str(device))
        if key not in self._mask_cache:
            idx = torch.arange(seqlen, device=device)
            future = idx[None, :] > idx[:, None]
            too_old = idx[None, :] < (idx[:, None] - self.local_window_size + 1)
            mask = torch.zeros((seqlen, seqlen), device=device, dtype=torch.float32)
            mask.masked_fill_(future | too_old, float("-inf"))
            self._mask_cache[key] = mask.view(1, 1, seqlen, seqlen)
        return self._mask_cache[key]

    def forward(self, x: Tensor) -> Tensor:
        batch, seqlen, _ = x.shape
        q = self.q_proj(x).reshape(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        attn_mask = self._local_mask(seqlen, x.device)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None if attn_mask is None else attn_mask.to(dtype=q.dtype),
            is_causal=attn_mask is None,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(batch, seqlen, self.model_dim)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, model_dim: int, ff_mult: int):
        super().__init__()
        hidden = model_dim * ff_mult
        self.in_proj = CastedLinear(model_dim, hidden * 2, bias=False)
        self.out_proj = CastedLinear(hidden, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        ff_mult: int,
        rope_base: float,
        local_window_size: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(model_dim)
        self.attn = CausalSelfAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_base=rope_base,
            local_window_size=local_window_size,
        )
        self.ffn_norm = RMSNorm(model_dim)
        self.ffn = SwiGLU(model_dim, ff_mult)
        self.attn_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
        self.ffn_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.ffn_scale.to(dtype=x.dtype)[None, None, :] * self.ffn(self.ffn_norm(x))
        return x


class CausalDepthwiseConvStem(nn.Module):
    def __init__(self, model_dim: int, kernel_size: int):
        super().__init__()
        self.norm = RMSNorm(model_dim)
        self.depthwise = nn.Conv1d(model_dim, model_dim, kernel_size, groups=model_dim, bias=False)
        self.pointwise = nn.Conv1d(model_dim, model_dim, 1, bias=False)
        self.pad = kernel_size - 1
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x).transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        h = self.pointwise(self.depthwise(h)).transpose(1, 2)
        return x + self.scale.to(dtype=x.dtype) * h


class TransformerBackbone(nn.Module):
    def __init__(self, args: Hyperparameters, variant: str):
        super().__init__()
        self.variant = variant
        self.stem = CausalDepthwiseConvStem(args.model_dim, args.conv_kernel_size) if variant == "transformer_rope_gqa_convstem" else None
        local_layers = args.num_layers // 2 if variant == "transformer_rope_gqa_localglobal" else 0
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim=args.model_dim,
                    num_heads=args.num_heads,
                    num_kv_heads=args.num_kv_heads,
                    ff_mult=args.ff_mult,
                    rope_base=args.rope_base,
                    local_window_size=args.local_window_size if idx < local_layers else 0,
                )
                for idx in range(args.num_layers)
            ]
        )
        self.final_norm = RMSNorm(args.model_dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.stem is not None:
            x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class ExplicitScaleProjector(nn.Module):
    def __init__(self, input_groups: int, model_dim: int, num_slots: int):
        super().__init__()
        in_dim = input_groups * (1 + num_slots) * model_dim
        out_dim = (1 + num_slots) * model_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.out_dim = out_dim
        self.model_dim = model_dim
        self.num_slots = num_slots

    def forward(self, summary: Tensor, slots: Tensor) -> tuple[Tensor, Tensor]:
        batch, coarse_steps, groups, dim = summary.shape
        flat = torch.cat(
            [
                summary.reshape(batch, coarse_steps, groups * dim),
                slots.reshape(batch, coarse_steps, groups * self.num_slots * dim),
            ],
            dim=-1,
        )
        out = self.net(flat)
        coarse_summary = out[..., : self.model_dim]
        coarse_slots = out[..., self.model_dim :].reshape(batch, coarse_steps, self.num_slots, self.model_dim)
        return coarse_summary, coarse_slots


class ScaleHorizonPredictor(nn.Module):
    def __init__(self, model_dim: int, num_slots: int, scale_values: tuple[int, ...], horizon_values: tuple[int, ...]):
        super().__init__()
        self.scale_values = scale_values
        self.horizon_values = horizon_values
        self.num_slots = num_slots
        self.norm = RMSNorm(model_dim)
        self.scale_emb = nn.Embedding(len(scale_values), model_dim)
        self.horizon_emb = nn.Embedding(len(horizon_values), model_dim)
        self.net = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim * (1 + num_slots)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context_states: Tensor, scale_index: int, horizon_index: int) -> tuple[Tensor, Tensor]:
        scale_bias = self.scale_emb.weight[scale_index].view(1, 1, -1)
        horizon_bias = self.horizon_emb.weight[horizon_index].view(1, 1, -1)
        cond = self.norm(context_states + scale_bias + horizon_bias)
        out = self.net(cond)
        pred_summary = context_states + out[..., : context_states.size(-1)]
        pred_slots = out[..., context_states.size(-1) :].reshape(
            context_states.size(0),
            context_states.size(1),
            self.num_slots,
            context_states.size(-1),
        )
        return pred_summary, pred_slots


def build_backbone_module(args: Hyperparameters) -> nn.Module:
    if args.backbone_kind in {
        "transformer_rope_gqa_base",
        "transformer_rope_gqa_convstem",
        "transformer_rope_gqa_localglobal",
    }:
        return TransformerBackbone(args, args.backbone_kind)
    raise ValueError(f"Unsupported backbone kind {args.backbone_kind}")


def build_patch_encoder(args: Hyperparameters) -> nn.Module:
    if args.patch_encoder_kind == "mlp_baseline":
        return PatchSlotEncoder(
            patch_size=args.patch_size,
            num_slots=args.num_slots,
            slot_bytes=args.slot_bytes,
            byte_embed_dim=args.byte_embed_dim,
            model_dim=args.model_dim,
        )
    if args.patch_encoder_kind == "patch_transformer":
        return PatchTransformerEncoder(
            patch_size=args.patch_size,
            num_slots=args.num_slots,
            byte_embed_dim=args.byte_embed_dim,
            model_dim=args.model_dim,
            num_heads=args.patch_encoder_heads,
            num_layers=args.patch_encoder_layers,
            ff_mult=args.patch_encoder_ff_mult,
        )
    if args.patch_encoder_kind == "latent_queries":
        return LatentQueryEncoder(
            patch_size=args.patch_size,
            num_slots=args.num_slots,
            byte_embed_dim=args.byte_embed_dim,
            model_dim=args.model_dim,
            num_heads=args.patch_encoder_heads,
            num_layers=args.patch_encoder_layers,
            ff_mult=args.patch_encoder_ff_mult,
        )
    if args.patch_encoder_kind == "conv_patch":
        return ConvPatchEncoder(
            patch_size=args.patch_size,
            num_slots=args.num_slots,
            slot_bytes=args.slot_bytes,
            byte_embed_dim=args.byte_embed_dim,
            model_dim=args.model_dim,
            kernel_size=args.conv_kernel_size,
        )
    raise ValueError(f"Unsupported patch encoder kind {args.patch_encoder_kind}")


@dataclass
class FeatureBatch:
    features: Tensor
    prev_patches: Tensor
    target_patches: Tensor
    byte_mask: Tensor
    full_patch_mask: Tensor


class PureJEPAByteBackbone(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.byte_emb = nn.Embedding(args.vocab_size, args.byte_embed_dim)
        self.patch_encoder = build_patch_encoder(args)
        self.target_byte_emb = copy.deepcopy(self.byte_emb) if args.objective_kind == "slot_ema_teacher" else None
        self.target_encoder = copy.deepcopy(self.patch_encoder) if args.objective_kind == "slot_ema_teacher" else None
        if self.target_byte_emb is not None:
            for param in self.target_byte_emb.parameters():
                param.requires_grad = False
        if self.target_encoder is not None:
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        self.patch_bos = nn.Parameter(torch.zeros(1, 1, args.model_dim))
        self.context_mask_token = nn.Parameter(torch.zeros(1, 1, args.model_dim))
        self.context_model = build_backbone_module(args)
        self.predictor = ScaleHorizonPredictor(args.model_dim, args.num_slots, args.multiscale_groups, args.predict_horizons)
        self.scale_projectors = nn.ModuleDict(
            {
                str(group): ExplicitScaleProjector(group // args.patch_size, args.model_dim, args.num_slots)
                for group in args.multiscale_groups
                if group > args.patch_size
            }
        )
        self.sigreg = SIGReg()

    def _prepare_patch_batch(self, input_ids: Tensor, target_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        valid_positions = torch.ones_like(target_ids, dtype=torch.bool)
        pad_len = (-target_ids.size(1)) % self.args.patch_size
        if pad_len > 0:
            pad_tokens = target_ids.new_full((target_ids.size(0), pad_len), self.args.pad_id)
            pad_mask = torch.zeros((target_ids.size(0), pad_len), dtype=torch.bool, device=target_ids.device)
            target_ids = torch.cat([target_ids, pad_tokens], dim=1)
            valid_positions = torch.cat([valid_positions, pad_mask], dim=1)
        prev_seq = torch.cat([input_ids[:, :1], target_ids[:, :-1]], dim=1)
        patch_shape = (target_ids.size(0), target_ids.size(1) // self.args.patch_size, self.args.patch_size)
        patches = target_ids.reshape(patch_shape)
        prev_patches = prev_seq.reshape(patch_shape)
        valid_patch_positions = valid_positions.reshape(patch_shape)
        full_patch_mask = valid_patch_positions.all(dim=-1)
        return patches, prev_patches, valid_patch_positions, full_patch_mask

    def _encode_patches(self, patches: Tensor) -> tuple[Tensor, Tensor]:
        patch_emb = self.byte_emb(patches)
        return self.patch_encoder(patch_emb.float())

    def _encode_targets(self, patches: Tensor, online_summary: Tensor, online_slots: Tensor) -> tuple[Tensor, Tensor]:
        if self.target_encoder is None or self.target_byte_emb is None:
            return online_summary.detach(), online_slots.detach()
        with torch.no_grad():
            target_summary, target_slots = self.target_encoder(self.target_byte_emb(patches).float())
        return target_summary, target_slots

    def _context_states(self, patch_summary: Tensor, apply_mask: bool) -> Tensor:
        bos = self.patch_bos.expand(patch_summary.size(0), 1, -1)
        context_inputs = torch.cat([bos, patch_summary[:, :-1]], dim=1)
        if apply_mask and self.training and self.args.masked_context_prob > 0.0:
            keep = torch.rand(context_inputs.shape[:2], device=context_inputs.device) >= self.args.masked_context_prob
            keep[:, 0] = True
            context_inputs = torch.where(keep.unsqueeze(-1), context_inputs, self.context_mask_token.expand_as(context_inputs))
        return self.context_model(context_inputs)

    def _coarse_targets(
        self,
        target_summary: Tensor,
        target_slots: Tensor,
        full_patch_mask: Tensor,
        group_bytes: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        factor = group_bytes // self.args.patch_size
        usable_patches = (target_summary.size(1) // factor) * factor
        if usable_patches <= 0:
            empty_summary = target_summary[:, :0]
            empty_slots = target_slots[:, :0]
            empty_mask = full_patch_mask[:, :0]
            starts = torch.arange(0, 0, device=target_summary.device)
            return empty_summary, empty_slots, empty_mask, starts
        if factor == 1:
            starts = torch.arange(0, usable_patches, factor, device=target_summary.device)
            return (
                target_summary[:, :usable_patches],
                target_slots[:, :usable_patches],
                full_patch_mask[:, :usable_patches],
                starts,
            )
        grouped_summary = target_summary[:, :usable_patches].reshape(
            target_summary.size(0), -1, factor, target_summary.size(-1)
        )
        grouped_slots = target_slots[:, :usable_patches].reshape(
            target_slots.size(0), -1, factor, self.args.num_slots, target_slots.size(-1)
        )
        coarse_summary, coarse_slots = self.scale_projectors[str(group_bytes)](grouped_summary, grouped_slots)
        coarse_mask = full_patch_mask[:, :usable_patches].reshape(full_patch_mask.size(0), -1, factor).all(dim=-1)
        starts = torch.arange(0, usable_patches, factor, device=target_summary.device)
        return coarse_summary, coarse_slots, coarse_mask, starts

    def _summary_aux_loss(self, pred_summary: Tensor, target_summary: Tensor, mask: Tensor) -> Tensor:
        per_patch = (pred_summary.float() - target_summary.float()).square().mean(dim=-1)
        return masked_mean(per_patch, mask)

    def _slot_prediction_loss(self, pred_slots: Tensor, target_slots: Tensor, mask: Tensor) -> Tensor:
        flat_pred = pred_slots.float().reshape(pred_slots.size(0), pred_slots.size(1), -1)
        flat_target = target_slots.float().reshape(target_slots.size(0), target_slots.size(1), -1)
        if self.args.objective_kind in {"slot_l2", "slot_ema_teacher", "masked_slot_jepa"}:
            per_patch = (flat_pred - flat_target).square().mean(dim=-1)
            return masked_mean(per_patch, mask)
        if self.args.objective_kind == "slot_cosine":
            cos = F.cosine_similarity(flat_pred, flat_target, dim=-1)
            return masked_mean(1.0 - cos, mask)
        if self.args.objective_kind == "slot_vicreg":
            base = masked_mean((flat_pred - flat_target).square().mean(dim=-1), mask)
            valid = mask.reshape(-1)
            pred_valid = flat_pred.reshape(-1, flat_pred.size(-1))[valid]
            if pred_valid.size(0) < 2:
                return base
            std = torch.sqrt(pred_valid.var(dim=0, unbiased=False) + 1e-4)
            var_penalty = F.relu(1.0 - std).mean()
            centered = pred_valid - pred_valid.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / max(pred_valid.size(0) - 1, 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            cov_penalty = off_diag.square().mean()
            return base + self.args.vicreg_var_weight * var_penalty + self.args.vicreg_cov_weight * cov_penalty
        raise ValueError(f"Unsupported OBJECTIVE_KIND={self.args.objective_kind}")

    def _combined_latents(self, summary: Tensor, slots: Tensor) -> Tensor:
        return torch.cat([summary, slots.reshape(summary.size(0), summary.size(1), -1)], dim=-1)

    @torch.no_grad()
    def update_ema(self) -> None:
        if self.target_encoder is None or self.target_byte_emb is None:
            return
        decay = self.args.ema_decay
        for target_param, online_param in zip(self.target_byte_emb.parameters(), self.byte_emb.parameters()):
            target_param.lerp_(online_param, 1.0 - decay)
        for target_param, online_param in zip(self.target_encoder.parameters(), self.patch_encoder.parameters()):
            target_param.lerp_(online_param, 1.0 - decay)

    def extract_backbone_state(self, input_ids: Tensor, target_ids: Tensor, apply_context_mask: bool = False) -> dict[str, Tensor]:
        patches, prev_patches, valid_patch_positions, full_patch_mask = self._prepare_patch_batch(input_ids, target_ids)
        patch_summary, patch_slots = self._encode_patches(patches)
        target_summary, target_slots = self._encode_targets(patches, patch_summary, patch_slots)
        context_states = self._context_states(
            patch_summary,
            apply_mask=apply_context_mask and self.args.objective_kind == "masked_slot_jepa",
        )
        byte_mask = valid_patch_positions & (patches >= BYTE260_OFFSET)
        pred_summary, pred_slots = self.predictor(context_states, 0, self.args.predict_horizons.index(1))
        features = torch.cat([context_states, pred_summary, pred_slots.reshape(pred_slots.size(0), pred_slots.size(1), -1)], dim=-1)
        return {
            "patches": patches,
            "prev_patches": prev_patches,
            "byte_mask": byte_mask,
            "full_patch_mask": full_patch_mask,
            "patch_summary": patch_summary,
            "patch_slots": patch_slots,
            "target_summary": target_summary,
            "target_slots": target_slots,
            "context_states": context_states,
            "pred_summary": pred_summary,
            "pred_slots": pred_slots,
            "features": features,
        }

    def compute_losses(self, input_ids: Tensor, target_ids: Tensor) -> dict[str, Tensor]:
        state = self.extract_backbone_state(input_ids, target_ids, apply_context_mask=True)
        context_states = state["context_states"]
        target_summary = state["target_summary"]
        target_slots = state["target_slots"]
        full_patch_mask = state["full_patch_mask"]

        scale_losses: list[Tensor] = []
        for scale_index, group_bytes in enumerate(self.args.multiscale_groups):
            scale_summary, scale_slots, coarse_mask, coarse_starts = self._coarse_targets(
                target_summary,
                target_slots,
                full_patch_mask,
                group_bytes,
            )
            if scale_summary.size(1) == 0:
                continue
            coarse_context = context_states[:, coarse_starts]
            for horizon_index, horizon in enumerate(self.args.predict_horizons):
                if horizon > scale_summary.size(1):
                    continue
                pred_len = scale_summary.size(1) - horizon + 1
                pred_input = coarse_context[:, :pred_len]
                target_summary_h = scale_summary[:, horizon - 1 :]
                target_slots_h = scale_slots[:, horizon - 1 :]
                mask = coarse_mask[:, horizon - 1 :]
                if not torch.any(mask):
                    continue
                pred_summary, pred_slots = self.predictor(pred_input, scale_index, horizon_index)
                slot_loss = self._slot_prediction_loss(pred_slots, target_slots_h, mask)
                summary_loss = self._summary_aux_loss(pred_summary, target_summary_h, mask)
                scale_losses.append(slot_loss + self.args.patch_summary_weight * summary_loss)
        jepa_loss = torch.stack(scale_losses).mean() if scale_losses else context_states.new_zeros(())

        valid_lengths = state["full_patch_mask"].sum(dim=1)
        max_valid = int(valid_lengths.min().item()) if valid_lengths.numel() else 0
        combined = self._combined_latents(state["patch_summary"], state["patch_slots"])
        sigreg_latents = combined[:, :max_valid].float() if max_valid > 1 else combined[:, :0].float()
        sigreg_loss = self.sigreg(sigreg_latents) if self.args.sigreg_weight > 0.0 else jepa_loss.new_zeros(())
        total_loss = self.args.jepa_weight * jepa_loss + self.args.sigreg_weight * sigreg_loss
        return {
            "loss": total_loss,
            "jepa_loss": jepa_loss,
            "sigreg_loss": sigreg_loss,
        }

    def extract_probe_features(self, input_ids: Tensor, target_ids: Tensor) -> FeatureBatch:
        state = self.extract_backbone_state(input_ids, target_ids)
        return FeatureBatch(
            features=state["features"],
            prev_patches=state["prev_patches"],
            target_patches=state["patches"],
            byte_mask=state["byte_mask"],
            full_patch_mask=state["full_patch_mask"],
        )

    def export_checkpoint(self) -> dict[str, Tensor]:
        return self.state_dict()

    def load_export_checkpoint(self, state_dict: dict[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(f"Checkpoint load mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")


class CheapProbe(nn.Module):
    def __init__(self, feature_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.norm = RMSNorm(feature_dim)
        self.out = nn.Linear(feature_dim, patch_size * BYTE_VOCAB_SIZE)

    def forward(self, features: Tensor) -> Tensor:
        logits = self.out(self.norm(features))
        return logits.view(features.size(0), features.size(1), self.patch_size, BYTE_VOCAB_SIZE)


class StrongProbe(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        patch_size: int,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        ff_mult: int,
        rope_base: float,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.byte_emb = nn.Embedding(vocab_size, hidden_dim)
        self.cond_proj = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim=hidden_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    ff_mult=ff_mult,
                    rope_base=rope_base,
                    local_window_size=0,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, BYTE_VOCAB_SIZE, bias=False)

    def forward(self, features: Tensor, prev_patches: Tensor) -> Tensor:
        batch_size, num_patches, patch_size = prev_patches.shape
        decoder_in = self.byte_emb(prev_patches).reshape(batch_size * num_patches, patch_size, -1)
        cond = self.cond_proj(features).reshape(batch_size * num_patches, 1, -1)
        x = decoder_in + cond
        for block in self.blocks:
            x = block(x)
        logits = self.out(self.final_norm(x))
        return logits.reshape(batch_size, num_patches, patch_size, BYTE_VOCAB_SIZE)


def build_probe(args: Hyperparameters, feature_dim: int) -> nn.Module:
    if args.probe_kind == "cheap":
        return CheapProbe(feature_dim, args.patch_size)
    if args.probe_kind == "strong":
        return StrongProbe(
            feature_dim=feature_dim,
            patch_size=args.patch_size,
            vocab_size=args.vocab_size,
            hidden_dim=args.decoder_hidden,
            num_layers=args.decoder_layers,
            num_heads=args.decoder_num_heads,
            num_kv_heads=args.decoder_num_kv_heads,
            ff_mult=args.decoder_ff_mult,
            rope_base=args.rope_base,
        )
    raise ValueError(f"Unsupported probe kind {args.probe_kind}")


def probe_loss_from_logits(logits: Tensor, target_patches: Tensor, byte_mask: Tensor) -> Tensor:
    targets = (target_patches - BYTE260_OFFSET).clamp_min(0)
    losses = F.cross_entropy(logits.reshape(-1, BYTE_VOCAB_SIZE), targets.reshape(-1), reduction="none").reshape_as(target_patches)
    return masked_mean(losses, byte_mask)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class OptimizerBundle:
    adamw: torch.optim.Optimizer | None
    muon: Muon | None = None
    muon_params: tuple[nn.Parameter, ...] = ()


def _decay_muon_params(params: Iterable[nn.Parameter], lr: float, weight_decay: float) -> None:
    if weight_decay <= 0.0:
        return
    scale = 1.0 - lr * weight_decay
    for param in params:
        param.data.mul_(scale)


def _is_muon_param(name: str, param: nn.Parameter) -> bool:
    return param.ndim == 2 and name.startswith("context_model.")


def build_backbone_optimizer(model: PureJEPAByteBackbone, args: Hyperparameters) -> OptimizerBundle:
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_muon_param(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    adamw = torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    muon = None
    if muon_params:
        muon = Muon(muon_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    return OptimizerBundle(adamw=adamw, muon=muon, muon_params=tuple(muon_params))


def build_probe_optimizer(parameters: Iterable[nn.Parameter], lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))


def lr_for_step(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(warmup_steps, 1))
    progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_backbone_optimizer_lr(bundle: OptimizerBundle, model_lr: float, matrix_lr: float) -> None:
    if bundle.adamw is not None:
        for group in bundle.adamw.param_groups:
            group["lr"] = model_lr
    if bundle.muon is not None:
        for group in bundle.muon.param_groups:
            group["lr"] = matrix_lr


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def maybe_init_distributed(device: torch.device) -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=1800))
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    return local_rank, world_size


def close_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def prepare_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return device
    return torch.device("cpu")


def eval_backbone(args: Hyperparameters, model: PureJEPAByteBackbone, device: torch.device, val_tokens: Tensor) -> tuple[float, float]:
    seq_len = args.train_seq_len
    batch_seqs = max(args.val_batch_size // seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    jepa_sum = 0.0
    sigreg_sum = 0.0
    batch_count = 0

    model.eval()
    with torch.inference_mode():
        for seq_start in range(0, total_seqs, batch_seqs):
            seq_end = min(seq_start + batch_seqs, total_seqs)
            raw_start = seq_start * seq_len
            raw_end = seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end]
            x = local[:-1].reshape(-1, seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
            y = local[1:].reshape(-1, seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                losses = model.compute_losses(x, y)
            jepa_sum += float(losses["jepa_loss"].item())
            sigreg_sum += float(losses["sigreg_loss"].item())
            batch_count += 1
    model.train()
    if batch_count == 0:
        return 0.0, 0.0
    return jepa_sum / batch_count, sigreg_sum / batch_count


def eval_probe(
    args: Hyperparameters,
    backbone: PureJEPAByteBackbone,
    probe: nn.Module,
    device: torch.device,
    val_tokens: Tensor,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    batch_seqs = max(args.val_batch_size // seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    loss_sum = 0.0
    byte_count = 0

    backbone.eval()
    probe.eval()
    with torch.inference_mode():
        for seq_start in range(0, total_seqs, batch_seqs):
            seq_end = min(seq_start + batch_seqs, total_seqs)
            raw_start = seq_start * seq_len
            raw_end = seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end]
            x = local[:-1].reshape(-1, seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
            y = local[1:].reshape(-1, seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                batch = backbone.extract_probe_features(x, y)
                features = batch.features.detach() if args.probe_detach_backbone else batch.features
                if args.probe_kind == "cheap":
                    logits = probe(features)
                else:
                    logits = probe(features, batch.prev_patches)
                batch_loss = probe_loss_from_logits(logits, batch.target_patches, batch.byte_mask)
            batch_bytes = int(batch.byte_mask.sum().item())
            loss_sum += float(batch_loss.item()) * batch_bytes
            byte_count += batch_bytes
    backbone.train()
    probe.train()
    val_loss = loss_sum / max(byte_count, 1)
    return val_loss, bpb_from_nats(val_loss)


def output_root_path(output_root: str) -> Path:
    return Path(output_root)


def logs_root(output_root: str) -> Path:
    return output_root_path(output_root) / "logs"


def artifacts_root(output_root: str) -> Path:
    return output_root_path(output_root) / "artifacts"


def checkpoint_dir_for(run_id: str, output_root: str) -> Path:
    return artifacts_root(output_root) / run_id / "checkpoints"


def backbone_summary_path_for(run_id: str, output_root: str) -> Path:
    return artifacts_root(output_root) / run_id / "backbone_run.json"


def probe_result_path_for(run_id: str, checkpoint_label: str, probe_kind: str, output_root: str) -> Path:
    return artifacts_root(output_root) / run_id / "probe_results" / f"{checkpoint_label}__{probe_kind}.json"


def log_factory(run_id: str, output_root: str) -> tuple[Path, callable]:
    logfile = logs_root(output_root) / f"{run_id}.txt"
    logfile.parent.mkdir(parents=True, exist_ok=True)

    def log0(msg: str, console: bool = True) -> None:
        if console and rank0_only():
            print(msg)
        if rank0_only():
            with logfile.open("a", encoding="utf-8") as f:
                print(msg, file=f)

    return logfile, log0


def run_backbone(args: Hyperparameters) -> None:
    device = prepare_device()
    local_rank, world_size = maybe_init_distributed(device)
    del local_rank
    random.seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    np.random.seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    torch.manual_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    logfile, log0 = log_factory(args.run_id, args.output_root)
    code = Path(__file__).read_text(encoding="utf-8")
    if rank0_only():
        log0(code, console=False)
        log0("=" * 100, console=False)
        log0(f"Running Python {sys.version}", console=False)
        log0(f"Running PyTorch {torch.__version__}", console=False)
        if device.type == "cuda":
            log0(
                subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
                console=False,
            )
        log0("=" * 100, console=False)
        log0(
            "mode:backbone "
            f"run_id:{args.run_id} phase:{args.run_phase} backbone_kind:{args.backbone_kind} "
            f"patch_encoder_kind:{args.patch_encoder_kind} "
            f"patch_size:{args.patch_size} predict_horizons:{','.join(map(str, args.predict_horizons))} "
            f"multiscale_groups:{','.join(map(str, args.multiscale_groups))} train_shards:{args.train_shards}"
        )

    train_files = select_data_files(args.train_files, args.train_shards, dist.get_rank() if dist.is_initialized() else 0, world_size)
    actual_train_files = len(train_files)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, args.val_max_seqs) if rank0_only() else None
    model = PureJEPAByteBackbone(args).to(device)
    model_for_train: nn.Module = model
    if dist.is_available() and dist.is_initialized():
        model_for_train = nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = build_backbone_optimizer(model, args)
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    batch_seqs = args.train_batch_tokens // args.train_seq_len
    if rank0_only():
        dataset_dir = Path(args.data_path).resolve()
        log0(f"train_loader:dataset:{dataset_dir.name} train_shards_local:{actual_train_files} train_shards_requested:{args.train_shards}")
        log0(f"val_loader:pattern:{args.val_files} periodic_tokens:{0 if val_tokens is None else val_tokens.numel() - 1}")
        log0(f"model_params:{num_params}")
        log0(
            f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} batch_seqs:{batch_seqs} "
            f"iterations:{args.iterations} world_size:{world_size}"
        )

    train_loader = TokenLoader(train_files)
    checkpoint_dir = checkpoint_dir_for(args.run_id, args.output_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_time_ms = 0.0
    t0 = time.perf_counter()
    train_bytes_seen = 0.0
    checkpoint_targets = list(args.checkpoint_bytes)
    checkpoint_records: list[dict[str, object]] = []
    train_points: list[dict[str, object]] = []
    val_points: list[dict[str, object]] = []
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def one_step(do_backward: bool) -> tuple[dict[str, Tensor], int]:
        x_cpu, y_cpu = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len)
        local_bytes = count_payload_bytes(y_cpu)
        x = x_cpu.to(device=device, dtype=torch.int64, non_blocking=True)
        y = y_cpu.to(device=device, dtype=torch.int64, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses = model_for_train.module.compute_losses(x, y) if isinstance(model_for_train, nn.parallel.DistributedDataParallel) else model_for_train.compute_losses(x, y)
        if do_backward:
            if optimizer.adamw is not None:
                optimizer.adamw.zero_grad(set_to_none=True)
            if optimizer.muon is not None:
                optimizer.muon.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            if optimizer.adamw is not None:
                optimizer.adamw.step()
            if optimizer.muon is not None:
                _decay_muon_params(optimizer.muon_params, args.matrix_lr, args.weight_decay)
                optimizer.muon.step()
            (model_for_train.module if isinstance(model_for_train, nn.parallel.DistributedDataParallel) else model_for_train).update_ema()
        return {key: value.detach() for key, value in losses.items()}, local_bytes

    def save_checkpoint(label: str, step: int, source: str, val_jepa_loss: float | None, val_sigreg_loss: float | None) -> None:
        if not rank0_only():
            return
        path = checkpoint_dir / f"{label}.pt"
        payload = {
            "run_id": args.run_id,
            "run_phase": args.run_phase,
            "backbone_kind": args.backbone_kind,
            "hyperparameters": asdict(args),
            "model_state_dict": model.export_checkpoint(),
            "step": step,
            "train_bytes_seen": train_bytes_seen,
            "train_time_ms": training_time_ms,
            "model_params": num_params,
            "train_shards_used": args.train_shards,
            "local_train_shards_used": actual_train_files,
            "source": source,
        }
        torch.save(payload, path)
        record = {
            "label": label,
            "path": str(path),
            "step": step,
            "train_bytes_seen": train_bytes_seen,
            "train_time_ms": training_time_ms,
            "val_jepa_loss": val_jepa_loss,
            "val_sigreg_loss": val_sigreg_loss,
            "source": source,
        }
        checkpoint_records.append(record)
        log0(
            f"checkpoint_saved label:{label} step:{step} train_bytes_seen:{int(train_bytes_seen)} "
            f"train_time:{training_time_ms:.0f}ms path:{path}"
        )

    for warmup_step in range(args.warmup_steps):
        lr = lr_for_step(warmup_step, args.iterations, args.lr, args.warmup_steps, args.min_lr_ratio)
        matrix_lr = lr_for_step(warmup_step, args.iterations, args.matrix_lr, args.warmup_steps, args.min_lr_ratio)
        set_backbone_optimizer_lr(optimizer, lr, matrix_lr)
        _, local_bytes = one_step(do_backward=True)
        train_bytes_seen += all_reduce_sum(local_bytes, device)
        if rank0_only():
            log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

    step = 0
    while True:
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        local_timed_out = max_wallclock_ms is not None and step > 0 and approx_training_time_ms >= max_wallclock_ms
        timed_out = all_reduce_any(local_timed_out, device)
        last_step = step >= args.iterations or timed_out
        should_validate = last_step or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0)
        val_jepa_loss = None
        val_sigreg_loss = None
        if should_validate:
            barrier()
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            if rank0_only() and val_tokens is not None:
                val_jepa_loss, val_sigreg_loss = eval_backbone(args, model, device, val_tokens)
                point = {
                    "step": step,
                    "total_steps": args.iterations,
                    "val_jepa_loss": val_jepa_loss,
                    "val_sigreg_loss": val_sigreg_loss,
                    "train_time_ms": training_time_ms,
                    "step_avg_ms": training_time_ms / max(step, 1),
                    "train_bytes_seen": train_bytes_seen,
                }
                val_points.append(point)
                log0(
                    f"step:{step}/{args.iterations} val_jepa_loss:{val_jepa_loss:.4f} val_sigreg_loss:{val_sigreg_loss:.4f} "
                    f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms "
                    f"train_bytes_seen:{int(train_bytes_seen)}"
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        while checkpoint_targets and train_bytes_seen >= checkpoint_targets[0]:
            label = f"ckpt_{checkpoint_targets[0]}"
            save_checkpoint(label, step, "threshold", val_jepa_loss, val_sigreg_loss)
            checkpoint_targets.pop(0)
        if args.stop_after_last_checkpoint and not checkpoint_targets:
            last_step = True

        if last_step:
            break

        lr = lr_for_step(step, args.iterations, args.lr, args.warmup_steps, args.min_lr_ratio)
        matrix_lr = lr_for_step(step, args.iterations, args.matrix_lr, args.warmup_steps, args.min_lr_ratio)
        set_backbone_optimizer_lr(optimizer, lr, matrix_lr)
        losses, local_bytes = one_step(do_backward=True)
        step += 1
        train_bytes_seen += all_reduce_sum(local_bytes, device)
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if rank0_only() and args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            point = {
                "step": step,
                "total_steps": args.iterations,
                "train_loss": float(losses["loss"].item()),
                "jepa_loss": float(losses["jepa_loss"].item()),
                "sigreg_loss": float(losses["sigreg_loss"].item()),
                "train_time_ms": approx_training_time_ms,
                "step_avg_ms": approx_training_time_ms / max(step, 1),
                "train_bytes_seen": train_bytes_seen,
            }
            train_points.append(point)
            log0(
                f"step:{step}/{args.iterations} train_loss:{losses['loss'].item():.4f} "
                f"jepa_loss:{losses['jepa_loss'].item():.4f} sigreg_loss:{losses['sigreg_loss'].item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"train_bytes_seen:{int(train_bytes_seen)}"
            )

    barrier()
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_time_ms += 1000.0 * (time.perf_counter() - t0)
    if rank0_only():
        save_checkpoint("final", step, "final", val_points[-1]["val_jepa_loss"] if val_points else None, val_points[-1]["val_sigreg_loss"] if val_points else None)
        if device.type == "cuda":
            log0(
                f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
                f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
            )
        else:
            log0("peak memory allocated: 0 MiB reserved: 0 MiB")
        summary = {
            "run_mode": "backbone",
            "run_id": args.run_id,
            "run_phase": args.run_phase,
            "backbone_kind": args.backbone_kind,
            "patch_encoder_kind": args.patch_encoder_kind,
            "config": asdict(args),
            "model_params": num_params,
            "train_shards_used": args.train_shards,
            "local_train_shards_used": actual_train_files,
            "gpu_count": world_size,
            "elapsed_ms": training_time_ms,
            "elapsed_gpu_hours": (training_time_ms / 3_600_000.0) * world_size,
            "final_step": step,
            "train_bytes_seen": train_bytes_seen,
            "train_points": train_points,
            "val_points": val_points,
            "checkpoint_records": checkpoint_records,
            "peak_alloc_mib": torch.cuda.max_memory_allocated() // 1024 // 1024 if device.type == "cuda" else 0,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() // 1024 // 1024 if device.type == "cuda" else 0,
            "log_path": str(logfile),
        }
        save_json(backbone_summary_path_for(args.run_id, args.output_root), summary)
    close_distributed()


def load_backbone_checkpoint(
    path: Path,
    device: torch.device,
    probe_args: Hyperparameters,
) -> tuple[Hyperparameters, PureJEPAByteBackbone, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    args = Hyperparameters()
    for key, value in dict(payload["hyperparameters"]).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.run_mode = "probe"
    args.probe_kind = probe_args.probe_kind
    args.probe_checkpoint = str(path)
    args.probe_detach_backbone = probe_args.probe_detach_backbone
    args.probe_val_mode = probe_args.probe_val_mode
    args.probe_train_batch_tokens = probe_args.probe_train_batch_tokens
    args.probe_train_shards = probe_args.probe_train_shards
    args.probe_iterations = probe_args.probe_iterations
    args.probe_max_wallclock_seconds = probe_args.probe_max_wallclock_seconds
    args.probe_val_loss_every = probe_args.probe_val_loss_every
    args.probe_train_log_every = probe_args.probe_train_log_every
    args.probe_lr = probe_args.probe_lr
    args.probe_weight_decay = probe_args.probe_weight_decay
    args.probe_grad_clip_norm = probe_args.probe_grad_clip_norm
    args.decoder_hidden = probe_args.decoder_hidden
    args.decoder_layers = probe_args.decoder_layers
    args.decoder_num_heads = probe_args.decoder_num_heads
    args.decoder_num_kv_heads = probe_args.decoder_num_kv_heads
    args.decoder_ff_mult = probe_args.decoder_ff_mult
    model = PureJEPAByteBackbone(args).to(device)
    model.load_export_checkpoint(payload["model_state_dict"])
    model.eval()
    return args, model, payload


def run_probe(args: Hyperparameters) -> None:
    if not args.probe_checkpoint:
        raise ValueError("RUN_MODE=probe requires PROBE_CHECKPOINT")
    device = prepare_device()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = Path(args.probe_checkpoint).resolve()
    checkpoint_args, backbone, checkpoint_payload = load_backbone_checkpoint(checkpoint_path, device, args)
    for param in backbone.parameters():
        param.requires_grad = False

    probe_args = args
    # Keep the run-time probe config, but inherit structural settings from the checkpoint.
    probe_args.patch_size = checkpoint_args.patch_size
    probe_args.train_seq_len = checkpoint_args.train_seq_len
    probe_args.val_batch_size = checkpoint_args.val_batch_size
    probe_args.model_dim = checkpoint_args.model_dim
    probe_args.byte_embed_dim = checkpoint_args.byte_embed_dim
    probe_args.vocab_size = checkpoint_args.vocab_size
    probe_args.decoder_layers = probe_args.decoder_layers

    checkpoint_label = checkpoint_path.stem
    probe_run_id = f"{args.run_id}__{checkpoint_label}__{args.probe_kind}"
    logfile, log0 = log_factory(probe_run_id, args.output_root)
    if rank0_only():
        log0(
            "mode:probe "
            f"run_id:{args.run_id} checkpoint_label:{checkpoint_label} probe_kind:{args.probe_kind} "
            f"probe_val_mode:{args.probe_val_mode} detach_backbone:{args.probe_detach_backbone}"
        )

    train_files = select_data_files(args.train_files, args.probe_train_shards)
    val_max_seqs = args.final_val_max_seqs if args.probe_val_mode == "full" else args.val_max_seqs
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, val_max_seqs)
    train_loader = TokenLoader(train_files)
    feature_dim = checkpoint_args.model_dim * (2 + checkpoint_args.num_slots)
    probe = build_probe(probe_args, feature_dim).to(device)
    optimizer = build_probe_optimizer(probe.parameters(), args.probe_lr, args.probe_weight_decay)
    num_probe_params = sum(param.numel() for param in probe.parameters() if param.requires_grad)
    batch_seqs = args.probe_train_batch_tokens // args.train_seq_len
    training_time_ms = 0.0
    t0 = time.perf_counter()
    train_bytes_seen = 0
    train_points: list[dict[str, object]] = []
    val_points: list[dict[str, object]] = []
    max_wallclock_ms = 1000.0 * args.probe_max_wallclock_seconds if args.probe_max_wallclock_seconds > 0 else None

    def probe_step(do_backward: bool) -> tuple[Tensor, int]:
        x_cpu, y_cpu = train_loader.next_batch(args.probe_train_batch_tokens, args.train_seq_len)
        local_bytes = count_payload_bytes(y_cpu)
        x = x_cpu.to(device=device, dtype=torch.int64, non_blocking=True)
        y = y_cpu.to(device=device, dtype=torch.int64, non_blocking=True)
        with torch.no_grad():
            batch = backbone.extract_probe_features(x, y)
        features = batch.features.detach() if args.probe_detach_backbone else batch.features
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if args.probe_kind == "cheap":
                logits = probe(features)
            else:
                logits = probe(features, batch.prev_patches)
            loss = probe_loss_from_logits(logits, batch.target_patches, batch.byte_mask)
        if do_backward:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.probe_grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), args.probe_grad_clip_norm)
            optimizer.step()
        return loss.detach(), local_bytes

    for warmup_step in range(args.probe_warmup_steps):
        lr = lr_for_step(warmup_step, args.probe_iterations, args.probe_lr, args.probe_warmup_steps, args.min_lr_ratio)
        set_optimizer_lr(optimizer, lr)
        _, local_bytes = probe_step(do_backward=True)
        train_bytes_seen += local_bytes
        log0(f"probe_warmup_step:{warmup_step + 1}/{args.probe_warmup_steps}")

    step = 0
    while True:
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        timed_out = max_wallclock_ms is not None and step > 0 and approx_training_time_ms >= max_wallclock_ms
        last_step = step >= args.probe_iterations or timed_out
        should_validate = last_step or (args.probe_val_loss_every > 0 and step > 0 and step % args.probe_val_loss_every == 0)
        if should_validate:
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_probe(args, backbone, probe, device, val_tokens)
            point = {
                "step": step,
                "total_steps": args.probe_iterations,
                "val_loss": val_loss,
                "val_bpb": val_bpb,
                "train_time_ms": training_time_ms,
                "step_avg_ms": training_time_ms / max(step, 1),
                "train_bytes_seen": train_bytes_seen,
            }
            val_points.append(point)
            log0(
                f"step:{step}/{args.probe_iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms "
                f"train_bytes_seen:{train_bytes_seen}"
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        lr = lr_for_step(step, args.probe_iterations, args.probe_lr, args.probe_warmup_steps, args.min_lr_ratio)
        set_optimizer_lr(optimizer, lr)
        train_loss, local_bytes = probe_step(do_backward=True)
        step += 1
        train_bytes_seen += local_bytes
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.probe_train_log_every > 0 and (step <= 10 or step % args.probe_train_log_every == 0):
            point = {
                "step": step,
                "total_steps": args.probe_iterations,
                "train_loss": float(train_loss.item()),
                "train_time_ms": approx_training_time_ms,
                "step_avg_ms": approx_training_time_ms / max(step, 1),
                "train_bytes_seen": train_bytes_seen,
            }
            train_points.append(point)
            log0(
                f"step:{step}/{args.probe_iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"train_bytes_seen:{train_bytes_seen}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated() // 1024 // 1024
        peak_reserved = torch.cuda.max_memory_reserved() // 1024 // 1024
    else:
        peak_alloc = 0
        peak_reserved = 0
    training_time_ms += 1000.0 * (time.perf_counter() - t0)
    result = {
        "run_mode": "probe",
        "run_id": args.run_id,
        "probe_run_id": probe_run_id,
        "probe_kind": args.probe_kind,
        "probe_val_mode": args.probe_val_mode,
        "probe_detach_backbone": args.probe_detach_backbone,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": checkpoint_label,
        "checkpoint_step": checkpoint_payload["step"],
        "checkpoint_train_bytes": checkpoint_payload["train_bytes_seen"],
        "backbone_kind": checkpoint_payload["backbone_kind"],
        "probe_config": asdict(args),
        "probe_model_params": num_probe_params,
        "elapsed_ms": training_time_ms,
        "elapsed_gpu_hours": training_time_ms / 3_600_000.0,
        "train_bytes_seen": train_bytes_seen,
        "train_points": train_points,
        "val_points": val_points,
        "best_val_bpb": min(point["val_bpb"] for point in val_points) if val_points else float("nan"),
        "final_val": val_points[-1] if val_points else None,
        "peak_alloc_mib": peak_alloc,
        "peak_reserved_mib": peak_reserved,
        "log_path": str(logfile),
    }
    result_path = probe_result_path_for(args.run_id, checkpoint_label, args.probe_kind, args.output_root)
    save_json(result_path, result)
    log0(f"probe_result_json:{result_path}")


def synthetic_batch(batch: int, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
    x = torch.randint(BYTE260_OFFSET, BYTE260_VOCAB_SIZE, (batch, seq_len), device=device)
    y = torch.randint(BYTE260_OFFSET, BYTE260_VOCAB_SIZE, (batch, seq_len), device=device)
    return x, y


def run_self_tests() -> None:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    base_env = dict(os.environ)
    try:
        os.environ.update(
            {
                "RUN_MODE": "probe",
                "BACKBONE_KIND": "transformer_rope_gqa_base",
                "OBJECTIVE_KIND": "slot_ema_teacher",
                "PATCH_SIZE": "8",
                "NUM_SLOTS": "4",
                "SLOT_BYTES": "2",
                "MODEL_DIM": "64",
                "NUM_LAYERS": "2",
                "NUM_HEADS": "4",
                "NUM_KV_HEADS": "2",
                "FF_MULT": "2",
                "TRAIN_SEQ_LEN": "32",
                "TRAIN_BATCH_TOKENS": "32",
                "VAL_BATCH_SIZE": "32",
                "PREDICT_HORIZONS": "1,2",
                "MULTISCALE_GROUPS": "8,32",
            }
        )
        args = Hyperparameters()
        backbone = PureJEPAByteBackbone(args).to(device)
        x, y = synthetic_batch(2, args.train_seq_len, device)
        feature_batch = backbone.extract_probe_features(x, y)
        feature_dim = args.model_dim * (2 + args.num_slots)
        cheap_probe = CheapProbe(feature_dim, args.patch_size).to(device)
        logits = cheap_probe(feature_batch.features.detach())
        loss = probe_loss_from_logits(logits, feature_batch.target_patches, feature_batch.byte_mask)
        loss.backward()
        backbone_grad_total = 0.0
        for param in backbone.parameters():
            if param.grad is not None:
                backbone_grad_total += float(param.grad.abs().sum().item())
        if backbone_grad_total != 0.0:
            raise AssertionError(f"Expected no backbone gradients from detached cheap probe, got {backbone_grad_total}")

        leak_x, leak_y = synthetic_batch(1, 32, device)
        original = backbone.extract_probe_features(leak_x, leak_y).features.detach()
        mutated_y = leak_y.clone()
        patch_start = args.patch_size
        patch_end = patch_start + args.patch_size
        mutated_y[:, patch_start:patch_end] = torch.randint(BYTE260_OFFSET, BYTE260_VOCAB_SIZE, (1, args.patch_size), device=device)
        mutated = backbone.extract_probe_features(leak_x, mutated_y).features.detach()
        patch_index = 1
        if not torch.allclose(original[:, patch_index], mutated[:, patch_index], atol=1e-6, rtol=0.0):
            raise AssertionError("Current-patch features changed after mutating the target patch; leakage check failed")

        strong_probe = StrongProbe(
            feature_dim=feature_dim,
            patch_size=args.patch_size,
            vocab_size=args.vocab_size,
            hidden_dim=args.decoder_hidden,
            num_layers=max(args.decoder_layers, 2),
            num_heads=args.decoder_num_heads,
            num_kv_heads=args.decoder_num_kv_heads,
            ff_mult=args.decoder_ff_mult,
            rope_base=args.rope_base,
        ).to(device)
        strong_logits = strong_probe(feature_batch.features.detach(), feature_batch.prev_patches)
        strong_loss = probe_loss_from_logits(strong_logits, feature_batch.target_patches, feature_batch.byte_mask)
        strong_loss.backward()
        probe_backbone_grad_total = 0.0
        for param in backbone.parameters():
            if param.grad is not None:
                probe_backbone_grad_total += float(param.grad.abs().sum().item())
        if probe_backbone_grad_total != 0.0:
            raise AssertionError(f"Expected no backbone gradients from detached strong probe, got {probe_backbone_grad_total}")

        optimizer = build_backbone_optimizer(backbone, args)
        losses = backbone.compute_losses(x, y)
        if optimizer.adamw is not None:
            optimizer.adamw.zero_grad(set_to_none=True)
        if optimizer.muon is not None:
            optimizer.muon.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if optimizer.adamw is not None:
            optimizer.adamw.step()
        if optimizer.muon is not None:
            _decay_muon_params(optimizer.muon_params, args.matrix_lr, 0.0)
            optimizer.muon.step()
        before_update = None
        if backbone.target_encoder is not None:
            before_update = [param.detach().clone() for param in backbone.target_encoder.parameters()]
        backbone.update_ema()
        if backbone.target_encoder is not None and before_update is not None:
            moved = 0.0
            for old, new in zip(before_update, backbone.target_encoder.parameters()):
                moved += float((old - new.detach()).abs().sum().item())
            if moved <= 0.0:
                raise AssertionError("EMA teacher parameters did not update")
        raw = io.BytesIO()
        torch.save({"model_state_dict": backbone.export_checkpoint(), "hyperparameters": asdict(args)}, raw)
        raw.seek(0)
        loaded = torch.load(raw, map_location=device)
        restored = PureJEPAByteBackbone(args).to(device)
        restored.load_export_checkpoint(loaded["model_state_dict"])
        feature_a = backbone.extract_probe_features(x, y).features.detach()
        feature_b = restored.extract_probe_features(x, y).features.detach()
        if not torch.allclose(feature_a, feature_b, atol=1e-6, rtol=0.0):
            raise AssertionError("Checkpoint replay changed extracted features")

        print("self_test:ok")
    finally:
        os.environ.clear()
        os.environ.update(base_env)


def main() -> None:
    args = Hyperparameters()
    if args.self_test:
        run_self_tests()
        return
    if args.run_mode == "backbone":
        run_backbone(args)
        return
    if args.run_mode == "probe":
        run_probe(args)
        return
    raise ValueError(f"Unsupported RUN_MODE={args.run_mode}")


if __name__ == "__main__":
    main()
