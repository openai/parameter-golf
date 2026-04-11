from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


REPO_ROOT = Path(__file__).resolve().parent
SHARD_HEADER_INTS = 256
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
BYTE_OFFSET = 4
BYTE_VOCAB_SIZE = 256
SPECIAL_TOKEN_COUNT = BYTE_OFFSET
MODEL_VOCAB_SIZE = BYTE_VOCAB_SIZE + SPECIAL_TOKEN_COUNT
SUBMISSION_SIZE_LIMIT_BYTES = 16_000_000
ARTIFACT_NAME = "final_model.int8.ptz"
RAW_MODEL_NAME = "final_model.pt"
SUBMISSION_NAME = "submission.json"


def env_flag(name: str, default: bool) -> bool:
    return bool(int(os.environ.get(name, "1" if default else "0")))


class BaseHyperparameters:
    def __init__(self):
        default_data_path = str(REPO_ROOT / "data" / "datasets" / "fineweb10B_byte260")
        self.data_path = os.environ.get("DATA_PATH", default_data_path)
        self.train_files = os.path.join(self.data_path, "fineweb_train_*.bin")
        self.val_files = os.path.join(self.data_path, "fineweb_val_*.bin")
        self.run_id = os.environ.get("RUN_ID", f"clean_byte_jepa_{uuid.uuid4().hex[:8]}")
        self.seed = int(os.environ.get("SEED", 1337))
        self.model_mode = os.environ.get("MODEL_MODE", "byte_jepa")
        self.byte_data_offset = int(os.environ.get("BYTE_DATA_OFFSET", BYTE_OFFSET))

        self.iterations = int(os.environ.get("ITERATIONS", 20_000))
        self.train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 262_144))
        self.val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 262_144))
        self.train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
        self.val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
        self.train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))
        self.val_max_bytes = int(os.environ.get("VAL_MAX_BYTES", 0))
        self.max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0.0))
        self.grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 1))

        self.vocab_size = int(os.environ.get("VOCAB_SIZE", MODEL_VOCAB_SIZE))
        self.model_dim = int(os.environ.get("MODEL_DIM", 384))
        self.num_layers = int(os.environ.get("NUM_LAYERS", 6))
        self.num_heads = int(os.environ.get("NUM_HEADS", 6))
        self.num_kv_heads = int(os.environ.get("NUM_KV_HEADS", self.num_heads))
        self.mlp_mult = int(os.environ.get("MLP_MULT", 4))
        self.tie_embeddings = env_flag("TIE_EMBEDDINGS", True)
        self.tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.01))
        self.rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
        self.qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
        self.logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
        self.residual_scale_scheme = os.environ.get("RESIDUAL_SCALE_SCHEME", "inv_sqrt_2depth")
        self.attn_residual_scale_init = float(os.environ.get("ATTN_RESIDUAL_SCALE_INIT", 0.0))
        self.mlp_residual_scale_init = float(os.environ.get("MLP_RESIDUAL_SCALE_INIT", 0.0))
        self.enable_compile = env_flag("ENABLE_COMPILE", False)

        self.lr = float(os.environ.get("LR", 3e-4))
        self.min_lr_scale = float(os.environ.get("MIN_LR_SCALE", 0.1))
        self.weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.1))
        self.beta1 = float(os.environ.get("BETA1", 0.9))
        self.beta2 = float(os.environ.get("BETA2", 0.95))
        self.adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
        self.grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
        self.warmup_steps = int(os.environ.get("WARMUP_STEPS", 200))

        self.jepa_enabled = env_flag("JEPA_ENABLED", True)
        self.jepa_arch = os.environ.get("JEPA_ARCH", "lewm_chunk")
        self.latent_weight = float(os.environ.get("LATENT_WEIGHT", 0.2))
        self.reg_weight = float(os.environ.get("REG_WEIGHT", 1.0))
        self.jepa_target_momentum = float(os.environ.get("JEPA_TARGET_MOMENTUM", 0.996))
        self.chunk_size = int(os.environ.get("CHUNK_SIZE", 32))
        self.jepa_latent_dim = int(os.environ.get("JEPA_LATENT_DIM", 256))
        self.jepa_pred_hidden = int(os.environ.get("JEPA_PRED_HIDDEN", 384))
        self.jepa_pred_depth = int(os.environ.get("JEPA_PRED_DEPTH", 2))
        self.jepa_pred_heads = int(os.environ.get("JEPA_PRED_HEADS", 4))
        self.pred_loss_mode = os.environ.get("PRED_LOSS_MODE", "mse")
        self.latent_reg_mode = os.environ.get("LATENT_REG_MODE", "sigreg_cf")
        self.reg_slices = int(os.environ.get("REG_SLICES", 1024))
        self.reg_sigma = float(os.environ.get("REG_SIGMA", 1.0))
        self.sigreg_knots = int(os.environ.get("SIGREG_KNOTS", 17))
        self.sigreg_max_t = float(os.environ.get("SIGREG_MAX_T", 3.0))
        self.hier_latent_enabled = env_flag("HIER_LATENT_ENABLED", False)
        self.slow_chunk_span = int(os.environ.get("SLOW_CHUNK_SPAN", 4))
        self.slow_latent_weight = float(os.environ.get("SLOW_LATENT_WEIGHT", 0.5))

        self.track = os.environ.get("TRACK", "non-record-unlimited-compute-16mb")
        self.artifact_path = Path(os.environ.get("ARTIFACT_PATH", str(Path(__file__).resolve().parent / ARTIFACT_NAME)))
        self.raw_model_path = Path(os.environ.get("RAW_MODEL_PATH", str(Path(__file__).resolve().parent / RAW_MODEL_NAME)))
        self.submission_path = Path(
            os.environ.get("SUBMISSION_PATH", str(Path(__file__).resolve().parent / SUBMISSION_NAME))
        )
        self.log_dir = Path(os.environ.get("LOG_DIR", str(Path(__file__).resolve().parent / "logs")))

        if self.model_mode != "byte_jepa":
            raise ValueError(f"MODEL_MODE must be 'byte_jepa', got {self.model_mode!r}")
        if self.vocab_size != MODEL_VOCAB_SIZE:
            raise ValueError(f"VOCAB_SIZE must be {MODEL_VOCAB_SIZE}, got {self.vocab_size}")
        if self.jepa_arch not in {"lewm_chunk", "ema_aux"}:
            raise ValueError(f"Unsupported JEPA_ARCH={self.jepa_arch!r}")
        if self.pred_loss_mode not in {"mse", "cosine"}:
            raise ValueError(f"Unsupported PRED_LOSS_MODE={self.pred_loss_mode!r}")
        if self.latent_reg_mode not in {"sigreg_cf", "sliced_mmd_gaussian"}:
            raise ValueError(f"Unsupported LATENT_REG_MODE={self.latent_reg_mode!r}")
        if self.chunk_size <= 0:
            raise ValueError(f"CHUNK_SIZE must be positive, got {self.chunk_size}")
        if self.jepa_latent_dim <= 0:
            raise ValueError(f"JEPA_LATENT_DIM must be positive, got {self.jepa_latent_dim}")
        if self.jepa_latent_dim > self.model_dim:
            raise ValueError("JEPA_LATENT_DIM must be <= MODEL_DIM")
        if self.jepa_pred_hidden <= 0:
            raise ValueError(f"JEPA_PRED_HIDDEN must be positive, got {self.jepa_pred_hidden}")
        if self.jepa_pred_depth <= 0:
            raise ValueError(f"JEPA_PRED_DEPTH must be positive, got {self.jepa_pred_depth}")
        if self.jepa_pred_heads <= 0:
            raise ValueError(f"JEPA_PRED_HEADS must be positive, got {self.jepa_pred_heads}")
        if self.reg_slices <= 0:
            raise ValueError(f"REG_SLICES must be positive, got {self.reg_slices}")
        if self.reg_sigma <= 0.0:
            raise ValueError(f"REG_SIGMA must be positive, got {self.reg_sigma}")
        if self.sigreg_knots < 2:
            raise ValueError(f"SIGREG_KNOTS must be >= 2, got {self.sigreg_knots}")
        if self.sigreg_max_t <= 0.0:
            raise ValueError(f"SIGREG_MAX_T must be positive, got {self.sigreg_max_t}")
        if self.slow_chunk_span <= 0:
            raise ValueError(f"SLOW_CHUNK_SPAN must be positive, got {self.slow_chunk_span}")
        if self.slow_latent_weight < 0.0:
            raise ValueError(f"SLOW_LATENT_WEIGHT must be >= 0, got {self.slow_latent_weight}")
        if self.jepa_arch != "lewm_chunk":
            if self.pred_loss_mode != "mse":
                raise ValueError("PRED_LOSS_MODE is only supported for JEPA_ARCH=lewm_chunk")
            if self.latent_reg_mode != "sigreg_cf":
                raise ValueError("LATENT_REG_MODE is only supported for JEPA_ARCH=lewm_chunk")
            if self.hier_latent_enabled:
                raise ValueError("HIER_LATENT_ENABLED is only supported for JEPA_ARCH=lewm_chunk")


def maybe_autocast(device: torch.device, enabled: bool = True):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def load_data_shard(file: Path, byte_offset: int) -> Tensor:
    header_bytes = SHARD_HEADER_INTS * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=SHARD_HEADER_INTS)
    if header.size != SHARD_HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    token_ids = tokens_np.astype(np.int32, copy=False)
    mapped = np.empty_like(token_ids, dtype=np.int32)
    byte_mask = token_ids >= int(byte_offset)
    mapped[byte_mask] = token_ids[byte_mask] - int(byte_offset)
    mapped[~byte_mask] = BYTE_VOCAB_SIZE + token_ids[~byte_mask]
    if mapped.size and (mapped.min() < 0 or mapped.max() >= MODEL_VOCAB_SIZE):
        raise ValueError(f"byte260 shard {file} maps outside [0, {MODEL_VOCAB_SIZE - 1}]")
    return torch.from_numpy(mapped.astype(np.int64, copy=False))


def list_shards(pattern: str) -> list[Path]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files


def load_validation_tokens(pattern: str, seq_len: int, byte_offset: int, max_bytes: int) -> Tensor:
    files = list_shards(pattern)
    tokens = torch.cat([load_data_shard(file, byte_offset) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if max_bytes > 0:
        usable = min(usable, (max_bytes // seq_len) * seq_len)
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len} and VAL_MAX_BYTES={max_bytes}")
    return tokens[: usable + 1]


class TokenStream:
    def __init__(self, pattern: str, byte_offset: int):
        self.files = list_shards(pattern)
        self.byte_offset = byte_offset
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0], self.byte_offset)
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx], self.byte_offset)
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device, byte_offset: int):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern, byte_offset)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if global_tokens % (self.world_size * grad_accum_steps * seq_len) != 0:
            raise ValueError(
                f"TRAIN_BATCH_TOKENS={global_tokens} must be divisible by WORLD_SIZE*GRAD_ACCUM_STEPS*TRAIN_SEQ_LEN"
            )
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def residual_scale_init_for_depth(depth: int, scheme: str, override: float) -> float:
    if override > 0.0:
        return override
    if scheme == "none":
        return 1.0
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    if scheme == "inv_sqrt_depth":
        return depth ** -0.5
    if scheme == "inv_sqrt_2depth":
        return (2.0 * depth) ** -0.5
    raise ValueError(f"Unsupported RESIDUAL_SCALE_SCHEME={scheme!r}")


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for _, param in module.named_parameters():
            if param.ndim < 2 and param.dtype != torch.float32:
                param.data = param.data.float()


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


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
        return self._cos_cached.to(dtype=dtype).clone(), self._sin_cached.to(dtype=dtype).clone()


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
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

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_heads != self.num_kv_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
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
        attn_scale_init: float,
        mlp_scale_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.full((dim,), attn_scale_init, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.full((dim,), mlp_scale_init, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class ByteGPT(nn.Module):
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
        residual_scale_scheme: str,
        attn_residual_scale_init: float,
        mlp_residual_scale_init: float,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        attn_scale_init = residual_scale_init_for_depth(
            num_layers, residual_scale_scheme, attn_residual_scale_init
        )
        mlp_scale_init = residual_scale_init_for_depth(num_layers, residual_scale_scheme, mlp_residual_scale_init)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    attn_scale_init,
                    mlp_scale_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def encode(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[bi](x, x0)
        return self.final_norm(x)

    def logits_from_hidden(self, hidden: Tensor) -> Tensor:
        logits = F.linear(hidden, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(hidden)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


class LatentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            CastedLinear(input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            CastedLinear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LatentSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = CastedLinear(dim, inner_dim * 3, bias=False)
        self.to_out = CastedLinear(inner_dim, dim, bias=False) if inner_dim != dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(x.size(0), x.size(1), self.heads, -1).transpose(1, 2)
        k = k.view(x.size(0), x.size(1), self.heads, -1).transpose(1, 2)
        v = v.view(x.size(0), x.size(1), self.heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.to_out(out)


class LatentFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            CastedLinear(dim, hidden_dim, bias=True),
            nn.GELU(),
            CastedLinear(hidden_dim, dim, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LatentBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int):
        super().__init__()
        self.attn = LatentSelfAttention(dim, heads=heads, dim_head=dim_head)
        self.mlp = LatentFeedForward(dim, mlp_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class LatentTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int, heads: int, mlp_dim: int):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("JEPA_PRED_HIDDEN must be divisible by JEPA_PRED_HEADS")
        self.input_proj = CastedLinear(input_dim, hidden_dim, bias=True) if input_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [LatentBlock(hidden_dim, heads=heads, dim_head=hidden_dim // heads, mlp_dim=mlp_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = CastedLinear(hidden_dim, output_dim, bias=True) if hidden_dim != output_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x)


class LatentARPredictor(nn.Module):
    def __init__(
        self,
        max_steps: int,
        depth: int,
        heads: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        mlp_dim: int,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_steps, input_dim) * (input_dim**-0.5))
        self.transformer = LatentTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        t = x.size(1)
        if t > self.pos_embedding.size(1):
            raise ValueError(f"Latent sequence length {t} exceeds predictor max_steps {self.pos_embedding.size(1)}")
        return self.transformer(x + self.pos_embedding[:, :t, :].to(dtype=x.dtype))


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024, max_t: float = 3.0):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0.0, max_t, knots, dtype=torch.float32)
        dt = max_t / float(knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    def forward(self, proj: Tensor) -> Tensor:
        if proj.ndim != 3:
            raise ValueError(f"SIGReg expects (T, B, D), got {tuple(proj.shape)}")
        if proj.size(-2) < 2:
            return proj.new_zeros(())
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        a = a.div_(a.norm(p=2, dim=0).clamp_min(1e-12))
        x_t = (proj @ a).unsqueeze(-1) * self.t.to(dtype=proj.dtype)
        err = (x_t.cos().mean(-3) - self.phi.to(dtype=proj.dtype)).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights.to(dtype=proj.dtype)) * proj.size(-2)
        return statistic.mean()



def eval_val(
    args: Hyperparameters,
    model: ByteJEPAModel,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_tokens // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_TOKENS must provide at least one sequence per rank; "
            f"got VAL_BATCH_TOKENS={args.val_batch_tokens}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with maybe_autocast(device):
                batch_loss = model.ce_loss(x, y).detach()
            batch_token_count = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * batch_token_count
            token_count += batch_token_count
            byte_count += (y < BYTE_VOCAB_SIZE).to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / torch.clamp_min(token_count, 1.0)
    val_bpb = (loss_sum / math.log(2.0)) / torch.clamp_min(byte_count, 1.0)
    model.train()
    return float(val_loss.item()), float(val_bpb.item())


INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                t = t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE)
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def write_submission_json(
    args: Hyperparameters,
    code_bytes: int,
    artifact_bytes: int,
    val_loss: float,
    val_bpb: float,
    build_meta: dict[str, object],
) -> None:
    payload = {
        "track": args.track,
        "val_loss": float(val_loss),
        "val_bpb": float(val_bpb),
        "bytes_code": int(code_bytes),
        "bytes_model_int8_zlib": int(artifact_bytes),
        "bytes_total": int(code_bytes + artifact_bytes),
        "submission_valid_lzma": int(code_bytes + artifact_bytes <= SUBMISSION_SIZE_LIMIT_BYTES),
        "model_mode": args.model_mode,
        "jepa_enabled": int(args.jepa_enabled),
        "jepa_arch": args.jepa_arch,
        "build_meta": build_meta,
    }
    args.submission_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_optimizer(raw_model: ByteJEPAModel, args: Hyperparameters, device: torch.device) -> torch.optim.Optimizer:
    trainable = [param for param in raw_model.parameters() if param.requires_grad]
    fused = device.type == "cuda"
    return torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=fused,
    )


def lr_for_step(args: Hyperparameters, step: int) -> float:
    if step < args.warmup_steps:
        return args.lr * float(step + 1) / float(max(args.warmup_steps, 1))
    if args.iterations <= args.warmup_steps:
        return args.lr
    progress = (step - args.warmup_steps) / float(max(args.iterations - args.warmup_steps, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_scale + (1.0 - args.min_lr_scale) * cosine)



class Hyperparameters(BaseHyperparameters):
    def __init__(self):
        super().__init__()
        self.run_id = self.run_id.replace("clean_byte_jepa_", "byte_jepa_")
        self.projector_hidden = int(os.environ.get("JEPA_PROJECTOR_HIDDEN", 2048))
        self.context_hidden = int(os.environ.get("JEPA_CONTEXT_HIDDEN", 2048))
        self.pred_dim_head = int(os.environ.get("JEPA_PRED_DIM_HEAD", 64))
        self.pred_dropout = float(os.environ.get("JEPA_PRED_DROPOUT", 0.0))
        self.pred_emb_dropout = float(os.environ.get("JEPA_PRED_EMB_DROPOUT", 0.0))
        if self.projector_hidden <= 0:
            raise ValueError(f"JEPA_PROJECTOR_HIDDEN must be positive, got {self.projector_hidden}")
        if self.context_hidden <= 0:
            raise ValueError(f"JEPA_CONTEXT_HIDDEN must be positive, got {self.context_hidden}")
        if self.pred_dim_head <= 0:
            raise ValueError(f"JEPA_PRED_DIM_HEAD must be positive, got {self.pred_dim_head}")
        if self.pred_dropout < 0.0:
            raise ValueError(f"JEPA_PRED_DROPOUT must be >= 0, got {self.pred_dropout}")
        if self.pred_emb_dropout < 0.0:
            raise ValueError(f"JEPA_PRED_EMB_DROPOUT must be >= 0, got {self.pred_emb_dropout}")


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LeWMSIGReg(torch.nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024, max_t: float = 3.0):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0.0, max_t, knots, dtype=torch.float32)
        dt = max_t / float(knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    def forward(self, proj: Tensor) -> Tensor:
        if proj.ndim != 3:
            raise ValueError(f"SIGReg expects (T, B, D), got {tuple(proj.shape)}")
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        a = a.div_(a.norm(p=2, dim=0).clamp_min(1e-12))
        x_t = (proj @ a).unsqueeze(-1) * self.t.to(dtype=proj.dtype)
        err = (x_t.cos().mean(-3) - self.phi.to(dtype=proj.dtype)).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights.to(dtype=proj.dtype)) * proj.size(-2)
        return statistic.mean()


class LeWMFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LeWMAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        bsz, seqlen, _ = q.shape
        q = q.view(bsz, seqlen, self.heads, -1).transpose(1, 2)
        k = k.view(bsz, seqlen, self.heads, -1).transpose(1, 2)
        v = v.view(bsz, seqlen, self.heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.to_out(out)


class LeWMConditionalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = LeWMAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = LeWMFeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LeWMBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = LeWMAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = LeWMFeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LeWMTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        block_class: type[nn.Module] = LeWMBlock,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        for _ in range(depth):
            self.layers.append(block_class(hidden_dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x) if isinstance(block, LeWMBlock) else block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class LeWMMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn: type[nn.Module] | None = nn.BatchNorm1d,
        act_fn: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LeWMARPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = LeWMTransformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=LeWMConditionalBlock,
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        t = x.size(1)
        x = x + self.pos_embedding[:, :t].to(dtype=x.dtype)
        x = self.dropout(x)
        return self.transformer(x, c)


def apply_sequence_mlp(module: nn.Module, x: Tensor) -> Tensor:
    bsz, steps, dim = x.shape
    y = module(x.reshape(bsz * steps, dim))
    return y.reshape(bsz, steps, -1)


class ByteJEPAModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.student = ByteGPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            residual_scale_scheme=args.residual_scale_scheme,
            attn_residual_scale_init=args.attn_residual_scale_init,
            mlp_residual_scale_init=args.mlp_residual_scale_init,
        )
        self.jepa_arch = args.jepa_arch
        self.jepa_enabled = args.jepa_enabled
        self.max_chunks = (args.train_seq_len + args.chunk_size - 1) // args.chunk_size

        self.latent_bos = nn.Parameter(torch.zeros(args.jepa_latent_dim, dtype=torch.float32))
        self.context_bos = nn.Parameter(torch.zeros(args.jepa_latent_dim, dtype=torch.float32))
        self.sigreg = LeWMSIGReg(knots=args.sigreg_knots, num_proj=args.reg_slices, max_t=args.sigreg_max_t)

        self.projector = LeWMMLP(
            input_dim=args.model_dim,
            hidden_dim=args.projector_hidden,
            output_dim=args.jepa_latent_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.context_encoder = LeWMMLP(
            input_dim=args.model_dim,
            hidden_dim=args.context_hidden,
            output_dim=args.jepa_latent_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.predictor = LeWMARPredictor(
            num_frames=self.max_chunks,
            depth=args.jepa_pred_depth,
            heads=args.jepa_pred_heads,
            mlp_dim=2 * args.jepa_pred_hidden,
            input_dim=args.jepa_latent_dim,
            hidden_dim=args.jepa_pred_hidden,
            output_dim=args.jepa_pred_hidden,
            dim_head=args.pred_dim_head,
            dropout=args.pred_dropout,
            emb_dropout=args.pred_emb_dropout,
        )
        self.pred_proj = LeWMMLP(
            input_dim=args.jepa_pred_hidden,
            hidden_dim=args.projector_hidden,
            output_dim=args.jepa_latent_dim,
            norm_fn=nn.BatchNorm1d,
        )

        if self.jepa_arch == "lewm_chunk":
            self.latent_to_hidden = CastedLinear(args.jepa_latent_dim, args.model_dim, bias=False)
            nn.init.zeros_(self.latent_to_hidden.weight)
            self.target_encoder = None
            self.target_projector = None
        else:
            self.latent_to_hidden = None
            self.target_encoder = copy.deepcopy(self.student)
            self.target_projector = copy.deepcopy(self.projector)
            for param in self.target_encoder.parameters():
                param.requires_grad_(False)
            for param in self.target_projector.parameters():
                param.requires_grad_(False)
            self.target_encoder.eval()
            self.target_projector.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.jepa_arch == "ema_aux":
            self.target_encoder.eval()
            self.target_projector.eval()
        return self

    def update_target_encoder(self, momentum: float) -> None:
        if self.jepa_arch != "ema_aux":
            return
        with torch.no_grad():
            for target, online in zip(self.target_encoder.parameters(), self.student.parameters(), strict=True):
                target.data.lerp_(online.data, 1.0 - momentum)
            for target, online in zip(self.target_projector.parameters(), self.projector.parameters(), strict=True):
                target.data.lerp_(online.data, 1.0 - momentum)
            for target, online in zip(self.target_encoder.buffers(), self.student.buffers(), strict=True):
                target.copy_(online)
            for target, online in zip(self.target_projector.buffers(), self.projector.buffers(), strict=True):
                target.copy_(online)

    def _chunk_pool(self, hidden: Tensor) -> Tensor:
        bsz, seq_len, dim = hidden.shape
        n_chunks = (seq_len + self.args.chunk_size - 1) // self.args.chunk_size
        padded = n_chunks * self.args.chunk_size
        if padded != seq_len:
            hidden = torch.cat([hidden, hidden.new_zeros((bsz, padded - seq_len, dim))], dim=1)
        hidden = hidden.reshape(bsz, n_chunks, self.args.chunk_size, dim)
        if padded == seq_len:
            return hidden.mean(dim=2)
        counts = torch.full((n_chunks,), self.args.chunk_size, device=hidden.device, dtype=hidden.dtype)
        counts[-1] = seq_len - (n_chunks - 1) * self.args.chunk_size
        return hidden.sum(dim=2) / counts.view(1, n_chunks, 1)

    def _broadcast_chunk_tensor(self, chunk_values: Tensor, seq_len: int) -> Tensor:
        out = torch.repeat_interleave(chunk_values, repeats=self.args.chunk_size, dim=1)
        return out[:, :seq_len, :]

    def _shift_with_bos(self, x: Tensor, bos: Tensor) -> Tensor:
        prefix = bos.to(dtype=x.dtype)[None, None, :].expand(x.size(0), 1, -1)
        return torch.cat([prefix, x[:, :-1, :]], dim=1).contiguous()

    def _compute_chunk_streams(self, student_hidden: Tensor) -> tuple[Tensor, Tensor]:
        chunk_summaries = self._chunk_pool(student_hidden)
        chunk_latents = apply_sequence_mlp(self.projector, chunk_summaries)
        context_latents = apply_sequence_mlp(self.context_encoder, chunk_summaries)
        return chunk_latents, context_latents

    def _predict_chunk_latents(self, chunk_latents: Tensor, context_latents: Tensor) -> Tensor:
        x = self._shift_with_bos(chunk_latents, self.latent_bos)
        c = self._shift_with_bos(context_latents, self.context_bos)
        pred_hidden = self.predictor(x, c)
        return apply_sequence_mlp(self.pred_proj, pred_hidden)

    def _latent_stats(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        flat = z.reshape(-1, z.size(-1)).float()
        if flat.size(0) < 2:
            zero = flat.new_zeros(())
            return zero, zero, zero
        mean = flat.mean(dim=0)
        centered = flat - mean
        var = centered.pow(2).mean(dim=0).clamp_min(1e-5)
        normed = centered / var.sqrt()
        cov = (normed.T @ normed) / float(normed.size(0))
        return var.sqrt().mean(), mean.norm() / math.sqrt(mean.numel()), torch.diag(cov).mean()

    def _compose_lewm_hidden(self, student_hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        chunk_latents, context_latents = self._compute_chunk_streams(student_hidden)
        pred_latents = self._predict_chunk_latents(chunk_latents, context_latents)
        latent_hidden = self.latent_to_hidden(pred_latents.to(dtype=student_hidden.dtype))
        fused_hidden = student_hidden + self._broadcast_chunk_tensor(latent_hidden, student_hidden.size(1))
        return fused_hidden, chunk_latents, pred_latents

    def predict_logits(self, input_ids: Tensor) -> Tensor:
        student_hidden = self.student.encode(input_ids)
        if self.jepa_arch == "lewm_chunk":
            fused_hidden, _, _ = self._compose_lewm_hidden(student_hidden)
            return self.student.logits_from_hidden(fused_hidden)
        return self.student.logits_from_hidden(student_hidden)

    def ce_loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.predict_logits(input_ids)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean")

    def export_inference_state(self) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {
            f"student.{name}": tensor.detach().cpu() for name, tensor in self.student.state_dict().items()
        }
        if self.jepa_arch == "lewm_chunk":
            state["latent_bos"] = self.latent_bos.detach().cpu()
            state["context_bos"] = self.context_bos.detach().cpu()
            for prefix, module in (
                ("projector.", self.projector),
                ("context_encoder.", self.context_encoder),
                ("predictor.", self.predictor),
                ("pred_proj.", self.pred_proj),
                ("latent_to_hidden.", self.latent_to_hidden),
            ):
                state.update({f"{prefix}{name}": tensor.detach().cpu() for name, tensor in module.state_dict().items()})
        return state

    def load_inference_state(self, state: dict[str, Tensor]) -> None:
        student_state = {name.removeprefix("student."): tensor for name, tensor in state.items() if name.startswith("student.")}
        self.student.load_state_dict(student_state, strict=True)
        if self.jepa_arch != "lewm_chunk":
            return
        with torch.no_grad():
            self.latent_bos.copy_(state["latent_bos"].to(device=self.latent_bos.device, dtype=self.latent_bos.dtype))
            self.context_bos.copy_(state["context_bos"].to(device=self.context_bos.device, dtype=self.context_bos.dtype))
        for prefix, module in (
            ("projector.", self.projector),
            ("context_encoder.", self.context_encoder),
            ("predictor.", self.predictor),
            ("pred_proj.", self.pred_proj),
            ("latent_to_hidden.", self.latent_to_hidden),
        ):
            module_state = {name.removeprefix(prefix): tensor for name, tensor in state.items() if name.startswith(prefix)}
            module.load_state_dict(module_state, strict=True)

    def compute_losses(self, input_ids: Tensor, target_ids: Tensor):
        student_hidden = self.student.encode(input_ids)
        zero = student_hidden.new_zeros(())

        if self.jepa_arch == "lewm_chunk":
            fused_hidden, chunk_latents, pred_latents = self._compose_lewm_hidden(student_hidden)
            logits = self.student.logits_from_hidden(fused_hidden)
            ce_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean")
            latent_std, latent_mean_norm, latent_cov_trace = self._latent_stats(chunk_latents)
            if not self.jepa_enabled:
                return ce_loss, ce_loss.detach(), zero, zero, latent_std.detach(), latent_mean_norm.detach(), latent_cov_trace.detach()
            pred_loss = F.mse_loss(pred_latents.float(), chunk_latents.detach().float(), reduction="mean")
            reg_loss = self.sigreg(chunk_latents.transpose(0, 1).float())
            total_loss = ce_loss + self.args.latent_weight * (pred_loss + self.args.reg_weight * reg_loss)
            return (
                total_loss,
                ce_loss.detach(),
                pred_loss.detach(),
                reg_loss.detach(),
                latent_std.detach(),
                latent_mean_norm.detach(),
                latent_cov_trace.detach(),
            )

        logits = self.student.logits_from_hidden(student_hidden)
        ce_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean")
        online_chunk_latents, context_latents = self._compute_chunk_streams(student_hidden)
        latent_std, latent_mean_norm, latent_cov_trace = self._latent_stats(online_chunk_latents)
        if not self.jepa_enabled:
            return ce_loss, ce_loss.detach(), zero, zero, latent_std.detach(), latent_mean_norm.detach(), latent_cov_trace.detach()
        pred_latents = self._predict_chunk_latents(online_chunk_latents, context_latents)
        with torch.no_grad():
            target_hidden = self.target_encoder.encode(input_ids)
            target_chunk_summaries = self._chunk_pool(target_hidden)
            target_latents = apply_sequence_mlp(self.target_projector, target_chunk_summaries).detach()
        pred_loss = F.mse_loss(pred_latents.float(), target_latents.float(), reduction="mean")
        total_loss = ce_loss + self.args.latent_weight * pred_loss
        return (
            total_loss,
            ce_loss.detach(),
            pred_loss.detach(),
            zero,
            latent_std.detach(),
            latent_mean_norm.detach(),
            latent_cov_trace.detach(),
        )

    def forward(self, input_ids: Tensor, target_ids: Tensor):
        return self.compute_losses(input_ids, target_ids)


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed:
        if device.type == "cuda":
            dist.init_process_group(backend=backend, device_id=device)
        else:
            dist.init_process_group(backend=backend)
        dist.barrier()
    master_process = rank == 0

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.log_dir.mkdir(parents=True, exist_ok=True)
    logfile = args.log_dir / f"{args.run_id}.txt"

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg, flush=True)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        f"model_mode:{args.model_mode} jepa_arch:{args.jepa_arch} "
        f"jepa_enabled:{int(args.jepa_enabled)} chunk_size:{args.chunk_size}"
    )

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, args.byte_data_offset, args.val_max_bytes)
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device, args.byte_data_offset)
    log0(f"val_bpb:enabled tokenizer_kind=byte260 byte_offset:{args.byte_data_offset}")
    log0(f"val_loader:bytes:{val_tokens.numel() - 1}")

    raw_model = ByteJEPAModel(args).to(device)
    for module in raw_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
        if isinstance(module, Rotary):
            module.inv_freq.data = module.inv_freq.data.float()
    restore_low_dim_params_to_fp32(raw_model)
    if device.type == "cuda":
        raw_model.bfloat16()
        restore_low_dim_params_to_fp32(raw_model)
    if args.enable_compile and hasattr(torch, "compile"):
        raw_model.student = torch.compile(raw_model.student, dynamic=False, fullgraph=False)
    if distributed and device.type == "cuda":
        model: nn.Module = DDP(raw_model, device_ids=[local_rank], broadcast_buffers=False)
    elif distributed:
        model = DDP(raw_model)
    else:
        model = raw_model
    optimizer = build_optimizer(raw_model, args, device)

    log0(
        f"train_batch_tokens:{args.train_batch_tokens} val_batch_tokens:{args.val_batch_tokens} "
        f"train_seq_len:{args.train_seq_len} iterations:{args.iterations}"
    )
    log0(
        f"model_dim:{args.model_dim} num_layers:{args.num_layers} num_heads:{args.num_heads} "
        f"num_kv_heads:{args.num_kv_heads} mlp_mult:{args.mlp_mult}"
    )
    log0(
        f"jepa_latent_dim:{args.jepa_latent_dim} jepa_pred_hidden:{args.jepa_pred_hidden} "
        f"jepa_pred_depth:{args.jepa_pred_depth} jepa_pred_heads:{args.jepa_pred_heads} "
        f"projector_hidden:{args.projector_hidden} context_hidden:{args.context_hidden} "
        f"pred_dim_head:{args.pred_dim_head} latent_weight:{args.latent_weight} reg_weight:{args.reg_weight}"
    )
    if args.jepa_arch == "lewm_chunk":
        log0(
            f"sigreg:enabled reg_slices:{args.reg_slices} "
            f"sigreg_knots:{args.sigreg_knots} sigreg_max_t:{args.sigreg_max_t}"
        )
    if args.jepa_arch == "ema_aux":
        log0(f"ema_target_momentum:{args.jepa_target_momentum}")

    def zero_grad_all() -> None:
        optimizer.zero_grad(set_to_none=True)

    step = 0
    stop_after_step: int | None = None
    training_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, raw_model, rank, world_size, device, args.grad_accum_steps, val_tokens)
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f} "
                f"train_time:{training_time_ms:.0f}ms"
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        zero_grad_all()
        accum_total = torch.zeros((), device=device)
        accum_ce = torch.zeros((), device=device)
        accum_jepa = torch.zeros((), device=device)
        accum_reg = torch.zeros((), device=device)
        accum_std = torch.zeros((), device=device)
        accum_mean = torch.zeros((), device=device)
        accum_cov = torch.zeros((), device=device)

        lr = lr_for_step(args, step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for micro_step in range(args.grad_accum_steps):
            if distributed and isinstance(model, DDP):
                model.require_backward_grad_sync = micro_step == args.grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, args.grad_accum_steps)
            with maybe_autocast(device):
                total_loss, ce_loss, jepa_loss, reg_loss, latent_std, latent_mean_norm, latent_cov_trace = model(x, y)
            scaled_loss = total_loss / float(args.grad_accum_steps)
            scaled_loss.backward()
            accum_total += total_loss.detach()
            accum_ce += ce_loss
            accum_jepa += jepa_loss
            accum_reg += reg_loss
            accum_std += latent_std
            accum_mean += latent_mean_norm
            accum_cov += latent_cov_trace

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip_norm)
        optimizer.step()
        raw_model.update_target_encoder(args.jepa_target_momentum)
        zero_grad_all()

        step += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        latent_std_value = float((accum_std / args.grad_accum_steps).item())
        if args.jepa_enabled and latent_std_value > 0.0 and latent_std_value < 1e-4:
            raise RuntimeError(f"latent collapse detected: latent_std={latent_std_value:.8f}")
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(
                f"step:{step}/{args.iterations} train_loss:{(accum_total / args.grad_accum_steps).item():.6f} "
                f"ce:{(accum_ce / args.grad_accum_steps).item():.6f} "
                f"jepa:{(accum_jepa / args.grad_accum_steps).item():.6f} "
                f"sigreg:{(accum_reg / args.grad_accum_steps).item():.6f} "
                f"latent_std:{latent_std_value:.6f} "
                f"latent_mean_norm:{(accum_mean / args.grad_accum_steps).item():.6f} "
                f"latent_cov_trace:{(accum_cov / args.grad_accum_steps).item():.6f} "
                f"train_time:{elapsed_ms:.0f}ms"
            )

        reached_cap = max_wallclock_ms is not None and elapsed_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if device.type == "cuda":
        log0(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
        )

    if master_process:
        torch.save(raw_model.export_inference_state(), args.raw_model_path)
        model_bytes = args.raw_model_path.stat().st_size
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(raw_model.export_inference_state())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    if master_process:
        args.artifact_path.write_bytes(quant_blob)
        artifact_bytes = args.artifact_path.stat().st_size
        code_bytes = len(code.encode("utf-8"))
        log0(f"serialized_artifact_lzma:{artifact_bytes} bytes")
        log0(f"code_bytes_lzma:{code_bytes} bytes")
        log0(f"Total submission size lzma: {artifact_bytes + code_bytes} bytes")
        log0(f"submission_valid_lzma:{int(artifact_bytes + code_bytes <= SUBMISSION_SIZE_LIMIT_BYTES)}")
        log0(
            f"int8_payload_bytes:{quant_stats['int8_payload_bytes']} baseline_tensor_bytes:{quant_stats['baseline_tensor_bytes']}"
        )

    if distributed:
        dist.barrier()
    quant_state = torch.load(io.BytesIO(zlib.decompress(args.artifact_path.read_bytes())), map_location="cpu")
    raw_model.load_inference_state(dequantize_state_dict_int8(quant_state))

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_eval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, raw_model, rank, world_size, device, args.grad_accum_steps, val_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.6f} val_bpb:{q_val_bpb:.6f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_eval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if master_process:
        code_bytes = len(code.encode("utf-8"))
        artifact_bytes = args.artifact_path.stat().st_size
        build_meta = {
            "trainer_variant": "byte_lewm_standard",
            "jepa_enabled": int(args.jepa_enabled),
            "jepa_arch": args.jepa_arch,
            "chunk_size": args.chunk_size,
            "train_seq_len": args.train_seq_len,
            "model_dim": args.model_dim,
            "num_layers": args.num_layers,
            "jepa_latent_dim": args.jepa_latent_dim,
            "jepa_pred_hidden": args.jepa_pred_hidden,
            "jepa_pred_depth": args.jepa_pred_depth,
            "jepa_pred_heads": args.jepa_pred_heads,
            "projector_hidden": args.projector_hidden,
            "context_hidden": args.context_hidden,
            "pred_dim_head": args.pred_dim_head,
            "pred_dropout": args.pred_dropout,
            "pred_emb_dropout": args.pred_emb_dropout,
            "latent_weight": args.latent_weight,
            "reg_weight": args.reg_weight,
            "reg_slices": args.reg_slices,
            "sigreg_knots": args.sigreg_knots,
            "sigreg_max_t": args.sigreg_max_t,
            "jepa_target_momentum": args.jepa_target_momentum,
        }
        write_submission_json(args, code_bytes, artifact_bytes, q_val_loss, q_val_bpb, build_meta)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
