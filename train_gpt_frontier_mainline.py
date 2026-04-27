from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from flash_attn_interface import causal_attention, configure_attention_logging, flash_attention_import_summary
from frontier_checkpoint import atomic_json_dump
from frontier_loader import build_train_loader, load_data_shard
from frontier_quant import (
    CONTROL_TENSOR_NAME_PATTERNS,
    QuantConfig,
    classify_tensor,
    dequantize_state_dict,
    quant_config_from_env,
    quant_gap_report,
    quant_summary_json,
    quant_surface_record,
    quantize_state_dict,
    serialize_quantized_payload,
)
from frontier_tokenizer import (
    build_sentencepiece_luts,
    sentencepiece_tokenizer_stats,
    spm,
    validate_dataset_tokenizer_pair,
)
from research.submission_metrics import canonical_submission_eval


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    return bool(int(_env(key, "1" if default else "0")))


def _resolved_parallel_start() -> int:
    if "PARALLEL_RESIDUAL_START" in os.environ:
        return int(os.environ["PARALLEL_RESIDUAL_START"])
    return int(_env("PARALLEL_START_LAYER", "-1"))


def _resolved_loop_bounds(num_layers: int) -> tuple[int, int]:
    recur_layers = _env("RECUR_LAYERS", "").strip()
    if recur_layers:
        parsed = sorted({int(item.strip()) for item in recur_layers.split(",") if item.strip()})
        if not parsed:
            raise ValueError("RECUR_LAYERS was set but no layer indices could be parsed")
        if any(idx < 0 or idx >= num_layers for idx in parsed):
            raise ValueError(f"RECUR_LAYERS={recur_layers!r} includes an index outside [0, {num_layers - 1}]")
        expected = list(range(parsed[0], parsed[-1] + 1))
        if parsed != expected:
            raise ValueError(f"RECUR_LAYERS must describe one contiguous segment, got {recur_layers!r}")
        return parsed[0], parsed[-1]
    loop_start = int(_env("LOOP_START", "3"))
    loop_end = int(_env("LOOP_END", "5"))
    if not (0 <= loop_start <= loop_end < num_layers):
        raise ValueError(
            f"Loop bounds must satisfy 0 <= LOOP_START <= LOOP_END < NUM_LAYERS; "
            f"got LOOP_START={loop_start} LOOP_END={loop_end} NUM_LAYERS={num_layers}"
        )
    return loop_start, loop_end


@dataclass
class Hyperparameters:
    data_path: str
    train_files: str
    val_files: str
    tokenizer_path: str
    run_id: str
    seed: int
    artifact_dir: Path
    summary_path: Path
    iterations: int
    warmdown_iters: int
    warmup_steps: int
    train_batch_tokens: int
    train_seq_len: int
    eval_seq_len: int
    val_batch_size: int
    val_loss_every: int
    train_log_every: int
    max_wallclock_seconds: float
    gptq_reserve_seconds: float
    vocab_size: int
    num_layers: int
    xsa_last_n: int
    model_dim: int
    embedding_dim: int
    num_heads: int
    num_kv_heads: int
    mlp_mult: float
    tie_embeddings: bool
    skip_gates_enabled: bool
    per_pass_embeddings: bool
    qk_gain_init: float
    rope_base: float
    rope_dims: int
    rope_train_seq_len: int
    ln_scale: bool
    num_loops: int
    loop_start: int
    loop_end: int
    enable_looping_at: float
    parallel_residual_start: int
    embed_lr: float
    head_lr: float
    tied_embed_lr: float
    tied_embed_init_std: float
    matrix_lr: float
    scalar_lr: float
    muon_momentum: float
    muon_backend_steps: int
    muon_momentum_warmup_start: float
    muon_momentum_warmup_steps: int
    muon_row_normalize: bool
    beta1: float
    beta2: float
    adam_eps: float
    grad_clip_norm: float
    muon_wd: float
    adam_wd: float
    embed_wd: float
    ema_decay: float
    enable_async_loader: bool
    loader_prefetch_size: int
    shuffled_loader: bool
    sliding_window_enabled: bool
    eval_stride: int
    ttt_enabled: bool
    ttt_lr: float
    ttt_epochs: int
    ttt_momentum: float
    ttt_chunk_tokens: int
    ttt_train_batch_seqs: int
    ttt_grad_clip: float
    prefix_matcher_enabled: bool
    prequant_ttt_enabled: bool
    enable_torch_compile: bool
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    grad_accum_steps: int
    quant_config: QuantConfig

    @classmethod
    def from_env(cls) -> Hyperparameters:
        data_path = _env("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
        vocab_size = int(_env("VOCAB_SIZE", "8192"))
        num_layers = int(_env("NUM_LAYERS", "11"))
        model_dim = int(_env("MODEL_DIM", "512"))
        num_heads = int(_env("NUM_HEADS", "8"))
        if model_dim % num_heads != 0:
            raise ValueError(f"MODEL_DIM={model_dim} must be divisible by NUM_HEADS={num_heads}")
        head_dim = model_dim // num_heads
        rope_dims = min(int(_env("ROPE_DIMS", "16")), head_dim)
        if rope_dims % 2 != 0:
            raise ValueError(f"ROPE_DIMS must be even after clamping to head_dim={head_dim}; got {rope_dims}")
        loop_start, loop_end = _resolved_loop_bounds(num_layers)
        distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        world_size = int(_env("WORLD_SIZE", "1"))
        if world_size <= 0:
            raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
        grad_accum_steps = 8 // world_size
        artifact_dir = Path(_env("ARTIFACT_DIR", "."))
        summary_path = Path(_env("SUMMARY_PATH", str(artifact_dir / "run_summary.json")))
        return cls(
            data_path=data_path,
            train_files=os.path.join(data_path, "fineweb_train_*.bin"),
            val_files=os.path.join(data_path, "fineweb_val_*.bin"),
            tokenizer_path=_env("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model"),
            run_id=_env("RUN_ID", str(uuid.uuid4())),
            seed=int(_env("SEED", "1337")),
            artifact_dir=artifact_dir,
            summary_path=summary_path,
            iterations=int(_env("ITERATIONS", "20000")),
            warmdown_iters=int(_env("WARMDOWN_ITERS", "3500")),
            warmup_steps=int(_env("WARMUP_STEPS", "20")),
            train_batch_tokens=int(_env("TRAIN_BATCH_TOKENS", "786432")),
            train_seq_len=int(_env("TRAIN_SEQ_LEN", "2048")),
            eval_seq_len=int(_env("EVAL_SEQ_LEN", "2048")),
            val_batch_size=int(_env("VAL_BATCH_SIZE", "524288")),
            val_loss_every=int(_env("VAL_LOSS_EVERY", "4000")),
            train_log_every=int(_env("TRAIN_LOG_EVERY", "100")),
            max_wallclock_seconds=float(_env("MAX_WALLCLOCK_SECONDS", "600")),
            gptq_reserve_seconds=float(_env("GPTQ_RESERVE_SECONDS", "12")),
            vocab_size=vocab_size,
            num_layers=num_layers,
            xsa_last_n=int(_env("XSA_LAST_N", "11")),
            model_dim=model_dim,
            embedding_dim=int(_env("EMBEDDING_DIM", _env("MODEL_DIM", "512"))),
            num_heads=num_heads,
            num_kv_heads=int(_env("NUM_KV_HEADS", "4")),
            mlp_mult=float(_env("MLP_MULT", "4.0")),
            tie_embeddings=_env_bool("TIE_EMBEDDINGS", True),
            skip_gates_enabled=_env_bool("SKIP_GATES_ENABLED", True),
            per_pass_embeddings=_env_bool("PER_PASS_EMBEDDINGS", False),
            qk_gain_init=float(_env("QK_GAIN_INIT", "5.0")),
            rope_base=float(_env("ROPE_BASE", "10000")),
            rope_dims=rope_dims,
            rope_train_seq_len=int(_env("ROPE_TRAIN_SEQ_LEN", _env("TRAIN_SEQ_LEN", "2048"))),
            ln_scale=_env_bool("LN_SCALE", True),
            num_loops=int(_env("NUM_LOOPS", "0")),
            loop_start=loop_start,
            loop_end=loop_end,
            enable_looping_at=float(_env("ENABLE_LOOPING_AT", "0.35")),
            parallel_residual_start=_resolved_parallel_start(),
            embed_lr=float(_env("EMBED_LR", "0.6")),
            head_lr=float(_env("HEAD_LR", "0.008")),
            tied_embed_lr=float(_env("TIED_EMBED_LR", "0.035")),
            tied_embed_init_std=float(_env("TIED_EMBED_INIT_STD", "0.005")),
            matrix_lr=float(_env("MATRIX_LR", "0.022")),
            scalar_lr=float(_env("SCALAR_LR", "0.02")),
            muon_momentum=float(_env("MUON_MOMENTUM", "0.99")),
            muon_backend_steps=int(_env("MUON_BACKEND_STEPS", "5")),
            muon_momentum_warmup_start=float(_env("MUON_MOMENTUM_WARMUP_START", "0.92")),
            muon_momentum_warmup_steps=int(_env("MUON_MOMENTUM_WARMUP_STEPS", "1500")),
            muon_row_normalize=_env_bool("MUON_ROW_NORMALIZE", True),
            beta1=float(_env("BETA1", "0.9")),
            beta2=float(_env("BETA2", "0.95")),
            adam_eps=float(_env("ADAM_EPS", "1e-8")),
            grad_clip_norm=float(_env("GRAD_CLIP_NORM", "0.3")),
            muon_wd=float(_env("MUON_WD", "0.095")),
            adam_wd=float(_env("ADAM_WD", "0.02")),
            embed_wd=float(_env("EMBED_WD", "0.085")),
            ema_decay=float(_env("EMA_DECAY", "0.9965")),
            enable_async_loader=_env_bool("ENABLE_ASYNC_LOADER", False),
            loader_prefetch_size=int(_env("LOADER_PREFETCH_SIZE", "2")),
            shuffled_loader=_env_bool("SHUFFLED_TRAIN_LOADER", True),
            sliding_window_enabled=_env_bool("SLIDING_WINDOW_ENABLED", True),
            eval_stride=int(_env("EVAL_STRIDE", "64")),
            ttt_enabled=_env_bool("TTT_ENABLED", False),
            ttt_lr=float(_env("TTT_LR", "0.005")),
            ttt_epochs=int(_env("TTT_EPOCHS", "3")),
            ttt_momentum=float(_env("TTT_MOMENTUM", "0.9")),
            ttt_chunk_tokens=int(_env("TTT_CHUNK_TOKENS", "32768")),
            ttt_train_batch_seqs=int(_env("TTT_TRAIN_BATCH_SEQS", _env("TTT_BATCH_SEQS", "32"))),
            ttt_grad_clip=float(_env("TTT_GRAD_CLIP", "1.0")),
            prefix_matcher_enabled=_env_bool("PREFIX_MATCHER_ENABLED", False),
            prequant_ttt_enabled=_env_bool("PREQUANT_TTT_ENABLED", False),
            enable_torch_compile=_env_bool("ENABLE_TORCH_COMPILE", True),
            distributed=distributed,
            rank=int(_env("RANK", "0")),
            world_size=world_size,
            local_rank=int(_env("LOCAL_RANK", "0")),
            grad_accum_steps=grad_accum_steps,
            quant_config=quant_config_from_env(),
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def track(self) -> str:
        if self.prequant_ttt_enabled:
            return "prequant_ttt"
        if self.ttt_enabled:
            return "score_first_ttt"
        return "fixed_predictor"


class ValidationData:
    def __init__(self, args: Hyperparameters, device: torch.device):
        if spm is None:
            raise ImportError("sentencepiece is required to run the frontier mainline trainer")
        self.sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        if int(self.sp.vocab_size()) != args.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}"
            )
        self.val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_sentencepiece_luts(
            self.sp,
            args.vocab_size,
            device,
        )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for EVAL_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, *, base: float, train_seq_len: int, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        )
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
            rope_dims = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rope_dims / max(rope_dims - 2, 1)))
                inv_freq = 1.0 / (
                    new_base ** (torch.arange(0, rope_dims, 2, dtype=torch.float32, device=device) / rope_dims)
                )
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
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


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        rope_base: float,
        rope_train_seq_len: int,
        rope_dims: int,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        if num_heads % num_kv_heads != 0:
            raise ValueError("NUM_HEADS must be divisible by NUM_KV_HEADS")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = rope_dims
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=rope_train_seq_len, rope_dims=rope_dims)
        self.use_xsa = False

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        batch, seq_len, heads, dim = y.shape
        kv_heads = v.size(-2)
        group = heads // kv_heads
        y_grouped = y.reshape(batch, seq_len, kv_heads, group, dim)
        v_norm = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_grouped * v_norm).sum(dim=-1, keepdim=True) * v_norm
        return (y_grouped - proj).reshape(batch, seq_len, heads, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape
        q = self.c_q(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seq_len, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = causal_attention(q, k, v, enable_gqa=(self.num_kv_heads != self.num_heads))
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(batch, seq_len, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        *,
        rope_base: float,
        rope_train_seq_len: int,
        rope_dims: int,
        qk_gain_init: float,
        layer_idx: int,
        ln_scale: bool,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base=rope_base,
            rope_train_seq_len=rope_train_seq_len,
            rope_dims=rope_dims,
            qk_gain_init=qk_gain_init,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            return (
                x_in
                + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
                + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
            )
        x_out = x_in + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        return x_out + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x_out) * self.ln_scale_factor
        )


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.tie_embeddings = args.tie_embeddings
        self.logit_softcap = float(_env("LOGIT_SOFTCAP", "30.0"))
        if self.logit_softcap <= 0.0:
            raise ValueError(f"LOGIT_SOFTCAP must be positive, got {self.logit_softcap}")
        self.tied_embed_init_std = args.tied_embed_init_std
        self.tok_emb = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.embed_proj = (
            CastedLinear(args.embedding_dim, args.model_dim, bias=False)
            if args.embedding_dim != args.model_dim
            else None
        )
        self.head_proj = (
            CastedLinear(args.model_dim, args.embedding_dim, bias=False)
            if args.embedding_dim != args.model_dim
            else None
        )
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    args.model_dim,
                    args.num_heads,
                    args.num_kv_heads,
                    args.mlp_mult,
                    rope_base=args.rope_base,
                    rope_train_seq_len=args.rope_train_seq_len,
                    rope_dims=args.rope_dims,
                    qk_gain_init=args.qk_gain_init,
                    layer_idx=idx,
                    ln_scale=args.ln_scale,
                )
                for idx in range(args.num_layers)
            ]
        )
        if args.xsa_last_n > 0:
            for idx in range(max(0, args.num_layers - args.xsa_last_n), args.num_layers):
                self.blocks[idx].attn.use_xsa = True
        if args.parallel_residual_start >= 0:
            for idx in range(args.parallel_residual_start, args.num_layers):
                self.blocks[idx].parallel = True
        self.looping_active = False
        if args.num_loops > 0:
            loop_segment = list(range(args.loop_start, args.loop_end + 1))
            all_indices = list(range(args.loop_start))
            for _ in range(args.num_loops + 1):
                all_indices.extend(loop_segment)
            all_indices.extend(range(args.loop_end + 1, args.num_layers))
            num_encoder = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_encoder]
            self.decoder_indices = all_indices[num_encoder:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, args.num_layers))
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, args.model_dim, dtype=torch.float32))
        self.skip_gates = (
            nn.Parameter(torch.zeros(self.num_skip_weights, args.model_dim, dtype=torch.float32))
            if args.skip_gates_enabled
            else None
        )
        self.pass_embeddings = (
            nn.Embedding(max(1, len(self.encoder_indices) + len(self.decoder_indices)), args.model_dim)
            if args.per_pass_embeddings
            else None
        )
        self.pass_embedding_scale = (
            nn.Parameter(torch.tensor(0.02, dtype=torch.float32)) if self.pass_embeddings is not None else None
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else CastedLinear(args.embedding_dim, args.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        if self.pass_embeddings is not None:
            nn.init.zeros_(self.pass_embeddings.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def describe_architecture(self) -> dict[str, object]:
        return {
            "encoder_indices": list(self.encoder_indices),
            "decoder_indices": list(self.decoder_indices),
            "parallel_residual_layers": [idx for idx, block in enumerate(self.blocks) if block.parallel],
            "xsa_layers": [idx for idx, block in enumerate(self.blocks) if block.attn.use_xsa],
            "looping_active": self.looping_active,
            "per_pass_embeddings": self.pass_embeddings is not None,
        }

    def _add_pass_embedding(self, x: Tensor, pass_index: int) -> Tensor:
        if self.pass_embeddings is None or self.pass_embedding_scale is None:
            return x
        embed_idx = min(pass_index, self.pass_embeddings.num_embeddings - 1)
        pass_embed = self.pass_embeddings.weight[embed_idx].to(dtype=x.dtype)
        return x + self.pass_embedding_scale.to(dtype=x.dtype) * pass_embed[None, None, :]

    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips: list[Tensor] = []
        enc_iter = self.encoder_indices if self.looping_active else list(range(self.num_encoder_layers))
        dec_iter = self.decoder_indices if self.looping_active else list(range(self.num_encoder_layers, self.args.num_layers))
        pass_index = 0
        for block_idx in enc_iter:
            x = self._add_pass_embedding(x, pass_index)
            x = self.blocks[block_idx](x, x0)
            skips.append(x)
            pass_index += 1
        for skip_idx, block_idx in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    gate = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, gate)
                else:
                    x = x + scaled_skip
            x = self._add_pass_embedding(x, pass_index)
            x = self.blocks[block_idx](x, x0)
            pass_index += 1
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        return x

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.forward_hidden(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when TIE_EMBEDDINGS=0")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )


def restore_fp32_params(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
            param.data = param.data.float()


def zeropower_via_newtonschulz5(grad: Tensor, *, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16()
    x /= x.norm() + eps
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        momentum: float,
        backend_steps: int,
        weight_decay: float = 0.0,
        row_normalize: bool = False,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=True,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
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
            total_params = sum(int(param.numel()) for param in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for idx, param in enumerate(params):
                if idx % world_size == rank and param.grad is not None:
                    grad = param.grad
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    grad = grad.add(buf, alpha=momentum)
                    if group.get("row_normalize", False):
                        row_norms = grad.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                        grad = grad / row_norms.to(dtype=grad.dtype)
                    grad = zeropower_via_newtonschulz5(grad, steps=backend_steps)
                    grad *= max(1.0, grad.size(0) / grad.size(1)) ** 0.5
                    updates_flat[curr : curr + param.numel()] = grad.reshape(-1)
                curr += param.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for param in params:
                weight_decay = group.get("weight_decay", 0.0)
                if weight_decay > 0.0:
                    param.data.mul_(1.0 - lr * weight_decay)
                grad = updates_flat[curr : curr + param.numel()].view_as(param).to(dtype=param.dtype)
                param.add_(grad, alpha=-lr)
                curr += param.numel()
        return loss


class Optimizers:
    def __init__(self, args: Hyperparameters, model: GPT):
        embed_params: list[Tensor] = []
        matrix_params: list[Tensor] = []
        scalar_params: list[Tensor] = []
        head_params: list[Tensor] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name == "lm_head.weight":
                head_params.append(param)
                continue
            if name.startswith("tok_emb.") or "pass_embeddings." in name:
                embed_params.append(param)
                continue
            if param.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
                matrix_params.append(param)
            else:
                scalar_params.append(param)

        token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
        self.optimizer_tok = torch.optim.AdamW(
            [{"params": embed_params, "lr": token_lr, "base_lr": token_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.embed_wd,
            fused=torch.cuda.is_available(),
        )
        self.optimizer_muon = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            weight_decay=args.muon_wd,
            row_normalize=args.muon_row_normalize,
        )
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_wd,
            fused=torch.cuda.is_available(),
        )
        self.optimizers: list[torch.optim.Optimizer] = [
            self.optimizer_tok,
            self.optimizer_muon,
            self.optimizer_scalar,
        ]
        if head_params:
            self.optimizer_head = torch.optim.Adam(
                [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr}],
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
                fused=torch.cuda.is_available(),
            )
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()


def make_model(args: Hyperparameters, device: torch.device) -> GPT:
    model = GPT(args).to(device).bfloat16()
    restore_fp32_params(model)
    return model


def load_train_batch(
    loader,
    args: Hyperparameters,
) -> tuple[Tensor, Tensor]:
    return loader.next_batch(args.train_batch_tokens, args.train_seq_len, args.grad_accum_steps)


def _metric_from_totals(loss_sum: Tensor, token_count: Tensor, byte_count: Tensor) -> tuple[float, float]:
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return float(val_loss), float(val_bpb)


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    device: torch.device,
    val_data: ValidationData,
) -> tuple[float, float]:
    seq_len = args.eval_seq_len
    local_batch_tokens = args.val_batch_size // (args.world_size * args.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={args.world_size}, "
            f"GRAD_ACCUM_STEPS={args.grad_accum_steps}, EVAL_SEQ_LEN={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * args.rank) // args.world_size
    seq_end = (total_seqs * (args.rank + 1)) // args.world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * batch_token_count
            token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]).to(
                dtype=torch.int16
            )
            byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _metric_from_totals(loss_sum, token_count, byte_count)


def eval_val_sliding_window(
    args: Hyperparameters,
    model: GPT,
    device: torch.device,
    val_data: ValidationData,
    *,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    seq_len = args.eval_seq_len
    context_size = seq_len - args.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, args.eval_stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_start = total_windows * args.rank // args.world_size
    my_end = total_windows * (args.rank + 1) // args.world_size
    my_windows = window_starts[my_start:my_end]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_idx in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[batch_idx : batch_idx + batch_seqs]
            batch_size = len(batch_ws)
            x_batch = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
            window_lengths: list[int] = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                window_lengths.append(wlen)
                chunk = val_data.val_tokens[ws : we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(batch_size, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = window_lengths[i]
                score_start = 0 if ws == 0 else context_size
                scored_nll = nll[i, score_start:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - score_start)
                tgt = y_batch[i, score_start:wlen]
                prev = x_batch[i, score_start:wlen]
                token_bytes = val_data.base_bytes_lut[tgt].to(torch.float64)
                token_bytes += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += token_bytes.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _metric_from_totals(loss_sum, token_count, byte_count)


def eval_val_score_first_ttt(
    args: Hyperparameters,
    model: GPT,
    device: torch.device,
    val_data: ValidationData,
    *,
    batch_seqs: int | None = None,
) -> tuple[float, float]:
    seq_len = args.eval_seq_len
    stride = args.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    batch_seqs = args.ttt_train_batch_seqs if batch_seqs is None else batch_seqs
    ttt_chunk = args.ttt_chunk_tokens
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        window_len = min(ws + seq_len, total_tokens) - ws
        score_start = 0 if ws == 0 else context_size
        scored_start = ws + score_start
        chunk_idx = min(scored_start // ttt_chunk, num_chunks - 1)
        if window_len > score_start:
            chunk_windows[chunk_idx].append(ws)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ttt_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    for chunk_idx, windows in enumerate(chunk_windows):
        if not windows:
            continue
        my_start = len(windows) * args.rank // args.world_size
        my_end = len(windows) * (args.rank + 1) // args.world_size
        my_windows = windows[my_start:my_end]
        model.eval()
        with torch.no_grad():
            for batch_idx in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[batch_idx : batch_idx + batch_seqs]
                batch_size = len(batch_ws)
                x_batch = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
                window_lengths: list[int] = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    window_lengths.append(wlen)
                    chunk = val_data.val_tokens[ws : we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(batch_size, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = window_lengths[i]
                    score_start = 0 if ws == 0 else context_size
                    scored_nll = nll[i, score_start:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - score_start)
                    tgt = y_batch[i, score_start:wlen]
                    prev = x_batch[i, score_start:wlen]
                    token_bytes = val_data.base_bytes_lut[tgt].to(torch.float64)
                    token_bytes += (
                        val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]
                    ).to(torch.float64)
                    byte_count += token_bytes.sum()

        is_last_chunk = chunk_idx == num_chunks - 1
        if is_last_chunk or args.ttt_epochs <= 0:
            continue
        chunk_start = chunk_idx * ttt_chunk
        chunk_end = min((chunk_idx + 1) * ttt_chunk, total_tokens)
        chunk_seqs = (chunk_end - chunk_start) // seq_len
        if chunk_seqs <= 0:
            continue
        my_seq_start = chunk_seqs * args.rank // args.world_size
        my_seq_end = chunk_seqs * (args.rank + 1) // args.world_size
        my_chunk_seqs = my_seq_end - my_seq_start
        if my_chunk_seqs <= 0:
            continue
        chunk_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * chunk_idx / max(num_chunks - 1, 1)))
        for group in optimizer.param_groups:
            group["lr"] = chunk_lr
        model.train()
        for _epoch in range(args.ttt_epochs):
            for bs in range(0, my_chunk_seqs, args.ttt_train_batch_seqs):
                be = min(bs + args.ttt_train_batch_seqs, my_chunk_seqs)
                actual_bs = my_seq_start + bs
                start_tok = chunk_start + actual_bs * seq_len
                end_tok = chunk_start + (my_seq_start + be) * seq_len + 1
                if end_tok > val_data.val_tokens.numel():
                    continue
                local = val_data.val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                x = local[:-1].reshape(-1, seq_len)
                y = local[1:].reshape(-1, seq_len)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = model(x, y)
                loss.backward()
                if args.world_size > 1:
                    for param in ttt_params:
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                optimizer.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    model.eval()
    return _metric_from_totals(loss_sum, token_count, byte_count)


def collect_hessians(
    model: GPT,
    calibration_inputs: list[Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    hessians: dict[str, Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(_module: nn.Module, inp: tuple[Tensor, ...], _out: Tensor) -> None:
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            h = hessians.setdefault(name, torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device))
            h.addmm_(x.T, x)

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65_536:
            tensor_name = f"{name}.weight"
            tensor_class = classify_tensor(tensor_name)
            if tensor_class in {"attn", "mlp"}:
                hooks.append(module.register_forward_hook(make_hook(tensor_name)))
    if model.tie_embeddings:
        hook_module: nn.Module = model.head_proj if model.head_proj is not None else model.final_norm
        hooks.append(hook_module.register_forward_hook(make_hook("tok_emb.weight")))

    was_looping = model.looping_active
    model.eval()
    with torch.inference_mode():
        for batch in calibration_inputs:
            x = batch.to(device=device, dtype=torch.int64, non_blocking=True)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    model.looping_active = was_looping
    return {name: tensor.cpu() / max(len(calibration_inputs), 1) for name, tensor in hessians.items()}


def build_calibration_inputs(
    args: Hyperparameters,
    recent_inputs: list[Tensor],
    calibration_loader,
    device: torch.device,
) -> list[Tensor]:
    if args.quant_config.gptq_mode != "full" or args.quant_config.calibration_batches <= 0:
        return []
    target = args.quant_config.calibration_batches
    strategy = args.quant_config.calibration_strategy
    recent = list(recent_inputs)
    recent_batch_count = 0
    if strategy == "recent":
        recent_batch_count = min(target, len(recent))
    elif strategy == "mixed":
        recent_batch_count = min(target // 2, len(recent))
    elif strategy == "loop_heavy":
        recent_batch_count = min(max(target // 2, target - 4), len(recent))
    batches: list[Tensor] = []
    if recent_batch_count > 0:
        if recent_batch_count <= len(recent):
            batches.extend(recent[-recent_batch_count:])
        else:
            for idx in range(recent_batch_count):
                batches.append(recent[idx % len(recent)])
    while len(batches) < target:
        x, _ = calibration_loader.next_batch(args.train_batch_tokens, args.train_seq_len, args.grad_accum_steps)
        batches.append(x.detach().cpu())
    return batches


def timed_eval(
    label: str,
    eval_fn,
    *args,
    log_fn,
    exact_evals: dict[str, dict[str, float]],
    **kwargs,
) -> tuple[float, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    val_loss, val_bpb = eval_fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    log_fn(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms")
    log_fn(f"{label}_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")
    exact_evals[label] = {"val_loss": val_loss, "val_bpb": val_bpb, "eval_time_ms": elapsed_ms}
    return val_loss, val_bpb


def mainline_surface_record(args: Hyperparameters) -> dict[str, object]:
    surface = {
        "entrypoint": "train_gpt_frontier_mainline.py",
        "track": args.track,
        "data_path": args.data_path,
        "tokenizer_path": args.tokenizer_path,
        "vocab_size": args.vocab_size,
        "num_layers": args.num_layers,
        "model_dim": args.model_dim,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "mlp_mult": args.mlp_mult,
        "xsa_last_n": args.xsa_last_n,
        "skip_gates_enabled": args.skip_gates_enabled,
        "per_pass_embeddings": args.per_pass_embeddings,
        "qk_gain_init": args.qk_gain_init,
        "num_loops": args.num_loops,
        "loop_start": args.loop_start,
        "loop_end": args.loop_end,
        "enable_looping_at": args.enable_looping_at,
        "parallel_residual_start": args.parallel_residual_start,
        "enable_async_loader": args.enable_async_loader,
        "loader_prefetch_size": args.loader_prefetch_size,
        "shuffled_loader": args.shuffled_loader,
        "prefix_matcher_enabled": args.prefix_matcher_enabled,
        "prequant_ttt_enabled": args.prequant_ttt_enabled,
        "ttt_enabled": args.ttt_enabled,
        "quant_surface": quant_surface_record(args.quant_config),
        "warnings": [],
    }
    if args.prefix_matcher_enabled:
        surface["warnings"].append("PREFIX_MATCHER_ENABLED is challenger-only and remains a no-op in the stable trainer.")
    if args.prequant_ttt_enabled:
        surface["warnings"].append("PREQUANT_TTT_ENABLED is challenger-only and remains a no-op in the stable trainer.")
    tokenizer_path = Path(args.tokenizer_path)
    if tokenizer_path.is_file():
        try:
            surface["tokenizer_stats"] = sentencepiece_tokenizer_stats(str(tokenizer_path))
        except Exception as exc:  # noqa: BLE001
            surface["tokenizer_stats_error"] = str(exc)
    dataset_dir = Path(args.data_path)
    if dataset_dir.is_dir():
        try:
            dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
                args.data_path,
                args.tokenizer_path,
            )
            surface["dataset_manifest"] = {
                "dataset_name": dataset_name,
                "actual_train_files": actual_train_files,
                "expected_train_files": expected_train_files,
            }
        except Exception as exc:  # noqa: BLE001
            surface["dataset_manifest_error"] = str(exc)
    return surface


def write_surface_files(args: Hyperparameters) -> dict[str, object]:
    surface = mainline_surface_record(args)
    if not args.is_main_process:
        return surface
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    quant_surface = quant_surface_record(args.quant_config)
    (args.artifact_dir / "mainline_surface.json").write_text(
        json.dumps(surface, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.artifact_dir / "quant_surface.json").write_text(
        json.dumps(quant_surface, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.prefix_matcher_enabled or args.prequant_ttt_enabled:
        (args.artifact_dir / "rule_audit_seed.txt").write_text(
            "\n".join(
                [
                    f"track: {args.track}",
                    f"prefix_matcher_enabled: {args.prefix_matcher_enabled}",
                    f"prequant_ttt_enabled: {args.prequant_ttt_enabled}",
                    "manual_review_required: true",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    return surface


def write_summary(
    args: Hyperparameters,
    *,
    status: str,
    step: int,
    train_time_ms: float,
    training_best_val_loss: float | None,
    training_best_val_bpb: float | None,
    last_train_loss: float | None,
    last_val_loss: float | None,
    last_val_bpb: float | None,
    named_evals_exact: dict[str, dict[str, float]],
    export_seconds: float | None = None,
    note: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    if not args.is_main_process:
        return
    label, payload = canonical_submission_eval({"named_evals_exact": named_evals_exact}, track=args.track)
    summary = {
        "run_id": args.run_id,
        "status": status,
        "step": step,
        "iterations": args.iterations,
        "track": args.track,
        "train_time_ms": round(train_time_ms, 3),
        "max_wallclock_seconds": args.max_wallclock_seconds,
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "last_val_bpb": last_val_bpb,
        "training_best_val_loss": training_best_val_loss,
        "training_best_val_bpb": training_best_val_bpb,
        "best_val_loss": None if payload is None else payload.get("val_loss", training_best_val_loss),
        "best_val_bpb": None if payload is None else payload.get("val_bpb", training_best_val_bpb),
        "final_submission_metric_label": label,
        "official_submission_metric_label": label,
        "final_submission_loss": None if payload is None else payload.get("val_loss"),
        "final_submission_bpb": None if payload is None else payload.get("val_bpb"),
        "named_evals_exact": named_evals_exact,
        "export_seconds": export_seconds,
        "note": note,
    }
    if extra:
        summary.update(extra)
    atomic_json_dump(summary, args.summary_path)


def run_training(args: Hyperparameters, device: torch.device, log_fn) -> None:
    validation = ValidationData(args, device)
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    log_fn(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log_fn(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log_fn(f"val_loader:shards pattern={args.val_files} tokens:{validation.val_tokens.numel() - 1}")

    base_model = make_model(args, device)
    model: nn.Module = base_model
    if args.enable_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False)
    log_fn(f"model_params:{sum(param.numel() for param in base_model.parameters())}")
    log_fn(
        "architecture:"
        f" encoder={base_model.encoder_indices}"
        f" decoder={base_model.decoder_indices}"
        f" parallel_from={args.parallel_residual_start}"
        f" xsa_last_n={args.xsa_last_n}"
        f" per_pass_embeddings={args.per_pass_embeddings}"
    )
    log_fn(
        "quant_surface:"
        + json.dumps(
            quant_surface_record(args.quant_config),
            sort_keys=True,
        )
    )
    if args.prefix_matcher_enabled:
        log_fn("challenger_notice: PREFIX_MATCHER_ENABLED remains a no-op in the stable mainline trainer")
    if args.prequant_ttt_enabled:
        log_fn("challenger_notice: PREQUANT_TTT_ENABLED remains a no-op in the stable mainline trainer")

    optimizers = Optimizers(args, base_model)
    train_loader = build_train_loader(
        args.train_files,
        args.rank,
        args.world_size,
        device,
        async_enabled=args.enable_async_loader,
        prefetch_size=args.loader_prefetch_size,
        shuffled=args.shuffled_loader,
        seed=args.seed,
    )
    recent_inputs: deque[Tensor] = deque(maxlen=max(8, min(args.quant_config.calibration_batches, 32)))
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms = max(max_wallclock_ms - 1000.0 * args.gptq_reserve_seconds, 0.0)
        log_fn(
            f"gptq:reserving {args.gptq_reserve_seconds:.0f}s for export/calibration "
            f"effective_train_budget_ms:{max_wallclock_ms:.0f}"
        )

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def zero_grad_all() -> None:
        optimizers.zero_grad_all()

    def train_step() -> Tensor:
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(args.grad_accum_steps):
            if args.distributed:
                assert isinstance(model, DDP)
                model.require_backward_grad_sync = micro_step == args.grad_accum_steps - 1
            x, y = load_train_batch(train_loader, args)
            recent_inputs.append(x.detach().cpu())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / args.grad_accum_steps).backward()
        train_loss /= args.grad_accum_steps
        return train_loss

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        base_model.train()
        for warmup_step in range(args.warmup_steps):
            warmup_loss = train_step()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log_fn(f"warmup_step:{warmup_step + 1}/{args.warmup_steps} loss:{warmup_loss.item():.4f}")
        if args.num_loops > 0:
            base_model.looping_active = True
            for warmup_step in range(args.warmup_steps):
                warmup_loss = train_step()
                for opt in optimizers:
                    opt.step()
                zero_grad_all()
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                    log_fn(f"loop_warmup_step:{warmup_step + 1}/{args.warmup_steps} loss:{warmup_loss.item():.4f}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if args.distributed and isinstance(model, DDP):
            model.require_backward_grad_sync = True
        if hasattr(train_loader, "close"):
            train_loader.close()
        train_loader = build_train_loader(
            args.train_files,
            args.rank,
            args.world_size,
            device,
            async_enabled=args.enable_async_loader,
            prefetch_size=args.loader_prefetch_size,
            shuffled=args.shuffled_loader,
            seed=args.seed,
        )
        recent_inputs.clear()

    ema_state = {name: tensor.detach().float().clone() for name, tensor in base_model.state_dict().items()}
    training_time_ms = 0.0
    stop_after_step: int | None = None
    step = 0
    best_val_loss: float | None = None
    best_val_bpb: float | None = None
    last_val_loss: float | None = None
    last_val_bpb: float | None = None
    last_train_loss: float | None = None
    named_evals_exact: dict[str, dict[str, float]] = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - started)
            val_loss, val_bpb = eval_val(args, model, device, validation)
            last_val_loss = val_loss
            last_val_bpb = val_bpb
            if best_val_bpb is None or val_bpb < best_val_bpb:
                best_val_loss = val_loss
                best_val_bpb = val_bpb
                log_fn(f"best_val_update:step:{step} val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f}")
            log_fn(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            write_summary(
                args,
                status="running",
                step=step,
                train_time_ms=training_time_ms,
                training_best_val_loss=best_val_loss,
                training_best_val_bpb=best_val_bpb,
                last_train_loss=last_train_loss,
                last_val_loss=last_val_loss,
                last_val_bpb=last_val_bpb,
                named_evals_exact=named_evals_exact,
                note="validation",
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - started)
        scale = lr_mul(step, elapsed_ms)
        if args.num_loops > 0 and not base_model.looping_active:
            threshold_reached = False
            if max_wallclock_ms is not None:
                threshold_reached = elapsed_ms / max(max_wallclock_ms, 1e-9) >= args.enable_looping_at
            else:
                threshold_reached = step / max(args.iterations, 1) >= args.enable_looping_at
            if threshold_reached:
                base_model.looping_active = True
                log_fn(
                    f"layer_loop:enabled step:{step} frac:{args.enable_looping_at:.3f} "
                    f"encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
                )

        train_loss = train_step()
        last_train_loss = float(train_loss.item())
        momentum_frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1.0 - momentum_frac) * args.muon_momentum_warmup_start + momentum_frac * args.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        with torch.no_grad():
            for name, tensor in base_model.state_dict().items():
                ema_state[name].mul_(args.ema_decay).add_(tensor.detach().float(), alpha=1.0 - args.ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - started)
        if args.train_log_every > 0 and (step <= 5 or step % args.train_log_every == 0):
            tok_s = step * args.train_batch_tokens / max(approx_training_time_ms / 1000.0, 1e-9)
            log_fn(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"tok_s:{tok_s:.0f}"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if args.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    log_fn(
        f"peak memory allocated:{torch.cuda.max_memory_allocated() // 1024 // 1024 if torch.cuda.is_available() else 0}MiB "
        f"reserved:{torch.cuda.max_memory_reserved() // 1024 // 1024 if torch.cuda.is_available() else 0}MiB"
    )
    current_state = base_model.state_dict()
    ema_cast = {name: tensor.to(dtype=current_state[name].dtype) for name, tensor in ema_state.items()}
    base_model.load_state_dict(ema_cast, strict=True)

    timed_eval("post_ema", eval_val, args, model, device, validation, log_fn=log_fn, exact_evals=named_evals_exact)

    export_started = time.perf_counter()
    code = Path(__file__).read_text(encoding="utf-8")
    code_bytes = len(code.encode("utf-8"))
    export_state = {name: tensor.detach().cpu() for name, tensor in base_model.state_dict().items()}
    if args.is_main_process:
        torch.save(export_state, args.artifact_dir / "final_model.pt")
        log_fn(f"Serialized model: {(args.artifact_dir / 'final_model.pt').stat().st_size} bytes")
        log_fn(f"Code size: {code_bytes} bytes")

    calibration_loader = build_train_loader(
        args.train_files,
        args.rank,
        args.world_size,
        device,
        async_enabled=False,
        prefetch_size=1,
        shuffled=True,
        seed=args.seed + 17,
    )
    calibration_inputs = build_calibration_inputs(args, list(recent_inputs), calibration_loader, device)
    if hasattr(calibration_loader, "close"):
        calibration_loader.close()
    if args.quant_config.calibration_strategy == "loop_heavy" and args.num_loops > 0:
        base_model.looping_active = True
    hessians = collect_hessians(base_model, calibration_inputs, device) if calibration_inputs else {}
    quant_payload, quant_summary = quantize_state_dict(
        export_state,
        config=args.quant_config,
        hessians=hessians,
    )
    raw_payload, compressed_payload = serialize_quantized_payload(quant_payload, compressor=args.quant_config.compressor)
    compressed_path = args.artifact_dir / "final_model.ptz"
    if args.is_main_process:
        compressed_path.write_bytes(compressed_payload)
        (args.artifact_dir / "quant_summary.json").write_text(quant_summary_json(quant_summary), encoding="utf-8")
        log_fn(f"Serialized model mixed+{args.quant_config.compressor}: {len(compressed_payload)} bytes")
        log_fn(f"Total submission size: {len(compressed_payload) + code_bytes} bytes")
    export_seconds = time.perf_counter() - export_started

    eval_model = make_model(args, device)
    eval_model.load_state_dict(dequantize_state_dict(quant_payload), strict=True)
    eval_model.looping_active = base_model.looping_active
    timed_eval("final_roundtrip", eval_val, args, eval_model, device, validation, log_fn=log_fn, exact_evals=named_evals_exact)
    if args.sliding_window_enabled and 0 < args.eval_stride < args.eval_seq_len:
        timed_eval(
            "final_sliding_window",
            eval_val_sliding_window,
            args,
            eval_model,
            device,
            validation,
            log_fn=log_fn,
            exact_evals=named_evals_exact,
        )
    if args.ttt_enabled and args.sliding_window_enabled:
        timed_eval(
            "legal_ttt",
            eval_val_score_first_ttt,
            args,
            eval_model,
            device,
            validation,
            log_fn=log_fn,
            exact_evals=named_evals_exact,
        )

    final_label, final_payload = canonical_submission_eval({"named_evals_exact": named_evals_exact}, track=args.track)
    gap = quant_gap_report(best_val_bpb, None if final_payload is None else float(final_payload["val_bpb"]))
    loader_stats = train_loader.loader_stats().__dict__
    if hasattr(train_loader, "close"):
        train_loader.close()
    write_summary(
        args,
        status="completed",
        step=step,
        train_time_ms=training_time_ms,
        training_best_val_loss=best_val_loss,
        training_best_val_bpb=best_val_bpb,
        last_train_loss=last_train_loss,
        last_val_loss=last_val_loss,
        last_val_bpb=last_val_bpb,
        named_evals_exact=named_evals_exact,
        export_seconds=export_seconds,
        note="final_export_complete",
        extra={
            "loader_stats": loader_stats,
            "quant_surface": quant_surface_record(args.quant_config),
            "quant_summary": quant_summary,
            "artifact_bytes": len(compressed_payload) + code_bytes,
            "compressed_model_bytes": len(compressed_payload),
            "compressed_payload_bytes": len(raw_payload),
            **gap,
            "final_model_path": str(compressed_path),
            "final_submission_metric_label": final_label,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable SP8192 frontier mainline trainer")
    parser.add_argument("--dry-run", action="store_true", help="Write resolved surfaces and exit before CUDA setup")
    args_ns = parser.parse_args()
    args = Hyperparameters.from_env()
    surface = write_surface_files(args)
    if args_ns.dry_run or _env_bool("MAINLINE_DRY_RUN", False):
        write_summary(
            args,
            status="dry_run",
            step=0,
            train_time_ms=0.0,
            training_best_val_loss=None,
            training_best_val_bpb=None,
            last_train_loss=None,
            last_val_loss=None,
            last_val_bpb=None,
            named_evals_exact={},
            note="dry_run",
            extra={"mainline_surface": surface},
        )
        print(json.dumps(surface, indent=2, sort_keys=True))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training; use --dry-run for local config validation")

    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(device)
    if args.distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    log_dir = Path(_env("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{args.run_id}.txt"

    def log_fn(message: str, *, console: bool = True) -> None:
        if not args.is_main_process:
            return
        if console:
            print(message)
        with logfile.open("a", encoding="utf-8") as f:
            print(message, file=f)

    configure_attention_logging(log_fn if args.is_main_process else None)
    attention_summary = flash_attention_import_summary()
    log_fn("=" * 100, console=False)
    log_fn("Hyperparameters:", console=True)
    for key, value in sorted(vars(args).items()):
        if key == "quant_config":
            log_fn(f"  quant_config: {json.dumps(quant_surface_record(value), sort_keys=True)}", console=True)
        else:
            log_fn(f"  {key}: {value}", console=True)
    log_fn("=" * 100, console=False)
    log_fn(f"Running Python {sys.version}", console=False)
    log_fn(f"Running PyTorch {torch.__version__}", console=False)
    log_fn(
        "attention_runtime:"
        f" flash_attn_available={attention_summary['flash_attn_available']}"
        f" source={attention_summary['flash_attn_source']}"
        f" import_error={attention_summary['flash_attn_import_error']}",
        console=False,
    )
    if which("nvidia-smi") is not None:
        log_fn(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
            console=False,
        )
    log_fn("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    run_training(args, device, log_fn)
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
