#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)
    DEVICE = torch.device("cuda", LOCAL_RANK)
else:
    DEVICE = torch.device("cpu")
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    iterations = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    eval_max_tokens = int(os.environ.get("EVAL_MAX_TOKENS", "0"))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    model_dim = int(os.environ.get("MODEL_DIM", 768))
    num_heads = int(os.environ.get("NUM_HEADS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 2))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    embed_dim = int(os.environ.get("EMBED_DIM", 254))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 1000000.0))
    rope_dim = int(os.environ.get("ROPE_DIM", 16))
    parallel_residual_start_layer = int(os.environ.get("PARALLEL_RESIDUAL_START_LAYER", 10_000))
    mlp_act = os.environ.get("MLP_ACT", "swiglu")
    leaky_relu_slope = float(os.environ.get("LEAKY_RELU_SLOPE", 0.5))
    layer_schedule = os.environ.get("LAYER_SCHEDULE", "")
    qk_gain = float(os.environ.get("QK_GAIN", 1.0))
    softcap_mode = os.environ.get("SOFTCAP_MODE", "tanh")
    z_loss_coef = float(os.environ.get("Z_LOSS_COEF", 0.0))
    weight_l1_coef = float(os.environ.get("WEIGHT_L1_COEF", 0.0))
    sign_balance_coef = float(os.environ.get("SIGN_BALANCE_COEF", 0.0))
    sign_balance_temp = float(os.environ.get("SIGN_BALANCE_TEMP", 0.05))
    init_model_npz = os.environ.get("INIT_MODEL_NPZ", "")

    quant_mode = os.environ.get("QUANT_MODE", "binary")
    quant_group_size = int(os.environ.get("QUANT_GROUP_SIZE", 128))
    quantize_embeddings = bool(int(os.environ.get("QUANTIZE_EMBEDDINGS", "0")))
    binary_center_mode = os.environ.get("BINARY_CENTER_MODE", "none")
    binary_ste_clip = float(os.environ.get("BINARY_STE_CLIP", 0.0))
    skip_roundtrip_eval = bool(int(os.environ.get("SKIP_ROUNDTRIP_EVAL", "0")))
    skip_final_val = bool(int(os.environ.get("SKIP_FINAL_VAL", "0")))
    save_model = bool(int(os.environ.get("SAVE_MODEL", "1")))
    save_debug_zlib = bool(int(os.environ.get("SAVE_DEBUG_ZLIB", "0")))

    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_prefix_tokens = int(os.environ.get("TTT_PREFIX_TOKENS", 4_194_304))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 1))
    ttt_batch_tokens = int(os.environ.get("TTT_BATCH_TOKENS", 65_536))
    ttt_lr = float(os.environ.get("TTT_LR", 1e-4))
    ttt_weight_decay = float(os.environ.get("TTT_WEIGHT_DECAY", 0.0))
    ttt_param_set = os.environ.get("TTT_PARAM_SET", "embed")
    ttt_mode = os.environ.get("TTT_MODE", "prefix")
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32_768))
    ttt_optimizer = os.environ.get("TTT_OPTIMIZER", "adamw")
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.0))
    ttt_grad_clip_norm = float(os.environ.get("TTT_GRAD_CLIP_NORM", 0.0))

    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    optimizer_name = os.environ.get("OPTIMIZER_NAME", "split_muon")
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    compile_model = bool(int(os.environ.get("TORCH_COMPILE", "1")))
    sdpa_enable_gqa = bool(int(os.environ.get("SDPA_ENABLE_GQA", "1")))
    ddp_enabled = bool(int(os.environ.get("DDP", "1")))

    out_dir = os.environ.get("OUT_DIR", "logs_h100_binary")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.local_train_batch_tokens // self.grad_accum_steps

    @property
    def local_train_batch_tokens(self) -> int:
        world = WORLD_SIZE if self.ddp_enabled else 1
        return max((self.train_batch_tokens // world // self.train_seq_len) * self.train_seq_len, self.train_seq_len)

    @property
    def effective_train_batch_tokens(self) -> int:
        world = WORLD_SIZE if self.ddp_enabled else 1
        return self.local_train_batch_tokens * world

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        warmdown_ms = self.warmdown_iters * step_ms
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,raw_scales,norm.weight,q_norm,k_norm",
    ).split(",") if p
)

FAMILY_MEDIAN_SCALE_RATIO = {
    "q_proj": 1.4171, "k_proj": 1.6763, "v_proj": 2.2108, "o_proj": 1.6933,
    "gate_proj": 1.8615, "up_proj": 2.0254, "down_proj": 1.5191, "embed_tokens": 1.0,
}


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int64, copy=False)


class TokenStream:
    def __init__(self, pattern: str, log_fn=None, dataset_name: str = ""):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = RANK % len(self.files) if WORLD_SIZE > 1 else 0
        self.epoch = 1
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = ((RANK // max(len(self.files), 1)) * 104729) % max(int(self.tokens.size) - 1, 1) if WORLD_SIZE > 1 else 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn:
                self.log_fn(f"WARNING: starting epoch:{self.epoch} dataset:{self.dataset_name} train_shards:{len(self.files)}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


class TokenLoader:
    def __init__(self, pattern: str, log_fn=None, dataset_name: str = ""):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        usable = (batch_tokens // seq_len) * seq_len
        chunk = self.stream.take(usable + 1)
        x = torch.from_numpy(chunk[:-1].reshape(-1, seq_len)).to(DEVICE, non_blocking=True)
        y = torch.from_numpy(chunk[1:].reshape(-1, seq_len)).to(DEVICE, non_blocking=True)
        return x, y


def ste_binary(x: torch.Tensor, center_mode: str, clip_value: float) -> torch.Tensor:
    if center_mode == "none":
        centered = x
    elif center_mode == "group_mean":
        centered = x - x.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unsupported BINARY_CENTER_MODE={center_mode}")
    q = torch.where(centered >= 0, torch.ones_like(x), -torch.ones_like(x))
    base = x.clamp(-clip_value, clip_value) if clip_value > 0 else x
    return base + (q - base).detach()


class RMSNormWeight(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight.to(x.dtype)


class GroupedSTELinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, args: Hyperparameters, family: str):
        super().__init__()
        if in_dim % args.quant_group_size != 0:
            raise ValueError(f"in_dim={in_dim} must be divisible by group size")
        self.group_size = args.quant_group_size
        self.center_mode = args.binary_center_mode
        self.clip_value = args.binary_ste_clip
        latent = torch.randn(out_dim, in_dim, dtype=torch.float32) / math.sqrt(in_dim)
        groups = latent.view(out_dim, in_dim // self.group_size, self.group_size)
        base_scale = groups.abs().mean(dim=-1) * FAMILY_MEDIAN_SCALE_RATIO[family]
        self.weight_logits = nn.Parameter(latent)
        self.raw_scales = nn.Parameter(torch.log(torch.expm1(base_scale) + 1e-6))

    def scales(self) -> torch.Tensor:
        return F.softplus(self.raw_scales) + 1e-6

    def quantized_weight(self) -> torch.Tensor:
        rows, cols = self.weight_logits.shape
        groups = self.weight_logits.view(rows, cols // self.group_size, self.group_size)
        signed = ste_binary(groups, self.center_mode, self.clip_value)
        return (signed * self.scales()[:, :, None]).reshape(rows, cols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantized_weight().to(x.dtype))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float, max_seq_len: int):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        cos = self.cos_cached[:, :, :seq_len, :].to(x.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(x.dtype)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class QwenAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, args: Hyperparameters):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_dim = args.rope_dim if args.rope_dim > 0 else self.head_dim
        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = GroupedSTELinear(dim, dim, args, "q_proj")
        self.k_proj = GroupedSTELinear(dim, kv_dim, args, "k_proj")
        self.v_proj = GroupedSTELinear(dim, kv_dim, args, "v_proj")
        self.o_proj = GroupedSTELinear(dim, dim, args, "o_proj")
        self.q_norm = RMSNormWeight(self.head_dim)
        self.k_norm = RMSNormWeight(self.head_dim)
        self.rope = RotaryEmbedding(self.rope_dim, args.rope_base, args.train_seq_len)
        self.scale = self.head_dim ** -0.5
        self.qk_gain = args.qk_gain
        self.sdpa_enable_gqa = args.sdpa_enable_gqa

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        if self.rope_dim == self.head_dim:
            return self.rope(x)
        return torch.cat((self.rope(x[..., :self.rope_dim]), x[..., self.rope_dim:]), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.apply_rope(self.q_norm(q).to(COMPUTE_DTYPE)) * self.qk_gain
        k = self.apply_rope(self.k_norm(k).to(COMPUTE_DTYPE)) * self.qk_gain
        enable_gqa = self.sdpa_enable_gqa and self.num_kv_heads != self.num_heads
        if self.num_kv_heads != self.num_heads and not enable_gqa:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v.to(q.dtype),
            dropout_p=0.0,
            is_causal=True,
            scale=self.scale,
            enable_gqa=enable_gqa,
        )
        return self.o_proj(y.transpose(1, 2).reshape(b, t, d))


class QwenMLP(nn.Module):
    def __init__(self, dim: int, args: Hyperparameters):
        super().__init__()
        hidden = int(dim * args.mlp_mult)
        self.mlp_act = args.mlp_act
        self.leaky_relu_slope = args.leaky_relu_slope
        self.gate_proj = GroupedSTELinear(dim, hidden, args, "gate_proj")
        self.up_proj = GroupedSTELinear(dim, hidden, args, "up_proj")
        self.down_proj = GroupedSTELinear(hidden, dim, args, "down_proj")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.mlp_act == "swiglu":
            act = F.silu(gate)
        elif self.mlp_act == "lrelu2_glu":
            act = F.leaky_relu(gate, negative_slope=self.leaky_relu_slope).square()
        else:
            raise ValueError(f"unsupported MLP_ACT={self.mlp_act}")
        return self.down_proj(act * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, layer_idx: int, args: Hyperparameters):
        super().__init__()
        self.parallel_residual = layer_idx >= args.parallel_residual_start_layer
        self.input_layernorm = RMSNormWeight(dim)
        self.post_attention_layernorm = RMSNormWeight(dim)
        self.attn = QwenAttention(dim, args.num_heads, args.num_kv_heads, args)
        self.mlp = QwenMLP(dim, args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parallel_residual:
            base = x
            return base + self.attn(self.input_layernorm(base)) + self.mlp(self.post_attention_layernorm(base))
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.embed_dim = args.embed_dim if args.embed_dim > 0 else args.model_dim
        self.tok_emb = nn.Embedding(args.vocab_size, self.embed_dim)
        nn.init.normal_(self.tok_emb.weight, std=args.tied_embed_init_std)
        if self.embed_dim != args.model_dim:
            self.embed_proj = nn.Linear(self.embed_dim, args.model_dim, bias=False)
            self.output_proj = nn.Linear(args.model_dim, self.embed_dim, bias=False)
            nn.init.normal_(self.embed_proj.weight, std=1.0 / math.sqrt(self.embed_dim))
            nn.init.normal_(self.output_proj.weight, std=1.0 / math.sqrt(args.model_dim))
        self.blocks = nn.ModuleList([Block(args.model_dim, i, args) for i in range(args.num_layers)])
        self.exec_order = [int(p) for p in args.layer_schedule.split(",") if p.strip()] if args.layer_schedule else list(range(args.num_layers))
        self.final_norm = RMSNormWeight(args.model_dim)

    def embed_weight(self) -> torch.Tensor:
        return self.tok_emb.weight

    def softcap(self, logits: torch.Tensor) -> torch.Tensor:
        c = self.args.logit_softcap
        z = logits / c
        if self.args.softcap_mode == "tanh":
            return c * torch.tanh(z)
        if self.args.softcap_mode == "poly5":
            z2 = z * z
            return c * torch.clamp(z - z * z2 / 3.0 + 2.0 * z * z2 * z2 / 15.0, -1.0, 1.0)
        raise ValueError(f"unsupported SOFTCAP_MODE={self.args.softcap_mode}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids).to(COMPUTE_DTYPE)
        if hasattr(self, "embed_proj"):
            x = self.embed_proj(x).to(COMPUTE_DTYPE)
        for idx in self.exec_order:
            x = self.blocks[idx](x)
        return self.final_norm(x)

    def loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        x = self.forward(input_ids)
        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        x = x.reshape(-1, self.embed_dim)
        y = target_ids.reshape(-1)
        logits = self.softcap(x @ self.embed_weight().to(x.dtype).T)
        loss = F.cross_entropy(logits.float(), y)
        if self.args.z_loss_coef > 0:
            loss = loss + self.args.z_loss_coef * torch.logsumexp(logits.float(), dim=-1).square().mean()
        return loss


def zeropower_newtonschulz5(g: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (x.norm() + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    return x.T if transposed else x


class MuonState:
    def __init__(self, params: list[nn.Parameter], args: Hyperparameters):
        self.params = params
        self.args = args
        self.buffers = [torch.zeros_like(p) for p in params]

    @torch.no_grad()
    def step(self, step: int, lr_mul: float):
        t = min(step / max(self.args.muon_momentum_warmup_steps, 1), 1.0)
        momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        for p, buf in zip(self.params, self.buffers):
            if p.grad is None:
                continue
            g = p.grad
            buf.mul_(momentum).add_(g)
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            p.add_(g_ortho.to(p.dtype), alpha=-lr * scale)


class SplitOptim:
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        named = list(model.named_parameters())
        embed, matrix, scalar = [], [], []
        for name, p in named:
            base_name = name.removeprefix("_orig_mod.")
            if base_name.startswith(("tok_emb.", "embed_proj.", "output_proj.")):
                embed.append(p)
            elif p.ndim == 2 and not any(pat in base_name for pat in CONTROL_TENSOR_NAME_PATTERNS):
                matrix.append(p)
            else:
                scalar.append(p)
        self.matrix_keys = matrix
        self.scalar_keys = scalar
        self.embed_keys = embed
        self.muon = MuonState(matrix, args)
        self.adam_embed = torch.optim.Adam(embed, lr=args.tied_embed_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)
        self.adam_scalar = torch.optim.Adam(scalar, lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)

    def zero_grad(self):
        for opt in (self.adam_embed, self.adam_scalar):
            opt.zero_grad(set_to_none=True)
        for p in self.matrix_keys:
            p.grad = None

    def step(self, step: int, lr_mul: float):
        self.muon.step(step, lr_mul)
        for group in self.adam_embed.param_groups:
            group["lr"] = self.args.tied_embed_lr * lr_mul
        for group in self.adam_scalar.param_groups:
            group["lr"] = self.args.scalar_lr * lr_mul
        self.adam_embed.step()
        self.adam_scalar.step()


def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int):
    table_size = max(int(sp.vocab_size()), vocab_size)
    base = np.zeros((table_size,), dtype=np.int16)
    lead = np.zeros((table_size,), dtype=np.bool_)
    boundary = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        boundary[token_id] = False
        if sp.is_byte(token_id):
            base[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            lead[token_id] = True
            piece = piece[1:]
        base[token_id] = len(piece.encode("utf-8"))
    return base, lead, boundary


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(f) for f in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


@torch.no_grad()
def eval_val(args, model, val_tokens, base_lut, lead_lut, boundary_lut, log_fn=None):
    model.eval()
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    val_batch_seqs = max(val_batch_tokens // args.train_seq_len, 1)
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        seq_end = min(seq_start + val_batch_seqs, total_seqs)
        raw_start = seq_start * args.train_seq_len
        raw_end = seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = torch.from_numpy(x_np).to(DEVICE)
        y = torch.from_numpy(y_np).to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=COMPUTE_DTYPE, enabled=DEVICE.type == "cuda"):
            loss = model.loss(x, y)
        n = float(y.numel())
        total_loss_sum += float(loss.item()) * n
        prev = x_np.reshape(-1)
        tgt = y_np.reshape(-1)
        bytes_np = base_lut[tgt].astype(np.int16, copy=True)
        bytes_np += (lead_lut[tgt] & ~boundary_lut[prev]).astype(np.int16, copy=False)
        total_tokens += n
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    model.train()
    val_loss = total_loss_sum / total_tokens
    val_bpb = (val_loss / math.log(2.0)) * (total_tokens / total_bytes)
    return val_loss, val_bpb


def ttt_parameters(model: nn.Module, param_set: str) -> list[nn.Parameter]:
    params = []
    for name, p in model.named_parameters():
        base_name = name.removeprefix("_orig_mod.")
        if param_set == "embed":
            keep = base_name.startswith(("tok_emb.", "embed_proj.", "output_proj.", "final_norm."))
        elif param_set == "scalar":
            keep = p.ndim < 2 or any(pat in base_name for pat in CONTROL_TENSOR_NAME_PATTERNS)
        elif param_set == "all":
            keep = True
        else:
            raise ValueError(f"unsupported TTT_PARAM_SET={param_set!r}; expected embed, scalar, or all")
        if keep:
            params.append(p)
    if not params:
        raise ValueError(f"TTT_PARAM_SET={param_set!r} selected no parameters")
    return params


def train_ttt_on_scored_prefix(args, model, scored_tokens: np.ndarray, log_fn=None):
    usable = ((scored_tokens.size - 1) // args.train_seq_len) * args.train_seq_len
    if usable <= 0 or args.ttt_epochs <= 0:
        return
    params = ttt_parameters(model, args.ttt_param_set)
    opt = torch.optim.AdamW(params, lr=args.ttt_lr, weight_decay=args.ttt_weight_decay)
    batch_tokens = max((args.ttt_batch_tokens // args.train_seq_len) * args.train_seq_len, args.train_seq_len)
    model.train()
    t0 = time.perf_counter()
    steps = 0
    for epoch in range(args.ttt_epochs):
        for start in range(0, usable, batch_tokens):
            end = min(start + batch_tokens, usable)
            end = start + ((end - start) // args.train_seq_len) * args.train_seq_len
            if end <= start:
                continue
            chunk = scored_tokens[start:end + 1]
            x = torch.from_numpy(chunk[:-1].reshape(-1, args.train_seq_len)).to(DEVICE)
            y = torch.from_numpy(chunk[1:].reshape(-1, args.train_seq_len)).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=COMPUTE_DTYPE, enabled=DEVICE.type == "cuda"):
                loss = model.loss(x, y)
            loss.backward()
            opt.step()
            steps += 1
            del x, y, loss
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    if log_fn:
        elapsed = 1000.0 * (time.perf_counter() - t0)
        log_fn(f"ttt_train:tokens:{usable} epochs:{args.ttt_epochs} steps:{steps} param_set:{args.ttt_param_set} lr:{args.ttt_lr} time:{elapsed:.0f}ms")
    del opt
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    model.eval()


def score_token_chunk(args, model, chunk: np.ndarray, base_lut, lead_lut, boundary_lut):
    usable = ((chunk.size - 1) // args.train_seq_len) * args.train_seq_len
    if usable <= 0:
        return 0.0, 0.0, 0.0
    chunk = chunk[: usable + 1]
    x_np = chunk[:-1].reshape(-1, args.train_seq_len)
    y_np = chunk[1:].reshape(-1, args.train_seq_len)
    x = torch.from_numpy(x_np).to(DEVICE)
    y = torch.from_numpy(y_np).to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=COMPUTE_DTYPE, enabled=DEVICE.type == "cuda"):
        loss = model.loss(x, y)
    loss_value = float(loss.item())
    n = float(y.numel())
    prev = x_np.reshape(-1)
    tgt = y_np.reshape(-1)
    bytes_np = base_lut[tgt].astype(np.int16, copy=True)
    bytes_np += (lead_lut[tgt] & ~boundary_lut[prev]).astype(np.int16, copy=False)
    total_bytes = float(bytes_np.astype(np.float64).sum())
    del x, y, loss
    return loss_value * n, n, total_bytes


def make_ttt_optimizer(args, params):
    if args.ttt_optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.ttt_lr, momentum=args.ttt_momentum, weight_decay=args.ttt_weight_decay)
    if args.ttt_optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.ttt_lr, weight_decay=args.ttt_weight_decay)
    raise ValueError(f"unsupported TTT_OPTIMIZER={args.ttt_optimizer!r}; expected adamw or sgd")


def train_ttt_on_scored_chunk(args, model, opt, scored_tokens: np.ndarray, lr: float):
    usable = ((scored_tokens.size - 1) // args.train_seq_len) * args.train_seq_len
    if usable <= 0 or args.ttt_epochs <= 0:
        return 0
    batch_tokens = max((args.ttt_batch_tokens // args.train_seq_len) * args.train_seq_len, args.train_seq_len)
    for group in opt.param_groups:
        group["lr"] = lr
    model.train()
    steps = 0
    for _ in range(args.ttt_epochs):
        for start in range(0, usable, batch_tokens):
            end = min(start + batch_tokens, usable)
            end = start + ((end - start) // args.train_seq_len) * args.train_seq_len
            if end <= start:
                continue
            chunk = scored_tokens[start:end + 1]
            x = torch.from_numpy(chunk[:-1].reshape(-1, args.train_seq_len)).to(DEVICE)
            y = torch.from_numpy(chunk[1:].reshape(-1, args.train_seq_len)).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=COMPUTE_DTYPE, enabled=DEVICE.type == "cuda"):
                loss = model.loss(x, y)
            loss.backward()
            if args.ttt_grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], args.ttt_grad_clip_norm)
            opt.step()
            steps += 1
            del x, y, loss
    model.eval()
    return steps


def eval_val_chunked_score_first_ttt(args, model, val_tokens, base_lut, lead_lut, boundary_lut, log_fn=None):
    model.eval()
    total_tokens_available = ((val_tokens.size - 1) // args.train_seq_len) * args.train_seq_len
    chunk_tokens = max((args.ttt_chunk_tokens // args.train_seq_len) * args.train_seq_len, args.train_seq_len)
    total_chunks = max((total_tokens_available + chunk_tokens - 1) // chunk_tokens, 1)
    params = ttt_parameters(model, args.ttt_param_set)
    opt = make_ttt_optimizer(args, params)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    total_ttt_steps = 0
    t0 = time.perf_counter()
    for chunk_idx, start in enumerate(range(0, total_tokens_available, chunk_tokens), start=1):
        end = min(start + chunk_tokens, total_tokens_available)
        end = start + ((end - start) // args.train_seq_len) * args.train_seq_len
        if end <= start:
            continue
        chunk = val_tokens[start:end + 1]
        loss_sum, n, byte_count = score_token_chunk(args, model, chunk, base_lut, lead_lut, boundary_lut)
        total_loss_sum += loss_sum
        total_tokens += n
        total_bytes += byte_count
        progress = (chunk_idx - 1) / max(total_chunks - 1, 1)
        lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        steps = train_ttt_on_scored_chunk(args, model, opt, chunk, lr)
        total_ttt_steps += steps
        if log_fn and (chunk_idx == 1 or chunk_idx == total_chunks or chunk_idx % 25 == 0):
            log_fn(
                f"ttt_chunk_progress:{chunk_idx}/{total_chunks} scored_tokens:{int(total_tokens)} "
                f"steps:{total_ttt_steps} lr:{lr:.6g}"
            )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    if log_fn:
        elapsed = 1000.0 * (time.perf_counter() - t0)
        log_fn(
            f"ttt_chunked_train:chunks:{total_chunks} chunk_tokens:{chunk_tokens} epochs:{args.ttt_epochs} "
            f"steps:{total_ttt_steps} optimizer:{args.ttt_optimizer} param_set:{args.ttt_param_set} "
            f"lr:{args.ttt_lr} momentum:{args.ttt_momentum} time:{elapsed:.0f}ms"
        )
    del opt
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    val_loss = total_loss_sum / total_tokens
    val_bpb = (val_loss / math.log(2.0)) * (total_tokens / total_bytes)
    return val_loss, val_bpb


def eval_val_score_first_ttt(args, model, val_tokens, base_lut, lead_lut, boundary_lut, log_fn=None):
    if args.ttt_mode == "chunked":
        return eval_val_chunked_score_first_ttt(args, model, val_tokens, base_lut, lead_lut, boundary_lut, log_fn=log_fn)
    if args.ttt_mode != "prefix":
        raise ValueError(f"unsupported TTT_MODE={args.ttt_mode!r}; expected prefix or chunked")
    model.eval()
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    val_batch_seqs = max(val_batch_tokens // args.train_seq_len, 1)
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    did_ttt = False
    ttt_after_tokens = max(args.ttt_prefix_tokens, args.train_seq_len)
    for batch_idx, seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        seq_end = min(seq_start + val_batch_seqs, total_seqs)
        raw_start = seq_start * args.train_seq_len
        raw_end = seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = torch.from_numpy(x_np).to(DEVICE)
        y = torch.from_numpy(y_np).to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=COMPUTE_DTYPE, enabled=DEVICE.type == "cuda"):
            loss = model.loss(x, y)
        n = float(y.numel())
        total_loss_sum += float(loss.item()) * n
        prev = x_np.reshape(-1)
        tgt = y_np.reshape(-1)
        bytes_np = base_lut[tgt].astype(np.int16, copy=True)
        bytes_np += (lead_lut[tgt] & ~boundary_lut[prev]).astype(np.int16, copy=False)
        total_tokens += n
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0):
            log_fn(f"ttt_val_progress:{batch_idx}/{total_batches}")
        if not did_ttt and total_tokens >= ttt_after_tokens and batch_idx < total_batches:
            scored_last_token_pos = raw_end - 1
            if log_fn:
                log_fn(f"ttt_score_first_pause:scored_tokens:{int(total_tokens)} scored_last_token_pos:{scored_last_token_pos}")
            train_ttt_on_scored_prefix(args, model, val_tokens[:raw_end], log_fn=log_fn)
            did_ttt = True
    val_loss = total_loss_sum / total_tokens
    val_bpb = (val_loss / math.log(2.0)) * (total_tokens / total_bytes)
    return val_loss, val_bpb


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def state_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}


def load_npz_into_model(model: nn.Module, path: str):
    data = np.load(path)
    own = model.state_dict()
    loaded = {}
    for key in data.files:
        if key in own and tuple(data[key].shape) == tuple(own[key].shape):
            loaded[key] = torch.from_numpy(data[key]).to(own[key].dtype)
    if not loaded:
        raise ValueError(f"loaded no matching tensors from {path}")
    own.update(loaded)
    model.load_state_dict(own, strict=True)


def estimate_native_bit_payload_bytes(flat_state: dict[str, np.ndarray], quant_mode: str):
    bit_bytes = scale_bytes = other_bytes = 0
    for name, arr in flat_state.items():
        if name.endswith(".weight_logits"):
            bit_bytes += (arr.size + 7) // 8
        elif name.endswith(".raw_scales"):
            scale_bytes += arr.size * 2
        else:
            other_bytes += arr.size * 2
    return bit_bytes + scale_bytes + other_bytes, {"bit_bytes": bit_bytes, "scale_bytes": scale_bytes, "other_bytes": other_bytes}


@torch.no_grad()
def average_gradients(params: list[nn.Parameter]) -> None:
    if not dist.is_available() or not dist.is_initialized() or WORLD_SIZE <= 1:
        return
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(WORLD_SIZE)


def should_log() -> bool:
    return RANK == 0


def main():
    args = Hyperparameters()
    if args.ddp_enabled and WORLD_SIZE > 1:
        dist.init_process_group(backend="nccl" if DEVICE.type == "cuda" else "gloo")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    if should_log():
        print(logfile)

    def log(msg: str, console: bool = True):
        if not should_log():
            return
        if console:
            print(msg, flush=True)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    log(Path(__file__).read_text(encoding="utf-8"), console=False)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    if args.eval_max_tokens > 0:
        keep = max((args.eval_max_tokens // args.train_seq_len) * args.train_seq_len + 1, args.train_seq_len + 1)
        val_tokens = val_tokens[: min(keep, val_tokens.size)]
    base_lut, lead_lut, boundary_lut = build_sentencepiece_luts(sp, args.vocab_size)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=Path(args.data_path).name)

    model = GPT(args).to(DEVICE)
    if args.init_model_npz:
        load_npz_into_model(model, args.init_model_npz)
        log(f"init_model_npz:{args.init_model_npz}")
    if args.compile_model and DEVICE.type == "cuda":
        model = torch.compile(model)
    opt = SplitOptim(model, args)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"run_id:{args.run_id}")
    log(f"torch_version:{torch.__version__} device:{DEVICE} compile:{args.compile_model} ddp_allreduce:{args.ddp_enabled and WORLD_SIZE > 1} world_size:{WORLD_SIZE}")
    log(f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings} embed_dim:{args.embed_dim}")
    log(f"quant_mode:{args.quant_mode} quant_group_size:{args.quant_group_size} binary_center_mode:{args.binary_center_mode} rope_dim:{args.rope_dim} mlp_act:{args.mlp_act} qk_gain:{args.qk_gain} softcap_mode:{args.softcap_mode}")
    log(f"iterations:{args.iterations} train_batch_tokens:{args.effective_train_batch_tokens} local_train_batch_tokens:{args.local_train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} val_batch_size:{args.val_batch_size} warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")
    log(f"optimizer:{args.optimizer_name} matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}")

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0
    model.train()
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if should_log() and ((last_step and not args.skip_final_val) or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0)):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            if args.ttt_enabled:
                val_loss, val_bpb = eval_val_score_first_ttt(args, model, val_tokens, base_lut, lead_lut, boundary_lut, log_fn=log)
                log(f"step:{step}/{args.iterations} ttt_val_loss:{val_loss:.4f} ttt_val_bpb:{val_bpb:.4f} train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms")
            else:
                val_loss, val_bpb = eval_val(args, model, val_tokens, base_lut, lead_lut, boundary_lut, log_fn=log)
                log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms")
            t0 = time.perf_counter()
        if last_step:
            if args.skip_final_val:
                train_time_ms += 1000.0 * (time.perf_counter() - t0)
                log(f"skipping_final_val:1 train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()
        opt.zero_grad()
        train_loss_value = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = train_loader.next_batch(args.microbatch_tokens, args.train_seq_len)
            with torch.autocast(device_type=DEVICE.type, dtype=COMPUTE_DTYPE, enabled=DEVICE.type == "cuda"):
                loss = model.loss(x, y) / args.grad_accum_steps
            loss.backward()
            train_loss_value += float(loss.detach().item())
        average_gradients(list(model.parameters()))
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        opt.step(step, lr_mul)
        torch.cuda.synchronize() if DEVICE.type == "cuda" else None
        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{args.effective_train_batch_tokens / (step_ms / 1000.0):.0f}")
        if max_wallclock_ms is not None and stop_after_step is None:
            stop_now = torch.tensor(
                [1 if approx_train_time_ms >= max_wallclock_ms else 0],
                device=DEVICE,
                dtype=torch.int32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stop_now, op=dist.ReduceOp.MAX)
            if int(stop_now.item()):
                stop_after_step = step

    if should_log() and args.save_model:
        out_path = out_dir / f"{args.run_id}_mlx_model.npz"
        flat_state = state_to_numpy(unwrap_model(model))
        np.savez(out_path, **flat_state)
        log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")
        native_bytes, breakdown = estimate_native_bit_payload_bytes(flat_state, args.quant_mode)
        log(f"estimated_native_payload:{native_bytes} bytes (bit:{breakdown['bit_bytes']} scale_fp16:{breakdown['scale_bytes']} other_fp16:{breakdown['other_bytes']})")
        if args.save_debug_zlib:
            quant_blob = zlib.compress(pickle.dumps(flat_state, protocol=pickle.HIGHEST_PROTOCOL), level=9)
            quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
            quant_path.write_bytes(quant_blob)
            log(f"serialized_model_int8_zlib:{quant_path.stat().st_size} bytes")
    elif should_log():
        log("skipping_model_save:1")
    if args.skip_roundtrip_eval:
        log("skipping_final_int8_zlib_roundtrip_eval:1")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
