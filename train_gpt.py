from __future__ import annotations

import argparse
import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from contextlib import nullcontext
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# Enable flash/mem-efficient SDP on CUDA (H100); math-only fallback on CPU for dev.
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
else:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# -----------------------------
# HYPERPARAMETERS
# -----------------------------

def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)



def hyperparameters() -> SimpleNamespace:
    return SimpleNamespace(
        dataset=env_str("DATASET", "fineweb"),
        data_path=env_str("DATA_PATH", "./data/datasets/fineweb10B_sp1024"),
        train_files=env_str("TRAIN_FILES", "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"),
        val_files=env_str("VAL_FILES", "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"),
        tokenizer_path=env_str("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"),
        vocab_size=env_int("VOCAB_SIZE", 1024),
        num_layers=env_int("NUM_LAYERS", 2),
        model_dim=env_int("MODEL_DIM", 128),
        num_heads=env_int("NUM_HEADS", 4),
        num_kv_heads=env_int("NUM_KV_HEADS", 4),
        mlp_mult=env_int("MLP_MULT", 1),
        tie_embeddings=bool(int(os.environ.get("TIE_EMBEDDINGS", "1"))),
        tied_embed_init_std=env_float("TIED_EMBED_INIT_STD", 0.02),
        logit_softcap=env_float("LOGIT_SOFTCAP", 1.0),
        rope_base=env_float("ROPE_BASE", 10000.0),
        qk_gain_init=env_float("QK_GAIN_INIT", 1.0),
        shared_block_size=env_int("SHARED_BLOCK_SIZE", 1),
        recurrence_steps=env_int("RECURRENCE_STEPS", 1),
        recurrence_gate_init=env_float("RECURRENCE_GATE_INIT", 0.43),
        train_seq_len=env_int("TRAIN_SEQ_LEN", 64),
        train_batch_tokens=env_int("TRAIN_BATCH_TOKENS", 64),
        grad_accum_steps=env_int("GRAD_ACCUM_STEPS", 1),
        iterations=env_int("ITERATIONS", 80),
        warmup_steps=env_int("WARMUP_STEPS", 0),
        warmup_restore_model=bool(int(os.environ.get("WARMUP_RESTORE_MODEL", "1"))),
        checkpoint_every=env_int("CHECKPOINT_EVERY", 0),
        preload_shards=bool(int(os.environ.get("PRELOAD_SHARDS", "0"))),
        warmdown_iters=env_int("WARMDOWN_ITERS", 0),
        max_wallclock_seconds=env_float("MAX_WALLCLOCK_SECONDS", 0.0),
        beta1=env_float("BETA1", 0.9),
        beta2=env_float("BETA2", 0.95),
        adam_eps=env_float("ADAM_EPS", 1e-8),
        embed_lr=env_float("EMBED_LR", 3e-4),
        tied_embed_lr=env_float("TIED_EMBED_LR", 2e-3),
        head_lr=env_float("HEAD_LR", 4e-4),
        matrix_lr=env_float("MATRIX_LR", 3e-5),
        scalar_lr=env_float("SCALAR_LR", 2e-3),
        muon_momentum=env_float("MUON_MOMENTUM", 0.95),
        muon_backend_steps=env_int("MUON_BACKEND_STEPS", 3),
        muon_momentum_warmup_start=env_float("MUON_MOMENTUM_WARMUP_START", 0.85),
        muon_momentum_warmup_steps=env_int("MUON_MOMENTUM_WARMUP_STEPS", 750),
        grad_clip_norm=env_float("GRAD_CLIP_NORM", 1.0),
        val_batch_size=env_int("VAL_BATCH_SIZE", 256),
        val_loss_every=env_int("VAL_LOSS_EVERY", 80),
        val_heartbeat_every=env_int("VAL_HEARTBEAT_EVERY", 0),
        eval_seq_len=env_int("EVAL_SEQ_LEN", 0),
        eval_stride=env_int("EVAL_STRIDE", 0),
        train_log_every=env_int("TRAIN_LOG_EVERY", 1),
        seed=env_int("SEED", 3928752),
        dtype=env_str("DTYPE", "float32"),
        compile=bool(int(os.environ.get("COMPILE", "0"))),
        attn_noise_scale=env_float("ATTN_NOISE_SCALE", 0.0),
        attn_noise_decay=env_float("ATTN_NOISE_DECAY", 1.0),
        x_gate_init=env_float("X_GATE_INIT", 0.85),
        id_pull_init=env_float("ID_PULL_INIT", 0.0),
        ttt_enabled=bool(int(os.environ.get("TTT_ENABLED", "0"))),
        ttt_lora_rank=env_int("TTT_LORA_RANK", 8),
        ttt_lora_lr=env_float("TTT_LORA_LR", 0.01),
        ttt_chunk_size=env_int("TTT_CHUNK_SIZE", 256),
        ttt_eval_seq_len=env_int("TTT_EVAL_SEQ_LEN", 1024),
        ttt_batch_size=env_int("TTT_BATCH_SIZE", 64),
    )


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

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
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# DATA + TOKENIZER UTILS
# -----------------------------

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
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
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
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        if local_tokens < seq_len:
            raise ValueError(
                f"TRAIN_BATCH_TOKENS too small: local_tokens={local_tokens}, seq_len={seq_len}, "
                f"world_size={self.world_size}, grad_accum_steps={grad_accum_steps}. "
                f"Need TRAIN_BATCH_TOKENS >= {seq_len * self.world_size * grad_accum_steps}."
            )
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)



def load_validation_tokens(pattern: str, seq_len: int, vocab_size: int | None = None) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    shards = []
    for file in files:
        toks = load_data_shard(file).to(torch.int64).contiguous()
        if vocab_size is not None and toks.numel():
            tmax = int(toks.max().item())
            if tmax >= vocab_size:
                raise ValueError(
                    f"[val_loader] token id exceeds vocab after shard decode: max={tmax} vs vocab={vocab_size} file={file}"
                )
        shards.append(toks)

    tokens = torch.cat(shards).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]



def kHop(sp, expected_vocab_size=None, strict=True):
    vocab_size = int(sp.vocab_size())
    if expected_vocab_size is not None and vocab_size != expected_vocab_size and strict:
        raise ValueError(f"Tokenizer vocab ({vocab_size}) != expected ({expected_vocab_size})")
    return vocab_size



def build_sentencepiece_luts(sp, vocab_size, device):
    base_bytes = []
    has_leading_space = []
    is_boundary_token = []

    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        leading_space = piece.startswith("▁")
        has_leading_space.append(leading_space)
        normalized = piece[1:] if leading_space else piece
        normalized = normalized.replace("<0x0A>", "\n")
        try:
            b = normalized.encode("utf-8")
        except Exception:
            b = b""
        base_bytes.append(len(b))
        is_boundary_token.append(leading_space)

    base_bytes_lut = torch.tensor(base_bytes, dtype=torch.int32, device=device)
    has_leading_space_lut = torch.tensor(has_leading_space, dtype=torch.bool, device=device)
    is_boundary_token_lut = torch.tensor(is_boundary_token, dtype=torch.bool, device=device)
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


# -----------------------------
# MODEL MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight", "skip_weights", "x_gate", "id_pull"
)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
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
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                adjusted_base = self.base * (scale ** (self.dim / (self.dim - 2)))
                inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)



def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)



def pifPxy(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    training: bool = True,
    noise_scale: float = 0.0,
    noise_decay: float = 1.0,
    step_idx: int | None = None,
) -> Tensor:
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    if is_causal:
        tq, tk = q.size(-2), k.size(-2)
        causal_mask = torch.ones((tq, tk), device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    if training and noise_scale > 0.0:
        scale = noise_scale if step_idx is None else noise_scale * (noise_decay ** step_idx)
        scores = scores + scale * torch.randn_like(scores)

    attn = torch.softmax(scores, dim=-1)
    if training and dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return torch.matmul(attn, v)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        train_seq_len: int = 1024,
        dropout: float = 0.0,
        attn_noise_scale: float = 0.0,
        attn_noise_decay: float = 1.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.attn_noise_scale = attn_noise_scale
        self.attn_noise_decay = attn_noise_decay
        self._train_step = 0

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)

    def forward(self, x: Tensor, q_delta: Tensor | None = None, v_delta: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        if q_delta is not None:
            q = q + q_delta
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x)
        if v_delta is not None:
            v = v + v_delta
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if k.size(1) != q.size(1):
            if q.size(1) % k.size(1) != 0:
                raise RuntimeError(f"num_heads={q.size(1)} not divisible by num_kv_heads={k.size(1)}")
            repeat = q.size(1) // k.size(1)
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if self.training and self.attn_noise_scale > 0.0:
            y = pifPxy(
                q, k, v,
                dropout_p=self.dropout,
                is_causal=True,
                training=True,
                noise_scale=self.attn_noise_scale,
                noise_decay=self.attn_noise_decay,
                step_idx=self._train_step,
            )
            self._train_step += 1
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
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
        train_seq_len: int = 1024,
        attn_noise_scale: float = 0.0,
        attn_noise_decay: float = 1.0,
        x_gate_init: float = 0.85,
        id_pull_init: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            train_seq_len=train_seq_len,
            attn_noise_scale=attn_noise_scale,
            attn_noise_decay=attn_noise_decay,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.x_gate = nn.Parameter(torch.full((dim,), x_gate_init, dtype=torch.float32))
        self.id_pull = nn.Parameter(torch.full((dim,), id_pull_init, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor, q_delta_fn=None, v_delta_fn=None) -> Tensor:
        dtype = x.dtype
        pull = torch.tanh(self.id_pull).to(dtype=dtype)[None, None, :]
        x_gate = torch.sigmoid(self.x_gate).to(dtype=dtype)[None, None, :]
        mix = self.resid_mix.to(dtype=dtype)
        live = mix[0][None, None, :]
        ident = mix[1][None, None, :]

        x_carried = live * x + ident * x0 + pull * x0
        x_normed = self.attn_norm(x_carried)
        qd = q_delta_fn(x_normed) if q_delta_fn is not None else None
        vd = v_delta_fn(x_normed) if v_delta_fn is not None else None
        attn_out = self.attn(x_normed, qd, vd)
        mlp_out = self.mlp(self.mlp_norm(x_carried))

        candidate = x_carried
        candidate = candidate + self.attn_scale.to(dtype=dtype)[None, None, :] * attn_out
        candidate = candidate + self.mlp_scale.to(dtype=dtype)[None, None, :] * mlp_out

        return x_gate * x + (1.0 - x_gate) * candidate


@dataclass
class KMotifGateConfig:
    base_gate: float = 0.43
    late_gate: float = 0.18
    gate_min: float = 0.12
    gate_max: float = 0.62
    motif_eps: float = 0.10
    smooth: float = 0.35
    decay_motif_amplitude: bool = True


class KMotifGateWalk:
    """Bounded dynamic gate driven by |3n-2| kMotif walk with {-2,5} residue modulation."""

    def __init__(self, cfg: KMotifGateConfig):
        self.cfg = cfg
        self._prev_gate = cfg.base_gate

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    @staticmethod
    def _residue_branch(m: int) -> float:
        r7 = m % 7
        if r7 == 5:
            return +1.00
        elif r7 == 2:
            return -0.75
        elif r7 in (1, 6):
            return +0.40
        elif r7 in (0, 3):
            return -0.25
        else:
            return +0.15

    @staticmethod
    def _microcycle(m: int) -> float:
        return (-0.12, -0.04, +0.04, +0.12)[m % 4]

    def value(self, step_idx: int, max_steps: int) -> float:
        progress = 0.0 if max_steps <= 1 else step_idx / float(max_steps - 1)
        base = self._lerp(self.cfg.base_gate, self.cfg.late_gate, progress)
        m = abs(3 * step_idx - 2)
        motif = self._residue_branch(m) + self._microcycle(m)
        eps = self.cfg.motif_eps
        if self.cfg.decay_motif_amplitude:
            eps = self._lerp(eps, eps * 0.45, progress)
        raw_gate = base * (1.0 + eps * motif)
        raw_gate = self._clamp(raw_gate, self.cfg.gate_min, self.cfg.gate_max)
        gate = self._lerp(raw_gate, self._prev_gate, self.cfg.smooth)
        gate = self._clamp(gate, self.cfg.gate_min, self.cfg.gate_max)
        self._prev_gate = gate
        return gate


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
        train_seq_len: int = 1024,
        shared_block_size: int = 1,
        recurrence_steps: int = 1,
        recurrence_gate_init: float = 0.45,
        attn_noise_scale: float = 0.0,
        attn_noise_decay: float = 1.0,
        x_gate_init: float = 0.85,
        id_pull_init: float = 0.0,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.shared_block_size = max(1, shared_block_size)
        self.recurrence_steps = max(1, recurrence_steps)
        self.shared_blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    train_seq_len=train_seq_len,
                    attn_noise_scale=attn_noise_scale,
                    attn_noise_decay=attn_noise_decay,
                    x_gate_init=x_gate_init,
                    id_pull_init=id_pull_init,
                )
                for _ in range(self.shared_block_size)
            ]
        )
        self.recurrence_gates = nn.Parameter(
            self._kmotif_gate_init(self.recurrence_steps, self.shared_block_size, model_dim, recurrence_gate_init)
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    @staticmethod
    def _kmotif_gate_init(recurrence_steps: int, shared_block_size: int, model_dim: int, avg_tau: float) -> Tensor:
        """Non-uniform gate init following |3n-2| kMotif pattern, averaging to avg_tau."""
        weights = torch.tensor([abs(3 * (n + 1) - 2) for n in range(recurrence_steps)], dtype=torch.float32)
        per_step = avg_tau * recurrence_steps * weights / weights.sum()
        # shape: (recurrence_steps,) → (recurrence_steps, shared_block_size, model_dim)
        return per_step[:, None, None].expand(recurrence_steps, shared_block_size, model_dim).contiguous()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
        for i, block in enumerate(self.shared_blocks):
            with torch.no_grad():
                phase = torch.sigmoid(torch.tensor(3.0 * (i / max(len(self.shared_blocks) - 1, 1) - 0.5)))
                block.resid_mix.data[0] = phase * torch.ones(block.resid_mix.shape[1])
                block.resid_mix.data[1] = (1 - phase) * torch.ones(block.resid_mix.shape[1])

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        for step in range(self.recurrence_steps):
            for bi, block in enumerate(self.shared_blocks):
                x_prev = x
                if lora is not None:
                    x = block(x, x0, q_delta_fn=lora.q_loras[bi], v_delta_fn=lora.v_loras[bi])
                else:
                    x = block(x, x0)
                gate = torch.sigmoid(self.recurrence_gates[step, bi]).to(dtype=x.dtype)[None, None, :]
                x = gate * x + (1.0 - gate) * x_prev

        x = self.final_norm(x)

        if lora is not None:
            if self.tie_embeddings:
                logits = F.linear(x, self.tok_emb.weight)
            else:
                logits = self.lm_head(x)
            logits = logits + lora.lm_head_lora(x)
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
            bsz, sl, V = logits.shape
            return F.cross_entropy(
                logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
            ).reshape(bsz, sl)

        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        for step in range(self.recurrence_steps):
            for bi, block in enumerate(self.shared_blocks):
                x_prev = x
                x = block(x, x0)
                gate = torch.sigmoid(self.recurrence_gates[step, bi]).to(dtype=x.dtype)[None, None, :]
                x = gate * x + (1.0 - gate) * x_prev
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight.to(x.dtype))
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


# -----------------------------
# TEST-TIME TRAINING (LoRA)
# -----------------------------

TTT_BOS_ID = 1


class BatchedLinearLoRA(nn.Module):
    """LoRA for a linear layer, with independent weights per batch element.
    Computes x @ Aᵀ @ Bᵀ  =  x @ (BA)ᵀ,  i.e. the LoRA delta is ΔW = BA."""
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()


class BatchedTTTLoRA(nn.Module):
    """All LoRA adapters for one batch: LM head and Q/V per shared block."""
    def __init__(self, bsz: int, model: GPT, rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.shared_blocks:
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_v.weight.shape[0], rank))

    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()


def _ttt_reset_optimizer(opt: torch.optim.Optimizer) -> None:
    for group in opt.param_groups:
        for p in group["params"]:
            s = opt.state.get(p)
            if not s:
                continue
            s["exp_avg"].zero_()
            s["exp_avg_sq"].zero_()
            s["step"].fill_(0)


def _ttt_build_optimizer(lora: BatchedTTTLoRA, lr: float, beta1: float, beta2: float) -> torch.optim.Adam:
    return torch.optim.Adam(lora.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-10)


def _ttt_find_docs(all_tokens: Tensor) -> list[tuple[int, int]]:
    """Return (start_offset, length) for each document, identified by BOS boundaries."""
    bos_positions = (all_tokens == TTT_BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else all_tokens.numel()
        if i + 1 < len(bos_positions):
            end += 1
        if end - start >= 2:
            docs.append((start, end - start))
    return docs


def _ttt_chunk_window(ci: int, pred_len: int, num_chunks: int, chunk_size: int, eval_seq_len: int):
    """Return (win_start, win_len, chunk_offset, chunk_len) for chunk `ci` of a doc."""
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def eval_val_ttt(
    args: SimpleNamespace,
    model: GPT,
    rank: int,
    world_size: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Evaluate with batched LoRA test-time training. Returns (val_loss, val_bpb)."""
    files = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _ttt_find_docs(all_tokens)

    rank_docs = docs[(len(docs) * rank) // world_size: (len(docs) * (rank + 1)) // world_size]
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len
    batch_size = args.ttt_batch_size
    lora_rank = args.ttt_lora_rank

    rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    lora = BatchedTTTLoRA(batch_size, model, lora_rank).to(device)
    opt = _ttt_build_optimizer(lora, args.ttt_lora_lr, args.beta1, args.beta2)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    use_cuda = device.type == "cuda"

    for bi in range(0, len(rank_docs), batch_size):
        batch = rank_docs[bi:bi + batch_size]
        bsz = len(batch)

        if bsz == batch_size:
            cur_lora, cur_opt = lora, opt
            cur_lora.reset()
            _ttt_reset_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, model, lora_rank).to(device)
            cur_opt = _ttt_build_optimizer(cur_lora, args.ttt_lora_lr, args.beta1, args.beta2)

        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)

        for ci in range(max_nc):
            chunk_stats = _ttt_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
            context_size, chunk_offset = chunk_stats[1], chunk_stats[2]

            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc - 1 for nc in num_chunks)

            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            doc_info = []
            for b in range(bsz):
                if not active[b]:
                    doc_info.append((0, 0))
                    continue
                ds, dl = batch[b]
                ws, wl, co, cl = _ttt_chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                chunk = all_tokens[ds + ws: ds + ws + wl + 1]
                toks = chunk.to(dtype=torch.int64, device=device)
                x[b, :wl] = toks[:-1]
                y[b, :wl] = toks[1:]
                doc_info.append((co, cl))

            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()
            if needs_train:
                with ctx:
                    ptl = model(x, y, lora=cur_lora)
            else:
                with torch.no_grad(), ctx:
                    ptl = model(x, y, lora=cur_lora)

            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]:
                        continue
                    co, cl = doc_info[b]
                    lbl = ptl[b, co:co + cl].to(torch.float64)
                    prev = x[b, co:co + cl]
                    tgt = y[b, co:co + cl]
                    tok_bytes = base_bytes_lut[tgt].to(torch.float64)
                    tok_bytes += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    loss_sum += lbl.sum()
                    byte_sum += tok_bytes.sum()
                    token_count += cl

            if needs_train:
                mask = torch.tensor([float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device)
                per_doc = ptl[:, chunk_offset:chunk_offset + chunk_size].mean(dim=-1)
                cur_opt.zero_grad()
                (per_doc * mask).sum().backward()
                cur_opt.step()

    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb


# -----------------------------
# EVAL + QUANTIZATION
# -----------------------------

def eval_val(
    args: SimpleNamespace,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len_override: int = 0,
) -> tuple[float, float]:
    seq_len = seq_len_override if seq_len_override > 0 else args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        batch_idx = 0
        total_batches = max(1, math.ceil((seq_end - seq_start) / local_batch_seqs))
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    batch_loss = model(x, y).detach()
            else:
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
            batch_idx += 1
            if rank == 0 and args.val_heartbeat_every > 0 and (
                batch_idx % args.val_heartbeat_every == 0 or batch_idx == total_batches
            ):
                print(f"val_progress:{batch_idx}/{total_batches}")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())



def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t



def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
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
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)

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

        if "tok_emb" in name:
            kept = t.to(dtype=torch.float16).contiguous()
            passthrough[name] = kept
            passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
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
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# TRAINING
# -----------------------------

def modelWrapFn(model, args=None, device=None):
    if device is not None:
        model = model.to(device)
    # enable cudnn autotune for stable input sizes on GPU
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    dtype = getattr(args, "dtype", "float32") if args else "float32"
    model = model.bfloat16() if dtype == "bfloat16" else model.float()
    if getattr(args, "compile", False):
        try:
            model = torch.compile(model)
        except Exception:
            pass
    return model


last_loss = None


def heartbeat(step, loss, model, optimizer_muon):
    global last_loss
    try:
        g = torch.sigmoid(model.recurrence_gates.detach()).mean().item()
        m = optimizer_muon.param_groups[0]["momentum"]
        d = 0.0 if last_loss is None else loss - last_loss
        last_loss = loss
        print(f"hb | step:{step} loss:{loss:.4f} dL:{d:+.5f} G:{g:.3f} M:{m:.3f}", flush=True)
    except Exception:
        pass



def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = hyperparameters()

    parser = argparse.ArgumentParser()
    # minimal CLI overrides so env/config can be overridden without editing code
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--train_files", type=str, default=None)
    parser.add_argument("--val_files", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--num_kv_heads", type=int, default=None)
    parser.add_argument("--train_seq_len", type=int, default=None)
    parser.add_argument("--train_batch_tokens", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tie_embeddings", type=int, choices=[0,1], default=None)
    parser.add_argument("--compile", type=int, choices=[0,1], default=None)
    ns = parser.parse_args()
    # apply CLI overrides if provided
    if ns.tokenizer_path is not None:
        args.tokenizer_path = ns.tokenizer_path
    if ns.train_files is not None:
        args.train_files = ns.train_files
    if ns.val_files is not None:
        args.val_files = ns.val_files
    if ns.data_path is not None:
        args.data_path = ns.data_path
    if ns.vocab_size is not None:
        args.vocab_size = ns.vocab_size
    if ns.model_dim is not None:
        args.model_dim = ns.model_dim
    if ns.num_heads is not None:
        args.num_heads = ns.num_heads
    if ns.num_kv_heads is not None:
        args.num_kv_heads = ns.num_kv_heads
    if ns.train_seq_len is not None:
        args.train_seq_len = ns.train_seq_len
    if ns.train_batch_tokens is not None:
        args.train_batch_tokens = ns.train_batch_tokens
    if ns.iterations is not None:
        args.iterations = ns.iterations
    if ns.seed is not None:
        args.seed = ns.seed
    if ns.tie_embeddings is not None:
        args.tie_embeddings = bool(ns.tie_embeddings)
    if ns.compile is not None:
        args.compile = bool(ns.compile)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    grad_accum_steps = max(1, args.grad_accum_steps)
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        # pass an integer device id (local_rank / device.index) to init_process_group
        device_id = device.index if hasattr(device, "index") and device.type == "cuda" else local_rank
        try:
            dist.init_process_group(backend="nccl", device_id=device_id)
        except Exception:
            # fallback for CPU or non-nccl environments
            dist.init_process_group(backend="gloo")
        dist.barrier()
    master_process = rank == 0

    # Validate batch-token divisibility to avoid silent shape errors in loaders
    if args.train_batch_tokens % (max(1, world_size) * grad_accum_steps * args.train_seq_len) != 0:
        raise ValueError(
            f"TRAIN_BATCH_TOKENS ({args.train_batch_tokens}) must be divisible by WORLD_SIZE*GRAD_ACCUM_STEPS*TRAIN_SEQ_LEN ({world_size}*{grad_accum_steps}*{args.train_seq_len})"
        )
    if args.val_batch_size % (max(1, world_size) * grad_accum_steps * args.train_seq_len) != 0:
        raise ValueError(
            f"VAL_BATCH_SIZE ({args.val_batch_size}) must be divisible by WORLD_SIZE*GRAD_ACCUM_STEPS*TRAIN_SEQ_LEN ({world_size}*{grad_accum_steps}*{args.train_seq_len})"
        )

    logfile = None
    # prefer /workspace for output on RunPod (root overlay is only 5GB)
    _ws = Path("/workspace/wpSoupBuild") if Path("/workspace").is_dir() else Path(".")
    if master_process:
        log_dir = _ws / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
        logfile = str(log_dir / f"{run_id}.txt")
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)
    _flash = device.type == "cuda"
    log0(f"sdp_backends:flash={_flash} mem_efficient={_flash} math=True manual_gqa_expand=True")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor()
    ok = sp.Load(args.tokenizer_path)
    if not ok:
        raise FileNotFoundError(f"Could not load tokenizer: {args.tokenizer_path}")
    tokenizer_vocab_size = kHop(sp, args.vocab_size, strict=False)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, vocab_size=tokenizer_vocab_size)
    dataset_dir = Path(args.data_path).resolve()
    # prefer explicit train_files pattern when counting shards
    try:
        actual_train_files = len(list(glob.glob(args.train_files))) if args.train_files else len(list(dataset_dir.glob("fineweb_train_*.bin")))
    except Exception:
        actual_train_files = 0
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    model = GPT(
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
        train_seq_len=args.train_seq_len,
        shared_block_size=args.shared_block_size,
        recurrence_steps=args.recurrence_steps,
        recurrence_gate_init=args.recurrence_gate_init,
        attn_noise_scale=args.attn_noise_scale,
        attn_noise_decay=args.attn_noise_decay,
        x_gate_init=args.x_gate_init,
        id_pull_init=args.id_pull_init,
    )
    model = modelWrapFn(model, args, device)
    restore_low_dim_params_to_fp32(model)
    # determine autocast dtype from args.dtype for GPU runs
    autocast_dtype = None
    if use_cuda:
        if args.dtype == "bfloat16":
            autocast_dtype = torch.bfloat16
        elif args.dtype == "float16" or args.dtype == "fp16":
            autocast_dtype = torch.float16
        else:
            autocast_dtype = None

    block_named_params = list(model.shared_blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    # recurrence_gates are walk-scheduled, not learned
    model.recurrence_gates.requires_grad_(False)
    gate_walk = KMotifGateWalk(KMotifGateConfig())
    gate_weights = torch.tensor(
        [abs(3 * (n + 1) - 2) for n in range(model.recurrence_steps)], dtype=torch.float32
    )
    gate_weights_norm = model.recurrence_steps * gate_weights / gate_weights.sum()
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr

    optimizer_tok = torch.optim.AdamW(
        [{"params": [model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=0.01,
        fused=False,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=0.01,
        fused=False,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=False,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"motif:shared_block_size:{args.shared_block_size} recurrence_steps:{args.recurrence_steps} recurrence_gate_init:{args.recurrence_gate_init}")
    log0(f"gate_walk:base={gate_walk.cfg.base_gate} late={gate_walk.cfg.late_gate} eps={gate_walk.cfg.motif_eps} smooth={gate_walk.cfg.smooth}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} head_lr:{args.head_lr if model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # reconcile legacy "num_layers" flag with shared_block_size
    if getattr(args, "num_layers", None) is not None and args.shared_block_size == 1 and args.num_layers > 1:
        log0(f"note: using num_layers={args.num_layers} as shared_block_size")
        args.shared_block_size = args.num_layers

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # optionally preload shards into memory for faster I/O during dev runs
    if getattr(args, "preload_shards", False):
        try:
            # force stream to read entire pattern into memory by taking a large chunk
            _ = train_loader.stream.take(10_000_000)
        except Exception:
            pass

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                if use_cuda and autocast_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        warmup_loss = model(x, y)
                else:
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        # option to restore model and optimizer states after warmup
        if getattr(args, "warmup_restore_model", True):
            model.load_state_dict(initial_model_state, strict=True)
            for opt, state in zip(optimizers, initial_optimizer_states):
                opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0)
        if should_validate:
            if use_cuda:
                torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        with torch.no_grad():
            walk_tau = gate_walk.value(step, args.iterations)
            model.recurrence_gates.data.copy_(
                (walk_tau * gate_weights_norm)[:, None, None].expand_as(model.recurrence_gates)
            )
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            if use_cuda and autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss = model(x, y)
            else:
                loss = model(x, y)
            heartbeat(step + 1, float(loss.item()), model, optimizer_muon)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        with torch.no_grad():
            for p in matrix_params:
                p.mul_(1.0 - 0.02 * optimizer_muon.param_groups[0]["lr"])
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

        # periodic checkpointing (save CPU-state dict to records/<RUN_ID>/)
        if getattr(args, "checkpoint_every", 0) > 0 and master_process and step % args.checkpoint_every == 0:
            try:
                run_id_ck = os.environ.get("RUN_ID", "run")
                ck_dir = _ws / "records" / run_id_ck
                ck_dir.mkdir(parents=True, exist_ok=True)
                export_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                ck_path = ck_dir / f"checkpoint_step_{step}.pt"
                torch.save(export_state, ck_path)
                log0(f"Saved checkpoint: {ck_path}")
            except Exception:
                pass

    peak_alloc = torch.cuda.max_memory_allocated() // 1024 // 1024 if use_cuda else 0
    peak_reserved = torch.cuda.max_memory_reserved() // 1024 // 1024 if use_cuda else 0
    log0(f"peak memory allocated: {peak_alloc} MiB reserved: {peak_reserved} MiB")

    export_state = dict(model.state_dict())
    if master_process:
        _final_pt = str(_ws / "final_model.pt")
        torch.save(export_state, _final_pt)
        model_bytes = os.path.getsize(_final_pt)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(export_state)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    _final_int8 = str(_ws / "final_model.int8.ptz")
    if master_process:
        with open(_final_int8, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(_final_int8)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=False)

    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")

    if getattr(args, "ttt_enabled", False):
        log0(f"ttt_lora:starting rank={args.ttt_lora_rank} lr={args.ttt_lora_lr} "
             f"chunk={args.ttt_chunk_size} eval_seq={args.ttt_eval_seq_len} batch={args.ttt_batch_size}")
        ttt_loss, ttt_bpb = eval_val_ttt(
            args, model, rank, world_size, device,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        log0(f"ttt_lora val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
