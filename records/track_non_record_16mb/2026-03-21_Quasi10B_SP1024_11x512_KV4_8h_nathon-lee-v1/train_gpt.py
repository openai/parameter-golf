"""
train_gpt.py — 11-layer GPT for the 10min/16MB parameter-golf track

Architecture:
  - 11 transformer layers, d=512, 8 heads / 4 KV heads, 3× ReluSquared MLP
  - U-Net style encoder-decoder skip connections
  - Partial RoPE: rotary on 16 of 64 head dims for position-free generalization
  - LN Scale: RMSNorm damped by 1/sqrt(layer+1) for deep gradient stability
  - SmearGate: per-dim gate blending current + previous token embeddings
  - BigramHash(2048, dim=64->512): hash-based bigram context embeddings

Training:
  - Muon optimizer (Newton-Schulz) for 2D weights, momentum warmup 0.85->0.99
  - Adam (beta1=0.8, beta2=0.95) for scalars/embeddings, WD=0.04
  - Wallclock-aware cosine warmdown over last ~3000 steps
  - Orthogonal init with muP output-projection scaling
  - EMA (decay=0.997) of all parameters; EMA weights used for export

Compression:
  - Uniform int5 per-row quantization + int8 fallback
  - zstd-22 compression (zlib-9 fallback)
  - Target artifact ≤ 15.8 MB

Evaluation:
  - Sliding window with stride=64 for near-max context scoring
  - Full-model SGD TTT: 3 epochs over val, first 2 blocks frozen

Hardware target: 1×H100 PCIe, 6400s training + ≤240s eval+TTT
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import sys
import time
import uuid
import zlib
from pathlib import Path

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", 1)))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 50000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", 1)))

    # CHANGED: 4096->2048, 128->64
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 64))

    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", 1)))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", 1)))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))

    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))

    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))

    swa_pct = float(os.environ.get("SWA_PCT", 0.4))


# -----------------------------
# UTILITIES
# -----------------------------

def log0(msg: str) -> None:
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(msg, flush=True)

def _read_code() -> str:
    try:
        return Path(__file__).read_text()
    except Exception:
        return ""

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0


# -----------------------------
# DATA LOADING (with proper header skipping)
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    """Load a binary shard, skipping the 1024-byte (256 x int32) header."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[:usable + 1]


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
            chunks.append(self.tokens[self.pos:self.pos + k])
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
        local = chunk[start:start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# BPB COMPUTATION
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size: int, device: torch.device):
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


# -----------------------------
# QUANTIZATION
# -----------------------------

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict) -> Tensor:
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def _classify_param(name: str) -> str:
    if "mlp" in name:
        return "mlp"
    if ".attn." in name or ".c_q." in name or ".c_k." in name or ".c_v." in name:
        return "attn"
    return "other"

def quantize_intN_per_row(t: Tensor, clip_range: int = 31):
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / clip_range).clamp_min(1e-12).to(torch.float16)
        scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -clip_range, clip_range).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / clip_range, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale

def quantize_state_dict_mixed(state_dict: dict[str, Tensor]):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    stats = {"param_count": 0, "int8_payload_bytes": 0}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += int(t.numel())

        if not t.is_floating_point():
            result[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            meta[name] = "passthrough_nonfloat"
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            result[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            meta[name] = "passthrough"
            continue

        cat = _classify_param(name)
        if cat == "mlp" and t.ndim >= 2:
            q, s = quantize_intN_per_row(t, clip_range=15)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int5"}
            stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
        elif cat == "attn" and t.ndim >= 2:
            # CHANGED: int6 (clip_range=31) -> int5 (clip_range=15)
            q, s = quantize_intN_per_row(t, clip_range=15)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int5"}
            stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
            stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj = {
        "__quant_format__": "mixed_int5_int6_int8_v1",
        "result": result,
        "meta": meta,
        "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    return obj, stats

def dequantize_state_dict_mixed(obj, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    result = obj["result"]
    meta = obj["meta"]
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    out: dict[str, Tensor] = {}

    for name, orig in template_sd.items():
        info = meta[name]
        orig_dtype = orig.dtype

        if isinstance(info, str) and info.startswith("passthrough"):
            t = result[name]
            orig_dt_str = passthrough_orig_dtypes.get(name)
            if isinstance(orig_dt_str, str):
                t = t.to(dtype=getattr(torch, orig_dt_str))
            elif t.dtype != orig_dtype and t.is_floating_point():
                t = t.to(orig_dtype)
            out[name] = t
            continue

        q = result[name + ".q"]
        s = result[name + ".scale"]
        if q.ndim == 2 and s.ndim >= 1:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * s.float()).to(orig_dtype)

    return out


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
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len or self._cos_cached.device != device):
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


def apply_rotary_emb_partial(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    if rope_dims >= x.size(-1):
        return apply_rotary_emb(x, cos, sin)
    x_rot = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    half = rope_dims // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat([rotated, x_pass], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, rope_dims: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_dims = rope_dims if rope_dims > 0 else self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.rope_dims, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        if self.rope_dims < self.head_dim:
            q = apply_rotary_emb_partial(q, cos, sin, self.rope_dims)
            k = apply_rotary_emb_partial(k, cos, sin, self.rope_dims)
        else:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
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
        return self.proj(torch.relu(self.fc(x)).square())


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
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self._p1 = 36313
        self._p2 = 27191

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.long()
        B = self.bigram_vocab_size
        out = torch.zeros_like(t)
        if t.size(-1) > 1:
            out[..., 1:] = (self._p1 * t[..., 1:] + self._p2 * t[..., :-1]) % B
        return out

    def forward(self, token_ids: Tensor) -> Tensor:
        idx = self.bigram_hash(token_ids)
        h = self.embed(idx)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, rope_dims: int = 0,
                 layer_idx: int = 0, ln_scale: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self._ln_div = math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed = self.attn_norm(x)
        if self._ln_div != 1.0:
            normed = normed / self._ln_div
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(normed)
        normed2 = self.mlp_norm(x)
        if self._ln_div != 1.0:
            normed2 = normed2 / self._ln_div
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(normed2)
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, tie_embeddings: bool,
                 tied_embed_init_std: float, logit_softcap: float, rope_base: float,
                 qk_gain_init: float, bigram_vocab_size: int = 0, bigram_dim: int = 128,
                 rope_dims: int = 0, ln_scale: bool = False):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.model_dim = model_dim
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, std=tied_embed_init_std)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  rope_dims=rope_dims, layer_idx=i, ln_scale=ln_scale)
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        n_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and min(module.weight.shape) >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * n_layers))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               target_ids.reshape(-1), reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
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
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


# -----------------------------
# EVALUATION
# -----------------------------

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = max(1, local_batch_tokens // args.train_seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, seq_end)
            raw_s = batch_start * args.train_seq_len
            raw_e = batch_end * args.train_seq_len + 1
            local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y).detach()
            n = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * n
            val_token_count += n
            prev = x.reshape(-1)
            tgt = y.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.int16)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (val_loss_sum, val_token_count, val_byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = float(val_loss_sum / val_token_count)
    bpt = val_loss / math.log(2.0)
    tpb = float(val_token_count / val_byte_count)
    model.train()
    return val_loss, float(bpt * tpb)


def eval_val_sliding(args, model, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride: int, batch_seqs: int = 32):
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = list(range(0, total_tokens - seq_len + 1, stride))
    last_ws = window_starts[-1] if window_starts else 0
    if last_ws + seq_len < total_tokens:
        window_starts.append(total_tokens - seq_len)
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = float(loss_sum / token_count)
    bpb = float(val_loss / math.log(2) * token_count.item() / byte_count.item())
    model.train()
    return val_loss, bpb


# -----------------------------
# TTT (Test-Time Training)
# -----------------------------

def run_ttt_sgd(args, model, rank, world_size, device, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    model.train()
    frozen = set()
    for i in range(min(args.ttt_freeze_blocks, len(model.blocks))):
        for p in model.blocks[i].parameters():
            p.requires_grad_(False)
            frozen.add(id(p))
    model.tok_emb.weight.requires_grad_(False)
    frozen.add(id(model.tok_emb.weight))
    if model.bigram is not None:
        for p in model.bigram.parameters():
            p.requires_grad_(False)
            frozen.add(id(p))
    for p in model.smear.parameters():
        p.requires_grad_(False)
        frozen.add(id(p))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=args.ttt_lr, momentum=args.ttt_momentum)
    seq_len = args.train_seq_len
    total = val_tokens.numel() - 1
    n_chunks = total // seq_len

    for epoch in range(args.ttt_epochs):
        for ci in range(n_chunks):
            s = ci * seq_len
            chunk = val_tokens[s:s + seq_len + 1].to(device=device, dtype=torch.int64)
            x, y = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()
            optimizer.step()

    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()


# -----------------------------
# MAIN
# -----------------------------

def main() -> None:
    args = Hyperparameters()
    code = _read_code()

    distributed = int(os.environ.get("RANK", -1)) >= 0
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master = rank == 0
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log0(f"=== Parameter Golf: 11L PartialRoPE+LNScale+EMA+SWA+TTT ===")
    log0(f"run_id:{args.run_id} seed:{args.seed}")

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    ).to(device)
    restore_low_dim_params_to_fp32(model)

    n_params = sum(p.numel() for p in model.parameters())
    log0(f"model_params:{n_params}")

    # EMA shadow
    ema_state = None
    if args.ema_enabled:
        ema_state = {n: t.detach().clone() for n, t in model.state_dict().items()}

    # SWA collection
    swa_state = None
    swa_count = 0

    # Optimizer groups
    matrix_params, scalar_params, embed_params, head_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "tok_emb" in name:
            embed_params.append(p)
        elif name == "lm_head.weight":
            head_params.append(p)
        elif p.ndim == 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    optimizers: list = []
    if matrix_params:
        optimizers.append(Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                               backend_steps=args.muon_backend_steps))
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    if embed_params:
        optimizers.append(torch.optim.Adam(embed_params, lr=token_lr, betas=(args.beta1, args.beta2),
                                           eps=args.adam_eps))
    if scalar_params:
        optimizers.append(torch.optim.Adam(scalar_params, lr=args.scalar_lr, betas=(args.beta1, args.beta2),
                                           eps=args.adam_eps))
    if head_params:
        optimizers.append(torch.optim.Adam(head_params, lr=args.head_lr, betas=(args.beta1, args.beta2),
                                           eps=args.adam_eps))

    for opt in optimizers:
        if not isinstance(opt, Muon):
            for pg in opt.param_groups:
                pg["_base_lr"] = pg["lr"]

    total_seqs_per_step = args.train_batch_tokens // args.train_seq_len  # 512
    seqs_per_rank = total_seqs_per_step // world_size
    max_micro_seqs = int(os.environ.get("MICRO_BATCH_SEQS", 0))
    if max_micro_seqs <= 0:
        max_micro_seqs = seqs_per_rank  # 1 GPU=512, 8 GPU=64
    grad_accum_steps = max(1, seqs_per_rank // max_micro_seqs)
    micro_batch_seqs = seqs_per_rank // grad_accum_steps
    actual_tokens = grad_accum_steps * micro_batch_seqs * args.train_seq_len * world_size
    log0(f"grad_accum_steps:{grad_accum_steps} micro_batch_seqs:{micro_batch_seqs} tokens_per_step:{actual_tokens}")

    grad_scale = 1.0 / grad_accum_steps

    if distributed:
        ddp_model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    else:
        ddp_model = model

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    # Warmup compile
    if args.warmup_steps > 0:
        init_sd = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        ddp_model.train()
        for _ in range(args.warmup_steps):
            zero_grad_all()
            for micro in range(grad_accum_steps):
                if distributed:
                    ddp_model.require_backward_grad_sync = micro == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = ddp_model(x, y)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
        model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opt):
            o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        if args.ema_enabled:
            ema_state = {n: t.detach().clone() for n, t in model.state_dict().items()}

    # Main training loop
    training_time_ms = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    stopped_early = False
    while True:
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        last_step = step >= args.iterations
        if max_wallclock_ms is not None and elapsed_ms >= max_wallclock_ms:
            last_step = True
            stopped_early = True

        should_val = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_val:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vbpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stopped_early:
                log0(f"wallclock_stop at step:{step} train_time:{training_time_ms:.0f}ms")
            break

        # LR schedule: linear warmdown
        elapsed_ms_now = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        step_ms = elapsed_ms_now / max(step, 1) if step > 0 else 100.0
        if max_wallclock_ms is not None:
            remaining_ms = max(max_wallclock_ms - elapsed_ms_now, 0.0)
            warmdown_ms = args.warmdown_iters * step_ms
            lr_scale = min(remaining_ms / max(warmdown_ms, 1e-9), 1.0) if remaining_ms < warmdown_ms else 1.0
        else:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if step >= warmdown_start:
                lr_scale = max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            else:
                lr_scale = 1.0

        # Momentum warmup
        mom = args.muon_momentum
        if step < args.muon_momentum_warmup_steps:
            frac = step / args.muon_momentum_warmup_steps
            mom = args.muon_momentum_warmup_start + frac * (args.muon_momentum - args.muon_momentum_warmup_start)

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro in range(grad_accum_steps):
            if distributed:
                ddp_model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = ddp_model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        for opt in optimizers:
            if isinstance(opt, Muon):
                for pg in opt.param_groups:
                    pg["lr"] = args.matrix_lr * lr_scale
                    pg["momentum"] = mom
                opt.step()
            else:
                for pg in opt.param_groups:
                    pg["lr"] = pg["_base_lr"] * lr_scale
                opt.step()

        # Weight decay for Muon params
        if args.muon_wd > 0:
            with torch.no_grad():
                for p in matrix_params:
                    p.mul_(1 - args.matrix_lr * lr_scale * args.muon_wd)

        # EMA update
        if args.ema_enabled and ema_state is not None:
            with torch.no_grad():
                for name, param in model.state_dict().items():
                    ema_state[name].mul_(args.ema_decay).add_(param, alpha=1 - args.ema_decay)

        # SWA collection in last swa_pct of training
        if args.swa_pct > 0 and lr_scale < args.swa_pct:
            if swa_state is None:
                swa_state = {n: t.detach().clone() for n, t in model.state_dict().items()}
                swa_count = 1
            else:
                swa_count += 1
                with torch.no_grad():
                    for n, t in model.state_dict().items():
                        swa_state[n] += t

        if args.train_log_every > 0 and step % args.train_log_every == 0:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} lr_scale:{lr_scale:.4f}")

        step += 1

    # Choose best weights: SWA > EMA > raw
    if swa_state is not None and swa_count > 0:
        log0(f"loading_swa_weights count={swa_count}")
        with torch.no_grad():
            for n in swa_state:
                swa_state[n] /= swa_count
        model.load_state_dict(swa_state, strict=True)
    elif args.ema_enabled and ema_state is not None:
        log0("loading_ema_weights")
        model.load_state_dict(ema_state, strict=True)

    # Quantize & compress
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if master:
        torch.save(sd_cpu, "final_model.pt")
        log0(f"raw_model_bytes:{os.path.getsize('final_model.pt')}")

    quant_obj, q_stats = quantize_state_dict_mixed(sd_cpu)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    if _COMPRESSOR == "zstd":
        quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
    else:
        quant_blob = zlib.compress(quant_raw, 9)

    if master:
        with open("final_model.ptz", "wb") as f:
            f.write(quant_blob)
        artifact_bytes = os.path.getsize("final_model.ptz")
        code_bytes = len(code.encode("utf-8"))
        log0(f"artifact_bytes:{artifact_bytes} code_bytes:{code_bytes} total:{artifact_bytes + code_bytes}")

    if distributed:
        dist.barrier()

    # Roundtrip: decompress + dequantize
    with open("final_model.ptz", "rb") as f:
        blob = f.read()
    if _COMPRESSOR == "zstd":
        decompressed = zstandard.ZstdDecompressor().decompress(blob)
    else:
        decompressed = zlib.decompress(blob)
    loaded_obj = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
    deq_sd = dequantize_state_dict_mixed(loaded_obj, sd_cpu)
    model.load_state_dict(deq_sd, strict=True)

    # TTT before final eval
    if args.ttt_enabled:
        log0("starting_ttt")
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        run_ttt_sgd(args, model, rank, world_size, device, val_tokens,
                    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        torch.cuda.synchronize()
        log0(f"ttt_done:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")

    # Sliding window eval
    torch.cuda.synchronize()
    t_eval = time.perf_counter()
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        log0(f"eval:sliding stride={args.eval_stride}")
        final_loss, final_bpb = eval_val_sliding(
            args, model, rank, world_size, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, batch_seqs=args.eval_batch_seqs,
        )
    else:
        log0("eval:standard")
        final_loss, final_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
    torch.cuda.synchronize()
    log0(f"final val_loss:{final_loss:.6f} val_bpb:{final_bpb:.6f} eval_time:{1000.0 * (time.perf_counter() - t_eval):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()