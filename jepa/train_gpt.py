"""
Causal JEPA for the Parameter Golf challenge.

Architecture: a standard causal transformer (GQA + partial RoPE + BigramHash +
LeakyReLU(0.5)² + U-Net skips) with a JEPA auxiliary loss that encourages the
hidden representations to be predictive of future states.

The JEPA component (predictor + EMA target encoder) is used only during training
and is NOT included in the saved artifact. At inference, the model is a regular
autoregressive next-byte predictor evaluated with sliding-window + TTT.
"""

from __future__ import annotations

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
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import zstandard
except ImportError:
    zstandard = None


def env_flag(name: str, default: bool) -> bool:
    return bool(int(os.environ.get(name, "1" if default else "0")))


# ── hyperparameters ──────────────────────────────────────────────────────────

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/byte260_export/datasets/fineweb10B_byte260")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 200))
    max_val_tokens = int(os.environ.get("MAX_VAL_TOKENS", 131_072))
    final_max_val_tokens = int(os.environ.get("FINAL_MAX_VAL_TOKENS", 131_072))
    final_full_val = env_flag("FINAL_FULL_VAL", False)
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 20))

    iterations = int(os.environ.get("ITERATIONS", 200_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 10))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 260))
    pad_id = int(os.environ.get("PAD_ID", 0))
    bos_id = int(os.environ.get("BOS_ID", 1))
    byte_offset = int(os.environ.get("BYTE_OFFSET", 4))
    byte_count = int(os.environ.get("BYTE_COUNT", 256))

    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    partial_rope_dim = int(os.environ.get("PARTIAL_ROPE_DIM", 16))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 4096))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 64))

    jepa_weight = float(os.environ.get("JEPA_WEIGHT", 0.1))
    jepa_latent_dim = int(os.environ.get("JEPA_LATENT_DIM", 256))
    jepa_horizon = int(os.environ.get("JEPA_HORIZON", 8))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    int6_enabled = env_flag("INT6_ENABLED", True)
    use_zstd = env_flag("USE_ZSTD", True)
    zstd_level = int(os.environ.get("ZSTD_LEVEL", 22))

    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 16))

    ttt_enabled = env_flag("TTT_ENABLED", True)
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))

    use_compile = env_flag("USE_COMPILE", os.name != "nt")

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

    @property
    def num_skips(self) -> int:
        return (self.num_layers - 1) // 2


# ── muon optimizer ───────────────────────────────────────────────────────────

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.transpose(-2, -1) if transposed else X


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
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total = sum(int(p.numel()) for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.float32)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad.float()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / max(g.size(1), 1)) ** 0.5
                    flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                p.add_(flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype), alpha=-lr)
                curr += p.numel()
        return loss


# ── int6 quantization ────────────────────────────────────────────────────────

CONTROL_PATTERNS = tuple(
    p for p in os.environ.get("CONTROL_PATTERNS", "attn_scale,mlp_scale,skip_weights,smear").split(",") if p
)
INT_KEEP_FLOAT_MAX = 65_536
INT_CLIP_Q = 0.9999984
INT_CLIP_CANDIDATES = (1.0, 0.95, 0.9, 0.85, 0.8)


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def compress_bytes(payload: bytes, use_zstd: bool, zstd_level: int) -> tuple[bytes, str]:
    if use_zstd and zstandard is not None:
        return zstandard.ZstdCompressor(level=zstd_level).compress(payload), f"zstd-{zstd_level}"
    return zlib.compress(payload, level=9), "zlib-9"


def decompress_bytes(payload: bytes) -> bytes:
    try:
        return zlib.decompress(payload)
    except Exception:
        pass
    if zstandard is not None:
        return zstandard.ZstdDecompressor().decompress(payload)
    raise ValueError("Cannot decompress")


def quantize_scale_vector(scale: Tensor) -> tuple[Tensor, Tensor]:
    s = scale.float().clamp_min(1e-12)
    log_s = s.log()
    lo, hi = log_s.min(), log_s.max()
    if float((hi - lo).abs()) < 1e-12:
        q = torch.zeros_like(s, dtype=torch.uint8)
    else:
        q = torch.clamp(torch.round((log_s - lo) * (255.0 / (hi - lo))), 0, 255).to(torch.uint8)
    return q.contiguous(), torch.stack((lo, hi)).to(torch.float16).contiguous()


def dequantize_scale_vector(sq: Tensor, meta: Tensor) -> Tensor:
    lo, hi = meta.float()
    if float((hi - lo).abs()) < 1e-12:
        return torch.full(sq.shape, torch.exp(lo).item(), dtype=torch.float32)
    return torch.exp(lo + (sq.float() / 255.0) * (hi - lo))


def quantize_float_tensor(t: Tensor, levels: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        base = torch.quantile(t32.abs(), INT_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        best_q = best_s = best_err = None
        for mult in INT_CLIP_CANDIDATES:
            clip = (base * mult).clamp_min(1.0 / levels)
            clipped = torch.clamp(t32, -clip[:, None], clip[:, None])
            s = (clip / levels).clamp_min(1.0 / levels)
            q = torch.clamp(torch.round(clipped / s[:, None]), -levels, levels).to(torch.int8)
            err = ((q.float() * s[:, None]) - t32).pow(2).mean(dim=1)
            if best_err is None:
                best_q, best_s, best_err = q, s, err
            else:
                better = err < best_err
                best_q = torch.where(better[:, None], q, best_q)
                best_s = torch.where(better, s, best_s)
                best_err = torch.where(better, err, best_err)
        return best_q.contiguous(), best_s.to(torch.float16).contiguous()

    base = float(torch.quantile(t32.abs().flatten(), INT_CLIP_Q).item()) if t32.numel() else 0.0
    best_q = best_s = best_err = None
    for mult in INT_CLIP_CANDIDATES:
        clip = base * mult
        s = torch.tensor(clip / levels if clip > 0 else 1.0, dtype=torch.float32)
        clipped = torch.clamp(t32, -clip, clip) if clip > 0 else torch.zeros_like(t32)
        q = torch.clamp(torch.round(clipped / s), -levels, levels).to(torch.int8)
        err = torch.mean(((q.float() * s) - t32).pow(2))
        if best_err is None or err < best_err:
            best_q, best_s, best_err = q, s, err
    return best_q.contiguous(), best_s


def quantize_state_dict(sd: dict[str, Tensor], int6: bool):
    levels = 31 if int6 else 127
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    scale_qs: dict[str, Tensor] = {}
    scale_metas: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig: dict[str, str] = {}
    qmeta: dict[str, dict] = {}
    stats = {"param_count": 0, "payload_bytes": 0, "baseline_bytes": 0}

    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += int(t.numel())
        stats["baseline_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            passthrough[name] = t
            stats["payload_bytes"] += tensor_nbytes(t)
            continue
        if "byte_emb" in name or t.numel() <= INT_KEEP_FLOAT_MAX:
            if any(p in name for p in CONTROL_PATTERNS):
                kept = t.float().contiguous()
            elif t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(torch.float16).contiguous()
            else:
                kept = t
            passthrough[name] = kept
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue

        q, s = quantize_float_tensor(t, levels=levels)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        if levels < 127:
            qmeta.setdefault(name, {})["levels"] = levels
        quantized[name] = q
        if s.ndim > 0 and s.numel() > 32:
            sq, sm = quantize_scale_vector(s)
            scale_qs[name] = sq
            scale_metas[name] = sm
            qmeta.setdefault(name, {})["scale_scheme"] = "qscale"
            stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(sq) + tensor_nbytes(sm)
        else:
            scales[name] = s
            stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
        dtypes[name] = str(t.dtype).removeprefix("torch.")

    obj: dict = {
        "__quant_format__": "causal_jepa_int6_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if scale_qs:
        obj["scale_qs"] = scale_qs
        obj["scale_metas"] = scale_metas
    if passthrough_orig:
        obj["passthrough_orig_dtypes"] = passthrough_orig
    return obj, stats


def dequantize_state_dict(obj: dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        if qmeta.get(name, {}).get("scale_scheme") == "qscale":
            s = dequantize_scale_vector(obj["scale_qs"][name], obj["scale_metas"][name])
        else:
            s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        od = orig_dtypes.get(name)
        if isinstance(od, str):
            out_t = out_t.to(getattr(torch, od)).contiguous()
        out[name] = out_t
    return out


# ── data loading ─────────────────────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    n = int(header[2])
    off = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(file, dtype="<u2", count=n, offset=off)
    if tokens.size != n:
        raise ValueError(f"Short read: {file}")
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


def load_val_tokens(pattern: str) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    return torch.cat([load_data_shard(f) for f in files]).contiguous()


def cap_val_tokens(tokens: Tensor, max_tokens: int) -> Tensor:
    if max_tokens <= 0 or tokens.numel() <= max_tokens:
        return tokens
    return tokens[:max_tokens].contiguous()


def build_byte_lut(vocab_size: int, byte_offset: int, byte_count: int, device: torch.device) -> Tensor:
    out = torch.zeros(vocab_size, dtype=torch.int16, device=device)
    hi = min(vocab_size, byte_offset + byte_count)
    out[byte_offset:hi] = 1
    return out


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        parts: list[Tensor] = []
        rem = n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(rem, avail)
            parts.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            rem -= k
        return parts[0] if len(parts) == 1 else torch.cat(parts)


class DistributedLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum: int) -> Tensor:
        denom = self.world_size * grad_accum * seq_len
        if global_tokens % denom != 0:
            raise ValueError(f"TRAIN_BATCH_TOKENS={global_tokens} not divisible by {denom}")
        local_seqs = global_tokens // denom
        chunk = self.stream.take(local_seqs * seq_len * self.world_size)
        start = self.rank * local_seqs * seq_len
        local = chunk[start : start + local_seqs * seq_len].to(torch.int64)
        return local.reshape(local_seqs, seq_len).to(self.device, non_blocking=True)


# ── model components ─────────────────────────────────────────────────────────

def get_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def maybe_compile(obj, *, enabled: bool):
    if not enabled or not hasattr(torch, "compile"):
        return obj
    try:
        return torch.compile(obj, dynamic=False, fullgraph=False)
    except Exception:
        return obj


def make_adam(param_groups, *, beta1: float, beta2: float, eps: float):
    kwargs = dict(betas=(beta1, beta2), eps=eps)
    if torch.cuda.is_available():
        try:
            return torch.optim.Adam(param_groups, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(param_groups, **kwargs)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_len: int = 2048):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        self.dim = dim
        self.base = base
        self.train_len = train_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self._cos: Tensor | None = None
        self._sin: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos is None or self._cached_len != seq_len or self._cos.device != device:
            if seq_len > self.train_len:
                scale = seq_len / self.train_len
                adj_base = self.base * (scale ** (self.dim / max(self.dim - 2, 1)))
                inv = 1.0 / (adj_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
            else:
                inv = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv.dtype)
            freqs = torch.outer(t, inv)
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
            self._cached_len = seq_len
        return self._cos.to(dtype), self._sin.to(dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rd = cos.size(-1) * 2
    if rd == x.size(-1):
        h = x.size(-1) // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    xr, xp = x[..., :rd], x[..., rd:]
    h = rd // 2
    x1, x2 = xr[..., :h], xr[..., h:]
    return torch.cat((torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1), xp), dim=-1)


class BigramHash(nn.Module):
    def __init__(self, hash_vocab: int, bigram_dim: int, model_dim: int, bos_id: int = 1):
        super().__init__()
        self.hash_vocab = hash_vocab
        self.bos_id = bos_id
        self.embed = nn.Embedding(hash_vocab, bigram_dim)
        self.proj = nn.Linear(bigram_dim, model_dim, bias=False)

    def forward(self, ids: Tensor) -> Tensor:
        prev = F.pad(ids[:, :-1], (1, 0), value=self.bos_id)
        hashed = (prev * 263 + ids) % self.hash_vocab
        return self.proj(self.embed(hashed))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.weight.to(x.dtype), eps=self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, rotary: Rotary):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.rotary = rotary

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        kv = self.kv_proj(x).reshape(B, T, 2, self.num_kv_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]
        q = F.rms_norm(q, (self.head_dim,)).transpose(1, 2)
        k = F.rms_norm(k, (self.head_dim,)).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class TemporalSmear(nn.Module):
    """Learned per-dim EMA gate: blends x[t] with x[t-1] for local byte context."""
    def __init__(self, dim: int):
        super().__init__()
        self.smear = nn.Parameter(torch.full((dim,), -2.0))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.smear.to(x.dtype))
        return torch.lerp(x, F.pad(x[:, :-1], (0, 0, 1, 0)), g)


class CausalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, mlp_hidden: int, rotary: Rotary):
        super().__init__()
        self.smear = TemporalSmear(dim)
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, head_dim, rotary)
        self.mlp = MLP(dim, mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = self.smear(x)
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# ── CausalJEPA model ────────────────────────────────────────────────────────

class CausalJEPA(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.pad_id = args.pad_id
        self.num_layers = args.num_layers
        self.num_skips = args.num_skips
        self.logit_softcap = args.logit_softcap
        self.jepa_weight = args.jepa_weight
        self.jepa_horizon = args.jepa_horizon
        self.ema_decay = args.ema_decay

        dim = args.model_dim
        hd = args.head_dim
        mlp_hidden = int(dim * args.mlp_mult)
        rotary = Rotary(args.partial_rope_dim, base=args.rope_base, train_len=args.train_seq_len)

        self.byte_emb = nn.Embedding(args.vocab_size, dim)
        self.bigram_hash = BigramHash(args.bigram_vocab_size, args.bigram_dim, dim, args.bos_id)
        self.rotary = rotary
        self.blocks = nn.ModuleList([
            CausalBlock(dim, args.num_heads, args.num_kv_heads, hd, mlp_hidden, rotary)
            for _ in range(args.num_layers)
        ])
        self.skip_weights = nn.Parameter(torch.full((self.num_skips, dim), 0.1))
        self.final_norm = RMSNorm(dim)

        ld = args.jepa_latent_dim
        self.jepa_predictor = nn.Sequential(
            nn.Linear(dim, ld, bias=False), nn.GELU(), nn.Linear(ld, ld, bias=False)
        )
        self.jepa_decode_proj = nn.Linear(ld, dim, bias=False)
        self.jepa_target_proj = nn.Linear(dim, ld, bias=False)

        self._init_weights()

        self.ema_byte_emb = copy.deepcopy(self.byte_emb)
        self.ema_blocks = copy.deepcopy(self.blocks)
        self.ema_skip_weights = nn.Parameter(self.skip_weights.data.clone())
        self.ema_final_norm = copy.deepcopy(self.final_norm)
        self.ema_target_proj = copy.deepcopy(self.jepa_target_proj)
        self.jepa_target_proj.requires_grad_(False)
        for p in self._ema_params():
            p.requires_grad_(False)

    def _init_weights(self):
        nn.init.normal_(self.byte_emb.weight, std=0.02)
        nn.init.normal_(self.bigram_hash.embed.weight, std=0.02)
        scale = (2 * self.num_layers) ** -0.5
        for block in self.blocks:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.xavier_uniform_(block.attn.o_proj.weight)
            block.attn.o_proj.weight.data.mul_(scale)
            nn.init.xavier_uniform_(block.mlp.proj.weight)
            block.mlp.proj.weight.data.mul_(scale)
        for m in (self.jepa_predictor, self.jepa_target_proj):
            for p in m.modules():
                if isinstance(p, nn.Linear):
                    nn.init.xavier_uniform_(p.weight)
        nn.init.zeros_(self.jepa_decode_proj.weight)

    def _ema_params(self):
        yield from self.ema_byte_emb.parameters()
        yield from self.ema_blocks.parameters()
        yield self.ema_skip_weights
        yield from self.ema_final_norm.parameters()
        yield from self.ema_target_proj.parameters()

    def _online_params(self):
        yield from self.byte_emb.parameters()
        yield from self.blocks.parameters()
        yield self.skip_weights
        yield from self.final_norm.parameters()
        yield from self.jepa_target_proj.parameters()

    @torch.no_grad()
    def _sync_ema(self):
        for ep, op in zip(self._ema_params(), self._online_params()):
            ep.data.copy_(op.data)

    @torch.no_grad()
    def update_ema(self):
        d = self.ema_decay
        for ep, op in zip(self._ema_params(), self._online_params()):
            ep.data.lerp_(op.data, 1.0 - d)

    def _run_backbone(self, ids: Tensor, byte_emb, blocks, skip_w, final_norm) -> Tensor:
        x = byte_emb(ids) + self.bigram_hash(ids)
        skips: list[Tensor] = []
        pop_idx = 0
        for i, block in enumerate(blocks):
            x = block(x)
            if i < self.num_skips:
                skips.append(x)
            elif i >= self.num_layers - self.num_skips:
                x = x + skip_w[pop_idx][None, None, :].to(x.dtype) * skips.pop()
                pop_idx += 1
        return final_norm(x)

    def _jepa_logits(self, hidden: Tensor) -> Tensor:
        """Augment hidden states with JEPA predictions before computing logits.
        Predictor output is detached so the predictor is trained only by JEPA
        loss; decode_proj learns from CE how to use the predictor's output."""
        pred_latent = self.jepa_predictor(hidden)
        augmented = hidden + self.jepa_decode_proj(pred_latent.detach())
        logits = F.linear(augmented, self.byte_emb.weight)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward_logits(self, ids: Tensor) -> Tensor:
        hidden = self._run_backbone(ids, self.byte_emb, self.blocks, self.skip_weights, self.final_norm)
        return self._jepa_logits(hidden)

    @torch.no_grad()
    def _ema_forward(self, ids: Tensor) -> Tensor:
        return self._run_backbone(ids, self.ema_byte_emb, self.ema_blocks, self.ema_skip_weights, self.ema_final_norm)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self._run_backbone(input_ids, self.byte_emb, self.blocks, self.skip_weights, self.final_norm)

        pred_latent = self.jepa_predictor(hidden)
        augmented = hidden + self.jepa_decode_proj(pred_latent.detach())
        logits = F.linear(augmented[:, :-1], self.byte_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        ce = F.cross_entropy(logits.float().reshape(-1, self.vocab_size), input_ids[:, 1:].reshape(-1))

        K = self.jepa_horizon
        if K > 0 and hidden.size(1) > K:
            with torch.no_grad():
                ema_h = self._ema_forward(input_ids)
                target = self.ema_target_proj(ema_h[:, K:])
            jepa_loss = F.mse_loss(
                F.normalize(pred_latent[:, :-K].float(), dim=-1),
                F.normalize(target.float(), dim=-1),
            )
            total = ce + self.jepa_weight * jepa_loss
        else:
            jepa_loss = ce.new_zeros(())
            total = ce
        return total, ce.detach(), jepa_loss.detach()

    def serializable_state_dict(self) -> dict[str, Tensor]:
        skip = ("ema_", "jepa_target_proj.", "rotary.")
        return {k: v for k, v in self.state_dict().items() if not any(k.startswith(s) for s in skip)}

    def load_serializable(self, sd: dict[str, Tensor]):
        self.load_state_dict(sd, strict=False)
        self._sync_ema()


    @torch.no_grad()
    def zero_pad_emb_(self):
        if 0 <= self.pad_id < self.byte_emb.num_embeddings:
            self.byte_emb.weight[self.pad_id].zero_()


# ── evaluation ───────────────────────────────────────────────────────────────

def eval_val_sliding(
    args: Hyperparameters,
    model: CausalJEPA,
    rank: int, world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    byte_lut: Tensor,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    ctx_len = args.train_seq_len
    stride = args.eval_stride
    total = val_tokens.numel() - 1
    ws_list = [ws for ws in range(0, total, stride) if min(ws + ctx_len, total) - ws >= 1]
    my_s = (len(ws_list) * rank) // world_size
    my_e = (len(ws_list) * (rank + 1)) // world_size
    my_ws = ws_list[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_ws), args.eval_batch_seqs):
            batch_ws = my_ws[bi : bi + args.eval_batch_seqs]
            bsz = len(batch_ws)
            x = torch.zeros(bsz, ctx_len, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, ctx_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + ctx_len, total)
                wl = end - ws
                wlens.append(wl)
                chunk = val_tokens[ws : end + 1].to(dtype=torch.int64, device=device)
                x[i, :wl] = chunk[:-1]
                y[i, :wl] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model.forward_logits(x)
            nll = F.cross_entropy(
                logits.reshape(-1, model.vocab_size).float(), y.reshape(-1), reduction="none"
            ).reshape(bsz, ctx_len)
            for i, ws in enumerate(batch_ws):
                wl = wlens[i]
                s = 0 if ws == 0 else max(wl - stride, 0)
                scored = nll[i, s:wl].to(torch.float64)
                loss_sum += scored.sum()
                tok_count += float(wl - s)
                byte_cnt += byte_lut[y[i, s:wl]].to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_cnt, op=dist.ReduceOp.SUM)

    val_loss = float((loss_sum / tok_count).item())
    val_bpb = float(loss_sum.item() / (math.log(2.0) * byte_cnt.item()))
    model.train()
    return val_loss, val_bpb


def eval_val_ttt(
    args: Hyperparameters,
    model: CausalJEPA,
    device: torch.device,
    val_tokens: Tensor,
    byte_lut: Tensor,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    ctx_len = args.train_seq_len
    total = val_tokens.numel() - 1
    chunk_size = args.ttt_chunk_tokens

    original = {k: v.cpu().clone() for k, v in model.state_dict().items()
                if not k.startswith("ema_") and not k.startswith("jepa_") and not k.startswith("rotary.")}

    ttt_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith("ema_") and not n.startswith("jepa_")]
    ttt_opt = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

    chunk_starts = list(range(0, total, chunk_size))
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt = torch.zeros((), device=device, dtype=torch.float64)

    for ci, cs in enumerate(chunk_starts):
        ce = min(cs + chunk_size, total)

        # SCORE
        model.eval()
        scored_windows = []
        with torch.inference_mode():
            for ws in range(cs, ce, ctx_len):
                we = min(ws + ctx_len, ce)
                wl = we - ws
                if wl < 1:
                    continue
                chunk = val_tokens[ws : we + 1].to(dtype=torch.int64, device=device)
                x = chunk[:-1].unsqueeze(0)
                y_tgt = chunk[1:].unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model.forward_logits(x).float()
                nll = -F.log_softmax(logits, dim=-1).gather(-1, y_tgt.unsqueeze(-1)).squeeze(-1)
                nll_sum += nll.to(torch.float64).sum()
                tok_cnt += float(y_tgt.numel())
                byte_cnt += byte_lut[y_tgt[0]].to(torch.float64).sum()
                if wl == ctx_len:
                    scored_windows.append(chunk.detach())

        # ADAPT (skip last chunk)
        if ci < len(chunk_starts) - 1 and scored_windows:
            model.train()
            batch = torch.stack(scored_windows)
            for _ep in range(args.ttt_epochs):
                perm = torch.randperm(batch.size(0), device=device)
                for si in range(0, batch.size(0), max(1, batch.size(0) // 4)):
                    mini = batch[perm[si : si + max(1, batch.size(0) // 4)]]
                    x_m = mini[:, :-1]
                    y_m = mini[:, 1:]
                    ttt_opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model.forward_logits(x_m)
                    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size).float(), y_m.reshape(-1))
                    loss.backward()
                    if args.ttt_grad_clip > 0:
                        gn = torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        if not torch.isfinite(gn):
                            ttt_opt.zero_grad(set_to_none=True)
                            continue
                    ttt_opt.step()

    model.load_state_dict(original, strict=False)
    model.train()
    val_loss = float((nll_sum / tok_cnt).item())
    val_bpb = float(nll_sum.item() / (math.log(2.0) * byte_cnt.item()))
    return val_loss, val_bpb


# ── training ─────────────────────────────────────────────────────────────────

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    env_ws = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = env_ws > 1
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    world_size = env_ws if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    default_ga = max(8 // world_size, 1) if 8 % world_size == 0 else 1
    grad_accum = int(os.environ.get("GRAD_ACCUM_STEPS", str(default_ga)))
    grad_scale = 1.0 / grad_accum

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl")
        dist.barrier()
    master = rank == 0

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = get_amp_dtype()

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True):
        if not master:
            return
        if console:
            print(msg)
        if logfile:
            try:
                with open(logfile, "a", encoding="utf-8") as f:
                    print(msg, file=f)
            except OSError:
                pass

    log0(f"code_bytes:{len(code.encode('utf-8'))}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path(args.data_path).resolve()
    n_shards = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if n_shards == 0:
        raise FileNotFoundError(f"No byte shards in {dataset_dir}")

    val_tokens = load_val_tokens(args.val_files)
    periodic_val = cap_val_tokens(val_tokens, args.max_val_tokens)
    final_val = val_tokens if args.final_full_val else cap_val_tokens(val_tokens, args.final_max_val_tokens)
    byte_lut = build_byte_lut(args.vocab_size, args.byte_offset, args.byte_count, device)
    log0(f"data:{dataset_dir.name} shards:{n_shards} val:{val_tokens.numel()} periodic:{periodic_val.numel()} final:{final_val.numel()}")

    base_model = CausalJEPA(args).to(device)
    zeropower_via_newtonschulz5 = maybe_compile(zeropower_via_newtonschulz5, enabled=args.use_compile)
    train_model = maybe_compile(base_model, enabled=args.use_compile)
    if distributed:
        model = DDP(train_model, device_ids=[local_rank], broadcast_buffers=False)
    else:
        model = train_model

    named = [(n, p) for n, p in base_model.named_parameters() if p.requires_grad]
    embed_names = {"byte_emb.weight", "bigram_hash.embed.weight"}
    embed_ps = [p for n, p in named if n in embed_names]
    matrix_ps = [p for n, p in named if p.ndim == 2 and n not in embed_names]
    scalar_ps = [p for n, p in named if p.ndim < 2 and n not in embed_names]

    optimizers: list[torch.optim.Optimizer] = []
    if embed_ps:
        opt_e = make_adam([{"params": embed_ps, "lr": args.embed_lr, "base_lr": args.embed_lr}],
                         beta1=args.beta1, beta2=args.beta2, eps=args.adam_eps)
        optimizers.append(opt_e)
    opt_m = Muon(matrix_ps, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in opt_m.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizers.append(opt_m)
    if scalar_ps:
        opt_s = make_adam([{"params": scalar_ps, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                         beta1=args.beta1, beta2=args.beta2, eps=args.adam_eps)
        optimizers.append(opt_s)

    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    n_saved = sum(v.numel() for v in base_model.serializable_state_dict().values())
    log0(f"params_total:{n_params} params_saved:{n_saved}")
    log0(f"arch: dim={args.model_dim} layers={args.num_layers} heads={args.num_heads}/{args.num_kv_heads} "
         f"mlp={args.mlp_mult}x rope_dim={args.partial_rope_dim} bigram={args.bigram_vocab_size}x{args.bigram_dim} "
         f"skips={args.num_skips} softcap={args.logit_softcap} smear=1")
    log0(f"jepa: weight={args.jepa_weight} latent_dim={args.jepa_latent_dim} "
         f"horizon={args.jepa_horizon} ema={args.ema_decay}")
    log0(f"train: seq={args.train_seq_len} batch_tokens={args.train_batch_tokens} ga={grad_accum} "
         f"warmdown={args.warmdown_iters} warmup={args.warmup_steps}")
    log0(f"eval: stride={args.eval_stride} ttt={int(args.ttt_enabled)} int6={int(args.int6_enabled)} "
         f"zstd={int(args.use_zstd)} seed={args.seed}")

    loader = DistributedLoader(args.train_files, rank, world_size, device)

    def zero_grad():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    max_wc_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wc_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            if ws <= step < args.iterations:
                return max((args.iterations - step) / args.warmdown_iters, 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem = max(max_wc_ms - elapsed_ms, 0.0)
        return rem / max(wd_ms, 1e-9) if rem <= wd_ms else 1.0

    # Warmup (prime optimizer momentum buffers, then reset)
    if args.warmup_steps > 0:
        init_sd = {n: t.cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad()
            for ms in range(grad_accum):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum - 1
                batch = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss, _, _ = model(batch)
                (loss * grad_scale).backward()
            for o in optimizers:
                o.step()
            base_model.zero_pad_emb_()
            zero_grad()
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opt):
            o.load_state_dict(s)
        base_model._sync_ema()
        base_model.zero_pad_emb_()
        zero_grad()
        loader = DistributedLoader(args.train_files, rank, world_size, device)

    train_ms = 0.0
    stop_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last = step == args.iterations or (stop_step is not None and step >= stop_step)

        if last or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            ev = final_val if last else periodic_val
            vl, vb = eval_val_sliding(args, base_model, rank, world_size, device, ev, byte_lut, amp_dtype)
            scope = "final" if last else "periodic"
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"val_scope:{scope} val_tokens:{ev.numel()} "
                 f"train_time:{train_ms:.0f}ms step_avg:{train_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last:
            if stop_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock train_time:{train_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed = train_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed)
        zero_grad()
        t_loss = torch.zeros((), device=device)
        t_ce = torch.zeros((), device=device)
        t_jepa = torch.zeros((), device=device)
        for ms in range(grad_accum):
            if distributed:
                model.require_backward_grad_sync = ms == grad_accum - 1
            batch = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, ce_d, jepa_d = model(batch)
            t_loss += loss.detach()
            t_ce += ce_d
            t_jepa += jepa_d
            (loss * grad_scale).backward()
        t_loss /= grad_accum
        t_ce /= grad_accum
        t_jepa /= grad_accum

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        mm = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_m.param_groups:
            g["momentum"] = mm
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers:
            o.step()
        base_model.update_ema()
        base_model.zero_pad_emb_()
        zero_grad()

        step += 1
        approx_ms = train_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{t_loss.item():.4f} "
                 f"train_ce:{t_ce.item():.4f} train_jepa:{t_jepa.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")

        cap = max_wc_ms is not None and approx_ms >= max_wc_ms
        if distributed and max_wc_ms is not None:
            ct = torch.tensor(int(cap), device=device)
            dist.all_reduce(ct, op=dist.ReduceOp.MAX)
            cap = bool(ct.item())
        if stop_step is None and cap:
            stop_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if master:
        torch.save(base_model.serializable_state_dict(), "final_model.pt")
        log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes")
        log0(f"Code size: {len(code.encode('utf-8'))} bytes")

    qobj, qstats = quantize_state_dict(base_model.serializable_state_dict(), int6=args.int6_enabled)
    buf = io.BytesIO()
    torch.save(qobj, buf)
    raw = buf.getvalue()
    blob, comp_name = compress_bytes(raw, args.use_zstd, args.zstd_level)
    if master:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(blob)
        sz = os.path.getsize("final_model.int6.ptz")
        code_sz = len(code.encode("utf-8"))
        ratio = qstats["baseline_bytes"] / max(qstats["payload_bytes"], 1)
        log0(f"Quantized model ({comp_name}): {sz} bytes (payload:{qstats['payload_bytes']} ratio:{ratio:.2f}x)")
        log0(f"Total submission: {sz + code_sz} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        blob_disk = f.read()
    qstate = torch.load(io.BytesIO(decompress_bytes(blob_disk)), map_location="cpu", weights_only=False)
    base_model.load_serializable(dequantize_state_dict(qstate))
    base_model.zero_pad_emb_()

    torch.cuda.synchronize()
    t_qe = time.perf_counter()
    q_loss, q_bpb = eval_val_sliding(args, base_model, rank, world_size, device, final_val, byte_lut, amp_dtype)
    torch.cuda.synchronize()
    scope = "full" if args.final_full_val else "capped"
    log0(f"final_quant_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} "
         f"eval_scope:{scope} val_tokens:{final_val.numel()} eval_time:{1000*(time.perf_counter()-t_qe):.0f}ms")
    log0(f"final_quant_roundtrip_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")

    if args.ttt_enabled and world_size == 1:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_ttt(args, base_model, device, final_val, byte_lut, amp_dtype)
        torch.cuda.synchronize()
        log0(f"final_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
             f"ttt_gain:{ttt_bpb - q_bpb:.4f} ttt_time:{1000*(time.perf_counter()-t_ttt):.0f}ms")
        log0(f"final_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
