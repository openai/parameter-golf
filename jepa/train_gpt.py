"""
Two-stage byte-level JEPA for the Parameter Golf challenge.

Stage 1 trains a latent JEPA backbone over raw byte patches with an EMA teacher.
Stage 2 freezes that backbone and trains a patch-local decoder that turns
predicted patch latents into exact byte probabilities.
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
    model_family = os.environ.get("MODEL_FAMILY", "two_stage_jepa").strip().lower()
    compressed_ut_family = model_family in {"compressed_ut", "utcompress", "ut_byte_ce"}
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_bytes")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 200))
    max_val_tokens = int(os.environ.get("MAX_VAL_TOKENS", 131_072))
    final_max_val_tokens = int(os.environ.get("FINAL_MAX_VAL_TOKENS", 131_072))
    final_full_val = env_flag("FINAL_FULL_VAL", True)
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 20))

    iterations = int(os.environ.get("ITERATIONS", 200_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 10))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 256))
    pad_id = int(os.environ.get("PAD_ID", 0))
    bos_id = int(os.environ.get("BOS_ID", 0))
    byte_offset = int(os.environ.get("BYTE_OFFSET", 0))
    byte_count = int(os.environ.get("BYTE_COUNT", 256))
    byte_dim = int(os.environ.get("BYTE_DIM", 384 if compressed_ut_family else 256))
    compress_stride = int(os.environ.get("COMPRESS_STRIDE", 4))
    ut_steps = int(os.environ.get("UT_STEPS", 4))
    decoder_type = os.environ.get("DECODER_TYPE", "slot_ar").strip().lower()
    conv_kernels = tuple(int(x) for x in os.environ.get("CONV_KERNELS", "2,3,5").split(",") if x.strip())
    conv_dilations = tuple(int(x) for x in os.environ.get("CONV_DILATIONS", "1,2,2").split(",") if x.strip())
    pred_dim = int(os.environ.get("PRED_DIM", 0))
    pred_steps = int(os.environ.get("PRED_STEPS", 2))
    pred_heads = int(os.environ.get("PRED_HEADS", 4))
    pred_kv_heads = int(os.environ.get("PRED_KV_HEADS", 2))
    pred_mlp_mult = float(os.environ.get("PRED_MLP_MULT", 2.0))
    jepa_weight = float(os.environ.get("JEPA_WEIGHT", 1.0))
    ce_aux_weight = float(os.environ.get("CE_AUX_WEIGHT", 0.1))
    joint_ce_weight = float(os.environ.get("JOINT_CE_WEIGHT", 1.0))

    model_dim = int(os.environ.get("MODEL_DIM", 1024 if compressed_ut_family else 512))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0 if compressed_ut_family else 3.0))
    partial_rope_dim = int(os.environ.get("PARTIAL_ROPE_DIM", 32 if compressed_ut_family else 16))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 4096))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 96 if compressed_ut_family else 64))

    train_stage = os.environ.get("TRAIN_STAGE", "jepa").strip().lower()
    load_model_path = os.environ.get("LOAD_MODEL_PATH", "").strip()

    patch_size = int(os.environ.get("PATCH_SIZE", 8))
    slot_size = int(os.environ.get("SLOT_SIZE", 2))
    jepa_latent_dim = int(os.environ.get("JEPA_LATENT_DIM", 128))
    jepa_slot_dim = int(os.environ.get("JEPA_SLOT_DIM", 32))
    jepa_horizon = int(os.environ.get("JEPA_HORIZON", 1))
    jepa_pred_steps = int(os.environ.get("JEPA_PRED_STEPS", os.environ.get("JEPA_PRED_LAYERS", 2)))
    jepa_pred_layers = jepa_pred_steps
    jepa_pred_heads = int(os.environ.get("JEPA_PRED_HEADS", 4))
    jepa_slot_pred_layers = int(os.environ.get("JEPA_SLOT_PRED_LAYERS", 2))
    jepa_memory_dim = int(os.environ.get("JEPA_MEMORY_DIM", 64))
    jepa_memory_decay = float(os.environ.get("JEPA_MEMORY_DECAY", 0.95))
    summary_pred_weight = float(os.environ.get("SUMMARY_PRED_WEIGHT", 1.0))
    slot_pred_weight = float(os.environ.get("SLOT_PRED_WEIGHT", 1.0))
    summary_stab_weight = float(os.environ.get("SUMMARY_STAB_WEIGHT", 0.05))
    sigreg_weight = float(os.environ.get("SIGREG_WEIGHT", 0.05))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    decoder_dim = int(os.environ.get("DECODER_DIM", 192 if compressed_ut_family else 256))
    decoder_layers = int(os.environ.get("DECODER_LAYERS", 1 if compressed_ut_family else 3))
    decoder_heads = int(os.environ.get("DECODER_HEADS", 4))
    decoder_mlp_mult = float(os.environ.get("DECODER_MLP_MULT", 4.0 if compressed_ut_family else 2.0))
    decoder_bridge_layers = int(os.environ.get("DECODER_BRIDGE_LAYERS", 1))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    decoder_backbone_lr_mult = float(os.environ.get("DECODER_BACKBONE_LR_MULT", 0.1))
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
    ttt_enabled = env_flag("TTT_ENABLED", False)
    ttt_lr = float(os.environ.get("TTT_LR", 5e-4))
    ttt_weight_decay = float(os.environ.get("TTT_WEIGHT_DECAY", 0.0))
    ttt_beta1 = float(os.environ.get("TTT_BETA1", 0.9))
    ttt_beta2 = float(os.environ.get("TTT_BETA2", 0.999))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 1))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_scope = os.environ.get("TTT_SCOPE", "decoder").strip().lower()
    ttt_grad_clip_norm = float(os.environ.get("TTT_GRAD_CLIP_NORM", 1.0))

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
    def __init__(self, hash_vocab: int, bigram_dim: int, model_dim: int, bos_token: int):
        super().__init__()
        self.hash_vocab = hash_vocab
        self.bos_token = int(bos_token)
        self.embed = nn.Embedding(hash_vocab, bigram_dim)
        self.proj = nn.Linear(bigram_dim, model_dim, bias=False)

    def forward(self, ids: Tensor) -> Tensor:
        prev = torch.cat((ids.new_full((ids.size(0), 1), self.bos_token), ids[:, :-1]), dim=1)
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


JEPAState = Tensor


class JEPALatentMLP(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or (2 * dim)
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.gelu(self.fc(x)))


class JEPACausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Latent dim {dim} must divide num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q = F.rms_norm(qkv[:, :, 0], (self.head_dim,)).transpose(1, 2)
        k = F.rms_norm(qkv[:, :, 1], (self.head_dim,)).transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().reshape(B, T, D))


class LatentTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = JEPACausalSelfAttention(dim, num_heads)
        self.mlp = JEPALatentMLP(dim)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class UniversalLatentPredictor(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_steps: int):
        super().__init__()
        self.num_steps = max(int(num_steps), 1)
        self.step_embed = nn.Parameter(torch.zeros(self.num_steps, dim))
        self.block = LatentTransformerBlock(dim, num_heads)
        self.final_norm = RMSNorm(dim)
        nn.init.normal_(self.step_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        for step in range(self.num_steps):
            x = self.block(x + self.step_embed[step].to(x.dtype)[None, None, :])
        return self.final_norm(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Decoder dim {dim} must divide num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        B, T, D = x.shape
        C = cond.size(1)
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        kv = self.kv_proj(cond).reshape(B, C, 2, self.num_heads, self.head_dim)
        k = F.rms_norm(kv[:, :, 0], (self.head_dim,)).transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        q = F.rms_norm(q, (self.head_dim,)).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out_proj(y.transpose(1, 2).contiguous().reshape(B, T, D))


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: float):
        super().__init__()
        self.self_norm = RMSNorm(dim)
        self.cross_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.self_attn = JEPACausalSelfAttention(dim, num_heads)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.mlp = JEPALatentMLP(dim, hidden=int(dim * mlp_mult))
        self.self_scale = nn.Parameter(torch.ones(dim))
        self.cross_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = x + self.self_scale.to(x.dtype)[None, None, :] * self.self_attn(self.self_norm(x))
        x = x + self.cross_scale.to(x.dtype)[None, None, :] * self.cross_attn(self.cross_norm(x), cond)
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class SlotLatentPredictor(nn.Module):
    def __init__(self, context_dim: int, slot_dim: int, num_slots: int, num_layers: int, num_heads: int):
        super().__init__()
        if slot_dim % num_heads != 0:
            raise ValueError(f"Slot dim {slot_dim} must divide num_heads={num_heads}")
        self.num_slots = num_slots
        self.slot_queries = nn.Parameter(torch.zeros(num_slots, slot_dim))
        self.context_seed = nn.Linear(context_dim, slot_dim, bias=False)
        self.summary_seed = nn.Linear(context_dim, slot_dim, bias=False)
        self.context_cond = nn.Linear(context_dim, slot_dim, bias=False)
        self.summary_cond = nn.Linear(context_dim, slot_dim, bias=False)
        self.blocks = nn.ModuleList([
            DecoderBlock(slot_dim, num_heads, mlp_mult=2.0)
            for _ in range(max(num_layers, 1))
        ])
        self.final_norm = RMSNorm(slot_dim)
        self.out_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        nn.init.normal_(self.slot_queries, std=0.02)
        for module in (
            self.context_seed,
            self.summary_seed,
            self.context_cond,
            self.summary_cond,
            self.out_proj,
        ):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, context_token: Tensor, summary_pred: Tensor) -> Tensor:
        B, P, _ = summary_pred.shape
        x = self.slot_queries.to(summary_pred.dtype)[None, None, :, :].expand(B, P, -1, -1)
        x = x + self.context_seed(context_token).unsqueeze(2) + self.summary_seed(summary_pred).unsqueeze(2)
        cond = torch.cat(
            (
                self.context_cond(context_token).unsqueeze(2),
                self.summary_cond(summary_pred).unsqueeze(2),
            ),
            dim=2,
        )
        x = x.reshape(B * P, self.num_slots, -1)
        cond = cond.reshape(B * P, 2, -1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.out_proj(self.final_norm(x))
        return x.view(B, P, self.num_slots, -1)


class LatentPatchBridge(nn.Module):
    def __init__(self, patch_size: int, dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.patch_size = patch_size
        self.byte_queries = nn.Parameter(torch.zeros(patch_size, dim))
        self.context_seed = nn.Linear(dim, dim, bias=False)
        self.summary_seed = nn.Linear(dim, dim, bias=False)
        self.blocks = nn.ModuleList([
            DecoderBlock(dim, num_heads, mlp_mult=2.0)
            for _ in range(max(num_layers, 1))
        ])
        self.final_norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.byte_queries, std=0.02)
        for module in (self.context_seed, self.summary_seed, self.out_proj):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, context_token: Tensor, summary_token: Tensor, cond: Tensor) -> Tensor:
        B, P, _ = context_token.shape
        C = cond.size(2)
        x = self.byte_queries.to(summary_token.dtype)[None, None, :, :].expand(B, P, -1, -1)
        x = x + self.context_seed(context_token).unsqueeze(2) + self.summary_seed(summary_token).unsqueeze(2)
        x = x.reshape(B * P, self.patch_size, -1)
        cond = cond.reshape(B * P, C, -1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.out_proj(self.final_norm(x))
        return x.view(B, P, self.patch_size, -1)


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = nn.Linear(dim, 2 * hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, val = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.silu(gate) * val)


class RotaryUTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_dim: int,
        rope_base: float,
        train_len: int,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"UT dim {dim} must divide num_heads={num_heads}")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")
        head_dim = dim // num_heads
        rope_dim = min(max(rope_dim, 2), head_dim)
        if rope_dim % 2 != 0:
            rope_dim -= 1
        if rope_dim <= 0:
            raise ValueError(f"Invalid rope dim for head_dim={head_dim}")
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(
            dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary=Rotary(rope_dim, base=rope_base, train_len=train_len),
        )
        self.mlp = SwiGLUMLP(dim, hidden=int(dim * mlp_mult))
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class CausalDepthwiseConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int):
        super().__init__()
        if kernel_size <= 1:
            raise ValueError(f"kernel_size must be >1, got {kernel_size}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.norm = RMSNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x).transpose(1, 2)
        left = self.dilation * (self.kernel_size - 1)
        y = F.pad(y, (left, 0))
        y = self.pointwise(self.depthwise(y)).transpose(1, 2)
        return x + self.scale.to(x.dtype)[None, None, :] * y


class ByteNgramFrontEnd(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        byte_dim: int,
        bigram_vocab_size: int,
        bigram_dim: int,
        conv_kernels: tuple[int, ...],
        conv_dilations: tuple[int, ...],
    ):
        super().__init__()
        if len(conv_kernels) != len(conv_dilations):
            raise ValueError("CONV_KERNELS and CONV_DILATIONS must have the same length")
        self.byte_emb = nn.Embedding(vocab_size, byte_dim)
        self.bigram_hash = BigramHash(bigram_vocab_size, bigram_dim, byte_dim, bos_token=vocab_size)
        self.input_norm = RMSNorm(byte_dim)
        self.conv_blocks = nn.ModuleList(
            [CausalDepthwiseConvBlock(byte_dim, k, d) for k, d in zip(conv_kernels, conv_dilations)]
        )

    def forward(self, ids: Tensor) -> Tensor:
        x = self.byte_emb(ids) + self.bigram_hash(ids)
        x = self.input_norm(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class ChunkCompressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int):
        super().__init__()
        if stride <= 0:
            raise ValueError(f"COMPRESS_STRIDE must be positive, got {stride}")
        self.stride = stride
        self.proj = nn.Conv1d(in_dim, out_dim, kernel_size=stride, stride=stride, bias=False)
        self.norm = RMSNorm(out_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int]:
        B, T, _ = x.shape
        pad = (-T) % self.stride
        y = x.transpose(1, 2)
        if pad > 0:
            y = F.pad(y, (0, pad))
        y = self.proj(y).transpose(1, 2)
        return self.norm(y), pad


class ParallelSlotByteDecoder(nn.Module):
    def __init__(self, dim: int, vocab_size: int, stride: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.stride = stride
        self.norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, stride * vocab_size, bias=False)

    def forward(self, x: Tensor, target_len: int) -> Tensor:
        B, P, _ = x.shape
        logits = self.out_proj(self.norm(x)).view(B, P * self.stride, self.vocab_size)
        return logits[:, :target_len]


class TinyCausalDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: float, use_cross_attention: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.cross_norm = RMSNorm(dim) if use_cross_attention else None
        self.mlp_norm = RMSNorm(dim)
        self.attn = JEPACausalSelfAttention(dim, num_heads)
        self.cross_attn = CrossAttention(dim, num_heads) if use_cross_attention else None
        self.mlp = SwiGLUMLP(dim, hidden=int(dim * mlp_mult))
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.cross_scale = nn.Parameter(torch.ones(dim)) if use_cross_attention else None
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        if self.cross_attn is not None:
            if cond is None:
                raise RuntimeError("decoder block requires cond when cross attention is enabled")
            x = x + self.cross_scale.to(x.dtype)[None, None, :] * self.cross_attn(self.cross_norm(x), cond)
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class SlotByteBridge(nn.Module):
    def __init__(self, patch_size: int, dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.patch_size = patch_size
        self.byte_queries = nn.Parameter(torch.zeros(patch_size, dim))
        self.context_seed = nn.Linear(dim, dim, bias=False)
        self.blocks = nn.ModuleList([
            TinyCausalDecoderBlock(dim, num_heads, mlp_mult=2.0, use_cross_attention=True)
            for _ in range(max(num_layers, 1))
        ])
        self.final_norm = RMSNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.byte_queries, std=0.02)
        for module in (self.context_seed, self.out_proj):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, context_token: Tensor, cond: Tensor) -> Tensor:
        B, P, _ = context_token.shape
        x = self.byte_queries.to(context_token.dtype)[None, None, :, :].expand(B, P, -1, -1)
        x = x + self.context_seed(context_token).unsqueeze(2)
        x = x.view(B * P, self.patch_size, -1)
        cond = cond.view(B * P, cond.size(2), -1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.out_proj(self.final_norm(x))
        return x.view(B, P, self.patch_size, -1)


class SlotAutoregressiveByteDecoder(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        vocab_size: int,
        stride: int,
        decoder_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_mult: float,
        bridge_layers: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.stride = stride
        self.context_proj = nn.Linear(slot_dim, decoder_dim, bias=False)
        self.byte_emb = nn.Embedding(vocab_size, decoder_dim)
        self.start_token = nn.Parameter(torch.zeros(decoder_dim))
        self.pos_emb = nn.Parameter(torch.zeros(stride, decoder_dim))
        self.latent_bridge = (
            SlotByteBridge(stride, decoder_dim, num_heads, bridge_layers)
            if bridge_layers > 0
            else None
        )
        if self.latent_bridge is not None:
            self.bridge_scale = nn.Parameter(torch.ones(decoder_dim))
        else:
            self.register_parameter("bridge_scale", None)
        self.blocks = nn.ModuleList(
            [
                TinyCausalDecoderBlock(decoder_dim, num_heads, mlp_mult, use_cross_attention=True)
                for _ in range(max(num_layers, 1))
            ]
        )
        self.norm = RMSNorm(decoder_dim)
        self.out_proj = nn.Linear(decoder_dim, vocab_size, bias=False)
        nn.init.normal_(self.start_token, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.xavier_uniform_(self.context_proj.weight)

    def forward(self, slot_context: Tensor, ids: Tensor) -> Tensor:
        B, T = ids.shape
        P = slot_context.size(1)
        S = self.stride
        pad = P * S - T
        padded = F.pad(ids, (0, pad), value=0) if pad > 0 else ids
        targets = padded.view(B, P, S)
        if S > 1:
            prev_bytes = self.byte_emb(targets[:, :, :-1])
            byte_inputs = torch.cat(
                (self.start_token.to(prev_bytes.dtype)[None, None, None, :].expand(B, P, 1, -1), prev_bytes),
                dim=2,
            )
        else:
            byte_inputs = self.start_token.to(slot_context.dtype)[None, None, None, :].expand(B, P, 1, -1)
        x = byte_inputs
        x = x + self.pos_emb.to(x.dtype)[None, None, :, :]
        cond = self.context_proj(slot_context).unsqueeze(2)
        if self.latent_bridge is not None:
            bridge = self.latent_bridge(cond[:, :, 0], cond)
            x = x + self.bridge_scale.to(x.dtype)[None, None, None, :] * bridge
        x = x.view(B * P, S, -1)
        cond = cond.view(B * P, 1, -1)
        for block in self.blocks:
            x = block(x, cond)
        logits = self.out_proj(self.norm(x)).view(B, P * S, self.vocab_size)
        return logits[:, :T]


class CompressedUTPredictor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        pred_dim: int,
        out_dim: int,
        num_steps: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_dim: int,
        rope_base: float,
        train_len: int,
    ):
        super().__init__()
        self.num_steps = max(int(num_steps), 1)
        self.in_proj = nn.Linear(in_dim, pred_dim, bias=False)
        self.step_embed = nn.Parameter(torch.zeros(self.num_steps, pred_dim))
        self.block = RotaryUTBlock(
            dim=pred_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult,
            rope_dim=rope_dim,
            rope_base=rope_base,
            train_len=train_len,
        )
        self.norm = RMSNorm(pred_dim)
        self.out_proj = nn.Linear(pred_dim, out_dim, bias=False)
        nn.init.normal_(self.step_embed, std=0.02)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for step in range(self.num_steps):
            x = self.block(x + self.step_embed[step].to(x.dtype)[None, None, :])
        return self.out_proj(self.norm(x))


class CompressedUTByteModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        if args.train_stage not in {"ce", "jepa", "joint", "decoder"}:
            raise ValueError(f"Compressed UT supports TRAIN_STAGE=ce/jepa/joint/decoder, got {args.train_stage!r}")
        if args.byte_offset != 0 or args.byte_count != args.vocab_size:
            raise ValueError("Compressed UT currently expects raw-byte tokens with BYTE_OFFSET=0 and BYTE_COUNT=VOCAB_SIZE")
        if args.decoder_type not in {"parallel", "slot_ar"}:
            raise ValueError(f"Unsupported DECODER_TYPE={args.decoder_type!r}; expected 'parallel' or 'slot_ar'")

        self.args = args
        self.vocab_size = args.vocab_size
        self.pad_id = args.pad_id
        self.train_stage = args.train_stage
        self.compress_stride = args.compress_stride
        self.ut_steps = max(args.ut_steps, 1)
        self.decoder_type = args.decoder_type
        self.has_jepa_stage = args.train_stage in {"jepa", "joint"}
        self.pred_dim = args.pred_dim if args.pred_dim > 0 else max(args.model_dim // 2, 1)
        self.jepa_weight = args.jepa_weight
        self.ce_aux_weight = args.ce_aux_weight
        self.joint_ce_weight = args.joint_ce_weight
        self.sigreg_weight = args.sigreg_weight

        self.frontend = ByteNgramFrontEnd(
            vocab_size=args.vocab_size,
            byte_dim=args.byte_dim,
            bigram_vocab_size=args.bigram_vocab_size,
            bigram_dim=args.bigram_dim,
            conv_kernels=args.conv_kernels,
            conv_dilations=args.conv_dilations,
        )
        self.compressor = ChunkCompressor(args.byte_dim, args.model_dim, args.compress_stride)
        max_slots = (args.train_seq_len + args.compress_stride - 1) // args.compress_stride + 2
        rope_dim = args.partial_rope_dim if args.partial_rope_dim > 0 else (args.model_dim // args.num_heads)
        self.encoder_step_embed = nn.Parameter(torch.zeros(self.ut_steps, args.model_dim))
        self.encoder_bos = nn.Parameter(torch.zeros(args.model_dim))
        self.encoder_block = RotaryUTBlock(
            dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            rope_dim=rope_dim,
            rope_base=args.rope_base,
            train_len=max_slots,
        )
        self.encoder_norm = RMSNorm(args.model_dim)
        self.predictor: CompressedUTPredictor | None = None
        if self.has_jepa_stage:
            pred_rope_dim = args.partial_rope_dim if args.partial_rope_dim > 0 else (self.pred_dim // max(args.pred_heads, 1))
            self.predictor = CompressedUTPredictor(
                in_dim=args.model_dim,
                pred_dim=self.pred_dim,
                out_dim=args.model_dim,
                num_steps=args.pred_steps,
                num_heads=args.pred_heads,
                num_kv_heads=args.pred_kv_heads,
                mlp_mult=args.pred_mlp_mult,
                rope_dim=pred_rope_dim,
                rope_base=args.rope_base,
                train_len=max_slots,
            )
        if args.decoder_type == "parallel":
            self.decoder = ParallelSlotByteDecoder(args.model_dim, args.vocab_size, args.compress_stride)
        else:
            self.decoder = SlotAutoregressiveByteDecoder(
                slot_dim=args.model_dim,
                vocab_size=args.vocab_size,
                stride=args.compress_stride,
                decoder_dim=args.decoder_dim,
                num_layers=args.decoder_layers,
                num_heads=args.decoder_heads,
                mlp_mult=args.decoder_mlp_mult,
                bridge_layers=args.decoder_bridge_layers,
            )
        nn.init.normal_(self.encoder_step_embed, std=0.02)
        nn.init.normal_(self.encoder_bos, std=0.02)
        self.configure_training_stage(args.train_stage)

    def configure_training_stage(self, stage: str):
        if stage not in {"ce", "jepa", "joint", "decoder"}:
            raise ValueError(f"Unsupported stage {stage!r}")
        self.train_stage = stage
        for p in self.parameters():
            p.requires_grad_(False)
        if stage in {"ce", "jepa", "joint"}:
            for p in self.parameters():
                p.requires_grad_(True)
            return
        for module in (self.frontend, self.compressor, self.encoder_block, self.encoder_norm):
            for p in module.parameters():
                p.requires_grad_(True)
        for param in (self.encoder_step_embed, self.encoder_bos):
            param.requires_grad_(True)
        for p in self.decoder.parameters():
            p.requires_grad_(True)

    def parameter_lr_scale(self, name: str) -> float:
        if self.train_stage == "decoder" and not name.startswith("decoder."):
            return self.args.decoder_backbone_lr_mult
        return 1.0

    def _shift_slots(self, x: Tensor) -> Tensor:
        bos = self.encoder_bos.to(x.dtype)[None, None, :].expand(x.size(0), 1, -1)
        if x.size(1) <= 1:
            return bos[:, : x.size(1)]
        return torch.cat((bos, x[:, :-1]), dim=1)

    def _encode_slots(self, ids: Tensor) -> tuple[Tensor, int]:
        x = self.frontend(ids)
        slots, pad = self.compressor(x)
        for step in range(self.ut_steps):
            slots = self.encoder_block(slots + self.encoder_step_embed[step].to(slots.dtype)[None, None, :])
        return self.encoder_norm(slots), pad

    def _slot_valid_counts(self, total_tokens: int, num_slots: int) -> tuple[int, int]:
        full_slots = min(total_tokens // self.compress_stride, num_slots)
        valid_targets = min(max(full_slots - 1, 0), max(num_slots - 1, 0))
        return full_slots, valid_targets

    def _slot_target_mask(self, batch_size: int, total_tokens: int, num_slots: int, device: torch.device) -> Tensor:
        _full_slots, valid_targets = self._slot_valid_counts(total_tokens, num_slots)
        mask = torch.zeros(batch_size, max(num_slots - 1, 0), dtype=torch.bool, device=device)
        if valid_targets > 0:
            mask[:, :valid_targets] = True
        return mask

    def _masked_mse(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        extra_dims = pred.ndim - mask.ndim
        extra_scale = 1
        if extra_dims > 0:
            for size in pred.shape[-extra_dims:]:
                extra_scale *= int(size)
        if mask.ndim < pred.ndim:
            mask = mask.view(*mask.shape, *([1] * (pred.ndim - mask.ndim)))
        diff = (pred.float() - target.float()).pow(2)
        denom = (mask.float().sum() * float(extra_scale)).clamp_min(1.0)
        return (diff * mask.float()).sum() / denom

    def _sigreg(self, x: Tensor) -> Tensor:
        flat = x.float().reshape(-1, x.size(-1))
        if flat.size(0) < 2:
            return flat.new_zeros(())
        mean = flat.mean(dim=0)
        centered = flat - mean
        var = centered.pow(2).mean(dim=0).clamp_min(1e-5)
        normed = centered / var.sqrt()
        cov = (normed.T @ normed) / float(normed.size(0))
        offdiag = cov - torch.diag(torch.diag(cov))
        return mean.square().mean() + F.relu(1.0 - var.sqrt()).square().mean() + 0.01 * offdiag.square().mean()

    def _decode_from_context(self, slot_context: Tensor, ids: Tensor) -> Tensor:
        if self.decoder_type == "parallel":
            return self.decoder(slot_context, ids.size(1))
        return self.decoder(slot_context, ids)

    def _ce_from_logits(self, logits: Tensor, ids: Tensor) -> tuple[Tensor, Tensor]:
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size).float(), ids.reshape(-1), reduction="none").view_as(ids)
        mask = torch.ones_like(ids, dtype=torch.float32)
        if mask.size(1) > 0:
            mask[:, 0] = 0.0
        total = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return total, total.detach()

    def _ce_logits(self, ids: Tensor, *, backbone_no_grad: bool = False) -> Tensor:
        ctx = torch.no_grad() if backbone_no_grad else torch.enable_grad()
        with ctx:
            slots, _pad = self._encode_slots(ids)
            slot_context = self._shift_slots(slots)
        logits = self._decode_from_context(slot_context, ids)
        return self.args.logit_softcap * torch.tanh(logits / self.args.logit_softcap)

    def forward_logits(
        self,
        ids: Tensor,
        jepa_state: JEPAState | None = None,
        *,
        return_jepa_state: bool = False,
    ) -> Tensor | tuple[Tensor, JEPAState | None]:
        del jepa_state
        logits = self._ce_logits(ids)
        if return_jepa_state:
            return logits, None
        return logits

    def forward_ce(self, ids: Tensor, *, backbone_no_grad: bool = False) -> tuple[Tensor, Tensor]:
        logits = self._ce_logits(ids, backbone_no_grad=backbone_no_grad)
        return self._ce_from_logits(logits, ids)

    def forward_ttt(self, ids: Tensor, *, scope: str) -> Tensor:
        if scope != "decoder":
            raise ValueError(f"Compressed UT TTT supports only TTT_SCOPE='decoder', got {scope!r}")
        return self.forward_ce(ids, backbone_no_grad=True)[0]

    def _forward_slot_objective(self, ids: Tensor, *, ce_weight: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        slots, _pad = self._encode_slots(ids)
        slot_context = self._shift_slots(slots)
        logits = self._decode_from_context(slot_context, ids)
        logits = self.args.logit_softcap * torch.tanh(logits / self.args.logit_softcap)
        ce_loss, ce_det = self._ce_from_logits(logits, ids)

        predictor_in = slots[:, :-1]
        target = slots[:, 1:].detach()
        if self.predictor is None:
            raise RuntimeError(f"TRAIN_STAGE={self.train_stage!r} requires a predictor for slot objectives")
        pred = self.predictor(predictor_in) if predictor_in.size(1) > 0 else target
        target_mask = self._slot_target_mask(ids.size(0), ids.size(1), slots.size(1), ids.device)
        jepa_loss = (
            self._masked_mse(pred, target, target_mask)
            if pred.size(1) > 0 and target_mask.any()
            else slots.new_zeros(())
        )
        full_slots, _valid_targets = self._slot_valid_counts(ids.size(1), slots.size(1))
        sigreg = self._sigreg(slots[:, :full_slots]) if full_slots > 0 else slots.new_zeros(())
        total = self.jepa_weight * jepa_loss + self.sigreg_weight * sigreg + ce_weight * ce_loss
        zero = total.detach().new_zeros(())
        return total, jepa_loss.detach(), ce_det, sigreg.detach(), zero

    def forward_jepa(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self._forward_slot_objective(ids, ce_weight=self.ce_aux_weight)

    def forward_joint(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self._forward_slot_objective(ids, ce_weight=self.joint_ce_weight)

    def forward(self, ids: Tensor):
        if self.train_stage == "jepa":
            return self.forward_jepa(ids)
        if self.train_stage == "joint":
            return self.forward_joint(ids)
        loss, ce = self.forward_ce(ids)
        zero = loss.detach().new_zeros(())
        return loss, ce, zero, zero, zero

    def serializable_state_dict(self) -> dict[str, Tensor]:
        return dict(self.state_dict())

    def load_serializable(self, sd: dict[str, Tensor]):
        self.load_state_dict(sd, strict=False)

    @torch.no_grad()
    def zero_pad_emb_(self):
        valid_lo = self.args.byte_offset
        valid_hi = min(self.vocab_size, self.args.byte_offset + self.args.byte_count)
        if 0 <= self.pad_id < self.frontend.byte_emb.num_embeddings and not (valid_lo <= self.pad_id < valid_hi):
            self.frontend.byte_emb.weight[self.pad_id].zero_()
        if hasattr(self.decoder, "byte_emb") and 0 <= self.pad_id < self.decoder.byte_emb.num_embeddings and not (
            valid_lo <= self.pad_id < valid_hi
        ):
            self.decoder.byte_emb.weight[self.pad_id].zero_()

    @torch.no_grad()
    def update_teacher(self):
        return

    @torch.no_grad()
    def _sync_teacher(self):
        return


class JEPALatentMemory(nn.Module):
    """Tiny Titans-style memory over patch context states."""

    def __init__(self, dim: int, memory_dim: int, decay_init: float):
        super().__init__()
        self.memory_dim = max(memory_dim, 1)
        self.q_proj = nn.Linear(dim, self.memory_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.memory_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.memory_dim, bias=False)
        self.out_proj = nn.Linear(self.memory_dim, dim, bias=False)
        self.out_proj.weight.data.zero_()
        decay_init = min(max(decay_init, 1e-4), 1.0 - 1e-4)
        self.decay_logit = nn.Parameter(torch.tensor(math.log(decay_init / (1.0 - decay_init)), dtype=torch.float32))
        self.write_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def initial_state(self, batch_size: int, device: torch.device) -> JEPAState:
        return torch.zeros(batch_size, self.memory_dim, self.memory_dim, device=device, dtype=torch.float32)

    def forward(self, x: Tensor, state: JEPAState | None = None) -> tuple[Tensor, JEPAState]:
        B, T, _ = x.shape
        mem = self.initial_state(B, x.device) if state is None else state.float()
        decay = torch.sigmoid(self.decay_logit.float())
        write = torch.sigmoid(self.write_logit.float())
        outs: list[Tensor] = []
        for t in range(T):
            q = self.q_proj(x[:, t]).float()
            read = torch.einsum("bm,bmn->bn", q, mem)
            read = F.rms_norm(read.to(x.dtype), (self.memory_dim,))
            outs.append(self.out_proj(read))
            k = self.k_proj(x[:, t]).float()
            v = self.v_proj(x[:, t]).float()
            mem = decay * mem + write * torch.einsum("bm,bn->bmn", k, v)
        return torch.stack(outs, dim=1), mem


class BytePatchEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        patch_size: int,
        slot_size: int,
        model_dim: int,
        summary_dim: int,
        slot_dim: int,
        bigram_vocab_size: int,
        bigram_dim: int,
    ):
        super().__init__()
        if patch_size <= 0 or patch_size % slot_size != 0:
            raise ValueError(f"PATCH_SIZE={patch_size} must be a positive multiple of SLOT_SIZE={slot_size}")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.patch_size = patch_size
        self.slot_size = slot_size
        self.num_slots = patch_size // slot_size
        self.model_dim = model_dim
        self.byte_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram_hash = BigramHash(bigram_vocab_size, bigram_dim, model_dim, bos_token=vocab_size)
        self.patch_pos = nn.Parameter(torch.zeros(patch_size, model_dim))
        self.slot_pos = nn.Parameter(torch.zeros(self.num_slots, slot_dim))
        self.summary_proj = nn.Linear(patch_size * model_dim, summary_dim, bias=False)
        self.slot_proj = nn.Linear(slot_size * model_dim, slot_dim, bias=False)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        B, T = ids.shape
        K = self.patch_size
        P = (T + K - 1) // K
        pad = P * K - T
        padded = F.pad(ids, (0, pad), value=self.pad_id) if pad > 0 else ids
        byte_mask = torch.ones(B, T, device=ids.device, dtype=torch.bool)
        if pad > 0:
            byte_mask = F.pad(byte_mask, (0, pad), value=False)
        patch_mask = byte_mask.view(B, P, K).all(dim=-1)

        flat = padded.reshape(B, P * K)
        x = self.byte_emb(flat) + self.bigram_hash(flat)
        x = x.view(B, P, K, self.model_dim)
        x = F.rms_norm(x, (self.model_dim,))
        x = x + self.patch_pos.to(x.dtype)[None, None, :, :]

        summary = self.summary_proj(x.reshape(B, P, K * self.model_dim))
        summary = F.rms_norm(summary, (summary.size(-1),))

        slots = x.view(B, P, self.num_slots, self.slot_size, self.model_dim)
        slots = self.slot_proj(slots.reshape(B, P, self.num_slots, self.slot_size * self.model_dim))
        slots = slots + self.slot_pos.to(slots.dtype)[None, None, :, :]
        slots = F.rms_norm(slots, (slots.size(-1),))
        return summary, slots, byte_mask.view(B, P, K), patch_mask


class PatchLocalDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        patch_size: int,
        decoder_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_mult: float,
        bridge_layers: int,
        summary_dim: int,
        slot_dim: int,
        num_slots: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.num_slots = num_slots
        self.byte_emb = nn.Embedding(vocab_size, decoder_dim)
        self.bos_emb = nn.Parameter(torch.zeros(decoder_dim))
        self.byte_pos = nn.Parameter(torch.zeros(patch_size, decoder_dim))
        self.context_proj = nn.Linear(summary_dim, decoder_dim, bias=False)
        self.summary_proj = nn.Linear(summary_dim, decoder_dim, bias=False)
        self.slot_proj = nn.Linear(slot_dim, decoder_dim, bias=False)
        self.latent_bridge = (
            LatentPatchBridge(patch_size, decoder_dim, num_heads, bridge_layers)
            if bridge_layers > 0
            else None
        )
        if self.latent_bridge is not None:
            self.bridge_scale = nn.Parameter(torch.ones(decoder_dim))
        else:
            self.register_parameter("bridge_scale", None)
        self.blocks = nn.ModuleList([
            DecoderBlock(decoder_dim, num_heads, mlp_mult)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(decoder_dim)
        self.lm_head = nn.Linear(decoder_dim, vocab_size, bias=False)
        nn.init.normal_(self.bos_emb, std=0.02)

    def forward(
        self,
        ids: Tensor,
        byte_mask: Tensor,
        context_prev: Tensor,
        summary_pred: Tensor,
        slot_pred: Tensor,
    ) -> tuple[Tensor, Tensor]:
        B, T = ids.shape
        P = context_prev.size(1)
        K = self.patch_size
        pad = P * K - T
        padded = F.pad(ids, (0, pad), value=0) if pad > 0 else ids
        patches = padded.view(B, P, K)

        bos = self.bos_emb.to(self.byte_emb.weight.dtype)[None, None, None, :].expand(B, P, 1, -1)
        prev = self.byte_emb(patches[:, :, :-1])
        x = torch.cat((bos, prev), dim=2) + self.byte_pos.to(self.byte_emb.weight.dtype)[None, None, :, :]
        context_tokens = self.context_proj(context_prev)
        summary_tokens = self.summary_proj(summary_pred)
        slot_tokens = self.slot_proj(slot_pred)

        cond = torch.cat(
            (
                context_tokens.unsqueeze(2),
                summary_tokens.unsqueeze(2),
                slot_tokens,
            ),
            dim=2,
        )
        if self.latent_bridge is not None:
            bridge = self.latent_bridge(context_tokens, summary_tokens, cond)
            x = x + self.bridge_scale.to(x.dtype)[None, None, None, :] * bridge

        x = x.view(B * P, K, -1)
        cond = cond.view(B * P, cond.size(2), -1)
        for block in self.blocks:
            x = block(x, cond)
        logits = self.lm_head(self.final_norm(x)).view(B, P * K, self.vocab_size)
        flat_mask = byte_mask.reshape(B, P * K)
        return logits[:, :T], flat_mask[:, :T]


class CausalJEPA(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        if args.patch_size % args.slot_size != 0:
            raise ValueError(f"PATCH_SIZE={args.patch_size} must be divisible by SLOT_SIZE={args.slot_size}")
        if args.train_stage not in {"jepa", "decoder"}:
            raise ValueError(f"Unsupported TRAIN_STAGE={args.train_stage!r}")
        if args.byte_offset < 0 or args.byte_count <= 0 or args.byte_offset + args.byte_count > args.vocab_size:
            raise ValueError(
                f"Invalid byte token range offset={args.byte_offset} count={args.byte_count} vocab={args.vocab_size}"
            )

        self.args = args
        self.vocab_size = args.vocab_size
        self.pad_id = args.pad_id
        self.bos_id = args.bos_id
        self.train_stage = args.train_stage
        self.has_jepa_stage = True
        self.patch_size = args.patch_size
        self.slot_size = args.slot_size
        self.num_slots = args.patch_size // args.slot_size
        self.summary_dim = args.jepa_latent_dim
        self.slot_dim = args.jepa_slot_dim
        self.jepa_horizon = max(1, args.jepa_horizon)
        self.summary_pred_weight = args.summary_pred_weight
        self.slot_pred_weight = args.slot_pred_weight
        self.summary_stab_weight = args.summary_stab_weight
        self.sigreg_weight = args.sigreg_weight
        self.ema_decay = args.ema_decay

        self.patch_encoder = BytePatchEncoder(
            vocab_size=args.vocab_size,
            pad_id=args.pad_id,
            patch_size=args.patch_size,
            slot_size=args.slot_size,
            model_dim=args.model_dim,
            summary_dim=args.jepa_latent_dim,
            slot_dim=args.jepa_slot_dim,
            bigram_vocab_size=args.bigram_vocab_size,
            bigram_dim=args.bigram_dim,
        )
        max_patches = (args.train_seq_len + args.patch_size - 1) // args.patch_size + 2
        self.patch_context_pos = nn.Parameter(torch.zeros(max_patches, args.jepa_latent_dim))
        self.context_encoder = nn.ModuleList([
            LatentTransformerBlock(args.jepa_latent_dim, args.jepa_pred_heads)
            for _ in range(args.num_layers)
        ])
        self.context_norm = RMSNorm(args.jepa_latent_dim)

        self.context_bos = nn.Parameter(torch.zeros(args.jepa_latent_dim))
        self.predict_bos = nn.Parameter(torch.zeros(args.jepa_latent_dim))
        self.predictor_memory = (
            JEPALatentMemory(args.jepa_latent_dim, args.jepa_memory_dim, args.jepa_memory_decay)
            if args.jepa_memory_dim > 0
            else None
        )
        self.predictor = UniversalLatentPredictor(
            dim=args.jepa_latent_dim,
            num_heads=args.jepa_pred_heads,
            num_steps=args.jepa_pred_steps,
        )
        self.predictor_summary = nn.Linear(args.jepa_latent_dim, args.jepa_latent_dim, bias=False)
        self.slot_predictor = SlotLatentPredictor(
            context_dim=args.jepa_latent_dim,
            slot_dim=args.jepa_slot_dim,
            num_slots=self.num_slots,
            num_layers=args.jepa_slot_pred_layers,
            num_heads=args.jepa_pred_heads,
        )

        self.decoder = PatchLocalDecoder(
            vocab_size=args.vocab_size,
            patch_size=args.patch_size,
            decoder_dim=args.decoder_dim,
            num_layers=args.decoder_layers,
            num_heads=args.decoder_heads,
            mlp_mult=args.decoder_mlp_mult,
            bridge_layers=args.decoder_bridge_layers,
            summary_dim=args.jepa_latent_dim,
            slot_dim=args.jepa_slot_dim,
            num_slots=self.num_slots,
        )

        self.teacher_patch_encoder = copy.deepcopy(self.patch_encoder)
        self.teacher_context_encoder = copy.deepcopy(self.context_encoder)
        self.teacher_context_norm = copy.deepcopy(self.context_norm)
        for p in self.teacher_patch_encoder.parameters():
            p.requires_grad_(False)
        for p in self.teacher_context_encoder.parameters():
            p.requires_grad_(False)
        for p in self.teacher_context_norm.parameters():
            p.requires_grad_(False)

        self._init_weights()
        self._sync_teacher()
        self.configure_training_stage(args.train_stage)

    def _init_weights(self):
        nn.init.normal_(self.patch_context_pos, std=0.02)
        for module in (self.predictor_summary,):
            nn.init.xavier_uniform_(module.weight)

    def configure_training_stage(self, stage: str):
        self.train_stage = stage
        for p in self.parameters():
            p.requires_grad_(False)

        if stage == "jepa":
            for module in (
                self.patch_encoder,
                self.context_encoder,
                self.context_norm,
                self.predictor_memory,
                self.predictor,
                self.predictor_summary,
                self.slot_predictor,
            ):
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad_(True)
            for param in (self.patch_context_pos, self.context_bos, self.predict_bos):
                param.requires_grad_(True)
        elif stage == "decoder":
            for p in self.decoder.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"Unsupported stage {stage!r}")

        for p in self.teacher_patch_encoder.parameters():
            p.requires_grad_(False)
        for p in self.teacher_context_encoder.parameters():
            p.requires_grad_(False)
        for p in self.teacher_context_norm.parameters():
            p.requires_grad_(False)

    def _shift_with_bos(self, x: Tensor, bos: Tensor, steps: int) -> Tensor:
        steps = max(int(steps), 1)
        prefix = bos.to(dtype=x.dtype)[None, None, :].expand(x.size(0), steps, -1)
        if x.size(1) <= steps:
            return prefix[:, : x.size(1)]
        return torch.cat((prefix, x[:, :-steps]), dim=1)

    def _encode_context(self, summary: Tensor, blocks, norm_module: RMSNorm) -> Tensor:
        P = summary.size(1)
        if P > self.patch_context_pos.size(0):
            raise ValueError(f"Patch count {P} exceeds learned patch context positions {self.patch_context_pos.size(0)}")
        x = summary + self.patch_context_pos[:P].to(summary.dtype)[None, :, :]
        for block in blocks:
            x = block(x)
        return norm_module(x)

    def _predict_patch_latents(
        self,
        context_states: Tensor,
        jepa_state: JEPAState | None = None,
        *,
        return_state: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, JEPAState | None]:
        pred_in = self._shift_with_bos(context_states, self.predict_bos, self.jepa_horizon)
        context_prev = self._shift_with_bos(context_states, self.context_bos, self.jepa_horizon)
        next_state = jepa_state
        if self.predictor_memory is not None:
            mem_out, next_state = self.predictor_memory(pred_in, jepa_state)
            pred_in = pred_in + mem_out.to(pred_in.dtype)
        pred_in = self.predictor(pred_in)
        summary_pred = self.predictor_summary(pred_in)
        slot_pred = self.slot_predictor(pred_in, summary_pred)
        if return_state:
            return context_prev, summary_pred, slot_pred, next_state
        return context_prev, summary_pred, slot_pred

    @torch.no_grad()
    def _teacher_targets(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        teacher_summary_local, teacher_slots, _byte_mask, _patch_mask = self.teacher_patch_encoder(ids)
        teacher_context = self._encode_context(
            teacher_summary_local, self.teacher_context_encoder, self.teacher_context_norm
        )
        return teacher_summary_local, teacher_slots, teacher_context

    def _sigreg(self, x: Tensor) -> Tensor:
        flat = x.float().reshape(-1, x.size(-1))
        if flat.size(0) < 2:
            return flat.new_zeros(())
        mean = flat.mean(dim=0)
        centered = flat - mean
        var = centered.pow(2).mean(dim=0).clamp_min(1e-5)
        normed = centered / var.sqrt()
        cov = (normed.T @ normed) / float(normed.size(0))
        offdiag = cov - torch.diag(torch.diag(cov))
        return mean.square().mean() + F.relu(1.0 - var.sqrt()).square().mean() + 0.01 * offdiag.square().mean()

    def _masked_mse(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if mask.ndim < pred.ndim:
            mask = mask.view(*mask.shape, *([1] * (pred.ndim - mask.ndim)))
        diff = (pred.float() - target.float()).pow(2)
        denom = mask.float().sum().clamp_min(1.0)
        return (diff * mask.float()).sum() / denom

    def _student_latents(
        self,
        ids: Tensor,
        jepa_state: JEPAState | None = None,
        *,
        return_state: bool = False,
    ):
        summary_local, slots_local, byte_mask, patch_mask = self.patch_encoder(ids)
        context_states = self._encode_context(summary_local, self.context_encoder, self.context_norm)
        if return_state:
            context_prev, summary_pred, slot_pred, next_state = self._predict_patch_latents(
                context_states, jepa_state, return_state=True
            )
            return summary_local, slots_local, context_states, context_prev, summary_pred, slot_pred, byte_mask, patch_mask, next_state
        context_prev, summary_pred, slot_pred = self._predict_patch_latents(context_states, jepa_state)
        return summary_local, slots_local, context_states, context_prev, summary_pred, slot_pred, byte_mask, patch_mask

    def forward_jepa(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            summary_local,
            slots_local,
            _context_states,
            _context_prev,
            summary_pred,
            slot_pred,
            _byte_mask,
            patch_mask,
        ) = self._student_latents(ids)
        teacher_summary_local, teacher_slots, teacher_context = self._teacher_targets(ids)

        valid_mask = patch_mask.clone()
        if valid_mask.size(1) > 0:
            valid_mask[:, : self.jepa_horizon] = False

        summary_loss = self._masked_mse(
            F.normalize(summary_pred, dim=-1),
            F.normalize(teacher_context.detach(), dim=-1),
            valid_mask,
        )
        slot_mask = valid_mask.unsqueeze(-1).expand(-1, -1, self.num_slots)
        slot_loss = self._masked_mse(
            F.normalize(slot_pred, dim=-1),
            F.normalize(teacher_slots.detach(), dim=-1),
            slot_mask,
        )
        summary_stab = self._masked_mse(
            F.normalize(summary_local, dim=-1),
            F.normalize(teacher_summary_local.detach(), dim=-1),
            patch_mask,
        )
        sigreg = self._sigreg(summary_local[patch_mask]) if patch_mask.any() else summary_local.new_zeros(())
        total = (
            self.summary_pred_weight * summary_loss
            + self.slot_pred_weight * slot_loss
            + self.summary_stab_weight * summary_stab
            + self.sigreg_weight * sigreg
        )
        return total, summary_loss.detach(), slot_loss.detach(), summary_stab.detach(), sigreg.detach()

    def _decoder_logits(
        self,
        ids: Tensor,
        jepa_state: JEPAState | None = None,
        *,
        backbone_no_grad: bool = False,
        return_state: bool = False,
    ):
        student_fn = self._student_latents
        ctx = torch.no_grad() if backbone_no_grad else torch.enable_grad()
        with ctx:
            if return_state:
                (
                    _summary_local,
                    _slots_local,
                    _context_states,
                    context_prev,
                    summary_pred,
                    slot_pred,
                    byte_mask,
                    _patch_mask,
                    next_state,
                ) = student_fn(ids, jepa_state, return_state=True)
                logits, flat_mask = self.decoder(ids, byte_mask, context_prev, summary_pred, slot_pred)
                return logits, flat_mask, next_state
            (
                _summary_local,
                _slots_local,
                _context_states,
                context_prev,
                summary_pred,
                slot_pred,
                byte_mask,
                _patch_mask,
            ) = student_fn(ids)
        return self.decoder(ids, byte_mask, context_prev, summary_pred, slot_pred)

    def forward_decoder(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        logits, byte_mask = self._decoder_logits(ids, backbone_no_grad=True)
        logits = self.args.logit_softcap * torch.tanh(logits / self.args.logit_softcap)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size).float(), ids.reshape(-1), reduction="none").view_as(ids)
        mask = byte_mask.float()
        if mask.size(1) > 0:
            mask[:, 0] = 0.0
        total = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return total, total.detach()

    def forward_ttt(self, ids: Tensor, *, scope: str) -> Tensor:
        if scope != "decoder":
            raise ValueError(f"Causal JEPA TTT supports only TTT_SCOPE='decoder', got {scope!r}")
        return self.forward_decoder(ids)[0]

    def forward_logits(
        self,
        ids: Tensor,
        jepa_state: JEPAState | None = None,
        *,
        return_jepa_state: bool = False,
    ) -> Tensor | tuple[Tensor, JEPAState | None]:
        if return_jepa_state:
            logits, _mask, next_state = self._decoder_logits(ids, jepa_state, return_state=True)
            return self.args.logit_softcap * torch.tanh(logits / self.args.logit_softcap), next_state
        logits, _mask = self._decoder_logits(ids)
        return self.args.logit_softcap * torch.tanh(logits / self.args.logit_softcap)

    def forward(self, ids: Tensor):
        if self.train_stage == "jepa":
            return self.forward_jepa(ids)
        if self.train_stage == "decoder":
            loss, ce = self.forward_decoder(ids)
            zero = loss.detach().new_zeros(())
            return loss, ce, zero, zero, zero
        raise ValueError(f"Unsupported training stage {self.train_stage!r}")

    @torch.no_grad()
    def _sync_teacher(self):
        self.teacher_patch_encoder.load_state_dict(self.patch_encoder.state_dict(), strict=True)
        self.teacher_context_norm.load_state_dict(self.context_norm.state_dict(), strict=True)
        for tb, sb in zip(self.teacher_context_encoder, self.context_encoder):
            tb.load_state_dict(sb.state_dict(), strict=True)

    @torch.no_grad()
    def update_teacher(self):
        d = self.ema_decay
        for tp, sp in zip(self.teacher_patch_encoder.parameters(), self.patch_encoder.parameters()):
            tp.data.lerp_(sp.data, 1.0 - d)
        for tp, sp in zip(self.teacher_context_norm.parameters(), self.context_norm.parameters()):
            tp.data.lerp_(sp.data, 1.0 - d)
        for tblock, sblock in zip(self.teacher_context_encoder, self.context_encoder):
            for tp, sp in zip(tblock.parameters(), sblock.parameters()):
                tp.data.lerp_(sp.data, 1.0 - d)

    def serializable_state_dict(self) -> dict[str, Tensor]:
        skip = ("teacher_patch_encoder.", "teacher_context_encoder.", "teacher_context_norm.")
        return {k: v for k, v in self.state_dict().items() if not any(k.startswith(s) for s in skip)}

    def load_serializable(self, sd: dict[str, Tensor]):
        self.load_state_dict(sd, strict=False)

    @torch.no_grad()
    def zero_pad_emb_(self):
        valid_lo = self.args.byte_offset
        valid_hi = min(self.vocab_size, self.args.byte_offset + self.args.byte_count)
        if 0 <= self.pad_id < self.patch_encoder.byte_emb.num_embeddings and not (valid_lo <= self.pad_id < valid_hi):
            self.patch_encoder.byte_emb.weight[self.pad_id].zero_()
        if 0 <= self.pad_id < self.decoder.byte_emb.num_embeddings and not (valid_lo <= self.pad_id < valid_hi):
            self.decoder.byte_emb.weight[self.pad_id].zero_()


def build_model(args: Hyperparameters) -> nn.Module:
    family = args.model_family
    if family in {"two_stage_jepa", "jepa"}:
        return CausalJEPA(args)
    if family in {"compressed_ut", "utcompress", "ut_byte_ce"}:
        return CompressedUTByteModel(args)
    raise ValueError(f"Unsupported MODEL_FAMILY={family!r}")


# ── evaluation ───────────────────────────────────────────────────────────────

def eval_val_jepa(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    amp_dtype: torch.dtype,
) -> tuple[float, float, float, float, float]:
    ctx_len = args.train_seq_len
    usable = (val_tokens.numel() // ctx_len) * ctx_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for TRAIN_SEQ_LEN={ctx_len}")
    total_seqs = usable // ctx_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    totals = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(5)]
    count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, max(args.eval_batch_seqs, 1)):
            batch_seq_end = min(batch_seq_start + max(args.eval_batch_seqs, 1), seq_end)
            raw_start = batch_seq_start * ctx_len
            raw_end = batch_seq_end * ctx_len
            batch = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            batch = batch.reshape(-1, ctx_len)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                total, summary, slot, stab, reg = model(batch)
            bsz = float(batch.size(0))
            for acc, value in zip(totals, (total, summary, slot, stab, reg)):
                acc += value.to(torch.float64) * bsz
            count += bsz

    if dist.is_available() and dist.is_initialized():
        for acc in totals:
            dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    out = [float((acc / count.clamp_min(1.0)).item()) for acc in totals]
    model.train()
    return out[0], out[1], out[2], out[3], out[4]


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
    total = val_tokens.numel()
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
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + ctx_len, total)
                wl = end - ws
                wlens.append(wl)
                chunk = val_tokens[ws:end].to(dtype=torch.int64, device=device)
                x[i, :wl] = chunk
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model.forward_logits(x)
            nll = F.cross_entropy(
                logits.reshape(-1, model.vocab_size).float(), x.reshape(-1), reduction="none"
            ).reshape(bsz, ctx_len)
            for i, ws in enumerate(batch_ws):
                wl = wlens[i]
                s = 1 if ws == 0 else max(wl - stride, 0)
                if wl - s <= 0:
                    continue
                scored = nll[i, s:wl].to(torch.float64)
                loss_sum += scored.sum()
                tok_count += float(wl - s)
                byte_cnt += byte_lut[x[i, s:wl]].to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_cnt, op=dist.ReduceOp.SUM)

    val_loss = float((loss_sum / tok_count.clamp_min(1.0)).item())
    val_bpb = float(loss_sum.item() / (math.log(2.0) * byte_cnt.clamp_min(1.0).item()))
    model.train()
    return val_loss, val_bpb


def select_ttt_named_parameters(model: nn.Module, scope: str) -> list[tuple[str, nn.Parameter]]:
    named = list(model.named_parameters())
    if scope == "decoder":
        return [(name, param) for name, param in named if name.startswith("decoder.")]
    if scope == "all":
        return named
    raise ValueError(f"Unsupported TTT_SCOPE={scope!r}")


def eval_val_ttt_score_first(
    args: Hyperparameters,
    model: nn.Module,
    device: torch.device,
    val_tokens: Tensor,
    byte_lut: Tensor,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    named_ttt_params = select_ttt_named_parameters(model, args.ttt_scope)
    if not named_ttt_params:
        raise ValueError(f"No parameters selected for TTT_SCOPE={args.ttt_scope!r}")
    for param in model.parameters():
        param.requires_grad_(False)
    for _, param in named_ttt_params:
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [param for _, param in named_ttt_params],
        lr=args.ttt_lr,
        betas=(args.ttt_beta1, args.ttt_beta2),
        eps=args.adam_eps,
        weight_decay=args.ttt_weight_decay,
        fused=False,
    )

    total_tokens = val_tokens.numel()
    if total_tokens <= 1:
        raise ValueError("Validation split too short for TTT eval")
    ctx_len = max(int(args.train_seq_len), 2)
    stride = max(int(args.eval_stride), 1)
    chunk_tokens = max(int(args.ttt_chunk_tokens), ctx_len)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    for chunk_start in range(0, total_tokens, chunk_tokens):
        chunk_end = min(chunk_start + chunk_tokens, total_tokens)
        history_start = max(0, chunk_start - ctx_len + 1)
        with torch.inference_mode():
            for ws in range(history_start, chunk_end, stride):
                we = min(ws + ctx_len, total_tokens)
                if we - ws <= 1:
                    continue
                x = val_tokens[ws:we].to(device=device, dtype=torch.int64, non_blocking=True).unsqueeze(0)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    logits = model.forward_logits(x)
                nll = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size).float(),
                    x.reshape(-1),
                    reduction="none",
                ).view(-1).to(torch.float64)
                score_start = max(max((we - ws) - stride, 1), chunk_start - ws)
                score_end = min(we - ws, chunk_end - ws)
                if score_end <= score_start:
                    continue
                scored_tokens = x[0, score_start:score_end]
                loss_sum += nll[score_start:score_end].sum()
                tok_count += float(score_end - score_start)
                byte_count += byte_lut[scored_tokens].to(torch.float64).sum()

        model.train()
        for _ in range(max(args.ttt_epochs, 1)):
            for ws in range(chunk_start, chunk_end, ctx_len):
                seg_end = min(ws + ctx_len, chunk_end)
                seg_start = max(0, ws - 1)
                if seg_end - seg_start <= 1:
                    continue
                ids = val_tokens[seg_start:seg_end].to(device=device, dtype=torch.int64, non_blocking=True).unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    loss = model.forward_ttt(ids, scope=args.ttt_scope)
                loss.backward()
                if args.ttt_grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_([param for _, param in named_ttt_params], args.ttt_grad_clip_norm)
                optimizer.step()
                model.zero_pad_emb_()
        model.eval()

    val_loss = float((loss_sum / tok_count.clamp_min(1.0)).item())
    val_bpb = float(loss_sum.item() / (math.log(2.0) * byte_count.clamp_min(1.0).item()))
    model.train()
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

    if args.train_stage == "decoder" and not args.load_model_path:
        raise ValueError("TRAIN_STAGE=decoder requires LOAD_MODEL_PATH from a completed JEPA stage checkpoint")

    base_model = build_model(args).to(device)
    if args.load_model_path:
        load_obj = torch.load(args.load_model_path, map_location="cpu", weights_only=False)
        if isinstance(load_obj, dict) and "state_dict" in load_obj:
            load_obj = load_obj["state_dict"]
        if not isinstance(load_obj, dict):
            raise ValueError(f"Unsupported checkpoint format at {args.load_model_path!r}")
        base_model.load_state_dict(load_obj, strict=False)
        base_model.configure_training_stage(args.train_stage)
    base_model.zero_pad_emb_()

    zeropower_via_newtonschulz5 = maybe_compile(zeropower_via_newtonschulz5, enabled=args.use_compile)
    train_model = maybe_compile(base_model, enabled=args.use_compile)
    if distributed:
        model = DDP(train_model, device_ids=[local_rank], broadcast_buffers=False)
    else:
        model = train_model

    param_lr_scale_fn = getattr(base_model, "parameter_lr_scale", None)
    named = []
    for n, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        lr_scale = float(param_lr_scale_fn(n)) if callable(param_lr_scale_fn) else 1.0
        named.append((n, p, lr_scale))
    embed_names = {
        "patch_encoder.byte_emb.weight",
        "patch_encoder.bigram_hash.embed.weight",
        "decoder.byte_emb.weight",
        "frontend.byte_emb.weight",
        "frontend.bigram_hash.embed.weight",
    }
    embed_ps: dict[float, list[Tensor]] = {}
    matrix_ps: dict[float, list[Tensor]] = {}
    scalar_ps: dict[float, list[Tensor]] = {}
    for n, p, lr_scale in named:
        if n in embed_names:
            embed_ps.setdefault(lr_scale, []).append(p)
        elif p.ndim == 2:
            matrix_ps.setdefault(lr_scale, []).append(p)
        elif p.ndim < 2:
            scalar_ps.setdefault(lr_scale, []).append(p)

    def scaled_groups(buckets: dict[float, list[Tensor]], base_lr: float) -> list[dict[str, object]]:
        groups: list[dict[str, object]] = []
        for lr_scale, params in sorted(buckets.items(), key=lambda item: item[0]):
            if not params:
                continue
            scaled_lr = base_lr * lr_scale
            groups.append({"params": params, "lr": scaled_lr, "base_lr": scaled_lr})
        return groups

    optimizers: list[torch.optim.Optimizer] = []
    embed_groups = scaled_groups(embed_ps, args.embed_lr)
    if embed_groups:
        opt_e = make_adam(
            embed_groups,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.adam_eps,
        )
        optimizers.append(opt_e)
    matrix_groups = scaled_groups(matrix_ps, args.matrix_lr)
    if matrix_groups:
        opt_m = Muon(matrix_groups, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
        optimizers.append(opt_m)
    else:
        opt_m = None
    scalar_groups = scaled_groups(scalar_ps, args.scalar_lr)
    if scalar_groups:
        opt_s = make_adam(
            scalar_groups,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.adam_eps,
        )
        optimizers.append(opt_s)

    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    n_saved = sum(v.numel() for v in base_model.serializable_state_dict().values())
    log0(f"model_family:{args.model_family} stage:{args.train_stage} params_trainable:{n_params} params_saved:{n_saved}")
    if args.model_family in {"two_stage_jepa", "jepa"}:
        log0(
            f"arch: patch={args.patch_size} slot={args.slot_size} slots={args.patch_size // args.slot_size} "
            f"patch_dim={args.model_dim} summary_dim={args.jepa_latent_dim} slot_dim={args.jepa_slot_dim} "
            f"context_layers={args.num_layers} pred_steps={args.jepa_pred_steps} slot_pred_layers={args.jepa_slot_pred_layers} "
            f"pred_heads={args.jepa_pred_heads} "
            f"memory_dim={args.jepa_memory_dim} bigram={args.bigram_vocab_size}x{args.bigram_dim}"
        )
        log0(
            f"losses: summary={args.summary_pred_weight} slot={args.slot_pred_weight} "
            f"stab={args.summary_stab_weight} sigreg={args.sigreg_weight} ema={args.ema_decay}"
        )
        log0(
            f"decoder: dim={args.decoder_dim} layers={args.decoder_layers} "
            f"bridge_layers={args.decoder_bridge_layers} heads={args.decoder_heads} mlp={args.decoder_mlp_mult}x"
        )
    else:
        log0(
            f"arch: byte_dim={args.byte_dim} model_dim={args.model_dim} stride={args.compress_stride} "
            f"ut_steps={args.ut_steps} heads={args.num_heads}/{args.num_kv_heads} "
            f"rope={args.partial_rope_dim}@{args.rope_base:g} bigram={args.bigram_vocab_size}x{args.bigram_dim}"
        )
        log0(
            f"frontend: conv_kernels={','.join(str(k) for k in args.conv_kernels)} "
            f"conv_dilations={','.join(str(d) for d in args.conv_dilations)} decoder={args.decoder_type} "
            f"dec_dim={args.decoder_dim} dec_layers={args.decoder_layers} dec_bridge={args.decoder_bridge_layers} "
            f"dec_heads={args.decoder_heads} mlp={args.mlp_mult}x dec_mlp={args.decoder_mlp_mult}x"
        )
        if args.train_stage in {"jepa", "joint"}:
            pred_dim = args.pred_dim if args.pred_dim > 0 else max(args.model_dim // 2, 1)
            if args.train_stage == "jepa":
                log0(
                    f"predictor: dim={pred_dim} steps={args.pred_steps} heads={args.pred_heads}/{args.pred_kv_heads} "
                    f"mlp={args.pred_mlp_mult}x jepa={args.jepa_weight} ce_aux={args.ce_aux_weight} sigreg={args.sigreg_weight}"
                )
            else:
                log0(
                    f"predictor: dim={pred_dim} steps={args.pred_steps} heads={args.pred_heads}/{args.pred_kv_heads} "
                    f"mlp={args.pred_mlp_mult}x jepa={args.jepa_weight} ce={args.joint_ce_weight} sigreg={args.sigreg_weight}"
                )
        elif args.train_stage == "decoder":
            log0(f"decoder_finetune: backbone_lr_mult={args.decoder_backbone_lr_mult}")
    log0(
        f"tokens: vocab={args.vocab_size} bytes={args.byte_offset}:{args.byte_offset + args.byte_count - 1} pad_id={args.pad_id}"
    )
    log0(f"train: seq={args.train_seq_len} batch_tokens={args.train_batch_tokens} ga={grad_accum} "
         f"warmdown={args.warmdown_iters} warmup={args.warmup_steps}")
    log0(f"eval: stride={args.eval_stride} int6={int(args.int6_enabled)} "
         f"zstd={int(args.use_zstd)} seed={args.seed}")
    if args.ttt_enabled:
        log0(
            f"ttt: scope={args.ttt_scope} lr={args.ttt_lr:g} wd={args.ttt_weight_decay:g} "
            f"epochs={args.ttt_epochs} chunk_tokens={args.ttt_chunk_tokens} clip={args.ttt_grad_clip_norm:g}"
        )

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
                    loss = model(batch)[0]
                (loss * grad_scale).backward()
            for o in optimizers:
                o.step()
            if args.train_stage in {"jepa", "joint"}:
                base_model.update_teacher()
            base_model.zero_pad_emb_()
            zero_grad()
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opt):
            o.load_state_dict(s)
        if args.train_stage in {"jepa", "joint"}:
            base_model._sync_teacher()
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
            scope = "final" if last else "periodic"
            if args.train_stage in {"jepa", "joint"}:
                vl, vs, vslot, vstab, vreg = eval_val_jepa(args, base_model, rank, world_size, device, ev, amp_dtype)
                if args.model_family in {"compressed_ut", "utcompress", "ut_byte_ce"}:
                    log0(
                        f"step:{step}/{args.iterations} val_jepa:{vs:.4f} val_ce:{vslot:.4f} "
                        f"val_sigreg:{vstab:.4f} val_aux:{vreg:.4f} val_total:{vl:.4f} "
                        f"val_scope:{scope} val_tokens:{ev.numel()} "
                        f"train_time:{train_ms:.0f}ms step_avg:{train_ms / max(step, 1):.2f}ms"
                    )
                else:
                    log0(
                        f"step:{step}/{args.iterations} val_jepa:{vl:.4f} val_summary:{vs:.4f} "
                        f"val_slot:{vslot:.4f} val_stab:{vstab:.4f} val_sigreg:{vreg:.4f} "
                        f"val_scope:{scope} val_tokens:{ev.numel()} "
                        f"train_time:{train_ms:.0f}ms step_avg:{train_ms / max(step, 1):.2f}ms"
                    )
            else:
                vl, vb = eval_val_sliding(args, base_model, rank, world_size, device, ev, byte_lut, amp_dtype)
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
        t_a = torch.zeros((), device=device)
        t_b = torch.zeros((), device=device)
        t_c = torch.zeros((), device=device)
        t_d = torch.zeros((), device=device)
        for ms in range(grad_accum):
            if distributed:
                model.require_backward_grad_sync = ms == grad_accum - 1
            batch = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, a_d, b_d, c_d, d_d = model(batch)
            t_loss += loss.detach()
            t_a += a_d
            t_b += b_d
            t_c += c_d
            t_d += d_d
            (loss * grad_scale).backward()
        t_loss /= grad_accum
        t_a /= grad_accum
        t_b /= grad_accum
        t_c /= grad_accum
        t_d /= grad_accum

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        mm = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        if opt_m is not None:
            for g in opt_m.param_groups:
                g["momentum"] = mm
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers:
            o.step()
        if args.train_stage in {"jepa", "joint"}:
            base_model.update_teacher()
        base_model.zero_pad_emb_()
        zero_grad()

        step += 1
        approx_ms = train_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            if args.train_stage in {"jepa", "joint"}:
                if args.model_family in {"compressed_ut", "utcompress", "ut_byte_ce"}:
                    log0(
                        f"step:{step}/{args.iterations} train_loss:{t_loss.item():.4f} "
                        f"train_jepa:{t_a.item():.4f} train_ce:{t_b.item():.4f} "
                        f"train_sigreg:{t_c.item():.4f} train_aux:{t_d.item():.4f} "
                        f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms"
                    )
                else:
                    log0(
                        f"step:{step}/{args.iterations} train_loss:{t_loss.item():.4f} "
                        f"train_summary:{t_a.item():.4f} train_slot:{t_b.item():.4f} "
                        f"train_stab:{t_c.item():.4f} train_sigreg:{t_d.item():.4f} "
                        f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms"
                    )
            else:
                log0(
                    f"step:{step}/{args.iterations} train_loss:{t_loss.item():.4f} "
                    f"train_ce:{t_a.item():.4f} "
                    f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms"
                )

        cap = max_wc_ms is not None and approx_ms >= max_wc_ms
        if distributed and max_wc_ms is not None:
            ct = torch.tensor(int(cap), device=device)
            dist.all_reduce(ct, op=dist.ReduceOp.MAX)
            cap = bool(ct.item())
        if stop_step is None and cap:
            stop_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if master:
        ckpt_name = "stage1_model.pt" if args.train_stage == "jepa" else "checkpoint_model.pt"
        torch.save(base_model.state_dict(), ckpt_name)
        log0(f"Checkpoint: {ckpt_name} bytes:{os.path.getsize(ckpt_name)}")
        log0(f"Code size: {len(code.encode('utf-8'))} bytes")

    if args.train_stage == "jepa":
        if distributed:
            dist.destroy_process_group()
        return

    if master:
        torch.save(base_model.serializable_state_dict(), "final_model.pt")
        log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes")

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
    base_model.configure_training_stage("decoder")
    base_model.zero_pad_emb_()

    torch.cuda.synchronize()
    t_qe = time.perf_counter()
    q_loss, q_bpb = eval_val_sliding(args, base_model, rank, world_size, device, final_val, byte_lut, amp_dtype)
    torch.cuda.synchronize()
    scope = "full" if args.final_full_val else "capped"
    log0(f"final_quant_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} "
         f"eval_scope:{scope} val_tokens:{final_val.numel()} eval_time:{1000*(time.perf_counter()-t_qe):.0f}ms")
    log0(f"final_quant_roundtrip_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")

    if args.ttt_enabled:
        if distributed:
            dist.barrier()
            if not master:
                dist.destroy_process_group()
                return
            dist.destroy_process_group()
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_ttt_score_first(args, base_model, device, final_val, byte_lut, amp_dtype)
        torch.cuda.synchronize()
        log0(
            f"final_ttt_score_first val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
            f"eval_scope:{scope} val_tokens:{final_val.numel()} eval_time:{1000*(time.perf_counter()-t_ttt):.0f}ms"
        )
        log0(f"final_ttt_score_first_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
        return

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
