"""
10L LeakyReLU²  MLP3x + EMA + SlidingEval + Int6 QAT + LZMA-9

Architecture: 10 transformer layers, 512 dim, 8 heads / 4 KV (GQA), MLP 3x (1536)
              with LeakyReLU(0.5)²  activation, U-Net skip connections, RoPE, tied embeddings.
Training:     Muon optimizer with weight decay, EMA(0.997), Int6 QAT (late-stage STE).
Eval:         Sliding window (stride=256) on EMA weights post int6+LZMA roundtrip.

Hard stop: let's keep the script under 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import lzma
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
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------------------------
# GLOBAL QAT FLAG – toggled on during the last fraction of training
# ---------------------------------------------------------------------------
_QAT_ENABLED = False

# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------

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

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape – 10 layers, 3x MLP, GQA
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))

    # EMA
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    ema_start_step = int(os.environ.get("EMA_START_STEP", 100))

    # QAT  – enable STE fake-quantization for the last (1 - qat_start_frac) of training
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.80))
    qat_bits = int(os.environ.get("QAT_BITS", 6))

    # Sliding-window evaluation
    eval_stride = int(os.environ.get("EVAL_STRIDE", 256))
    sliding_eval = bool(int(os.environ.get("SLIDING_EVAL", "1")))

    # Compression: lzma (default, stdlib), zstd, or zlib
    compress_method = os.environ.get("COMPRESS_METHOD", "lzma")


# ---------------------------------------------------------------------------
# CONTROL-TENSOR PATTERNS (kept at higher precision during quantization)
# ---------------------------------------------------------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if p
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16


# ---------------------------------------------------------------------------
# QAT: FAKE INT6 QUANTIZATION WITH STRAIGHT-THROUGH ESTIMATOR
# ---------------------------------------------------------------------------

def fake_quantize_per_row(w: Tensor, bits: int = 6) -> Tensor:
    """Per-row symmetric fake-quantize.  Forward uses quantized weights,
    backward uses STE (gradients flow through as if the rounding didn't happen)."""
    max_val = (1 << (bits - 1)) - 1  # 31 for 6-bit
    amax = w.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = amax / max_val
    q = (w / scale).round().clamp(-max_val, max_val)
    dq = q * scale
    return w + (dq - w).detach()


# ---------------------------------------------------------------------------
# MUON OPTIMIZER  (from modded-nanogpt)
# ---------------------------------------------------------------------------

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
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total = sum(int(p.numel()) for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            cur = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    s = self.state[p]
                    if "momentum_buffer" not in s:
                        s["momentum_buffer"] = torch.zeros_like(g)
                    buf = s["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[cur: cur + p.numel()] = g.reshape(-1)
                cur += p.numel()
            if distributed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            cur = 0
            for p in params:
                g = flat[cur: cur + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                cur += p.numel()
        return loss


# ---------------------------------------------------------------------------
# TOKENIZER-AGNOSTIC BPB EVALUATION HELPERS
# ---------------------------------------------------------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vs = int(sp.vocab_size())
    table_size = max(sp_vs, vocab_size)
    base_bytes_np = np.zeros(table_size, dtype=np.int16)
    has_space_np = np.zeros(table_size, dtype=np.bool_)
    is_boundary_np = np.ones(table_size, dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files for: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Val split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


# ---------------------------------------------------------------------------
# POST-TRAINING INT6 QUANTIZATION + COMPRESSION
# ---------------------------------------------------------------------------

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def pack_int6(t: Tensor) -> tuple[Tensor, int]:
    """Pack int8 tensor (values [-31,31]) into 6-bit packed uint8.
    4 values → 3 bytes.  Returns (packed_uint8, original_numel)."""
    numel = t.numel()
    flat = (t.flatten().to(torch.int16) + 31).to(torch.uint8)  # [0, 62]
    pad = (4 - numel % 4) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    flat = flat.reshape(-1, 4)
    v0, v1, v2, v3 = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    b0 = (v0 << 2) | (v1 >> 4)
    b1 = ((v1 & 0xF) << 4) | (v2 >> 2)
    b2 = ((v2 & 0x3) << 6) | v3
    return torch.stack([b0, b1, b2], dim=1).flatten().contiguous(), numel


def unpack_int6(packed: Tensor, numel: int) -> Tensor:
    """Unpack 6-bit packed uint8 → int8 tensor (values [-31,31])."""
    packed = packed.reshape(-1, 3)
    b0, b1, b2 = packed[:, 0], packed[:, 1], packed[:, 2]
    v0 = b0 >> 2
    v1 = ((b0 & 0x3) << 4) | (b1 >> 4)
    v2 = ((b1 & 0xF) << 2) | (b2 >> 6)
    v3 = b2 & 0x3F
    flat = torch.stack([v0, v1, v2, v3], dim=1).flatten()[:numel]
    return (flat.to(torch.int16) - 31).to(torch.int8)


def quantize_float_tensor_int6(t: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row int6 for 2-D, per-tensor int6 for 1-D / scalars."""
    t32 = t.float()
    if t32.ndim == 2:
        amax = t32.abs().amax(dim=1).clamp(min=1e-5)
        scale = amax / 31.0
        q = (t32 / scale[:, None]).round().clamp(-31, 31).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    amax = float(t32.abs().max().item()) if t32.numel() else 0.0
    scale = torch.tensor(max(amax / 31.0, 1.0 / 31.0), dtype=torch.float32)
    q = (t32 / scale).round().clamp(-31, 31).to(torch.int8)
    return q.contiguous(), scale


def keep_float_tensor(name: str, t: Tensor, pt_dtypes: dict[str, str]) -> Tensor:
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        pt_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_state_dict_int6(state_dict: dict[str, Tensor]):
    quantized, scales, dtypes = {}, {}, {}
    passthrough: dict[str, Tensor] = {}
    pt_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int6_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int6_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, pt_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor_int6(t)
        packed, orig_numel = pack_int6(q)
        qmeta[name] = {"shape": list(q.shape), "numel": orig_numel}
        if s.ndim > 0:
            qmeta[name]["scheme"] = "per_row"
            qmeta[name]["axis"] = 0
        quantized[name] = packed
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int6_payload_bytes"] += tensor_nbytes(packed) + tensor_nbytes(s)
    obj: dict = {
        "__quant_format__": "int6_packed_v2",
        "quantized": quantized, "scales": scales, "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if pt_dtypes:
        obj["passthrough_orig_dtypes"] = pt_dtypes
    return obj, stats


def dequantize_state_dict_int6(obj: dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    pt_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, packed in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        shape = meta.get("shape")
        numel = meta.get("numel")
        if shape is not None and numel is not None:
            q = unpack_int6(packed, numel).reshape(shape)
        else:
            q = packed  # fallback: already int8
        if meta.get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig = pt_dtypes.get(name)
        if isinstance(orig, str):
            out_t = out_t.to(dtype=getattr(torch, orig)).contiguous()
        out[name] = out_t
    return out


def compress_bytes(data: bytes, method: str = "lzma") -> bytes:
    if method == "lzma":
        return lzma.compress(data, preset=9)
    return zlib.compress(data, level=9)


def decompress_bytes(data: bytes, method: str = "lzma") -> bytes:
    if method == "lzma":
        return lzma.decompress(data)
    return zlib.decompress(data)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_data_shard(file: Path) -> Tensor:
    hdr_bytes = 256 * np.dtype("<i4").itemsize
    tok_bytes = np.dtype("<u2").itemsize
    hdr = np.fromfile(file, dtype="<i4", count=256)
    if hdr.size != 256 or int(hdr[0]) != 20240520 or int(hdr[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    n = int(hdr[2])
    if file.stat().st_size != hdr_bytes + n * tok_bytes:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens = np.fromfile(file, dtype="<u2", count=n, offset=hdr_bytes)
    if tokens.size != n:
        raise ValueError(f"Short read: {file}")
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        left = n
        while left > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(left, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, accum: int) -> tuple[Tensor, Tensor]:
        local = global_tokens // (self.world_size * accum)
        span = local + 1
        chunk = self.stream.take(span * self.world_size)
        start = self.rank * span
        local_t = chunk[start: start + span].to(dtype=torch.int64)
        x, y = local_t[:-1].reshape(-1, seq_len), local_t[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ---------------------------------------------------------------------------
# TRANSFORMER MODULES
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """FP32 weights; cast to compute dtype at matmul time.
    When _QAT_ENABLED is True, applies per-row fake int6 quantization (STE)."""

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if self.training and _QAT_ENABLED and self.weight.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            w = fake_quantize_per_row(w, bits=Hyperparameters.qat_bits)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._len = 0
        self._cos: Tensor | None = None
        self._sin: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos is None or self._len != seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
            self._len = seq_len
        return self._cos.to(dtype=dtype), self._sin.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            try:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            except TypeError:
                reps = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(reps, dim=1)
                v = v.repeat_interleave(reps, dim=1)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, D))


class MLP(nn.Module):
    """LeakyReLU(0.5)^2 gated MLP – more expressive activation than ReLU^2."""

    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, tie_embeddings: bool,
                 tied_embed_init_std: float, logit_softcap: float,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0, skips = x, []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if target_ids is None:
            return logits  # for sliding-window eval
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               target_ids.reshape(-1), reduction="mean")


# ---------------------------------------------------------------------------
# EMA (EXPONENTIAL MOVING AVERAGE)
# ---------------------------------------------------------------------------

class EMA:
    """Polyak-style EMA of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        for n, p in model.named_parameters():
            self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            self.shadow[n].lerp_(p.data, 1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> dict[str, Tensor]:
        """Copy EMA weights into model; return backup of originals."""
        backup: dict[str, Tensor] = {}
        for n, p in model.named_parameters():
            backup[n] = p.data.clone()
            p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model: nn.Module, backup: dict[str, Tensor]):
        for n, p in model.named_parameters():
            p.data.copy_(backup[n])


# ---------------------------------------------------------------------------
# STANDARD EVALUATION (non-overlapping, used during training)
# ---------------------------------------------------------------------------

def eval_val(
    args, model, rank, world_size, device, grad_accum_steps,
    val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_seqs = max(local_batch_tokens // args.train_seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    s0 = (total_seqs * rank) // world_size
    s1 = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(s0, s1, local_seqs):
            be = min(bs + local_seqs, s1)
            raw_s, raw_e = bs * args.train_seq_len, be * args.train_seq_len + 1
            local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bl = model(x, y).detach()
            n = float(y.numel())
            loss_sum += bl.to(torch.float64) * n
            tok_cnt += n
            prev, tgt = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.int16)
            tb += (has_space_lut[tgt] & ~is_boundary_lut[prev]).to(torch.int16)
            byte_cnt += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_cnt, byte_cnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = loss_sum / tok_cnt
    bpt = vl.item() / math.log(2.0)
    tpb = tok_cnt.item() / byte_cnt.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)


# ---------------------------------------------------------------------------
# SLIDING-WINDOW EVALUATION (overlapping, used for final BPB)
# ---------------------------------------------------------------------------

def eval_val_sliding(
    args, model, rank, world_size, device,
    val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
    stride: int = 256,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total = val_tokens.numel() - 1
    starts = list(range(0, total - seq_len + 1, stride))
    if not starts:
        starts = [0]

    # Distribute windows round-robin across ranks
    rank_starts = starts[rank::world_size]
    batch_seqs = max(1, args.val_batch_size // seq_len)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(rank_starts), batch_seqs):
            batch = rank_starts[bi: bi + batch_seqs]
            x_b = torch.stack([val_tokens[s: s + seq_len] for s in batch]).to(device=device, dtype=torch.int64)
            y_b = torch.stack([val_tokens[s + 1: s + seq_len + 1] for s in batch]).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x_b)  # [B, seq_len, V]
            ptl = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y_b.reshape(-1), reduction="none"
            ).reshape(len(batch), seq_len)

            for i, ws in enumerate(batch):
                cf = 0 if ws == 0 else (seq_len - stride)
                nl = ptl[i, cf:]
                loss_sum += nl.to(torch.float64).sum()
                tok_cnt += nl.numel()
                ny, nx = y_b[i, cf:], x_b[i, cf:]
                tb = base_bytes_lut[ny].to(torch.int16)
                tb += (has_space_lut[ny] & ~is_boundary_lut[nx]).to(torch.int16)
                byte_cnt += tb.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_cnt, byte_cnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = loss_sum / tok_cnt
    bpt = vl.item() / math.log(2.0)
    tpb = tok_cnt.item() / byte_cnt.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)


# ---------------------------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------------------------

def main() -> None:
    global zeropower_via_newtonschulz5, _QAT_ENABLED

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # ---- distributed + CUDA setup ----
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "GRAD_ACCUM_STEPS" in os.environ:
        grad_accum_steps = int(os.environ["GRAD_ACCUM_STEPS"])
    else:
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
        grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 80, console=False)
    log0(f"Python {sys.version}  PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout, console=False)
    log0("=" * 80, console=False)

    # ---- tokenizer + validation ----
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Vocab mismatch: {args.vocab_size} vs {int(sp.vocab_size())}")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_space_lut, is_boundary_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val tokens:{val_tokens.numel() - 1}")

    # ---- model ----
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # ---- optimizers ----
    block_np = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_np if p.ndim == 2 and not any(c in n for c in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_np if p.ndim < 2 or any(c in n for c in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    tok_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.Adam([{"params": [base_model.tok_emb.weight], "lr": tok_lr, "base_lr": tok_lr}],
                               betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                  betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar]
    if base_model.lm_head is not None:
        opt_head = torch.optim.Adam([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
                                    betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, opt_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"params:{n_params}  layers:{args.num_layers}  dim:{args.model_dim}  "
         f"heads:{args.num_heads}/{args.num_kv_heads}  mlp_mult:{args.mlp_mult}  "
         f"ema_decay:{args.ema_decay}  muon_wd:{args.muon_wd}  qat_bits:{args.qat_bits}")
    log0(f"world:{world_size}  accum:{grad_accum_steps}  batch_tokens:{args.train_batch_tokens}  "
         f"warmdown:{args.warmdown_iters}  sliding_eval:{args.sliding_eval}  stride:{args.eval_stride}")

    # ---- EMA ----
    ema = EMA(base_model, decay=args.ema_decay)

    # ---- data loader ----
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_wall_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def zero_grad():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wall_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        sms = elapsed_ms / max(step, 1)
        wms = args.warmdown_iters * sms
        rms = max(max_wall_ms - elapsed_ms, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

    # ---- warmup (compile priming) ----
    if args.warmup_steps > 0:
        init_sd = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for o in optimizers:
                o.step()
            zero_grad()
            if ws + 1 == args.warmup_steps or (ws + 1) % 10 == 0:
                log0(f"warmup {ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opt, strict=True):
            o.load_state_dict(s)
        zero_grad()
        ema = EMA(base_model, decay=args.ema_decay)  # re-init EMA after warmup
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ---- training ----
    train_ms, stop_step = 0.0, None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last = step == args.iterations or (stop_step is not None and step >= stop_step)

        # Periodic validation (standard, fast)
        do_val = last or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if do_val:
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                              val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"time:{train_ms:.0f}ms avg:{train_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last:
            if stop_step is not None and step < args.iterations:
                log0(f"wallclock_stop step:{step}/{args.iterations} time:{train_ms:.0f}ms")
            break

        elapsed = train_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed)

        # Enable QAT in the last fraction of training
        if not _QAT_ENABLED and args.qat_start_frac > 0:
            if max_wall_ms:
                frac = elapsed / max_wall_ms
            else:
                frac = step / max(args.iterations, 1)
            if frac >= args.qat_start_frac:
                _QAT_ENABLED = True
                log0(f"QAT_ENABLED at step:{step} frac:{frac:.3f} bits:{args.qat_bits}")

        zero_grad()
        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac_m = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for g in opt_muon.param_groups:
            g["momentum"] = (1 - frac_m) * args.muon_momentum_warmup_start + frac_m * args.muon_momentum
            g["lr"] = g["base_lr"] * scale
        for o in optimizers:
            if o is not opt_muon:
                for g in o.param_groups:
                    g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers:
            o.step()

        # Muon weight decay (decoupled, applied after step)
        if args.muon_wd > 0:
            wd_lr = args.matrix_lr * scale
            wd_factor = 1.0 - wd_lr * args.muon_wd
            if wd_factor < 1.0:
                with torch.no_grad():
                    for p in matrix_params:
                        p.mul_(wd_factor)

        zero_grad()
        step += 1

        # EMA update
        if step >= args.ema_start_step:
            ema.update(base_model)

        approx_ms = train_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"time:{approx_ms:.0f}ms avg:{approx_ms / step:.2f}ms"
                 f"{' QAT' if _QAT_ENABLED else ''}")

        reached = max_wall_ms is not None and approx_ms >= max_wall_ms
        if distributed and max_wall_ms is not None:
            rt = torch.tensor(int(reached), device=device)
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
            reached = bool(rt.item())
        if stop_step is None and reached:
            stop_step = step

    log0(f"peak_mem alloc:{torch.cuda.max_memory_allocated() // 1024 // 1024}MiB "
         f"reserved:{torch.cuda.max_memory_reserved() // 1024 // 1024}MiB")

    # ---- Apply EMA weights for serialization ----
    log0("Applying EMA weights for serialization...")
    ema_backup = ema.apply_to(base_model)

    # ---- INT6 quantization + compression ----
    quant_obj, qstats = quantize_state_dict_int6(base_model.state_dict())
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    blob = compress_bytes(raw, args.compress_method)
    ext = {"lzma": "ptlzma", "zlib": "ptz"}.get(args.compress_method, "ptc")

    if master:
        with open(f"final_model.int6.{ext}", "wb") as f:
            f.write(blob)
        model_bytes = len(blob)
        code_bytes = len(code.encode("utf-8"))
        ratio = qstats["baseline_tensor_bytes"] / max(qstats["int6_payload_bytes"], 1)
        log0(f"int6+{args.compress_method}: {model_bytes} bytes "
             f"(payload:{qstats['int6_payload_bytes']} raw:{len(raw)} ratio:{ratio:.2f}x)")
        log0(f"code: {code_bytes} bytes")
        total = model_bytes + code_bytes
        log0(f"TOTAL ARTIFACT: {total} bytes  {'PASS' if total <= 16_000_000 else 'OVER 16MB!'}")

    # ---- Roundtrip: reload quantized model and evaluate ----
    if distributed:
        dist.barrier()
    with open(f"final_model.int6.{ext}", "rb") as f:
        disk_blob = f.read()
    rt_state = torch.load(io.BytesIO(decompress_bytes(disk_blob, args.compress_method)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int6(rt_state), strict=True)

    torch.cuda.synchronize()
    t_eval = time.perf_counter()

    # Standard eval on roundtripped model
    q_vl, q_vb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                          val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut)
    log0(f"roundtrip_standard val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f}")

    # Sliding-window eval for better BPB
    if args.sliding_eval:
        sw_vl, sw_vb = eval_val_sliding(args, model, rank, world_size, device,
                                        val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
                                        stride=args.eval_stride)
        torch.cuda.synchronize()
        log0(f"roundtrip_sliding val_loss:{sw_vl:.4f} val_bpb:{sw_vb:.4f} stride:{args.eval_stride} "
             f"eval_time:{1000.0 * (time.perf_counter() - t_eval):.0f}ms")
        log0(f"roundtrip_sliding_exact val_loss:{sw_vl:.8f} val_bpb:{sw_vb:.8f}")
    else:
        torch.cuda.synchronize()
        log0(f"roundtrip eval_time:{1000.0 * (time.perf_counter() - t_eval):.0f}ms")

    log0(f"final_exact val_loss:{q_vl:.8f} val_bpb:{q_vb:.8f}")

    # Restore original weights (cleanup)
    ema.restore(base_model, ema_backup)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
