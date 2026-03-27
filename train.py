"""
Winning architecture: 11-layer hybrid transformer with 3 quadratic attention layers
and 8 RWKV-style token-shift layers. ~16M params, 512 dim.
Supports multi-GPU via DDP. Launch with:
    torchrun --nproc_per_node=8 train.py
"""

from __future__ import annotations

import copy
import gc
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
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Flash Attention 3
from kernels import get_kernel
_cap = torch.cuda.get_device_capability()
_fa3_repo = "varunneal/flash-attention-3" if _cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(_fa3_repo).flash_attn_interface

# ---------------------------------------------------------------------------
# Hyperparameters
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
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))

    # Training length
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 300.0))

    # Architecture
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = True
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 262_144))

    # Hybrid layer config
    attn_layers = [4, 7, 10]
    window_pattern = "VVVVVVVVVVL"  # V=128, L=full context

    # SOTA techniques
    bigram_vocab_size = 2048
    bigram_dim = 128
    xsa_last_n = 4
    partial_rope_dims = 16
    leaky_relu_slope = 0.5
    ve_dim = 128
    ve_layers = [9, 10]

    # Optimizer
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.030))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    embed_lr = float(os.environ.get("EMBED_LR", 0.035))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # EMA / QAT
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))


# ---------------------------------------------------------------------------
# Muon Optimizer
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, wd: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, wd=wd),
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

            wd = group.get("wd", 0.0)
            curr = 0
            for p in params:
                if wd > 0:
                    p.mul_(1 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# ---------------------------------------------------------------------------
# Quantization (int6 range, stored as int8, zlib compressed)
# ---------------------------------------------------------------------------

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
    "smear", "mix_r", "mix_k", "gate", "scale",
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT6_CLIP_PERCENTILE = 99.99984
INT6_CLIP_Q = INT6_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor_int6(t: Tensor) -> tuple[Tensor, Tensor]:
    """Int6 quantization: range [-32, 31] stored as int8."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT6_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale_val = (clip_abs / 31.0).clamp_min(1.0 / 31.0)
        q = torch.clamp(torch.round(clipped / scale_val[:, None]), -32, 31).to(torch.int8).contiguous()
        return q, scale_val.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT6_CLIP_Q).item()) if t32.numel() else 0.0
    scale_val = torch.tensor(clip_abs / 31.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(
        torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale_val), -32, 31
    ).to(torch.int8).contiguous()
    return q, scale_val


def quantize_state_dict_int6(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "baseline_tensor_bytes", "int6_payload_bytes"), 0
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            passthrough[name] = t
            stats["int6_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += tensor_nbytes(kept)
            continue

        q, s = quantize_float_tensor_int6(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int6_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int6_per_row_v1",
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


def dequantize_state_dict_int6(obj: dict[str, object]) -> dict[str, Tensor]:
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


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
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
            chunks.append(self.tokens[self.pos : self.pos + k])
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
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ---------------------------------------------------------------------------
# Evaluation (BPB metric)
# ---------------------------------------------------------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
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


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Weights stored fp32 for optimizer, cast to input dtype for compute. Supports QAT."""
    _qat_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                s = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / s[:, None]), -32, 31) * s[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()  # straight-through estimator
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


# ---------------------------------------------------------------------------
# Rotary Embeddings (Partial RoPE)
# ---------------------------------------------------------------------------

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 4096):
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
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb_partial(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    """Apply RoPE only to the first rope_dims dimensions."""
    if rope_dims >= x.size(-1):
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    # Partial: only rotate first rope_dims dims
    half = rope_dims // 2
    x_rope = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rotated = torch.cat((x1 * cos[..., :half] + x2 * sin[..., :half],
                           x1 * (-sin[..., :half]) + x2 * cos[..., :half]), dim=-1)
    return torch.cat((x_rotated, x_pass), dim=-1)


# ---------------------------------------------------------------------------
# Bigram Hash Embedding
# ---------------------------------------------------------------------------

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


# ---------------------------------------------------------------------------
# SmearGate
# ---------------------------------------------------------------------------

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


# ---------------------------------------------------------------------------
# TokenShiftMix (RWKV-style)
# ---------------------------------------------------------------------------

class TokenShiftMix(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mix_r = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_k = nn.Parameter(torch.ones(dim) * 0.5)
        self.W_r = CastedLinear(dim, dim, bias=False)
        self.W_k = CastedLinear(dim, dim, bias=False)
        self.W_v = CastedLinear(dim, dim, bias=False)
        self.W_v._zero_init = True

    def forward(self, x: Tensor, window_size=None, v_embed=None) -> Tensor:
        B, T, C = x.shape
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))

        mix_r = torch.sigmoid(self.mix_r.to(dtype=x.dtype))[None, None, :]
        mix_k = torch.sigmoid(self.mix_k.to(dtype=x.dtype))[None, None, :]
        xr = mix_r * x + (1 - mix_r) * x_prev
        xk = mix_k * x + (1 - mix_k) * x_prev

        r = torch.sigmoid(self.W_r(xr))
        k = torch.relu(self.W_k(xk)).square()
        v = self.W_v(k)
        return r * v


# ---------------------------------------------------------------------------
# Causal Self-Attention (with FA3, partial RoPE, XSA, value embedding)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        partial_rope_dims: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_dims = partial_rope_dims
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.use_xsa = False

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Project attention output away from value direction."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj_val = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj_val).reshape(B, T, H, D)

    def forward(self, x: Tensor, window_size: tuple[int, int], v_embed: Tensor | None = None) -> Tensor:
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim)

        # QK norm
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Partial RoPE
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb_partial(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb_partial(k, cos, sin, self.rope_dims)

        # Per-head Q scaling
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        # Flash Attention 3 with sliding window
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)

        # XSA: project away from value direction
        if self.use_xsa:
            y = self._xsa_efficient(y, v)

        y = y.contiguous().reshape(B, T, C)
        return self.proj(y)


# ---------------------------------------------------------------------------
# MLP (LeakyReLU squared)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, leaky_slope: float = 0.5):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.leaky_slope = leaky_slope

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), self.leaky_slope)
        return self.proj(x.square())


# ---------------------------------------------------------------------------
# Block (Attention Layer)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int,
        partial_rope_dims: int = 16,
        leaky_slope: float = 0.5,
        ln_scale: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, partial_rope_dims)
        self.mlp = MLP(dim, mlp_mult, leaky_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, window_size: tuple[int, int], v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, window_size, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out


# ---------------------------------------------------------------------------
# AltBlock (Token-Shift Layer)
# ---------------------------------------------------------------------------

class AltBlock(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, leaky_slope: float = 0.5):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.mixer = TokenShiftMix(dim)
        self.mlp = MLP(dim, mlp_mult, leaky_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, window_size: tuple[int, int], v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.mixer(self.attn_norm(x_in), window_size, v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out))
        return x_out


# ---------------------------------------------------------------------------
# GPT Model (Hybrid Transformer)
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        dim = args.model_dim
        self.logit_softcap = args.logit_softcap

        # Embeddings
        self.tok_emb = nn.Embedding(args.vocab_size, dim)
        self.bigram = BigramHashEmbedding(args.bigram_vocab_size, args.bigram_dim, dim)
        self.smear = SmearGate(dim)

        # Value Embedding (shared, with per-layer scales)
        kv_dim = args.num_kv_heads * (dim // args.num_heads)
        self.ve_embed = nn.Embedding(args.vocab_size, args.ve_dim)
        self.ve_proj = CastedLinear(args.ve_dim, kv_dim, bias=False)
        self.ve_scales = nn.ParameterDict({
            str(i): nn.Parameter(torch.ones(1, dtype=torch.float32) * 0.1)
            for i in args.ve_layers
        })

        # Build hybrid layers
        self.window_sizes = self._compute_window_sizes(args)
        self.blocks = nn.ModuleList()
        for i in range(args.num_layers):
            if i in args.attn_layers:
                block = Block(
                    dim, args.num_heads, args.num_kv_heads, args.mlp_mult,
                    args.rope_base, args.qk_gain_init, i,
                    args.partial_rope_dims, args.leaky_relu_slope,
                )
                # Enable XSA on attention layers within the last xsa_last_n layers
                if i >= args.num_layers - args.xsa_last_n:
                    block.attn.use_xsa = True
            else:
                block = AltBlock(dim, args.mlp_mult, args.leaky_relu_slope)
            self.blocks.append(block)

        self.final_norm = RMSNorm()
        self._init_weights()

    def _compute_window_sizes(self, args: Hyperparameters) -> list[tuple[int, int]]:
        pattern = args.window_pattern.upper()
        window_sizes = []
        for i in range(args.num_layers):
            c = pattern[i % len(pattern)]
            if c == 'V':
                window_sizes.append((128, 128))
            elif c == 'L':
                window_sizes.append((args.train_seq_len, args.train_seq_len))
            else:
                window_sizes.append((args.train_seq_len, args.train_seq_len))
        # Force last layer to full context
        window_sizes[-1] = (args.train_seq_len, args.train_seq_len)
        return window_sizes

    def _init_weights(self) -> None:
        dim = self.args.model_dim
        num_layers = self.args.num_layers

        # Tied embedding: small normal
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)

        # VE embedding
        nn.init.normal_(self.ve_embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.ve_proj.weight)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        args = self.args
        B, T = input_ids.shape

        # Embedding pipeline
        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x

        # Precompute VE for layers that need it
        ve_base = None
        if args.ve_layers:
            ve_base = self.ve_proj(self.ve_embed(input_ids))  # (B, T, kv_dim)

        # Forward through hybrid layers
        for i, block in enumerate(self.blocks):
            v_embed = None
            if ve_base is not None and str(i) in self.ve_scales:
                v_embed = ve_base * self.ve_scales[str(i)].to(dtype=ve_base.dtype)
            x = block(x, x0, self.window_sizes[i], v_embed=v_embed)

        # Output head
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits = F.linear(x, self.tok_emb.weight)  # tied embeddings
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Distributed + CUDA setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Compute grad accumulation steps
    tokens_per_micro = (args.train_batch_tokens // world_size)
    device_batch_size = tokens_per_micro // args.train_seq_len
    if device_batch_size < 1:
        device_batch_size = 1
    actual_tokens_per_step = device_batch_size * args.train_seq_len * world_size
    grad_accum_steps = max(1, args.train_batch_tokens // actual_tokens_per_step)
    grad_scale = 1.0 / grad_accum_steps

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(f"World size: {world_size}, Grad accum: {grad_accum_steps}, Device batch: {device_batch_size}")

    # Tokenizer + validation setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Model setup
    base_model = GPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer setup
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]

    # Embedding params: tok_emb + bigram + ve_embed
    embed_params = [base_model.tok_emb.weight, base_model.bigram.embed.weight, base_model.ve_embed.weight]
    # Scalar control params not in blocks
    other_scalar_params = [
        base_model.bigram.scale,
        base_model.smear.gate,
    ]
    for key in base_model.ve_scales:
        other_scalar_params.append(base_model.ve_scales[key])
    scalar_params.extend(other_scalar_params)

    # VE proj and bigram proj are matrix params
    ve_proj_params = [p for p in base_model.ve_proj.parameters() if p.ndim == 2]
    bigram_proj_params = [p for p in base_model.bigram.proj.parameters() if p.ndim == 2]
    matrix_params.extend(ve_proj_params)
    matrix_params.extend(bigram_proj_params)

    optimizer_embed = torch.optim.Adam(
        [{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2),
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        wd=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_embed, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"Model params: {n_params:,} ({n_params / 1e6:.1f}M)")
    log0(f"Attention layers: {args.attn_layers}")
    log0(f"Window pattern: {args.window_pattern}")
    log0(f"Train seq len: {args.train_seq_len}")
    log0(f"Train batch tokens: {args.train_batch_tokens:,}")
    log0(f"Max wallclock: {args.max_wallclock_seconds}s")

    # Data loader
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # EMA state
    ema_state: dict[str, Tensor] = {}
    for name, t in base_model.state_dict().items():
        ema_state[name] = t.detach().float().clone()

    # Warmup (primes torch.compile, then reset)
    if args.warmup_steps > 0:
        initial_model_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        initial_ema = {name: t.clone() for name, t in ema_state.items()}
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if warmup_step + 1 == args.warmup_steps or (warmup_step + 1) % 5 == 0:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        # Reset everything
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        ema_state = initial_ema
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Main training loop
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Enable QAT near end of training
        CastedLinear._qat_enabled = scale < args.late_qat_threshold

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        # LR schedule
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        # Grad clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # EMA update
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0)
        if should_log:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"lr_scale:{scale:.3f} train_time:{approx_training_time_ms:.0f}ms "
                f"step_avg:{approx_training_time_ms / step:.1f}ms"
            )

        # GC management
        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()

        # Wallclock cap sync
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"Training complete. Steps: {step}")
    log0(f"Peak VRAM: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # Apply EMA weights
    log0("Applying EMA weights...")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)

    # Eval before quantization
    if master_process:
        torch.cuda.synchronize()
        val_loss, val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        log0(f"EMA val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # Quantize + compress
    quant_obj, quant_stats = quantize_state_dict_int6(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)

    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int6.ptz")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Compressed model: {quant_file_bytes:,} bytes")
        log0(f"Code size: {code_bytes:,} bytes")
        log0(f"Total artifact: {quant_file_bytes + code_bytes:,} bytes")

    # Roundtrip validation
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int6(quant_state), strict=True)

    torch.cuda.synchronize()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    log0(f"Roundtrip int6+zlib val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    log0(f"Roundtrip exact val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
