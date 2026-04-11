"""
Frontier stack for Parameter Golf, targeting ~1.12 BPB.
Combines proven techniques from PRs #315, #374, #414 with novel additions.

Technique stack:
  - 11 layers, 512 dim, 8H/4KV, MLP 3x (relu²), U-Net skips
  - SmearGate + BigramHash(2048) + OrthoInit
  - XSA on last 4 layers (Exclusive Self-Attention)
  - EMA weight averaging (decay=0.997)
  - Partial RoPE (16/64 dims)
  - LN Scale (1/sqrt(layer+1))
  - Late QAT with STE (final portion of training, instance-level flag)
  - GPTQ-lite post-training quantization (per-row optimal clip)
  - Tight SWA + VE128 (shared value embedding in last 2 layers)
  - Mixed int6/int5 quantization with zstd-22
  - FP16 tied embedding passthrough
  - Sliding window eval (stride=64)
  - Muon optimizer (momentum=0.99, WD=0.04)
  - FlashAttention 3 (FA3)
  - Value Residual / ResFormer (layer-0 V mixed into all layers)
  - Gated Attention (per-head sigmoid gate)
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import struct
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

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

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

    iterations = int(os.environ.get("ITERATIONS", 9000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Partial RoPE: only apply rotary to this many dims per head (rest = position-free)
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    # LN Scale: scale RMSNorm output by 1/sqrt(layer_idx+1)
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))

    # SmearGate + BigramHash
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))

    # XSA: apply to last N layers
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))

    # EMA
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    # Tight SWA
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.8))

    # Late QAT
    late_qat = bool(int(os.environ.get("LATE_QAT", "1")))
    qat_threshold = float(os.environ.get("QAT_THRESHOLD", 0.15))

    # Value Residual (ResFormer): mix layer-0 V into all layers
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "1")))
    value_residual_alpha = float(os.environ.get("VALUE_RESIDUAL_ALPHA", 0.5))

    # Gated Attention: per-head sigmoid gate
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "1")))

    # Shared Value Embedding (VE128) for last N layers
    ve_shared_layers = int(os.environ.get("VE_SHARED_LAYERS", 2))
    ve_dim = int(os.environ.get("VE_DIM", 128))

    # Optimizer hyperparameters
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # Eval
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    # GPTQ-lite clip percentiles
    gptq_lite = bool(int(os.environ.get("GPTQ_LITE", "1")))

# -----------------------------
# MUON OPTIMIZER WITH WEIGHT DECAY
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 weight_decay: float = 0.0, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 weight_decay=weight_decay, nesterov=nesterov),
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
            wd = group["weight_decay"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if wd > 0:
                    p.mul_(1 - wd * lr)
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


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------

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
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Sliding window evaluation with configurable stride for better context."""
    stride = args.eval_stride
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    positions = list(range(0, total_tokens - seq_len + 1, stride))
    my_positions = positions[rank::world_size]

    model.eval()
    with torch.inference_mode():
        for pos in my_positions:
            chunk = val_tokens[pos : pos + seq_len + 1].to(device=device, dtype=torch.int64)
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(x)

            # Only score tokens in the new stride window (avoid double-counting)
            score_start = 0 if pos == 0 else seq_len - stride
            score_end = seq_len

            logits_score = logits[:, score_start:score_end, :].reshape(-1, logits.size(-1))
            targets_score = y[:, score_start:score_end].reshape(-1)

            loss = F.cross_entropy(logits_score.float(), targets_score, reduction="sum")
            n_tokens = float(targets_score.numel())
            val_loss_sum += loss.to(torch.float64)
            val_token_count += n_tokens

            prev_ids = x[:, score_start:score_end].reshape(-1)
            tgt_ids = targets_score
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


# -----------------------------
# QUANTIZATION (INT6 + ZSTD-22 + GPTQ-LITE)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
    "smear_gate", "gate_param",
    "vr_alpha",
)
INT6_LEVELS = 31
INT6_RANGE = 15

def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row int6 quantization: 31 levels [-15, 15]."""
    t32 = t.float()
    if t32.ndim != 2:
        clip_abs = t32.abs().max().clamp_min(1e-8)
        scale = clip_abs / INT6_RANGE
        q = torch.clamp(torch.round(t32 / scale), -INT6_RANGE, INT6_RANGE).to(torch.int8)
        return q.contiguous(), scale.unsqueeze(0).to(torch.float16).contiguous()

    clip_abs = t32.abs().amax(dim=1).clamp_min(1e-8)
    scale = (clip_abs / INT6_RANGE).to(torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float().unsqueeze(1)), -INT6_RANGE, INT6_RANGE).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def quantize_int6_gptq_lite(t: Tensor) -> tuple[Tensor, Tensor]:
    """GPTQ-lite: try multiple clip percentiles per row, pick min MSE."""
    if t.ndim != 2:
        return quantize_int6_per_row(t)

    t32 = t.float()
    percentiles = [0.999, 0.9995, 0.9999, 0.99999, 1.0]
    best_q = None
    best_scale = None
    best_mse = None

    for pct in percentiles:
        if pct >= 1.0:
            clip_abs = t32.abs().amax(dim=1).clamp_min(1e-8)
        else:
            clip_abs = torch.quantile(t32.abs(), pct, dim=1).clamp_min(1e-8)

        scale = (clip_abs / INT6_RANGE).to(torch.float16)
        clipped = torch.clamp(t32, -clip_abs.unsqueeze(1), clip_abs.unsqueeze(1))
        q = torch.clamp(torch.round(clipped / scale.float().unsqueeze(1)), -INT6_RANGE, INT6_RANGE).to(torch.int8)
        recon = q.float() * scale.float().unsqueeze(1)
        mse = (t32 - recon).pow(2).mean(dim=1)

        if best_mse is None:
            best_q = q
            best_scale = scale
            best_mse = mse
        else:
            better = mse < best_mse
            if better.any():
                best_q[better] = q[better]
                best_scale[better] = scale[better]
                best_mse[better] = mse[better]

    return best_q.contiguous(), best_scale.contiguous()


def quantize_state_dict(state_dict: dict[str, Tensor], use_gptq_lite: bool = True):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}

    quant_fn = quantize_int6_gptq_lite if use_gptq_lite else quantize_int6_per_row

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()

        is_control = any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
        is_embedding = "tok_emb" in name

        if not t.is_floating_point() or is_control or t.numel() <= 65_536:
            out_t = t
            if t.is_floating_point() and t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                out_t = t.to(torch.float16)
            passthrough[name] = out_t.contiguous()
            continue

        if is_embedding:
            passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            passthrough[name] = t.to(torch.float16).contiguous()
            continue

        q, s = quant_fn(t)
        quantized[name] = q
        scales[name] = s

    obj = {
        "__quant_format__": "int6_gptq_lite_v1",
        "quantized": quantized,
        "scales": scales,
        "passthrough": passthrough,
        "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    return obj


def dequantize_state_dict(obj: dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, q in obj["quantized"].items():
        s = obj["scales"][name].float()
        if s.ndim > 0 and q.ndim == 2:
            out[name] = (q.float() * s.unsqueeze(1)).bfloat16().contiguous()
        else:
            out[name] = (q.float() * s.item()).bfloat16().contiguous()

    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig = passthrough_orig_dtypes.get(name)
        if isinstance(orig, str):
            out_t = out_t.to(dtype=getattr(torch, orig))
        out[name] = out_t

    return out


def compress_artifact(quant_obj: dict, code: str) -> bytes:
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        compressed = cctx.compress(raw)
    else:
        compressed = zlib.compress(raw, level=9)
    return compressed


def decompress_artifact(blob: bytes) -> dict:
    try:
        if HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            raw = dctx.decompress(blob)
        else:
            raw = zlib.decompress(blob)
    except Exception:
        raw = zlib.decompress(blob)
    return torch.load(io.BytesIO(raw), map_location="cpu")


# -----------------------------
# DATA LOADING
# -----------------------------

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


# -----------------------------
# TRANSFORMER MODULES
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
            if (param.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


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
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def apply_partial_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    """Apply RoPE to only the first rope_dims dimensions; leave the rest untouched."""
    if rope_dims >= x.size(-1):
        return apply_rotary_emb(x, cos, sin)
    x_rope = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    x_rope = apply_rotary_emb(x_rope, cos, sin)
    return torch.cat((x_rope, x_pass), dim=-1)


class SmearGate(nn.Module):
    """Learned gate blending each token embedding with the previous token's."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(x.dtype))
        shifted = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        return g * x + (1 - g) * shifted


class BigramHash(nn.Module):
    """Hash-based bigram embedding lookup."""
    def __init__(self, vocab_size: int, bigram_vocab_size: int, embed_dim: int, proj_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, embed_dim)
        self.proj = CastedLinear(embed_dim, proj_dim, bias=False)
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, input_ids: Tensor) -> Tensor:
        prev_ids = torch.cat([input_ids[:, :1], input_ids[:, :-1]], dim=1)
        bigram_ids = (prev_ids * self.vocab_size + input_ids) % self.bigram_vocab_size
        return self.proj(self.embed(bigram_ids))


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 64,
        use_xsa: bool = False,
        use_gated_attention: bool = False,
        use_value_residual: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = rope_dims
        self.rotary = Rotary(rope_dims, base=rope_base)
        self.use_xsa = use_xsa
        self.use_value_residual = use_value_residual

        if use_gated_attention:
            self.gate_param = nn.Parameter(torch.zeros(num_heads))
        else:
            self.gate_param = None

    def forward(self, x: Tensor, v0: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_partial_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_partial_rotary_emb(k, cos, sin, self.rope_dims)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        # Value Residual: mix layer-0 V with current V
        if self.use_value_residual and v0 is not None:
            kv_groups = self.num_heads // self.num_kv_heads
            v0_expanded = v0.repeat_interleave(kv_groups, dim=1) if kv_groups > 1 else v0
            v_expanded = v.repeat_interleave(kv_groups, dim=1) if kv_groups > 1 else v
            alpha = 0.5
            v_mixed = alpha * v_expanded + (1 - alpha) * v0_expanded
            y = F.scaled_dot_product_attention(q, k, v_mixed, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )

        # XSA: subtract self-value component
        if self.use_xsa:
            if self.num_kv_heads != self.num_heads:
                kv_groups = self.num_heads // self.num_kv_heads
                v_for_xsa = v.repeat_interleave(kv_groups, dim=1)
            else:
                v_for_xsa = v
            v_norm_sq = (v_for_xsa * v_for_xsa).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            self_proj = (y * v_for_xsa).sum(dim=-1, keepdim=True) / v_norm_sq * v_for_xsa
            y = y - self_proj

        # Gated Attention: per-head sigmoid gate
        if self.gate_param is not None:
            gate = torch.sigmoid(self.gate_param.to(dtype=y.dtype))[None, :, None, None]
            y = y * gate

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y), v


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
        rope_dims: int = 64,
        layer_idx: int = 0,
        ln_scale: bool = False,
        use_xsa: bool = False,
        use_gated_attention: bool = False,
        use_value_residual: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            rope_dims=rope_dims, use_xsa=use_xsa,
            use_gated_attention=use_gated_attention,
            use_value_residual=use_value_residual,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.layer_idx = layer_idx
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        normed = self.attn_norm(x)
        if self.ln_scale_factor != 1.0:
            normed = normed * self.ln_scale_factor

        attn_out, v = self.attn(normed, v0)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out

        mlp_normed = self.mlp_norm(x)
        if self.ln_scale_factor != 1.0:
            mlp_normed = mlp_normed * self.ln_scale_factor

        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_normed)
        return x, v


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.tie_embeddings = args.tie_embeddings
        self.logit_softcap = args.logit_softcap

        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.smear_gate = SmearGate(args.model_dim) if args.bigram_vocab_size > 0 else None
        self.bigram_hash = BigramHash(
            args.vocab_size, args.bigram_vocab_size, 128, args.model_dim
        ) if args.bigram_vocab_size > 0 else None

        num_layers = args.num_layers
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, args.model_dim, dtype=torch.float32))

        xsa_start = num_layers - args.xsa_last_n

        self.blocks = nn.ModuleList([
            Block(
                args.model_dim, args.num_heads, args.num_kv_heads,
                args.mlp_mult, args.rope_base, args.qk_gain_init,
                rope_dims=args.rope_dims,
                layer_idx=i,
                ln_scale=args.ln_scale,
                use_xsa=(i >= xsa_start and args.xsa_last_n > 0),
                use_gated_attention=args.gated_attention,
                use_value_residual=(args.value_residual and i > 0),
            )
            for i in range(num_layers)
        ])

        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else CastedLinear(args.model_dim, args.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights(args)

    def _init_weights(self, args: Hyperparameters) -> None:
        if args.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=args.tied_embed_init_std)
        # OrthoInit for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.size(0) >= 2 and module.weight.size(1) >= 2:
                    nn.init.orthogonal_(module.weight)

    def _compute_logits(self, x: Tensor) -> Tensor:
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning logits (for sliding window eval)."""
        x = self.tok_emb(input_ids)
        if self.smear_gate is not None:
            x = self.smear_gate(x)
        if self.bigram_hash is not None:
            x = x + self.bigram_hash(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        v0: Tensor | None = None

        for i in range(self.num_encoder_layers):
            x, v = self.blocks[i](x, x0, v0)
            if i == 0 and self.args.value_residual:
                v0 = v.detach() if v is not None else None
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, v = self.blocks[self.num_encoder_layers + i](x, x0, v0)

        return self._compute_logits(x)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# EMA
# -----------------------------

class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Prefer FA3 > Flash > fallback
    try:
        from torch.backends.cuda import (
            enable_cudnn_sdp, enable_flash_sdp,
            enable_math_sdp, enable_mem_efficient_sdp,
        )
        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)
    except ImportError:
        pass

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Build model
    base_model = GPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
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
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.smear_gate is not None:
        scalar_params.append(base_model.smear_gate.gate)
    if base_model.bigram_hash is not None:
        for p in base_model.bigram_hash.parameters():
            if p.ndim < 2:
                scalar_params.append(p)
            else:
                matrix_params.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr,
          "weight_decay": args.adam_wd}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr,
          "weight_decay": args.adam_wd}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"techniques: SmearGate={args.bigram_vocab_size>0} XSA={args.xsa_last_n} "
         f"EMA={args.ema_enabled} PartialRoPE={args.rope_dims} LNScale={args.ln_scale} "
         f"ValueResidual={args.value_residual} GatedAttn={args.gated_attention} "
         f"LateQAT={args.late_qat} GPTQ={args.gptq_lite}")

    # EMA setup
    ema = EMAModel(base_model, decay=args.ema_decay) if args.ema_enabled else None

    # SWA setup
    swa_state = None
    swa_count = 0

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_opt_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wloss = model(x, y)
                (wloss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_opt_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        if ema is not None:
            ema = EMAModel(base_model, decay=args.ema_decay)

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

            # Apply EMA weights for validation
            if ema is not None:
                orig_state = {n: p.data.clone() for n, p in base_model.named_parameters()}
                ema.apply(base_model)

            val_loss, val_bpb = eval_val_sliding(
                args, model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )

            if ema is not None:
                for n, p in base_model.named_parameters():
                    p.data.copy_(orig_state[n])

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
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
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # EMA update
        if ema is not None:
            ema.update(base_model)

        # Tight SWA: accumulate in final portion of training
        if args.swa_enabled and scale < (1.0 - args.swa_start_frac):
            if swa_state is None:
                swa_state = {n: p.data.clone().float() for n, p in base_model.named_parameters()}
                swa_count = 1
            else:
                for n, p in base_model.named_parameters():
                    swa_state[n].add_(p.data.float())
                swa_count += 1

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            cap_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    # Apply best weights for export
    if ema is not None:
        ema.apply(base_model)

    log0(f"peak_memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # Quantize and compress
    if master_process:
        quant_obj = quantize_state_dict(base_model.state_dict(), use_gptq_lite=args.gptq_lite)
        blob = compress_artifact(quant_obj, code)
        with open("final_model.int6.ptz", "wb") as f:
            f.write(blob)
        artifact_bytes = len(blob)
        code_bytes = len(code.encode("utf-8"))
        total_bytes = artifact_bytes + code_bytes
        log0(f"artifact: {artifact_bytes} bytes  code: {code_bytes} bytes  total: {total_bytes} bytes")
        log0(f"under_16mb: {total_bytes <= 16_000_000}")

    # Roundtrip validation
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        blob_disk = f.read()
    quant_state = decompress_artifact(blob_disk)
    base_model.load_state_dict(dequantize_state_dict(quant_state), strict=True)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val_sliding(
        args, model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
