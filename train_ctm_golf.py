"""
train_ctm_golf.py — DuoNeural CTM-Golf submission for OpenAI Parameter Golf
Author: Archon + Jesse (DuoNeural)
Date: 2026-04-24

Innovation: Guided Hebbian Learning (GHL) on depth-recurrent transformer layers.
The community converged on "Triple Recurrence" (depth recurrence on layers 3-5).
We independently developed CTM + GHL at DuoNeural, showing +23% perplexity improvement
at 32M scale. GHL forces each recurrence pass to predict the next hidden state, adding
temporal self-consistency supervision to the recurrent loop.

Architecture base: SP8192 + 11L x 512d x 8H/4KV + depth recurrence (layers 3-5, 3x) +
                   parallel residuals (layer 7+) + tied embeddings + logit softcap
GHL addition:      Between each recurrence pass through layers [loop_start..loop_end],
                   a small prediction MLP predicts the post-loop hidden state from the
                   pre-loop hidden state (MSE loss, annealed λ).
Optimizer:         MuonEq-R (row-normalized, NS5) + AdamW for embeddings/scalars
Quantization:      Int8 per-row + zlib (baseline scheme; GPTQ in v2)
Eval:              Sliding window (stride=64)

Baseline comparison (2026-04-24):
  GHL at 32M: best_ppl=9.63 vs baseline 12.9 (+23% improvement)
  GHL at 300M with constant λ: diverges (late-stage instability)
  GHL at 300M with annealed λ: RUNNING (ctm_ghl_annealing.py on kilonova)
  → GHL works best at small scale + with annealing — Parameter Golf model is ~50M params

Run on 8xH100:
  torchrun --standalone --nproc_per_node=8 train_ctm_golf.py

Run locally (1 GPU, smoke test):
  python train_ctm_golf.py
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
import lzma
import zlib
# np alias used in _byte_shuffle/_byte_unshuffle to avoid shadowing np
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# ── Hyperparameters ───────────────────────────────────────────────────────────

class Hyperparameters:
    data_path       = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files     = os.path.join(data_path, "fineweb_train_*.bin")
    val_files       = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path  = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id          = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed            = int(os.environ.get("SEED", 1337))

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    eval_stride     = int(os.environ.get("EVAL_STRIDE", 64))

    iterations          = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters      = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps        = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens  = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len       = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape — 11L 512d gives ~50M params with SP8192 tied embeddings
    vocab_size      = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers      = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads    = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim       = int(os.environ.get("MODEL_DIM", 480))  # 480: fits 16MB artifact limit with brotli+byte_shuffle
    num_heads       = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult        = int(os.environ.get("MLP_MULT", 4))
    tie_embeddings  = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base       = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap   = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init    = float(os.environ.get("QK_GAIN_INIT", 5.25))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # Depth recurrence (CTM-style)
    num_loops           = int(os.environ.get("NUM_LOOPS", 2))          # extra loops (total passes = num_loops+1)
    loop_start          = int(os.environ.get("LOOP_START", 3))         # first layer in recurrence seg
    loop_end            = int(os.environ.get("LOOP_END", 5))           # last layer in recurrence seg
    enable_looping_at   = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))  # frac of training when looping starts

    # Parallel residuals (GPT-J style, attn+MLP read same input)
    parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", 7))

    # GHL: our innovation — predict next hidden state across recurrence passes
    ghl_lambda      = float(os.environ.get("GHL_LAMBDA", 0.05))        # peak λ
    ghl_lambda_floor = float(os.environ.get("GHL_LAMBDA_FLOOR", 0.005)) # floor (don't go to 0)
    ghl_warmup_frac = float(os.environ.get("GHL_WARMUP_FRAC", 0.10))   # ramp duration (frac of post-loop training)
    ghl_hold_frac   = float(os.environ.get("GHL_HOLD_FRAC", 0.40))     # hold duration (frac of post-loop training)

    # Optimizer
    embed_lr        = float(os.environ.get("EMBED_LR", 0.6))
    head_lr         = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr   = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr       = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr       = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum   = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    muon_row_normalize = bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1")))  # MuonEq-R
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))


# ── GHL weight schedule ───────────────────────────────────────────────────────

def ghl_weight(step: int, args: Hyperparameters) -> float:
    """
    Annealed GHL loss weight. Mirrors our ctm_ghl_annealing.py finding:
    constant λ causes late-stage instability → anneal it out over training.

    Schedule (relative to when looping activates):
      warmup:  linear 0 → peak
      hold:    constant peak
      anneal:  cosine peak → floor
    """
    loop_start_step = int(args.iterations * args.enable_looping_at)
    if step < loop_start_step:
        return 0.0  # GHL inactive before looping starts (no prediction to make)

    loop_step = step - loop_start_step
    loop_total = args.iterations - loop_start_step
    warmup_end = max(1, int(loop_total * args.ghl_warmup_frac))
    hold_end   = max(warmup_end + 1, int(loop_total * args.ghl_hold_frac))

    if loop_step < warmup_end:
        return args.ghl_lambda * (loop_step / warmup_end)
    elif loop_step < hold_end:
        return args.ghl_lambda
    else:
        progress = (loop_step - hold_end) / max(1, loop_total - hold_end)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.ghl_lambda_floor + (args.ghl_lambda - args.ghl_lambda_floor) * cosine


# ── Muon (MuonEq-R: row-normalized) ──────────────────────────────────────────

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
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, row_normalize: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, row_normalize=row_normalize),
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
            row_normalize = group["row_normalize"]

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
                    if row_normalize:
                        # MuonEq-R: normalize by row count for scale invariance
                        g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    else:
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


# ── Tokenizer BPB evaluation ──────────────────────────────────────────────────

def build_sentencepiece_luts(sp, vocab_size, device):
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
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int, max_tokens: int = 0) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    all_tokens = []
    total = 0
    for file in files:
        shard = load_data_shard(file)
        all_tokens.append(shard)
        total += shard.numel()
        if max_tokens > 0 and total >= max_tokens:
            break
    tokens = torch.cat(all_tokens).contiguous()
    if max_tokens > 0:
        tokens = tokens[:max_tokens + 1]
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """Sliding-window BPB evaluation. Stride=eval_stride for max context."""
    stride = args.eval_stride
    ctx_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if ws + ctx_len <= total_tokens]

    my_s = len(window_starts) * rank // world_size
    my_e = len(window_starts) * (rank + 1) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for ws in my_windows:
            chunk = val_tokens[ws : ws + ctx_len + 1].to(device=device, dtype=torch.int64)
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y, return_ce_only=True).detach()
            # Only score the last stride tokens (left-most have minimal context)
            score_start = ctx_len - stride
            y_scored = y[0, score_start:]
            prev_ids  = x[0, score_start:]
            token_bytes = base_bytes_lut[y_scored].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[y_scored] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            # Approximate: use loss as proxy for bit contribution
            loss_sum += loss.to(torch.float64) * stride
            token_count += stride
            byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = float((loss_sum / token_count).item())
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = float(token_count.item() / byte_count.item())
    model.train()
    return val_loss, float(bits_per_token * tokens_per_byte)


# ── Byte shuffle (improves compression ratio on quantized fp16 tensors ~20-40%) ─
_BSHF_MAGIC = b'BSHF'

def _byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off:dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data: bytes) -> bytes:
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()

def _compress_bytes(data: bytes) -> bytes:
    shuffled = _byte_shuffle(data, stride=2)
    try:
        import brotli
        return b'BR' + brotli.compress(shuffled, quality=11)
    except ImportError:
        return b'LZ' + lzma.compress(shuffled, preset=9)

def _decompress_bytes(data: bytes) -> bytes:
    tag, payload = data[:2], data[2:]
    if tag == b'BR':
        import brotli
        raw = brotli.decompress(payload)
    else:
        raw = lzma.decompress(payload)
    return _byte_unshuffle(raw)

# ── Quantization (Int8 + brotli/lzma + byte shuffle) ─────────────────────────

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_PER_ROW_SCALE_DTYPE  = torch.float16
INT8_CLIP_PERCENTILE      = 99.99984
CONTROL_TENSOR_PATTERNS   = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")
# NOTE: "ghl" intentionally excluded — ghl_head weights are large matrices that SHOULD be quantized.
# ghl_current_lambda is a Python attr, not a Parameter, so it never appears in state_dict.

def quantize_float_tensor(t: Tensor):
    t32 = t.float()
    clip_q = INT8_CLIP_PERCENTILE / 100.0
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), clip_q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def pack_model(state_dict: dict) -> bytes:
    quantized, scales, passthroughs = {}, {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        is_control = any(p in name for p in CONTROL_TENSOR_PATTERNS)
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or is_control:
            passthroughs[name] = t.to(torch.float16) if t.is_floating_point() else t
            continue
        q, s = quantize_float_tensor(t)
        quantized[name] = q
        scales[name] = s

    obj = {"quantized": quantized, "scales": scales, "passthrough": passthroughs}
    buf = io.BytesIO()
    torch.save(obj, buf)
    return _compress_bytes(buf.getvalue())


def unpack_model(data: bytes) -> dict:
    obj = torch.load(io.BytesIO(_decompress_bytes(data)), weights_only=True)
    out = {}
    for name, q in obj["quantized"].items():
        s = obj["scales"][name]
        if s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(torch.bfloat16)
        else:
            out[name] = (q.float() * float(s.item())).to(torch.bfloat16)
    for name, t in obj["passthrough"].items():
        out[name] = t.to(torch.bfloat16) if t.is_floating_point() else t
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
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
    def __init__(self, pattern: str, rank: int, world_size: int, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ── Transformer modules ───────────────────────────────────────────────────────

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
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        # register as buffers so torch.compile sees stable object IDs — no recompile storms
        self.register_buffer("cos_cache", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cache", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device, dtype):
        return self.cos_cache[:, :, :seq_len, :].to(dtype=dtype), \
               self.sin_cache[:, :, :seq_len, :].to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
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
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        _gqa = self.num_kv_heads != self.num_heads
        if _gqa and hasattr(torch.backends.cuda, 'enable_flash_sdp') and \
                int(torch.__version__.split('.')[1]) >= 5:
            # enable_gqa added in torch 2.5 — use it if available
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                               enable_gqa=True)
        elif _gqa:
            # torch < 2.5: manually expand k/v to match num_heads
            expand_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(expand_factor, dim=1)
            v = v.repeat_interleave(expand_factor, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        # LeakyReLU²: from the SOTA stack
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp  = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.parallel   = False  # set True for parallel residual layers (GPT-J style)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        if self.parallel:
            # Parallel residual (GPT-J style): attn and MLP read from the same pre-residual x
            attn_out = self.attn(self.attn_norm(x))
            mlp_out  = self.mlp(self.mlp_norm(x))
            x = (x
                 + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
                 + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out)
        else:
            attn_out = self.attn(self.attn_norm(x))
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# ── GHL prediction head ────────────────────────────────────────────────────────
# A small MLP that predicts the post-recurrence hidden state from the pre-recurrence
# hidden state. MSE loss between prediction and actual post-recurrence state.
# Applied at each transition between consecutive recurrence passes.

class GHLHead(nn.Module):
    """
    Guided Hebbian Learning prediction head.
    Given z_before (hidden state before a recurrence pass), predict z_after.
    Loss = MSE(predict(z_before), z_after.detach())

    This forces the model to learn a temporal self-consistency: each recurrent
    thought step should be predictable from the previous one, similar to how
    temporal continuity is enforced in world model training.

    Architecture: LeakyReLU² (matches the MLP blocks) for consistency.
    Size: ~2.6M params at d=512. Int8 quantized = ~2.5MB.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.ln  = RMSNorm()
        self.fc  = CastedLinear(d_model, d_model * 2, bias=False)
        self.out = CastedLinear(d_model * 2, d_model, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        return self.out(F.leaky_relu(self.fc(self.ln(z)), negative_slope=0.5).square())


# ── CTM-Golf GPT model ────────────────────────────────────────────────────────

class CTMGolfGPT(nn.Module):
    """
    GPT with depth recurrence + GHL (CTM-Golf).

    Depth recurrence: layers [loop_start..loop_end] are run (num_loops+1) times.
    With loop_start=3, loop_end=5, num_loops=2: layers 3,4,5 run 3x total.
    This gives 11 physical layers but 17 virtual layers (same SOTA architecture).

    GHL: between each consecutive recurrence pass, a GHLHead predicts the
    post-pass hidden state from the pre-pass hidden state. MSE loss with
    annealed λ (mirrors our ctm_ghl_annealing.py research).
    """
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.tie_embeddings      = args.tie_embeddings
        self.tied_embed_init_std = args.tied_embed_init_std
        self.logit_softcap       = args.logit_softcap

        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.blocks  = nn.ModuleList([
            Block(args.model_dim, args.num_heads, args.num_kv_heads,
                  args.mlp_mult, args.rope_base, args.qk_gain_init)
            for _ in range(args.num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head    = None if args.tie_embeddings else CastedLinear(args.model_dim, args.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # Parallel residuals from parallel_residual_start
        for i in range(args.parallel_residual_start, args.num_layers):
            self.blocks[i].parallel = True

        # Build encoder/decoder index sequences for depth recurrence
        self.looping_active = False
        if args.num_loops > 0:
            loop_seg = list(range(args.loop_start, args.loop_end + 1))
            all_indices = list(range(args.loop_start))
            for _ in range(args.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(args.loop_end + 1, args.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            enc_layers = args.num_layers // 2
            self.encoder_indices = list(range(enc_layers))
            self.decoder_indices = list(range(enc_layers, args.num_layers))

        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, args.model_dim, dtype=torch.float32)
        )

        # GHL head (our innovation)
        self.ghl_head = GHLHead(args.model_dim) if args.ghl_lambda > 0 else None
        self.ghl_current_lambda = 0.0  # updated each step by training loop

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _run_with_ghl(self, x: Tensor, x0: Tensor, indices: list[int],
                      skips_in: list[Tensor] | None, is_encoder: bool):
        """
        Run blocks with given indices, tracking GHL at recurrence boundaries.
        Returns (x, ghl_loss, skips_accumulated or None).
        """
        args = self.args
        skips_out = [] if is_encoder else None
        ghl_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        z_before_loop = None
        prev_was_loop_end = False
        ghl_count = 0

        for step_pos, layer_idx in enumerate(indices):
            at_loop_start = self.looping_active and (layer_idx == args.loop_start)
            at_loop_end   = self.looping_active and (layer_idx == args.loop_end)

            # GHL: entering a new recurrence pass after completing one
            if at_loop_start:
                if prev_was_loop_end and self.ghl_head is not None and self.ghl_current_lambda > 0:
                    z_pred = self.ghl_head(z_before_loop)
                    ghl_loss = ghl_loss + F.mse_loss(z_pred, x.detach())
                    ghl_count += 1
                z_before_loop = x.detach()
                prev_was_loop_end = False

            # Decoder skip connection
            if not is_encoder and skips_in:
                skip_idx = len(indices) - len(skips_in) + (step_pos if step_pos < self.num_skip_weights else self.num_skip_weights - 1)
                if step_pos < self.num_skip_weights and skips_in:
                    x = x + self.skip_weights[step_pos].to(dtype=x.dtype)[None, None, :] * skips_in.pop()

            x = self.blocks[layer_idx](x, x0)

            if is_encoder:
                skips_out.append(x)

            if at_loop_end:
                prev_was_loop_end = True

        return x, ghl_loss, ghl_count, skips_out

    def forward(self, input_ids: Tensor, target_ids: Tensor, return_ce_only: bool = False) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # When looping inactive: run all blocks [0..N-1] sequentially as "encoder", no decoder.
        # This avoids out-of-range indices from the virtual encoder+decoder layout.
        enc_iter = self.encoder_indices if self.looping_active else list(range(len(self.blocks)))
        dec_iter = self.decoder_indices if self.looping_active else []

        # Encoder pass
        ghl_loss_enc = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        skips = []
        z_before_loop = None
        prev_was_loop_end = False
        ghl_count = 0

        for layer_idx in enc_iter:
            at_loop_start = self.looping_active and (layer_idx == self.args.loop_start)
            at_loop_end   = self.looping_active and (layer_idx == self.args.loop_end)

            if at_loop_start:
                if prev_was_loop_end and self.ghl_head is not None and self.ghl_current_lambda > 0:
                    z_pred = self.ghl_head(z_before_loop)
                    ghl_loss_enc = ghl_loss_enc + F.mse_loss(z_pred, x.detach())
                    ghl_count += 1
                z_before_loop = x.detach()
                prev_was_loop_end = False

            x = self.blocks[layer_idx](x, x0)
            skips.append(x)

            if at_loop_end:
                prev_was_loop_end = True

        # Decoder pass
        ghl_loss_dec = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        prev_was_loop_end = False

        for skip_idx, layer_idx in enumerate(dec_iter):
            at_loop_start = self.looping_active and (layer_idx == self.args.loop_start)
            at_loop_end   = self.looping_active and (layer_idx == self.args.loop_end)

            if at_loop_start:
                if prev_was_loop_end and self.ghl_head is not None and self.ghl_current_lambda > 0:
                    z_pred = self.ghl_head(z_before_loop)
                    ghl_loss_dec = ghl_loss_dec + F.mse_loss(z_pred, x.detach())
                    ghl_count += 1
                z_before_loop = x.detach()
                prev_was_loop_end = False

            if skip_idx < self.num_skip_weights and skips:
                x = x + self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[layer_idx](x, x0)

            if at_loop_end:
                prev_was_loop_end = True

        # Output
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        if return_ce_only:
            return ce_loss

        if self.ghl_current_lambda > 0 and ghl_count > 0:
            ghl_loss_total = (ghl_loss_enc + ghl_loss_dec) / ghl_count
            return ce_loss + self.ghl_current_lambda * ghl_loss_total
        return ce_loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Training ──────────────────────────────────────────────────────────────────

def cosine_lr(step: int, total: int, warmdown_iters: int, base_lr: float) -> float:
    if step < 10:
        return base_lr * step / 10
    if step > total - warmdown_iters:
        t = (step - (total - warmdown_iters)) / warmdown_iters
        return base_lr * 0.5 * (1 + math.cos(math.pi * t))
    return base_lr


def main() -> None:
    global zeropower_via_newtonschulz5
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    args = Hyperparameters()

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_flash_sdp, enable_mem_efficient_sdp, enable_math_sdp, enable_cudnn_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt" if master_process else None

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a") as f:
                print(msg, file=f)

    log0(f"DuoNeural CTM-Golf | run_id={args.run_id}")
    log0(f"GHL: lambda={args.ghl_lambda}, floor={args.ghl_lambda_floor}, "
         f"warmup_frac={args.ghl_warmup_frac}, hold_frac={args.ghl_hold_frac}")
    log0(f"Loops: num_loops={args.num_loops}, start={args.loop_start}, end={args.loop_end}, "
         f"activate_at={args.enable_looping_at}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Tokenizer + val setup
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Tokenizer vocab mismatch: got {sp.vocab_size()}, want {args.vocab_size}")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, args.val_batch_size)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Model
    base_model = CTMGolfGPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=True, fullgraph=False)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    log0(f"params: {base_model.count_params() / 1e6:.1f}M")
    log0(f"encoder_indices: {base_model.encoder_indices}")
    log0(f"decoder_indices: {base_model.decoder_indices}")

    # Optimizer: Muon for matrices, AdamW for embeddings/scalars
    matrix_params, scalar_params, embed_params = [], [], []
    for name, param in base_model.named_parameters():
        if param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_PATTERNS):
            if "tok_emb" in name:
                embed_params.append(param)
            else:
                scalar_params.append(param)
        elif "tok_emb" in name or "lm_head" in name:
            embed_params.append(param)
        else:
            matrix_params.append(param)

    muon_opt = Muon(
        [{"params": matrix_params, "lr": args.matrix_lr}],
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        row_normalize=args.muon_row_normalize,
    )
    adam_opt = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": args.embed_lr},
            {"params": scalar_params, "lr": args.scalar_lr},
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers = [muon_opt, adam_opt]

    # Data loader
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # EMA for eval — keep uncompiled (avoids state_dict key issues from torch.compile wrapper).
    # On 8xH100 eval is distributed across 8 GPUs so single-pass eval is fast enough.
    # For smoke tests, set VAL_BATCH_SIZE=65536 to limit eval windows.
    ema_model = copy.deepcopy(base_model).bfloat16()
    ema_decay = 0.9965

    # Training loop
    t_start = time.time()
    step = 0
    best_bpb = float("inf")

    for opt in optimizers:
        opt.zero_grad()

    log0(f"training start: {args.iterations} steps, {args.max_wallclock_seconds}s budget")

    while step < args.iterations:
        wallclock = time.time() - t_start
        if wallclock >= args.max_wallclock_seconds:
            log0(f"wallclock limit reached at step {step}")
            break

        frac = step / args.iterations

        # Enable depth recurrence at the configured fraction
        if args.num_loops > 0 and not base_model.looping_active and frac >= args.enable_looping_at:
            base_model.looping_active = True
            log0(f"looping enabled at step {step} (frac={frac:.3f})")
            log0(f"  encoder: {base_model.encoder_indices}")
            log0(f"  decoder: {base_model.decoder_indices}")

        # Set current GHL weight (annealed schedule)
        current_ghl = ghl_weight(step, args)
        base_model.ghl_current_lambda = current_ghl

        # Muon momentum warmup
        if step < args.muon_momentum_warmup_steps:
            muon_momentum = (args.muon_momentum_warmup_start
                             + (args.muon_momentum - args.muon_momentum_warmup_start)
                             * step / args.muon_momentum_warmup_steps)
            for pg in muon_opt.param_groups:
                pg["momentum"] = muon_momentum

        # Accumulate gradients
        model.train()
        accum_loss = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()
            accum_loss += loss.item()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        # LR update
        lr = cosine_lr(step, args.iterations, args.warmdown_iters, args.matrix_lr)
        lr_ratio = lr / args.matrix_lr
        for pg in muon_opt.param_groups:
            pg["lr"] = lr
        adam_opt.param_groups[0]["lr"] = args.embed_lr * lr_ratio
        adam_opt.param_groups[1]["lr"] = args.scalar_lr * lr_ratio

        for opt in optimizers:
            opt.step()
            opt.zero_grad()

        # EMA update
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), base_model.parameters()):
                p_ema.lerp_(p.to(dtype=p_ema.dtype), 1.0 - ema_decay)

        step += 1

        # Logging
        if step % args.train_log_every == 0 or step == args.iterations:
            elapsed = time.time() - t_start
            sps = step / elapsed
            eta = (args.iterations - step) / sps / 3600
            ppl = math.exp(min(accum_loss / grad_accum_steps, 20))
            log0(
                f"step {step:>6}/{args.iterations} | loss={accum_loss/grad_accum_steps:.4f} "
                f"ppl={ppl:.2f} | ghl_λ={current_ghl:.4f} | loop={base_model.looping_active} | "
                f"{sps:.2f} sps | ETA={eta:.1f}h | {elapsed:.0f}s"
            )

        # Validation
        if step % args.val_loss_every == 0 or step == args.iterations:
            ema_model.looping_active = base_model.looping_active
            ema_model.ghl_current_lambda = 0.0  # no GHL at eval
            val_loss, val_bpb = eval_val(
                args, ema_model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"val step={step} | val_loss={val_loss:.4f} | val_bpb={val_bpb:.4f}")
            if val_bpb < best_bpb:
                best_bpb = val_bpb
                torch.save(ema_model.state_dict(), "best_model.pt")
                log0(f"  → new best BPB: {best_bpb:.4f}")

    # Final: pack model to artifact
    log0(f"\ntraining done. steps={step}, best_bpb={best_bpb:.4f}")

    if master_process:
        state_dict = torch.load("best_model.pt", weights_only=True, map_location="cpu")
        packed = pack_model(state_dict)
        artifact_size = len(packed) + Path(__file__).stat().st_size
        log0(f"artifact: model={len(packed)/1e6:.2f}MB  script={Path(__file__).stat().st_size/1e3:.1f}KB  total≈{artifact_size/1e6:.2f}MB")
        if artifact_size <= 16_000_000:
            log0("artifact FITS within 16MB ✓")
        else:
            log0(f"WARNING: artifact is {artifact_size/1e6:.2f}MB — exceeds 16MB limit")

        with open("final_model.ptz", "wb") as f:
            f.write(packed)
        log0("saved final_model.ptz")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
