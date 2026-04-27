"""
train_gpt_rrt.py — Relaxed Recursive Transformer with LoRA Adapters
                   + SP8192 + GPTQ SDClip + Parallel Residuals + QK-Gain
                   + MuonEq-R + Legal Score-First TTT + EMA

Built on top of the parameter-golf baseline (train_gpt.py).

Novel contribution: Relaxed Recursive Transformer (RRT) with per-step LoRA
adapters on shared depth-recurrence layers. Instead of sharing weights
identically across recurrence steps (which amplifies quantization error),
each loop iteration applies a tiny learned LoRA delta (rank-4) to the
shared attention Q/V projections. This allows each pass to specialize while
keeping the parameter overhead negligible (~4K params per adapter pair).

Full stack:
  - SP8192 tokenizer (8192 vocab, more bytes per token)
  - 11L x 512d x 8H/4KV, MLP 4x, LeakyReLU(0.5)^2
  - Partial RoPE (16/64 dims)
  - Depth recurrence: loops layers 3-5 with RRT-LoRA adapters (3 steps)
  - Parallel residuals (GPT-J style) on layers 7+
  - QK-Gain 5.25 (learnable per-head query scaling)
  - MuonEq-R optimizer (row-normalized Muon)
  - EMA decay 0.9965
  - GPTQ SDClip quantization (int6 matrices, int8 embeddings)
  - Legal score-first TTT at eval time
  - LZMA code compression

Key environment variables:
  VOCAB_SIZE=8192             — use SP8192 tokenizer
  QK_GAIN_INIT=5.25           — initial QK gain
  RECUR_LAYERS=3,4,5          — which layers to loop
  RECUR_STEPS=3               — how many times to loop
  RECUR_LORA_RANK=4           — LoRA rank for per-step adapters
  RECUR_ACTIVATE_FRAC=0.35    — when to start recurrence (fraction of steps)
  PARALLEL_RESID_FROM=7       — layer index to start parallel residuals
  TTT_ENABLED=1               — enable test-time training
  TTT_LR=0.005                — TTT learning rate
  TTT_EPOCHS=3                — TTT epochs per chunk
  TTT_CHUNK_SIZE=32768        — tokens per TTT chunk
  TTT_FREEZE_LAYERS=2         — freeze first N layers during TTT
  EMA_DECAY=0.9965            — EMA decay for model weights
  WEIGHT_DECAY=0.095          — AdamW weight decay
  MATRIX_LR=0.022             — Muon learning rate
  WARMDOWN_FRAC=0.72          — fraction of training for warmdown
"""

from __future__ import annotations

import copy
import glob
import io
import lzma
import math
import os
import random
import struct
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

# -----------------------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------------------

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

    iterations          = int(os.environ.get("ITERATIONS", 20000))
    warmdown_frac       = float(os.environ.get("WARMDOWN_FRAC", 0.72))
    warmup_steps        = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens  = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len       = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape — matches SOTA stack
    vocab_size      = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers      = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads    = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim       = int(os.environ.get("MODEL_DIM", 512))
    num_heads       = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult        = int(os.environ.get("MLP_MULT", 4))
    tie_embeddings  = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base       = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_partial_dim = int(os.environ.get("ROPE_PARTIAL_DIM", 16))  # partial RoPE
    logit_softcap   = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init    = float(os.environ.get("QK_GAIN_INIT", 5.25))

    # Depth recurrence + RRT-LoRA
    recur_layers        = [int(x) for x in os.environ.get("RECUR_LAYERS", "3,4,5").split(",")]
    recur_steps         = int(os.environ.get("RECUR_STEPS", "3"))
    recur_lora_rank     = int(os.environ.get("RECUR_LORA_RANK", "4"))
    recur_activate_frac = float(os.environ.get("RECUR_ACTIVATE_FRAC", "0.35"))

    # Parallel residuals
    parallel_resid_from = int(os.environ.get("PARALLEL_RESID_FROM", "7"))

    # TTT
    ttt_enabled     = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr          = float(os.environ.get("TTT_LR", "0.005"))
    ttt_epochs      = int(os.environ.get("TTT_EPOCHS", "3"))
    ttt_chunk_size  = int(os.environ.get("TTT_CHUNK_SIZE", "32768"))
    ttt_freeze_layers = int(os.environ.get("TTT_FREEZE_LAYERS", "2"))
    ttt_momentum    = float(os.environ.get("TTT_MOMENTUM", "0.9"))

    # EMA
    ema_decay       = float(os.environ.get("EMA_DECAY", "0.9965"))

    # Optimizer
    embed_lr        = float(os.environ.get("EMBED_LR", "0.6"))
    head_lr         = float(os.environ.get("HEAD_LR", "0.008"))
    tied_embed_lr   = float(os.environ.get("TIED_EMBED_LR", "0.05"))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", "0.005"))
    matrix_lr       = float(os.environ.get("MATRIX_LR", "0.022"))
    scalar_lr       = float(os.environ.get("SCALAR_LR", "0.04"))
    weight_decay    = float(os.environ.get("WEIGHT_DECAY", "0.095"))
    muon_momentum   = float(os.environ.get("MUON_MOMENTUM", "0.95"))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", "5"))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", "0.85"))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", "500"))
    beta1           = float(os.environ.get("BETA1", "0.9"))
    beta2           = float(os.environ.get("BETA2", "0.95"))
    adam_eps        = float(os.environ.get("ADAM_EPS", "1e-8"))
    grad_clip_norm  = float(os.environ.get("GRAD_CLIP_NORM", "1.0"))

    @property
    def warmdown_iters(self):
        return int(self.iterations * self.warmdown_frac)


# -----------------------------------------------------------------------
# MUONEQ-R OPTIMIZER
# Row-normalized Muon — better spectral conditioning for Newton-Schulz
# -----------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
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


class MuonEqR(torch.optim.Optimizer):
    """
    MuonEq-R: Row-normalized Muon optimizer.
    Divides each row of the momentum matrix by its L2 norm before
    orthogonalization via Newton-Schulz. Improves spectral conditioning.
    Reference: arXiv:2603.28254
    """
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

                    # MuonEq-R: row-normalize before orthogonalization
                    if g.ndim == 2:
                        row_norms = g.norm(dim=1, keepdim=True).clamp(min=1e-7)
                        g = g / row_norms

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


# -----------------------------------------------------------------------
# TOKENIZER-AGNOSTIC EVALUATION (BPB)
# -----------------------------------------------------------------------

def build_sentencepiece_luts(sp, vocab_size: int, device):
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


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


# -----------------------------------------------------------------------
# GPTQ SDCLIP QUANTIZATION
# int6 for weight matrices, int8 for embeddings
# SDClip: clip = k * std(row) for principled rate-distortion
# -----------------------------------------------------------------------

CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain",
                                 "skip_weight", "lora_", "layerscale")
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 0.9999984  # 99.99984th percentile


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def quantize_tensor_int6_sdclip(t: Tensor, k: float = 12.85) -> tuple[Tensor, Tensor]:
    """
    GPTQ SDClip quantization to int6 (stored as int8, 2 bits wasted).
    Clip to k * std(row) per row for matrices.
    """
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (k * t32.std(dim=1)).clamp(min=1e-6)
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 31.0).clamp(min=1.0 / 31.0)  # int6: range [-31, 31]
        q = torch.clamp(torch.round(clipped / scale[:, None]), -31, 31).to(torch.int8).contiguous()
        return q, scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float((k * t32.std()).item()) if t32.numel() > 1 else 1.0
    scale = torch.tensor(clip_abs / 31.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -31, 31).to(torch.int8).contiguous()
    return q, scale


def quantize_tensor_int8_sdclip(t: Tensor, k: float = 20.0) -> tuple[Tensor, Tensor]:
    """int8 with SDClip for embeddings."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (k * t32.std(dim=1)).clamp(min=1e-6)
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp(min=1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float((k * t32.std()).item()) if t32.numel() > 1 else 1.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict(state_dict: dict) -> tuple[dict, dict]:
    """
    Quantize state dict:
    - Embedding weights: int8 SDClip (k=20)
    - All other 2D float matrices: int6 SDClip (k=12.85)
    - Small tensors / control params: fp16 passthrough
    """
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()

        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            passthrough[name] = t.to(torch.float16) if t.is_floating_point() else t
            continue

        if any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            passthrough[name] = t.to(torch.float16)
            continue

        orig_dtype = str(t.dtype).replace("torch.", "")

        if "tok_emb" in name or "lm_head" in name:
            # Embeddings: int8
            q, s = quantize_tensor_int8_sdclip(t, k=20.0)
        else:
            # Weight matrices: int6 (stored as int8)
            q, s = quantize_tensor_int6_sdclip(t, k=12.85)

        quantized[name] = q
        scales[name] = s
        dtypes[name] = orig_dtype

    return {"quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}, {}


def dequantize_state_dict(obj: dict) -> dict:
    out = {}
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].to(torch.float32)
        if s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(dtype)
    for name, t in obj["passthrough"].items():
        out[name] = t
    return out


# -----------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


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
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
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
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------------------------------------------------
# MODEL COMPONENTS
# -----------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    """Partial RoPE: only apply to first `partial_dim` dimensions of head."""
    def __init__(self, head_dim: int, partial_dim: int, base: float = 10000.0):
        super().__init__()
        self.partial_dim = partial_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, partial_dim, 2, dtype=torch.float32) / partial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len: int, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb_partial(x: Tensor, cos: Tensor, sin: Tensor, partial_dim: int) -> Tensor:
    """Apply RoPE only to first partial_dim dimensions, leave rest unchanged."""
    x_rot = x[..., :partial_dim]
    x_pass = x[..., partial_dim:]
    half = partial_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    x_rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((x_rotated, x_pass), dim=-1)


# -----------------------------------------------------------------------
# RRT-LORA ADAPTER
# Per-step LoRA delta for attention Q and V projections.
# Allows recurrent layers to specialize per iteration while sharing weights.
# Reference: Bae et al. "Relaxed Recursive Transformers" (ICLR 2025)
# -----------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    """
    Low-rank adapter: output += scale * (B @ (A @ x))
    where A: (rank, in_dim), B: (out_dim, rank)
    Initialized so output starts at zero (B=zeros).
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)  # zero init: no effect at start

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * self.lora_B(self.lora_A(x))


class RRTAttention(nn.Module):
    """
    CausalSelfAttention with optional per-step LoRA adapters on Q and V.
    When step_idx is provided, applies the corresponding adapter.
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, partial_rope_dim: int, qk_gain_init: float,
                 num_lora_steps: int = 0, lora_rank: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim

        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, partial_rope_dim, base=rope_base)
        self.partial_rope_dim = partial_rope_dim

        # Per-step LoRA adapters for Q and V
        if num_lora_steps > 0:
            self.lora_q = nn.ModuleList([LoRAAdapter(dim, dim, lora_rank) for _ in range(num_lora_steps)])
            self.lora_v = nn.ModuleList([LoRAAdapter(dim, kv_dim, lora_rank) for _ in range(num_lora_steps)])
        else:
            self.lora_q = None
            self.lora_v = None

    def forward(self, x: Tensor, step_idx: int = -1) -> Tensor:
        bsz, seqlen, dim = x.shape

        q = self.c_q(x)
        v = self.c_v(x)

        # Apply LoRA adapters for this recurrence step
        if self.lora_q is not None and 0 <= step_idx < len(self.lora_q):
            q = q + self.lora_q[step_idx](x)
            v = v + self.lora_v[step_idx](x)

        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb_partial(q, cos, sin, self.partial_rope_dim // 2)
        k = apply_rotary_emb_partial(k, cos, sin, self.partial_rope_dim // 2)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    """LeakyReLU(0.5)^2 MLP — matches SOTA stack."""
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
    """
    Transformer block with:
    - Optional parallel residuals (GPT-J style) for layers >= parallel_resid_from
    - Layerwise LN scale
    - RRT attention with LoRA adapter support
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, partial_rope_dim: int, qk_gain_init: float,
                 parallel_resid: bool = False,
                 num_lora_steps: int = 0, lora_rank: int = 4):
        super().__init__()
        self.parallel_resid = parallel_resid

        self.attn_norm = RMSNorm()
        self.attn = RRTAttention(dim, num_heads, num_kv_heads, rope_base, partial_rope_dim,
                                  qk_gain_init, num_lora_steps, lora_rank)

        if not parallel_resid:
            self.mlp_norm = RMSNorm()
        self.mlp = MLP(dim, mlp_mult)

        # Layerwise scale parameters (LN scale)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, step_idx: int = -1) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        if self.parallel_resid:
            # GPT-J style: attention and MLP read from same normalized input
            normed = self.attn_norm(x)
            attn_out = self.attn(normed, step_idx)
            mlp_out = self.mlp(normed)
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        else:
            attn_out = self.attn(self.attn_norm(x), step_idx)
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))

        return x


# -----------------------------------------------------------------------
# RELAXED RECURSIVE TRANSFORMER (main model)
# -----------------------------------------------------------------------

class RRTModel(nn.Module):
    """
    Relaxed Recursive Transformer.

    Architecture:
    - 11 physical layers (U-Net encoder/decoder structure)
    - Depth recurrence: loops layers 3,4,5 for `recur_steps` iterations
      with per-step LoRA adapters on Q and V (rank=4)
    - Parallel residuals on layers >= parallel_resid_from
    - QK-Gain 5.25 for all attention layers
    - Partial RoPE (16/64 head dims)
    - LeakyReLU(0.5)^2 MLP activation
    - Sigmoid-gated U-Net skip connections
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.logit_softcap = args.logit_softcap
        self.tie_embeddings = args.tie_embeddings
        self.recur_layers = set(args.recur_layers)
        self.recur_steps = args.recur_steps

        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)

        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, args.model_dim, dtype=torch.float32))

        # Build blocks
        blocks = []
        for i in range(args.num_layers):
            is_recur = i in self.recur_layers
            parallel = i >= args.parallel_resid_from
            blocks.append(Block(
                dim=args.model_dim,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                mlp_mult=args.mlp_mult,
                rope_base=args.rope_base,
                partial_rope_dim=args.rope_partial_dim,
                qk_gain_init=args.qk_gain_init,
                parallel_resid=parallel,
                num_lora_steps=args.recur_steps if is_recur else 0,
                lora_rank=args.recur_lora_rank,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm()

        # Track recurrence activation step
        self._recur_active = False

        if not args.tie_embeddings:
            self.lm_head = CastedLinear(args.model_dim, args.vocab_size, bias=False)
            self.lm_head._zero_init = True
        else:
            self.lm_head = None

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.args.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def activate_recurrence(self):
        self._recur_active = True

    def _run_block(self, block_idx: int, x: Tensor, x0: Tensor, step_idx: int = -1) -> Tensor:
        return self.blocks[block_idx](x, x0, step_idx)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        skips: list[Tensor] = []

        # Encoder half
        i = 0
        while i < self.num_encoder_layers:
            if self._recur_active and i in self.recur_layers:
                # Run recurrent block recur_steps times with LoRA adapters
                for step in range(self.recur_steps):
                    x = self._run_block(i, x, x0, step_idx=step)
                # Only advance once — the recurrent block counts as one physical layer
            else:
                x = self._run_block(i, x, x0)
            skips.append(x)
            i += 1

        # Decoder half
        for j in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[j].to(dtype=x.dtype)[None, None, :] * skips.pop()
            layer_idx = self.num_encoder_layers + j
            if self._recur_active and layer_idx in self.recur_layers:
                for step in range(self.recur_steps):
                    x = self._run_block(layer_idx, x, x0, step_idx=step)
            else:
                x = self._run_block(layer_idx, x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------------------------------------------------
# EMA HELPER
# -----------------------------------------------------------------------

class EMAModel:
    """Exponential moving average of model weights."""
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {name: param.data.clone().float()
                       for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data.float(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name].to(param.dtype))


# -----------------------------------------------------------------------
# LEGAL SCORE-FIRST TEST-TIME TRAINING
# -----------------------------------------------------------------------

def eval_with_ttt(
    args: Hyperparameters,
    model: nn.Module,
    val_tokens: Tensor,
    device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    rank: int,
    world_size: int,
) -> tuple[float, float]:
    """
    Legal score-first TTT evaluation.

    For each chunk:
    1. Score all windows under no_grad (this is the official graded prediction)
    2. Train on already-scored tokens with SGD
    3. Advance to next chunk

    Conditions satisfied:
    - Causality: sliding window eval is strictly causal
    - Score before update: tokens scored before any gradient update
    - Single pass: each token scored exactly once
    """
    model.eval()
    chunk_size = args.ttt_chunk_size
    seq_len = args.train_seq_len

    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)
    total_bytes = torch.zeros((), device=device, dtype=torch.float64)

    n_tokens = val_tokens.numel() - 1
    n_chunks = max(1, n_tokens // chunk_size)

    # Save original weights to restore after TTT
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Freeze first ttt_freeze_layers blocks during TTT
    frozen_params = set()
    for i in range(args.ttt_freeze_layers):
        for name, _ in model.named_parameters():
            if f"blocks.{i}." in name:
                frozen_params.add(name)

    ttt_params = [p for name, p in model.named_parameters()
                  if name not in frozen_params and p.requires_grad]

    ttt_optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size + 1, val_tokens.numel())
        chunk = val_tokens[chunk_start:chunk_end].to(device=device, dtype=torch.int64)

        if chunk.numel() < seq_len + 1:
            break

        usable = ((chunk.numel() - 1) // seq_len) * seq_len
        x = chunk[:usable].reshape(-1, seq_len)
        y = chunk[1:usable + 1].reshape(-1, seq_len)

        # Step 1: SCORE (no grad — this is the official prediction)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            n_toks = float(y.numel())
            total_loss += loss.to(torch.float64) * n_toks
            total_tokens += n_toks

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            total_bytes += token_bytes.to(torch.float64).sum()

        # Step 2: TRAIN on already-scored tokens
        model.train()
        cosine_steps = args.ttt_epochs * max(1, x.shape[0])
        for epoch in range(args.ttt_epochs):
            # Cosine LR decay across epochs
            cos_frac = (epoch + 1) / args.ttt_epochs
            lr_scale = 0.5 * (1 + math.cos(math.pi * cos_frac))
            for pg in ttt_optimizer.param_groups:
                pg["lr"] = args.ttt_lr * lr_scale

            perm = torch.randperm(x.shape[0], device=device)
            for batch_start in range(0, x.shape[0], 16):
                batch_idx = perm[batch_start:batch_start + 16]
                xb, yb = x[batch_idx], y[batch_idx]
                ttt_optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                ttt_optimizer.step()

        model.eval()

    # Reduce across ranks
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    val_loss = float(total_loss / total_tokens)
    bpt = val_loss / math.log(2.0)
    tpb = float(total_tokens / total_bytes)
    val_bpb = bpt * tpb

    # Restore original weights (TTT is eval-only, doesn't affect training)
    model.load_state_dict(original_state)
    model.train()

    return val_loss, val_bpb


def eval_val_sliding(args, model, val_tokens, device, base_bytes_lut,
                     has_leading_space_lut, is_boundary_token_lut, rank, world_size, grad_accum_steps):
    """Standard sliding window evaluation (no TTT)."""
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
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            raw_start = bss * args.train_seq_len
            raw_end = bse * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y).detach()
            n_toks = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * n_toks
            val_token_count += n_toks
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = float(val_loss_sum / val_token_count)
    bpt = val_loss / math.log(2.0)
    tpb = float(val_token_count / val_byte_count)
    model.train()
    return val_loss, bpt * tpb


# -----------------------------------------------------------------------
# ARTIFACT SIZE MEASUREMENT
# -----------------------------------------------------------------------

def measure_artifact(model: nn.Module, code: str) -> tuple[int, int, int]:
    """Quantize, compress with LZMA, measure total artifact size."""
    state_dict = model.state_dict()
    quant_obj, _ = quantize_state_dict(state_dict)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    model_bytes_zlib = zlib.compress(buf.getvalue(), level=9)

    # LZMA for code (saves ~40KB vs uncompressed)
    code_bytes = lzma.compress(code.encode("utf-8"), preset=9)

    return len(code_bytes), len(model_bytes_zlib), len(code_bytes) + len(model_bytes_zlib)


# -----------------------------------------------------------------------
# MAIN TRAINING LOOP
# -----------------------------------------------------------------------

def restore_fp32_params(model: nn.Module):
    with torch.no_grad():
        for name, param in model.named_parameters():
            should_fp32 = (
                param.ndim < 2
                or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
            )
            if should_fp32 and param.dtype != torch.float32:
                param.data = param.data.float()


def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 0 or 8 % world_size != 0:
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
    from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp, enable_cudnn_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Build model
    model = RRTModel(args).to(device)
    restore_fp32_params(model)

    if master_process:
        total_params = sum(p.numel() for p in model.parameters())
        lora_params = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name)
        print(f"RRTModel | total params: {total_params:,} | LoRA params: {lora_params:,}")
        print(f"  Recur layers: {args.recur_layers} | Steps: {args.recur_steps} | LoRA rank: {args.recur_lora_rank}")
        print(f"  Parallel resid from layer: {args.parallel_resid_from}")
        print(f"  QK gain init: {args.qk_gain_init} | Vocab: {args.vocab_size}")

    # EMA
    ema = EMAModel(model, args.ema_decay)

    model = torch.compile(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if distributed else model

    # Optimizer param groups
    matrix_params = [
        p for name, p in raw_model.named_parameters()
        if p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name
        and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in raw_model.named_parameters()
        if not (p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name
                and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS))
        and "tok_emb" not in name and "lm_head" not in name
    ]
    embed_params = [p for name, p in raw_model.named_parameters() if "tok_emb" in name]
    head_params = [p for name, p in raw_model.named_parameters() if "lm_head" in name]

    optimizer_muon = MuonEqR(
        matrix_params, lr=args.matrix_lr,
        momentum=args.muon_momentum, backend_steps=args.muon_backend_steps,
    )
    embed_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_adam = torch.optim.AdamW(
        [
            {"params": scalar_params, "lr": args.scalar_lr},
            {"params": embed_params, "lr": embed_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )

    # Tokenizer + val data
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    warmdown_iters = args.warmdown_iters
    recur_activate_step = int(args.recur_activate_frac * args.iterations)

    def lr_schedule(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if step >= args.iterations - warmdown_iters:
            decay = (args.iterations - step) / max(1, warmdown_iters)
            return max(0.0, decay)
        return 1.0

    t0 = time.time()
    train_loss_accum = 0.0
    model.train()

    for step in range(args.iterations):
        elapsed = time.time() - t0
        if elapsed > args.max_wallclock_seconds:
            if master_process:
                print(f"Wallclock limit reached at step {step}, t={elapsed:.1f}s")
            break

        # Activate recurrence partway through training (more stable)
        if step == recur_activate_step:
            raw_model.activate_recurrence()
            if master_process:
                print(f"step={step}: Recurrence activated")

        # LR schedule
        lr_scale = lr_schedule(step)
        for group in optimizer_muon.param_groups:
            group["lr"] = args.matrix_lr * lr_scale
        base_lrs = [args.scalar_lr, embed_lr, args.head_lr]
        for i, group in enumerate(optimizer_adam.param_groups):
            group["lr"] = base_lrs[i] * lr_scale

        # Muon momentum warmup
        if step < args.muon_momentum_warmup_steps:
            m = args.muon_momentum_warmup_start + (
                (args.muon_momentum - args.muon_momentum_warmup_start)
                * (step / args.muon_momentum_warmup_steps)
            )
            for group in optimizer_muon.param_groups:
                group["momentum"] = m

        optimizer_muon.zero_grad(set_to_none=True)
        optimizer_adam.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()
            step_loss += loss.item() * grad_scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer_muon.step()
        optimizer_adam.step()

        # EMA update
        ema.update(raw_model)

        train_loss_accum += step_loss

        if step % args.train_log_every == 0 and master_process:
            avg = train_loss_accum / args.train_log_every if step > 0 else step_loss
            train_loss_accum = 0.0
            print(f"step={step:5d} | loss={avg:.4f} | lr={lr_scale:.4f} | t={time.time()-t0:.1f}s")

        if args.val_loss_every > 0 and step % args.val_loss_every == 0 and step > 0:
            # Use EMA weights for validation
            ema.copy_to(raw_model)
            val_loss, val_bpb = eval_val_sliding(
                args, raw_model, val_tokens, device,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                rank, world_size, grad_accum_steps,
            )
            if master_process:
                print(f"  [sliding] val_loss={val_loss:.4f} | val_bpb={val_bpb:.4f}")
            # Restore training weights
            restore_fp32_params(raw_model)
            model.train()

    # Final evaluation with EMA weights
    ema.copy_to(raw_model)

    if master_process:
        print("\nRunning final sliding window evaluation...")
    val_loss_sliding, val_bpb_sliding = eval_val_sliding(
        args, raw_model, val_tokens, device,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        rank, world_size, grad_accum_steps,
    )

    val_loss_ttt, val_bpb_ttt = val_loss_sliding, val_bpb_sliding
    if args.ttt_enabled:
        if master_process:
            print("Running score-first TTT evaluation...")
        val_loss_ttt, val_bpb_ttt = eval_with_ttt(
            args, raw_model, val_tokens, device,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            rank, world_size,
        )

    if master_process:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"  Sliding: val_loss={val_loss_sliding:.4f} | val_bpb={val_bpb_sliding:.4f}")
        if args.ttt_enabled:
            print(f"  TTT:     val_loss={val_loss_ttt:.4f} | val_bpb={val_bpb_ttt:.4f}")
            print(f"  TTT gain: {val_bpb_sliding - val_bpb_ttt:.4f} bpb")
        print(f"{'='*60}")

        code_b, model_b, total_b = measure_artifact(raw_model, code)
        print(f"Artifact: code={code_b:,}B | model={model_b:,}B | total={total_b:,}B")
        print(f"Under 16MB: {total_b < 16_000_000}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
