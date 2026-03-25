"""
GolfStudent v2: 16MB Hybrid LM (LinearRecurrence + Attention)
d=352, L=14 (10 Recurrence + 4 Attention), vocab=1024, weight-tied, SwiGLU

Submission for openai/parameter-golf challenge.

Key techniques:
- Weight-tied embedding / lm-head
- Hybrid architecture: linear-recurrence (O(L)) + attention every 3rd layer
- Value Residuals: learned skip gates every 3 blocks (init=0, tanh-gated)
- Muon optimizer (momentum 0.85→0.99 warmup) for matrix params
- EMA (decay=0.997) for final checkpoint
- Schedule-free final 120s window (LR floor=10%, EMA decay=0.97)
- GPTQ-lite: 5 clip percentiles per row, min-MSE INT8 + zlib compression
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import struct
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

# ─────────────────────────────────────────────
# HYPERPARAMETERS  (all env-var overridable)
# ─────────────────────────────────────────────
class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH",      "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path,          "fineweb_train_*.bin")
    val_files      = os.path.join(data_path,          "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID",          str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED",        "1337"))

    # Validation
    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE",  "524288"))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY",  "1000"))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", "200"))

    # Training length
    iterations             = int(os.environ.get("ITERATIONS",             "20000"))
    warmdown_iters         = int(os.environ.get("WARMDOWN_ITERS",         "200"))    # ~last 80s on 8xH100 (last 13% of 10-min run). 3500 was too large (>600s budget).
    warmup_steps           = int(os.environ.get("WARMUP_STEPS",           "0"))   # warmup breaks weight tying via load_state_dict; skip for correctness
    train_batch_tokens     = int(os.environ.get("TRAIN_BATCH_TOKENS",     "524288"))
    train_seq_len          = int(os.environ.get("TRAIN_SEQ_LEN",          "1024"))
    max_wallclock_seconds  = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "600.0"))

    # Model shape (vocab=1024 saves ~885KB vs 4096, allowing d=288/L=14)
    vocab_size   = int(os.environ.get("VOCAB_SIZE",   "1024"))
    num_layers   = int(os.environ.get("NUM_LAYERS",   "14"))
    model_dim    = int(os.environ.get("MODEL_DIM",    "352"))  # ~15.4MB trained (96% budget)
    num_heads    = int(os.environ.get("NUM_HEADS",    "8"))
    ffn_mult     = int(os.environ.get("FFN_MULT",     "3"))   # SwiGLU hidden = ffn_mult * d
    attn_every   = int(os.environ.get("ATTN_EVERY",   "3"))   # attention every N layers

    # Optimizer (tuned: lower LRs reduce post-quant degradation)
    embed_lr     = float(os.environ.get("EMBED_LR",    "0.05"))
    matrix_lr    = float(os.environ.get("MATRIX_LR",   "0.025"))
    scalar_lr    = float(os.environ.get("SCALAR_LR",   "0.025"))
    muon_momentum              = float(os.environ.get("MUON_MOMENTUM",              "0.99"))   # stronger
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", "0.85"))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS",   "1500"))  # longer warmup
    muon_backend_steps  = int(os.environ.get("MUON_BACKEND_STEPS",  "5"))
    muon_weight_decay   = float(os.environ.get("MUON_WEIGHT_DECAY", "0.04"))  # decoupled WD
    beta1    = float(os.environ.get("BETA1",    "0.9"))
    beta2    = float(os.environ.get("BETA2",    "0.95"))
    adam_eps = float(os.environ.get("ADAM_EPS", "1e-8"))

    # EMA / training quality
    ema_decay = float(os.environ.get("EMA_DECAY", "0.997"))   # stronger than 0.999
    grad_clip = float(os.environ.get("GRAD_CLIP", "0.3"))     # tighter clip

    # Schedule-Free final window (replaces linear warmdown LR → 0 in last N seconds)
    # Keep LR at a constant floor so gradients stay sharp; EMA averages out the noise.
    schedule_free_window_s  = float(os.environ.get("SF_WINDOW_S",   "120.0"))  # final 2 min
    schedule_free_lr_floor  = float(os.environ.get("SF_LR_FLOOR",   "0.10"))   # 10% of peak
    schedule_free_ema_decay = float(os.environ.get("SF_EMA_DECAY",  "0.97"))   # faster averaging


# ─────────────────────────────────────────────
# MUON OPTIMIZER
# Background: https://kellerjordan.github.io/posts/muon/
# ─────────────────────────────────────────────
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
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                     nesterov=nesterov, weight_decay=weight_decay))

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
            lr        = group["lr"]
            momentum  = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov  = group["nesterov"]

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
                # Decoupled weight decay before the gradient step
                if group.get("weight_decay", 0.0) > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
def load_data_shard(path: Path) -> Tensor:
    """Load a contest binary shard. Header: 256×int32 (1024 bytes), magic=20240520."""
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", offset=256 * 4, count=num_tokens)
    return torch.tensor(tokens.astype(np.int32), dtype=torch.int32)


class DistributedTokenLoader:
    """Streams fixed-size token chunks from binary shards across ranks."""
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No data files found: {pattern}")
        random.shuffle(self.files)
        self._file_idx = 0
        self._tokens: Tensor | None = None
        self._pos = 0

    def _load_next_file(self):
        if self._file_idx >= len(self.files):
            self._file_idx = 0
            random.shuffle(self.files)
        self._tokens = load_data_shard(Path(self.files[self._file_idx]))
        self._file_idx += 1
        self._pos = 0

    def next_batch(self, batch_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = batch_tokens // (self.world_size * grad_accum_steps)
        local_seqs   = local_tokens // seq_len
        need = local_seqs * seq_len + 1

        while self._tokens is None or self._pos + need > len(self._tokens):
            self._load_next_file()

        chunk = self._tokens[self._pos: self._pos + need].to(self.device, dtype=torch.int64, non_blocking=True)
        self._pos += local_tokens
        x = chunk[:-1].reshape(local_seqs, seq_len)
        y = chunk[1:].reshape(local_seqs, seq_len)
        return x, y


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


# ─────────────────────────────────────────────
# TOKENIZER LOOKUP TABLES
# ─────────────────────────────────────────────
def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab = int(sp.vocab_size())
    table_size = max(sp_vocab, vocab_size)
    base_bytes_np       = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_np      = np.ones((table_size,),  dtype=np.bool_)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np,        dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np,  dtype=torch.bool,  device=device),
        torch.tensor(is_boundary_np,        dtype=torch.bool,  device=device),
    )


# ─────────────────────────────────────────────
# MODEL — GolfStudent Hybrid
# d=288, L=14 (LinearRecurrence x10 + Attention x4, every 3rd layer)
# ─────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: Tensor, k: Tensor, seq_len: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    head_dim = q.shape[-1]
    device   = q.device
    dtype    = q.dtype
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos      = torch.arange(seq_len, device=device).float()
    freqs    = torch.einsum("i,j->ij", pos, inv_freq)
    emb      = torch.cat([freqs, freqs], dim=-1).to(dtype)
    cos      = emb.cos()[None, None, :, :]
    sin      = emb.sin()[None, None, :, :]
    q_rot    = q * cos + rotate_half(q) * sin
    k_rot    = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, mult: int = 3):
        super().__init__()
        hidden = dim * mult
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LinearRecurrenceLayer(nn.Module):
    """Causal gated MLP with 1-step shift (stable, no loops, bfloat16-safe).

    Acts as a soft "convolution" layer: output at t mixes input[t] and input[t-1]
    through a learned gating mechanism. Replaces the unstable linear recurrence that
    had exp() overflow issues. O(T) in compute, no sequential loops, torch.compile safe.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm      = RMSNorm(dim)
        self.in_proj   = nn.Linear(dim, dim * 4, bias=False)   # gate + up + shift_gate + shift_up
        self.out_proj  = nn.Linear(dim * 2, dim, bias=False)
        self.ffn       = SwiGLUFFN(dim)
        self.norm2     = RMSNorm(dim)
        self.mix_alpha = nn.Parameter(torch.ones(dim) * 0.5)   # learnable shift mixing

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        h = self.norm(x)
        # Causal 1-step shift: x_prev[t] = x[t-1], zeros for t=0
        h_shift = torch.cat([torch.zeros(B, 1, D, device=x.device, dtype=x.dtype), h[:, :-1, :]], dim=1)
        # Learnable interpolation between current and previous token
        alpha = torch.sigmoid(self.mix_alpha)
        h_mix = alpha * h + (1 - alpha) * h_shift

        # Double-gated projection
        proj = self.in_proj(h_mix)                              # (B, T, 4D)
        g1, u1, g2, u2 = proj.chunk(4, dim=-1)
        branch1 = F.silu(g1) * u1                               # (B, T, D)
        branch2 = F.silu(g2) * u2                               # (B, T, D)

        out = self.out_proj(torch.cat([branch1, branch2], dim=-1))
        x   = x + out
        x   = x + self.ffn(self.norm2(x))
        return x


class AttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.norm  = RMSNorm(dim)
        self.qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.proj  = nn.Linear(dim, dim,     bias=False)
        self.ffn   = SwiGLUFFN(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, T)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        out  = self.proj(attn)
        x    = x + out
        x    = x + self.ffn(self.norm2(x))
        return x


class GolfStudent(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        V, D, L    = args.vocab_size, args.model_dim, args.num_layers
        self.embed    = nn.Embedding(V, D)
        layers = []
        for i in range(L):
            if (i + 1) % args.attn_every == 0:
                layers.append(AttentionLayer(D, args.num_heads))
            else:
                layers.append(LinearRecurrenceLayer(D))
        self.blocks   = nn.ModuleList(layers)
        self.norm_out = RMSNorm(D)
        # Weight tying: lm_head shares weights with embed
        self.lm_head  = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.embed.weight
        # Value residuals: learned skip-connections every 3 blocks (init=0 → ramps via gradient)
        n_groups = max(1, L // args.attn_every)   # 14//3 = 4 groups
        self.vr_alphas = nn.Parameter(torch.zeros(n_groups))
        # Orthogonal init on all matrices (improves Muon convergence + quantization)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.ndim == 2 and m.weight is not self.embed.weight:
                nn.init.orthogonal_(m.weight, gain=1.0)

    def forward(self, x: Tensor, targets: Tensor | None = None) -> Tensor:
        h = self.embed(x)
        vr_checkpoints: list[Tensor] = []
        vr_idx = 0
        for i, block in enumerate(self.blocks):
            # Store checkpoint at start of each 3-block group
            if i % 3 == 0:
                vr_checkpoints.append(h)
            h = block(h)
            # At end of each 3-block group, inject value residual from previous group
            if (i + 1) % 3 == 0:
                group = (i + 1) // 3 - 1   # group index just completed
                if group > 0 and group - 1 < len(self.vr_alphas):
                    alpha = torch.tanh(self.vr_alphas[group - 1])
                    h = h + alpha * vr_checkpoints[group - 1]
                vr_idx += 1
        h = self.norm_out(h)
        logits = self.lm_head(h)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


# ─────────────────────────────────────────────
# INT8 + ZLIB QUANTIZATION (exact contest format)
# ─────────────────────────────────────────────
CONTROL_TENSOR_NAME_PATTERNS = ("log_decay", "weight",)   # keep small/special tensors in FP16
INT8_KEEP_FLOAT_MAX_NUMEL   = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE    = torch.float16
INT8_CLIP_PERCENTILE        = 99.99984
INT8_CLIP_Q                 = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict) -> Tensor:
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


# GPTQ-lite: 5 clip percentiles per row, pick min reconstruction MSE
_GPTQ_CLIP_QS = [0.9990, 0.9995, 0.9999, 0.99999, float(INT8_CLIP_Q)]


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # GPTQ-lite: try each clip percentile, keep the scale with minimum MSE
        best_q, best_s, best_mse = None, None, float("inf")
        for clip_q in _GPTQ_CLIP_QS:
            clip_abs = (
                torch.quantile(t32.abs(), clip_q, dim=1)
                if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
            )
            clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
            scale   = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            q       = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
            recon   = q.float() * scale[:, None]
            mse     = float((t32 - recon).pow(2).mean())
            if mse < best_mse:
                best_mse, best_q, best_s = mse, q, scale
        return best_q.contiguous(), best_s.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    # 1-D / scalar: single-percentile (original behaviour)
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict):
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes = {}, {}, {}, {}, {}
    qmeta: dict = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"), 0
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"]           += int(t.numel())
        stats["num_tensors"]           += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"]  += tensor_nbytes(t)
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
        scales[name]    = s
        dtypes[name]    = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales":    scales,
        "dtypes":    dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_state: dict) -> dict:
    quantized   = quant_state["quantized"]
    scales      = quant_state["scales"]
    dtypes      = quant_state["dtypes"]
    passthrough = quant_state["passthrough"]
    qmeta       = quant_state.get("qmeta", {})
    out = {}
    for name, q in quantized.items():
        s    = scales[name]
        dtype = getattr(torch, dtypes[name])
        if qmeta.get(name, {}).get("scheme") == "per_row":
            w = q.float() * s.float().unsqueeze(1)
        else:
            w = q.float() * s.float()
        out[name] = w.to(dtype)
    out.update({k: v for k, v in passthrough.items()})
    return out


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
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
    is_boundary_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs   = max(1, local_batch_tokens // args.train_seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start  = (total_seqs * rank) // world_size
    seq_end    = (total_seqs * (rank + 1)) // world_size

    loss_sum  = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Unwrap DDP for validation — avoids collective sync during inference,
    # matches what top leaderboard entries do in eval.
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()
    with torch.inference_mode():
        for bs in range(seq_start, seq_end, local_batch_seqs):
            be   = min(bs + local_batch_seqs, seq_end)
            rs   = bs * args.train_seq_len
            re   = be * args.train_seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = eval_model(x, y).detach()
            n = float(y.numel())
            loss_sum  += batch_loss.to(torch.float64) * n
            tok_count += n
            prev_ids = x.reshape(-1)
            tgt_ids  = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).to(dtype=torch.int16)
            byte_count  += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum,   op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count,  op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    eval_model.train()
    val_loss = float(loss_sum / tok_count)
    val_bpb  = float(loss_sum / byte_count / math.log(2))
    return val_loss, val_bpb


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    code = open(__file__).read()

    # ── Distributed setup ────────────────────────────────────────────────────
    distributed = dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank        = dist.get_rank()
        world_size  = dist.get_world_size()
        local_rank  = int(os.environ["LOCAL_RANK"])
    else:
        rank = local_rank = 0
        world_size = 1

    master_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    def log0(msg: str) -> None:
        if master_process:
            print(msg, flush=True)

    args = Hyperparameters()
    torch.manual_seed(args.seed + rank)

    grad_accum_steps    = max(1, args.train_batch_tokens // (args.train_seq_len * world_size * 64))
    grad_scale          = 1.0 / grad_accum_steps

    # ── Tokenizer LUTs ───────────────────────────────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # ── Validation tokens (loaded once on all ranks) ─────────────────────────
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    # ── Model ─────────────────────────────────────────────────────────────────
    base_model = GolfStudent(args).to(device).to(torch.bfloat16)
    n_params   = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")

    # Note: torch.compile with dynamic=True causes shape-inference failures
    # in inductor when mixing cumsum + dynamic sequence length. Running eager —
    # cumsum and SDPA are already optimized CUDA kernels, overhead is minimal.
    model: nn.Module = (
        DDP(base_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else base_model
    )

    # ── EMA shadow (CPU float32) ──────────────────────────────────────────────
    ema_state = {k: v.detach().cpu().float().clone()
                 for k, v in base_model.state_dict().items()}

    def update_ema():
        d = args.ema_decay
        with torch.no_grad():
            for k, v in base_model.state_dict().items():
                ema_state[k].mul_(d).add_(v.detach().cpu().float(), alpha=1.0 - d)

    # ── Optimizers ────────────────────────────────────────────────────────────
    block_named = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named if p.ndim == 2]
    scalar_params = [p for n, p in block_named if p.ndim < 2]

    optimizer_embed = torch.optim.Adam(
        [{"params": [base_model.embed.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr,
        momentum=args.muon_momentum, backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
    )
    for g in optimizer_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_embed, optimizer_muon, optimizer_scalar]

    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"embed_lr:{args.embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} seq_len:{args.train_seq_len} "
         f"iterations:{args.iterations} max_wallclock:{args.max_wallclock_seconds:.0f}s "
         f"warmdown_iters:{args.warmdown_iters} ema_decay:{args.ema_decay}")

    # ── Data loader ───────────────────────────────────────────────────────────
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
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        remaining_ms  = max(max_wallclock_ms - elapsed_ms, 0.0)
        sf_ms = args.schedule_free_window_s * 1000.0
        # Schedule-free window: constant LR floor — keep gradients sharp, EMA averages
        if remaining_ms <= sf_ms:
            return args.schedule_free_lr_floor
        warmdown_ms   = args.warmdown_iters * step_ms
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms + sf_ms else 1.0

    # ── Warmup (prime compile paths, then restore weights) ────────────────────
    if args.warmup_steps > 0:
        init_state  = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opts   = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = (ms == grad_accum_steps - 1)
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wloss = model(x, y)
                (wloss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(init_state, strict=True)
        # Re-tie lm_head.weight — load_state_dict restores it as an independent
        # Parameter, breaking weight sharing. Must explicitly re-tie.
        base_model.lm_head.weight = base_model.embed.weight
        for opt, st in zip(optimizers, init_opts):
            opt.load_state_dict(st)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── Main training loop ────────────────────────────────────────────────────
    training_time_ms  = 0.0
    stop_after_step   = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                               val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = (ms == grad_accum_steps - 1)
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in optimizer_muon.param_groups:
            g["momentum"] = muon_mom

        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale

        torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # Schedule-free EMA: use faster decay in final SF window for sharper averaging
        in_sf = (max_wallclock_ms is not None and
                 max(max_wallclock_ms - (training_time_ms + 1000.0*(time.perf_counter()-t0)), 0)
                 <= args.schedule_free_window_s * 1000.0)
        ema_d = args.schedule_free_ema_decay if in_sf else args.ema_decay
        with torch.no_grad():
            for k, v in base_model.state_dict().items():
                ema_state[k].mul_(ema_d).add_(v.detach().cpu().float(), alpha=1.0 - ema_d)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms lr_mul:{scale:.4f}")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if master_process:
        log0(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")

    # ── Serialization + roundtrip validation ─────────────────────────────────
    # Use EMA weights for export (consistently better BPB)
    if master_process:
        # Load EMA state into model for export
        ema_state_dict = {k: v.to(device=device).to(torch.bfloat16) for k, v in ema_state.items()}
        base_model.load_state_dict(ema_state_dict, strict=True)
        base_model.lm_head.weight = base_model.embed.weight   # re-tie after load
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes  = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf  = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw  = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)

    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        qf_bytes   = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio      = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"Serialized model int8+zlib: {qf_bytes} bytes "
             f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)")
        log0(f"Total submission size int8+zlib: {qf_bytes + code_bytes} bytes")
        total_bytes = qf_bytes + code_bytes
        if total_bytes > 16_000_000:
            log0(f"WARNING: {total_bytes} > 16,000,000 — OVER LIMIT")
        else:
            log0(f"SIZE OK: {total_bytes:,} / 16,000,000 ({total_bytes/16e6*100:.1f}%)")

    if distributed:
        dist.barrier()

    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0*(time.perf_counter()-t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
