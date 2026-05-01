"""
Stochastic Feature Transformer (SFT)

Architecture: SP8192, 11L×512d, GQA 8H/4KV, MLP 4×, 3-layer depth recurrence,
parallel residuals (from layer 7), partial RoPE, QK-Gain 5.25, U-net skip connections,
EMA, GPTQ int6 + Brotli-11, score-first TTT.
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
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

class Hyperparameters:
    # Data paths (SP8192 tokenizer + dataset)
    data_dir = os.environ.get("DATA_DIR", "./data/")
    datasets_dir = os.environ.get("DATASETS_DIR", None)
    tokenizer_path = os.environ.get("TOKENIZER_PATH", None)
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))

    # Validation
    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))

    # Training
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    # MLP
    mlp_expansion = float(os.environ.get("MLP_EXPANSION", 4.0))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.25))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tie_embeddings = True
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    ln_scale = True

    # Depth recurrence: loop layers [loop_start..loop_end] num_loops extra times
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 5))
    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))

    # Parallel residuals
    parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", 7))

    # Optimizer
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.022))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_wd = float(os.environ.get("MUON_WD", 0.095))
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # EMA
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))

    # Quantization
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 20.0))
    compressor = os.environ.get("COMPRESSOR", "brotli")
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 64))
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 12.0))

    # TTT (Test-Time Training)
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.005))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))

    # Sliding window eval
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    sliding_window_enabled = bool(int(os.environ.get("SLIDING_WINDOW_ENABLED", "1")))

    def __init__(self):
        if self.datasets_dir is None:
            self.datasets_dir = os.path.join(self.data_dir, f"datasets/fineweb10B_sp{self.vocab_size}")
        if self.tokenizer_path is None:
            self.tokenizer_path = os.path.join(self.data_dir, f"tokenizers/fineweb_{self.vocab_size}_bpe.model")
        self.train_files = os.path.join(self.datasets_dir, "fineweb_train_*.bin")
        self.val_files = os.path.join(self.datasets_dir, "fineweb_val_*.bin")


# ============================================================================
# CONTROL TENSOR PATTERNS (kept in higher precision)
# ============================================================================

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_gate", "skip_weight", "ln_scale_param")


# ============================================================================
# MUON OPTIMIZER (row-normalized variant)
# ============================================================================

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
    # Row normalization (MuonEq-R)
    row_norms = X.norm(dim=-1, keepdim=True).clamp(min=eps)
    X = X / row_norms
    X *= max(1, X.size(0) / X.size(1)) ** 0.5
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 weight_decay: float = 0.0, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      weight_decay=weight_decay, nesterov=nesterov))

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
            wd = group["weight_decay"]
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
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ============================================================================
# BPB EVALUATION (byte-level scoring for tokenizer-agnostic comparison)
# ============================================================================

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


def eval_val(
    model: nn.Module,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len: int,
    batch_tokens: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[float, float]:
    local_batch_seqs = max(batch_tokens // (world_size * seq_len), 1)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for bs in range(seq_start, seq_end, local_batch_seqs):
            be = min(bs + local_batch_seqs, seq_end)
            raw_s = bs * seq_len
            raw_e = be * seq_len + 1
            local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            n_tok = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * n_tok
            val_token_count += n_tok
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
    bpt = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)


def eval_val_sliding(
    model: nn.Module,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len: int,
    stride: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    total_tokens = val_tokens.numel() - 1
    n_windows = max((total_tokens - seq_len) // stride + 1, 1)
    win_start = (n_windows * rank) // world_size
    win_end = (n_windows * (rank + 1)) // world_size

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for w in range(win_start, win_end):
            offset = w * stride
            x = val_tokens[offset : offset + seq_len].to(device=device, dtype=torch.int64).unsqueeze(0)
            y = val_tokens[offset + 1 : offset + seq_len + 1].to(device=device, dtype=torch.int64).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                per_tok = model(x, y, return_per_token_loss=True)  # [1, seq_len]
            # Only score the last `stride` tokens (or all if first window)
            score_start = 0 if w == 0 else seq_len - stride
            score_end = seq_len
            scored_loss = per_tok[0, score_start:score_end]
            scored_x = x[0, score_start:score_end]
            scored_y = y[0, score_start:score_end]
            n = float(scored_loss.numel())
            loss_sum += scored_loss.to(torch.float64).sum()
            token_count += n
            tb = base_bytes_lut[scored_y].to(torch.int16)
            tb += (has_leading_space_lut[scored_y] & ~is_boundary_token_lut[scored_x]).to(torch.int16)
            byte_count += tb.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bpt = val_loss.item() / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
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

    def next_batch(self, global_tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
        per_rank = global_tokens // self.world_size + 1
        chunk = self.stream.take(per_rank * self.world_size)
        start = self.rank * per_rank
        local = chunk[start : start + per_rank].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ============================================================================
# MODEL: STOCHASTIC FEATURE TRANSFORMER
# ============================================================================

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
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
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


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, rope_dims: int, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_dims = rope_dims
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(rope_dims, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # QK-RMSNorm
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # Partial RoPE: only apply to first rope_dims dimensions
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q_rope, q_pass = q[..., :self.rope_dims], q[..., self.rope_dims:]
        k_rope, k_pass = k[..., :self.rope_dims], k[..., self.rope_dims:]
        q_rope = apply_rotary_emb(q_rope, cos, sin)
        k_rope = apply_rotary_emb(k_rope, cos, sin)
        q = torch.cat([q_rope, q_pass], dim=-1)
        k = torch.cat([k_rope, k_pass], dim=-1)
        # QK-Gain
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, expansion: float = 4.0):
        super().__init__()
        hidden = int(expansion * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_expansion: float,
                 rope_base: float, rope_dims: int, qk_gain_init: float, layer_idx: int,
                 parallel_residual: bool = False, ln_scale_val: float = 1.0):
        super().__init__()
        self.parallel_residual = parallel_residual
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, rope_dims, qk_gain_init)
        self.mlp = MLP(dim, mlp_expansion)
        self.attn_scale = nn.Parameter(torch.full((dim,), ln_scale_val, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.full((dim,), ln_scale_val, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        h = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        if self.parallel_residual:
            # GPT-J style: attention and MLP read from same normalized input
            normed = self.attn_norm(h)
            attn_out = self.attn(normed)
            mlp_out = self.mlp(self.mlp_norm(h))
            x = h + self.attn_scale.to(x.dtype)[None, None, :] * attn_out \
                  + self.mlp_scale.to(x.dtype)[None, None, :] * mlp_out
        else:
            attn_out = self.attn(self.attn_norm(h))
            h = h + self.attn_scale.to(x.dtype)[None, None, :] * attn_out
            mlp_out = self.mlp(self.mlp_norm(h))
            x = h + self.mlp_scale.to(x.dtype)[None, None, :] * mlp_out
        return x


class SFT(nn.Module):
    """Stochastic Feature Transformer."""

    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.logit_softcap = args.logit_softcap
        self.tie_embeddings = args.tie_embeddings
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)

        # LN scale: 1/sqrt(2*num_layers + 1) for residual scaling
        ln_scale_val = 1.0 / math.sqrt(2 * args.num_layers + 1) if args.ln_scale else 1.0

        # U-net: encoder layers = first half, decoder layers = second half
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip = min(self.num_encoder_layers, self.num_decoder_layers)

        # Sigmoid-gated skip connections (U-net)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip, args.model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip, args.model_dim, dtype=torch.float32))

        # Blocks
        self.blocks = nn.ModuleList()
        for i in range(args.num_layers):
            parallel = (i >= args.parallel_residual_start)
            self.blocks.append(Block(
                dim=args.model_dim,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                mlp_expansion=args.mlp_expansion,
                rope_base=args.rope_base,
                rope_dims=args.rope_dims,
                qk_gain_init=args.qk_gain_init,
                layer_idx=i,
                parallel_residual=parallel,
                ln_scale_val=ln_scale_val,
            ))

        self.final_norm = RMSNorm()
        self.lm_head = None  # Tied embeddings

        # Depth recurrence config (set dynamically during training)
        self._looping_enabled = False
        self.loop_start = args.loop_start
        self.loop_end = args.loop_end
        self.num_loops = args.num_loops

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.args.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def enable_looping(self):
        self._looping_enabled = True

    def _build_encoder_decoder(self) -> tuple[list[int], list[int]]:
        """Build encoder/decoder schedules matching SOTA depth recurrence pattern.
        With loop_start=3, loop_end=5, num_loops=2, 11 layers:
          encoder: [0,1,2,3,4,5,3,4]  decoder: [5,3,4,5,6,7,8,9,10]
        """
        n = self.args.num_layers
        ls, le, nl = self.loop_start, self.loop_end, self.num_loops
        if not self._looping_enabled or nl <= 0:
            mid = n // 2
            return list(range(mid)), list(range(mid, n))
        # Encoder: first pass [0..le], then [ls..le-1] repeated (nl-1) times
        enc = list(range(le + 1))
        for _ in range(nl - 1):
            enc.extend(range(ls, le))
        # Decoder: [le], then [ls..le] repeated (nl-1) times, then [le+1..n-1]
        dec = [le]
        for _ in range(nl - 1):
            dec.extend(range(ls, le + 1))
        dec.extend(range(le + 1, n))
        return enc, dec

    def forward(self, input_ids: Tensor, target_ids: Tensor,
                return_per_token_loss: bool = False) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        enc_sched, dec_sched = self._build_encoder_decoder()
        skips: list[Tensor] = []

        # Encoder phase: run layers and save skip connections
        for i, layer_id in enumerate(enc_sched):
            x = self.blocks[layer_id](x, x0)
            if i < self.num_skip:
                skips.append(x)

        # Decoder phase: consume skip connections (LIFO) and run layers
        for i, layer_id in enumerate(dec_sched):
            if skips and i < self.num_skip:
                gate = torch.sigmoid(self.skip_gates[i].to(x.dtype))[None, None, :]
                x = x + gate * self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[layer_id](x, x0)

        x = self.final_norm(x)
        targets = target_ids.reshape(-1)

        # Tied embeddings
        logits = F.linear(x.reshape(-1, x.size(-1)), self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        if return_per_token_loss:
            per_tok = F.cross_entropy(logits.float(), targets, reduction="none")
            return per_tok.reshape(input_ids.shape)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9965):
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}
        self.active = False

    def update(self, model: nn.Module):
        if not self.active:
            # First call: just copy
            for name, p in model.named_parameters():
                self.shadow[name].copy_(p.data)
            self.active = True
            return
        d = self.decay
        for name, p in model.named_parameters():
            self.shadow[name].mul_(d).add_(p.data, alpha=1.0 - d)

    def apply(self, model: nn.Module):
        """Replace model weights with EMA weights."""
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: dict[str, Tensor]):
        """Restore model weights from backup."""
        for name, p in model.named_parameters():
            p.data.copy_(backup[name])


# ============================================================================
# GPTQ QUANTIZATION (SDClip + column-wise error redistribution)
# ============================================================================

def sdclip_quantize(W: Tensor, bits: int, clip_sigmas: float) -> tuple[Tensor, Tensor]:
    """SDClip quantization: clip at k*sigma per row, then round to N-bit."""
    qmax = (1 << (bits - 1)) - 1
    W = W.float()
    row_std = W.std(dim=1, keepdim=True).clamp(min=1e-8)
    clip_val = clip_sigmas * row_std
    W_clipped = W.clamp(-clip_val, clip_val)
    row_max = W_clipped.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scale = row_max / qmax
    Q = (W_clipped / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return Q, scale.squeeze(1).to(torch.float16)


def gptq_quantize(W: Tensor, H: Tensor, bits: int, clip_sigmas: float) -> tuple[Tensor, Tensor]:
    """Full GPTQ with Hessian-guided error redistribution + SDClip."""
    qmax = (1 << (bits - 1)) - 1
    W = W.float().clone()
    n_out, n_in = W.shape

    # Pre-compute per-row scales via SDClip
    row_std = W.std(dim=1, keepdim=True).clamp(min=1e-8)
    clip_val = clip_sigmas * row_std
    W.clamp_(-clip_val, clip_val)
    row_max = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scale = row_max / qmax  # (n_out, 1)

    # Damping for numerical stability
    damp = 0.01 * H.diagonal().mean().clamp(min=1e-6)
    H = H.clone()
    H.diagonal().add_(damp)

    # Compute H_inv
    try:
        H_inv = torch.linalg.inv(H)
    except Exception:
        H.diagonal().add_(damp * 100)
        H_inv = torch.linalg.inv(H)

    # Column-by-column GPTQ
    Q = torch.zeros(n_out, n_in, dtype=torch.int8, device=W.device)
    for j in range(n_in):
        w = W[:, j]
        q = (w / scale.squeeze(1)).round().clamp(-qmax, qmax)
        Q[:, j] = q.to(torch.int8)
        err = w - q * scale.squeeze(1)
        if j + 1 < n_in:
            correction = H_inv[j, j + 1:] / H_inv[j, j].clamp(min=1e-12)
            W[:, j + 1:] -= err.unsqueeze(1) * correction.unsqueeze(0)

    return Q, scale.squeeze(1).to(torch.float16)


def collect_hessians(model: nn.Module, loader, device: torch.device,
                     n_batches: int, seq_len: int) -> dict[str, Tensor]:
    """Collect Hessians H = X^T @ X for each Linear layer via forward hooks."""
    hessians: dict[str, Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, inputs, outputs):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            H = x.T @ x
            if name in hessians:
                hessians[name].add_(H)
            else:
                hessians[name] = H.clone()
        return hook_fn

    for name, mod in model.named_modules():
        if isinstance(mod, CastedLinear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            x, y = loader.next_batch(seq_len * 8, seq_len)  # Small batch for calibration
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                model(x, y)

    for h in hooks:
        h.remove()
    return hessians


def quantize_model(model: nn.Module, hessians: dict[str, Tensor], args: Hyperparameters,
                   log_fn=None) -> dict[str, object]:
    """Quantize model weights: GPTQ int6 for matrices, int8 for embeddings."""
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict] = {}

    gptq_names = []
    for name, param in model.named_parameters():
        t = param.detach().cpu().contiguous()

        if not t.is_floating_point():
            passthrough[name] = t
            continue

        is_control = any(pat in name for pat in CONTROL_PATTERNS)
        is_embedding = "tok_emb" in name

        if is_control or t.numel() <= 65536:
            # Small tensors: keep as fp16
            orig_dtype = str(t.dtype).removeprefix("torch.")
            passthrough_orig_dtypes[name] = orig_dtype
            passthrough[name] = t.to(torch.float16).contiguous()
            continue

        if is_embedding:
            bits = args.embed_bits
            clip_sig = args.embed_clip_sigmas
        else:
            bits = args.matrix_bits
            clip_sig = args.matrix_clip_sigmas

        # Use GPTQ if Hessian available, otherwise SDClip
        H = hessians.get(name.replace(".weight", ""))
        if H is not None and t.ndim == 2:
            Q, S = gptq_quantize(t, H.cpu(), bits, clip_sig)
            gptq_names.append(name)
        else:
            Q, S = sdclip_quantize(t, bits, clip_sig)

        quantized[name] = Q
        scales[name] = S
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        if t.ndim == 2:
            qmeta[name] = {"scheme": "per_row", "axis": 0, "bits": bits}

    if log_fn:
        log_fn(f"Quantized weights:")
        log_fn(f"  gptq (int{args.matrix_bits}): {', '.join(n.split('.')[-2] + '.' + n.split('.')[-1] for n in gptq_names[:6])}...")
        log_fn(f"  passthrough (float16): {', '.join(n.split('.')[-1] for n in list(passthrough.keys())[:6])}...")

    obj = {
        "__quant_format__": "sft_gptq_intN_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj


def dequantize_state_dict(obj: dict[str, object]) -> dict[str, Tensor]:
    """Dequantize a compressed state dict back to full precision."""
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].float()
        if q.ndim == 2:
            out[name] = (q.float() * s.unsqueeze(1)).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * s).to(dtype=dtype).contiguous()

    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig = passthrough_orig_dtypes.get(name)
        if isinstance(orig, str):
            out_t = out_t.to(dtype=getattr(torch, orig)).contiguous()
        out[name] = out_t

    return out


def compress_artifact(obj: dict, compressor: str = "brotli") -> bytes:
    """Serialize and compress the quantized model."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw = buf.getvalue()
    if compressor == "brotli" and HAS_BROTLI:
        return brotli.compress(raw, quality=11)
    return zlib.compress(raw, level=9)


def decompress_artifact(blob: bytes, compressor: str = "brotli") -> dict:
    """Decompress and deserialize a model artifact."""
    if compressor == "brotli" and HAS_BROTLI:
        raw = brotli.decompress(blob)
    else:
        raw = zlib.decompress(blob)
    return torch.load(io.BytesIO(raw), map_location="cpu")


# ============================================================================
# TEST-TIME TRAINING (Score-First SGD — Legal per Issue #1017)
# ============================================================================

def eval_val_ttt(
    model: nn.Module,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    args: Hyperparameters,
    rank: int,
    world_size: int,
    device: torch.device,
    log_fn=None,
) -> tuple[float, float]:
    """
    Score-first TTT: for each chunk of val tokens:
    1. Score the chunk under no_grad (accumulate BPB)
    2. Train model on the scored chunk with SGD

    This is LEGAL per Issue #1017 Condition 3: score-before-update.
    """
    seq_len = args.eval_seq_len
    stride = args.eval_stride
    chunk_tokens = args.ttt_chunk_tokens
    total_tokens = val_tokens.numel() - 1
    n_chunks = max(total_tokens // chunk_tokens, 1)

    if log_fn:
        log_fn(f"ttt:start chunks={n_chunks} ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs}")

    # Collect trainable parameters for TTT (only MLP output projections)
    ttt_params = []
    for name, p in model.named_parameters():
        if "mlp.proj" in name and p.requires_grad:
            ttt_params.append(p)

    if not ttt_params:
        # Fallback: train all parameters
        ttt_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_tokens
        chunk_end = min(chunk_start + chunk_tokens, total_tokens)
        if chunk_end - chunk_start < seq_len:
            continue

        # Distribute chunks across ranks
        if chunk_idx % world_size != rank:
            continue

        chunk = val_tokens[chunk_start : chunk_end + 1].to(device=device, dtype=torch.int64)

        # === PHASE 1: Score chunk under no_grad ===
        model.eval()
        with torch.inference_mode():
            n_windows = max((chunk.numel() - 1 - seq_len) // stride + 1, 1)
            for w in range(n_windows):
                offset = w * stride
                if offset + seq_len >= chunk.numel():
                    break
                x = chunk[offset : offset + seq_len].unsqueeze(0)
                y = chunk[offset + 1 : offset + seq_len + 1].unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    per_tok = model(x, y, return_per_token_loss=True)
                score_start = 0 if w == 0 else seq_len - stride
                scored_loss = per_tok[0, score_start:]
                scored_x = x[0, score_start:]
                scored_y = y[0, score_start:]
                loss_sum += scored_loss.to(torch.float64).sum()
                token_count += float(scored_loss.numel())
                tb = base_bytes_lut[scored_y].to(torch.int16)
                tb += (has_leading_space_lut[scored_y] & ~is_boundary_token_lut[scored_x]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()

        # === PHASE 2: Train on scored chunk ===
        model.train()
        n_seqs = (chunk.numel() - 1) // seq_len
        for epoch in range(args.ttt_epochs):
            for s in range(n_seqs):
                x = chunk[s * seq_len : (s + 1) * seq_len].unsqueeze(0)
                y = chunk[s * seq_len + 1 : (s + 1) * seq_len + 1].unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(ttt_params, args.grad_clip_norm)
                optimizer.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count.clamp(min=1)
    bpt = val_loss.item() / math.log(2.0)
    tpb = token_count.item() / byte_count.clamp(min=1).item()
    return float(val_loss.item()), float(bpt * tpb)


# ============================================================================
# TRAINING
# ============================================================================

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # ----- Distributed + CUDA -----
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

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

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Hyperparameters:")
    for k, v in sorted(vars(args).items()):
        if not k.startswith("_"):
            log0(f"  {k}: {v}")

    # ----- Tokenizer + Validation -----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer vocab_size={int(sp.vocab_size())}")

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"train_shards: {len(list(Path(args.datasets_dir).glob('fineweb_train_*.bin')))}")
    log0(f"val_tokens: {val_tokens.numel() - 1}")

    # ----- Model -----
    base_model = SFT(args).to(device).bfloat16()
    for mod in base_model.modules():
        if isinstance(mod, CastedLinear):
            mod.float()
    # Keep control params in fp32
    with torch.no_grad():
        for name, p in base_model.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")

    # Reserve time for GPTQ at the end
    effective_wallclock = args.max_wallclock_seconds - args.gptq_reserve_seconds
    log0(f"gptq:reserving {args.gptq_reserve_seconds}s, effective={effective_wallclock * 1000:.0f}ms")

    # ----- Optimizer -----
    block_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)]
    scalar_params = [p for n, p in block_params if p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.skip_gates.numel() > 0:
        scalar_params.append(base_model.skip_gates)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr,
          "weight_decay": args.embed_wd}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [opt_tok, opt_muon, opt_scalar]

    # ----- EMA -----
    ema = EMA(base_model, decay=args.ema_decay)

    # ----- Data Loader -----
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * effective_wallclock if effective_wallclock > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_frac <= 0:
            return 1.0
        if max_wallclock_ms is None:
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_frac * max_wallclock_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            return remaining_ms / max(warmdown_ms, 1e-9)
        return 1.0

    # ----- Warmup (compile priming) -----
    if args.warmup_steps > 0:
        initial_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_opt_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens // grad_accum_steps, args.train_seq_len)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wloss = model(x, y)
                (wloss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if ws + 1 == args.warmup_steps or (ws + 1) % 10 == 0:
                log0(f"warmup_step: {ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_state, strict=True)
        for opt, state in zip(optimizers, initial_opt_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ----- Depth recurrence warmup -----
    # Enable looping after enable_looping_at fraction of training
    loop_enable_step = None
    if args.num_loops > 0:
        # Will be enabled after warmdown_frac-based computation
        pass

    # ----- Main Training Loop -----
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        # Validation
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vbpb = eval_val(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                                is_boundary_token_lut, args.train_seq_len, args.val_batch_tokens,
                                rank, world_size, device)
            log0(f"{step}/{args.iterations} val_loss: {vl:.4f} val_bpb: {vbpb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        # Enable depth recurrence at the right time
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if max_wallclock_ms and not base_model._looping_enabled and args.num_loops > 0:
            frac = elapsed_ms / max_wallclock_ms
            if frac >= args.enable_looping_at:
                base_model.enable_looping()
                enc_s, dec_s = base_model._build_encoder_decoder()
                log0(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{enc_s} decoder:{dec_s}")
                # Re-warmup with looping
                if args.warmup_steps > 0:
                    loop_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
                    loop_opt_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
                    for lws in range(args.warmup_steps):
                        zero_grad_all()
                        for ms in range(grad_accum_steps):
                            if distributed:
                                model.require_backward_grad_sync = ms == grad_accum_steps - 1
                            x, y = train_loader.next_batch(args.train_batch_tokens // grad_accum_steps, args.train_seq_len)
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                                lw_loss = model(x, y)
                            (lw_loss * grad_scale).backward()
                        for opt in optimizers:
                            opt.step()
                        zero_grad_all()
                        if lws + 1 == args.warmup_steps or (lws + 1) % 10 == 0:
                            log0(f"loop_warmup_step: {lws + 1}/{args.warmup_steps}")
                    base_model.load_state_dict(loop_state, strict=True)
                    for opt, state in zip(optimizers, loop_opt_states, strict=True):
                        opt.load_state_dict(state)
                    zero_grad_all()
                    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

        # LR schedule
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()

        # Forward + backward
        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens // grad_accum_steps, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups:
            g["momentum"] = muon_mom

        # Apply LR scale
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale

        # Gradient clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # EMA update
        ema.update(base_model)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log = args.train_log_every > 0 and (step <= 5 or step % args.train_log_every == 0)
        if should_log:
            tok_per_s = int(step * args.train_batch_tokens / (approx_ms / 1000.0))
            log0(f"{step}/{args.iterations} train_loss: {train_loss.item():.4f} "
                 f"train_time: {approx_ms / 60000:.1f}m tok/s: {tok_per_s}")

        # Wallclock cap
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
         f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    # ----- Apply EMA -----
    log0("ema:applying EMA weights")
    ema.apply(base_model)

    # Pre-quant eval
    torch.cuda.synchronize()
    t_pre = time.perf_counter()
    pre_vl, pre_vbpb = eval_val(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                                 is_boundary_token_lut, args.train_seq_len, args.val_batch_tokens,
                                 rank, world_size, device)
    torch.cuda.synchronize()
    log0(f"pre-quantization post-ema val_loss:{pre_vl:.8f} val_bpb:{pre_vbpb:.8f} "
         f"eval_time:{1000 * (time.perf_counter() - t_pre):.0f}ms")

    # ----- Save raw model -----
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes")
        log0(f"Code size: {len(code.encode('utf-8'))} bytes")

    # ----- GPTQ Quantization -----
    log0("GPTQ:collecting Hessians from calibration data...")
    torch.cuda.synchronize()
    t_gptq = time.perf_counter()

    calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    hessians = collect_hessians(base_model, calib_loader, device,
                                args.gptq_calibration_batches, args.train_seq_len)
    log0(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t_gptq:.1f}s")

    quant_obj = quantize_model(base_model, hessians, args, log_fn=log0)
    artifact_blob = compress_artifact(quant_obj, args.compressor)

    if master_process:
        artifact_path = "final_model.int6.ptz"
        with open(artifact_path, "wb") as f:
            f.write(artifact_blob)
        artifact_bytes = os.path.getsize(artifact_path)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model quantized+{args.compressor}: {artifact_bytes} bytes")
        log0(f"Total submission size quantized+{args.compressor}: {artifact_bytes + code_bytes} bytes")

    # ----- Roundtrip validation -----
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        disk_blob = f.read()
    quant_state = decompress_artifact(disk_blob, args.compressor)
    base_model.load_state_dict(dequantize_state_dict(quant_state), strict=True)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_vl, q_vbpb = eval_val(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                             is_boundary_token_lut, args.train_seq_len, args.val_batch_tokens,
                             rank, world_size, device)
    torch.cuda.synchronize()
    log0(f"quantized val_loss:{q_vl:.8f} val_bpb:{q_vbpb:.8f} "
         f"eval_time:{1000 * (time.perf_counter() - t_qeval):.0f}ms")

    # ----- Sliding window eval (uses base_model for per-token loss) -----
    if args.sliding_window_enabled:
        torch.cuda.synchronize()
        t_sw = time.perf_counter()
        sw_vl, sw_vbpb = eval_val_sliding(base_model, val_tokens, base_bytes_lut, has_leading_space_lut,
                                           is_boundary_token_lut, args.eval_seq_len, args.eval_stride,
                                           rank, world_size, device)
        torch.cuda.synchronize()
        log0(f"quantized_sliding_window val_loss:{sw_vl:.8f} val_bpb:{sw_vbpb:.8f} "
             f"eval_time:{1000 * (time.perf_counter() - t_sw):.0f}ms")

    # ----- TTT (uses base_model for per-token loss + gradient updates) -----
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_vl, ttt_vbpb = eval_val_ttt(base_model, val_tokens, base_bytes_lut, has_leading_space_lut,
                                          is_boundary_token_lut, args, rank, world_size, device,
                                          log_fn=log0)
        torch.cuda.synchronize()
        log0(f"quantized_ttt val_loss:{ttt_vl:.8f} val_bpb:{ttt_vbpb:.8f} "
             f"eval_time:{1000 * (time.perf_counter() - t_ttt):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
