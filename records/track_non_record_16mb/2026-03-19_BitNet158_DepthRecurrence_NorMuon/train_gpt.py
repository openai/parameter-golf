"""
BitNet b1.58 training script for the OpenAI Parameter Golf challenge.

Features:
- BitLinear: ternary weights {-1,0,+1} with STE, int8 activations
- Depth recurrence: N unique blocks run R times (same compute, smaller file)
- Sequence length warmup + YaRN: train from seq_len=128 to 1024
- Sliding window evaluation: each token predicted with full context
- Ternary packing: 2 bits/weight serialization
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
from tqdm import tqdm
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as grad_checkpoint


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

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 65_536))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 20))
    val_max_seqs = int(os.environ.get("VAL_MAX_SEQS", 256))
    val_stride_factor = int(os.environ.get("VAL_STRIDE_FACTOR", 2))  # stride = seq_len // val_stride_factor
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))

    iterations = int(os.environ.get("ITERATIONS", 10000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 2000))
    lr_warmup_steps = int(os.environ.get("LR_WARMUP_STEPS", 100))
    min_lr_ratio = float(os.environ.get("MIN_LR_RATIO", 0.1))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 131_072))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Sequence length warmup + YaRN
    seq_len_start = int(os.environ.get("SEQ_LEN_START", 128))
    seq_len_warmup_steps = int(os.environ.get("SEQ_LEN_WARMUP_STEPS", 2000))

    # Early stopping
    early_stop_patience = int(os.environ.get("EARLY_STOP_PATIENCE", 5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    # Depth recurrence: total effective layers = num_unique_layers * recurrence_factor
    num_unique_layers = int(os.environ.get("NUM_UNIQUE_LAYERS", 8))
    recurrence_factor = int(os.environ.get("RECURRENCE_FACTOR", 3))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 1024))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    bitnet_eps = float(os.environ.get("BITNET_EPS", 1e-8))

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
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    # Ternary commitment loss: pushes weights toward {-1, 0, +1} to reduce quantization gap
    ternary_commit_lambda = float(os.environ.get("TERNARY_COMMIT_LAMBDA", 0.005))
    # Gradient checkpointing: trades compute for memory (recomputes block activations on backward)
    use_gradient_checkpointing = bool(int(os.environ.get("GRADIENT_CHECKPOINTING", "1")))


# -----------------------------
# SEQUENCE LENGTH WARMUP + YaRN
# -----------------------------

def get_current_seq_len(step: int, args: Hyperparameters) -> int:
    """Geometric (log-linear) warmup: seq_len_start → train_seq_len over seq_len_warmup_steps."""
    if args.seq_len_warmup_steps <= 0 or step >= args.seq_len_warmup_steps:
        return args.train_seq_len
    if args.seq_len_start >= args.train_seq_len:
        return args.train_seq_len
    progress = step / args.seq_len_warmup_steps
    log_start = math.log2(args.seq_len_start)
    log_end = math.log2(args.train_seq_len)
    log_cur = log_start + (log_end - log_start) * progress
    # Snap to nearest power of 2, then clamp
    snapped = 2 ** round(log_cur)
    return max(args.seq_len_start, min(snapped, args.train_seq_len))


def compute_yarn_base(current_seq_len: int, max_seq_len: int, base: float, head_dim: int) -> float:
    """NTK-aware RoPE base scaling for sequence length warmup (YaRN-lite)."""
    if current_seq_len >= max_seq_len:
        return base
    scale = current_seq_len / max_seq_len
    return base * (scale ** (head_dim / max(head_dim - 2, 1)))


def update_rotary_yarn(model: nn.Module, current_seq_len: int, max_seq_len: int,
                       base: float, head_dim: int) -> None:
    """Update all Rotary modules with the YaRN-adjusted base for the current seq_len."""
    yarn_base = compute_yarn_base(current_seq_len, max_seq_len, base, head_dim)
    for module in model.modules():
        if isinstance(module, Rotary):
            module.set_base(yarn_base)


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
                    # NorMuon: per-neuron row-wise RMS normalization (replaces uniform scaling)
                    row_rms = g.pow(2).mean(dim=tuple(range(1, g.ndim))).sqrt_().clamp_(min=1e-8)
                    g = g / row_rms.view(-1, *([1] * (g.ndim - 1)))
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
# BPB EVALUATION
# -----------------------------

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
    args: Hyperparameters,
    base_model: nn.Module,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    show_progress: bool = False,
) -> tuple[float, float]:
    """Sliding window evaluation: stride = seq_len // val_stride_factor."""
    seq_len = args.train_seq_len
    stride = max(1, seq_len // args.val_stride_factor)
    total_tokens = val_tokens.numel() - 1  # max start index for x

    # Collect window start positions, respecting val_max_seqs limit
    window_starts = list(range(0, total_tokens - seq_len + 1, stride))
    if args.val_max_seqs > 0:
        window_starts = window_starts[: args.val_max_seqs]

    val_loss_sum = 0.0
    val_token_count = 0
    val_byte_count = 0.0

    base_model.eval()
    # Use a modest batch size for memory efficiency
    batch_sz = max(1, args.val_batch_size // seq_len)

    with torch.inference_mode():
        it = range(0, len(window_starts), batch_sz)
        if show_progress:
            it = tqdm(it, total=math.ceil(len(window_starts) / batch_sz), desc="val", leave=False)
        for bi in it:
            batch_ws = window_starts[bi : bi + batch_sz]
            xs = torch.stack([
                val_tokens[w : w + seq_len].to(dtype=torch.int64) for w in batch_ws
            ]).to(device)
            ys = torch.stack([
                val_tokens[w + 1 : w + seq_len + 1].to(dtype=torch.int64) for w in batch_ws
            ]).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model(xs)  # [B, T, V]

            for i, w in enumerate(batch_ws):
                # Skip the first `stride` tokens for non-first windows (cold-start bias)
                skip = stride if w > 0 else 0
                lg = logits[i, skip:].float()       # [T-skip, V]
                tgt = ys[i, skip:]                   # [T-skip]
                prev = xs[i, skip:]                  # [T-skip]

                token_losses = F.cross_entropy(lg, tgt, reduction="none")
                val_loss_sum += token_losses.sum().item()
                val_token_count += tgt.numel()

                tbytes = base_bytes_lut[tgt].to(dtype=torch.int16)
                tbytes += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(dtype=torch.int16)
                val_byte_count += tbytes.to(torch.float64).sum().item()

    base_model.train()
    val_loss = val_loss_sum / max(val_token_count, 1)
    bpt = val_loss / math.log(2.0)
    tpb = val_token_count / max(val_byte_count, 1.0)
    return float(val_loss), float(bpt * tpb)


# -----------------------------
# TERNARY PACKING (2 bits/weight)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",") if p
)


def pack_ternary(w_ternary_int8: Tensor) -> tuple[bytes, int]:
    flat = w_ternary_int8.flatten().to(torch.int8)
    n = flat.numel()
    codes = (flat + 1).to(torch.uint8)
    pad = (4 - n % 4) % 4
    if pad:
        codes = torch.cat([codes, torch.zeros(pad, dtype=torch.uint8)])
    codes = codes.reshape(-1, 4)
    packed = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    return packed.cpu().numpy().tobytes(), n


def unpack_ternary(data: bytes, n: int, shape: tuple) -> Tensor:
    packed = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    codes = torch.stack([
        packed & 0x03, (packed >> 2) & 0x03,
        (packed >> 4) & 0x03, (packed >> 6) & 0x03,
    ], dim=1).reshape(-1)[:n]
    return (codes.to(torch.int8) - 1).reshape(shape).float()


def serialize_ternary_model(model: nn.Module, eps: float = 1e-8) -> dict:
    bitlinear_weight_names: set[str] = {
        name + ".weight"
        for name, module in model.named_modules()
        if isinstance(module, BitLinear)
    }
    ternary: dict = {}
    other: dict = {}
    for param_name, param in model.named_parameters():
        t = param.detach().float().cpu()
        if param_name in bitlinear_weight_names:
            alpha = t.abs().mean().clamp(min=eps)
            w_ternary = (t / alpha).clamp(-1, 1).round().to(torch.int8)
            packed_bytes, n = pack_ternary(w_ternary)
            ternary[param_name] = {"packed": packed_bytes, "shape": tuple(w_ternary.shape),
                                   "n": n, "alpha": float(alpha.item())}
        else:
            other[param_name] = t.half()
    return {"__format__": "bitnet_ternary_2bit_v1", "ternary": ternary, "other": other}


def deserialize_ternary_model(model: nn.Module, obj: dict) -> None:
    state: dict = {}
    for name, info in obj["ternary"].items():
        w_ternary = unpack_ternary(info["packed"], info["n"], info["shape"])
        state[name] = (w_ternary * info["alpha"]).to(torch.bfloat16)
    for name, t in obj["other"].items():
        state[name] = t.bfloat16()
    model.load_state_dict(state, strict=True)


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
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
        local_tokens = (global_tokens // (self.world_size * grad_accum_steps) // seq_len) * seq_len
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# MODEL MODULES
# -----------------------------

class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, eps: float = 1e-8):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        with torch.no_grad():
            alpha = w.abs().mean().clamp(min=self.eps)
            w_ternary = (w / alpha).clamp(-1, 1).round()
        w_q = w + (w_ternary * alpha - w).detach()
        with torch.no_grad():
            gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps) / 127.0
            x_int = (x / gamma).clamp(-128, 127).round()
        x_q = x + (x_int * gamma - x).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x_q, w_q / alpha, bias) * alpha


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.base = base
        self._current_base = base
        # Pre-allocate buffers at max_seq_len so dispatch key set never changes.
        # register_buffer ensures consistent AutogradCUDA keys for torch.compile.
        cos_buf, sin_buf = self._build_cache(max_seq_len, base)
        self.register_buffer("_cos_cached", cos_buf, persistent=False)
        self.register_buffer("_sin_cached", sin_buf, persistent=False)
        self._cached_seq_len = max_seq_len

    def _build_cache(self, seq_len: int, base: float) -> tuple[Tensor, Tensor]:
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]

    def set_base(self, new_base: float) -> None:
        if new_base != self._current_base:
            self._current_base = new_base
            # Rebuild in-place so the buffer tensors keep their existing dispatch key set.
            cos_new, sin_new = self._build_cache(self._cached_seq_len, new_base)
            self._cos_cached.copy_(cos_new.to(device=self._cos_cached.device))
            self._sin_cached.copy_(sin_new.to(device=self._sin_cached.device))

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        cos = self._cos_cached[:, :, :seq_len, :]
        sin = self._sin_cached[:, :, :seq_len, :]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, bitnet_eps: float):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = BitLinear(dim, dim, bias=False, eps=bitnet_eps)
        self.c_k = BitLinear(dim, kv_dim, bias=False, eps=bitnet_eps)
        self.c_v = BitLinear(dim, kv_dim, bias=False, eps=bitnet_eps)
        self.proj = BitLinear(dim, dim, bias=False, eps=bitnet_eps)
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
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, bitnet_eps: float):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = BitLinear(dim, hidden, bias=False, eps=bitnet_eps)
        self.proj = BitLinear(hidden, dim, bias=False, eps=bitnet_eps)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float, qk_gain_init: float, bitnet_eps: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, bitnet_eps)
        self.mlp = MLP(dim, mlp_mult, bitnet_eps)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    """
    GPT with depth recurrence: num_unique_layers blocks, each run recurrence_factor times.
    Total effective depth = num_unique_layers * recurrence_factor.
    U-Net skip connections span the full effective depth.
    Returns logits if target_ids is None, else cross-entropy loss.
    """
    def __init__(self, vocab_size: int, num_unique_layers: int, recurrence_factor: int,
                 model_dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 tie_embeddings: bool, tied_embed_init_std: float, logit_softcap: float,
                 rope_base: float, qk_gain_init: float, bitnet_eps: float,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        assert logit_softcap > 0
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.num_unique_layers = num_unique_layers
        self.recurrence_factor = recurrence_factor
        self.use_gradient_checkpointing = use_gradient_checkpointing
        total_eff = num_unique_layers * recurrence_factor
        self.num_encoder_layers = total_eff // 2
        self.num_decoder_layers = total_eff - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, bitnet_eps)
            for _ in range(num_unique_layers)
        ])
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else BitLinear(model_dim, vocab_size, bias=False, eps=bitnet_eps)

        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x0 = x
        skips: list[Tensor] = []
        use_ckpt = self.use_gradient_checkpointing and self.training

        for eff_i in range(self.num_encoder_layers):
            block = self.blocks[eff_i % self.num_unique_layers]
            x = grad_checkpoint(block, x, x0, use_reentrant=False) if use_ckpt else block(x, x0)
            skips.append(x)
        for eff_i in range(self.num_decoder_layers):
            skip_i = self.num_encoder_layers + eff_i
            if skips and eff_i < self.num_skip_weights:
                x = x + self.skip_weights[eff_i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block = self.blocks[skip_i % self.num_unique_layers]
            x = grad_checkpoint(block, x, x0, use_reentrant=False) if use_ckpt else block(x, x0)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

        if target_ids is None:
            return logits  # [B, T, V] for sliding window eval
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1), reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.home() / ".cache" / "torchinductor"))
    torch._dynamo.config.cache_size_limit = 32
    use_compile = not bool(int(os.environ.get("NO_COMPILE", "0")))
    if use_compile:
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
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(True); enable_math_sdp(True)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            tqdm.write(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Vocab size mismatch: {int(sp.vocab_size())} vs {args.vocab_size}")

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device)

    total_eff = args.num_unique_layers * args.recurrence_factor
    head_dim = args.model_dim // args.num_heads

    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel()-1} stride_factor:{args.val_stride_factor}")
    log0(f"model_type:bitnet_b158 weight_quant:ternary act_quant:int8_absmax serialization:ternary_2bit")
    log0(f"depth_recurrence:unique_layers={args.num_unique_layers} recurrence={args.recurrence_factor} effective_depth={total_eff}")
    log0(f"seq_len_warmup:{args.seq_len_start}→{args.train_seq_len} over {args.seq_len_warmup_steps} steps (YaRN)")

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_unique_layers=args.num_unique_layers,
        recurrence_factor=args.recurrence_factor,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bitnet_eps=args.bitnet_eps,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, BitLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # For training, compile a version that always returns loss (target_ids is always provided)
    # For eval, use base_model directly (returns logits when target_ids=None)
    # fullgraph=True is incompatible with gradient checkpointing (graph breaks at checkpoint boundaries)
    compile_fullgraph = not args.use_gradient_checkpointing
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=compile_fullgraph) if use_compile else base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params
                     if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for name, p in block_named_params
                     if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    n_bitlinear = sum(p.numel() for n, m in base_model.named_modules()
                      if isinstance(m, BitLinear) for p in m.parameters())
    log0(f"model_params:{n_params} unique_bitlinear_params:{n_bitlinear}")
    log0(f"ternary_estimate:{n_bitlinear*2//8 + (n_params-n_bitlinear)*2} bytes before zlib")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"sdp_backends:cudnn=False flash=True mem_efficient=True math=True")
    log0(f"gradient_checkpointing:{args.use_gradient_checkpointing} ternary_commit_lambda:{args.ternary_commit_lambda}")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
         f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
         f"lr_warmup:{args.lr_warmup_steps} warmdown:{args.warmdown_iters} "
         f"min_lr_ratio:{args.min_lr_ratio} early_stop_patience:{args.early_stop_patience}")
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int) -> float:
        if args.lr_warmup_steps > 0 and step < args.lr_warmup_steps:
            return (step + 1) / args.lr_warmup_steps
        cooldown_start = args.iterations - args.warmdown_iters
        if args.warmdown_iters > 0 and step >= cooldown_start:
            progress = min((step - cooldown_start) / max(args.warmdown_iters, 1), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        return 1.0

    # Optimizer warmup (runs steps, then resets model weights but keeps optimizer state)
    if args.warmup_steps > 0:
        warmup_seq_len = get_current_seq_len(0, args)
        update_rotary_yarn(base_model, warmup_seq_len, args.train_seq_len, args.rope_base, head_dim)
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_opt_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for micro in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, warmup_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wloss = model(x, y)
                (wloss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_opt_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    last_val_bpb: float | None = None
    last_val_loss: float | None = None
    best_val_bpb = float("inf")
    patience_counter = 0
    early_stopped = False
    prev_seq_len = -1

    pbar = tqdm(total=args.iterations, desc="train", unit="step", disable=not master_process)
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        current_seq_len = get_current_seq_len(step, args)

        # Update YaRN-adjusted RoPE when seq_len changes
        if current_seq_len != prev_seq_len:
            update_rotary_yarn(base_model, current_seq_len, args.train_seq_len, args.rope_base, head_dim)
            if step > 0:
                log0(f"seq_len_warmup: step={step} seq_len={current_seq_len}")
            prev_seq_len = current_seq_len

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Always eval at full seq_len (reset YaRN to full base for eval)
            update_rotary_yarn(base_model, args.train_seq_len, args.train_seq_len, args.rope_base, head_dim)
            val_loss, val_bpb = eval_val(
                args, base_model, device, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                show_progress=master_process,
            )
            # Restore YaRN for training
            update_rotary_yarn(base_model, current_seq_len, args.train_seq_len, args.rope_base, head_dim)
            last_val_bpb = val_bpb
            last_val_loss = val_loss
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"seq_len:{current_seq_len} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")

            if step > 0:
                if val_bpb < best_val_bpb:
                    best_val_bpb = val_bpb
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.early_stop_patience:
                        log0(f"early_stop: patience={args.early_stop_patience} "
                             f"val_bpb={val_bpb:.4f} best={best_val_bpb:.4f}")
                        early_stopped = True
                        last_step = True

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            pbar.close()
            if stop_after_step is not None and step < args.iterations and not early_stopped:
                log0(f"stopping_early: wallclock train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        scale = lr_mul(step)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, current_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Ternary commitment loss: penalises distance between latent weights and nearest ternary value.
        # Computed once per step (not per micro-batch) directly on BitLinear weight tensors.
        if args.ternary_commit_lambda > 0:
            commit_loss = torch.zeros((), device=device)
            for m in base_model.modules():
                if isinstance(m, BitLinear):
                    w = m.weight  # float32
                    alpha = w.abs().mean().clamp(min=args.bitnet_eps)
                    w_norm = w / alpha
                    w_ternary = w_norm.clamp(-1, 1).round().detach()
                    commit_loss = commit_loss + (w_norm - w_ternary).pow(2).mean()
            (commit_loss * args.ternary_commit_lambda).backward()

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_mom
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        pbar.update(1)
        postfix: dict = {"loss": f"{train_loss.item():.4f}", "slen": str(current_seq_len),
                         "step_ms": f"{approx_ms/step:.0f}"}
        if last_val_bpb is not None:
            postfix["bpb"] = f"{last_val_bpb:.4f}"
        if last_val_loss is not None:
            postfix["val_loss"] = f"{last_val_loss:.4f}"
        pbar.set_postfix(postfix)

        should_log = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0)
        if should_log:
            extra = ""
            if last_val_bpb is not None:
                extra += f" val_bpb:{last_val_bpb:.4f} val_loss:{last_val_loss:.4f}"
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f}{extra} "
                 f"seq_len:{current_seq_len} train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            reached_cap = bool(t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB "
         f"reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")

    # Reset to full seq_len for final eval and serialization
    update_rotary_yarn(base_model, args.train_seq_len, args.train_seq_len, args.rope_base, head_dim)

    if master_process:
        torch.save(base_model.state_dict(), "final_model_bitnet.pt")
        log0(f"Serialized model (raw pt): {os.path.getsize('final_model_bitnet.pt')} bytes")

    # Ternary 2-bit serialization
    ternary_obj = serialize_ternary_model(base_model, eps=args.bitnet_eps)
    buf = io.BytesIO()
    torch.save(ternary_obj, buf)
    ternary_raw = buf.getvalue()
    ternary_blob = zlib.compress(ternary_raw, level=9)
    if master_process:
        with open("final_model_bitnet.ternary.ptz", "wb") as f:
            f.write(ternary_blob)
        ternary_bytes = os.path.getsize("final_model_bitnet.ternary.ptz")
        code_bytes = len(code.encode("utf-8"))
        total = ternary_bytes + code_bytes
        log0(f"Serialized model ternary+zlib: {ternary_bytes} bytes (raw_torch:{len(ternary_raw)})")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {total} bytes ({total/1e6:.2f} MB)")
        log0(f"Within 16MB limit: {total <= 16e6}")

    if distributed:
        dist.barrier()

    # Roundtrip check
    with open("final_model_bitnet.ternary.ptz", "rb") as f:
        blob = f.read()
    state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    deserialize_ternary_model(base_model, state)
    torch.cuda.synchronize()
    t_eval = time.perf_counter()
    q_loss, q_bpb = eval_val(args, base_model, device, val_tokens,
                              base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_ternary_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} "
         f"eval_time:{1000*(time.perf_counter()-t_eval):.0f}ms")
    log0(f"final_ternary_roundtrip_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
