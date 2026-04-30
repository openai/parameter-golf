"""
Combined depth-recurrent transformer fork for Parameter Golf.

Active feature set:
- #1 Depth-Recurrent Attention Blocks
- #2 Block Attention Residuals
- #4 Vertical Carry
- #8 Dynamic Teacher-Weighted Cross Entropy

Inactive / removed from the baseline:
- macro sidechannel
- encoder/decoder skip stack
- D2S patch path
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

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 250))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 5))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 393216))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    macro_interval = int(os.environ.get("MACRO_INTERVAL", 16))
    macro_xattn_dim = int(os.environ.get("MACRO_XATTN_DIM", 128))
    macro_gate_init = float(os.environ.get("MACRO_GATE_INIT", -2.5))
    macro_distill_weight = float(os.environ.get("MACRO_DISTILL_WEIGHT", 0.15))
    surprisal_weight_scale = float(os.environ.get("SURPRISAL_WEIGHT_SCALE", 1.0))
    vertical_carry_enabled = bool(int(os.environ.get("VERTICAL_CARRY", "1")))
    vertical_chunk_size = int(os.environ.get("VERTICAL_CHUNK_SIZE", 64))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    n_physical_blocks = int(os.environ.get("N_PHYSICAL_BLOCKS", 3))
    n_depth_iters = int(os.environ.get("N_DEPTH_ITERS", 4))
    block_size = int(os.environ.get("BLOCK_SIZE", 2))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 7))
    model_dim = int(os.environ.get("MODEL_DIM", 896))
    num_heads = int(os.environ.get("NUM_HEADS", 14))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qat_group_size = int(os.environ.get("QAT_GROUP_SIZE", 128))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.04))

    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))

    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 10240))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "0")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.4))
    swa_every = int(os.environ.get("SWA_EVERY", 50))

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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
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

            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
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
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
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
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
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


# -----------------------------
# POST-TRAINING QUANTIZATION (INT4 QAT + zlib)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,iter_embeds,resid_alpha,q_gain,block_attn,smear,bigram.scale,macro_gate,vertical_carry.gate",
    ).split(",")
    if pattern
)

# -----------------------------
# INT4 EXPORT QUANTIZATION
# -----------------------------

def quantize_weight_int4(weight: Tensor, group_size: int = 128) -> tuple[Tensor, Tensor]:
    """Quantize a float weight tensor to int4 with per-group scales."""
    w = weight.float().reshape(-1, group_size)
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / 7.0
    q = (w / scale).round().clamp(-8, 7).to(torch.int8)
    return q.reshape(weight.shape), scale.squeeze(-1).to(torch.float16)


def pack_int4(q: Tensor) -> Tensor:
    """Pack pairs of int4 values (-8..7) into single uint8 bytes. Halves storage."""
    q_uint = (q.to(torch.int16) + 8).to(torch.uint8)  # shift to [0..15]
    assert q_uint.shape[-1] % 2 == 0, f"cols must be even for int4 packing, got {q_uint.shape[-1]}"
    lo = q_uint[..., 0::2]
    hi = q_uint[..., 1::2]
    return (hi << 4 | lo).to(torch.uint8)


def unpack_int4(packed: Tensor, orig_cols: int) -> Tensor:
    """Unpack uint8 -> pairs of int4 values."""
    lo = (packed & 0x0F).to(torch.int8) - 8
    hi = (packed >> 4).to(torch.int8) - 8
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], orig_cols)


# Pattern-based QAT: any weight matching these suffixes gets int4 quantized.
QAT_WEIGHT_SUFFIXES = (
    "attn.c_q.weight", "attn.c_k.weight", "attn.c_v.weight", "attn.proj.weight",
    "mlp.w_gate.weight", "mlp.w_up.weight", "mlp.w_down.weight",
    "tok_emb.weight", "lm_head.weight",
    "macro_q.weight", "macro_k.weight", "macro_v.weight", "macro_pred.weight",
    "bigram.proj.weight", "vertical_carry.pool_proj.weight",
)


def quantize_state_dict_int4(state_dict: dict[str, Tensor], group_size: int = 128):
    """Quantize model state dict: int4 for QAT weights, fp16 for everything else."""
    int4_packed: dict[str, Tensor] = {}
    int4_scales: dict[str, Tensor] = {}
    int4_shapes: dict[str, list[int]] = {}
    int4_orig_cols: dict[str, int] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    stats = {"param_count": 0, "int4_bytes": 0, "passthrough_bytes": 0}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += int(t.numel())

        if any(name.endswith(s) for s in QAT_WEIGHT_SUFFIXES) and t.numel() % group_size == 0:
            q, scale = quantize_weight_int4(t, group_size)
            int4_orig_cols[name] = q.shape[-1]
            packed = pack_int4(q)
            int4_packed[name] = packed
            int4_scales[name] = scale
            int4_shapes[name] = list(t.shape)
            stats["int4_bytes"] += packed.numel() + scale.numel() * 2
        else:
            if t.is_floating_point() and t.dtype in (torch.float32, torch.bfloat16):
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                t = t.to(torch.float16)
            passthrough[name] = t
            stats["passthrough_bytes"] += int(t.numel()) * int(t.element_size())

    obj: dict[str, object] = {
        "__quant_format__": "int4_transgolf_v1",
        "int4_packed": int4_packed,
        "int4_scales": int4_scales,
        "int4_shapes": int4_shapes,
        "int4_orig_cols": int4_orig_cols,
        "int4_group_size": group_size,
        "passthrough": passthrough,
        "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    return obj, stats


def dequantize_state_dict_int4(obj: dict[str, object]) -> dict[str, Tensor]:
    """Dequantize int4 state dict back to float tensors."""
    group_size = obj["int4_group_size"]
    int4_orig_cols = obj.get("int4_orig_cols", {})
    out: dict[str, Tensor] = {}

    for name, packed in obj["int4_packed"].items():
        shape = obj["int4_shapes"][name]
        scale = obj["int4_scales"][name]
        orig_cols = int4_orig_cols.get(name, shape[-1])
        q = unpack_int4(packed, orig_cols).reshape(-1, group_size)
        scale_f = scale.float().unsqueeze(-1)
        w = (q.float() * scale_f).reshape(shape)
        out[name] = w.to(torch.bfloat16)

    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t

    return out


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
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
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


def fake_quantize_4bit(weight: Tensor, group_size: int = 128) -> Tensor:
    """Fake-quantize weight to 4-bit with per-group scales. STE for backward."""
    orig_shape = weight.shape
    w = weight.reshape(-1, group_size)
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / 7.0
    w_q = (w / scale).round().clamp(-8, 7)
    w_dq = (w_q * scale).reshape(orig_shape)
    return weight + (w_dq - weight).detach()

class QATLinear(nn.Linear):
    """Linear layer with 4-bit fake quantization (STE) during training."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, group_size: int = 128):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size

    def forward(self, x: Tensor) -> Tensor:
        w = fake_quantize_4bit(self.weight, self.group_size)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


def collect_matrix_params(base_model: "GPT") -> list[nn.Parameter]:
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.bigram is not None and base_model.bigram.proj is not None:
        matrix_params.append(base_model.bigram.proj.weight)
    matrix_params.extend([
        base_model.macro_q.weight,
        base_model.macro_k.weight,
        base_model.macro_v.weight,
        base_model.macro_pred.weight,
    ])
    if base_model.vertical_carry is not None:
        matrix_params.append(base_model.vertical_carry.pool_proj.weight)
    return matrix_params


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
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


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float, group_size: int = 128):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = QATLinear(dim, dim, bias=False, group_size=group_size)
        self.c_k = QATLinear(dim, kv_dim, bias=False, group_size=group_size)
        self.c_v = QATLinear(dim, kv_dim, bias=False, group_size=group_size)
        self.proj = QATLinear(dim, dim, bias=False, group_size=group_size)
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
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float, group_size: int = 128):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.w_gate = QATLinear(dim, hidden, bias=False, group_size=group_size)
        self.w_up = QATLinear(dim, hidden, bias=False, group_size=group_size)
        self.w_down = QATLinear(hidden, dim, bias=False, group_size=group_size)
        self.w_down._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SmearGate(nn.Module):
    """Blend each token's embedding with the previous token's embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int, group_size: int = 128):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = QATLinear(bigram_dim, model_dim, bias=False, group_size=group_size) if bigram_dim != model_dim else None
        if self.proj is not None:
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
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class BlockAttentionResidual(nn.Module):
    """Inter-block attention over completed block summaries."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.block_proj = nn.Linear(model_dim, 1, bias=False)
        self.block_norm = RMSNorm()

    @torch.compiler.disable
    def forward(self, blocks: list[Tensor], partial_block: Tensor) -> Tensor:
        sources = [*blocks, partial_block]
        scores = torch.stack(
            [self.block_proj(self.block_norm(src)).squeeze(-1) for src in sources],
            dim=0,
        )
        attn = torch.softmax(scores, dim=0)
        attended = torch.zeros_like(partial_block)
        for idx, src in enumerate(sources):
            attended = attended + attn[idx].unsqueeze(-1) * src
        return attended


class VerticalCarry(nn.Module):
    """Carry per-chunk summaries between depth iterations without leaking future tokens."""

    def __init__(self, model_dim: int, chunk_size: int = 64):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = chunk_size
        self.pool_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.pool_proj._zero_init = True
        self.gate = nn.Parameter(torch.full((model_dim,), -2.5, dtype=torch.float32))

    def summarize(self, x: Tensor) -> Tensor:
        batch, seqlen, dim = x.shape
        if seqlen % self.chunk_size != 0:
            raise ValueError(
                f"sequence length {seqlen} must be divisible by vertical chunk size {self.chunk_size}"
            )
        summaries = x.reshape(batch, seqlen // self.chunk_size, self.chunk_size, dim)[:, :, -1, :]
        return self.pool_proj(summaries)

    def inject(self, x: Tensor, carry: Tensor) -> Tensor:
        batch, seqlen, dim = x.shape
        if seqlen % self.chunk_size != 0:
            raise ValueError(
                f"sequence length {seqlen} must be divisible by vertical chunk size {self.chunk_size}"
            )
        n_chunks = seqlen // self.chunk_size
        expected_shape = (batch, n_chunks, dim)
        if tuple(carry.shape) != expected_shape:
            raise ValueError(f"carry must have shape {expected_shape}, got {tuple(carry.shape)}")
        carry = carry.to(device=x.device, dtype=x.dtype)
        shifted = torch.cat([x.new_zeros(batch, 1, dim), carry[:, :-1, :]], dim=1)
        context = shifted.unsqueeze(2).expand(-1, -1, self.chunk_size, -1).reshape(batch, seqlen, dim)
        gate = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        return x + gate * context


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float, rope_base: float, qk_gain_init: float, group_size: int = 128):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, group_size=group_size)
        self.mlp = MLP(dim, mlp_mult, group_size=group_size)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor, iter_cond: Tensor) -> Tensor:
        del x0
        s1, b1, s2, b2 = iter_cond
        h = self.attn_norm(x)
        h = h * (1.0 + s1.to(h.dtype)) + b1.to(h.dtype)
        attn_out = self.attn(h)
        h = self.mlp_norm(x)
        h = h * (1.0 + s2.to(h.dtype)) + b2.to(h.dtype)
        mlp_out = self.mlp(h)
        return (
            x
            + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        )


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_physical_blocks: int,
        n_depth_iters: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        block_size: int = 2,
        macro_interval: int = 16,
        macro_xattn_dim: int = 128,
        macro_gate_init: float = -2.5,
        macro_distill_weight: float = 0.15,
        surprisal_weight_scale: float = 1.0,
        vertical_carry_enabled: bool = True,
        vertical_chunk_size: int = 64,
        group_size: int = 128,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if n_physical_blocks <= 0:
            raise ValueError(f"n_physical_blocks must be positive, got {n_physical_blocks}")
        if n_depth_iters <= 0:
            raise ValueError(f"n_depth_iters must be positive, got {n_depth_iters}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if macro_interval <= 0:
            raise ValueError(f"macro_interval must be positive, got {macro_interval}")
        if macro_xattn_dim <= 0:
            raise ValueError(f"macro_xattn_dim must be positive, got {macro_xattn_dim}")
        if surprisal_weight_scale < 0.0:
            raise ValueError(f"surprisal_weight_scale must be non-negative, got {surprisal_weight_scale}")
        if vertical_chunk_size <= 0:
            raise ValueError(f"vertical_chunk_size must be positive, got {vertical_chunk_size}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.n_physical_blocks = n_physical_blocks
        self.n_depth_iters = n_depth_iters
        self.block_size = block_size
        self.macro_interval = macro_interval
        self.macro_distill_weight = macro_distill_weight
        self.macro_xattn_dim = min(macro_xattn_dim, model_dim)
        self.surprisal_weight_scale = surprisal_weight_scale
        self.vertical_carry_enabled = vertical_carry_enabled
        self.vertical_chunk_size = vertical_chunk_size
        self.group_size = group_size
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim, group_size=group_size) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.macro_q = QATLinear(model_dim, self.macro_xattn_dim, bias=False, group_size=group_size)
        self.macro_k = QATLinear(model_dim, self.macro_xattn_dim, bias=False, group_size=group_size)
        self.macro_v = QATLinear(model_dim, model_dim, bias=False, group_size=group_size)
        self.macro_pred = QATLinear(model_dim, model_dim, bias=False, group_size=group_size)
        self.macro_pred._zero_init = True
        self.macro_gate = nn.Parameter(torch.full((model_dim,), macro_gate_init, dtype=torch.float32))
        self.block_attn = BlockAttentionResidual(model_dim)
        self.vertical_carry = (
            VerticalCarry(model_dim, chunk_size=vertical_chunk_size) if vertical_carry_enabled else None
        )
        self.blocks = nn.ModuleList(
            [
                Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, group_size=group_size)
                for _ in range(n_physical_blocks)
            ]
        )
        self.iter_embeds = nn.Parameter(torch.randn(n_depth_iters, 4, model_dim) * 0.001)
        self.resid_alpha = nn.Parameter(torch.zeros(n_depth_iters))
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else QATLinear(model_dim, vocab_size, bias=False, group_size=group_size)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * self.n_depth_iters))

    def _run_depth_groups(
        self,
        x: Tensor,
        x0: Tensor,
        start_iter: int,
        end_iter: int,
        completed_blocks: list[Tensor],
    ) -> Tensor:
        carry = None
        iter_idx = start_iter
        while iter_idx < end_iter:
            if len(completed_blocks) > 1:
                x = self.block_attn(completed_blocks, x)
                carry = None
            group_len = min(self.block_size, end_iter - iter_idx)
            block_sum = torch.zeros_like(x)
            for _ in range(group_len):
                if self.vertical_carry is not None and carry is not None:
                    x = self.vertical_carry.inject(x, carry)
                alpha = torch.sigmoid(self.resid_alpha[iter_idx]).to(dtype=x.dtype)
                x_in = alpha * x + (1.0 - alpha) * x0
                block = self.blocks[iter_idx % self.n_physical_blocks]
                x = block(x_in, x0, self.iter_embeds[iter_idx])
                if self.vertical_carry is not None:
                    carry = self.vertical_carry.summarize(x)
                block_sum = block_sum + x
                iter_idx += 1
            completed_blocks.append(block_sum / group_len)
        return x

    def _apply_macro_sidechannel(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Causal interval sidechannel with a leaky teacher for surprisal weighting."""
        B, L, D = x.shape
        interval = self.macro_interval
        if L % interval != 0:
            raise ValueError(f"sequence length {L} must be divisible by macro_interval {interval}")
        n_intervals = L // interval

        raw_summaries = x.reshape(B, n_intervals, interval, D)[:, :, -1, :].contiguous()

        teacher = None
        need_teacher = self.training and (
            self.macro_distill_weight > 0.0 or self.surprisal_weight_scale > 0.0
        )
        if need_teacher:
            with torch.no_grad():
                teacher = raw_summaries + self.macro_pred(F.rms_norm(raw_summaries, (D,)))

        shifted = torch.cat([
            torch.zeros(B, 1, D, device=x.device, dtype=x.dtype),
            raw_summaries[:, :-1, :],
        ], dim=1)
        student = shifted + self.macro_pred(F.rms_norm(shifted, (D,)))

        token_interval = torch.arange(L, device=x.device) // interval
        macro_pos = torch.arange(n_intervals, device=x.device)
        mask = macro_pos.unsqueeze(0) < token_interval.unsqueeze(1)

        q = self.macro_q(x)
        k = self.macro_k(student)
        v = self.macro_v(student)
        scores = torch.bmm(q, k.transpose(-1, -2)) * (self.macro_xattn_dim ** -0.5)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        context = torch.bmm(attn, v)
        x = x + torch.sigmoid(self.macro_gate.to(dtype=x.dtype))[None, None, :] * context

        distill_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        interval_surprisal = None
        if teacher is not None and n_intervals > 1:
            cos_sim = F.cosine_similarity(
                student[:, 1:, :].float(),
                teacher[:, 1:, :].detach().float(),
                dim=-1,
            )
            interval_surprisal = 1.0 - cos_sim
            distill_loss = interval_surprisal.mean()
        return x, distill_loss, interval_surprisal

    def _forward_impl(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        emb_w = fake_quantize_4bit(self.tok_emb.weight, self.group_size)
        x = F.embedding(input_ids, emb_w)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        completed_blocks = [x0]
        macro_split = max(1, self.n_depth_iters // 2)
        x = self._run_depth_groups(x, x0, 0, macro_split, completed_blocks)
        x, distill_loss, interval_surprisal = self._apply_macro_sidechannel(x)
        x = self._run_depth_groups(x, x0, macro_split, self.n_depth_iters, completed_blocks)
        x = self.block_attn(completed_blocks, x)
        return x, distill_loss, interval_surprisal

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x, distill_loss, interval_surprisal = self._forward_impl(input_ids)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        batch_size, seqlen = target_ids.shape
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, fake_quantize_4bit(self.tok_emb.weight, self.group_size))
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        ce_loss_unreduced = F.cross_entropy(logits.float(), targets, reduction="none").reshape(batch_size, seqlen)
        if self.training and self.surprisal_weight_scale > 0.0 and interval_surprisal is not None:
            surprisal_padded = torch.cat([
                torch.ones(batch_size, 1, device=logits.device, dtype=torch.float32),
                1.0 + self.surprisal_weight_scale * interval_surprisal,
            ], dim=1)
            dynamic_weights = surprisal_padded.repeat_interleave(self.macro_interval, dim=1)
            dynamic_weights = dynamic_weights[:, :seqlen]
            dynamic_weights = dynamic_weights / dynamic_weights.mean(dim=1, keepdim=True).clamp_min(1e-6)
            ce_loss = (ce_loss_unreduced * dynamic_weights.detach().to(ce_loss_unreduced.dtype)).mean()
        else:
            ce_loss = ce_loss_unreduced.mean()
        if self.training and self.macro_distill_weight > 0.0:
            return ce_loss + self.macro_distill_weight * distill_loss
        return ce_loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x, _, _ = self._forward_impl(input_ids)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, fake_quantize_4bit(self.tok_emb.weight, self.group_size))
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
            if rank == 0 and (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(my_windows))
                pct = done / len(my_windows) * 100
                running_bpb = 0.0
                if token_count.item() > 0:
                    rl = (loss_sum / token_count).item()
                    running_bpb = rl / math.log(2.0) * (token_count.item() / byte_count.item())
                print(f"  sliding_eval [{pct:5.1f}%] {done}/{len(my_windows)} windows running_bpb={running_bpb:.6f}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if args.train_seq_len % args.macro_interval != 0:
        raise ValueError(
            f"TRAIN_SEQ_LEN={args.train_seq_len} must be divisible by MACRO_INTERVAL={args.macro_interval}"
        )
    if args.vertical_carry_enabled and args.train_seq_len % args.vertical_chunk_size != 0:
        raise ValueError(
            f"TRAIN_SEQ_LEN={args.train_seq_len} must be divisible by VERTICAL_CHUNK_SIZE={args.vertical_chunk_size}"
        )
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
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
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

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
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # MODEL + OPTIMIZER SETUP
    base_model = GPT(
        vocab_size=args.vocab_size,
        n_physical_blocks=args.n_physical_blocks,
        n_depth_iters=args.n_depth_iters,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        block_size=args.block_size,
        macro_interval=args.macro_interval,
        macro_xattn_dim=args.macro_xattn_dim,
        macro_gate_init=args.macro_gate_init,
        macro_distill_weight=args.macro_distill_weight,
        surprisal_weight_scale=args.surprisal_weight_scale,
        vertical_carry_enabled=args.vertical_carry_enabled,
        vertical_chunk_size=args.vertical_chunk_size,
        group_size=args.qat_group_size,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, QATLinear):
            module.float()
    base_model.tok_emb.weight.data = base_model.tok_emb.weight.data.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = collect_matrix_params(base_model)
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params.append(base_model.iter_embeds)
    scalar_params.append(base_model.resid_alpha)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    scalar_params.append(base_model.macro_gate)
    scalar_params.append(base_model.block_attn.block_proj.weight)
    if base_model.vertical_carry is not None:
        scalar_params.append(base_model.vertical_carry.gate)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})

    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=0.04,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(
        f"model:TransformGolfR1248 d={args.model_dim} n_physical_blocks={args.n_physical_blocks} "
        f"n_depth_iters={args.n_depth_iters} block_size:{args.block_size} "
        f"vertical_carry:{args.vertical_carry_enabled} vertical_chunk_size:{args.vertical_chunk_size}"
    )
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(
        "block_schedule:depth_recurrent shared_blocks:True residual_highway:True "
        "adaln:True block_attention_residuals:True"
    )
    log0("compile_fullgraph:False block_attention_graph_break:True")
    log0(
        f"macro_sidechannel:causal_teacher_student macro_interval:{args.macro_interval} "
        f"macro_xattn_dim:{args.macro_xattn_dim} macro_distill_weight:{args.macro_distill_weight:.3f} "
        f"surprisal_weight_scale:{args.surprisal_weight_scale:.3f}"
    )
    log0("d2s_enabled:False")
    log0(f"seed:{args.seed}")

    # DATA LOADER & MODEL WARMUP
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
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
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # MAIN TRAINING LOOP
    training_time_ms = 0.0
    stop_after_step: int | None = None
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
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
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
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

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # SWA: collect checkpoints during warmdown
        if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1

        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            tokens_seen = step * args.train_batch_tokens
            tok_s = tokens_seen / max(approx_training_time_ms / 1000.0, 1e-9)
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"tok/s:{tok_s:.0f}"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    log0(f"total_tokens_seen:{step * args.train_batch_tokens}")

    # Apply SWA if collected
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"swa:applying averaged {swa_count} checkpoints")
        current_state = base_model.state_dict()
        avg_state = {
            name: (tensor / swa_count).to(dtype=current_state[name].dtype)
            for name, tensor in swa_state.items()
        }
        base_model.load_state_dict(avg_state, strict=True)

    # SERIALIZATION + ROUNDTRIP VALIDATION
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model (raw): {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")

    # INT4 quantization + zlib export
    quant_obj, quant_stats = quantize_state_dict_int4(base_model.state_dict(), args.qat_group_size)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    if master_process:
        with open("final_model.int4.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int4.ptz")
        code_bytes = len(code.encode("utf-8"))
        log0(
            f"Serialized model int4+zlib: {quant_file_bytes} bytes "
            f"(int4:{quant_stats['int4_bytes']} passthrough:{quant_stats['passthrough_bytes']})"
        )
        log0(f"Total submission size int4+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int4.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu", weights_only=True)
    base_model.load_state_dict(dequantize_state_dict_int4(quant_state), strict=True)
    for module in base_model.modules():
        if isinstance(module, QATLinear):
            module.float()
    base_model.tok_emb.weight.data = base_model.tok_emb.weight.data.float()
    restore_low_dim_params_to_fp32(base_model)

    # Sliding window eval on int4-roundtripped weights
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        log0(f"final_eval_mode:sliding_window stride:{args.eval_stride} batch_seqs:{args.eval_batch_seqs}")
        q_val_loss, q_val_bpb = eval_val_sliding(
            args, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, batch_seqs=args.eval_batch_seqs,
        )
    else:
        log0("final_eval_mode:standard")
        q_val_loss, q_val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
    torch.cuda.synchronize()
    log0(
        f"final_int4_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int4_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
