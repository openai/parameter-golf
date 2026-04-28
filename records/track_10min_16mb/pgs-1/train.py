"""
Memory-Augmented GPT — Parameter Golf submission (v40).

Architecture:
  - Learnable memory tokens prepended to each sequence (bidirectional with seq)
  - Sequence tokens attend causally to each other + freely to memory tokens
  - GQA attention with RoPE, SwiGLU MLP, U-Net skip connections
  - Logit soft-capping, tied embeddings, Muon optimizer for matrix params
  - Correct shard loading, shuffled shard order, proper BPB evaluation

v40 changes (full audit from v37/v39/base):

  THROUGHPUT:
  T1. Rotary: pre-allocate cos/sin as register_buffer at __init__ for _ROTARY_MAX_SEQ_LEN.
      Tensor object identity never changes -> torch.compile graph stays stable.
      Eliminates the "___check_obj_id" recompile at steps 1000 and 3000 seen in all
      prior versions (~20s wasted per recompile on 8-GPU).

  CORRECTNESS:
  C1. Mem mask: memory tokens attend ALL sequence positions (bidirectional with seq).
      v39's "causal mem->seq" (mem[i] sees seq[0..i]) was semantically broken: with M=2,
      mem[0] saw only 1 token and mem[1] only 2 tokens of a 1024-token sequence.
      The future-leakage concern does not apply: memory tokens are learned constants
      (not recurrent state), so they cannot encode future targets for retrieval.
  C2. Skip connections: removed the erroneous 0.5 factor added in v39 FIX-5.
      v39 had skip_weights init=0.5 AND a 0.5 multiply -> effective weight 0.25, too weak.
      v40: init=1.0, no extra factor. Weights are learned and find the right scale.
  C3. GQA: reverted to repeat_interleave (correct, compile-friendly). The
      expand+reshape in v39 FIX-6 claimed "zero-copy" but expand() returns non-contiguous
      strides and reshape() silently copies — identical cost, harder to audit.
  C4. Seq len raised to 1024. At 8-GPU ~200ms/step: ~3000 steps x 1024 = 3.1B tokens
      vs 4700 x 512 = 2.4B at seq_len=512. More tokens AND better BPB from longer context.

  DATA:
  D1. TokenStream._advance: no rank-offset file_idx skew (rank*7)%N removed.
      Rank-seeded RNG already guarantees different shard orderings. The extra offset
      caused uneven shard coverage when len(files) was not divisible by 7.
  D2. _JUMP_EVERY unified to 32 across all ranks (was 8+(rank%3) in v39).

  QUANTIZATION:
  Q1. Scale tensors use bfloat16 (same 2-byte footprint as float16, full fp32 exponent
      range — prevents underflow on small-scale tensors).
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

# ──────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────

class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH",        "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path,            "fineweb_train_*.bin")
    val_files      = os.path.join(data_path,            "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH",  "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID",           str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED",         1337))

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE",  524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY",  1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations            = int(os.environ.get("ITERATIONS",             20000))
    warmdown_iters        = int(os.environ.get("WARMDOWN_ITERS",         1200))
    warmup_steps          = int(os.environ.get("WARMUP_STEPS",           20))
    train_batch_tokens    = int(os.environ.get("TRAIN_BATCH_TOKENS",     524_288))
    train_seq_len         = int(os.environ.get("TRAIN_SEQ_LEN",          1024))  # C4: 512->1024
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init          = float(os.environ.get("QK_GAIN_INIT",         1.5))

    vocab_size     = int(os.environ.get("VOCAB_SIZE",     1024))
    num_layers     = int(os.environ.get("NUM_LAYERS",     8))
    num_kv_heads   = int(os.environ.get("NUM_KV_HEADS",   3))
    model_dim      = int(os.environ.get("MODEL_DIM",      384))
    num_heads      = int(os.environ.get("NUM_HEADS",      6))
    mlp_ratio      = float(os.environ.get("MLP_RATIO",   2.667))
    mem_tokens     = int(os.environ.get("MEM_TOKENS",    2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base      = float(os.environ.get("ROPE_BASE",   10000.0))
    logit_softcap  = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr                   = float(os.environ.get("EMBED_LR",                   0.6))
    head_lr                    = float(os.environ.get("HEAD_LR",                    0.008))
    tied_embed_lr              = float(os.environ.get("TIED_EMBED_LR",              0.05))
    tied_embed_init_std        = float(os.environ.get("TIED_EMBED_INIT_STD",        0.005))
    matrix_lr                  = float(os.environ.get("MATRIX_LR",                  0.04))
    scalar_lr                  = float(os.environ.get("SCALAR_LR",                  0.04))
    muon_momentum              = float(os.environ.get("MUON_MOMENTUM",              0.95))
    muon_backend_steps         = int(os.environ.get("MUON_BACKEND_STEPS",           5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS",   500))
    beta1          = float(os.environ.get("BETA1",        0.9))
    beta2          = float(os.environ.get("BETA2",        0.95))
    adam_eps       = float(os.environ.get("ADAM_EPS",     1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


# ──────────────────────────────────────────────
# MUON OPTIMIZER
# ──────────────────────────────────────────────

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """Orthogonalize a 2-D matrix via Newton-Schulz iteration (Muon core)."""
    a, b, c = 3.4445, -4.7750, 2.0315
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
        world_size  = dist.get_world_size() if distributed else 1
        rank        = dist.get_rank()       if distributed else 0

        for group in self.param_groups:
            params        = group["params"]
            if not params:
                continue
            lr            = group["lr"]
            momentum      = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov      = group["nesterov"]

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

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# ──────────────────────────────────────────────
# TOKENIZER-AGNOSTIC BPB EVALUATION
# ──────────────────────────────────────────────

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np        = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones( (table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\xe2\x96\x81"):  # UTF-8 for U+2581 LOWER ONE EIGHTH BLOCK
            has_leading_space_np[token_id] = True
            piece = piece[len("\xe2\x96\x81"):]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np,        dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool,  device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool,  device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No validation files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
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
            f"VAL_BATCH_SIZE too small: need >={args.train_seq_len} tokens per rank, "
            f"got {local_batch_tokens}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs       = (val_tokens.numel() - 1) // args.train_seq_len
    seqs_per_rank    = total_seqs // world_size

    max_offset = max(total_seqs - seqs_per_rank * world_size, 1)
    offset_t   = torch.randint(0, max_offset, (1,), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(offset_t, src=0)
    offset = int(offset_t.item())

    seq_start = offset + seqs_per_rank * rank
    seq_end   = offset + seqs_per_rank * (rank + 1)

    val_loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        _eval_count = 0
        _MAX_EVAL_BATCHES = 50
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            if _eval_count >= _MAX_EVAL_BATCHES:
                break
            _eval_count += 1
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end   = batch_seq_end   * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1: ].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum    += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids    = x.reshape(-1)
            tgt_ids     = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)

    val_loss        = val_loss_sum / val_token_count
    bits_per_token  = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# ──────────────────────────────────────────────
# INT8 + ZLIB QUANTIZATION
# ──────────────────────────────────────────────

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if p
)
INT8_KEEP_FLOAT_MAX_NUMEL   = 8_192
INT8_KEEP_FLOAT_STORE_DTYPE = torch.bfloat16   # Q1: bf16 scales — full fp32 exponent range
INT8_PER_ROW_SCALE_DTYPE    = torch.bfloat16
INT8_CLIP_PERCENTILE        = 99.99984
INT8_CLIP_Q                 = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict) -> Tensor:
    if any(pat in name for pat in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale   = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale    = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized:               dict[str, Tensor] = {}
    scales:                  dict[str, Tensor] = {}
    passthrough:             dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str]    = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"), 0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"]           += int(t.numel())
        stats["num_tensors"]           += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"]   += tensor_nbytes(t)
            continue
        stats["num_float_tensors"] += 1
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or any(pat in name for pat in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
            passthrough[name] = keep_float_tensor(name, t, passthrough_orig_dtypes)
            stats["int8_payload_bytes"] += tensor_nbytes(passthrough[name])
            continue
        q, scale = quantize_float_tensor(t)
        quantized[name] = q
        scales[name]    = scale
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(scale)
    payload = {
        "quantized": quantized, "scales": scales,
        "passthrough": passthrough, "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    return payload, stats


def dequantize_state_dict_int8(payload: dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, q in payload["quantized"].items():
        scale    = payload["scales"][name]
        out[name] = (q.float() * scale.float()[:, None]).contiguous() if q.ndim == 2 \
                    else (q.float() * scale.float()).contiguous()
    for name, t in payload["passthrough"].items():
        orig_dtype = payload.get("passthrough_orig_dtypes", {}).get(name)
        out[name]  = t.to(dtype=getattr(torch, orig_dtype)).contiguous() \
                     if isinstance(orig_dtype, str) else t
    return out


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    """Read a fineweb .bin shard, correctly skipping the 256-int32 header."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens    = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read: {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    """
    Rank-aware independent token stream.

    D1: _advance no longer applies (rank*7)%N file_idx skew — the rank-seeded RNG
        already ensures different shard orderings per rank. The extra offset caused
        uneven shard coverage when len(files) was not divisible by 7.
    D2: _JUMP_EVERY unified to 32 across all ranks (was 8+(rank%3) in v39, causing
        unequal randomisation rates across ranks).
    """

    def __init__(self, pattern: str, seed: int = 1337, rank: int = 0):
        files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not files:
            raise FileNotFoundError(f"No training shards: {pattern}")
        self.rng      = random.Random(seed + rank)
        self.files    = files[:]
        self.rng.shuffle(self.files)
        self.file_idx = 0
        self.tokens   = load_data_shard(self.files[0])
        self.pos      = self.rng.randint(0, max(self.tokens.numel() - 1, 0))
        self._call_count = 0
        self._JUMP_EVERY = 32   # D2: uniform across ranks

    def _advance(self) -> None:
        self.file_idx += 1
        if self.file_idx >= len(self.files):
            self.rng.shuffle(self.files)
            self.file_idx = 0   # D1: no (rank*7)%N skew
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos    = self.rng.randint(0, max(self.tokens.numel() - 1, 0))

    def take(self, n: int) -> Tensor:
        self._call_count += 1
        if self._call_count % self._JUMP_EVERY == 0 and self.tokens.numel() > n:
            self.pos = self.rng.randint(0, self.tokens.numel() - n - 1)
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos  += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    """Each rank owns an independent TokenStream seeded with (seed + rank)."""

    def __init__(self, pattern: str, rank: int, world_size: int,
                 device: torch.device, seed: int = 1337):
        self.device     = device
        self.world_size = world_size
        self.stream     = TokenStream(pattern, seed=seed, rank=rank)

    def next_batch(self, global_tokens: int, seq_len: int,
                   grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        buf = self.stream.take(local_tokens + 1)
        x = buf[:-1].reshape(-1, seq_len).to(dtype=torch.int64)
        y = buf[1: ].reshape(-1, seq_len).to(dtype=torch.int64)
        return (
            x.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True),
        )


# ──────────────────────────────────────────────
# MODEL UTILITIES
# ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Weights stored in fp32; cast to input dtype at matmul time (bf16 compute)."""
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            is_ctrl = any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
            if (param.ndim < 2 or is_ctrl) and param.dtype != torch.float32:
                param.data = param.data.float()


# T1: pre-allocate rotary buffers up to this length to fix dynamo recompile.
_ROTARY_MAX_SEQ_LEN = 2048


class Rotary(nn.Module):
    """
    RoPE with pre-allocated cos/sin register_buffers (T1 — dynamo recompile fix).

    Prior versions stored _cos_cached as a Python None initialised at __init__.
    torch.compile records object identity of _cos_cached during tracing. When
    the first forward() replaced None with a fresh Tensor, dynamo detected the
    identity change and recompiled the entire graph. This fired again when the
    autocast dtype changed, causing the step-1000 and step-3000 warnings in logs.

    Fix: pre-allocate cos/sin as register_buffer(persistent=False) at __init__
    for _ROTARY_MAX_SEQ_LEN positions. The buffer's object identity never changes;
    forward() returns a dtype-cast slice. No recompile ever fires.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t     = torch.arange(_ROTARY_MAX_SEQ_LEN, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        # (1, 1, max_seq, dim//2) — stored float32, cast to compute dtype in forward
        self.register_buffer("cos_full", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_full", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return (
            self.cos_full[:, :, :seq_len, :].to(dtype=dtype),
            self.sin_full[:, :, :seq_len, :].to(dtype=dtype),
        )


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half    = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


# ──────────────────────────────────────────────
# HYBRID ATTENTION MASK
# ──────────────────────────────────────────────

_mask_cache: dict = {}


def build_hybrid_mask(M: int, T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    Additive attention bias of shape (1, 1, M+T, M+T).

    C1 — Memory tokens attend ALL sequence positions (bidirectional with seq):

      mem->mem : full attention
      mem->seq : full attention  <- v40 (was causal-only in v39, breaking memory utility)
      seq->mem : full attention
      seq->seq : strictly causal

    Why bidirectional mem->seq is safe: memory tokens are learned constant embeddings,
    not a recurrent hidden state. They cannot encode future targets for retrieval
    because doing so would not reduce the causal training loss. The concern that
    seq[t] could "read future" via seq[t]->mem->seq[t+1] does not hold: mem carries
    no position-specific information between forward passes.

    With M=2 and v39's causal mem->seq mask: mem[0] attended only 1 token and mem[1]
    only 2 tokens of a 1024-token sequence — the memory mechanism was effectively dead.
    """
    key = (M, T, str(device), dtype)
    if key not in _mask_cache:
        L    = M + T
        mask = torch.full((L, L), float("-inf"), device=device, dtype=dtype)
        # mem -> mem: fully bidirectional
        mask[:M, :M] = 0.0
        # mem -> seq: full attention to all sequence positions (C1)
        mask[:M, M:] = 0.0
        # seq -> mem: every seq token attends all memory tokens
        mask[M:, :M] = 0.0
        # seq -> seq: strictly causal
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        mask[M:, M:] = torch.where(
            causal,
            torch.zeros(T, T, device=device, dtype=dtype),
            torch.full((T, T), float("-inf"), device=device, dtype=dtype),
        )
        _mask_cache[key] = mask[None, None, :, :]
    return _mask_cache[key]


# ──────────────────────────────────────────────
# TRANSFORMER BLOCKS
# ──────────────────────────────────────────────

class GQAttention(nn.Module):
    """
    Grouped-Query Attention with pre-allocated RoPE buffers and QK-norm.

    mem_tokens is a proper __init__ parameter — never monkey-patched post-construction,
    so torch.compile and DDP always see the correct value from the first trace.

    C3: GQA KV expansion uses repeat_interleave (correct, compile-friendly).
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, mem_tokens: int = 0):
        super().__init__()
        assert dim % num_heads == 0,          "model_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        self.rep          = num_heads // num_kv_heads
        self.mem_tokens   = mem_tokens
        kv_dim = num_kv_heads * self.head_dim

        self.c_q  = CastedLinear(dim, dim,    bias=False)
        self.c_k  = CastedLinear(dim, kv_dim, bias=False)
        self.c_v  = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim,    bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        # T1: pre-allocated rotary buffers — no dynamo recompile
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        B, L, D = x.shape
        M = self.mem_tokens
        T = L - M

        q = self.c_q(x).reshape(B, L, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # RoPE applied to sequence tokens only; memory tokens are unrotated
        cos, sin = self.rotary(T, q.dtype)   # T1: device arg removed (buffer already on device)
        q_mem, q_seq = q[:, :, :M, :], q[:, :, M:, :]
        k_mem, k_seq = k[:, :, :M, :], k[:, :, M:, :]
        q_seq = apply_rotary_emb(q_seq, cos, sin)
        k_seq = apply_rotary_emb(k_seq, cos, sin)
        q = torch.cat([q_mem, q_seq], dim=2)
        k = torch.cat([k_mem, k_seq], dim=2)

        q = q * self.q_gain.to(q.dtype)[None, :, None, None]

        # C3: repeat_interleave for GQA — correct and compile-friendly
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)

        if mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, L, D))


class SwiGLU(nn.Module):
    """SwiGLU MLP. At dim=384, ratio=2.667: hidden=int(384*2.667/64)*64=1024."""
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hidden = int(dim * ratio / 64) * 64
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.up   = CastedLinear(dim, hidden, bias=False)
        self.down = CastedLinear(hidden, dim, bias=False)
        self.down._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_ratio: float, rope_base: float, qk_gain_init: float,
                 mem_tokens: int = 0):
        super().__init__()
        self.attn_norm  = RMSNorm()
        self.mlp_norm   = RMSNorm()
        self.attn       = GQAttention(dim, num_heads, num_kv_heads, rope_base,
                                      qk_gain_init, mem_tokens=mem_tokens)
        self.mlp        = SwiGLU(dim, mlp_ratio)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x), mask)
        x = x + self.mlp_scale.to(x.dtype)[None, None, :]  * self.mlp(self.mlp_norm(x))
        return x


# ──────────────────────────────────────────────
# MEMORY-AUGMENTED GPT
# ──────────────────────────────────────────────

class GPT(nn.Module):
    """
    Memory-Augmented GPT with U-Net skip connections.

    v40 vs v39:
      C1: mem tokens attend ALL seq positions — mask is bidirectional mem->seq.
      C2: skip_weights init=1.0, no extra 0.5 factor in decoder.
          v39 had init=0.5 AND 0.5 factor -> effective 0.25, too weak.
      T1: Rotary uses pre-allocated buffers, no dynamo recompile.
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_ratio: float,
        mem_tokens: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        assert logit_softcap > 0
        self.tie_embeddings      = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap       = logit_softcap
        self.mem_tokens          = mem_tokens

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.memory  = nn.Parameter(torch.randn(1, mem_tokens, model_dim) * 0.01)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        num_skips = min(self.num_encoder_layers, self.num_decoder_layers)
        self.num_skips    = num_skips
        # C2: init=1.0 (was 0.5 in v39). No extra 0.5 factor in decoder loop.
        self.skip_weights = nn.Parameter(
            torch.ones(num_skips, model_dim, dtype=torch.float32)
        )

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_ratio,
                  rope_base, qk_gain_init, mem_tokens=mem_tokens)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm()
        self.lm_head    = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        B, T = input_ids.shape

        tok = self.tok_emb(input_ids)
        tok = F.rms_norm(tok, (tok.size(-1),))
        tok = F.dropout(tok, p=0.05, training=self.training)

        mem = self.memory.expand(B, -1, -1).to(dtype=tok.dtype)
        x   = torch.cat([mem, tok], dim=1)
        M   = self.mem_tokens

        mask = build_hybrid_mask(M, T, input_ids.device, x.dtype)

        # U-Net encoder
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, mask)
            skips.append(x)

        # U-Net decoder — C2: no extra 0.5 factor; skip_weights are learned at init=1.0
        skip_idx = 0
        for i in range(self.num_decoder_layers):
            if skips and skip_idx < self.num_skips:
                sw = self.skip_weights[skip_idx].to(x.dtype)[None, None, :]
                x  = x + sw * skips.pop()
                skip_idx += 1
            x = self.blocks[self.num_encoder_layers + i](x, mask)

        x = x[:, M:, :]   # strip memory tokens

        x       = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits  = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)

        # Logit soft-cap: unconditional (train + eval) — no train/eval mismatch
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank        = int(os.environ.get("RANK",       "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale       = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(True); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(True)

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
    log0(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Expected SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    dataset_dir        = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens         = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer=sentencepiece path={args.tokenizer_path}")
    log0(f"dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_tokens:{val_tokens.numel() - 1}")

    base_model = GPT(
        vocab_size          = args.vocab_size,
        num_layers          = args.num_layers,
        model_dim           = args.model_dim,
        num_heads           = args.num_heads,
        num_kv_heads        = args.num_kv_heads,
        mlp_ratio           = args.mlp_ratio,
        mem_tokens          = args.mem_tokens,
        tie_embeddings      = args.tie_embeddings,
        tied_embed_init_std = args.tied_embed_init_std,
        logit_softcap       = args.logit_softcap,
        rope_base           = args.rope_base,
        qk_gain_init        = args.qk_gain_init,
    ).to(device).bfloat16()

    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else compiled_model
    )

    block_named   = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.memory)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr,
        momentum=args.muon_momentum, backend_steps=args.muon_backend_steps,
    )
    for g in optimizer_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(
        f"model_params:{n_params} layers:{args.num_layers} dim:{args.model_dim} "
        f"heads:{args.num_heads} kv_heads:{args.num_kv_heads} mem_tokens:{args.mem_tokens}"
    )
    log0(f"world_size:{world_size} grad_accum:{grad_accum_steps}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} wallclock_cap:{args.max_wallclock_seconds}s"
    )

    train_loader = DistributedTokenLoader(
        args.train_files, rank, world_size, device, seed=args.seed
    )

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations else 1.0
            )
        step_ms      = elapsed_ms / max(step, 1)
        warmdown_ms  = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        init_model_state  = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_states   = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for micro in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wloss = model(x, y)
                (wloss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup_step:{ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_model_state, strict=True)
        for opt, state in zip(optimizers, init_opt_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        # Stream NOT reset: warmup tokens consumed once, no double-exposure in main loop.

    training_time_ms  = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step       = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
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
                    f"stopping_early: wallclock_cap step:{step}/{args.iterations} "
                    f"train_time:{training_time_ms:.0f}ms"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for micro in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac     = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in optimizer_muon.param_groups:
            g["momentum"] = muon_mom

        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        torch.cuda.synchronize()
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0
                                          or stop_after_step is not None):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak_memory:{torch.cuda.max_memory_allocated() // 1024 // 1024}MiB "
        f"reserved:{torch.cuda.max_memory_reserved() // 1024 // 1024}MiB"
    )

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"model_bytes:{os.path.getsize('final_model.pt')} code_bytes:{len(code.encode('utf-8'))}")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf  = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw  = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)

    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        qfile  = os.path.getsize("final_model.int8.ptz")
        cbytes = len(code.encode("utf-8"))
        ratio  = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"int8+zlib:{qfile} bytes ratio:{ratio:.2f}x total_submission:{qfile + cbytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu", weights_only=True
    )
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
