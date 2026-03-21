"""
12L + Low-Rank Q + QAT + FTLE-guided per-row precision + Stride-OGD

Based on the SOTA record (SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit).
New techniques:
  1. Low-Rank Q factorization (rank=128) → ~8% faster steps, funding 12 layers
  2. QAT with STE during training (uniform int6) → smaller quant gap
  3. FTLE gradient sensitivity tracking → per-row precision allocation at export
  4. Stride-OGD at eval → online vocab bias adaptation for free BPB

Hard stop: train_gpt.py must never be longer than 1500 lines.
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
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 2500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 0))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape — 12 layers with Low-Rank Q
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    q_rank = int(os.environ.get("Q_RANK", 128))  # Low-Rank Q factorization rank

    # Optimizer hyperparameters
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.10))
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

    # QAT config
    qat_bits = int(os.environ.get("QAT_BITS", 6))
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.1))  # Start QAT after 10% of training

    # Stride-OGD eval config
    ogd_lr = float(os.environ.get("OGD_LR", 0.1))
    ogd_enabled = bool(int(os.environ.get("OGD_ENABLED", "1")))

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
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
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
    seq_len_override: int = 0,
) -> tuple[float, float]:
    seq_len = seq_len_override if seq_len_override > 0 else args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
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
# QAT: FAKE QUANTIZATION WITH STE
# -----------------------------

def fake_quant_ste(w: Tensor, bits: int) -> Tensor:
    """Fake quantization with straight-through estimator for QAT.
    Simulates reduced-precision quantization during training so the model
    learns to be robust to quantization errors."""
    if bits >= 8 or not w.is_floating_point() or w.ndim < 2:
        return w
    max_val = (1 << (bits - 1)) - 1  # e.g., 31 for 6-bit
    # Per-row scale (matching export quantization)
    row_max = w.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = row_max / max_val
    w_q = torch.clamp(torch.round(w / scale), -max_val, max_val) * scale
    # STE: forward uses quantized, backward flows through original
    return w + (w_q - w).detach()


# -----------------------------
# POST-TRAINING QUANTIZATION (FTLE-guided per-row precision)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_row_at_bits(row: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    """Quantize a single row to the given bit width, stored as int8."""
    max_val = (1 << (bits - 1)) - 1  # int8=127, int7=63, int6=31, int5=15, int4=7
    r32 = row.float()
    clip_abs = float(torch.quantile(r32.abs(), INT8_CLIP_Q).item()) if r32.numel() else 0.0
    scale = max(clip_abs / max_val, 1.0 / max_val)
    q = torch.clamp(torch.round(torch.clamp(r32, -clip_abs, clip_abs) / scale), -max_val, max_val).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)

def quantize_tensor_ftle(t: Tensor, ftle_scores: Tensor | None, target_avg_bits: float = 5.5) -> tuple[Tensor, Tensor]:
    """FTLE-guided per-row precision quantization for 2D tensors.
    High-sensitivity rows get more bits, low-sensitivity rows get fewer."""
    t32 = t.float()
    if t32.ndim != 2 or ftle_scores is None or ftle_scores.numel() != t32.shape[0]:
        # Fallback to standard int8 per-row
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    nrows = t32.shape[0]
    # Normalize FTLE scores to [0, 1]
    scores = ftle_scores.float()
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = torch.ones(nrows)

    # Allocate bits per row based on sensitivity score
    # Map score to bits: score=0 → min_bits, score=1 → max_bits
    # We solve for min/max to achieve target_avg_bits
    min_bits, max_bits = 4, 8
    # Linear mapping: bits_i = min_bits + score_i * (max_bits - min_bits)
    # Adjust to hit target average
    raw_bits = min_bits + scores * (max_bits - min_bits)
    current_avg = raw_bits.mean().item()
    if current_avg > 0:
        shift = target_avg_bits - current_avg
        raw_bits = (raw_bits + shift).clamp(min_bits, max_bits)
    bits_per_row = raw_bits.round().long().clamp(min_bits, max_bits)

    # Quantize each row at its assigned precision
    q_rows = []
    scale_rows = []
    for i in range(nrows):
        bits = int(bits_per_row[i].item())
        max_val = (1 << (bits - 1)) - 1
        row = t32[i]
        clip_abs = float(torch.quantile(row.abs(), INT8_CLIP_Q).item()) if row.numel() else 0.0
        s = max(clip_abs / max_val, 1.0 / max_val)
        qr = torch.clamp(torch.round(torch.clamp(row, -clip_abs, clip_abs) / s), -max_val, max_val).to(torch.int8)
        q_rows.append(qr)
        scale_rows.append(s)

    q = torch.stack(q_rows).contiguous()
    scale = torch.tensor(scale_rows, dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    return q, scale

def quantize_state_dict_mixed(state_dict: dict[str, Tensor], ftle_dict: dict[str, Tensor] | None = None,
                               target_avg_bits: float = 5.5):
    """Mixed-precision quantization guided by FTLE gradient sensitivity."""
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Keep tied embedding in fp16
        if "tok_emb" in name:
            kept = t.to(dtype=torch.float16).contiguous()
            passthrough[name] = kept
            passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        ftle_scores = ftle_dict.get(name) if ftle_dict else None
        q, s = quantize_tensor_ftle(t, ftle_scores, target_avg_bits=target_avg_bits)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "mixed_ftle_per_row_v1",
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

def dequantize_state_dict(obj: dict[str, object]) -> dict[str, Tensor]:
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
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
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
# FTLE GRADIENT SENSITIVITY TRACKER
# -----------------------------

class FTLETracker:
    """Tracks per-row gradient L2 norms during training for FTLE-guided quantization."""
    def __init__(self):
        self.accum: dict[str, Tensor] = {}
        self.count: int = 0

    def update(self, named_params):
        """Accumulate gradient norms for 2D weight matrices."""
        self.count += 1
        for name, p in named_params:
            if p.grad is not None and p.ndim == 2:
                row_norm = p.grad.detach().float().norm(dim=-1)
                if name not in self.accum:
                    self.accum[name] = torch.zeros_like(row_norm)
                self.accum[name] += row_norm

    def get_scores(self) -> dict[str, Tensor]:
        """Return normalized per-row sensitivity scores."""
        if self.count == 0:
            return {}
        return {name: acc / self.count for name, acc in self.accum.items()}


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


class QATCastedLinear(nn.Linear):
    """CastedLinear with optional QAT fake quantization during training."""
    def __init__(self, *args, qat_bits: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_bits = qat_bits
        self.qat_enabled = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if self.training and self.qat_enabled and self.qat_bits < 8:
            w = fake_quant_ste(w, self.qat_bits)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
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
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                adjusted_base = self.base * (scale ** (self.dim / (self.dim - 2)))
                inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        train_seq_len: int = 1024,
        q_rank: int = 0,
        qat_bits: int = 8,
    ):
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

        # Low-Rank Q factorization: dim -> q_rank -> dim
        self.use_low_rank_q = q_rank > 0 and q_rank < dim
        if self.use_low_rank_q:
            self.c_q_down = QATCastedLinear(dim, q_rank, bias=False, qat_bits=qat_bits)
            self.c_q_up = QATCastedLinear(q_rank, dim, bias=False, qat_bits=qat_bits)
        else:
            self.c_q = QATCastedLinear(dim, dim, bias=False, qat_bits=qat_bits)

        self.c_k = QATCastedLinear(dim, kv_dim, bias=False, qat_bits=qat_bits)
        self.c_v = QATCastedLinear(dim, kv_dim, bias=False, qat_bits=qat_bits)
        self.proj = QATCastedLinear(dim, dim, bias=False, qat_bits=qat_bits)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        if self.use_low_rank_q:
            q = self.c_q_up(self.c_q_down(x))
        else:
            q = self.c_q(x)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
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
    def __init__(self, dim: int, mlp_mult: int, qat_bits: int = 8):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = QATCastedLinear(dim, hidden, bias=False, qat_bits=qat_bits)
        self.proj = QATCastedLinear(hidden, dim, bias=False, qat_bits=qat_bits)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
        rope_base: float, qk_gain_init: float, train_seq_len: int = 1024,
        q_rank: int = 0, qat_bits: int = 8,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            train_seq_len=train_seq_len, q_rank=q_rank, qat_bits=qat_bits,
        )
        self.mlp = MLP(dim, mlp_mult, qat_bits=qat_bits)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
        num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float,
        logit_softcap: float, rope_base: float, qk_gain_init: float,
        train_seq_len: int = 1024, q_rank: int = 0, qat_bits: int = 8,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  train_seq_len=train_seq_len, q_rank=q_rank, qat_bits=qat_bits)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
            # Overtone init: shape embedding spectrum to power-law decay
            with torch.no_grad():
                U, S, V = torch.linalg.svd(self.tok_emb.weight.data, full_matrices=False)
                target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
                self.tok_emb.weight.data = (U * target_S[None, :]) @ V
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
        # Phase-transition resid_mix: early layers trust x0 more, late layers trust residual
        num_layers = len(self.blocks)
        for i, block in enumerate(self.blocks):
            with torch.no_grad():
                phase = torch.sigmoid(torch.tensor(3.0 * (i / max(num_layers - 1, 1) - 0.5)))
                block.resid_mix.data[0] = phase * torch.ones(block.resid_mix.shape[1])
                block.resid_mix.data[1] = (1 - phase) * torch.ones(block.resid_mix.shape[1])

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x_norm = self.final_norm(x)
        x_flat = x_norm.reshape(-1, x_norm.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning logits [batch, seq_len, vocab]. For sliding window eval."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight.to(x.dtype))
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


# -----------------------------
# SLIDING WINDOW EVAL + STRIDE-OGD
# -----------------------------

def eval_val_sliding_ogd(
    logits_fn,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len: int,
    stride: int,
    vocab_size: int,
    ogd_lr: float = 0.1,
    ogd_enabled: bool = True,
    eval_batch_seqs: int = 256,
) -> tuple[float, float]:
    """Sliding window evaluation with optional online gradient descent on vocab bias."""
    total = val_tokens.numel() - 1

    # Build windows: (start_pos, score_offset)
    windows: list[tuple[int, int]] = []
    p = 0
    while p + seq_len <= total:
        s = 0 if p == 0 else (seq_len - stride)
        windows.append((p, s))
        p += stride

    # Distribute across ranks
    n = len(windows)
    per_rank = (n + world_size - 1) // world_size
    my_start = rank * per_rank
    my_end = min(my_start + per_rank, n)
    my_windows = windows[my_start:my_end]

    # Stride-OGD: learnable vocab bias updated online
    vocab_bias = torch.zeros(vocab_size, device=device, dtype=torch.float32, requires_grad=ogd_enabled)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    for i in range(0, len(my_windows), eval_batch_seqs):
        batch = my_windows[i : i + eval_batch_seqs]
        bs = len(batch)

        x_list = [val_tokens[w : w + seq_len] for w, _ in batch]
        y_list = [val_tokens[w + 1 : w + seq_len + 1] for w, _ in batch]
        pad = eval_batch_seqs - bs
        if pad > 0:
            x_list.extend([x_list[-1]] * pad)
            y_list.extend([y_list[-1]] * pad)

        x = torch.stack(x_list).to(device=device, dtype=torch.int64)
        y = torch.stack(y_list).to(device=device, dtype=torch.int64)

        if ogd_enabled:
            # Forward with gradient tracking for OGD bias
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    raw_logits = logits_fn(x)
                logits = raw_logits.float() + vocab_bias[None, None, :]

            # Score this batch
            for b in range(bs):
                s = batch[b][1]
                scored_logits = logits[b, s:]
                scored_targets = y[b, s:]

                loss = F.cross_entropy(scored_logits, scored_targets, reduction="sum")
                loss_sum += loss.detach().to(torch.float64)
                ns = scored_targets.numel()
                tok_count += ns

                prev = x[b, s : s + ns]
                tgt = scored_targets
                tb = base_bytes_lut[tgt].to(torch.int16)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()

            # OGD update: gradient descent on vocab bias using this batch's loss
            # Use the scored tokens from all batch elements
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            accum = torch.zeros((), device=device)
            for b in range(bs):
                s_off = batch[b][1]
                sl = logits[b, s_off:]
                st = y[b, s_off:]
                accum = accum + F.cross_entropy(sl, st, reduction="mean")
            accum.backward()
            with torch.no_grad():
                if vocab_bias.grad is not None:
                    vocab_bias -= ogd_lr * vocab_bias.grad
                    vocab_bias.grad.zero_()
        else:
            # Standard sliding window eval (no OGD)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = logits_fn(x)

                for b in range(bs):
                    s = batch[b][1]
                    scored_logits = logits[b, s:]
                    scored_targets = y[b, s:]
                    loss = F.cross_entropy(scored_logits.float(), scored_targets, reduction="sum")
                    loss_sum += loss.to(torch.float64)
                    ns = scored_targets.numel()
                    tok_count += ns
                    prev = x[b, s : s + ns]
                    tgt = scored_targets
                    tb = base_bytes_lut[tgt].to(torch.int16)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.int16)
                    byte_count += tb.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / tok_count).item()
    bpb = val_loss / math.log(2.0) * (tok_count.item() / byte_count.item())
    return val_loss, bpb


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # DISTRIBUTED + CUDA SETUP
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

    # TOKENIZER + VALIDATION SETUP
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
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        train_seq_len=args.train_seq_len,
        q_rank=args.q_rank,
        qat_bits=args.qat_bits,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, (CastedLinear, QATCastedLinear)):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=0.01, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=0.01, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} (12L, Low-Rank Q r={args.q_rank})")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"qat_bits:{args.qat_bits} qat_start_frac:{args.qat_start_frac}")
    log0(f"ogd_enabled:{args.ogd_enabled} ogd_lr:{args.ogd_lr}")
    log0(f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} iterations:{args.iterations}")
    log0(f"seed:{args.seed}")

    # FTLE tracker
    ftle_tracker = FTLETracker()

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
    qat_activated = False
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
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        # Enable QAT after qat_start_frac of training
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if not qat_activated and max_wallclock_ms and elapsed_ms > args.qat_start_frac * max_wallclock_ms:
            for module in base_model.modules():
                if isinstance(module, QATCastedLinear):
                    module.qat_enabled = True
            qat_activated = True
            log0(f"step:{step} QAT enabled (elapsed:{elapsed_ms:.0f}ms)")

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

        # FTLE: track gradient sensitivity every 100 steps (cheap)
        if step % 100 == 0:
            ftle_tracker.update(base_model.named_parameters())

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
        # Decoupled weight decay for Muon-optimized matrix params
        with torch.no_grad():
            for p in matrix_params:
                p.mul_(1.0 - 0.02 * optimizer_muon.param_groups[0]["lr"])
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
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

    # FTLE scores for quantization
    ftle_scores = ftle_tracker.get_scores()
    log0(f"FTLE tracked {len(ftle_scores)} tensors over {ftle_tracker.count} samples")

    # SERIALIZATION + ROUNDTRIP VALIDATION
    export_state = dict(base_model.state_dict())

    if master_process:
        torch.save(export_state, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")

    # Try multiple target_avg_bits to find the best that fits under 16MB
    code_bytes = len(code.encode("utf-8"))
    best_bits = 5.5
    for try_bits in [6.0, 5.5, 5.0, 4.5]:
        quant_obj, quant_stats = quantize_state_dict_mixed(export_state, ftle_scores, target_avg_bits=try_bits)
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = zlib.compress(quant_raw, level=9)
        total_size = len(quant_blob) + code_bytes
        log0(f"target_bits:{try_bits:.1f} compressed:{len(quant_blob)} total:{total_size}")
        if total_size <= 16_000_000:
            best_bits = try_bits
            break

    log0(f"Selected target_avg_bits={best_bits}")
    quant_obj, quant_stats = quantize_state_dict_mixed(export_state, ftle_scores, target_avg_bits=best_bits)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model mixed+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict(quant_state), strict=False)

    # Eval: sliding window + OGD
    eval_sl = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    eval_stride = args.eval_stride
    if eval_sl != args.train_seq_len:
        val_tokens_eval = load_validation_tokens(args.val_files, eval_sl)
    else:
        val_tokens_eval = val_tokens

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()

    if eval_stride > 0:
        log0(f"Compiling forward_logits for sliding window + OGD eval (stride={eval_stride}, seq_len={eval_sl})...")
        compiled_logits = torch.compile(base_model.forward_logits, dynamic=False)
        eval_batch_seqs = 256
        warmup_x = torch.zeros(eval_batch_seqs, eval_sl, dtype=torch.int64, device=device)
        base_model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = compiled_logits(warmup_x)
        log0("Compilation done, starting eval...")

        q_val_loss, q_val_bpb = eval_val_sliding_ogd(
            compiled_logits, rank, world_size, device,
            val_tokens_eval, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            eval_sl, eval_stride, args.vocab_size,
            ogd_lr=args.ogd_lr, ogd_enabled=args.ogd_enabled,
            eval_batch_seqs=eval_batch_seqs,
        )
        base_model.train()
    else:
        q_val_loss, q_val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens_eval, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            seq_len_override=eval_sl if eval_sl != args.train_seq_len else 0,
        )

    torch.cuda.synchronize()
    log0(
        f"final_mixed_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_mixed_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
