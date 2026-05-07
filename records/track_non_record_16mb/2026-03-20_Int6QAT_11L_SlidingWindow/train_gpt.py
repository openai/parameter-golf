"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
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

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# v8 changes from baseline:
# - 12 transformer layers (was 9) — depth helps confirmed by v4 experiments
# - model_dim 512 (unchanged) — minimum viable width confirmed
# - mlp_hidden 1024 (2x default) — mlp1536 busts budget at our ~0.91x zlib ratio
#   (Muon near-orthogonal weights are high-entropy; only fits if QAT has 5k+ steps)
# - STE fake-int6 QAT — weights fake-quantised to [-31,31] during training via
#   Straight-Through Estimator; reduces quant penalty ~0.05->~0.001 bpb
# - load_quant_artifact: fixed dispatch to handle int6_mixed_per_row_v2 format
# - mixed post-training quantisation — int6 per-row on block matrices, int8 on embedding
# - sliding window eval (stride=64, chunk=256, ctx=1024) — ~0.045 bpb improvement
# - seq_len=4096 (was 1024) — longer context; confirmed better in PR #64
# - tuned Muon: momentum=0.99, matrix_lr=0.02, warmdown=3000 (from PR #64)
# Target: beat PR #64 StandardOptimal val_bpb=1.16287856

class Hyperparameters:
    # Data paths
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Sliding window eval: ctx must match train_seq_len for best results.
    # With ctx=4096, chunk=512: each scored token has 3584 tokens of context.
    # stride=chunk means we tile the val set once (same cost as flat eval).
    use_sliding_window_eval = bool(int(os.environ.get("USE_SLIDING_WINDOW_EVAL", "1")))
    sw_stride = int(os.environ.get("SW_STRIDE", 64))       # kept for compatibility
    sw_chunk = int(os.environ.get("SW_CHUNK", 512))        # tokens scored per window
    sw_ctx = int(os.environ.get("SW_CTX", 4096))           # must match train_seq_len

    # Training length
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    # mlp_hidden: 1024 (2x) — safe default at our actual ~0.91x zlib ratio on Muon weights.
    # 12L mlp1024 = 22.54M params -> 15.4MB after int6+zlib, within 16MB budget.
    # mlp1536 (28.84M params) would need 0.70x zlib to fit — only achievable with
    # longer QAT runs where weights cluster more tightly on the int6 grid.
    mlp_hidden = int(os.environ.get("MLP_HIDDEN", 1024))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))  # fallback if MLP_HIDDEN not set
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # STE fake-int6 quantisation-aware training.
    # When enabled, weights are fake-quantised to the int6 grid in the forward pass
    # using the Straight-Through Estimator (gradients pass through the rounding op).
    # Benefits: (1) model learns to pack information into int6 representation,
    # reducing post-training quant penalty from ~0.05 to ~0.001 bpb;
    # (2) trained weights sit on the int6 grid, lowering entropy, so zlib
    # compresses ~7% better, making mlp1536 fit in 16MB.
    use_ste_qat = bool(int(os.environ.get("USE_STE_QAT", "1")))
    ste_qat_numel_threshold = int(os.environ.get("STE_QAT_NUMEL_THRESHOLD", 65_536))

    # Quantisation: INT6_NUMEL_THRESHOLD controls which tensors get int6 vs int8.
    # Large weight matrices (attention projections, MLP projections) get int6.
    # Small tensors fall through to int8 or fp16 passthrough as before.
    int6_numel_threshold = int(os.environ.get("INT6_NUMEL_THRESHOLD", 65_536))

    # Optimiser hyperparameters (tuned from PR #64 analysis)
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

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
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def _count_bytes_for_tokens(
    tgt_ids: Tensor,
    prev_ids: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> Tensor:
    token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
    token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
    return token_bytes


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
    """Standard flat evaluation — fast, used for mid-training checkpoints."""
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
            val_byte_count += _count_bytes_for_tokens(
                tgt_ids, prev_ids, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
            ).to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding_window(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    doc_boundary_token_id: int = 0,
) -> tuple[float, float]:
    """
    Fast batched sliding window evaluation.

    Tiles the validation token stream with stride=SW_CHUNK and window=SW_CTX.
    Each window is processed in a single forward pass; only the last SW_CHUNK
    tokens are scored. Every scored token therefore has at least
    (SW_CTX - SW_CHUNK) = 768 tokens of context (with defaults ctx=1024, chunk=256).

    Uses GPT.forward_logits() for per-token scores, batched for speed.
    Runs in approximately the same wall time as flat eval.
    """
    ctx = args.sw_ctx
    chunk = args.sw_chunk

    # Unwrap to the base GPT model for forward_logits access
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod

    n_tokens = val_tokens.numel() - 1
    starts = list(range(0, n_tokens - ctx, chunk))
    if not starts:
        return float("nan"), float("nan")

    my_starts = [s for i, s in enumerate(starts) if i % world_size == rank]

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    batch_size = max(1, args.val_batch_size // ctx)

    m.eval()
    with torch.inference_mode():
        for b in range(0, len(my_starts), batch_size):
            batch_starts = my_starts[b : b + batch_size]
            x = torch.stack([
                val_tokens[s : s + ctx].to(device=device, dtype=torch.int64)
                for s in batch_starts
            ])  # [B, ctx]
            y = torch.stack([
                val_tokens[s + 1 : s + ctx + 1].to(device=device, dtype=torch.int64)
                for s in batch_starts
            ])  # [B, ctx]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = m.forward_logits(x)  # [B, ctx, V]

            # Score only last chunk positions
            logits_s = logits[:, -chunk:, :].reshape(-1, logits.size(-1))  # [B*chunk, V]
            y_s = y[:, -chunk:].reshape(-1)                                 # [B*chunk]
            x_s = x[:, -chunk:].reshape(-1)                                 # [B*chunk] prev tokens

            loss_per_tok = F.cross_entropy(logits_s.float(), y_s, reduction="none")
            val_loss_sum += loss_per_tok.sum().to(torch.float64)
            val_token_count += float(y_s.numel())
            val_byte_count += _count_bytes_for_tokens(
                y_s, x_s, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
            ).to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    if val_token_count == 0:
        return float("nan"), float("nan")
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    m.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTISATION
# -----------------------------
#
# Two quantisation schemes are used:
#   int8  — large tensors with numel <= INT6_NUMEL_THRESHOLD, and non-matrix tensors
#   int6  — large 2D weight matrices (attention projections, MLP) with numel > threshold
#
# Int6 packs two 6-bit signed values per byte using a custom bit-packing scheme.
# Each value is in [-31, 31]; scale is per-row fp16 (same as int8 path).
# This gives a 25% size reduction over int8 for the large matrices, which is
# the unlock that allows 12L×512d to fit within the 16MB budget.
#
# Format tag: "int6_per_row_v1"

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

# Int6 constants: values in [-31, 31], 64 levels per sign
INT6_MAX = 31
INT6_CLIP_PERCENTILE = 99.99984
INT6_CLIP_Q = INT6_CLIP_PERCENTILE / 100.0
INT6_PER_ROW_SCALE_DTYPE = torch.float16


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor_int8(t: Tensor) -> tuple[Tensor, Tensor]:
    """Standard per-row int8 quantisation (unchanged from baseline)."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_float_tensor_int6(t: Tensor) -> tuple[Tensor, Tensor, int]:
    """
    Per-row int6 quantisation for 2D weight matrices.

    Values are clipped to [-31, 31] (6-bit signed, symmetric).
    Four int6 values are packed into exactly 3 bytes — zero bit waste:
      byte0 = a[5:0] | b[1:0]<<6
      byte1 = b[5:2] | c[3:0]<<4
      byte2 = c[5:4] | d[5:0]<<2
    Rows are padded to a multiple of 4 for alignment, orig_cols stored for trimming.

    This gives exactly 0.75x the storage of int8, before zlib.

    Returns: (packed_uint8 tensor, per-row scale fp16 tensor, orig_cols int)
    """
    assert t.ndim == 2, "int6 quantisation only supported for 2D tensors"
    t32 = t.float()
    rows, cols = t32.shape

    clip_abs = (
        torch.quantile(t32.abs(), INT6_CLIP_Q, dim=1)
        if t32.numel()
        else torch.empty((rows,), dtype=torch.float32)
    )
    clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
    scale = (clip_abs / float(INT6_MAX)).clamp_min(1.0 / float(INT6_MAX))
    q_int = torch.clamp(torch.round(clipped / scale[:, None]), -INT6_MAX, INT6_MAX).to(torch.int8)

    q_np = q_int.numpy().astype(np.int8)
    pad = (4 - cols % 4) % 4
    if pad:
        q_np = np.pad(q_np, ((0, 0), (0, pad)), constant_values=0)
    q_off = (q_np + INT6_MAX).astype(np.uint8)  # [0, 62]
    n_groups = q_off.shape[1] // 4
    g = q_off.reshape(rows, n_groups, 4)
    a, b, c, d = g[:, :, 0], g[:, :, 1], g[:, :, 2], g[:, :, 3]
    byte0 = (a & 0x3F) | ((b & 0x03) << 6)
    byte1 = ((b >> 2) & 0x0F) | ((c & 0x0F) << 4)
    byte2 = ((c >> 4) & 0x03) | ((d & 0x3F) << 2)
    packed = np.stack([byte0, byte1, byte2], axis=2).reshape(rows, -1)

    packed_tensor = torch.from_numpy(packed.astype(np.uint8)).contiguous()
    scale_tensor = scale.to(dtype=INT6_PER_ROW_SCALE_DTYPE).contiguous()
    return packed_tensor, scale_tensor, cols


def dequantize_float_tensor_int6(packed: Tensor, scale: Tensor, orig_cols: int) -> Tensor:
    """Unpack 4-into-3-byte int6 encoding back to float."""
    rows = packed.shape[0]
    p = packed.numpy().astype(np.uint8).reshape(rows, -1, 3)
    b0 = p[:, :, 0].astype(np.uint16)
    b1 = p[:, :, 1].astype(np.uint16)
    b2 = p[:, :, 2].astype(np.uint16)
    a = b0 & 0x3F
    b = ((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)
    c = ((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)
    d = (b2 >> 2) & 0x3F
    q_off = np.stack([a, b, c, d], axis=2).reshape(rows, -1).astype(np.int16)
    q_f32 = (q_off - INT6_MAX).astype(np.float32)[:, :orig_cols]
    s = scale.to(dtype=torch.float32)
    return (torch.from_numpy(q_f32) * s[:, None]).contiguous()


def quantize_state_dict_mixed(state_dict: dict[str, Tensor], int6_numel_threshold: int):
    """
    Mixed int6/int8 quantisation:
    - Large 2D float matrices (numel > threshold): int6 per-row
    - Smaller 2D or non-matrix floats: int8 per-row/per-tensor (baseline)
    - Control tensors and small tensors: fp16 passthrough (baseline)
    """
    quantized_int8: dict[str, Tensor] = {}
    quantized_int6: dict[str, Tensor] = {}
    int6_orig_cols: dict[str, int] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes", "num_int6", "num_int8"),
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

        # Control tensors / small tensors → fp16 passthrough
        is_control = any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or is_control:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        orig_dtype = str(t.dtype).removeprefix("torch.")

        # Large 2D matrices → int6
        if t.ndim == 2 and t.numel() > int6_numel_threshold:
            packed, scale, orig_cols = quantize_float_tensor_int6(t)
            quantized_int6[name] = packed
            scales[name] = scale
            int6_orig_cols[name] = orig_cols
            dtypes[name] = orig_dtype
            qmeta[name] = {"scheme": "int6_per_row", "orig_cols": orig_cols}
            stats["int8_payload_bytes"] += tensor_nbytes(packed) + tensor_nbytes(scale)
            stats["num_int6"] += 1
        else:
            # Smaller matrices or 1D tensors → int8
            q, s = quantize_float_tensor_int8(t)
            if s.ndim > 0:
                qmeta[name] = {"scheme": "per_row", "axis": 0}
            quantized_int8[name] = q
            scales[name] = s
            dtypes[name] = orig_dtype
            stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
            stats["num_int8"] += 1

    # Pack all int6 data into a single flat uint8 tensor so zlib sees one
    # continuous block of structured data rather than 73 separate tensors
    # with interleaved pickle metadata (which breaks zlib's pattern detection).
    int6_names = sorted(quantized_int6.keys())
    int6_index: list[dict] = []
    if int6_names:
        chunks = []
        for name in int6_names:
            packed = quantized_int6[name]
            int6_index.append({
                "name": name,
                "rows": packed.shape[0],
                "packed_cols": packed.shape[1],
                "orig_cols": int6_orig_cols[name],
            })
            chunks.append(packed.reshape(-1))
        int6_flat = torch.cat(chunks)  # single [N] uint8 tensor
    else:
        int6_flat = torch.zeros(0, dtype=torch.uint8)

    obj: dict[str, object] = {
        "__quant_format__": "int6_mixed_per_row_v2",
        "int6_flat": int6_flat,        # single flat uint8 tensor
        "int6_index": int6_index,      # list of dicts with name/shape/cols
        "quantized_int8": quantized_int8,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_mixed(obj: dict[str, object]) -> dict[str, Tensor]:
    """Reconstruct state dict from mixed int6/int8 quantised artifact."""
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    fmt = obj.get("__quant_format__", "int6_mixed_per_row_v1")

    # Dequantise int6 tensors
    if fmt == "int6_mixed_per_row_v2":
        # New flat format: reconstruct individual tensors from the single flat buffer
        int6_flat: Tensor = obj["int6_flat"]
        int6_index: list = obj["int6_index"]
        offset = 0
        for entry in int6_index:
            name = entry["name"]
            rows = entry["rows"]
            packed_cols = entry["packed_cols"]
            orig_cols = entry["orig_cols"]
            n = rows * packed_cols
            packed = int6_flat[offset : offset + n].reshape(rows, packed_cols)
            offset += n
            dtype = getattr(torch, obj["dtypes"][name])
            scale = obj["scales"][name]
            out[name] = dequantize_float_tensor_int6(packed, scale, orig_cols).to(dtype=dtype).contiguous()
    else:
        # Legacy v1 format
        for name, packed in obj["quantized_int6"].items():
            dtype = getattr(torch, obj["dtypes"][name])
            scale = obj["scales"][name]
            orig_cols = obj["int6_orig_cols"][name]
            out[name] = dequantize_float_tensor_int6(packed, scale, orig_cols).to(dtype=dtype).contiguous()

    # Dequantise int8 tensors
    for name, q in obj["quantized_int8"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        if meta.get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()

    # Passthrough tensors (fp16/fp32 small tensors, control params)
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t

    return out


# Keep the old int8-only functions as fallback / for loading baseline checkpoints
def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
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
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor_int8(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
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


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
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


def load_quant_artifact(path: str) -> dict[str, Tensor]:
    """Load either int6_mixed or legacy int8 artifact, return state dict."""
    with open(path, "rb") as f:
        blob = f.read()
    obj = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    fmt = obj.get("__quant_format__", "int8_clean_per_row_v1")
    if fmt in ("int6_mixed_per_row_v1", "int6_mixed_per_row_v2"):
        return dequantize_state_dict_mixed(obj)
    else:
        return dequantize_state_dict_int8(obj)


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

# Global STE QAT flags — set by main() before training.
# QAT is implemented as an external weight-perturbation loop (not inside the compiled
# forward pass), so torch.compile(fullgraph=True) is unaffected.
# _STE_QAT_ENABLED: whether QAT is configured at all
# _STE_QAT_ACTIVE:  flips True at warmup boundary; controls the perturbation loop
_STE_QAT_ENABLED: bool = False
_STE_QAT_ACTIVE: bool = False
_STE_QAT_NUMEL_THRESHOLD: int = 65_536
_STE_QAT_WARMUP_STEPS: int = 200


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Keep weights in fp32 for optimiser quality, cast at matmul time for bf16 compute."""
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


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
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
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
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    """relu^2 MLP. mlp_hidden sets the hidden dimension directly."""
    def __init__(self, dim: int, mlp_hidden: int):
        super().__init__()
        self.fc = CastedLinear(dim, mlp_hidden, bias=False)
        self.proj = CastedLinear(mlp_hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_hidden: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_hidden)
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
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_hidden: int, tie_embeddings: bool,
                 tied_embed_init_std: float, logit_softcap: float, rope_base: float,
                 qk_gain_init: float):
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
            Block(model_dim, num_heads, num_kv_heads, mlp_hidden, rope_base, qk_gain_init)
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
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

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
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return raw logits [B, T, V] — used by sliding window eval."""
        bsz, seqlen = input_ids.shape
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
        x = self.final_norm(x)  # [B, T, D]
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)  # [B, T, V]


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Resolve mlp_hidden
    mlp_hidden = args.mlp_hidden

    # Enable STE QAT globally — must happen before model construction so that
    # CastedLinear picks up the flag when torch.compile traces the graph.
    global _STE_QAT_ENABLED, _STE_QAT_ACTIVE, _STE_QAT_NUMEL_THRESHOLD, _STE_QAT_WARMUP_STEPS
    _STE_QAT_ENABLED = args.use_ste_qat
    _STE_QAT_ACTIVE = False  # activates after warmup steps
    _STE_QAT_NUMEL_THRESHOLD = args.ste_qat_numel_threshold
    _STE_QAT_WARMUP_STEPS = 200

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

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

    # -----------------------------
    # TOKENIZER + VALIDATION SETUP
    # -----------------------------

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
    log0(f"sliding_window_eval:{args.use_sliding_window_eval} stride:{args.sw_stride} chunk:{args.sw_chunk} ctx:{args.sw_ctx}")

    # -----------------------------
    # MODEL + OPTIMISER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_hidden=mlp_hidden,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
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
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
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
    log0(f"model_params:{n_params}")
    log0(f"mlp_hidden:{mlp_hidden} num_layers:{args.num_layers} model_dim:{args.model_dim}")
    log0(f"ste_qat:{args.use_ste_qat} ste_qat_numel_threshold:{args.ste_qat_numel_threshold} ste_qat_warmup_steps:{_STE_QAT_WARMUP_STEPS} (activates at step {_STE_QAT_WARMUP_STEPS})")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
         f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
         f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}")
    log0(f"muon_momentum:{args.muon_momentum} muon_momentum_warmup_start:{args.muon_momentum_warmup_start} "
         f"muon_momentum_warmup_steps:{args.muon_momentum_warmup_steps}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
         f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
         f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")
    log0(f"int6_numel_threshold:{args.int6_numel_threshold}")
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

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

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

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
            # Use fast flat eval for mid-training checkpoints; sliding window only at end
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        # STE QAT: collect large matrix params to fake-quantise this step.
        # We do this outside the compiled graph — temporarily overwrite .data
        # with quantised values before forward, restore originals after backward.
        # The compiled model sees normal weights and compiles cleanly.
        qat_params: list[tuple[nn.Parameter, Tensor]] = []
        if _STE_QAT_ACTIVE:
            for p in matrix_params:
                if p.numel() > args.ste_qat_numel_threshold:
                    orig = p.data.clone()
                    q_scale = (p.data.float().abs().amax(dim=1) / INT6_MAX).clamp_min(1.0 / INT6_MAX)
                    p_q = torch.round(p.data.float() / q_scale[:, None]).clamp(-INT6_MAX, INT6_MAX) * q_scale[:, None]
                    p.data.copy_(p_q.to(p.data.dtype))
                    qat_params.append((p, orig))

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Restore fp32 weights — optimiser updates the original (non-quantised) values.
        # Gradients were computed on quantised weights, giving the STE effect.
        for p, orig in qat_params:
            p.data.copy_(orig)

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
        if _STE_QAT_ENABLED and not _STE_QAT_ACTIVE and step >= _STE_QAT_WARMUP_STEPS:
            _STE_QAT_ACTIVE = True
            log0(f"ste_qat:active step:{step}")
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
         f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    # -----------------------------
    # SLIDING WINDOW FINAL EVAL
    # -----------------------------

    if args.use_sliding_window_eval and master_process:
        log0("running sliding_window_eval on training-converged model...")
    if args.use_sliding_window_eval:
        sw_val_loss, sw_val_bpb = eval_val_sliding_window(
            args, model, rank, world_size, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        log0(f"sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
             f"(stride={args.sw_stride} chunk={args.sw_chunk} ctx={args.sw_ctx})")

    # -----------------------------
    # SERIALISATION + ROUNDTRIP VALIDATION
    # -----------------------------

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    # Mixed int6/int8 quantisation
    quant_obj, quant_stats = quantize_state_dict_mixed(base_model.state_dict(), args.int6_numel_threshold)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    artifact_name = "final_model.int6mixed.ptz"
    if master_process:
        with open(artifact_name, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(artifact_name)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"Serialized model int6mixed+zlib: {quant_file_bytes} bytes "
             f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} "
             f"payload_ratio:{ratio:.2f}x num_int6:{quant_stats['num_int6']} num_int8:{quant_stats['num_int8']})")
        log0(f"Total submission size int6mixed+zlib: {quant_file_bytes + code_bytes} bytes")
        within_budget = quant_file_bytes + code_bytes <= 16_000_000
        log0(f"Budget check: {'OK' if within_budget else 'FAIL — OVER 16MB'} "
             f"({quant_file_bytes + code_bytes:,} / 16,000,000 bytes)")

    if distributed:
        dist.barrier()
    quant_state = load_quant_artifact(artifact_name)
    base_model.load_state_dict(quant_state, strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Also run sliding window on the quantised model (this is the leaderboard score)
    if args.use_sliding_window_eval:
        q_sw_val_loss, q_sw_val_bpb = eval_val_sliding_window(
            args, model, rank, world_size, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        log0(f"final_roundtrip_sliding_window val_loss:{q_sw_val_loss:.4f} val_bpb:{q_sw_val_bpb:.4f} "
             f"(stride={args.sw_stride} chunk={args.sw_chunk} ctx={args.sw_ctx})")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

