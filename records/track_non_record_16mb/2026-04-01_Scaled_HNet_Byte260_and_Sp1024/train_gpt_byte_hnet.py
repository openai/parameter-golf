"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
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
import torch

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._higher_order_ops.associative_scan import associative_scan

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_byte260")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY", 0))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 260))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # H-Net tokenization.
    outer_layers = int(os.environ.get("OUTER_LAYERS", 0))
    target_avg_chunk_len = float(os.environ.get("TARGET_AVG_CHUNK_LEN", 6.0))
    ratio_loss_weight = float(os.environ.get("RATIO_LOSS_WEIGHT", 0.03))
    hnet_lr_diff = float(os.environ.get("HNET_LR_DIFF", 0.75))

    # Optimizer hyperparameters.
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
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Sliding window eval.
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))

    # EMA.
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    # QAT (late quantization-aware training).
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))

    # GPTQ calibration.
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 45.0))
    gptq_calib_batches = int(os.environ.get("GPTQ_CALIB_BATCHES", 256))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))
    prune_pct = float(os.environ.get("PRUNE_PCT", 0.03))

    compile_model = bool(int(os.environ.get("COMPILE_MODEL", "1")))
    chunk_divisor = int(os.environ.get("CHUNK_DIVISOR", "4"))
    model_save_path = os.environ.get("MODEL_SAVE_PATH", "final_model.pt")

# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
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
                    # Scale correction from Muon reference implementations.
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
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_byte_luts(vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    BYTE_OFFSET = 4  # pad=0, bos=1, eos=2, unk=3
    base_bytes = torch.zeros(vocab_size, dtype=torch.int16, device=device)
    base_bytes[BYTE_OFFSET:] = 1  # each byte token = 1 byte
    has_leading_space = torch.zeros(vocab_size, dtype=torch.bool, device=device)  # not applicable
    is_boundary = torch.ones(vocab_size, dtype=torch.bool, device=device)
    is_boundary[BYTE_OFFSET:] = False  # byte tokens are not boundary tokens
    return base_bytes, has_leading_space, is_boundary


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
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
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
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
    """Sliding window evaluation: each token scored with maximum context.

    Windows of train_seq_len advance by `stride`. Only the last `stride` tokens
    per window contribute to the score (first window scores all).
    """
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
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
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

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

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
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

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
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
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# ---------------------------------------------------------------------------
# Int6 Quantization + zstd Compression
# ---------------------------------------------------------------------------

INT6_MAX_VAL = 31

def quantize_float_tensor_int6(t: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a float tensor to 6-bit signed integers [-31, 31] with per-row scaling."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / INT6_MAX_VAL).clamp_min(1.0 / INT6_MAX_VAL)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -INT6_MAX_VAL, INT6_MAX_VAL).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / INT6_MAX_VAL if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -INT6_MAX_VAL, INT6_MAX_VAL).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int6(state_dict: dict[str, Tensor]):
    """Pure INT6 quantization for all large float tensors."""
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int6_payload_bytes"), 0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int6_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor_int6(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int6_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int6_per_row_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int6(obj: dict[str, object]) -> dict[str, Tensor]:
    """Dequantize int6 state dict — same logic as int8, scale multiplication is identical."""
    return dequantize_state_dict_int8(obj)


# ---------------------------------------------------------------------------
# GPTQ: Hessian-aware INT6 quantization
# ---------------------------------------------------------------------------

def _classify_param(name: str) -> str:
    """Classify parameter for mixed quantization: embeddings get INT8, rest gets INT6+GPTQ."""
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    return "quant"  # everything else gets INT6+GPTQ


def quantize_int6_gptq(weight: Tensor, hessian: Tensor | None = None,
                        clip_range: int = 31, block_size: int = 128) -> tuple[Tensor, Tensor]:
    """Full GPTQ: Hessian-aware int6 quantization with Cholesky error compensation."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _quantize_int6_multi_pct(t32, clip_range)

    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp

    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]

    try:
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        return _quantize_int6_multi_pct(t32, clip_range)

    best_q, best_scale, best_err = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()

        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()

        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse

    best_q = best_q[:, inv_perm]
    return best_q, best_scale


def _quantize_int6_multi_pct(t32: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    """Fallback: naive INT6 with multi-percentile search for best clip."""
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    # scalar/vector
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def collect_hessians(base_model: nn.Module, train_loader, args, device, grad_accum_steps,
                     num_batches: int = 256) -> dict[str, Tensor]:
    """Collect Hessian H = X^T X per CastedLinear layer for GPTQ."""
    hessians: dict[str, Tensor] = {}
    hooks = []

    for name, module in base_model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')

            def make_hook(pname, ncols):
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    xtx = (x.T @ x).cpu()
                    hessians[pname] += xtx
                return hook_fn
            h = module.register_forward_hook(make_hook(param_name, cols))
            hooks.append(h)

    base_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(num_batches):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            _ = base_model(x, y)

    for h in hooks:
        h.remove()
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H

    base_model.train()
    return hessians


def quantize_state_dict_int6_gptq(state_dict: dict[str, Tensor],
                                   hessians: dict[str, Tensor] | None = None,
                                   block_size: int = 128,
                                   prune_pct: float = 0.0):
    """Mixed INT6+GPTQ quantization: INT8 for embeddings, INT6+GPTQ for attn/mlp."""
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int6_payload_bytes", "gptq_layers", "naive_layers", "embed_layers"), 0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int6_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += tensor_nbytes(kept)
            continue

        cat = _classify_param(name)
        stats["num_float_tensors"] += 1

        if cat == "embed":
            # Embeddings get INT8 (127 levels) — more precision for critical params
            stats["embed_layers"] += 1
            q, s = quantize_float_tensor(t)  # existing INT8 quantizer
            if s.ndim > 0:
                qmeta[name] = {"scheme": "per_row", "axis": 0}
            quantized[name] = q
            scales[name] = s
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["int6_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
        else:
            # Everything else gets INT6+GPTQ
            H = hessians.get(name) if hessians else None
            if H is not None:
                stats["gptq_layers"] += 1
                q, s = quantize_int6_gptq(t, hessian=H, block_size=block_size)
            else:
                stats["naive_layers"] += 1
                q, s = _quantize_int6_multi_pct(t.float())
            if s.ndim > 0:
                qmeta[name] = {"scheme": "per_row", "axis": 0}
            quantized[name] = q
            scales[name] = s
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["int6_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    # Post-quant pruning: zero out smallest INT6 weights for better compression
    if prune_pct > 0:
        all_int6_vals = []
        for name in quantized:
            cat = _classify_param(name)
            if cat != "embed":
                all_int6_vals.append(quantized[name].flatten().abs().float())
        if all_int6_vals:
            all_vals = torch.cat(all_int6_vals)
            k = max(1, int(prune_pct * all_vals.numel()))
            threshold = all_vals.kthvalue(k).values.item()
            pruned_count = 0
            for name in quantized:
                cat = _classify_param(name)
                if cat != "embed":
                    mask = quantized[name].abs() <= int(threshold)
                    pruned_count += mask.sum().item()
                    quantized[name][mask] = 0
            stats["pruned_count"] = pruned_count
            stats["prune_threshold"] = threshold

    obj: dict[str, object] = {
        "__quant_format__": "int6_gptq_mixed_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def compress_bytes(data: bytes) -> bytes:
    """Compress using zstd-22 if available, otherwise zlib-9."""
    if _HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        return b"ZSTD" + cctx.compress(data)
    return b"ZLIB" + zlib.compress(data, level=9)


def decompress_bytes(data: bytes) -> bytes:
    """Decompress, auto-detecting zstd vs zlib from header."""
    if data[:4] == b"ZSTD":
        if not _HAS_ZSTD:
            raise RuntimeError("zstandard package required to decompress ZSTD data")
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data[4:])
    if data[:4] == b"ZLIB":
        return zlib.decompress(data[4:])
    return zlib.decompress(data)


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
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
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
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
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
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
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    # QAT: when _qat_enabled[0] is nonzero, fake-quantize weights during training (STE).
    # Using a tensor so torch.compile treats it as dynamic (not a compile-time constant).
    _qat_enabled: Tensor = torch.tensor([0], dtype=torch.int32)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / INT6_MAX_VAL).clamp_min(1.0 / INT6_MAX_VAL)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -INT6_MAX_VAL, INT6_MAX_VAL) * scale[:, None]).to(x.dtype)
            # STE: forward uses quantized, backward passes through
            # Blend: when _qat_enabled=0, alpha=0 so w_q contribution is zero (just w)
            alpha = CastedLinear._qat_enabled[0].float()
            w = w + alpha * (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
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
            or self._cos_cached.is_inference()  # stale from inference_mode eval
        ):
            with torch.no_grad():
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
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
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
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
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
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
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


class TransformerStage(nn.Module):
    """
    Same-resolution stage:
    - first half of blocks store skip activations
    - second half consume them in reverse order
    - final RMSNorm at the end

    GPT backbone factored into a reusable module.
    """
    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)

        if self.num_skip_weights > 0:
            self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, dim, dtype=torch.float32))
        else:
            self.skip_weights = nn.Parameter(torch.empty(0, dim, dtype=torch.float32))

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_mult=mlp_mult,
                    rope_base=rope_base,
                    qk_gain_init=qk_gain_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()

    def forward(self, x: Tensor) -> Tensor:
        if len(self.blocks) == 0:
            return x

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

        return self.final_norm(x)


def straight_through(soft: Tensor, hard: Tensor) -> Tensor:
    return soft + (hard - soft).detach()

def ste_to_one(x: Tensor) -> Tensor:
    # Forward value is 1, gradient wrt x is 1.
    return straight_through(x, torch.ones_like(x))

def frac_gradient(t: Tensor, frac: float = 1.0) -> Tensor:
    """Scale gradients by frac in backward, identity in forward."""
    if frac == 1.0:
        return t
    return straight_through(t * frac, t)

class RoutingModule(nn.Module):
    """
    Paper Eq. 4:
      q_t = W_q xhat_t
      k_t = W_k xhat_t
      p_t = 0.5 * (1 - cos(q_t, k_{t-1}))
      b_t = 1[p_t >= 0.5]
    with p_1 = 1.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = CastedLinear(d_model, d_model, bias=False)
        self.k_proj = CastedLinear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d_model))
            self.k_proj.weight.copy_(torch.eye(d_model))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = F.normalize(self.q_proj(x[:, 1:]), dim=-1)   # [B, L-1, D]
        k = F.normalize(self.k_proj(x[:, :-1]), dim=-1)  # [B, L-1, D]
        cos_sim = torch.einsum("bld,bld->bl", q, k)      # [B, L-1]

        p = ((1.0 - cos_sim) * 0.5).clamp(0.0, 1.0)
        p = F.pad(p, (1, 0), value=1.0)                  # [B, L]
        boundary_mask = p >= 0.5                         # [B, L]

        confidence = torch.where(boundary_mask, p, 1.0 - p)  # [B, L]
        return p, boundary_mask, confidence


class ChunkLayer(nn.Module):
    """
    Direct selection downsampler:
    keep positions where b_t = 1, gather them to the front, and right-pad with zeros.
    Uses a fixed max_chunks = L // CHUNK_DIVISOR for torch.compile compatibility.
    """

    def __init__(self, chunk_divisor: int = 4):
        super().__init__()
        self.CHUNK_DIVISOR = chunk_divisor

    def forward(self, hidden_states: Tensor, boundary_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            hidden_states: [B, L, D]
            boundary_mask: [B, L] (bool)

        Returns:
            chunked:   [B, max_chunks, D]  (max_chunks = L // CHUNK_DIVISOR)
            pad_mask:  [B, max_chunks] (bool)
            take_idx:  [B, max_chunks] original positions selected
        """
        _, L, D = hidden_states.shape
        max_chunks = L // self.CHUNK_DIVISOR  # fixed at compile time

        num_chunks = boundary_mask.sum(dim=-1)  # [B]
        num_chunks = num_chunks.clamp(max=max_chunks)

        token_idx = torch.arange(L, device=hidden_states.device)[None, :] + (~boundary_mask).long() * L
        sorted_idx = torch.argsort(token_idx, dim=1)
        take_idx = sorted_idx[:, :max_chunks]

        chunked = torch.gather(
            hidden_states,
            dim=1,
            index=take_idx.unsqueeze(-1).expand(-1, -1, D),
        )

        pad_mask = torch.arange(max_chunks, device=hidden_states.device)[None, :] < num_chunks[:, None]
        chunked = chunked * pad_mask.unsqueeze(-1).to(chunked.dtype)

        return chunked, pad_mask, take_idx


class DeChunkLayer(nn.Module):
    """
    Paper dechunking:
    1. smooth chunk states with parallel EMA via associative_scan (same-shape tuple)
    2. plug chunk states back to original sequence positions by cumulative chunk id
    """
    def forward(
        self,
        hidden_states: Tensor,   # [B, C, D] output of main stage on chunked sequence
        boundary_mask: Tensor,   # [B, L] hard boundaries on original sequence
        boundary_prob: Tensor,   # [B, L] p_t on original sequence
        take_idx: Tensor,        # [B, C] cached indices from ChunkLayer
    ) -> Tensor:
        B, L = boundary_mask.shape
        _, C, D = hidden_states.shape

        p_chunked = torch.gather(boundary_prob, dim=1, index=take_idx)  # [B, C]
        p_chunked = p_chunked.clamp(1e-4, 1.0 - 1e-4)

        # Parallel EMA via associative scan.
        # Recurrence: s_t = p_t * x_t + (1 - p_t) * s_{t-1}
        # Pack into single tensor: row = [decay, weighted_x_1, ..., weighted_x_D]
        # so both elements of the tuple have identical shape [B, C, D].
        decay = (1.0 - p_chunked).unsqueeze(-1).expand_as(hidden_states)  # [B, C, D]
        weighted_x = p_chunked.unsqueeze(-1) * hidden_states              # [B, C, D]

        # Fix position 0: decay=0 and weighted_x=x_0 so s_0 = x_0 exactly.
        # Without this, the clamp(1e-4, 1-1e-4) makes decay[:,0]=1e-4 instead of 0,
        # contaminating the first chunk state.
        decay = decay.clone()
        weighted_x = weighted_x.clone()
        decay[:, 0] = 0.0
        weighted_x[:, 0] = hidden_states[:, 0]

        # combine_fn: (d1, v1) o (d2, v2) = (d1*d2, v2 + d2*v1)
        def combine_fn(left: tuple[Tensor, Tensor], right: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
            d_left, v_left = left
            d_right, v_right = right
            return (d_left * d_right, v_right + d_right * v_left)

        _, smoothed = associative_scan(combine_fn, (decay, weighted_x), dim=1)

        # Plug chunk states back to original positions
        chunk_id = torch.cumsum(boundary_mask.long(), dim=1) - 1  # [B, L]
        chunk_id = chunk_id.clamp(max=C - 1)  # prevent OOB when boundaries > max_chunks
        out = torch.gather(
            smoothed,
            dim=1,
            index=chunk_id.unsqueeze(-1).expand(-1, -1, D),
        )
        return out

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        outer_layers: int,
        target_avg_chunk_len: float,
        ratio_loss_weight: float,
        hnet_lr_diff: float = 0.75,
        chunk_divisor: int = 4,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.hnet_lr_diff = hnet_lr_diff
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        main_layers = num_layers - 2 * outer_layers
        if main_layers < 0:
            raise ValueError(
                f"NUM_LAYERS must be >= 2 * OUTER_LAYERS, got "
                f"NUM_LAYERS={num_layers}, OUTER_LAYERS={outer_layers}"
            )

        self.main_stage = TransformerStage(
            num_layers=main_layers,
            dim=model_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult,
            rope_base=rope_base,
            qk_gain_init=qk_gain_init,
        )

        self.outer_layers = outer_layers
        self.main_layers = main_layers
        self.use_hnet = outer_layers > 0
        if self.use_hnet:
            self.encoder_stage = TransformerStage(
                num_layers=outer_layers,
                dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_mult=mlp_mult,
                rope_base=rope_base,
                qk_gain_init=qk_gain_init,
            )
            self.decoder_stage = TransformerStage(
                num_layers=outer_layers,
                dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_mult=mlp_mult,
                rope_base=rope_base,
                qk_gain_init=qk_gain_init,
            )
            self.routing0 = RoutingModule(model_dim)
            self.chunk0 = ChunkLayer(chunk_divisor=chunk_divisor)
            self.residual_proj0 = CastedLinear(model_dim, model_dim, bias=False)
            self.residual_proj0._zero_init = True
            self.dechunk0 = DeChunkLayer()
            self.target_avg_chunk_len = target_avg_chunk_len
            self.ratio_loss_weight = ratio_loss_weight

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

    def _compute_ratio_loss(self, boundary_mask: Tensor, boundary_prob: Tensor) -> Tensor:
        """
        Encourage the average boundary rate to match a target average chunk length.
        boundary_mask: [B, L] bool
        boundary_prob: [B, L] float
        """
        N = self.target_avg_chunk_len
        if N <= 1.0:
            raise ValueError(f"TARGET_AVG_CHUNK_LEN must be > 1, got {N}")

        F_val = boundary_mask.float().mean(dim=-1)   # hard boundary rate per example
        G_val = boundary_prob.mean(dim=-1)           # soft boundary rate per example

        ratio_loss = N / (N - 1.0) * (
            (N - 1.0) * F_val * G_val + (1.0 - F_val) * (1.0 - G_val)
        )
        return ratio_loss.mean() * self.ratio_loss_weight

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids) # [BATCH_SIZE, SEQ_LEN, MODEL_DIM] = [B, T, D]

        if self.use_hnet:
            # E -> M -> D (1 stage)
            # Encode
            x = self.encoder_stage(x)
            encoder_out = x

            # Routing and chunking
            p0, bmask0, conf0 = self.routing0(x)
            chunked0, pad_mask0, take_idx0 = self.chunk0(x, bmask0)


            chunked0 = frac_gradient(chunked0, self.hnet_lr_diff ** -1)

            # Main stage on chunked sequence
            chunked0 = self.main_stage(chunked0)

            dechunked0 = self.dechunk0(chunked0, bmask0, p0, take_idx0) # Dechunk (upsample)

            ste_conf0 = ste_to_one(conf0).unsqueeze(-1)  # [B, L, 1]
            x = ste_conf0 * dechunked0 + self.residual_proj0(encoder_out.to(self.residual_proj0.weight.dtype))


            x = frac_gradient(x, self.hnet_lr_diff)
            x = x.to(self.tok_emb.weight.dtype) # Cast back to embedding dtype for decoder stage

            # Decoder
            x = self.decoder_stage(x)

            aux_loss = self._compute_ratio_loss(bmask0, p0) if self.training else None

        else:
            x = self.main_stage(x)
            aux_loss = None


        x = x.reshape(-1, x.size(-1)) # x.size(-1) = MODEL_DIM, reshape to [B*T, D] for LM head
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        return ce_loss + aux_loss if aux_loss is not None else ce_loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning logits [B, T, V] instead of loss. Used for sliding window eval."""
        x = self.tok_emb(input_ids)

        if self.use_hnet:
            x = self.encoder_stage(x)
            encoder_out = x
            p0, bmask0, conf0 = self.routing0(x)
            chunked0, pad_mask0, take_idx0 = self.chunk0(x, bmask0)
            chunked0 = frac_gradient(chunked0, self.hnet_lr_diff ** -1)
            chunked0 = self.main_stage(chunked0)
            dechunked0 = self.dechunk0(chunked0, bmask0, p0, take_idx0)
            ste_conf0 = ste_to_one(conf0).unsqueeze(-1)
            x = ste_conf0 * dechunked0 + self.residual_proj0(encoder_out.to(self.residual_proj0.weight.dtype))
            x = frac_gradient(x, self.hnet_lr_diff)
            x = x.to(self.tok_emb.weight.dtype)
            x = self.decoder_stage(x)
        else:
            x = self.main_stage(x)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if args.compile_model:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

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

    # Fast math knobs
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
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_byte_luts(args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_kind=byte vocab_size={args.vocab_size}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

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
        outer_layers=args.outer_layers,
        target_avg_chunk_len=args.target_avg_chunk_len,
        ratio_loss_weight=args.ratio_loss_weight,
        hnet_lr_diff=args.hnet_lr_diff,
        chunk_divisor=args.chunk_divisor,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    CastedLinear._qat_enabled = torch.tensor([0], dtype=torch.int32, device=device)
    compiled_model = torch.compile(base_model, dynamic=False) if args.compile_model else base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    backbone_named_params = (
        [(f"main_stage.{n}", p) for n, p in base_model.main_stage.named_parameters()]
    )
    if base_model.use_hnet:
        backbone_named_params += (
            [(f"encoder_stage.{n}", p) for n, p in base_model.encoder_stage.named_parameters()] +
            [(f"decoder_stage.{n}", p) for n, p in base_model.decoder_stage.named_parameters()] +
            [(f"routing0.{n}", p) for n, p in base_model.routing0.named_parameters()] +
            [(f"residual_proj0.{n}", p) for n, p in base_model.residual_proj0.named_parameters()]
        )
    matrix_params = [
        p for name, p in backbone_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]

    scalar_params = [
        p for name, p in backbone_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
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
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"outer_layers:{args.outer_layers} main_layers:{base_model.main_layers} chunk_divisor:{args.chunk_divisor}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    gptq_reserve_ms = 1000.0 * args.gptq_reserve_seconds
    train_loop_cap_ms = max_wallclock_ms - gptq_reserve_ms if max_wallclock_ms is not None else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if train_loop_cap_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(train_loop_cap_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
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

    # EMA: initialize from current model weights (disabled when decay <= 0)
    ema_enabled = args.ema_decay > 0
    ema_state = None
    if ema_enabled:
        ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
        log0(f"ema:initialized with decay={args.ema_decay}")
    else:
        log0("ema:disabled")

    # QAT: will be enabled dynamically when LR scale drops below threshold
    if args.late_qat_threshold > 0:
        log0(f"late_qat:will enable when lr_scale < {args.late_qat_threshold}")

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
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            # if hasattr(base_model, "_debug_last_chunk_mask"):
            #     avg_chunk_rate = base_model._debug_last_chunk_mask.float().mean().item()
            #     avg_chunk_len = 1.0 / max(avg_chunk_rate, 1e-8)
            #     log0(
            #         f"chunk_stats step:{step}/{args.iterations} "
            #         f"avg_chunk_rate:{avg_chunk_rate:.4f} avg_chunk_len:{avg_chunk_len:.2f} "
            #         f"chunked_shape:{getattr(base_model, '_debug_last_chunked_shape', None)}"
            #     )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations} (reserving {args.gptq_reserve_seconds}s for GPTQ)"
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

        # EMA update
        if ema_state is not None:
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)

        # Late QAT: enable fake int6 quantization when LR drops low enough
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and CastedLinear._qat_enabled[0].item() == 0:
            CastedLinear._qat_enabled[0] = 1
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")

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

        # Periodic checkpoint saving (rank 0 only)
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0 and rank == 0:
            ckpt_path = str(Path(args.model_save_path).parent / f"{Path(args.model_save_path).stem}.step{step}.pt")
            torch.save(base_model.state_dict(), ckpt_path)
            log0(f"checkpoint:saved {ckpt_path} at step {step}")

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = train_loop_cap_ms is not None and approx_training_time_ms >= train_loop_cap_ms
        if distributed and train_loop_cap_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + GPTQ + ROUNDTRIP VALIDATION
    # -----------------------------

    # Disable QAT for serialization
    CastedLinear._qat_enabled[0] = 0

    # Apply EMA weights before saving (if enabled)
    if ema_state is not None:
        log0("ema:applying EMA weights")
        current_dtypes = {name: t.dtype for name, t in base_model.state_dict().items()}
        avg_state = {name: t.to(dtype=current_dtypes[name]) for name, t in ema_state.items()}
        del ema_state
        base_model.load_state_dict(avg_state, strict=True)

    # GPTQ calibration: collect Hessians
    torch.cuda.synchronize()
    t_gptq = time.perf_counter()
    log0(f"gptq:calibrating with {args.gptq_calib_batches} batches...")
    calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    hessians = collect_hessians(base_model, calib_loader, args, device, grad_accum_steps,
                                num_batches=args.gptq_calib_batches)
    torch.cuda.synchronize()
    gptq_time_ms = 1000.0 * (time.perf_counter() - t_gptq)
    log0(f"gptq:collected hessians for {len(hessians)} layers in {gptq_time_ms:.0f}ms")

    if master_process:
        torch.save(base_model.state_dict(), args.model_save_path)
        model_bytes = os.path.getsize(args.model_save_path)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    # INT6+GPTQ mixed quantization + zstd compression
    save_stem = Path(args.model_save_path).stem
    artifact_path = str(Path(args.model_save_path).parent / f"{save_stem}.int6.ptz")
    artifact_path_int8 = str(Path(args.model_save_path).parent / f"{save_stem}.int8.ptz")
    quant_obj, quant_stats = quantize_state_dict_int6_gptq(
        base_model.state_dict(),
        hessians=hessians,
        block_size=args.gptq_block_size,
        prune_pct=args.prune_pct,
    )
    if quant_stats.get("pruned_count", 0) > 0:
        log0(f"prune:zeroed {quant_stats['pruned_count']} int6 weights (threshold={quant_stats.get('prune_threshold', 0):.0f})")
    log0(f"gptq:quantized {quant_stats.get('gptq_layers', 0)} layers with GPTQ, "
         f"{quant_stats.get('naive_layers', 0)} naive, {quant_stats.get('embed_layers', 0)} INT8 embeds")
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = compress_bytes(quant_buf.getvalue())
    if master_process:
        with open(artifact_path, "wb") as f:
            f.write(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        compressor = "zstd-22" if _HAS_ZSTD else "zlib-9"
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int6_payload_bytes"], 1)
        total_artifact = len(quant_blob) + code_bytes
        log0(
            f"artifact int6+gptq+{compressor}: {len(quant_blob)} bytes + {code_bytes} code = {total_artifact} total "
            f"(payload:{quant_stats['int6_payload_bytes']} payload_ratio:{ratio:.2f}x)"
        )
        if total_artifact > 16_000_000:
            log0(f"WARNING: artifact {total_artifact} exceeds 16,000,000 byte cap by {total_artifact - 16_000_000} bytes!")
        else:
            log0(f"artifact headroom: {16_000_000 - total_artifact} bytes ({(16_000_000 - total_artifact)/1e6:.3f}MB)")

    # Also produce INT8+zlib for comparison
    quant_obj8, quant_stats8 = quantize_state_dict_int8(base_model.state_dict())
    quant_buf8 = io.BytesIO()
    torch.save(quant_obj8, quant_buf8)
    quant_blob8 = zlib.compress(quant_buf8.getvalue(), level=9)
    if master_process:
        with open(artifact_path_int8, "wb") as f:
            f.write(quant_blob8)
        log0(f"artifact int8+zlib: {len(quant_blob8)} bytes (for comparison)")

    # Roundtrip validation with INT6+GPTQ
    if distributed:
        dist.barrier()
    with open(artifact_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(decompress_bytes(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int6(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_gptq_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_gptq_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Sliding window evaluation (with INT6+GPTQ roundtripped weights already loaded)
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        log0(f"sliding_eval: stride={args.eval_stride} batch_seqs={args.eval_batch_seqs} seq_len={args.train_seq_len}")
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            batch_seqs=args.eval_batch_seqs,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_gptq_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_gptq_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
