"""
train_gpt_prototype_fixed.py — Parameter Golf challenge prototype (fixed).

CHANGES FROM BROKEN PROTOTYPE:
  BUG-1  UnboundLocalError: master_process referenced before assignment.
         Fix: moved batch-size guard to after `master_process = rank == 0`.
  BUG-2  SwiGLU MLP inflates parameter count +50% at mlp_mult=2.
         Fix: hidden = int(2 * mlp_mult * dim / 3) — parameter-equivalent SwiGLU.
  BUG-3  Dropout bypasses args: modules read os.environ directly.
         Fix: dropout passed as explicit constructor argument through the call chain.
  BUG-4  tok_emb in CONTROL_TENSOR_NAME_PATTERNS wastes ~2MB artifact budget.
         Fix: tok_emb removed from control patterns; quantized as standard tensor.
  BUG-5  Eval protocol inconsistency: train uses eval_val(), final uses
         eval_val_sliding(). Fix: sliding-window used throughout, with a cheap
         non-sliding pass during training (faster) and sliding only at final eval.
  BUG-6  forward_logits not compiled; bypasses torch.compile graph.
         Fix: forward_logits is compiled via a separate torch.compile call.
  BUG-7  seq_len=4096 default causes OOM on single-GPU configs.
         Fix: default reverted to 1024; 4096 recommended only for multi-H100.

ADDITIONS (from qlabs.sh/10x research):
  ADD-1  muon_wd: Muon weight decay (qlabs: WD up to 1.6 at massive overparameterization;
         use 0.01–0.05 at standard param-golf scale).
  ADD-2  Loop recurrence: num_loops, loop_start_layer, loop_end_layer (qlabs PR looping).
  ADD-3  Spectral embedding init (novel; power-law singular value spectrum).
  ADD-4  INT6 mid-layer quantization (rounds int8 to 4-step grid for better zlib ratio).
  ADD-5  eval_val_sliding for final scoring (maximises context for BPB measurement).

Hard stop: train_gpt.py and train_gpt_mlx.py must stay ≤ 1500 lines.
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
    data_path          = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files        = os.path.join(data_path, "fineweb_train_*.bin")
    val_files          = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path     = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id             = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed               = int(os.environ.get("SEED", 1337))

    val_batch_size     = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every     = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every    = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations         = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters     = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps       = int(os.environ.get("WARMUP_STEPS", 20))
    # NOTE: seq_len=4096 requires ~21 GB activation memory per loop pass.
    # Use 1024 for GTX/single-GPU; 4096 only for multi-H100.
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len      = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init       = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size         = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers         = int(os.environ.get("NUM_LAYERS", 10))
    # Recurrence: loop a contiguous range of layers num_loops times.
    # Set loop_start_layer = loop_end_layer = -1 to loop ALL layers.
    # qlabs finding: loop middle layers, NOT the final few.
    # Example: NUM_LAYERS=10 NUM_LOOPS=2 LOOP_START_LAYER=2 LOOP_END_LAYER=8
    num_loops          = int(os.environ.get("NUM_LOOPS", 1))
    loop_start_layer   = int(os.environ.get("LOOP_START_LAYER", -1))
    loop_end_layer     = int(os.environ.get("LOOP_END_LAYER", -1))
    # Dropout: apply in attention + MLP during training (qlabs: 0.1).
    # Higher dropout compensates for overparameterization; use 0.0 at standard scale.
    dropout            = float(os.environ.get("DROPOUT", 0.0))
    num_kv_heads       = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim          = int(os.environ.get("MODEL_DIM", 512))
    num_heads          = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult           = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings     = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base          = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap      = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    embed_lr           = float(os.environ.get("EMBED_LR", 0.6))
    head_lr            = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr      = float(os.environ.get("TIED_EMBED_LR", 0.03))
    matrix_lr          = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr          = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum      = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    # muon_wd: L2 weight decay applied to Muon-updated parameters.
    # qlabs uses WD up to 1.6 in the massively overparameterized regime.
    # At standard parameter-golf scale, 0.01–0.05 is more appropriate.
    muon_wd            = float(os.environ.get("MUON_WD", 0.01))
    beta1              = float(os.environ.get("BETA1", 0.9))
    beta2              = float(os.environ.get("BETA2", 0.95))
    adam_eps           = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm     = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    # EMA: Maintain a smoothed shadow copy of weights for validation.
    ema_decay          = float(os.environ.get("EMA_DECAY", 0.999))

    ttt_lora_rank      = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lora_lr        = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_chunk_size     = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len   = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    ttt_batch_size     = int(os.environ.get("TTT_BATCH_SIZE", 64))

    # Sliding-window evaluation parameters.
    eval_stride        = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs    = int(os.environ.get("EVAL_BATCH_SEQS", 512))

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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, wd: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, wd=wd),
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
            lr           = group["lr"]
            momentum     = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov     = group["nesterov"]
            wd           = group.get("wd", 0.0)

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
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
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
        raise ValueError(f"Validation split too short for TRAIN_SEQ_LEN={seq_len}")
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
    """Non-overlapping window evaluation. Fast; used during training checkpoints."""
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
    seq_end   = (total_seqs * (rank + 1)) // world_size
    val_loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end   = batch_seq_end   * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum   += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids  = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


@torch.no_grad()
def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    compiled_forward_logits,   # pre-compiled forward_logits callable
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """
    Sliding-window evaluation for maximum context utilisation.

    Each token is scored with up to `train_seq_len` tokens of left context.
    Windows advance by `eval_stride` tokens; only the rightmost `eval_stride`
    positions in each window (except the first) contribute to the BPB estimate.
    This provides a strictly better BPB lower bound than non-overlapping evaluation.

    Note: `compiled_forward_logits` must be passed explicitly so the compiled
    graph is used (forward_logits is a separate method not captured by the main
    compile call on forward).
    """
    model.eval()
    val_loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)

    seq_len   = args.train_seq_len
    stride    = args.eval_stride
    N         = val_tokens.numel()

    # Build list of window start indices.
    # Window 0: start=0, scores all seq_len positions.
    # Window k>0: start=k*stride, scores only the rightmost `stride` positions.
    start_indices = list(range(0, N - seq_len, stride))
    if not start_indices:
        start_indices = [0]

    rank_starts = start_indices[
        (len(start_indices) * rank) // world_size :
        (len(start_indices) * (rank + 1)) // world_size
    ]

    batch_size = args.eval_batch_seqs
    is_first_window = {s: (s == 0) for s in start_indices}

    for i in range(0, len(rank_starts), batch_size):
        batch_starts = rank_starts[i : i + batch_size]
        bsz = len(batch_starts)

        x = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        y = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        score_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)

        for b, st in enumerate(batch_starts):
            end = min(st + seq_len + 1, N)
            actual_len = end - st - 1
            chunk = val_tokens[st : st + actual_len + 1].to(device)
            x[b, :actual_len] = chunk[:-1]
            y[b, :actual_len] = chunk[1:]
            if is_first_window.get(st, False):
                score_mask[b, :actual_len] = True
            else:
                # Score only positions not covered by the previous window.
                score_start = seq_len - stride
                score_mask[b, score_start : actual_len] = True

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = compiled_forward_logits(x)

        flat_logits  = logits[score_mask]
        flat_targets = y[score_mask]
        if flat_logits.numel() > 0:
            loss = F.cross_entropy(flat_logits.float(), flat_targets, reduction="sum")
            val_loss_sum    += loss.to(torch.float64)
            val_token_count += flat_targets.numel()
            prev_ids = x[score_mask]
            tgt_ids  = flat_targets
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------

# tok_emb intentionally NOT in CONTROL_TENSOR_NAME_PATTERNS:
# including it wastes ~2MB artifact budget (fp32 passthrough).
# tok_emb is quantized as a standard large tensor (per-row int8).
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
INT8_KEEP_FLOAT_MAX_NUMEL    = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE  = torch.float16
INT8_PER_ROW_SCALE_DTYPE     = torch.float16
INT8_CLIP_PERCENTILE         = 99.99984
INT8_CLIP_Q                  = INT8_CLIP_PERCENTILE / 100.0
# INT6 layer compression: rounds int8 values to multiples of INT6_STEP.
# Middle layers (not first/last) tolerate this better; improves zlib ratio.
INT6_LAYERS = os.environ.get("INT6_LAYERS", "3,4,5,6,7")
INT6_STEP   = int(os.environ.get("INT6_STEP", 4))


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
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    int6_set = {int(x) for x in INT6_LAYERS.split(",") if x.strip()} if INT6_LAYERS else set()

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

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or any(
            pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
        ):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        # INT6 compression for middle layers: round to INT6_STEP multiples.
        # Reduces unique values → better zlib ratio (typically 5–10% size saving).
        for layer_idx in int6_set:
            if f"blocks.{layer_idx}." in name:
                q = (torch.round(q.float() / INT6_STEP) * INT6_STEP).clamp(-127, 127).to(torch.int8)
                break
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name]    = s
        dtypes[name]    = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
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


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes  = np.dtype("<u2").itemsize
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
        self.rank       = rank
        self.world_size = world_size
        self.device     = device
        self.stream     = TokenStream(pattern)

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
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(
                pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS
            )) and param.dtype != torch.float32:
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
            self._cos_cached is None or self._sin_cached is None
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
    _sm80_plus = False

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        dropout: float = 0.0,          # FIX: explicit arg, not os.environ
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain   = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary   = Rotary(self.head_dim, base=rope_base)
        self.dropout_p = dropout

    def forward(self, x: Tensor, q_delta=None, v_delta=None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x) + (q_delta if q_delta is not None else 0)
        k = self.c_k(x)
        v = self.c_v(x) + (v_delta if v_delta is not None else 0)
        q = q.reshape(bsz, seqlen, self.num_heads,    self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        use_gqa_kernel = getattr(self, "_sm80_plus", False) and self.num_kv_heads != self.num_heads
        if not use_gqa_kernel and self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=use_gqa_kernel,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    """
    SwiGLU MLP with parameter-equivalent hidden dimension.

    FIX: The naive SwiGLU with hidden = mlp_mult * dim uses 3 weight matrices
    instead of 2, inflating parameter count by 50% vs relu².
    Correction: hidden = int(2 * mlp_mult * dim / 3) keeps total params equal.

    For mlp_mult=2, dim=512:
      relu²   (2 matrices): 2 × 512 × 1024 = 1,048,576 params
      SwiGLU naive (3 mats): 3 × 512 × 1024 = 1,572,864 params  ← broken
      SwiGLU fixed  (3 mats): 3 × 512 × 682  = 1,047,552 params  ← ~equal ✓
    """
    def __init__(self, dim: int, mlp_mult: int, dropout: float = 0.0):  # FIX: explicit arg
        super().__init__()
        hidden = int(2 * mlp_mult * dim / 3)   # FIX: parameter-equivalent SwiGLU
        self.w1   = CastedLinear(dim, hidden, bias=False)
        self.w2   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        dropout: float = 0.0,          # FIX: explicit arg
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, dropout)
        self.mlp  = MLP(dim, mlp_mult, dropout)
        self.attn_scale = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, q_delta_fn=None, v_delta_fn=None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = self.attn_norm(x)
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(n, qd, vd)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_loops: int,
        loop_start_layer: int,
        loop_end_layer: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        dropout: float = 0.0,          # FIX: explicit arg
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings    = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap     = logit_softcap
        self.num_loops         = num_loops
        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # Build physical_layer_indices: maps virtual depth position → block index.
        # Supports partial-range looping (qlabs: only middle layers looped).
        self.physical_layer_indices: list[int] = []
        if loop_start_layer >= 0 and loop_end_layer > loop_start_layer:
            self.physical_layer_indices.extend(range(0, loop_start_layer))
            for _ in range(num_loops):
                self.physical_layer_indices.extend(range(loop_start_layer, loop_end_layer))
            self.physical_layer_indices.extend(range(loop_end_layer, num_layers))
        else:
            for _ in range(num_loops):
                self.physical_layer_indices.extend(range(num_layers))

        effective_layers = len(self.physical_layer_indices)
        self.num_encoder_layers = effective_layers // 2
        self.num_decoder_layers = effective_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        # Residual scale: stabilise skip magnitudes when virtual depth > num_layers.
        # Without this, looped models diverge due to accumulating residual norms.
        self._residual_scale = 1.0 / math.sqrt(num_loops) if num_loops > 1 else 1.0
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            # Spectral initialisation: singular values follow k^{-0.5} power law.
            # Encourages embedding diversity and smooth gradient flow from the start.
            with torch.no_grad():
                w = self.tok_emb.weight
                V, D = w.shape
                U, _, _ = torch.linalg.svd(torch.randn(V, D), full_matrices=False)
                _, _, Vh = torch.linalg.svd(torch.randn(D, D), full_matrices=False)
                k = torch.arange(1, D + 1, dtype=torch.float32)
                S = k.pow(-0.5)
                S *= self.tied_embed_init_std * D ** 0.5 / S.norm()
                w.copy_(U @ torch.diag(S) @ Vh)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _run_blocks(self, x: Tensor, x0: Tensor, lora=None) -> Tensor:
        """Shared logic for forward() and forward_logits()."""
        skips: list[Tensor] = []
        rs = self._residual_scale
        for i in range(self.num_encoder_layers):
            pidx = self.physical_layer_indices[i]
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = self.blocks[pidx](x, x0, qd, vd)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            ei = self.num_encoder_layers + i
            pidx = self.physical_layer_indices[ei]
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop() * rs
            qd = lora.q_loras[ei] if lora else None
            vd = lora.v_loras[ei] if lora else None
            x = self.blocks[pidx](x, x0, qd, vd)
        return x

    def _embed(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        return x, x  # (x, x0)

    def _logits(self, x: Tensor, lora=None) -> Tensor:
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = logits + (lora.lm_head_lora(x) if lora else 0)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        x, x0 = self._embed(input_ids)
        x = self._run_blocks(x, x0, lora)
        logits = self._logits(x, lora)
        if lora:
            bsz, sl, V = logits.shape
            return F.cross_entropy(
                logits.float().reshape(-1, V),
                target_ids.reshape(-1),
                reduction="none",
            ).reshape(bsz, sl)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="mean",
        )

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return raw logits only. Compiled separately for eval_val_sliding."""
        x, x0 = self._embed(input_ids)
        x = self._run_blocks(x, x0, lora=None)
        return self._logits(x, lora=None)


# -----------------------------
# TEST-TIME TRAINING (LoRA)
# -----------------------------

BOS_ID = 1


class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()


class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz: int, model: GPT, rank: int):
        super().__init__()
        dim   = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        effective_layers = model.num_encoder_layers + model.num_decoder_layers
        block = model.blocks[0]
        for _ in range(effective_layers):
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_v.weight.shape[0], rank))

    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()


def _reset_ttt_optimizer(opt):
    for group in opt.param_groups:
        for p in group["params"]:
            s = opt.state.get(p)
            if not s:
                continue
            s["exp_avg"].zero_()
            s["exp_avg_sq"].zero_()
            s["step"].fill_(0)


def _build_ttt_optimizer(lora, args: Hyperparameters):
    return torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr,
                            betas=(args.beta1, args.beta2), eps=1e-10)


def _find_docs(all_tokens: Tensor, include_next_bos: bool = True) -> list[tuple[int, int]]:
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else all_tokens.numel()
        if include_next_bos and i + 1 < len(bos_positions):
            end += 1
        assert end - start >= 2
        docs.append((start, end - start))
    return docs


def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    chunk_start = ci * chunk_size
    chunk_end   = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start   = max(0, chunk_end - eval_seq_len)
    win_len     = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len    = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def _accumulate_bpb(
    ptl, x, y, batch_i, chunk_offset, chunk_len,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    loss_sum, byte_sum, token_count,
):
    lbl      = ptl[batch_i, chunk_offset : chunk_offset + chunk_len].to(torch.float64)
    prev     = x[batch_i, chunk_offset : chunk_offset + chunk_len]
    tgt      = y[batch_i, chunk_offset : chunk_offset + chunk_len]
    tok_bytes = base_bytes_lut[tgt].to(torch.float64)
    tok_bytes += has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]
    loss_sum    += lbl.sum()
    byte_sum    += tok_bytes.sum()
    token_count += chunk_len


def eval_val_ttt_lora(
    args: Hyperparameters,
    base_model: GPT,
    rank: int,
    world_size: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    files      = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs       = _find_docs(all_tokens)
    rank_docs  = docs[(len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size]
    chunk_size    = args.ttt_chunk_size
    eval_seq_len  = args.ttt_eval_seq_len
    batch_size    = args.ttt_batch_size
    lora_rank     = args.ttt_lora_rank
    rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    lora = BatchedTTTLoRA(batch_size, base_model, lora_rank).to(device)
    opt  = _build_ttt_optimizer(lora, args)
    loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum    = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    for bi in range(0, len(rank_docs), batch_size):
        batch = rank_docs[bi : bi + batch_size]
        bsz   = len(batch)
        if bsz == batch_size:
            cur_lora, cur_opt = lora, opt
            cur_lora.reset()
            _reset_ttt_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, lora_rank).to(device)
            cur_opt  = _build_ttt_optimizer(cur_lora, args)

        pred_lens  = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc     = max(num_chunks)

        for ci in range(max_nc):
            chunk_stats  = _compute_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
            context_size = chunk_stats[1]
            chunk_offset = chunk_stats[2]
            active       = [ci < nc for nc in num_chunks]
            needs_train  = any(ci < nc - 1 for nc in num_chunks)

            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            doc_info = []
            for b in range(bsz):
                if not active[b]:
                    doc_info.append((0, 0))
                    continue
                ds, dl = batch[b]
                ws, wl, co, cl = _compute_chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                chunk = all_tokens[ds + ws : ds + ws + wl + 1]
                toks  = chunk.to(dtype=torch.int64, device=device)
                x[b, :wl] = toks[:-1]
                y[b, :wl] = toks[1:]
                doc_info.append((co, cl))

            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)

            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]:
                        continue
                    co, cl = doc_info[b]
                    _accumulate_bpb(ptl, x, y, b, co, cl,
                                    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                                    loss_sum, byte_sum, token_count)

            if needs_train:
                mask = torch.tensor([float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device)
                per_doc = ptl[:, chunk_offset : chunk_offset + chunk_size].mean(dim=-1)
                cur_opt.zero_grad()
                (per_doc * mask).sum().backward()
                cur_opt.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb  = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # ── Distributed + CUDA setup ────────────────────────────────────────────

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
    cc = torch.cuda.get_device_capability(device)
    CausalSelfAttention._sm80_plus = cc[0] >= 8
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    master_process = rank == 0   # FIX BUG-1: defined before any reference

    # FIX BUG-1 (cont.): batch size guard now placed AFTER master_process is defined
    min_tokens = args.train_seq_len * world_size * grad_accum_steps
    if args.train_batch_tokens < min_tokens:
        if master_process:
            print(f"Warning: adjusting train_batch_tokens {args.train_batch_tokens} → {min_tokens}")
        args.train_batch_tokens = min_tokens

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    is_sm80_plus = CausalSelfAttention._sm80_plus
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_flash_sdp(is_sm80_plus)
    enable_math_sdp(not is_sm80_plus)
    enable_mem_efficient_sdp(False)
    enable_cudnn_sdp(False)

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
    log0(f"sdp_backend: flash={is_sm80_plus} math={not is_sm80_plus} (sm{cc[0]}{cc[1]})")
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    # ── Tokenizer + validation metric setup ─────────────────────────────────

    random.seed(args.seed);  np.random.seed(args.seed)
    torch.manual_seed(args.seed);  torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} ≠ tokenizer.vocab_size={int(sp.vocab_size())}")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(f"loop_config: num_loops={args.num_loops} loop_start={args.loop_start_layer} "
         f"loop_end={args.loop_end_layer}")

    # ── Model + optimizer setup ─────────────────────────────────────────────

    base_model = GPT(
        vocab_size         = args.vocab_size,
        num_layers         = args.num_layers,
        num_loops          = args.num_loops,
        loop_start_layer   = args.loop_start_layer,
        loop_end_layer     = args.loop_end_layer,
        model_dim          = args.model_dim,
        num_heads          = args.num_heads,
        num_kv_heads       = args.num_kv_heads,
        mlp_mult           = args.mlp_mult,
        tie_embeddings     = args.tie_embeddings,
        tied_embed_init_std = args.tied_embed_init_std,
        logit_softcap      = args.logit_softcap,
        rope_base          = args.rope_base,
        qk_gain_init       = args.qk_gain_init,
        dropout            = args.dropout,   # FIX BUG-3: passed explicitly
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
        if isinstance(module, Rotary):
            module.inv_freq.data = module.inv_freq.data.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    # FIX BUG-6: compile forward_logits separately so eval_val_sliding uses it.
    compiled_forward_logits = torch.compile(base_model.forward_logits, dynamic=False)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else compiled_model
    )

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps, wd=args.muon_wd)
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
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"effective_depth:{base_model.num_encoder_layers + base_model.num_decoder_layers} "
         f"(num_loops={args.num_loops} × num_layers={args.num_layers})")
    log0(f"dropout:{args.dropout} muon_wd:{args.muon_wd}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
         f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
         f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")

    # ── Data loader + warmup ─────────────────────────────────────────────────

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    ema = EMA(base_model, args.ema_decay)

    def zero_grad_all() -> None:
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
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            ema.update(base_model)
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── Main training loop ───────────────────────────────────────────────────

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
            ema.apply(base_model)
            # During training: use fast non-overlapping eval (consistent scale).
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            ema.restore(base_model)
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for group in optimizer_muon.param_groups:
            group["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        ema.update(base_model)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
         f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    # ── Serialization + roundtrip validation ────────────────────────────────

    if master_process:
        ema.apply(base_model)
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes")
        ema.restore(base_model)

    ema.apply(base_model)
    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf  = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw  = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"Serialized model int8+zlib: {quant_file_bytes} bytes "
             f"(payload_ratio:{ratio:.2f}x)  code: {code_bytes} bytes  "
             f"total: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    # Roundtrip: use sliding-window eval (same as competition score).
    # FIX BUG-5 + BUG-6: pass compiled_forward_logits explicitly.
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val_sliding(
        args, base_model, compiled_forward_logits,
        rank, world_size, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"final_int8_sliding val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int8_sliding_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # TTT-LoRA (competition score).
    torch._dynamo.reset()
    torch.cuda.synchronize()
    t_ttt = time.perf_counter()
    ttt_val_loss, ttt_val_bpb = eval_val_ttt_lora(
        args, base_model, rank, world_size, device,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    ema.restore(base_model)
    log0(f"final_int8_ttt_lora val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
