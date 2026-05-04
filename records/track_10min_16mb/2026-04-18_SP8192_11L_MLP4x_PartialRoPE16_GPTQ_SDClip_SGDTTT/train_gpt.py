"""
Record submission: 7-Block Depth Recurrence + QK-Gain 4.0 + GPTQ + SGD All-Weights TTT
Track: 10min_16mb (8xH100 SXM)

Architecture: 10 layers (dim=512, 8 heads, 4 KV), no depth recurrence.
XSA on all 10 blocks. SwiGLU 3x MLP. SmearGate, U-Net skips, tied embeddings,
QK-Gain init=4.0.

Post-training: GPTQ (int5 MLP, int6 attn, int8 embed) + zstd-22 compression.
Eval: SGD all-weights TTT (lr=0.005, momentum=0.9, chunk=2048, score_cap=2048).

Run: torchrun --standalone --nproc_per_node=8 train_gpt.py
Multi-seed: SEED=314 / SEED=42 / SEED=999
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

def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride: return data
    src = np.frombuffer(data, dtype=np.uint8); n = len(src)
    out = np.empty(n, dtype=np.uint8); dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]; out[dest_off:dest_off+len(chunk)] = chunk; dest_off += len(chunk)
    return b'BSHF' + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != b'BSHF': return data
    stride = data[4]
    if stride < 2: return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5); n = len(payload)
    out = np.empty(n, dtype=np.uint8); src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off+chunk_len]; src_off += chunk_len
    return out.tobytes()
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default config: 10 layers at width 512
# - 10 unique blocks, no depth recurrence
# - 8 attention heads with 4 KV heads (GQA) and 3x SwiGLU MLP
# - QK-Gain init=4.0, XSA on all blocks, SmearGate, U-Net skips
# - vocab size 1024, sequence length 2048, tied embeddings
# - 524,288 train tokens per step, 10 minute wallclock cap on 8xH100

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 6500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.25))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    activation = os.environ.get("ACTIVATION", "leakyrelu2")  # "swiglu" or "leakyrelu2"

    # N-gram hash embedding.
    ngram_n = int(os.environ.get("NGRAM_N", 0))  # 0 = disabled, 5-7 recommended
    ngram_buckets = int(os.environ.get("NGRAM_BUCKETS", 8192))
    ngram_dim = int(os.environ.get("NGRAM_DIM", 64))

    # XSA config
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))  # XSA on all blocks

    # Depth recurrence config
    # LAYER_SCHEDULE: comma-separated block indices for forward pass order
    # e.g. "0,1,2,3,4,5,3,4,5,6,7,8" = 9 unique blocks, middle 3 shared (looped 2x)
    # Empty string = no recurrence (use num_layers unique blocks as normal)
    layer_schedule_str = os.environ.get("LAYER_SCHEDULE", "")
    layer_schedule = [int(x) for x in layer_schedule_str.split(",") if x.strip()] if layer_schedule_str else []
    # Progressive depth recurrence: phase1 is the initial schedule (typically no recurrence),
    # phase2 activates at RECURRENCE_ACTIVATE_AT fraction of iterations.
    layer_schedule_phase2_str = os.environ.get("LAYER_SCHEDULE_PHASE2", "")
    layer_schedule_phase2 = [int(x) for x in layer_schedule_phase2_str.split(",") if x.strip()] if layer_schedule_phase2_str else []
    recurrence_activate_at = float(os.environ.get("RECURRENCE_ACTIVATE_AT", 0.0))

    # QAT (Quantization-Aware Training) config
    qat_enabled = int(os.environ.get("QAT_ENABLED", 0))
    qat_start_step = int(os.environ.get("QAT_START_STEP", 2000))
    qat_bits_mlp = int(os.environ.get("QAT_BITS_MLP", 5))
    qat_bits_attn = int(os.environ.get("QAT_BITS_ATTN", 6))
    qat_bits_embed = int(os.environ.get("QAT_BITS_EMBED", 8))

    # EMA config
    ema_enabled = int(os.environ.get("EMA_ENABLED", 1))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.022))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    muon_wd = float(os.environ.get("MUON_WD", 0.095))
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))

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
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                wd = group.get("weight_decay", 0.0)
                if wd > 0 and p.ndim >= 2:
                    p.data.mul_(1.0 - lr * wd)
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
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale



# --- Fake quantization for QAT (Quantization-Aware Training) ---

def fake_quantize_ste(w: Tensor, max_val: int) -> Tensor:
    """Simulate int-N quantization with straight-through estimator.
    Forward: returns quantized value. Backward: gradient passes through as if no rounding.
    max_val: e.g. 15 for int5, 31 for int6, 127 for int8.
    """
    if w.ndim == 2:
        scale = (w.detach().abs().amax(dim=1, keepdim=True) / max_val).clamp_min(1e-12)
    else:
        scale = (w.detach().abs().max() / max_val).clamp_min(1e-12)
    q = torch.clamp(torch.round(w / scale), -max_val - 1, max_val) * scale
    return w + (q - w).detach()


def apply_qat_to_model(model, mlp_max: int = 15, attn_max: int = 31, embed_max: int = 127):
    """Apply fake quantization to all applicable weights in-place (STE).
    Call this at the start of each forward pass when QAT is active."""
    for name, param in model.named_parameters():
        if param.numel() <= 896 or not param.is_floating_point() or param.ndim < 2:
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS) or "smear" in name:
            continue
        if any(p in name for p in EMBED_QUANT_PATTERNS):
            param.data = fake_quantize_ste(param.data, embed_max)
        elif any(p in name for p in MLP_QUANT_PATTERNS):
            param.data = fake_quantize_ste(param.data, mlp_max)
        else:
            param.data = fake_quantize_ste(param.data, attn_max)


# --- Mixed quantization (int5-MLP, int6-attn, int8-embed) ---

def sdclip_quantize(t: Tensor, clip_sigmas: float, clip_range: int) -> tuple[Tensor, Tensor]:
    """SDClip quantization: clip = k * std(row). Lower entropy than min/max clipping."""
    t32 = t.float()
    if t32.ndim == 2:
        row_std = t32.std(dim=1)
        scale = (clip_sigmas * row_std / clip_range).clamp_min(1e-12)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -clip_range, clip_range).to(torch.int8)
        return q, scale.half()
    std_val = t32.std().item()
    scale = torch.tensor(max(clip_sigmas * std_val / clip_range, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale

# SDClip parameters (from SOTA #1493)
MATRIX_CLIP_SIGMAS = float(os.environ.get("MATRIX_CLIP_SIGMAS", "12.85"))
EMBED_CLIP_SIGMAS = float(os.environ.get("EMBED_CLIP_SIGMAS", "15.0"))
MATRIX_BITS = int(os.environ.get("MATRIX_BITS", "6"))
EMBED_BITS = int(os.environ.get("EMBED_BITS", "8"))

MLP_QUANT_PATTERNS = ("mlp.", "gate.", "up.")
EMBED_QUANT_PATTERNS = ("tok_emb",)

def quantize_state_dict_mixed(state_dict: dict) -> dict:
    result = {}
    for name, t in state_dict.items():
        t_cpu = t.detach().cpu().contiguous()
        if not t_cpu.is_floating_point() or t_cpu.numel() <= 65536:
            result[name] = t_cpu.to(torch.float16) if t_cpu.is_floating_point() else t_cpu
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS) or "smear" in name:
            result[name] = t_cpu.to(torch.float16).contiguous()
            continue
        if any(p in name for p in EMBED_QUANT_PATTERNS):
            cs, bits = EMBED_CLIP_SIGMAS, EMBED_BITS
        else:
            cs, bits = MATRIX_CLIP_SIGMAS, MATRIX_BITS
        clip_range = 2 ** (bits - 1) - 1
        q, scale = sdclip_quantize(t_cpu, cs, clip_range)
        result[name + ".__q"] = q
        result[name + ".__scale"] = scale
        result[name + ".__dtype"] = str(t_cpu.dtype)
    result["__quant_format__"] = "sdclip_v1"
    return result

def dequantize_state_dict_mixed(quant_dict: dict) -> dict:
    quant_dict.pop("__quant_format__", None)
    result = {}
    seen = set()
    for key in list(quant_dict.keys()):
        if key.endswith(".__q"):
            name = key[:-4]
            if name in seen: continue
            seen.add(name)
            q = quant_dict[name + ".__q"]
            scale = quant_dict[name + ".__scale"]
            dtype = getattr(torch, quant_dict[name + ".__dtype"].split(".")[-1])
            if q.ndim == 2 and scale.ndim == 1:
                result[name] = (q.float() * scale.float()[:, None]).to(dtype)
            else:
                result[name] = (q.float() * scale.float()).to(dtype)
        elif not any(key.endswith(s) for s in (".__scale", ".__dtype")):
            result[key] = quant_dict[key]
    return result


# -----------------------------
# GPTQ (Generalized Post-Training Quantization)
# -----------------------------
# Column-wise quantization with Hessian compensation.
# Minimizes output error (not weight error) using calibration data.

def collect_layer_inputs(model, calib_data, module_name, device):
    """Collect input activations for a specific module using calibration data."""
    inputs_list = []
    hooks = []
    def hook_fn(module, input, output):
        inp = input[0] if isinstance(input, tuple) else input
        inputs_list.append(inp.detach().reshape(-1, inp.shape[-1]).float())
    for name, module in model.named_modules():
        if name == module_name:
            hooks.append(module.register_forward_hook(hook_fn))
            break
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for x in calib_data:
            _ = model(x, torch.roll(x, -1, dims=1))
    for h in hooks:
        h.remove()
    return torch.cat(inputs_list, dim=0) if inputs_list else None


def gptq_quantize_weight(w, H, clip_sigmas=12.85, clip_range=31, block_size=128):
    """GPTQ with SDClip: Hessian-compensated quantization using k*std per-row scaling.
    Returns (Q_int8, scale_fp16) — quantized int values and per-row scales."""
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone().cpu()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)
    # Column ordering by Hessian diagonal (most important first)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    # SDClip scale: k * std(row) / clip_range
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
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
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


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
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rope_d = cos.size(-1) * 2
    if rope_d == x.size(-1):
        half = rope_d // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    x_rot, x_pass = x[..., :rope_d], x[..., rope_d:]
    half = rope_d // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((rotated, x_pass), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        use_xsa: bool = False,
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
        rope_dims_env = int(os.environ.get("ROPE_DIMS", self.head_dim))
        self.rope_dims = max(2, min(rope_dims_env, self.head_dim))
        if self.rope_dims % 2 != 0:
            raise ValueError("ROPE_DIMS must be even")
        self.rotary = Rotary(self.rope_dims, base=rope_base)
        self.use_xsa = use_xsa

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
        # Expand KV heads for GQA (compatible with PyTorch < 2.5 without enable_gqa)
        if self.num_kv_heads != self.num_heads:
            reps = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        # XSA: remove self-value projection from attention output
        if self.use_xsa:
            # v is already expanded for GQA above
            dot_yv = (y * v).sum(dim=-1, keepdim=True)
            dot_vv = (v * v).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            y = y - (dot_yv / dot_vv) * v
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, activation: str = "leakyrelu2"):
        super().__init__()
        self.activation = activation
        if activation == "leakyrelu2":
            # LeakyReLU²: 2 matrices, wider hidden (mlp_mult * dim)
            hidden = mlp_mult * dim
            hidden = ((hidden + 63) // 64) * 64
            self.up = CastedLinear(dim, hidden, bias=False)
            self.proj = CastedLinear(hidden, dim, bias=False)
            self.proj._zero_init = True
        else:
            # SwiGLU: 3 matrices, narrower hidden (2/3 * mlp_mult * dim)
            hidden = int(2 * mlp_mult * dim / 3)
            hidden = ((hidden + 63) // 64) * 64
            self.gate = CastedLinear(dim, hidden, bias=False)
            self.up = CastedLinear(dim, hidden, bias=False)
            self.proj = CastedLinear(hidden, dim, bias=False)
            self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "leakyrelu2":
            return self.proj(F.leaky_relu(self.up(x), negative_slope=0.5).square())
        else:
            return self.proj(F.silu(self.gate(x)) * self.up(x))



class SmearGate(nn.Module):
    """Blend each token embedding with the previous token's embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class NgramHashEmbedding(nn.Module):
    """Hash n-gram context into a learned embedding, projected to model dim."""
    def __init__(self, n: int, num_buckets: int, embed_dim: int, model_dim: int):
        super().__init__()
        self.n = n
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)
        self.proj = CastedLinear(embed_dim, model_dim, bias=False)
        self.proj._zero_init = True  # start as no-op
        # Fixed primes for hashing each position in the n-gram
        self.register_buffer("primes", torch.tensor([2, 3, 5, 7, 11, 13, 17, 19, 23][:n], dtype=torch.long))

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: [B, T]
        h = torch.zeros_like(token_ids, dtype=torch.long)
        for i in range(self.n):
            # Shift token_ids right by i positions (causal: only past context)
            shifted = torch.roll(token_ids.long(), shifts=i, dims=1)
            shifted[:, :i] = 0  # zero out non-causal positions
            h = (h + shifted * self.primes[i]) % self.num_buckets
        return self.proj(self.embed(h))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        use_xsa: bool = False,
        activation: str = "leakyrelu2",
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult, activation=activation)
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
        ngram_n: int = 0,
        ngram_buckets: int = 8192,
        ngram_dim: int = 64,
        xsa_last_n: int = 4,
        layer_schedule: list = None,
        layer_schedule_phase2: list = None,
        activation: str = "leakyrelu2",
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.ngram = NgramHashEmbedding(ngram_n, ngram_buckets, ngram_dim, model_dim) if ngram_n > 0 else None
        self.smear = SmearGate(model_dim)

        # Depth recurrence: layer_schedule (phase1) + optional phase2 that activates mid-training.
        phase1 = layer_schedule if layer_schedule else list(range(num_layers))
        phase2 = layer_schedule_phase2 if layer_schedule_phase2 else phase1
        max_schedule = phase2 if len(phase2) >= len(phase1) else phase1
        self.layer_schedule_phase1 = phase1
        self.layer_schedule_phase2 = phase2
        self.layer_schedule = list(phase1)
        num_unique_blocks = max(max(max_schedule), num_layers - 1) + 1
        max_effective = len(max_schedule)

        # Dynamic (set by set_phase). Init for phase1.
        self.num_encoder_layers = len(phase1) // 2
        self.num_decoder_layers = len(phase1) - self.num_encoder_layers
        # skip_weights sized for max schedule so phase2 has enough slots.
        max_enc = max_effective // 2
        max_dec = max_effective - max_enc
        self.num_skip_weights = min(max_enc, max_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self._recurrence_phase = 1
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    use_xsa=(i >= num_unique_blocks - xsa_last_n) if xsa_last_n < num_unique_blocks else True,
                    activation=activation,
                )
                for i in range(num_unique_blocks)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        import math
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def set_recurrence_phase(self, phase: int) -> None:
        if phase == self._recurrence_phase:
            return
        self._recurrence_phase = phase
        self.layer_schedule = list(self.layer_schedule_phase2 if phase == 2 else self.layer_schedule_phase1)
        eff = len(self.layer_schedule)
        self.num_encoder_layers = eff // 2
        self.num_decoder_layers = eff - self.num_encoder_layers

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.ngram is not None:
            x = x + self.ngram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        # Uses layer_schedule to support depth recurrence (shared blocks).
        for i in range(self.num_encoder_layers):
            block_idx = self.layer_schedule[i]
            x = self.blocks[block_idx](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block_idx = self.layer_schedule[self.num_encoder_layers + i]
            x = self.blocks[block_idx](x, x0)

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


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
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
        ngram_n=args.ngram_n,
        ngram_buckets=args.ngram_buckets,
        ngram_dim=args.ngram_dim,
        xsa_last_n=args.xsa_last_n,
        layer_schedule=args.layer_schedule if args.layer_schedule else None,
        layer_schedule_phase2=args.layer_schedule_phase2 if args.layer_schedule_phase2 else None,
        activation=args.activation,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # Resume from checkpoint if specified
    resume_from = os.environ.get("RESUME_FROM", "")
    resume_step = 0
    if resume_from and os.path.exists(resume_from):
        log0(f"Resuming from checkpoint: {resume_from}")
        resume_state = torch.load(resume_from, map_location="cpu", weights_only=False)
        base_model.load_state_dict(resume_state, strict=False)
        resume_step = int(os.environ.get("RESUME_STEP", "0"))
        log0(f"Resumed at step {resume_step}")

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.embed_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
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

    # --- Artifact size pre-check ---
    # Estimate mixed quant artifact size BEFORE training to avoid wasted compute.
    # Set SKIP_SIZE_CHECK=1 to override (e.g., for experiments where size doesn't matter).
    skip_size_check = int(os.environ.get("SKIP_SIZE_CHECK", "0"))
    max_artifact_bytes = int(os.environ.get("MAX_ARTIFACT_BYTES", "16000000"))  # 16MB = 16,000,000 bytes (competition limit)
    if rank == 0 and not skip_size_check:
        try:
            import brotli as _brotli_check
            _check_sd = {k: v.detach().cpu().float() for k, v in base_model.state_dict().items()}
            _check_quant = quantize_state_dict_mixed(_check_sd)
            _check_buf = io.BytesIO()
            torch.save(_check_quant, _check_buf)
            _check_raw = _check_buf.getvalue()
            _check_compressed = _brotli_check.compress(_byte_shuffle(_check_raw), quality=11)
            _check_code_size = len(code.encode("utf-8"))
            _check_total = len(_check_compressed) + _check_code_size
            log0(f"size_check: mixed_quant={len(_check_compressed)} code={_check_code_size} total={_check_total} limit={max_artifact_bytes}")
            if _check_total > max_artifact_bytes:
                log0(f"SIZE_CHECK_FAILED: {_check_total} bytes > {max_artifact_bytes} limit ({_check_total - max_artifact_bytes} bytes over)")
                log0(f"  Reduce MODEL_DIM or NUM_LAYERS. Set SKIP_SIZE_CHECK=1 to override.")
                sys.exit(1)
            else:
                headroom = max_artifact_bytes - _check_total
                log0(f"size_check: PASSED ({headroom} bytes headroom, {headroom/1024:.1f} KB)")
            del _check_sd, _check_quant, _check_buf, _check_raw, _check_compressed
        except Exception as e:
            log0(f"size_check: WARNING could not estimate size: {e}")

    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
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

    # SKIP_TRAINING: jump straight to serialization + GPTQ + TTT
    skip_training = int(os.environ.get("SKIP_TRAINING", "0"))
    if skip_training:
        log0("SKIP_TRAINING: skipping warmup and training loop")

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if not skip_training and args.warmup_steps > 0:
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

    # EMA initialization
    if args.ema_enabled:
        base_model._ema_state = {n: p.detach().clone().float() for n, p in base_model.state_dict().items()}
        log0(f"EMA: enabled, decay={args.ema_decay}")

    if args.qat_enabled:
        log0(f"QAT: enabled, start_step={args.qat_start_step}, bits_mlp={args.qat_bits_mlp}, bits_attn={args.qat_bits_attn}, bits_embed={args.qat_bits_embed}")
    _qat_logged = False

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if skip_training:
        stop_after_step = -1
        log0("SKIP_TRAINING: skipping main training loop")

    phase_swap_step = int(args.recurrence_activate_at * args.iterations) if args.recurrence_activate_at > 0 and args.layer_schedule_phase2 else -1
    if phase_swap_step > 0:
        log0(f"recurrence: phase2 swap scheduled at step {phase_swap_step} (schedule len {len(args.layer_schedule_phase2)})")
    _phase2_logged = False

    step = resume_step
    while True:
        if phase_swap_step > 0 and step >= phase_swap_step and not _phase2_logged:
            base_model.set_recurrence_phase(2)
            if args.ema_enabled and hasattr(base_model, '_ema_state'):
                base_model._ema_state = {n: p.detach().clone().float() for n, p in base_model.state_dict().items()}
                log0(f"EMA: reset at phase2 swap (step {step})")
            log0(f"recurrence: phase2 activated at step {step}")
            _phase2_logged = True
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
        # QAT: build param map once, save/restore weights around forward+backward
        _qat_active = args.qat_enabled and step >= args.qat_start_step
        if _qat_active and not _qat_logged:
            log0(f"QAT: activated at step {step}")
            _qat_logged = True
        _qat_saved = {}
        _qat_mlp_max = (2 ** (args.qat_bits_mlp - 1)) - 1
        _qat_attn_max = (2 ** (args.qat_bits_attn - 1)) - 1
        _qat_embed_max = (2 ** (args.qat_bits_embed - 1)) - 1
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            # QAT: save original weights, replace with fake-quantized for forward
            if _qat_active:
                _qat_saved.clear()
                for _qn, _qp in base_model.named_parameters():
                    if _qp.numel() <= 896 or not _qp.is_floating_point() or _qp.ndim < 2:
                        continue
                    if any(_pat in _qn for _pat in CONTROL_TENSOR_NAME_PATTERNS) or "smear" in _qn:
                        continue
                    _qat_saved[_qn] = _qp.data.clone()
                    if any(_pat in _qn for _pat in EMBED_QUANT_PATTERNS):
                        _qp.data = fake_quantize_ste(_qp.data, _qat_embed_max)
                    elif any(_pat in _qn for _pat in MLP_QUANT_PATTERNS):
                        _qp.data = fake_quantize_ste(_qp.data, _qat_mlp_max)
                    else:
                        _qp.data = fake_quantize_ste(_qp.data, _qat_attn_max)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
            # QAT: restore original weights before optimizer step
            if _qat_active:
                for _qn, _qp in base_model.named_parameters():
                    if _qn in _qat_saved:
                        _qp.data = _qat_saved[_qn]
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
        if args.ema_enabled and hasattr(base_model, '_ema_state'):
            with torch.no_grad():
                for n, p in base_model.state_dict().items():
                    base_model._ema_state[n].lerp_(p.float(), 1 - args.ema_decay)

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


        # SWA: collect weights every 200 steps during warmdown
        _swa_every = 200
        if not hasattr(base_model, '_swa_state'):
            base_model._swa_state = None
            base_model._swa_count = 0
        _est_total = int(max_wallclock_ms / (approx_training_time_ms / max(step, 1)))
        _warmdown_start = max(0, _est_total - args.warmdown_iters)
        if step >= _warmdown_start and step % _swa_every == 0 and step > 100:
            if base_model._swa_state is None:
                base_model._swa_state = {n: p.detach().cpu().clone().float() for n, p in base_model.state_dict().items()}
                base_model._swa_count = 1
                log0(f"SWA: started at step {step} (warmdown_start~{_warmdown_start})")
            else:
                for n, p in base_model.state_dict().items():
                    base_model._swa_state[n] += p.detach().cpu().float()
                base_model._swa_count += 1

        # Needed to sync whether we've reached the wallclock cap.
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

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.


    # SKIP_TRAINING: load existing checkpoint and jump to GPTQ+TTT
    if skip_training and os.path.exists("final_model.pt"):
        log0("SKIP_TRAINING: loading checkpoint from final_model.pt")
        base_model.load_state_dict(torch.load("final_model.pt", map_location="cpu", weights_only=True))
        base_model.to(device)
        log0("SKIP_TRAINING: checkpoint loaded, skipping to GPTQ+TTT")

    # Apply EMA weights if enabled (takes priority over SWA)
    elif args.ema_enabled and hasattr(base_model, '_ema_state'):
        log0("EMA: applying averaged weights")
        for n, t in base_model._ema_state.items():
            base_model.state_dict()[n].copy_(t.to(dtype=base_model.state_dict()[n].dtype))
        log0("EMA: applied")
    # Apply SWA averaged weights (if no EMA, or as fallback)
    elif hasattr(base_model, '_swa_state') and base_model._swa_state is not None and base_model._swa_count > 1:
        log0(f"SWA: averaging {base_model._swa_count} checkpoints")
        for n, t in base_model._swa_state.items():
            avg = (t / base_model._swa_count).to(dtype=base_model.state_dict()[n].dtype)
            base_model.state_dict()[n].copy_(avg)
        log0("SWA: applied")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu", weights_only=False)
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
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
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # GPTQ + Mixed quantization (int5-MLP + int6-attn + int8-embed + brotli-11)
    code_bytes = len(code.encode("utf-8"))
    try:
        import brotli

        # Build calibration data and collect Hessians for GPTQ+SDClip
        gptq_enabled = int(os.environ.get("GPTQ_ENABLED", "1"))
        if gptq_enabled:
            log0("GPTQ+SDClip: building calibration data...")
            n_calib = int(os.environ.get("GPTQ_N_CALIB", "64"))
            calib_shards = sorted(glob.glob(args.train_files))[:2]
            calib_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in calib_shards]).long()
            calib_data = []
            rng = random.Random(args.seed)
            for _ in range(n_calib):
                start = rng.randint(0, calib_tokens.numel() - args.train_seq_len - 1)
                calib_data.append(calib_tokens[start:start + args.train_seq_len].unsqueeze(0))
            del calib_tokens

            # Collect Hessians for all eligible layers
            log0("GPTQ+SDClip: collecting Hessians...")
            hessians = {}
            n_hcalib = min(32, len(calib_data))
            for name, param in base_model.named_parameters():
                if param.ndim != 2 or param.numel() < 4096: continue
                if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS) or "smear" in name: continue
                module_name = ".".join(name.split(".")[:-1])
                X = collect_layer_inputs(base_model, calib_data[:n_hcalib], module_name, device)
                if X is not None and X.shape[0] > 0:
                    hessians[name] = (X.float().T @ X.float()) / X.shape[0]
            del calib_data

            # GPTQ+SDClip quantize all eligible weights
            sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
            quant_result = {}
            quant_meta = {}
            for name, tensor in sd_cpu.items():
                t = tensor.contiguous()
                if not t.is_floating_point() or t.numel() <= 65536:
                    quant_result[name] = t.to(torch.float16) if t.is_floating_point() else t
                    quant_meta[name] = "passthrough"
                    continue
                if name not in hessians or hessians[name].shape[0] != t.shape[1]:
                    # Simple SDClip (no GPTQ Hessian) for layers with dim mismatch
                    if any(p in name for p in EMBED_QUANT_PATTERNS):
                        cs, bits = EMBED_CLIP_SIGMAS, EMBED_BITS
                    else:
                        cs, bits = MATRIX_CLIP_SIGMAS, MATRIX_BITS
                    clip_range = 2 ** (bits - 1) - 1
                    log0(f"  SDClip (no GPTQ) {name} (int{bits}, k={cs})...")
                    W = t.float()
                    row_std = W.std(dim=1)
                    s = (cs * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
                    Q = torch.clamp(torch.round(W / s.float().unsqueeze(1)), -clip_range, clip_range).to(torch.int8)
                    quant_result[name + ".q"] = Q
                    quant_result[name + ".scale"] = s
                    quant_meta[name] = f"sdclip (int{bits})"
                    continue
                if any(p in name for p in EMBED_QUANT_PATTERNS):
                    cs, bits = EMBED_CLIP_SIGMAS, EMBED_BITS
                else:
                    cs, bits = MATRIX_CLIP_SIGMAS, MATRIX_BITS
                clip_range = 2 ** (bits - 1) - 1
                log0(f"  GPTQ+SDClip {name} (int{bits}, k={cs})...")
                q, s = gptq_quantize_weight(t, hessians[name], clip_sigmas=cs, clip_range=clip_range)
                quant_result[name + ".q"] = q
                quant_result[name + ".scale"] = s
                quant_meta[name] = f"gptq_sdclip (int{bits})"
            del hessians

            # Also load dequantized GPTQ weights into base_model for TTT eval
            deq_state = {}
            for name, orig in sd_cpu.items():
                if name + ".q" in quant_result:
                    q, s = quant_result[name + ".q"], quant_result[name + ".scale"]
                    deq_state[name] = (q.float() * s.float().view(q.shape[0], *[1]*(q.ndim-1))).to(orig.dtype)
                elif name in quant_result:
                    t = quant_result[name]
                    deq_state[name] = t.to(orig.dtype) if t.dtype != orig.dtype else t
                else:
                    deq_state[name] = orig
            base_model.load_state_dict({k: v.to(device) for k, v in deq_state.items()}, strict=True)
            del sd_cpu
            log0("GPTQ+SDClip: done")

        # Save float+GPTQ state for TTT (TTT will modify weights)
        gptq_float_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

        # --- SGD All-Weights TTT eval on float+GPTQ model ---
        # IMPORTANT: TTT runs on the float model (after GPTQ but before mixed quantization).
        # Running TTT on dequantized mixed quant weights destroys quality because
        # int5 (31 levels) weights are too coarse for SGD adaptation.

        ttt_enabled = int(os.environ.get("TTT_ENABLED", "1"))
        ttt_max_docs = int(os.environ.get("TTT_MAX_DOCS", "0"))  # 0 = all docs
        ttt_min_doc_len = int(os.environ.get("TTT_MIN_DOC_LEN", "32"))
        if ttt_enabled:
            sgd_lr = float(os.environ.get("SGD_LR", "0.005"))
            sgd_momentum = float(os.environ.get("SGD_MOMENTUM", "0.9"))
            ttt_train_chunk = int(os.environ.get("TTT_TRAIN_CHUNK", "2048"))
            score_cap = int(os.environ.get("SCORE_CAP", "2048"))
            log0(f"Running SGD all-weights TTT eval (lr={sgd_lr}, momentum={sgd_momentum}, "
                 f"chunk={ttt_train_chunk}, score_cap={score_cap})...")

            # Create fresh UNCOMPILED model for TTT — torch.compile breaks SGD per-doc adaptation
            ttt_model = GPT(
                vocab_size=args.vocab_size, num_layers=args.num_layers,
                model_dim=args.model_dim, num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
                tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
                logit_softcap=args.logit_softcap, rope_base=args.rope_base,
                qk_gain_init=args.qk_gain_init, ngram_n=args.ngram_n,
                ngram_buckets=args.ngram_buckets, ngram_dim=args.ngram_dim,
                xsa_last_n=args.xsa_last_n,
                layer_schedule=args.layer_schedule if args.layer_schedule else None,
                layer_schedule_phase2=args.layer_schedule_phase2 if args.layer_schedule_phase2 else None,
                activation=args.activation,
            ).to(device).float()
            ttt_model.load_state_dict({k: v.to(device).float() for k, v in gptq_float_state.items()}, strict=True)
            base_state = {k: v.clone() for k, v in ttt_model.state_dict().items()}

            # Find document boundaries (BOS=1)
            bos_positions = (val_tokens == 1).nonzero(as_tuple=True)[0].cpu().numpy()
            n_all_docs = len(bos_positions)
            if ttt_max_docs > 0:
                n_all_docs = min(n_all_docs, ttt_max_docs)

            doc_list = []
            for d in range(n_all_docs):
                ds = int(bos_positions[d])
                de = int(bos_positions[d + 1]) if d + 1 < len(bos_positions) else val_tokens.numel()
                dl = de - ds
                if dl >= 5:
                    doc_list.append((ds, dl))
            n_ttt_docs = len(doc_list)

            short_docs = [(ds, dl) for ds, dl in doc_list if dl < ttt_min_doc_len]
            long_docs = [(ds, dl) for ds, dl in doc_list if dl >= ttt_min_doc_len]

            # Shard docs across GPUs
            my_short = short_docs[(len(short_docs) * rank) // world_size : (len(short_docs) * (rank + 1)) // world_size]
            my_long = long_docs[(len(long_docs) * rank) // world_size : (len(long_docs) * (rank + 1)) // world_size]
            log0(f"TTT: {n_ttt_docs} docs ({len(short_docs)} short, {len(long_docs)} long), "
                 f"rank {rank}: {len(my_short)} short + {len(my_long)} long")

            ttt_nll = torch.zeros((), device=device, dtype=torch.float64)
            ttt_bytes = torch.zeros((), device=device, dtype=torch.float64)
            ttt_tokens = torch.zeros((), device=device, dtype=torch.float64)
            t_ttt = time.perf_counter()

            # Short docs: score without TTT
            ttt_model.eval()
            for p in ttt_model.parameters():
                p.requires_grad = False
            with torch.no_grad():
                for ds, dl in my_short:
                    x = val_tokens[ds:ds + dl - 1].to(device=device, dtype=torch.int64).unsqueeze(0)
                    y = val_tokens[ds + 1:ds + dl].to(device=device, dtype=torch.int64).unsqueeze(0)
                    n = dl - 1
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = ttt_model(x, y)
                    ttt_nll += loss.to(torch.float64) * n
                    ttt_tokens += n
                    prev_ids = x.reshape(-1)[:n]
                    tgt_ids = y.reshape(-1)[:n]
                    tb = base_bytes_lut[tgt_ids].to(torch.float64)
                    tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
                    ttt_bytes += tb.sum()

            # Long docs: SGD all-weights TTT (train on chunks, then score)
            for di, (ds, dl) in enumerate(my_long):
                doc = val_tokens[ds:ds + dl].to(device)
                ttt_model.load_state_dict(base_state, strict=True)
                for p in ttt_model.parameters():
                    p.requires_grad = True
                ttt_model.train()
                optimizer = torch.optim.SGD(ttt_model.parameters(), lr=sgd_lr, momentum=sgd_momentum)

                # Train on all chunks
                for cs in range(0, dl - 1, ttt_train_chunk):
                    ce = min(cs + ttt_train_chunk, dl - 1)
                    if ce - cs < 2:
                        continue
                    tx = doc[cs:ce].long().unsqueeze(0)
                    ty = doc[cs + 1:ce + 1].long().unsqueeze(0)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = ttt_model(tx, ty)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Score (up to score_cap tokens)
                ttt_model.eval()
                for p in ttt_model.parameters():
                    p.requires_grad = False
                score_len = min(score_cap, dl - 1)
                sx = doc[:score_len].long().unsqueeze(0)
                sy = doc[1:score_len + 1].long().unsqueeze(0)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    sloss = ttt_model(sx, sy).detach()
                ttt_nll += sloss.to(torch.float64) * score_len
                ttt_tokens += score_len
                prev_ids = sx.reshape(-1)[:score_len]
                tgt_ids = sy.reshape(-1)[:score_len]
                tb = base_bytes_lut[tgt_ids].to(torch.float64)
                tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
                ttt_bytes += tb.sum()

                if (di + 1) % 1000 == 0:
                    elapsed = time.perf_counter() - t_ttt
                    running_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
                    log0(f"  TTT rank {rank}: {di+1}/{len(my_long)} long docs, bpb={running_bpb:.4f} elapsed={elapsed:.0f}s")

            # All-reduce across GPUs
            if distributed:
                dist.all_reduce(ttt_nll, op=dist.ReduceOp.SUM)
                dist.all_reduce(ttt_bytes, op=dist.ReduceOp.SUM)
                dist.all_reduce(ttt_tokens, op=dist.ReduceOp.SUM)
            ttt_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
            ttt_loss = ttt_nll.item() / max(ttt_tokens.item(), 1.0)
            log0(f"final_sgd_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
                 f"docs:{n_ttt_docs} eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")

        # --- Save quantized artifact (from GPTQ+SDClip result) ---
        if ttt_enabled:
            del ttt_model, base_state
            torch.cuda.empty_cache()
        # Use pre-computed quant_result from GPTQ+SDClip (not re-quantizing)
        mixed_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, mixed_buf)
        mixed_raw = mixed_buf.getvalue()
        mixed_blob = brotli.compress(_byte_shuffle(mixed_raw), quality=11)
        with open("final_model.mixed.ptz", "wb") as f:
            f.write(mixed_blob)
        log0(f"Serialized model GPTQ+SDClip+brotli: {len(mixed_blob)} bytes")
        _fits = "FITS" if (len(mixed_blob) + code_bytes) < 16_000_000 else "OVER"
        log0(f"Total mixed submission: {len(mixed_blob) + code_bytes} bytes ({_fits} 16MB)")
        # Roundtrip validation: decompress, dequantize, eval
        loaded = torch.load(
            io.BytesIO(_byte_unshuffle(brotli.decompress(mixed_blob))),
            map_location="cpu", weights_only=False
        )
        deq = {}
        template_sd = {k: v.cpu() for k, v in base_model.state_dict().items()}
        for name, orig in template_sd.items():
            if name + ".q" in loaded["w"]:
                q, s = loaded["w"][name + ".q"], loaded["w"][name + ".scale"]
                deq[name] = (q.float() * s.float().view(q.shape[0], *[1]*(q.ndim-1))).to(orig.dtype)
            elif name in loaded["w"]:
                t = loaded["w"][name]
                deq[name] = t.to(orig.dtype) if t.dtype != orig.dtype else t
            else:
                deq[name] = orig
        base_model.load_state_dict({k: v.to(device) for k, v in deq.items()}, strict=True)
        torch.cuda.synchronize()
        t_mqeval = time.perf_counter()
        mq_val_loss, mq_val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        log0(f"final_mixed_roundtrip val_loss:{mq_val_loss:.4f} val_bpb:{mq_val_bpb:.4f} "
             f"eval_time:{1000.0 * (time.perf_counter() - t_mqeval):.0f}ms")

    except ImportError:
        log0("WARNING: brotli not installed, skipping mixed quantization")
    except Exception as e:
        log0(f"WARNING: mixed quantization failed: {e}")
        import traceback; traceback.print_exc()


    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
