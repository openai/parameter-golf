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
from pathlib import Path

import brotli

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
# Hybrid ETD (Encode-Think-Decode) architecture:
# - Unique encoder layers map input to latent space (with U-Net skip storage)
# - Shared "think" layers loop multiple times (iterative refinement)
# - Unique decoder layers map back to output space (with U-Net skip consumption)
# This combines the best of both worlds: unique capacity where layers do
# fundamentally different jobs, and parameter-efficient looping where
# iterative refinement is needed.

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
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape - ETD hybrid architecture
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_encoder_layers = int(os.environ.get("NUM_ENCODER_LAYERS", 3))  # Unique front layers
    num_think_layers = int(os.environ.get("NUM_THINK_LAYERS", 3))      # Shared looped layers
    num_think_passes = int(os.environ.get("NUM_THINK_PASSES", 2))      # Times to loop think layers
    num_decoder_layers = int(os.environ.get("NUM_DECODER_LAYERS", 3))  # Unique back layers
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Encoder/Decoder block config
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))

    # Think block config (defaults to same as encoder/decoder if not set)
    # THINK_NUM_HEADS=0 means use NUM_HEADS, THINK_KV_HEADS=0 means use NUM_KV_HEADS
    # THINK_MLP_MULT=0 means use MLP_MULT, otherwise must be >= 1
    think_num_heads = int(os.environ.get("THINK_NUM_HEADS", 0)) or num_heads
    think_kv_heads = int(os.environ.get("THINK_KV_HEADS", 0)) or num_kv_heads
    think_mlp_mult = int(os.environ.get("THINK_MLP_MULT", 0))
    if think_mlp_mult == 0:
        think_mlp_mult = mlp_mult
    if think_mlp_mult < 1:
        raise ValueError(f"THINK_MLP_MULT must be >= 0 , got {think_mlp_mult}")

    # Optimizer hyperparameters
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

    # Mixed precision quantization
    quant_bits = int(os.environ.get("QUANT_BITS", 5))
    think_quant_bits = int(os.environ.get("THINK_QUANT_BITS", 5))
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 64))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    # Progressive recurrence curriculum (0.0 = disabled, start with full passes)
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.0))

    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # QuIP-style randomised Hadamard rotation wrapping GPTQ.
    use_rht = bool(int(os.environ.get("USE_RHT", "0")))
    # Peaky-weight recipe — Muon side.
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.09))
    muon_row_norm = bool(int(os.environ.get("MUON_ROW_NORM", "1")))
    # Peaky-weight recipe — Adam side. WD on the Adam-managed groups (embeddings,
    # head, scalars). Default 0 keeps existing behaviour; raise to peakify those
    # tensors too. Top PR uses adam_wd=0.02, embed_wd=0.085.
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.0))
    embed_weight_decay = float(os.environ.get("EMBED_WEIGHT_DECAY", 0.0))
    head_weight_decay = float(os.environ.get("HEAD_WEIGHT_DECAY", 0.0))
    # SDClip: per-row clip = k * std(row). 0 disables (falls back to percentile clip).
    # Matrices: peaky → start k≈4 at int5, k≈12.85 at int6.
    # Embeddings: keep MUCH wider (top PR uses 20.0) — embedding precision matters more.
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 0.0))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 20.0))

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
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0, row_norm=False):
        super().__init__(params, dict(
            lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov,
            weight_decay=weight_decay, row_norm=row_norm,
        ))

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
            weight_decay = group.get("weight_decay", 0.0)
            row_norm = group.get("row_norm", False)

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
                    if row_norm and g.ndim == 2:  # MuonEq-R: equalises per-row update scale
                        g = g / (g.norm(dim=-1, keepdim=True) + 1e-7)
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
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
# POST-TRAINING QUANTIZATION
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
INT8_CLIP_PERCENTILE = float(os.environ.get("INT8_CLIP_PERCENTILE", 99.99984))
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

def quantize_float_tensor(t, bits=8, use_rht=False, rht_seed_base=0):
    """Returns (q, scale, rht_meta)."""
    max_val = (1 << (bits - 1)) - 1
    t32 = t.float()
    rht_meta = None
    if t32.ndim == 2:
        rows, cols = t32.shape
        if use_rht and _is_pow2(rows) and _is_pow2(cols) and t32.numel() > 0:
            seed_row = (rht_seed_base * 2654435761) & 0xFFFFFFFF
            seed_col = (rht_seed_base * 2246822519 + 1) & 0xFFFFFFFF
            t32 = rht_apply(t32, seed_row, seed_col)
            rht_meta = (int(seed_row), int(seed_col))
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(max_val)).clamp_min(1.0 / float(max_val))
        q = torch.clamp(torch.round(clipped / scale[:, None]), -max_val, max_val).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(), rht_meta

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(max_val) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -max_val, max_val).to(torch.int8).contiguous()
    return q, scale, None

def quantize_state_dict_int8(state_dict, tensor_bits=None, hessians=None, use_rht=False,
                              clip_sigmas=0.0, embed_clip_sigmas=0.0):
    # Mixed precision: tensor_bits maps name substrings to bit widths, e.g. {"think_blocks": 8, "default": 6}.
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    rht_seeds: dict[str, tuple[int, int]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    def _get_bits(name: str) -> int:
        if tensor_bits is None:
            return 8
        for pattern, bits in tensor_bits.items():
            if pattern != "default" and pattern in name:
                return bits
        return tensor_bits.get("default", 8)

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
        bits = _get_bits(name)
        seed_base = abs(hash(name)) & 0xFFFFFFFF
        if hessians is not None and name in hessians:
            # Route the per-tensor SDClip k: embeddings need a much wider clip than matrices.
            cs = embed_clip_sigmas if "tok_emb" in name else clip_sigmas
            q, s, rht_meta = gptq_quantize_weight(
                t, hessians[name], bits=bits, use_rht=use_rht, rht_seed_base=seed_base,
                clip_sigmas=cs,
            )
            qmeta[name] = {"scheme": "per_row", "axis": 0, "bits": bits, "method": "gptq"}
        else:
            q, s, rht_meta = quantize_float_tensor(
                t, bits=bits, use_rht=use_rht, rht_seed_base=seed_base
            )
            if s.ndim > 0:
                qmeta[name] = {"scheme": "per_row", "axis": 0, "bits": bits}
            else:
                qmeta[name] = {"bits": bits}
        if rht_meta is not None:
            rht_seeds[name] = rht_meta
            qmeta[name]["rht"] = True
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
    if rht_seeds:
        obj["rht_seeds"] = rht_seeds
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    rht_seeds = obj.get("rht_seeds", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            w = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1))))
        else:
            scale = float(s.item())
            w = q.float() * scale
        if name in rht_seeds and w.ndim == 2:
            seed_row, seed_col = rht_seeds[name]
            w = rht_invert(w, int(seed_row), int(seed_col))
        out[name] = w.to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# RANDOMIZED HADAMARD TRANSFORM (QuIP-style incoherence; requires pow2 dims)
# -----------------------------

def _fwht(x: Tensor) -> Tensor:
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-3], n)
        h *= 2
    return x

def _rht_signs(seed: int, length: int, device, dtype=torch.float32) -> Tensor:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    bits = torch.randint(0, 2, (length,), generator=gen, dtype=torch.int64)
    return (bits.to(dtype=dtype) * 2.0 - 1.0).to(device=device)

def rht_apply(W: Tensor, seed_row: int, seed_col: int) -> Tensor:
    """W' = D_r · (1/√r) FWHT · W · (1/√c) FWHT · D_c. Orthonormal; rht_invert reverses."""
    rows, cols = W.shape
    dr = _rht_signs(seed_row, rows, W.device, dtype=W.dtype)
    dc = _rht_signs(seed_col, cols, W.device, dtype=W.dtype)
    W = W * dc[None, :]
    W = _fwht(W) / math.sqrt(cols)
    W = W * dr[:, None]
    W = _fwht(W.T).T / math.sqrt(rows)
    return W

def rht_invert(Wq: Tensor, seed_row: int, seed_col: int) -> Tensor:
    rows, cols = Wq.shape
    dr = _rht_signs(seed_row, rows, Wq.device, dtype=Wq.dtype)
    dc = _rht_signs(seed_col, cols, Wq.device, dtype=Wq.dtype)
    Wq = _fwht(Wq.T).T / math.sqrt(rows)
    Wq = Wq * dr[:, None]
    Wq = _fwht(Wq) / math.sqrt(cols)
    return (Wq * dc[None, :]).contiguous()


# -----------------------------
# GPTQ QUANTIZATION
# -----------------------------

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def gptq_quantize_weight(w, H, bits=6, block_size=128, use_rht=False, rht_seed_base=0, clip_sigmas=0.0):
    """Returns (q, s, rht_meta). rht_meta = (seed_row, seed_col) or None.
    clip_sigmas > 0 → SDClip (per-row k·std). 0 → percentile clip (default)."""
    clip_range = (1 << (bits - 1)) - 1
    w_float = w.float().clone()
    rows, cols = w_float.shape
    rht_meta = None

    # RHT: rotate W into an incoherent basis so outliers flatten. Hessian is
    # conjugated (H_eff = D_c · FWHT · H · FWHT · D_c / c) to stay consistent.
    # Only powers-of-two shapes qualify; non-pow2 falls through to vanilla GPTQ.
    if use_rht and _is_pow2(rows) and _is_pow2(cols):
        seed_row = (rht_seed_base * 2654435761) & 0xFFFFFFFF
        seed_col = (rht_seed_base * 2246822519 + 1) & 0xFFFFFFFF
        w_float = rht_apply(w_float, seed_row, seed_col)
        d_col = _rht_signs(seed_col, cols, H.device, dtype=torch.float32)
        H_rot = H.float() * d_col[None, :] * d_col[:, None]
        H_rot = _fwht(H_rot) / math.sqrt(cols)
        H = _fwht(H_rot.T).T / math.sqrt(cols)
        rht_meta = (int(seed_row), int(seed_col))

    W_orig = w_float
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H.diagonal().add_(0.01 * H.diag().mean())
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    if clip_sigmas > 0.0:
        # SDClip: scale = k·std(row)/clip_range. Wider clip lets long Laplacian-tail
        # weights through unclipped; only beats percentile clip when weights are peaky.
        row_std = W_orig.std(dim=1).clamp_min(1e-10)
        s = (clip_sigmas * row_std / clip_range).to(torch.float16)
    else:
        clip_abs = torch.quantile(W_orig.abs(), INT8_CLIP_Q, dim=1).clamp_min(1e-10)
        s = (clip_abs / clip_range).to(torch.float16)
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
    return Q[:, invperm], s, rht_meta


def collect_hessians(model: nn.Module, train_loader: "DistributedTokenLoader", args: Hyperparameters, grad_accum_steps: int) -> dict[str, Tensor]:
    hessians: dict[str, Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
            hessians[name].addmm_(x.T, x)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            hooks.append(module.register_forward_hook(make_hook(name + ".weight")))

    # Collect Hessian for tok_emb.weight used as lm_head by hooking final_norm output
    if model.tie_embeddings:
        def make_output_hook(name: str):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
                hessians[name].addmm_(x.T, x)
            return hook_fn
        hooks.append(model.final_norm.register_forward_hook(make_output_hook("tok_emb.weight")))

    model.eval()
    with torch.no_grad():
        for _ in range(args.gptq_calibration_batches):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            model(x, y)

    for hook in hooks:
        hook.remove()

    n = args.gptq_calibration_batches
    return {name: H.cpu() / n for name, H in hessians.items()}


# -----------------------------
# COMPRESSION
# -----------------------------
# Byte-shuffle deinterleaves every `stride`-th byte into contiguous groups
# before compression. For 16-bit data (fp16 scales, dtype metadata) this groups
# high and low bytes together, exposing lower entropy to the entropy coder and
# yielding better ratios than raw brotli on mixed-precision state dicts.

_BSHF_MAGIC = b"BSHF"

def byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(len(src), dtype=np.uint8)
    off = 0
    for p in range(stride):
        chunk = src[p::stride]
        out[off:off + len(chunk)] = chunk
        off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()

def byte_unshuffle(data: bytes) -> bytes:
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    off = 0
    for p in range(stride):
        k = n // stride + (1 if p < n % stride else 0)
        out[p::stride][:k] = payload[off:off + k]
        off += k
    return out.tobytes()


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


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
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
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
        return self.proj(x)


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
        self.has_mlp = mlp_mult > 0
        self.attn_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        if self.has_mlp:
            self.mlp_norm = RMSNorm()
            self.mlp = MLP(dim, mlp_mult)
            self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        if self.has_mlp:
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_encoder_layers: int,
        num_think_layers: int,
        num_think_passes: int,
        num_decoder_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        think_num_heads: int,
        think_kv_heads: int,
        think_mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_encoder_layers = num_encoder_layers
        self.num_think_layers = num_think_layers
        self.num_think_passes = num_think_passes
        self.active_think_passes = num_think_passes  # reduced by activate_passes() for progressive recurrence
        self.num_decoder_layers = num_decoder_layers

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        block_args = dict(dim=model_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult, rope_base=rope_base, qk_gain_init=qk_gain_init)
        think_block_args = dict(dim=model_dim, num_heads=think_num_heads, num_kv_heads=think_kv_heads,
            mlp_mult=think_mlp_mult, rope_base=rope_base, qk_gain_init=qk_gain_init)
        # ENCODE: unique front layers with U-Net skips, THINK: shared looped, DECODE: unique back layers.
        self.encoder_blocks = nn.ModuleList([Block(**block_args) for _ in range(num_encoder_layers)])
        self.think_blocks = nn.ModuleList([Block(**think_block_args) for _ in range(num_think_layers)])
        self.pass_emb = nn.Embedding(num_think_passes, model_dim)
        self.decoder_blocks = nn.ModuleList([Block(**block_args) for _ in range(num_decoder_layers)])

        # U-Net skip weights: connect encoder outputs to decoder inputs
        # Number of skip connections = min(encoder, decoder) layers
        self.num_skip_connections = min(num_encoder_layers, num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_connections, model_dim, dtype=torch.float32)
        )

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

    def activate_passes(self, passes: int) -> None:
        """Switch number of active think passes (progressive recurrence curriculum)."""
        self.active_think_passes = min(passes, self.num_think_passes)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # ---- ENCODE: unique layers, store skips ----
        skips: list[Tensor] = []
        for block in self.encoder_blocks:
            x = block(x, x0)
            skips.append(x)

        # ---- THINK: shared layers looped ----
        for pass_idx in range(self.active_think_passes):
            x = x + self.pass_emb(torch.tensor(pass_idx, device=x.device))[None, None, :]
            for block in self.think_blocks:
                x = block(x, x0)

        # ---- DECODE: unique layers, consume skips in reverse ----
        for i, block in enumerate(self.decoder_blocks):
            if i < self.num_skip_connections and skips:
                skip = skips.pop()
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
            x = block(x, x0)

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
        num_encoder_layers=args.num_encoder_layers,
        num_think_layers=args.num_think_layers,
        num_think_passes=args.num_think_passes,
        num_decoder_layers=args.num_decoder_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        think_num_heads=args.think_num_heads,
        think_kv_heads=args.think_kv_heads,
        think_mlp_mult=args.think_mlp_mult,
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

    # Collect all block params across encoder, think, decoder
    all_block_named_params = (
        list(base_model.encoder_blocks.named_parameters())
        + list(base_model.think_blocks.named_parameters())
        + list(base_model.decoder_blocks.named_parameters())
    )
    matrix_params = [
        p
        for name, p in all_block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in all_block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    # Add skip weights and pass embedding to scalar optimizer
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.extend(list(base_model.pass_emb.parameters()))

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.embed_weight_decay,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
        row_norm=args.muon_row_norm,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_weight_decay,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.head_weight_decay,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    total_effective_layers = args.num_encoder_layers + (args.num_think_layers * args.num_think_passes) + args.num_decoder_layers
    unique_layer_blocks = args.num_encoder_layers + args.num_think_layers + args.num_decoder_layers
    log0(f"model_params:{n_params}")
    log0(
        f"architecture:ETD encoder:{args.num_encoder_layers} "
        f"think:{args.num_think_layers}x{args.num_think_passes} "
        f"decoder:{args.num_decoder_layers} "
        f"effective_depth:{total_effective_layers} unique_blocks:{unique_layer_blocks} "
        f"skip_connections:{base_model.num_skip_connections}"
    )
    log0(
        f"enc/dec_blocks: heads={args.num_heads} kv={args.num_kv_heads} mlp={args.mlp_mult}x | "
        f"think_blocks: heads={args.think_num_heads} kv={args.think_kv_heads} mlp={args.think_mlp_mult}x"
    )
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

    if args.enable_looping_at > 0.0 and args.num_think_passes > 1:
        base_model.activate_passes(1)
        log0(f"progressive_recurrence: 1pass→{args.num_think_passes}passes@{args.enable_looping_at:.2f}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_frac <= 0:
            return 1.0
        if max_wallclock_ms is None:
            frac = step / max(args.iterations, 1)
        else:
            frac = elapsed_ms / max(max_wallclock_ms, 1e-9)
        if frac >= 1.0 - args.warmdown_frac:
            return max((1.0 - frac) / args.warmdown_frac, 0.0)
        return 1.0

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

    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}

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

        if args.enable_looping_at > 0.0 and args.num_think_passes > 1:
            _frac = (elapsed_ms / max_wallclock_ms) if max_wallclock_ms is not None else (step / max(args.iterations, 1))
            if base_model.active_think_passes < args.num_think_passes and _frac >= args.enable_looping_at:
                base_model.activate_passes(args.num_think_passes)
                log0(f"think_loop:enabled passes:{args.num_think_passes} step:{step} frac:{_frac:.3f}")

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

        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)

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

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------

    # Apply EMA weights before quantization
    log0("ema:applying EMA weights before quantization")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    tensor_bits = {
        "think_blocks": args.think_quant_bits,
        "tok_emb": args.embed_bits,
        "default": args.quant_bits,
    }
    log0(f"quantization: encoder/decoder={args.quant_bits}bit think={args.think_quant_bits}bit embed={args.embed_bits}bit clip_q={INT8_CLIP_Q}")

    if base_model.active_think_passes != args.num_think_passes:
        base_model.activate_passes(args.num_think_passes)
        log0(f"gptq:activating full recurrence passes:{args.num_think_passes}")

    hessians = None
    if args.gptq_calibration_batches > 0:
        log0(f"gptq:collecting hessians calibration_batches={args.gptq_calibration_batches}")
        train_loader_calib = DistributedTokenLoader(args.train_files, rank, world_size, device)
        hessians = collect_hessians(base_model, train_loader_calib, args, grad_accum_steps)
        log0(f"gptq:hessians collected for {len(hessians)} tensors")

    pre_quant = {n: t.detach().cpu().float().clone() for n, t in base_model.state_dict().items()}
    quant_obj, quant_stats = quantize_state_dict_int8(
        base_model.state_dict(), tensor_bits=tensor_bits, hessians=hessians,
        use_rht=args.use_rht, clip_sigmas=args.matrix_clip_sigmas,
        embed_clip_sigmas=args.embed_clip_sigmas,
    )
    if master_process:
        # Roundtrip sanity: catches RHT/inverse bugs before paying eval cost.
        dq = dequantize_state_dict_int8(quant_obj)
        worst = max(
            ((n, float((pre_quant[n] - dq[n].float()).abs().max().item())) for n in dq
             if pre_quant[n].ndim >= 2 and dq[n].shape == pre_quant[n].shape),
            key=lambda kv: kv[1], default=("", 0.0),
        )
        log0(f"roundtrip:worst tensor={worst[0]} max_abs_err={worst[1]:.4g}")
        del dq
    del pre_quant
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = brotli.compress(byte_shuffle(quant_raw), quality=11)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+brotli: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+brotli: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(byte_unshuffle(brotli.decompress(quant_blob_disk))), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_brotli_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_brotli_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()