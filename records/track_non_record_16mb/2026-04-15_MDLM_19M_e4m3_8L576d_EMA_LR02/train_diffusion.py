"""
Text diffusion (MDLM) training for Parameter Golf.

Implements Masked Diffusion Language Model (Sahoo et al. 2024, Nie et al. 2025):
- Bidirectional attention (full context at every position)
- Absorbing-state (masking) forward process with learned mask embedding
- Continuous-time MDLM training objective (weighted MLM)
- NELBO-based BPB evaluation
- Optional depth recurrence for parameter efficiency

Key difference from AR baseline: same transformer applied bidirectionally,
with masking noise → the same weights get T denoising iterations at eval time.
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
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Diffusion-specific
    mask_eps = float(os.environ.get("MASK_EPS", 1e-3))       # avoid t=0 singularity
    eval_t_samples = int(os.environ.get("EVAL_T_SAMPLES", 64))  # MC samples for NELBO (mid-training)
    eval_max_seqs = int(os.environ.get("EVAL_MAX_SEQS", 4096))  # cap val sequences (mid-training; 0=all)
    # Final post-quant eval uses a tighter NELBO bound — more t-samples and more
    # val sequences. Defaults below are aggressive because the user said eval time
    # is not a constraint.
    final_eval_t_samples = int(os.environ.get("FINAL_EVAL_T_SAMPLES", 256))
    final_eval_max_seqs = int(os.environ.get("FINAL_EVAL_MAX_SEQS", 0))  # 0 = all

    # Depth recurrence: 0 = standard, N = N unique layers cycled (num_layers/N) times
    num_unique_layers = int(os.environ.get("NUM_UNIQUE_LAYERS", 0))

    # Optimizer
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
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))  # MDLM default

    # Quantization: "int8", "fp8", "int4", "int2"
    quant_mode = os.environ.get("QUANT_MODE", "int8")
    quant_group_size = int(os.environ.get("QUANT_GROUP_SIZE", 32))

    # QAT: 0=disabled, 2/4/8=simulate that bit width during training via STE
    # Progressive QAT: comma-separated schedule "bits:frac,bits:frac,..."
    # e.g. "8:0.25,4:0.5,2:0.75" means 8-bit at 25%, 4-bit at 50%, 2-bit at 75%
    qat_bits = int(os.environ.get("QAT_BITS", 0))
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.0))
    qat_schedule = os.environ.get("QAT_SCHEDULE", "")  # overrides qat_bits if set

    # Loss weight clamp: cap the 1/t weight to reduce gradient variance at low t
    weight_clamp = float(os.environ.get("WEIGHT_CLAMP", 100.0))

    # Rounding regularization: λ * ||W - Q(W)||² pushes weights toward quant grid
    round_reg_lambda = float(os.environ.get("ROUND_REG_LAMBDA", 0.0))
    round_reg_start_frac = float(os.environ.get("ROUND_REG_START_FRAC", 0.5))  # ramp up in last half

    # EMA: exponential moving average of model weights. 0 = disabled. Standard
    # diffusion training trick — averages the noisy SGD trajectory into a smoother
    # final model. ema_start_frac controls when EMA tracking begins (typically
    # after warmup so we don't average meaningless early weights).
    ema_decay = float(os.environ.get("EMA_DECAY", 0.0))
    ema_start_frac = float(os.environ.get("EMA_START_FRAC", 0.1))


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
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
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
# EMA (exponential moving average of model weights)
# -----------------------------

class EMA:
    """Exponential moving average of model parameters with per-step decay update.

    Maintains a shadow copy of all model parameters **in fp32** regardless of the
    live model's dtype. This is critical: at decay=0.999 the per-step update is
    (1-d) × weight ≈ 5e-5 for typical weights, which is below bf16 precision and
    gets rounded away — leaving the EMA shadow effectively frozen at init. fp32
    accumulator preserves the small updates.

    Update after every optimizer step. At eval/serialization time, swap live
    weights with the (cast-back) EMA copy, do the work, then restore.

    For ternary/low-bit STE: the EMA averages the live fp32 weights (which the STE
    quantizes on the forward pass), not the quantized weights. The EMA forward at
    eval time still re-quantizes through whatever STE path is active.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                # fp32 accumulator regardless of live dtype.
                self.shadow[name] = p.detach().to(dtype=torch.float32).clone()
        self._backup: dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            # Promote p to fp32 so the (1-d) × weight term keeps its precision
            # even when the live param is bf16.
            shadow.mul_(d).add_(p.detach().to(dtype=torch.float32), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Save current weights into _backup and copy EMA into the live model."""
        self._backup.clear()
        for name, p in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            self._backup[name] = p.detach().clone()
            # Cast fp32 shadow back to the live param's dtype (bf16 or fp32).
            p.data.copy_(shadow.to(dtype=p.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Copy backed-up weights back into the live model."""
        for name, p in model.named_parameters():
            backup = self._backup.get(name)
            if backup is not None:
                p.data.copy_(backup)
        self._backup.clear()


# -----------------------------
# TOKENIZER BPB SETUP
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


# -----------------------------
# MDLM LOSS AND EVAL
# -----------------------------

def sample_t_low_discrepancy(batch_size: int, eps: float, device: torch.device) -> Tensor:
    """Sample t in [eps, 1] with low-discrepancy (stratified) sampling — one uniform
    offset shared across an even grid. Standard MDLM variance-reduction trick
    (Sahoo et al. 2024). Not antithetic sampling (which would pair t with 1-t)."""
    u = torch.rand(1, device=device).item()
    offset = torch.arange(batch_size, device=device, dtype=torch.float32) / batch_size
    return ((u / batch_size + offset) % 1.0) * (1 - eps) + eps


def mdlm_loss(logits: Tensor, x0: Tensor, mask: Tensor, t: Tensor,
              weight_clamp: float = 15.0) -> Tensor:
    """MDLM training loss: min(1/t, clamp) * sum_masked(-log p(x0|xt)) / L, averaged over batch.

    Args:
        logits: [B, L, V] model output logits
        x0: [B, L] clean token ids
        mask: [B, L] boolean mask (True = masked position)
        t: [B] noise level per sample
        weight_clamp: cap on 1/t weight to reduce gradient variance at low t
    """
    L = x0.shape[1]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    log_p_x0 = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    masked_nll = (-log_p_x0 * mask.float()).sum(dim=-1)  # [B]
    weight = (1.0 / t).clamp(max=weight_clamp)
    return (weight * masked_nll / L).mean()


def eval_val_diffusion(
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
    """Compute NELBO-based BPB for diffusion model.

    NELBO per token = (1-eps)/K * sum_k (1/t_k) * mean_masked_k(-log p)
    BPB = (NELBO_per_token / ln2) * tokens_per_byte
    """
    eps = args.mask_eps
    K = args.eval_t_samples
    seq_len = args.train_seq_len

    # Create non-overlapping sequences from val tokens
    total_tokens = val_tokens.numel()
    usable = (total_tokens // seq_len) * seq_len
    total_seqs = usable // seq_len
    # Cap total sequences for faster eval (diffusion eval is K× slower than AR)
    if args.eval_max_seqs > 0:
        total_seqs = min(total_seqs, args.eval_max_seqs)

    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = max(local_batch_tokens // seq_len, 1)
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    # Evenly spaced t values for low-variance NELBO estimation
    t_values = torch.linspace(eps, 1.0, K)

    nelbo_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, seq_end)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len
            x0 = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64).reshape(-1, seq_len)
            B, L = x0.shape

            # Accumulate NELBO over K t-samples
            batch_nelbo = torch.zeros((), device=device, dtype=torch.float64)
            for t_val in t_values:
                tv = t_val.item()
                mask = torch.rand(B, L, device=device) < tv
                # Ensure at least 1 masked position per sample for numerical stability
                if tv < 0.01:
                    force_mask = torch.zeros(B, L, device=device, dtype=torch.bool)
                    force_mask[:, 0] = True
                    mask = mask | force_mask

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = model(x0, mask)  # [B, L, V]

                log_probs = F.log_softmax(logits.float(), dim=-1)
                log_p_x0 = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
                masked_nll = (-log_p_x0 * mask.float()).sum(dim=-1)  # [B]
                # Don't divide by L here — token_count does that later
                weighted = (1.0 / tv) * masked_nll  # total NELBO contribution per sequence
                batch_nelbo += weighted.sum().to(torch.float64)

            # MC estimate of NELBO per token: (1-eps)/K * weighted_sum
            nelbo_sum += batch_nelbo * (1 - eps) / K
            token_count += B * L

            # Byte counting (same approach as AR baseline)
            ids = x0.reshape(-1)
            prev_ids = torch.cat(
                [torch.zeros(B, 1, device=device, dtype=torch.int64), x0[:, :-1]], dim=1
            ).reshape(-1)
            token_bytes = base_bytes_lut[ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(nelbo_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    nelbo_per_token = nelbo_sum / token_count
    bits_per_token = nelbo_per_token.item() / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    model.train()
    return float(nelbo_per_token.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION (int8, fp8, int4)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",") if p
)
SMALL_TENSOR_MAX_NUMEL = 65_536


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


# --- INT4 group quantization ---

def quantize_int4_grouped(t: Tensor, group_size: int = 32) -> dict:
    """Quantize to symmetric INT4 with a bf16 scale per group."""
    flat = t.flatten().float()
    pad = (-len(flat)) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    groups = flat.view(-1, group_size)
    scales = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    normalized = groups / scales
    q = (normalized * 7).round().clamp(-8, 7).to(torch.int8)
    # Pack two int4 values into one uint8
    q_flat = q.view(-1, 2)
    packed = ((q_flat[:, 0] & 0xF) | (q_flat[:, 1] << 4)).to(torch.uint8)
    return {
        "packed": packed.contiguous(),
        "scales": scales.squeeze(1).to(torch.bfloat16).contiguous(),
        "shape": list(t.shape),
        "pad": pad,
        "group_size": group_size,
    }


def dequantize_int4_grouped(entry: dict) -> Tensor:
    packed, scales = entry["packed"], entry["scales"].float()
    group_size, pad = entry["group_size"], entry["pad"]
    shape = torch.Size(entry["shape"])
    low = (packed.to(torch.int8) << 4) >> 4   # sign-extend low nibble
    high = packed.to(torch.int8) >> 4          # sign-extend high nibble
    unpacked = torch.stack([low, high], dim=1).flatten().float()
    groups = unpacked.view(-1, group_size)
    result = groups * (scales.unsqueeze(1) / 7.0)
    result = result.flatten()
    if pad:
        result = result[:-pad]
    return result.view(shape).to(torch.bfloat16)


# --- INT2 group quantization ---

def quantize_int2_grouped(t: Tensor, group_size: int = 32) -> dict:
    """Quantize to symmetric INT2 (4 levels) with a bf16 scale per group.

    Uses per-group ABSMEAN scaling to match the ste_quantize(bits=2) training
    path. The stored grid is {-1.5, -0.5, 0.5, 1.5} × absmean.
    """
    flat = t.flatten().float()
    pad = (-len(flat)) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    groups = flat.view(-1, group_size)
    # Absmean scaling (matches ste_quantize).
    scales = groups.abs().mean(dim=1, keepdim=True).clamp(min=1e-10)
    # 4 levels: {-2, -1, 0, 1} mapped from {-1.5, -0.5, 0.5, 1.5} * scale/1.5
    q = (groups / scales * 1.5).round().clamp(-2, 1).to(torch.int8)
    # Pack 4 values per byte (2 bits each)
    q_shifted = (q + 2).to(torch.uint8)  # shift to {0, 1, 2, 3}
    q_flat = q_shifted.view(-1, 4)
    packed = q_flat[:, 0] | (q_flat[:, 1] << 2) | (q_flat[:, 2] << 4) | (q_flat[:, 3] << 6)
    return {
        "packed": packed.contiguous(),
        "scales": scales.squeeze(1).to(torch.bfloat16).contiguous(),
        "shape": list(t.shape), "pad": pad, "group_size": group_size,
    }


def dequantize_int2_grouped(entry: dict) -> Tensor:
    packed, scales = entry["packed"], entry["scales"].float()
    group_size, pad = entry["group_size"], entry["pad"]
    shape = torch.Size(entry["shape"])
    v0 = (packed & 0x03).to(torch.int8) - 2
    v1 = ((packed >> 2) & 0x03).to(torch.int8) - 2
    v2 = ((packed >> 4) & 0x03).to(torch.int8) - 2
    v3 = ((packed >> 6) & 0x03).to(torch.int8) - 2
    unpacked = torch.stack([v0, v1, v2, v3], dim=1).flatten().float()
    groups = unpacked.view(-1, group_size)
    result = groups * (scales.unsqueeze(1) / 1.5)
    result = result.flatten()
    if pad:
        result = result[:-pad]
    return result.view(shape).to(torch.bfloat16)


# --- Ternary group quantization (BitNet b1.58 recipe) ---
#
# 3 levels {-1, 0, +1} × per-group absmean scale. Packs 5 trits per byte
# (3^5 = 243 < 256), giving ~1.60 bits/param on disk.

_TERNARY_POW3 = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)


def quantize_ternary_grouped(t: Tensor, group_size: int = 32) -> dict:
    """Quantize to ternary {-1, 0, +1} with per-group absmean scale. 5 trits per byte."""
    flat = t.flatten().float()
    pad = (-len(flat)) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    groups = flat.view(-1, group_size)
    # Absmean scale — robust to outliers, matches BitNet b1.58.
    scale = groups.abs().mean(dim=1, keepdim=True).clamp(min=1e-10)
    # Threshold at 0.5 × scale: round(w / scale) clamped to {-1, 0, 1}.
    q = (groups / scale).round().clamp(-1, 1).to(torch.int8)  # values in {-1, 0, 1}
    q_shifted = (q + 1).to(torch.uint8)  # {0, 1, 2}

    # Pack 5 trits per byte as base-3 integer.
    # Pad q_shifted so its flat length is divisible by 5.
    q_flat = q_shifted.reshape(-1)
    trit_pad = (-q_flat.numel()) % 5
    if trit_pad:
        q_flat = F.pad(q_flat, (0, trit_pad), value=0)
    q_groups_of_5 = q_flat.view(-1, 5).to(torch.int32)
    pow3 = _TERNARY_POW3.to(q_groups_of_5.device)
    packed = (q_groups_of_5 * pow3).sum(dim=1).to(torch.uint8).contiguous()
    return {
        "packed": packed,
        "scales": scale.squeeze(1).to(torch.bfloat16).contiguous(),
        "shape": list(t.shape),
        "pad": pad,
        "trit_pad": trit_pad,
        "group_size": group_size,
    }


def dequantize_ternary_grouped(entry: dict) -> Tensor:
    packed, scales = entry["packed"], entry["scales"].float()
    group_size, pad, trit_pad = entry["group_size"], entry["pad"], entry["trit_pad"]
    shape = torch.Size(entry["shape"])
    # Unpack 5 trits per byte.
    packed_int = packed.to(torch.int32)
    trits = torch.zeros((packed_int.numel(), 5), dtype=torch.int32, device=packed.device)
    x = packed_int.clone()
    for i in range(5):
        trits[:, i] = x % 3
        x = x // 3
    q_shifted = trits.reshape(-1)  # values in {0, 1, 2}
    if trit_pad:
        q_shifted = q_shifted[:-trit_pad]
    q = (q_shifted.to(torch.float32) - 1.0)  # back to {-1, 0, 1}
    groups = q.view(-1, group_size)
    result = (groups * scales.unsqueeze(1)).flatten()
    if pad:
        result = result[:-pad]
    return result.view(shape).to(torch.bfloat16)


# --- NVFP4 / E2M1 quantization ---

# E2M1: 1 sign + 2 exponent + 1 mantissa. 8 non-negative values:
_E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # max=6.0


def quantize_e2m1_grouped(t: Tensor, group_size: int = 32) -> dict:
    """Quantize to NVFP4/E2M1 with per-group bf16 scale. Packs 2 values per byte."""
    flat = t.flatten().float()
    pad = (-len(flat)) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    groups = flat.view(-1, group_size)
    scale = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 6.0
    scaled = groups / scale
    # Quantize: nearest E2M1 value via LUT
    lut = torch.tensor(_E2M1_POS, device=t.device)
    signs = (scaled < 0).to(torch.uint8) * 8  # bit 3 = sign
    mags = scaled.abs()
    codes = (mags.unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8) | signs
    # Pack 2 per byte
    flat_codes = codes.view(-1)
    packed = (flat_codes[0::2] | (flat_codes[1::2] << 4)).contiguous()
    return {
        "packed": packed,
        "scales": scale.squeeze(1).to(torch.bfloat16).contiguous(),
        "shape": list(t.shape), "pad": pad, "group_size": group_size,
    }


def dequantize_e2m1_grouped(entry: dict) -> Tensor:
    packed, scales = entry["packed"], entry["scales"].float()
    group_size, pad = entry["group_size"], entry["pad"]
    shape = torch.Size(entry["shape"])
    lut = torch.tensor(_E2M1_POS + [-v for v in _E2M1_POS], device=packed.device)  # 16 values
    low = (packed & 0x0F).long()
    high = ((packed >> 4) & 0x0F).long()
    flat = torch.zeros(packed.numel() * 2, device=packed.device)
    flat[0::2] = lut[low]
    flat[1::2] = lut[high]
    groups = flat.view(-1, group_size)
    result = (groups * scales.unsqueeze(1)).flatten()
    if pad:
        result = result[:-pad]
    return result.view(shape).to(torch.bfloat16)


# --- FP8 quantization ---
#
# float8_e4m3fn: 1 sign + 4 exponent + 3 mantissa, range ~±448, 8 bits/param.
#                Better mantissa precision, less dynamic range. Closer to "uniform"
#                in the high-density region around zero.
# float8_e5m2:   1 sign + 5 exponent + 2 mantissa, range ~±57344, 8 bits/param.
#                Wider dynamic range, less mantissa precision. Better for activations
#                or weights with heavy tails.
# Both are 1 byte/param so the budget math is identical to int8.

def quantize_fp8(t: Tensor) -> dict:
    """Quantize to float8_e4m3fn (no scale needed)."""
    return {"data": t.to(torch.float8_e4m3fn).contiguous(), "shape": list(t.shape), "fmt": "e4m3"}


def quantize_fp8_e5m2(t: Tensor) -> dict:
    """Quantize to float8_e5m2 (no scale needed)."""
    return {"data": t.to(torch.float8_e5m2).contiguous(), "shape": list(t.shape), "fmt": "e5m2"}


def quantize_fp8_cascade(t: Tensor) -> dict:
    """Cascade-quantize: fp32 -> fp8_e4m3 -> fp8_e5m2, store as e5m2.

    Both casts are lossy in different ways: e4m3 has 3 mantissa bits, e5m2 has 2.
    Cascading them snaps the weight to ~49 distinct values (verified empirically
    on Gaussian weights at scale 0.05) instead of ~256, while still being stored
    in an 8-bit format. The lower byte-level entropy compresses ~6% smaller under
    lzma than single-cast e5m2, and ~25% smaller than int8. Same dequant path
    as e5m2 since storage is identical.
    """
    cascaded = t.to(torch.float8_e4m3fn).to(torch.float8_e5m2)
    return {"data": cascaded.contiguous(), "shape": list(t.shape), "fmt": "e4m3->e5m2"}


def dequantize_fp8(entry: dict) -> Tensor:
    return entry["data"].to(torch.float32).to(torch.bfloat16).contiguous()


# --- Multi-mode quantize/dequantize ---

def quantize_state_dict(state_dict: dict[str, Tensor], mode: str = "int4",
                        group_size: int = 32) -> tuple[dict, dict]:
    """Quantize state dict with the given mode: 'int8', 'fp8', or 'int4'."""
    entries: dict[str, dict] = {}
    passthrough: dict[str, Tensor] = {}
    stats = {"param_count": 0, "payload_bytes": 0, "baseline_bytes": 0}

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["baseline_bytes"] += tensor_nbytes(t)

        # Small tensors and control tensors: keep as fp16
        if (not t.is_floating_point() or t.numel() <= SMALL_TENSOR_MAX_NUMEL
                or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)):
            stored = t.to(torch.float16).contiguous() if t.is_floating_point() else t
            passthrough[name] = stored
            stats["payload_bytes"] += tensor_nbytes(stored)
            continue

        if mode == "fp8":
            entries[name] = quantize_fp8(t)
            stats["payload_bytes"] += t.numel()  # 1 byte per param
        elif mode == "fp8_e5m2":
            entries[name] = quantize_fp8_e5m2(t)
            stats["payload_bytes"] += t.numel()  # 1 byte per param
        elif mode == "fp8_cascade":
            entries[name] = quantize_fp8_cascade(t)
            stats["payload_bytes"] += t.numel()  # 1 byte per param (pre-compression)
        elif mode == "int4":
            entries[name] = quantize_int4_grouped(t, group_size)
            e = entries[name]
            stats["payload_bytes"] += tensor_nbytes(e["packed"]) + tensor_nbytes(e["scales"])
        elif mode == "int2":
            entries[name] = quantize_int2_grouped(t, group_size)
            e = entries[name]
            stats["payload_bytes"] += tensor_nbytes(e["packed"]) + tensor_nbytes(e["scales"])
        elif mode == "ternary":
            entries[name] = quantize_ternary_grouped(t, group_size)
            e = entries[name]
            stats["payload_bytes"] += tensor_nbytes(e["packed"]) + tensor_nbytes(e["scales"])
        elif mode == "e2m1":
            entries[name] = quantize_e2m1_grouped(t, group_size)
            e = entries[name]
            stats["payload_bytes"] += tensor_nbytes(e["packed"]) + tensor_nbytes(e["scales"])
        else:  # int8 fallback
            t32 = t.float()
            if t32.ndim == 2:
                clip_abs = torch.quantile(t32.abs(), 0.9999984, dim=1)
                scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
                q = torch.clamp(torch.round(t32 / scale[:, None]), -127, 127).to(torch.int8)
                entries[name] = {"q": q.contiguous(), "scale": scale.to(torch.float16).contiguous(), "scheme": "per_row"}
            else:
                clip_abs = float(torch.quantile(t32.abs().flatten(), 0.9999984).item()) if t32.numel() else 0.0
                scale = torch.tensor(max(clip_abs / 127.0, 1.0 / 127.0), dtype=torch.float32)
                q = torch.clamp(torch.round(t32 / scale), -127, 127).to(torch.int8)
                entries[name] = {"q": q.contiguous(), "scale": scale, "scheme": "per_tensor"}
            e = entries[name]
            stats["payload_bytes"] += tensor_nbytes(e["q"]) + tensor_nbytes(e["scale"])

    obj = {"__quant_mode__": mode, "entries": entries, "passthrough": passthrough}
    return obj, stats


def dequantize_state_dict(obj: dict) -> dict[str, Tensor]:
    mode = obj["__quant_mode__"]
    out: dict[str, Tensor] = {}

    for name, entry in obj["entries"].items():
        if mode in ("fp8", "fp8_e5m2", "fp8_cascade"):
            out[name] = dequantize_fp8(entry)
        elif mode == "int4":
            out[name] = dequantize_int4_grouped(entry)
        elif mode == "int2":
            out[name] = dequantize_int2_grouped(entry)
        elif mode == "ternary":
            out[name] = dequantize_ternary_grouped(entry)
        elif mode == "e2m1":
            out[name] = dequantize_e2m1_grouped(entry)
        else:  # int8
            q, s = entry["q"], entry["scale"]
            if entry["scheme"] == "per_row":
                out[name] = (q.float() * s.float()[:, None]).to(torch.bfloat16).contiguous()
            else:
                out[name] = (q.float() * float(s.item())).to(torch.bfloat16).contiguous()

    for name, t in obj["passthrough"].items():
        out[name] = t.to(torch.bfloat16).contiguous() if t.is_floating_point() else t
    return out


# -----------------------------
# DATA LOADING (same as baseline)
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
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
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> Tensor:
        """Returns x0: [batch, seq_len] clean token sequences."""
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        return local.reshape(-1, seq_len).to(self.device, non_blocking=True)


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[:usable]


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


# --- QAT: Straight-Through Estimator for simulated quantization ---

def ste_quantize(w: Tensor, bits: int, group_size: int = 32) -> Tensor:
    """Simulate N-bit group quantization with straight-through estimator.
    Forward: returns quantize-then-dequantize(w). Backward: gradients flow through w unchanged.

    bits values:
      8  = INT8 symmetric, per-group absmax
      4  = INT4 symmetric, per-group absmax
      -4 = E2M1 (NVFP4) 4-bit float, per-group absmax
      3  = ternary {-1, 0, +1} with per-group ABSMEAN scaling (BitNet b1.58 recipe)
      2  = INT2 {-1.5, -0.5, 0.5, 1.5} with per-group ABSMEAN scaling
    """
    orig_shape = w.shape
    flat = w.reshape(-1)
    pad = (-len(flat)) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    groups = flat.reshape(-1, group_size)

    if bits in (3, 2):
        # Low-bit modes use absmean scaling (robust to outliers).
        scale = groups.abs().mean(dim=1, keepdim=True).clamp(min=1e-10)
    else:
        scale = groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)

    if bits == 8:
        # INT8 symmetric: 256 levels, range [-127, 127]
        q = (groups / scale * 127).round().clamp(-128, 127)
        dequant = q * scale / 127
    elif bits == -4:
        # E2M1 (NVFP4): non-uniform 4-bit float, scale by max/6.0
        e2m1_scale = scale / 6.0  # rescale so max E2M1 value maps to original scale
        scaled = groups / e2m1_scale
        lut = torch.tensor(_E2M1_POS, device=w.device)
        signs = (scaled < 0).float()
        nearest = (scaled.abs().unsqueeze(-1) - lut.to(w.device)).abs().argmin(dim=-1)
        dequant = lut.to(w.device)[nearest] * (1 - 2 * signs) * e2m1_scale
    elif bits == 4:
        # INT4 symmetric: 16 levels, range [-8, 7], scale by 7
        q = (groups / scale * 7).round().clamp(-8, 7)
        dequant = q * scale / 7
    elif bits == 3:
        # Ternary {-1, 0, +1} × absmean scale. BitNet b1.58 recipe.
        q = (groups / scale).round().clamp(-1, 1)
        dequant = q * scale
    elif bits == 2:
        # INT2 {-1.5, -0.5, 0.5, 1.5} × absmean scale (no zero state — ternary is
        # usually better, this is kept for A/B comparison).
        q = (groups / scale * 1.5).round().clamp(-2, 1)
        dequant = q * scale / 1.5
    else:
        return w  # no QAT

    result = dequant.reshape(-1)
    if pad:
        result = result[:-pad]
    return (result.reshape(orig_shape) - w).detach() + w


class CastedLinear(nn.Linear):
    # Class-level QAT config (set from main before model creation)
    _qat_bits: int = 0
    _qat_group_size: int = 32
    _qat_enabled: bool = False  # toggled during training

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 norm_input: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        # RMSNorm the input before the matmul. Used on projection layers at low bit
        # widths — the binary AR submission's NormedBinaryLinear trick. Pins the output
        # magnitude to the input magnitude instead of letting discrete weights eat it.
        self.norm_input = norm_input

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_input:
            x = F.rms_norm(x, (x.size(-1),))
        w = self.weight
        if CastedLinear._qat_enabled and CastedLinear._qat_bits > 0 and w.ndim == 2:
            w = ste_quantize(w.float(), CastedLinear._qat_bits, CastedLinear._qat_group_size)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
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
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len or self._cos_cached.device != device):
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


class SelfAttention(nn.Module):
    """Bidirectional self-attention (no causal mask) for diffusion."""

    # Set from main before model creation. Turns on RMSNorm-before-matmul on the
    # output projection (which receives un-normalized attention output).
    _norm_proj_input: bool = False

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False, norm_input=SelfAttention._norm_proj_input)
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
            q, k, v, attn_mask=None,
            is_causal=False,  # BIDIRECTIONAL for diffusion
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    # Set from main before model creation. Turns on RMSNorm-before-matmul on the
    # output projection (which receives un-normalized ReLU² output).
    _norm_proj_input: bool = False

    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False, norm_input=MLP._norm_proj_input)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = SelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# -----------------------------
# DIFFUSION TRANSFORMER
# -----------------------------

class DiffusionTransformer(nn.Module):
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
        num_unique_layers: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap

        # Embedding: vocab_size+1 for [MASK] token (standard MDLM approach)
        self.mask_id = vocab_size
        self.tok_emb = nn.Embedding(vocab_size + 1, model_dim)

        # Depth recurrence: N unique layers applied (num_layers/N) times each
        if num_unique_layers > 0 and num_layers % num_unique_layers == 0:
            self.num_unique_blocks = num_unique_layers
            self.recurrence_iters = num_layers // num_unique_layers
            self.use_skip_connections = False
        else:
            self.num_unique_blocks = num_layers
            self.recurrence_iters = 1
            self.use_skip_connections = True

        # U-Net skip connections (only without recurrence)
        if self.use_skip_connections:
            self.num_encoder_layers = num_layers // 2
            self.num_decoder_layers = num_layers - self.num_encoder_layers
            n_skip = min(self.num_encoder_layers, self.num_decoder_layers)
            self.skip_weights = nn.Parameter(torch.ones(n_skip, model_dim, dtype=torch.float32))
        else:
            self.num_encoder_layers = 0
            self.num_decoder_layers = 0
            self.skip_weights = nn.Parameter(torch.empty(0, model_dim))

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(self.num_unique_blocks)
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

    def forward(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        """Forward pass: input_ids + mask -> logits [B, L, vocab_size].

        Args:
            input_ids: [B, L] clean token ids (before masking)
            mask: [B, L] boolean mask (True = this position is masked)
        """
        # Standard MDLM: replace masked positions with [MASK] token id, then embed
        masked_ids = torch.where(mask, self.mask_id, input_ids)
        # Apply QAT to embedding weights if enabled (so quantization-aware for storage)
        if CastedLinear._qat_enabled and CastedLinear._qat_bits > 0:
            emb_w = ste_quantize(self.tok_emb.weight.float(), CastedLinear._qat_bits, CastedLinear._qat_group_size)
            x = F.embedding(masked_ids, emb_w.to(self.tok_emb.weight.dtype))
        else:
            x = self.tok_emb(masked_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        if self.use_skip_connections:
            skips: list[Tensor] = []
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)
        else:
            # Depth recurrence: cycle through unique blocks multiple times
            for _iter in range(self.recurrence_iters):
                for block in self.blocks:
                    x = block(x, x0)

        x = self.final_norm(x)
        if self.tie_embeddings:
            # Project to vocab_size only (exclude [MASK] embedding row)
            out_w = self.tok_emb.weight[:self.vocab_size]
            if CastedLinear._qat_enabled and CastedLinear._qat_bits > 0:
                out_w = ste_quantize(out_w.float(), CastedLinear._qat_bits, CastedLinear._qat_group_size)
            logits = F.linear(x, out_w.to(x.dtype))
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    print("MDLM: starting...", flush=True)
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    print(f"MDLM: config layers={args.num_layers} dim={args.model_dim} unique_layers={args.num_unique_layers}", flush=True)
    print("MDLM: compiling newton-schulz...", flush=True)
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    print("MDLM: newton-schulz compiled", flush=True)

    # --- Distributed + CUDA setup ---
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
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
    log0(f"MDLM Diffusion Training | num_unique_layers={args.num_unique_layers}")
    log0(f"Running PyTorch {torch.__version__}")
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)

    # --- Tokenizer + validation ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer vocab_size={int(sp.vocab_size())}")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val tokens:{val_tokens.numel()} sequences:{val_tokens.numel() // args.train_seq_len}")

    # --- QAT setup ---
    CastedLinear._qat_bits = args.qat_bits
    CastedLinear._qat_group_size = args.quant_group_size
    # Low-bit modes (ternary/int2): enable QAT from step 0, turn on normalized inputs
    # on projection layers. E2M1 and higher-bit modes leave these off so existing
    # behavior is unchanged.
    low_bit = args.qat_bits in (2, 3)
    qat_at_step_zero = args.qat_bits > 0 and args.qat_start_frac <= 0.0 and not args.qat_schedule
    CastedLinear._qat_enabled = qat_at_step_zero
    SelfAttention._norm_proj_input = low_bit
    MLP._norm_proj_input = low_bit

    # --- Model ---
    base_model = DiffusionTransformer(
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
        num_unique_layers=args.num_unique_layers,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"MDLM: model created, {n_params:,} params. Compiling model...", flush=True)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    print("MDLM: model compiled", flush=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # --- Optimizers (same split as baseline) ---
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
    recur_info = f" ({base_model.num_unique_blocks} unique x{base_model.recurrence_iters})" if base_model.recurrence_iters > 1 else ""
    log0(f"model_params:{n_params}{recur_info}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"layers:{args.num_layers} dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} mlp_mult:{args.mlp_mult}")
    log0(f"mask_eps:{args.mask_eps} eval_t_samples:{args.eval_t_samples} grad_clip:{args.grad_clip_norm} weight_clamp:{args.weight_clamp}")
    log0(f"quant_mode:{args.quant_mode} qat_bits:{args.qat_bits} qat_start_frac:{args.qat_start_frac} "
         f"norm_proj_input:{low_bit} qat_at_step_zero:{qat_at_step_zero}")

    # --- Data loader + warmup ---
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

    # Warmup: compile forward/backward, then restore weights
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x0 = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                B, L = x0.shape
                t = sample_t_low_discrepancy(B, args.mask_eps, device)
                mask = torch.rand(B, L, device=device) < t[:, None]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = model(x0, mask)
                    loss = mdlm_loss(logits, x0, mask, t, args.weight_clamp)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if warmup_step + 1 == args.warmup_steps or (warmup_step + 1) % 10 == 0:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # --- EMA setup (after warmup so the shadow starts at the post-warmup-restored
    # weights, which are the actual training initial weights) ---
    ema: EMA | None = None
    if args.ema_decay > 0:
        ema = EMA(base_model, decay=args.ema_decay)
        log0(f"ema:enabled decay:{args.ema_decay} start_frac:{args.ema_start_frac}")

    # --- Main training loop ---
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    ema_started = False  # becomes True after the first ema.update() call
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Mid-training eval uses EMA weights only once the shadow has started
            # updating. Before that, ema.shadow == initial weights and swapping
            # them in would produce a ~uniform-prior val_bpb.
            if ema is not None and ema_started:
                ema.apply_to(base_model)
            val_loss, val_bpb = eval_val_diffusion(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            if ema is not None and ema_started:
                ema.restore(base_model)
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

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x0 = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            B, L = x0.shape
            t = sample_t_low_discrepancy(B, args.mask_eps, device)
            mask = torch.rand(B, L, device=device) < t[:, None]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x0, mask)
                loss = mdlm_loss(logits, x0, mask, t, args.weight_clamp)
            # Rounding regularization: push weights toward quantization grid points
            if args.round_reg_lambda > 0 and CastedLinear._qat_bits > 0:
                elapsed = training_time_ms + 1000.0 * (time.perf_counter() - t0)
                train_frac = elapsed / max_wallclock_ms if max_wallclock_ms else step / args.iterations
                reg_scale = max(0.0, (train_frac - args.round_reg_start_frac) / max(1.0 - args.round_reg_start_frac, 1e-6))
                if reg_scale > 0:
                    round_loss = sum(
                        (p.float() - ste_quantize(p.float(), CastedLinear._qat_bits, CastedLinear._qat_group_size)).pow(2).mean()
                        for p in base_model.parameters()
                        if p.ndim == 2 and p.numel() > SMALL_TENSOR_MAX_NUMEL  # skip small/control tensors
                    )
                    loss = loss + args.round_reg_lambda * reg_scale * round_loss
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
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

        # EMA update (after weights have been updated). Skip the very first
        # ema_start_frac of training so the shadow doesn't average meaningless
        # early weights. Once started, every step contributes.
        if ema is not None:
            ema_frac = approx_training_time_ms / max_wallclock_ms if max_wallclock_ms else step / args.iterations
            if ema_frac >= args.ema_start_frac:
                ema.update(base_model)
                if not ema_started:
                    ema_started = True
                    log0(f"ema:started step:{step} ema_frac:{ema_frac:.4f}")

        # QAT scheduling: progressive or fixed
        frac = approx_training_time_ms / max_wallclock_ms if max_wallclock_ms else step / args.iterations
        if args.qat_schedule:
            # Progressive: parse "8:0.25,4:0.5,2:0.75" → apply highest-frac stage reached
            new_bits = 0
            for entry in args.qat_schedule.split(","):
                bits_str, frac_str = entry.strip().split(":")
                if frac >= float(frac_str):
                    new_bits = int(bits_str)
            if new_bits != CastedLinear._qat_bits:
                CastedLinear._qat_bits = new_bits
                CastedLinear._qat_enabled = new_bits > 0
                log0(f"QAT stage: {new_bits}-bit at step {step} ({frac:.1%} of training)")
        elif args.qat_bits > 0 and not CastedLinear._qat_enabled:
            if frac >= args.qat_start_frac:
                CastedLinear._qat_enabled = True
                log0(f"QAT enabled: {args.qat_bits}-bit at step {step} ({frac:.1%} of training)")

        should_log = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0)
        if should_log:
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

    CastedLinear._qat_enabled = False  # disable QAT for eval and serialization
    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # If EMA is active and has started updating, swap the live model to EMA
    # weights for the final serialization + final eval. We do not restore — the
    # artifact and reported number both use the EMA model, which is the standard
    # diffusion practice. If the run was so short that EMA never started, fall
    # back to the live weights rather than the frozen initial shadow.
    if ema is not None and ema_started:
        ema.apply_to(base_model)
        log0("ema:applied for serialization + final eval")
    elif ema is not None:
        log0("ema:skipped (never started — using live weights)")

    # --- Serialization + roundtrip validation ---
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes")

    quant_obj, quant_stats = quantize_state_dict(
        base_model.state_dict(), mode=args.quant_mode, group_size=args.quant_group_size
    )
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    raw_bytes = quant_buf.getvalue()
    # Try both zlib and lzma, use whichever is smaller
    import lzma as _lzma
    blob_zlib = zlib.compress(raw_bytes, level=9)
    blob_lzma = _lzma.compress(raw_bytes, preset=6)
    quant_blob = blob_lzma if len(blob_lzma) < len(blob_zlib) else blob_zlib
    compressor = "lzma" if len(blob_lzma) < len(blob_zlib) else "zlib"
    artifact_name = "final_model.int8.ptz"
    if master_process:
        with open(artifact_name, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(artifact_name)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_bytes"] / max(quant_stats["payload_bytes"], 1)
        log0(f"Serialized model {args.quant_mode}+{compressor}: {quant_file_bytes} bytes (ratio:{ratio:.2f}x) [zlib:{len(blob_zlib)} lzma:{len(blob_lzma)}]")
        log0(f"Total submission size: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open(artifact_name, "rb") as f:
        quant_blob_disk = f.read()
    try:
        decompressed = zlib.decompress(quant_blob_disk)
    except zlib.error:
        decompressed = _lzma.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    # Use the tighter final eval budget (more K, more sequences) for the number we
    # actually report. Mid-training evals are unaffected.
    _orig_k, _orig_cap = args.eval_t_samples, args.eval_max_seqs
    args.eval_t_samples = args.final_eval_t_samples
    args.eval_max_seqs = args.final_eval_max_seqs
    log0(f"final_eval K:{args.eval_t_samples} max_seqs:{args.eval_max_seqs}")
    q_val_loss, q_val_bpb = eval_val_diffusion(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    args.eval_t_samples, args.eval_max_seqs = _orig_k, _orig_cap
    torch.cuda.synchronize()
    log0(
        f"final_{args.quant_mode}_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_{args.quant_mode}_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()

    # Bypass Python's normal shutdown — torch._inductor.async_compile's atexit
    # handler hangs on its compile-worker pool shutdown for 300s, eating ~$2 of
    # idle 8xH100 time per run. Flush our streams first so no log lines are lost.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
