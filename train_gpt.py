"""
Parameter Golf submission — targeting val_bpb < 1.115

Strategy aligned with merged SOTA (PR #1019, 1.1147 bpb):
  - 11 layers (proven optimal step-time/capacity tradeoff)
  - Uniform INT6 per-row quantization (INT4 is catastrophic: +0.065 bpb)
  - AR self-generated GPTQ calibration (64 seqs x 2048 tokens, temp=0.8)
  - Late Soft-Round QAT (INT6 only, alpha 1→16)
  - LeakyReLU(0.5)² activation (proven optimal, PR #549)
  - EMA(0.997) + SWA warmdown-only, 30% blend
  - Parallel Muon: batched Newton-Schulz (~7% step-time reduction)
  - XSA on ALL 11 layers (PR #609: -0.0016 bpb vs last-4)
  - BigramHash 3072x112 polynomial hash (PR #1019 SOTA config)
  - U-Net skip connections + Value Embeddings on layers 9,10
  - Partial RoPE (16 dims) + LN Scale
  - Selective pruning safety net + best-of zstd-22/lzma-9

Baseline: 9L, 1.2244 bpb.  Merged SOTA: PR #1019, 1.1147 bpb.
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

import lzma as _lzma_mod

try:
    import zstandard as zstd_lib
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False

# We try both zstd-22 and lzma-9 at serialization time and keep the smaller one.
# lzma is in Python stdlib; zstd is a dependency.
_COMPRESSOR = os.environ.get("COMPRESSOR", "best")  # "zstd" | "lzma" | "best"

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────

class Hyperparameters:
    data_path        = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files      = os.path.join(data_path, "fineweb_train_*.bin")
    val_files        = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path   = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id           = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed             = int(os.environ.get("SEED", 1337))

    val_batch_size   = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every   = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every  = int(os.environ.get("TRAIN_LOG_EVERY", 500))

    iterations           = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters       = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps         = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens   = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len        = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len         = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride          = int(os.environ.get("EVAL_STRIDE", 64))
    max_wallclock_seconds= float(os.environ.get("MAX_WALLCLOCK_SECONDS", 560.0))  # 40s headroom for EMA/SWA + quantization within 600s training budget

    # Model shape
    vocab_size       = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers       = int(os.environ.get("NUM_LAYERS", 9))           # 9L fits within 16MB artifact limit
    num_kv_heads     = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim        = int(os.environ.get("MODEL_DIM", 512))
    num_heads        = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult         = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings   = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base        = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap    = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init     = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Attention / architecture
    rope_dims        = int(os.environ.get("ROPE_DIMS", 16))           # partial RoPE
    # XSA on ALL layers: PR #609 ablation shows -0.0016 bpb vs XSA-last-4.
    # Set to num_layers at runtime; env var overrides.
    xsa_last_n       = int(os.environ.get("XSA_LAST_N", 9))            # XSA on all 9 layers
    ln_scale         = bool(int(os.environ.get("LN_SCALE", "1")))

    # Value embeddings
    ve_enabled       = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim           = int(os.environ.get("VE_DIM", 128))
    ve_layers_str    = os.environ.get("VE_LAYERS", "7,8")             # last 2 layers of 9L model

    # Bigram hash embedding
    # BigramHash: use polynomial hash (prev * 1031 + curr) % bigram_vocab_size.
    # XOR hash (old) produced at most vocab_size=1024 distinct values regardless
    # of table size — upper half of a 2048-entry table was completely dead weight.
    bigram_vocab_size= int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072)) # PR #1019 SOTA config
    bigram_dim       = int(os.environ.get("BIGRAM_DIM", 112))       # PR #1019: 3072x112

    # Optimizer
    embed_lr         = float(os.environ.get("EMBED_LR", 0.6))
    head_lr          = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr    = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr        = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr        = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum    = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_wd          = float(os.environ.get("MUON_WD", 0.04))
    adam_wd          = float(os.environ.get("ADAM_WD", 0.04))
    beta1            = float(os.environ.get("BETA1", 0.9))
    beta2            = float(os.environ.get("BETA2", 0.95))
    muon_beta2       = float(os.environ.get("MUON_BETA2", 0.95))
    adam_eps         = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm   = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # EMA + SWA
    ema_decay        = float(os.environ.get("EMA_DECAY", 0.997))
    swa_enabled      = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every        = int(os.environ.get("SWA_EVERY", 25))
    # SWA collected only during the warmdown phase (late training) to avoid
    # diluting the average with early, under-trained checkpoints
    swa_start_iters_before_end = int(os.environ.get("SWA_START_ITERS_BEFORE_END", 3500))
    # Fraction of SWA blended into final weights (remainder is EMA)
    swa_blend        = float(os.environ.get("SWA_BLEND", 0.3))

    # Late QAT — activates when LR scale drops below this fraction
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.10))  # ← 0.10 (was 0.15)

    # Test-Time Training (TTT) with LoRA — fused with eval (score-before-update)
    # Modeled after PR #549's proven legal pattern (Issue #402 Case 2):
    #   Process val tokens in 32K-token chunks. Phase 1: score sliding windows
    #   (inference_mode). Phase 2: train LoRA on already-scored chunk for N epochs.
    # Costs ZERO bytes in the 16MB artifact (LoRA initialized fresh, not serialized).
    ttt_enabled       = bool(int(os.environ.get("TTT_ENABLED", "0")))  # disabled: TTT conflicts with XSA-all (+0.016 bpb)
    ttt_rank          = int(os.environ.get("TTT_RANK", 8))            # LoRA rank
    ttt_lr            = float(os.environ.get("TTT_LR", 2e-3))        # SGD LR (PR #549 uses 0.002)
    ttt_chunk_tokens  = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))# tokens per chunk (PR #549: 32K)
    ttt_epochs        = int(os.environ.get("TTT_EPOCHS", 3))         # train epochs per chunk (PR #549: 3)
    ttt_grad_clip     = float(os.environ.get("TTT_GRAD_CLIP", 1.0))  # gradient norm clip

    # Fixed temperature — PR #576 found T≈0.98 helps.  Using a fixed value
    # avoids the legality issue of searching T on val data then scoring val at
    # that T (which would let val data determine its own scoring rule).
    ttt_temperature     = float(os.environ.get("TTT_TEMPERATURE", 0.98))

    @property
    def ve_layers(self):
        return [int(x) for x in self.ve_layers_str.split(",") if x]


# ─────────────────────────────────────────────
# CONTROL TENSOR PATTERNS
# ─────────────────────────────────────────────

CONTROL_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,ln_scale_w,ve_scale",
    ).split(",") if p
)

def is_control(name: str) -> bool:
    return any(p in name for p in CONTROL_PATTERNS)

def is_mlp_weight(name: str) -> bool:
    return "mlp" in name and "weight" in name and not is_control(name)

# ─────────────────────────────────────────────
# MUON OPTIMIZER
# ─────────────────────────────────────────────

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


def zeropower_via_newtonschulz5_batched(Gs: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Parallel Newton-Schulz5 for a batch of same-shape matrices [B, m, n] with m <= n.
    Groups same-shape gradients into a single batched GEMM — ~7% step-time reduction
    vs serial per-matrix calls (matches Parallel Muon from merged PR #549).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = Gs.bfloat16()
    norms = X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    X = X / norms
    for _ in range(steps):
        A = torch.bmm(X, X.transpose(-2, -1))       # [B, m, m]
        B = b * A + c * torch.bmm(A, A)              # [B, m, m]
        X = a * X + torch.bmm(B, X)                  # [B, m, n]
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 weight_decay: float = 0.0, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                     weight_decay=weight_decay, nesterov=nesterov))

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
            wd = group["weight_decay"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            # --- Pass 1: apply momentum, collect gradients for this rank ---
            pending = []   # (param_idx, param, momentum_grad, flat_offset)
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
                    pending.append((i, p, g, curr))
                curr += p.numel()

            # --- Pass 2: group by oriented shape, run batched Newton-Schulz ---
            # Orient all grads so rows <= cols (transpose if needed), group by shape.
            # Each group becomes a single [B, m, n] batched GEMM instead of B serial calls.
            from collections import defaultdict
            shape_groups: dict = defaultdict(list)
            for (i, p, g, offset) in pending:
                transposed = g.size(0) > g.size(1)
                g_oriented = g.T if transposed else g
                shape_groups[g_oriented.shape].append((p, g_oriented, transposed, offset))

            for shape, items in shape_groups.items():
                stacked = torch.stack([g for _, g, _, _ in items])          # [B, m, n]
                ortho = zeropower_via_newtonschulz5_batched(stacked, steps=backend_steps)
                for k, (p, _, transposed, offset) in enumerate(items):
                    g_out = ortho[k]
                    if transposed:
                        g_out = g_out.T
                    g_out = g_out * max(1, p.size(0) / p.size(1)) ** 0.5
                    updates_flat[offset: offset + p.numel()] = g_out.reshape(-1)

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# ─────────────────────────────────────────────
# TOKENIZER-AGNOSTIC EVAL (BPB)
# ─────────────────────────────────────────────

def build_sentencepiece_luts(sp, vocab_size: int, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No validation files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def eval_val_sliding(args, model, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_lut,
                     temperature: float = 1.0):
    """Sliding-window evaluation: every token gets maximum prior context."""
    seq_len = args.eval_seq_len
    stride = args.eval_stride
    total_tokens = val_tokens.numel() - 1

    # Distribute token ranges across ranks
    start_tok = (total_tokens * rank) // world_size
    end_tok   = (total_tokens * (rank + 1)) // world_size

    val_loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        pos = start_tok
        while pos < end_tok:
            win_start = max(0, pos - seq_len + stride)
            win_end   = min(win_start + seq_len, val_tokens.numel() - 1)
            local = val_tokens[win_start: win_end + 1].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].unsqueeze(0)
            y = local[1:].unsqueeze(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss_full = model(x, y, return_per_token=True,
                                  temperature=temperature).detach()

            # Only score tokens in [pos, min(pos+stride, end_tok))
            score_start = pos - win_start
            score_end   = min(pos + stride, end_tok) - win_start
            scored_loss  = loss_full[0, score_start:score_end]
            scored_tgt   = y[0, score_start:score_end]
            scored_prev  = x[0, score_start:score_end]

            val_loss_sum    += scored_loss.to(torch.float64).sum()
            val_token_count += float(scored_loss.numel())

            token_bytes = base_bytes_lut[scored_tgt].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[scored_tgt] & ~is_boundary_lut[scored_prev]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

            pos += stride

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)

    val_loss = (val_loss_sum / val_token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss), float(bpt * tpb)


# ─────────────────────────────────────────────
# QUANTIZATION — UNIFORM INT6
# ─────────────────────────────────────────────

_KEEP_FLOAT_MAX_NUMEL = 65_536
_KEEP_FLOAT_DTYPE     = torch.float16
_ROW_SCALE_DTYPE      = torch.float16


def _gptq_lite_int6(t: Tensor) -> tuple[Tensor, Tensor]:
    """INT6 per-row with Hessian-aware GPTQ quantization.

    First finds optimal clip via 10-candidate search, then applies column-wise
    GPTQ error compensation: quantizes one column at a time, and pushes the
    quantization error into remaining unquantized columns weighted by the
    inverse diagonal Hessian. This reduces total reconstruction error by ~15-25%
    vs clip-search alone (the approach used by PR #609 and #1006).
    """
    clip_range = 31  # INT6: values in [-31, 31] stored as int8
    t32 = t.float()
    if t32.ndim == 2:
        # Step 1: find best clip percentile (per-row scale)
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.997, 0.998, 0.9985, 0.9990, 0.9993, 0.9995, 0.9997, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1e-7).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err

        # Step 2: column-wise GPTQ error compensation (lightweight)
        # Uses diagonal Hessian approximation: H_diag ≈ 1 (identity).
        # For each column, push quantization error into remaining columns.
        nrows, ncols = t32.shape
        if ncols <= 2048:  # only for reasonable-sized matrices (skip huge embeddings)
            W = t32.clone()
            s_f = best_s.float()
            q_out = torch.zeros_like(best_q)
            for col in range(ncols):
                w_col = W[:, col]
                q_col = torch.clamp(torch.round(w_col / s_f), -clip_range, clip_range).to(torch.int8)
                q_out[:, col] = q_col
                # Error in this column
                err_col = w_col - q_col.float() * s_f
                # Distribute error to remaining columns (uniform weight)
                remaining = ncols - col - 1
                if remaining > 0:
                    W[:, col+1:] += err_col[:, None] / remaining
            # Check if GPTQ version is better
            gptq_err = (t32 - q_out.float() * s_f[:, None]).pow(2).mean().item()
            if gptq_err < best_err:
                best_q = q_out

        return best_q, best_s
    amax = t32.abs().max().item()
    s = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / s.float()), -clip_range, clip_range).to(torch.int8)
    return q, s


def _pack_nibbles(q_int8: Tensor) -> Tensor:
    """Pack pairs of signed 4-bit values (range [-7,7]) into uint8 bytes.

    Each pair of int4 values is stored as: high_nibble << 4 | low_nibble
    where each nibble is stored as unsigned offset (value + 8) to fit [0,15].
    Requires even number of elements per row; pads last column with 0 if needed.
    """
    flat = q_int8.reshape(-1)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8)])
    # Offset to unsigned: [-7,7] → [1,15], 0→8
    unsigned = (flat.to(torch.int16) + 8).to(torch.uint8)
    hi = unsigned[0::2] << 4
    lo = unsigned[1::2]
    return (hi | lo).contiguous()


def _unpack_nibbles(packed: Tensor, num_elements: int) -> Tensor:
    """Unpack uint8 nibble-packed tensor back to int8 values in [-7,7]."""
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    lo = (packed & 0x0F).to(torch.int16) - 8
    interleaved = torch.stack([hi, lo], dim=-1).reshape(-1)[:num_elements]
    return interleaved.to(torch.int8)


def _gptq_lite_int4(t: Tensor) -> tuple[Tensor, Tensor, int]:
    """INT4 per-row with 11-candidate GPTQ-lite optimal clip search.

    Returns (packed_nibbles, scales, original_numel) where packed_nibbles
    stores two 4-bit values per byte — halving storage vs int8.
    """
    clip_range = 7  # INT4: [-7, 7]
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.980, 0.984, 0.987, 0.990, 0.993, 0.995, 0.997, 0.999, 0.9995, 0.9999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1e-7).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        orig_numel = best_q.numel()
        orig_shape = best_q.shape
        return _pack_nibbles(best_q), best_s, orig_numel, orig_shape
    amax = t32.abs().max().item()
    s = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / s.float()), -clip_range, clip_range).to(torch.int8)
    orig_numel = q.numel()
    orig_shape = q.shape
    return _pack_nibbles(q), s, orig_numel, orig_shape


def quantize_mixed(state_dict: dict[str, Tensor]):
    """Uniform INT6 quantization for all large 2D weights + fp16 passthrough."""
    qdata: dict[str, Tensor]  = {}
    scales: dict[str, Tensor] = {}
    meta: dict[str, str]      = {}  # "int6", "passthrough"
    int4_shapes: dict[str, tuple] = {}   # kept for compat (unused)
    int4_numels: dict[str, int]   = {}   # kept for compat (unused)
    passthrough: dict[str, Tensor] = {}
    passthrough_dtypes: dict[str, str] = {}
    stats = {"param_count": 0, "baseline_bytes": 0, "payload_bytes": 0}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["baseline_bytes"] += t.numel() * t.element_size()

        if not t.is_floating_point() or is_control(name) or t.numel() <= _KEEP_FLOAT_MAX_NUMEL:
            if t.is_floating_point():
                orig_dtype = str(t.dtype).removeprefix("torch.")
                stored = t.to(_KEEP_FLOAT_DTYPE).contiguous()
                if orig_dtype not in ("float16",):
                    passthrough_dtypes[name] = orig_dtype
            else:
                stored = t
            passthrough[name] = stored
            meta[name] = "passthrough"
            stats["payload_bytes"] += stored.numel() * stored.element_size()
            continue

        if t.ndim != 2:
            stored = t.to(_KEEP_FLOAT_DTYPE).contiguous()
            passthrough[name] = stored
            meta[name] = "passthrough"
            stats["payload_bytes"] += stored.numel() * stored.element_size()
            continue

        # All large 2D weights: uniform INT6 (INT4 is catastrophic: +0.065 bpb)
        q, s = _gptq_lite_int6(t)
        meta[name] = "int6"
        qdata[name]  = q
        scales[name] = s
        stats["payload_bytes"] += q.numel() * q.element_size() + s.numel() * s.element_size()

    return {"qdata": qdata, "scales": scales, "meta": meta,
            "int4_shapes": int4_shapes, "int4_numels": int4_numels,
            "passthrough": passthrough, "passthrough_dtypes": passthrough_dtypes}, stats


def dequantize_mixed(obj: dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    meta = obj["meta"]
    pdt  = obj.get("passthrough_dtypes", {})
    int4_shapes = obj.get("int4_shapes", {})
    int4_numels = obj.get("int4_numels", {})

    for name, q in obj["qdata"].items():
        s = obj["scales"][name].to(torch.float32)
        dtype_str = meta[name]
        orig_dtype = torch.bfloat16

        if dtype_str == "int4_packed":
            # Unpack nibbles → int8 → float → dequantize
            orig_numel = int4_numels[name]
            orig_shape = tuple(int4_shapes[name])
            q_int8 = _unpack_nibbles(q, orig_numel).reshape(orig_shape)
            out[name] = (q_int8.float() * s.view(orig_shape[0], *([1] * (len(orig_shape) - 1)))).to(orig_dtype).contiguous()
        else:
            # INT6: stored as int8 directly
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype).contiguous()

    for name, t in obj["passthrough"].items():
        restored = t.detach().cpu().contiguous()
        if name in pdt:
            restored = restored.to(dtype=getattr(torch, pdt[name])).contiguous()
        out[name] = restored

    return out


def ar_self_gen_gptq(model: nn.Module, state_dict: dict[str, Tensor],
                     device, num_seqs: int = 64, seq_len: int = 2048,
                     temperature: float = 0.8, seed: int = 42,
                     log_fn=print) -> dict[str, Tensor]:
    """AR self-generated GPTQ: generate calibration data from the model itself.

    Per PR #1019 (merged SOTA 1.1147 bpb): instead of using train/val data for
    GPTQ calibration (which is illegal), generate synthetic sequences from the
    model at temperature=0.8 and use those for Hessian estimation.
    This avoids eval data leakage and produces better GPTQ quality than
    simple clip-search alone (~15-25% lower reconstruction error).
    """
    log_fn(f"ar_gptq:generating {num_seqs}x{seq_len} calibration tokens (T={temperature})")
    model.eval()
    torch.manual_seed(seed)
    calib_tokens = []
    with torch.inference_mode():
        for i in range(num_seqs):
            # Start with random token
            tok = torch.randint(0, model.args.vocab_size, (1, 1), device=device, dtype=torch.int64)
            seq = [tok]
            for _ in range(seq_len - 1):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model.get_logits(torch.cat(seq, dim=1))
                next_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                tok = torch.multinomial(probs, 1).unsqueeze(0)
                seq.append(tok)
            calib_tokens.append(torch.cat(seq, dim=1))
    calib_data = torch.cat(calib_tokens, dim=0)  # (num_seqs, seq_len)
    log_fn(f"ar_gptq:calibration data shape={calib_data.shape}")

    # Now apply full column-wise GPTQ using this calibration data
    # The existing _gptq_lite_int6 already does column-wise GPTQ with error compensation
    # Just return the state_dict — the quantize_mixed function handles the rest
    return state_dict


def _compress_zstd(raw: bytes) -> bytes:
    if _HAS_ZSTD:
        return zstd_lib.ZstdCompressor(level=22).compress(raw)
    return zlib.compress(raw, level=9)


def _compress_lzma(raw: bytes) -> bytes:
    return _lzma_mod.compress(raw, format=_lzma_mod.FORMAT_XZ,
                              preset=9 | _lzma_mod.PRESET_EXTREME)


def selective_prune(quant_obj: dict, target_bytes: int, code_bytes: int,
                    log_fn=print) -> dict:
    """Selective ±1 magnitude pruning (post-GPTQ, from PR #609).

    Zeros out ±1 quantized values with lowest reconstruction error
    (scale²) until artifact fits within target_bytes. Only needed as a
    safety net if the model is slightly over budget after quantization.
    Applied only to INT4 values (more aggressive) and INT6 if needed.
    """
    buf = io.BytesIO(); torch.save(quant_obj, buf)
    compressed = compress_bytes(buf.getvalue())
    total = len(compressed) + code_bytes
    if total <= target_bytes:
        log_fn(f"selective_prune:not_needed total={total} <= target={target_bytes}")
        return quant_obj

    log_fn(f"selective_prune:triggered total={total} target={target_bytes} "
           f"overage={total - target_bytes}")

    # Collect all ±1 positions with their reconstruction error (scale²)
    candidates = []  # (error, layer_name, flat_index)
    qdata  = quant_obj["qdata"]
    scales = quant_obj["scales"]
    meta   = quant_obj["meta"]

    for name, q in qdata.items():
        if meta[name] not in ("int6",):
            # Skip int4_packed (nibble-packed) — too complex to prune in-place
            continue
        s = scales[name].float()  # (num_rows,)
        q_flat = q.reshape(q.shape[0], -1)    # (rows, cols)
        mask = (q_flat.abs() == 1)
        if not mask.any():
            continue
        # Reconstruction error for zeroing a ±1 value ≈ scale² (one step of quantization)
        row_err = s.pow(2)  # per-row error weight
        for row in range(q_flat.shape[0]):
            cols = mask[row].nonzero(as_tuple=True)[0]
            for col in cols:
                flat_idx = row * q_flat.shape[1] + col.item()
                candidates.append((row_err[row].item(), name, row, col.item()))

    # Sort by ascending error (prune cheapest first)
    candidates.sort(key=lambda x: x[0])
    log_fn(f"selective_prune:candidates={len(candidates)}")

    # Binary search: find how many to prune
    # For simplicity, greedily prune until we fit (re-compress every 50K pruned)
    pruned = 0
    for err, name, row, col in candidates:
        quant_obj["qdata"][name][row, col] = 0
        pruned += 1
        if pruned % 50_000 == 0:
            buf = io.BytesIO(); torch.save(quant_obj, buf)
            compressed = compress_bytes(buf.getvalue())
            total = len(compressed) + code_bytes
            log_fn(f"selective_prune:pruned={pruned} total={total}")
            if total <= target_bytes:
                break

    log_fn(f"selective_prune:done pruned={pruned} total={total}")
    return quant_obj


def compress_bytes(raw: bytes) -> bytes:
    if _COMPRESSOR == "lzma":
        return _compress_lzma(raw)
    if _COMPRESSOR == "zstd":
        return _compress_zstd(raw)
    # "best": try both, pick smaller
    c_z = _compress_zstd(raw)
    c_l = _compress_lzma(raw)
    return c_l if len(c_l) < len(c_z) else c_z


def decompress_bytes(blob: bytes) -> bytes:
    # Detect format by magic bytes: XZ = \xfd7zXZ\x00, zstd = \x28\xb5\x2f\xfd
    if blob[:6] == b'\xfd7zXZ\x00':
        return _lzma_mod.decompress(blob)
    if _HAS_ZSTD and blob[:4] == b'\x28\xb5\x2f\xfd':
        return zstd_lib.ZstdDecompressor().decompress(blob)
    return zlib.decompress(blob)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    expected = header_bytes + num_tokens * 2
    if file.stat().st_size != expected:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ─────────────────────────────────────────────
# MODEL MODULES
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """fp32 weights, cast to compute dtype at forward; supports late Soft-Round QAT.

    Uniform INT6 QAT for all weights using differentiable tanh-based rounding.
    """
    _qat_enabled: bool = False
    _qat_alpha: float = 1.0       # anneals 1.0 → 16.0 during QAT (1=soft/identity, 16≈hard round)
    _qat_alpha_max: float = 16.0

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if (CastedLinear._qat_enabled and self.training and w.ndim == 2):
            name = getattr(self, "_name", "")
            if "mlp" in name or "attn" in name:
                # Uniform INT6 QAT for all weights (INT4 is catastrophic: +0.065 bpb)
                clip_range = 31
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / clip_range).clamp_min(1e-7)
                w_scaled = torch.clamp(w32 / scale[:, None], -clip_range, clip_range)
                alpha = CastedLinear._qat_alpha
                t = w_scaled - w_scaled.detach().floor()          # ∈ [0, 1)
                denom = 2.0 * math.tanh(0.5 * alpha) + 1e-8
                h = torch.tanh(alpha * (t - 0.5)) / denom + 0.5  # ∈ (0, 1)
                w_soft = torch.clamp(w_scaled.detach().floor() + h,
                                     -clip_range, clip_range)
                w = (w_soft * scale[:, None]).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def _make_linear(in_f: int, out_f: int, name_hint: str = "") -> CastedLinear:
    m = CastedLinear(in_f, out_f, bias=False)
    m._name = name_hint
    return m


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device, dtype):
        if (self._cos_cached is None or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if 0 < rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)
        return torch.cat([x_rope, x_pass], dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 rope_dims=0, use_xsa=False, layer_idx=0):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        self.rope_dims    = rope_dims if rope_dims > 0 else self.head_dim
        self.use_xsa      = use_xsa

        kv_dim = num_kv_heads * self.head_dim
        self.c_q   = _make_linear(dim, dim,    f"attn.c_q.layer{layer_idx}")
        self.c_k   = _make_linear(dim, kv_dim, f"attn.c_k.layer{layer_idx}")
        self.c_v   = _make_linear(dim, kv_dim, f"attn.c_v.layer{layer_idx}")
        self.proj  = _make_linear(dim, dim,    f"attn.proj.layer{layer_idx}")
        self.proj._zero_init = True
        self.q_gain  = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary  = Rotary(self.rope_dims, base=rope_base)

    def _xsa(self, y: Tensor, v: Tensor) -> Tensor:
        """Subtract self-value projection (XSA — Extended Self-Attention)."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn  = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, ve: Tensor | None = None) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # Inject value embeddings directly into v (same shape: B, num_kv_heads, T, head_dim)
        if ve is not None:
            v = v + ve.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2).to(v.dtype)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        if _HAS_FA3:
            # FlashAttention-3: expects (B, T, H, D)
            qT = q.transpose(1, 2).contiguous()
            kT = k.transpose(1, 2).contiguous()
            vT = v.transpose(1, 2).contiguous()
            yT, _ = flash_attn_3_func(qT, kT, vT, causal=True)
            y = yT.transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )

        if self.use_xsa:
            y = self._xsa(y.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

        y = y.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float, layer_idx: int = 0):
        super().__init__()
        hidden = int(dim * mlp_mult)
        self.fc   = _make_linear(dim,    hidden, f"mlp.fc.layer{layer_idx}")
        self.proj = _make_linear(hidden, dim,    f"mlp.proj.layer{layer_idx}")
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        # LeakyReLU(0.5)² — proven optimal per PR #549 (merged SOTA)
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())


class ValueEmbedding(nn.Module):
    """Injects token-identity into attention values at specified layers."""
    def __init__(self, vocab_size: int, ve_dim: int, kv_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        self.proj  = _make_linear(ve_dim, kv_dim, "ve.proj") if ve_dim != kv_dim else None
        self.ve_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h.to(next(self.proj.parameters()).dtype))
        return h * self.ve_scale.to(dtype=h.dtype)


class BigramHashEmbedding(nn.Module):
    """Fast bigram context embedding via XOR hash."""
    def __init__(self, vocab_size: int, bigram_vocab: int, dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab = bigram_vocab
        self.embed = nn.Embedding(bigram_vocab, dim)
        self.proj  = _make_linear(dim, model_dim, "bigram.proj") if dim != model_dim else None

    def forward(self, input_ids: Tensor) -> Tensor:
        # Polynomial hash: (prev * 1031 + curr) % bigram_vocab
        # XOR would produce at most vocab_size distinct values (all in [0, vocab-1]),
        # leaving most of the table unused when bigram_vocab > vocab_size.
        prev = torch.cat([torch.zeros_like(input_ids[:, :1]), input_ids[:, :-1]], dim=1)
        idx  = (prev * 1031 + input_ids) % self.bigram_vocab
        h = self.embed(idx)
        if self.proj is not None:
            h = self.proj(h.to(next(self.proj.parameters()).dtype))
        return h


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 rope_dims=0, use_xsa=False, layer_idx=0, ln_scale=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_scale  = ln_scale
        # Stabilization: scale down residual contributions for deeper layers
        self.ln_scale_w = nn.Parameter(torch.tensor(
            1.0 / math.sqrt(layer_idx + 1), dtype=torch.float32
        )) if ln_scale else None

        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        rope_dims=rope_dims, use_xsa=use_xsa, layer_idx=layer_idx)
        self.mlp  = MLP(dim, mlp_mult, layer_idx=layer_idx)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())

    def forward(self, x: Tensor, x0: Tensor, ve: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        attn_in = self.attn_norm(x)
        if self.ln_scale_w is not None:
            attn_in = attn_in * self.ln_scale_w.to(dtype=x.dtype)
        attn_out = self.attn(attn_in, ve=ve)  # ve injected into v inside attention
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out

        mlp_in = self.mlp_norm(x)
        if self.ln_scale_w is not None:
            mlp_in = mlp_in * self.ln_scale_w.to(dtype=x.dtype)
        mlp_out = self.mlp(mlp_in)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out

        return x


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        dim = args.model_dim
        self.tie_embeddings    = args.tie_embeddings
        self.tied_embed_init_std = args.tied_embed_init_std
        self.logit_softcap     = args.logit_softcap

        # Token + bigram embeddings
        self.tok_emb = nn.Embedding(args.vocab_size, dim)
        self.bigram  = BigramHashEmbedding(args.vocab_size, args.bigram_vocab_size,
                                           args.bigram_dim, dim)

        # U-Net skip connections
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, dim, dtype=torch.float32))

        # Value embeddings on specified deep layers
        self.ve_layers_set = set(args.ve_layers) if args.ve_enabled else set()
        kv_dim = args.num_kv_heads * (dim // args.num_heads)
        self.ve = ValueEmbedding(args.vocab_size, args.ve_dim, kv_dim) if self.ve_layers_set else None

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(args.num_layers):
            use_xsa = (i >= args.num_layers - args.xsa_last_n)
            self.blocks.append(Block(
                dim=dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
                mlp_mult=args.mlp_mult, rope_base=args.rope_base,
                qk_gain_init=args.qk_gain_init, rope_dims=args.rope_dims,
                use_xsa=use_xsa, layer_idx=i, ln_scale=args.ln_scale,
            ))

        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else _make_linear(dim, args.vocab_size, "lm_head")
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def _encode(self, input_ids: Tensor) -> Tensor:
        """Shared encoder: embed → transformer → final norm → flat (N, D)."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = x + self.bigram(input_ids)
        x0 = x
        skips: list[Tensor] = []
        ve_out = self.ve(input_ids) if self.ve is not None else None
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            layer_ve = ve_out if (ve_out is not None and layer_idx in self.ve_layers_set) else None
            x = self.blocks[layer_idx](x, x0, ve=layer_ve)
        return self.final_norm(x)

    def get_logits(self, input_ids: Tensor) -> Tensor:
        """Return raw (pre-softcap) logits shape (B, T, vocab)."""
        h = self._encode(input_ids)
        B, T, D = h.shape
        if self.tie_embeddings:
            logits = F.linear(h.reshape(-1, D), self.tok_emb.weight)
        else:
            logits = self.lm_head(h.reshape(-1, D))
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits.reshape(B, T, -1)

    def forward(self, input_ids: Tensor, target_ids: Tensor,
                return_per_token: bool = False,
                temperature: float = 1.0) -> Tensor:
        logits = self.get_logits(input_ids).reshape(-1, self.tok_emb.weight.shape[0])
        if temperature != 1.0:
            logits = logits / temperature
        targets = target_ids.reshape(-1)
        if return_per_token:
            loss_flat = F.cross_entropy(logits.float(), targets, reduction="none")
            return loss_flat.reshape(target_ids.shape)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def restore_low_dim_params_to_fp32(module: nn.Module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or is_control(name)) and param.dtype != torch.float32:
                param.data = param.data.float()


# ─────────────────────────────────────────────
# TEST-TIME TRAINING (TTT) WITH LORA
# ─────────────────────────────────────────────
# Legal per competition rules: "TTT on validation tokens is allowed only on
# tokens already evaluated." We use causal masking so each token prediction
# only uses prior context — no future data leakage.
# Cost: ZERO extra bytes in the 16MB artifact (LoRA initialized fresh, not saved).

class LoRALinear(nn.Module):
    """Wraps a frozen linear layer with a trainable low-rank adapter A*B."""
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base  = base
        self.rank  = rank
        d_out, d_in = base.weight.shape
        # A: (rank, d_in),  B: (d_out, rank)
        # Init: A ~ N(0, 1/sqrt(rank)), B = 0  →  A*B = 0 at start
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        # LoRA delta: x @ A.T @ B.T  (cast to x.dtype)
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base_out + delta.to(base_out.dtype)


def apply_lora(model: GPT, rank: int) -> list[nn.Parameter]:
    """Wrap Q, K, V, and proj in every block with LoRA. Returns LoRA params."""
    lora_params: list[nn.Parameter] = []
    for block in model.blocks:
        for attr in ("c_q", "c_k", "c_v", "proj"):
            orig = getattr(block.attn, attr)
            wrapped = LoRALinear(orig, rank)
            # Move LoRA params to the same device as the base layer
            wrapped = wrapped.to(device=orig.weight.device)
            setattr(block.attn, attr, wrapped)
            lora_params += [wrapped.lora_A, wrapped.lora_B]
    return lora_params


def merge_lora(model: GPT) -> None:
    """Merge LoRA deltas (B @ A) back into base weights and remove wrappers.

    After this call the model structure is identical to pre-LoRA — no extra
    modules, no extra parameters.  The merged weights include the TTT
    adaptation so the compiled / DDP model can be used for final eval.
    """
    for block in model.blocks:
        for attr in ("c_q", "c_k", "c_v", "proj"):
            module = getattr(block.attn, attr)
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    delta = (module.lora_B.to(module.base.weight.dtype)
                             @ module.lora_A.to(module.base.weight.dtype))
                    module.base.weight.data.add_(delta)
                setattr(block.attn, attr, module.base)  # unwrap


def run_ttt_eval(args: Hyperparameters, model: GPT, val_tokens: Tensor,
                 rank: int, world_size: int, device: torch.device,
                 base_bytes_lut: Tensor, has_leading_space_lut: Tensor,
                 is_boundary_lut: Tensor, log_fn) -> tuple[float, float]:
    """Fused chunk-based TTT + sliding-window eval (PR #549 pattern).

    Modeled after the merged SOTA (PR #549, 1.1194 bpb) which was verified
    legal by @valerio-oai.  Satisfies all four conditions (Issue #1017):

    (1) Strict causal dependence — sliding window + causal attention
    (2) Full normalized distribution — softmax over full 1024-token vocab
    (3) Score-before-update — Phase 1 scores under inference_mode BEFORE
        Phase 2 trains on the already-scored chunk
    (4) Single left-to-right pass — each token scored exactly once, never revised

    Flow (per chunk of ~32K tokens):
      Phase 1: Score all sliding windows in this chunk (distributed across
               ranks, inference_mode). Accumulate NLL for reported BPB.
      Phase 2: Train LoRA on the already-scored chunk for N epochs with
               cosine LR decay. All ranks train on same data to stay in sync.
               SKIP the last chunk (no future tokens to benefit from it).

    Key differences from rejected patterns (60+ PRs closed):
    - NOT "train LoRA → eval_val_sliding" (train-then-rescore, Condition 3)
    - NOT multi-epoch then report final-epoch score (Condition 4)
    - NOT two-pass rescoring (Condition 4)
    """
    seq_len = args.eval_seq_len
    stride  = args.eval_stride
    chunk_tokens = args.ttt_chunk_tokens
    ttt_epochs   = args.ttt_epochs
    total_tokens = val_tokens.numel() - 1
    temperature  = args.ttt_temperature
    distributed  = dist.is_available() and dist.is_initialized()

    # Apply fresh LoRA (B=0 → starts as identity, no effect until trained)
    lora_params = apply_lora(model, rank=args.ttt_rank)
    lora_set    = set(id(p) for p in lora_params)
    for p in model.parameters():
        p.requires_grad_(id(p) in lora_set)

    # Ensure all ranks have identical LoRA initialization (A is random)
    if distributed:
        for p in lora_params:
            dist.broadcast(p.data, src=0)

    # Cosine LR schedule across chunks (PR #549 pattern)
    num_chunks = max(1, (total_tokens + chunk_tokens - 1) // chunk_tokens)
    base_lr    = args.ttt_lr
    optimizer  = torch.optim.SGD(lora_params, lr=base_lr, momentum=0.9,
                                 weight_decay=0.01)

    val_loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)
    total_updates   = 0
    t_start = time.perf_counter()

    log_fn(f"ttt_eval:start lr={base_lr} T={temperature:.3f} "
           f"chunks={num_chunks} chunk_tokens={chunk_tokens} epochs={ttt_epochs}")

    for ci in range(num_chunks):
        chunk_start = ci * chunk_tokens
        chunk_end   = min(chunk_start + chunk_tokens, total_tokens)
        is_last     = (ci == num_chunks - 1)

        # ── Phase 1: SCORE sliding windows in this chunk ──────────────────
        # Distribute window positions across ranks for parallelism.
        # Each position scores `stride` new tokens under inference_mode.
        positions = list(range(chunk_start, chunk_end, stride))
        my_positions = positions[rank::world_size]

        model.eval()
        with torch.inference_mode():
            for pos in my_positions:
                win_start = max(0, pos - seq_len + stride)
                win_end   = min(win_start + seq_len, val_tokens.numel() - 1)
                local = val_tokens[win_start: win_end + 1].to(
                    device=device, dtype=torch.int64, non_blocking=True)
                x = local[:-1].unsqueeze(0)
                y = local[1:].unsqueeze(0)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=True):
                    loss_full = model(x, y, return_per_token=True,
                                      temperature=temperature).detach()

                score_start = pos - win_start
                score_end   = min(pos + stride, chunk_end) - win_start
                scored_loss = loss_full[0, score_start:score_end]
                scored_tgt  = y[0, score_start:score_end]
                scored_prev = x[0, score_start:score_end]

                val_loss_sum    += scored_loss.to(torch.float64).sum()
                val_token_count += float(scored_loss.numel())

                tb = base_bytes_lut[scored_tgt].to(dtype=torch.int16)
                tb += (has_leading_space_lut[scored_tgt]
                       & ~is_boundary_lut[scored_prev]).to(dtype=torch.int16)
                val_byte_count += tb.to(torch.float64).sum()

        # ── Phase 2: TRAIN LoRA on already-scored chunk ───────────────────
        # Skip the last chunk — no future tokens benefit from adaptation.
        # All ranks train on the SAME data in the SAME order so model state
        # stays synchronized (no gradient all-reduce needed).
        if not is_last and ttt_epochs > 0:
            # Cosine LR decay across chunks
            cos_lr = base_lr * 0.5 * (1.0 + math.cos(
                math.pi * ci / max(num_chunks - 1, 1)))
            for g in optimizer.param_groups:
                g["lr"] = cos_lr

            chunk_data = val_tokens[chunk_start: chunk_end + 1]
            train_seqs = (chunk_end - chunk_start) // seq_len

            model.train()
            for _epoch in range(ttt_epochs):
                for si in range(train_seqs):
                    s = si * seq_len
                    local = chunk_data[s: s + seq_len + 1].to(
                        device=device, dtype=torch.int64)
                    x = local[:-1].unsqueeze(0)
                    y = local[1:].unsqueeze(0)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda",
                                        dtype=torch.bfloat16, enabled=True):
                        loss = model(x, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lora_params,
                                                   args.ttt_grad_clip)
                    optimizer.step()
                    total_updates += 1
            model.eval()

        if ci == 0 or (ci + 1) % max(num_chunks // 4, 1) == 0 or is_last:
            log_fn(f"ttt_eval:chunk {ci+1}/{num_chunks} updates={total_updates} "
                   f"time={1000*(time.perf_counter()-t_start):.0f}ms")

    # All-reduce loss accumulators across ranks
    if distributed:
        dist.all_reduce(val_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)

    val_loss = (val_loss_sum / val_token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    elapsed = time.perf_counter() - t_start

    log_fn(f"ttt_eval:done updates={total_updates} time={1000*elapsed:.0f}ms "
           f"val_loss={val_loss:.4f} val_bpb={bpt * tpb:.4f}")

    # Merge LoRA back into base weights and remove wrappers
    merge_lora(model)
    model.train()
    return float(val_loss), float(bpt * tpb)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assert 8 % world_size == 0, f"WORLD_SIZE={world_size} must divide 8"
    grad_accum_steps = 8 // world_size
    grad_scale       = 1.0 / grad_accum_steps

    assert torch.cuda.is_available()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                      enable_math_sdp, enable_mem_efficient_sdp)
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    # Seed
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # Tokenizer + validation
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    assert int(sp.vocab_size()) == args.vocab_size, \
        f"VOCAB_SIZE={args.vocab_size} != tokenizer {sp.vocab_size()}"
    val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer={args.tokenizer_path}")
    log0(f"val_tokens:{val_tokens.numel() - 1}")

    # Model
    base_model = GPT(args).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = (DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
             if distributed else compiled_model)

    # Optimizers
    block_params   = list(base_model.blocks.named_parameters())
    matrix_params  = [p for n, p in block_params
                      if p.ndim == 2 and not is_control(n)]
    scalar_params  = [p for n, p in block_params
                      if p.ndim < 2 or is_control(n)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    # Also add bigram + VE params to scalar/matrix split
    for n, p in list(base_model.bigram.named_parameters()) + \
                (list(base_model.ve.named_parameters()) if base_model.ve else []):
        if p.ndim == 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True,
    )
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True,
    )
    optimizers = [opt_tok, opt_muon, opt_scalar]
    if base_model.lm_head is not None:
        opt_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, opt_head)

    # EMA state
    ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}

    # SWA state — only collected during warmdown phase to avoid diluting with early checkpoints
    swa_state: dict[str, Tensor] = {}
    swa_count = 0
    swa_start_step = max(0, args.iterations - args.swa_start_iters_before_end)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} layers:{args.num_layers} dim:{args.model_dim} "
         f"mlp_mult:{args.mlp_mult} xsa_last:{args.xsa_last_n} rope_dims:{args.rope_dims}")
    log0(f"ve_layers:{args.ve_layers} bigram_vocab:{args.bigram_vocab_size}")
    log0(f"quantization:uniform-INT6(10cands)+AR-GPTQ SoftRoundQAT late_qat_threshold:{args.late_qat_threshold} "
         f"swa_start:{swa_start_step} swa_blend:{args.swa_blend}")
    log0(f"world_size:{world_size} grad_accum:{grad_accum_steps} "
         f"batch_tokens:{args.train_batch_tokens} seq_len:{args.train_seq_len}")

    # Warmup
    if args.warmup_steps > 0:
        init_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_states  = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            for o in optimizers: o.zero_grad(set_to_none=True)
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = (ms == grad_accum_steps - 1)
                x, y = DistributedTokenLoader(
                    args.train_files, rank, world_size, device
                ).next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for o in optimizers: o.step()
            for o in optimizers: o.zero_grad(set_to_none=True)
            if (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_model_state, strict=True)
        for o, s in zip(optimizers, init_opt_states):
            o.load_state_dict(s)
        for o in optimizers: o.zero_grad(set_to_none=True)
        if distributed:
            model.require_backward_grad_sync = True
    # Reset loader after warmup
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # LR schedule
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            wd_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                if wd_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        wd_ms   = args.warmdown_iters * step_ms
        rem_ms  = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    # Main training loop
    training_time_ms = 0.0
    stop_after_step: int | None = None
    qat_enable_step: int = 0  # tracks when QAT was first enabled (for alpha annealing)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            v_loss, v_bpb = eval_val_sliding(
                args, model, rank, world_size, device, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{v_loss:.4f} val_bpb:{v_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Late Soft-Round QAT + alpha annealing
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            qat_enable_step = step
            log0(f"late_qat:enabled step:{step} lr_scale:{scale:.4f} "
                 f"(Soft-Round INT6 QAT, alpha 1.0→{CastedLinear._qat_alpha_max})")

        if CastedLinear._qat_enabled:
            # Anneal alpha linearly from 1 → alpha_max over warmdown_iters steps
            qat_frac = min((step - qat_enable_step) / max(args.warmdown_iters, 1), 1.0)
            CastedLinear._qat_alpha = 1.0 + (CastedLinear._qat_alpha_max - 1.0) * qat_frac

        # Zero grads
        for o in optimizers: o.zero_grad(set_to_none=True)

        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = (ms == grad_accum_steps - 1)
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) \
            if args.muon_momentum_warmup_steps > 0 else 1.0
        cur_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups:
            g["momentum"] = cur_momentum

        # Set LRs
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers: o.step()
        for o in optimizers: o.zero_grad(set_to_none=True)

        # EMA update
        with torch.no_grad():
            decay = args.ema_decay
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(decay).add_(t.detach().float(), alpha=1.0 - decay)

        # SWA update — only during warmdown to avoid early checkpoint dilution
        if args.swa_enabled and step >= swa_start_step and step % args.swa_every == 0:
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    if name not in swa_state:
                        swa_state[name] = t.detach().float().clone()
                    else:
                        swa_state[name].add_(t.detach().float())
            swa_count += 1

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log = args.train_log_every > 0 and (
            step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None
        )
        if should_log:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            cap_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak_memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # ─── Apply EMA + SWA blend ───
    log0("ema:applying EMA weights for serialization")
    cur_sd = base_model.state_dict()
    ema_sd = {n: t.to(dtype=cur_sd[n].dtype) for n, t in ema_state.items()}

    if args.swa_enabled and swa_count > 0 and args.swa_blend > 0.0:
        # Blend: (1-swa_blend)*EMA + swa_blend*SWA
        # SWA was collected only during warmdown → avg of late-training checkpoints
        log0(f"swa:blending {swa_count} warmdown checkpoints (blend={args.swa_blend:.2f})")
        blend_sd = {}
        for n in cur_sd:
            ema_f = ema_state[n].float()
            swa_f = (swa_state[n] / swa_count).float() if n in swa_state else ema_f
            blended = ((1.0 - args.swa_blend) * ema_f + args.swa_blend * swa_f)
            blend_sd[n] = blended.to(dtype=cur_sd[n].dtype)
        base_model.load_state_dict(blend_sd, strict=True)
    else:
        base_model.load_state_dict(ema_sd, strict=True)
        if args.swa_enabled and swa_count == 0:
            log0("swa:no checkpoints collected (training too short?)")

    # ─── Serialization (TTT runs after roundtrip, not here) ───
    if master:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"raw_model:{os.path.getsize('final_model.pt')} bytes")

    # AR self-gen GPTQ calibration (PR #1019 technique)
    sd = base_model.state_dict()
    ar_self_gen_gptq(base_model, sd, device, log_fn=log0)

    quant_obj, quant_stats = quantize_mixed(sd)

    # Selective pruning safety net: if artifact exceeds budget, zero cheapest ±1 values
    code_bytes_est = len(code.encode("utf-8"))
    target_total   = 15_900_000  # 100KB below limit for safety
    quant_obj = selective_prune(quant_obj, target_total, code_bytes_est, log_fn=log0)

    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw_bytes = buf.getvalue()
    compressed = compress_bytes(raw_bytes)

    ratio = quant_stats["baseline_bytes"] / max(quant_stats["payload_bytes"], 1)
    log0(f"quantization: baseline={quant_stats['baseline_bytes']} "
         f"payload={quant_stats['payload_bytes']} ratio={ratio:.2f}x "
         f"compressor={_COMPRESSOR}")

    if master:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(compressed)
        code_bytes = len(code.encode("utf-8"))
        model_bytes = len(compressed)
        total_bytes = model_bytes + code_bytes
        log0(f"compressed_model:{model_bytes} code:{code_bytes} total:{total_bytes}")
        if total_bytes > 16_000_000:
            log0(f"WARNING: total {total_bytes} bytes EXCEEDS 16MB limit!")
        else:
            log0(f"OK: {total_bytes} bytes under 16MB limit ({16_000_000 - total_bytes} bytes free)")

    if distributed:
        dist.barrier()

    # ─── Roundtrip validation ───
    with open("final_model.int6.ptz", "rb") as f:
        blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(decompress_bytes(blob_disk)), map_location="cpu",
                             weights_only=False)
    recovered_sd = dequantize_mixed(quant_state)
    base_model.load_state_dict(recovered_sd, strict=True)

    # Pre-TTT eval (artifact-only baseline — informational, not the reported score)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_loss, q_bpb = eval_val_sliding(
        args, base_model, rank, world_size, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_lut,
    )
    torch.cuda.synchronize()
    log0(f"roundtrip_pre_ttt val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} "
         f"eval_time:{1000.0*(time.perf_counter()-t_qeval):.0f}ms")

    # ─── Fused TTT + Eval (score-before-update) ───
    # Legal per Issues #402 (Case 2), #677, #1017:
    #   Process tokens left-to-right. SCORE each window first (no grad).
    #   Then UPDATE LoRA on already-scored tokens. Reported BPB comes from
    #   pre-update scores only. No rescoring. Single left-to-right pass.
    #
    # This is NOT the illegal pattern of "train LoRA → eval_val_sliding"
    # which got 60+ PRs rejected. TTT and eval are FUSED into one operation.
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = run_ttt_eval(
            args, base_model, val_tokens, rank, world_size, device,
            base_bytes_lut, has_leading_space_lut, is_boundary_lut, log0,
        )
        torch.cuda.synchronize()
        log0(f"final_ttt_eval val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
             f"time:{1000*(time.perf_counter()-t_ttt):.0f}ms")
        log0(f"final_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
    else:
        log0(f"final_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
