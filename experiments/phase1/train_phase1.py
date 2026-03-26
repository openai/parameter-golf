#!/usr/bin/env python3
"""
Phase 1: Competitive standard-stack training script.

Monkey-patches the baseline train_gpt.py with:
  1. Int6 quantization (6-bit range in int8 storage, ~25% artifact savings)
  2. Zstd compression (level 22, better ratio than zlib)
  3. Sliding window evaluation (stride-based, more context per token)
  4. Muon weight decay
  5. Bigger model via env-var defaults (10 layers, 3x MLP)

Usage:
  torchrun --standalone --nproc_per_node=4 experiments/phase1/train_phase1.py
"""
from __future__ import annotations

import importlib.util
import io
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# ENV-VAR DEFAULTS (set before importing baseline so Hyperparameters picks them up)
# ---------------------------------------------------------------------------

_PHASE1_DEFAULTS = {
    "NUM_LAYERS": "10",
    "MLP_MULT": "3",
    "MUON_MOMENTUM": "0.99",
    # Sliding window eval settings
    "SLIDING_STRIDE": "64",
    # Quantization bits (6 = int6 range stored as int8)
    "QUANT_BITS": "6",
    # Compression method: zstd or zlib
    "COMPRESS_METHOD": "zstd",
}

for key, default in _PHASE1_DEFAULTS.items():
    os.environ.setdefault(key, default)


# ---------------------------------------------------------------------------
# IMPORT BASELINE MODULE (without executing main)
# ---------------------------------------------------------------------------

def _import_baseline():
    """Import train_gpt.py as a module without running main()."""
    baseline_path = str(Path(__file__).resolve().parent.parent.parent / "train_gpt.py")
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    spec = importlib.util.spec_from_file_location("train_gpt_base", baseline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline from {baseline_path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = "train_gpt_base"  # Prevent __main__ guard from firing
    sys.modules["train_gpt_base"] = mod
    spec.loader.exec_module(mod)
    return mod


base = _import_baseline()


# ---------------------------------------------------------------------------
# PATCH 1: Int6 Quantization
# ---------------------------------------------------------------------------
# Quantize to [-31, 31] instead of [-127, 127]. Stored as int8 tensors
# but the restricted range compresses much better under zstd/zlib, saving
# ~25% artifact bytes. This headroom lets us fit a bigger model (10L, 3x MLP).

QUANT_BITS = int(os.environ.get("QUANT_BITS", "6"))
if QUANT_BITS == 6:
    QUANT_MAX = 31
elif QUANT_BITS == 5:
    QUANT_MAX = 15
else:
    QUANT_MAX = 127  # standard int8

_original_quantize_float_tensor = base.quantize_float_tensor


def quantize_float_tensor_intN(t: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row N-bit symmetric quantization (stored as int8)."""
    qmax = QUANT_MAX
    t32 = t.float()
    clip_q = base.INT8_CLIP_Q

    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), clip_q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / qmax).clamp_min(1.0 / qmax)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax).to(torch.int8).contiguous()
        return q, scale.to(dtype=base.INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / qmax if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(
        torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qmax, qmax
    ).to(torch.int8).contiguous()
    return q, scale


if QUANT_BITS != 8:
    base.quantize_float_tensor = quantize_float_tensor_intN
    print(f"[phase1] int{QUANT_BITS} quantization enabled (range [-{QUANT_MAX}, {QUANT_MAX}])")


# ---------------------------------------------------------------------------
# PATCH 2: Zstd Compression
# ---------------------------------------------------------------------------
# Replace zlib with zstandard at level 22 for better compression ratio.
# The baseline uses zlib.compress/decompress in main(). We replace the module-
# level zlib reference so those calls route through zstd transparently.

COMPRESS_METHOD = os.environ.get("COMPRESS_METHOD", "zstd")

if COMPRESS_METHOD == "zstd":
    try:
        import zstandard
        import zlib as _real_zlib

        class _ZstdCompat:
            """Drop-in replacement for zlib that uses zstandard."""

            def __init__(self, level: int = 22):
                self._level = level

            def compress(self, data, level=None):
                lvl = level if level is not None else self._level
                return zstandard.ZstdCompressor(level=min(lvl, 22)).compress(data)

            def decompress(self, data):
                return zstandard.ZstdDecompressor().decompress(data)

            def __getattr__(self, name):
                return getattr(_real_zlib, name)

        base.zlib = _ZstdCompat()
        print(f"[phase1] zstd compression enabled (level 22)")
    except ImportError:
        print("[phase1] WARNING: zstandard not installed, falling back to zlib", file=sys.stderr)
        print("[phase1]   pip install zstandard", file=sys.stderr)


# ---------------------------------------------------------------------------
# PATCH 3: Sliding Window Evaluation
# ---------------------------------------------------------------------------
# Standard eval splits validation into non-overlapping sequences. Tokens near
# sequence boundaries have little context. Sliding window uses overlapping
# windows so every scored token has at least (seq_len - stride) tokens of
# context, improving BPB by ~0.03-0.05.

SLIDING_STRIDE = int(os.environ.get("SLIDING_STRIDE", "64"))
_original_eval_val = base.eval_val


def eval_val_sliding(
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
) -> tuple[float, float]:
    """Sliding window evaluation with configurable stride."""
    stride = SLIDING_STRIDE
    seq_len = args.train_seq_len

    if stride <= 0 or stride >= seq_len:
        # Disabled — fall back to standard eval
        return _original_eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )

    total_tokens = val_tokens.numel()

    # Build list of (window_offset, score_start_within_window, score_end_within_window)
    # First window scores all positions; subsequent windows score only the last `stride`.
    windows: list[tuple[int, int, int]] = []
    off = 0
    while off + seq_len < total_tokens:
        if off == 0:
            windows.append((off, 0, seq_len))
        else:
            windows.append((off, seq_len - stride, seq_len))
        off += stride

    # Distribute windows across ranks (contiguous partitioning)
    n = len(windows)
    win_start = (n * rank) // world_size
    win_end = (n * (rank + 1)) // world_size
    my_windows = windows[win_start:win_end]

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Batch size for sliding window (number of windows per forward pass)
    sw_batch = max(1, args.val_batch_size // (seq_len * world_size * grad_accum_steps))

    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(my_windows), sw_batch):
            batch_windows = my_windows[batch_start:batch_start + sw_batch]
            bsz = len(batch_windows)

            # Build batch tensors
            x_list, y_list = [], []
            for w_off, _, _ in batch_windows:
                x_list.append(val_tokens[w_off:w_off + seq_len])
                y_list.append(val_tokens[w_off + 1:w_off + seq_len + 1])
            x = torch.stack(x_list).to(device=device, dtype=torch.int64, non_blocking=True)
            y = torch.stack(y_list).to(device=device, dtype=torch.int64, non_blocking=True)

            # Capture per-token logits via F.cross_entropy interception
            captured = {}
            original_ce = F.cross_entropy

            def _intercept_ce(logits, targets, reduction="mean", **kwargs):
                per_tok = original_ce(logits, targets, reduction="none", **kwargs)
                captured["per_token_ce"] = per_tok.detach()
                return per_tok.mean() if reduction == "mean" else (
                    per_tok.sum() if reduction == "sum" else per_tok
                )

            F.cross_entropy = _intercept_ce
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    model(x, y)
            finally:
                F.cross_entropy = original_ce

            # per_token_ce shape: [bsz * seq_len]
            per_token_ce = captured["per_token_ce"].reshape(bsz, seq_len)
            prev_all = x  # [bsz, seq_len]
            tgt_all = y   # [bsz, seq_len]

            for i, (_, s_start, s_end) in enumerate(batch_windows):
                scored_ce = per_token_ce[i, s_start:s_end]
                scored_tgt = tgt_all[i, s_start:s_end]
                scored_prev = prev_all[i, s_start:s_end]

                val_loss_sum += scored_ce.to(torch.float64).sum()
                val_token_count += float(scored_ce.numel())

                token_bytes = base_bytes_lut[scored_tgt].to(dtype=torch.int16)
                token_bytes += (
                    has_leading_space_lut[scored_tgt] & ~is_boundary_token_lut[scored_prev]
                ).to(dtype=torch.int16)
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


if SLIDING_STRIDE > 0:
    base.eval_val = eval_val_sliding
    print(f"[phase1] sliding window eval enabled (stride={SLIDING_STRIDE})")


# ---------------------------------------------------------------------------
# PATCH 4: Muon Weight Decay
# ---------------------------------------------------------------------------
# Add weight decay to the Muon optimizer step. Top leaderboard entries use
# Muon WD ~0.04. The baseline Muon has no weight decay.

MUON_WD = float(os.environ.get("MUON_WD", "0.04"))

if MUON_WD > 0:
    _original_muon_step = base.Muon.step

    @torch.no_grad()
    def _muon_step_with_wd(self, closure=None):
        # Apply weight decay before the standard Muon step
        for group in self.param_groups:
            wd = group.get("weight_decay", MUON_WD)
            if wd > 0:
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is not None and p.ndim >= 2:
                        p.data.mul_(1.0 - lr * wd)
        return _original_muon_step(self, closure)

    base.Muon.step = _muon_step_with_wd
    print(f"[phase1] Muon weight decay enabled (WD={MUON_WD})")


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[phase1] config: QUANT_BITS={QUANT_BITS} COMPRESS={COMPRESS_METHOD} "
          f"STRIDE={SLIDING_STRIDE} MUON_WD={MUON_WD} "
          f"NUM_LAYERS={os.environ.get('NUM_LAYERS')} MLP_MULT={os.environ.get('MLP_MULT')}")
    base.main()
