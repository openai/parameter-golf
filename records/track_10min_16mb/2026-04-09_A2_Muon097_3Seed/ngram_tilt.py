"""N-gram tilt eval-time helper.

Wraps the C++ ContextMixer kernel from PR #1420 (legality argument in
issue #1017) via ctypes. Builds the open-addressing hash tables on rank 0,
broadcasts hints/betas to other ranks, and provides a torch helper that
applies the one-token exponential tilt to per-position NLL.

Math:
  p_tilt(t) = p_model(t) * exp(beta * 1[t==hint]) / Z
  Z = 1 + p_hint * (exp(beta) - 1)
  -log p_tilt(target) = nll + has_hint * (log(Z) - beta * 1[tgt==hint])

Score-before-update is enforced inside the C kernel: hint for position p
is read from the prefix-only hash tables BEFORE the kernel updates them
with token at position p.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


_HERE = Path(__file__).resolve().parent
# Look in ./ngram/ subdir first (dev layout), then current dir (submission layout)
if (_HERE / "ngram" / "fused_expert_kernel.cpp").exists():
    _NGRAM_DIR = _HERE / "ngram"
else:
    _NGRAM_DIR = _HERE
_LIB_PATH = _NGRAM_DIR / "libfused_ngram.so"
_SRC_PATH = _NGRAM_DIR / "fused_expert_kernel.cpp"

_lib = None


def _ensure_lib():
    global _lib
    if _lib is not None:
        return _lib
    if (not _LIB_PATH.exists()) or (
        _SRC_PATH.exists() and _SRC_PATH.stat().st_mtime_ns > _LIB_PATH.stat().st_mtime_ns
    ):
        subprocess.run(
            [
                "g++", "-O3", "-march=native", "-std=c++17",
                "-fPIC", "-shared",
                str(_SRC_PATH),
                "-o", str(_LIB_PATH),
            ],
            check=True,
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.ctxmixer_new.restype = ctypes.c_void_p
    lib.ctxmixer_new.argtypes = [
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_double, ctypes.c_int,
    ]
    lib.ctxmixer_delete.restype = None
    lib.ctxmixer_delete.argtypes = [ctypes.c_void_p]
    lib.ctxmixer_set_tokens.restype = None
    lib.ctxmixer_set_tokens.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
    ]
    lib.ctxmixer_set_luts.restype = None
    lib.ctxmixer_set_luts.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.ctxmixer_reset.restype = None
    lib.ctxmixer_reset.argtypes = [ctypes.c_void_p]
    lib.ctxmixer_get_hints_batch.restype = None
    lib.ctxmixer_get_hints_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double),
    ]
    _lib = lib
    return _lib


class NgramTiltState:
    """Owns the precomputed hints/betas for the entire validation stream.

    Construction is collective: all ranks call build_hints() but only
    rank 0 actually runs the C++ kernel; other ranks receive the hints
    via broadcast.
    """

    def __init__(
        self,
        val_tokens: torch.Tensor,
        has_leading_space_lut: torch.Tensor,
        is_boundary_token_lut: torch.Tensor,
        rank: int,
        world_size: int,
        device: torch.device,
        base_beta: float = 2.0,
        agree_bonus: float = 0.1,
        within_threshold: float = 0.25,
        within_beta: float = 0.92,
        word_threshold: float = 0.80,
        word_beta: float = 0.50,
        open_table_bits: int = 26,
        token_threshold_scale: float = 1.0,
        order_stride: int = 2,
        log=print,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device

        n_tok = val_tokens.numel()
        # Hints[i] = hint for position i (the token at val_tokens[i] given
        # the prefix val_tokens[:i]). Position 0 has no prefix => hint -1.
        # We compute for positions 1..n_tok-1.
        self.hints_cpu = torch.full((n_tok,), -1, dtype=torch.int32)
        self.betas_cpu = torch.zeros((n_tok,), dtype=torch.float64)

        if rank == 0:
            t0 = time.perf_counter()
            lib = _ensure_lib()
            tokens_np = val_tokens.cpu().numpy().astype(np.int64, copy=False)
            tokens_np = np.ascontiguousarray(tokens_np)
            # base_bytes is unused by the kernel hints (only LUTs that
            # determine word boundaries matter), but the API expects it.
            base_bytes_np = np.zeros(has_leading_space_lut.numel(), dtype=np.int16)
            ls_np = has_leading_space_lut.cpu().numpy().astype(np.uint8, copy=False)
            bd_np = is_boundary_token_lut.cpu().numpy().astype(np.uint8, copy=False)
            base_bytes_np = np.ascontiguousarray(base_bytes_np)
            ls_np = np.ascontiguousarray(ls_np)
            bd_np = np.ascontiguousarray(bd_np)

            mixer = lib.ctxmixer_new(
                base_beta, agree_bonus,
                within_threshold, within_beta,
                word_threshold, word_beta,
                open_table_bits, token_threshold_scale, order_stride,
            )
            if not mixer:
                raise RuntimeError("ctxmixer_new returned NULL")
            try:
                lib.ctxmixer_set_tokens(
                    mixer,
                    tokens_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(int(n_tok)),
                )
                lib.ctxmixer_set_luts(
                    mixer,
                    base_bytes_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                    ls_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    bd_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                )
                positions = np.arange(1, n_tok, dtype=np.int64)
                positions = np.ascontiguousarray(positions)
                out_hints = np.full(n_tok - 1, -1, dtype=np.int32)
                out_betas = np.zeros(n_tok - 1, dtype=np.float64)
                lib.ctxmixer_get_hints_batch(
                    mixer,
                    positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int(int(n_tok - 1)),
                    out_hints.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    out_betas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
            finally:
                lib.ctxmixer_delete(mixer)
            self.hints_cpu[1:] = torch.from_numpy(out_hints)
            self.betas_cpu[1:] = torch.from_numpy(out_betas)
            elapsed = time.perf_counter() - t0
            n_hits = int((out_hints >= 0).sum())
            log(
                f"ngram_tilt:precompute n_tok={n_tok} hints={n_hits} "
                f"({100*n_hits/(n_tok-1):.2f}%) elapsed={elapsed:.1f}s "
                f"base_beta={base_beta} within_beta={within_beta} agree_bonus={agree_bonus}"
            )

        # Move to device, broadcast from rank 0
        self.hints = self.hints_cpu.to(device=device, dtype=torch.int64)
        self.betas = self.betas_cpu.to(device=device, dtype=torch.float64)
        if world_size > 1:
            dist.broadcast(self.hints, src=0)
            dist.broadcast(self.betas, src=0)

    def tilt_nll(
        self,
        scored_nll: torch.Tensor,        # [N] float64, per-position NLL from base softmax
        scored_logits: torch.Tensor,     # [N, V] float, logits at scored positions
        target_ids: torch.Tensor,        # [N] int64, true target tokens
        global_positions: torch.Tensor,  # [N] int64, position index into the val stream
    ) -> torch.Tensor:
        """Apply n-gram tilt to per-position NLL.

        Returns mixed_nll [N] float64. When hint is -1 the tilt is a no-op.
        """
        hints = self.hints[global_positions]
        betas = self.betas[global_positions]
        has_hint = (hints >= 0).to(torch.float64)

        # Recover logsumexp from nll: nll = lse - logit_tgt  =>  lse = nll + logit_tgt
        logit_tgt = scored_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).to(torch.float64)
        safe_h = hints.clamp(min=0)
        logit_hint = scored_logits.gather(-1, safe_h.unsqueeze(-1)).squeeze(-1).to(torch.float64)
        lse = scored_nll + logit_tgt
        p_hint = (logit_hint - lse).exp().clamp(0.0, 1.0)
        Z = 1.0 + p_hint * (betas.exp() - 1.0)
        is_hit = (target_ids == hints).to(torch.float64)
        mixed_nll = scored_nll + has_hint * (Z.log() - betas * is_hit)
        return mixed_nll
