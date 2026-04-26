"""GoldenShift DSV Builder — Forward-Only Histogram + Matmul.

Keeps only what is needed for the pure GoldenShift DSV forward build:

  EigenTrainer
    .from_codebook_uint64()          -- construct from uint64 codebook
    .build_fwd_from_tokens()         -- forward-only histogram scan + matmul

Everything else (bilateral solver, eigensolver, PMI centering, backward
accumulation, AxisWeightScheduler, AnticipationEigenGate, HadamardEigenSolver,
SoftEMABundle, FullTeleportResult, FullTeleportStep, EigenSpiralBuilder,
absorb_bigrams, absorb_bigrams_chunked) has been removed.

Pipeline:
  1. Compute freq_table[tok] = count(tok)  [O(N), one pass, in _semantic_layer.py]
  2. Chunked histogram scan (distributed):
       fwd_hist[a, (c-1)*V + b] += 1/freq[b]   for each (a, b, lag c)
     All-reduce across ranks (NCCL).
  3. Single matmul per lag (rank 0 only):
       For lag c:
         result_c = fwd_hist_block_c @ CB_pm1        (V,V) @ (V,n_bits)
         sem_fwd_spectrum += roll_cols(result_c, offset_c)
     sem_fwd_pm1 = sign(sem_fwd_spectrum)
  4. Pack to uint64.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# -- GPU acceleration (optional, graceful fallback to CPU) --------------------
try:
    from _gpu import (
        gpu_available,
        gpu_matmul_f16,
        gpu_matmul_f16_dual,
        gpu_bincount_weighted,
        gpu_sign_f32,
        gpu_uint64_batch_to_pm1,
    )
    _GPU_AVAILABLE = gpu_available()
except ImportError:
    _GPU_AVAILABLE = False
    def gpu_matmul_f16(a, b): return a.astype('float32') @ b.astype('float32')
    def gpu_matmul_f16_dual(a1, a2, b):
        return (a1.astype('float32') @ b.astype('float32'),
                a2.astype('float32') @ b.astype('float32'))
    def gpu_bincount_weighted(idx, w, ml):
        if w is None:
            return np.bincount(idx.astype(np.int64), minlength=ml).astype(np.float64)
        return np.bincount(idx.astype(np.int64), weights=w.astype(np.float64), minlength=ml)
    def gpu_sign_f32(a):
        out = np.sign(a).astype(np.float32); out[out == 0.0] = 1.0; return out
    def gpu_uint64_batch_to_pm1(hvs):
        N, W = hvs.shape
        bits = np.unpackbits(hvs.view(np.uint8).reshape(N, W * 8), axis=1, bitorder='little')
        return bits.astype(np.float32) * 2.0 - 1.0

# -----------------------------------------------------------------------------
# pm1 <-> uint64 conversion helpers
# -----------------------------------------------------------------------------

def uint64_to_pm1(hv: np.ndarray) -> np.ndarray:
    """(W,) uint64 -> (W*64,) float32 in {-1, +1}."""
    bits = np.unpackbits(hv.view(np.uint8), bitorder='little')
    return (bits.astype(np.float32) * 2.0 - 1.0)


def pm1_to_uint64(pm1: np.ndarray) -> np.ndarray:
    """(W*64,) float32 -> (W,) uint64."""
    bits = (pm1 >= 0.0).astype(np.uint8)
    return np.packbits(bits, bitorder='little').view(np.uint64)


def batch_uint64_to_pm1(hvs: np.ndarray) -> np.ndarray:
    """(N, W) uint64 -> (N, W*64) float32 in {-1, +1}."""
    if _GPU_AVAILABLE:
        return gpu_uint64_batch_to_pm1(hvs)
    N, W = hvs.shape
    bits = np.unpackbits(hvs.view(np.uint8).reshape(N, W * 8), axis=1, bitorder='little')
    return bits.astype(np.float32) * 2.0 - 1.0


def batch_pm1_to_uint64(pm1s: np.ndarray) -> np.ndarray:
    """(N, W*64) float32 -> (N, W) uint64."""
    N, n_bits = pm1s.shape
    W = n_bits // 64
    bits = (pm1s >= 0.0).astype(np.uint8)
    return np.packbits(bits, axis=1, bitorder='little').view(np.uint64).reshape(N, W)


# -----------------------------------------------------------------------------
# EigenTrainer — forward-only GoldenShift DSV build
# -----------------------------------------------------------------------------

class EigenTrainer:
    """Forward-only GoldenShift DSV builder.

    Builds sem_fwd via a single-pass chunked histogram scan followed by
    C separate matmuls on the unrotated codebook with cheap output-column
    circular shifts (one per lag).

    Only sem_fwd is built. No backward accumulation. No PMI centering.
    No eigensolver. No bilateral build.

    Args:
        codebook_pm1  : (vocab_size, n_bits) float32 -- CB in pm1 space
    """

    def __init__(self, codebook_pm1: np.ndarray):
        self.CB_pm1     = codebook_pm1.astype(np.float32)
        self.vocab_size = codebook_pm1.shape[0]
        self.n_bits     = codebook_pm1.shape[1]

    @classmethod
    def from_codebook_uint64(
        cls,
        codebook_vecs: np.ndarray,   # (vocab_size, n_words) uint64
        **_kwargs,                   # absorb unused kwargs (goal_threshold etc.)
    ) -> "EigenTrainer":
        """Construct from uint64 codebook (converts to pm1 internally).

        Args:
            codebook_vecs : (vocab_size, n_words) uint64 -- raw codebook

        Returns:
            EigenTrainer instance
        """
        vocab_size, n_words = codebook_vecs.shape
        bits = np.unpackbits(
            codebook_vecs.view(np.uint8).reshape(vocab_size, n_words * 8),
            axis=1, bitorder='little'
        )
        cb_pm1 = bits.astype(np.float32) * 2.0 - 1.0   # (vocab_size, n_bits)
        obj = cls(cb_pm1)
        obj._CB_uint64 = codebook_vecs.astype(np.uint64)
        return obj

    def build_bilateral_from_tokens(
        self,
        tokens           : np.ndarray,
        ctx_len          : int   = 4,
        axis_word_shifts : Optional[list] = None,
        chunk_size       : int   = 2_000_000,
        verbose          : bool  = True,
        time_budget_s    : float = 600.0,
        dist_rank        : int   = 0,
        dist_world_size  : int   = 1,
        freq_weights     : Optional[np.ndarray] = None,
    ) -> dict:
        """Build forward DSV table via chunked histogram scan + matmul.

        Forward-only: only sem_fwd is built. No backward accumulation.
        No PMI centering. No eigensolver.

        Algorithm:
          1. Chunked histogram scan (distributed):
               fwd_hist[a, (c-1)*V + b] += weight   for each (a, b, lag c)
             where weight = 1/freq[b] if freq_weights provided, else 1.
             All-reduce across ranks (NCCL).
          2. C separate matmuls on unrotated CB_pm1 (rank 0 only):
               For lag c:
                 result_c = fwd_hist_block_c @ CB_pm1   (V,V) @ (V,n_bits)
                 sem_fwd_spectrum += roll_cols(result_c, offset_c)
             sem_fwd_pm1 = sign(sem_fwd_spectrum)
          3. Pack to uint64.

        The circular output-column rotation replaces the old 256 MB
        CB_composite_pm1 materialization with a cheap np.roll on the
        (V, n_bits) output — same mathematical result, 4x less peak RAM.

        Args:
            tokens           : (N,) token sequence
            ctx_len          : Number of lags (context depth)
            axis_word_shifts : List of (word_shift, bit_shift) per lag.
                               Pass GOLDEN_AXES._word_shifts/._bit_shifts[1..ctx_len].
                               If None: no rotation (all lags use same CB).
            chunk_size       : Positions per outer loop iteration (default 2M)
            verbose          : Print progress
            time_budget_s    : Stop early if exceeded
            dist_rank        : This rank's index in distributed group
            dist_world_size  : Total number of ranks
            freq_weights     : (V,) float32 -- token frequencies (raw counts).
                               When provided, accumulates 1/freq[b] per (a->b) pair.
                               When None: uniform weight=1 (backward compatible).

        Returns dict:
            sem_fwd_pm1  : (vocab_size, n_bits) float32  [rank 0 only, else None]
            sem_fwd_u64  : (vocab_size, n_words) uint64  [rank 0 only, else None]
            sem_bwd_pm1  : None  (not built)
            sem_bwd_u64  : None  (not built)
            total_pairs  : int
        """
        import time
        t0 = time.time()

        V  = self.vocab_size
        D  = self.n_bits
        C  = ctx_len
        N  = len(tokens)
        N_valid = max(0, N - C)
        n_words = D // 64

        tokens_i32 = np.clip(tokens.astype(np.int32), 0, V - 1)

        # -- Precompute inverse-frequency lookup table -------------------------
        if freq_weights is not None:
            _fw = np.asarray(freq_weights, dtype=np.float32)
            _fw = np.maximum(_fw, 1.0)
            inv_freq = (1.0 / _fw).astype(np.float32)   # (V,) float32
            _use_freq = True
            if verbose and dist_rank == 0:
                print(f"[GoldenShiftDSV] 1/freq weighting: "
                      f"min_freq={float(_fw.min()):.0f}, "
                      f"max_inv_freq={float(inv_freq.max()):.5f}")
        else:
            inv_freq = None
            _use_freq = False

        # -- Precompute per-lag bit offsets for output column rotation ---------
        # GoldenAxisShift(CB, lag=c) = roll_bits(CB, offset_c).
        # Equivalent: result_c = fwd_block_c @ CB_pm1, then roll output cols by +offset_c.
        # Cost: np.roll on (V, n_bits) float32 per lag -- O(V*n_bits), fast.
        if dist_rank == 0:
            if axis_word_shifts is not None and len(axis_word_shifts) >= C:
                lag_bit_offsets = [
                    (ws * 64 + bs) % D
                    for ws, bs in axis_word_shifts[:C]
                ]
                if verbose:
                    print(f"[GoldenShiftDSV] Lag bit offsets: {lag_bit_offsets}")
            else:
                lag_bit_offsets = [0] * C
                if verbose:
                    print(f"[GoldenShiftDSV] No axis shifts -- uniform CB for all lags")
        else:
            lag_bit_offsets = None

        # -- Step 1: Chunked forward histogram scan (distributed) -------------
        import os as _os
        try:
            import torch as _torch
            import torch.distributed as _td
            _lr  = int(_os.environ.get('LOCAL_RANK', dist_rank))
            _dev = _torch.device(f'cuda:{_lr}') if _torch.cuda.is_available() else _torch.device('cpu')
            _use_gpu = True
        except ImportError:
            _torch = None
            _td    = None
            _dev   = None
            _use_gpu = False

        if _use_gpu:
            fwd_acc = _torch.zeros(V * C * V, dtype=_torch.float32, device=_dev)
        else:
            fwd_hist_flat = np.zeros(V * C * V, dtype=np.float32)

        total_pairs = 0
        c_offsets   = np.arange(C, dtype=np.int64) * V   # (C,)

        shard_start = dist_rank * N_valid // dist_world_size
        shard_end   = min((dist_rank + 1) * N_valid // dist_world_size, N_valid)

        for start in range(shard_start, shard_end, chunk_size):
            if time.time() - t0 > time_budget_s * 0.90:
                if verbose:
                    print(f"[GoldenShiftDSV] Time budget reached at pos {start:,}")
                break

            end       = min(start + chunk_size, N_valid)
            chunk_len = end - start

            a_chunk = tokens_i32[start:end].astype(np.int64)   # (chunk,)

            b_chunk = np.lib.stride_tricks.sliding_window_view(
                tokens_i32[start : end + C], C + 1
            )[:chunk_len, 1:].astype(np.int64)   # (chunk, C)

            # fwd: row=a, col=(c-1)*V + b  ->  flat = a*(C*V) + col
            fwd_idx = (
                a_chunk[:, None] * (C * V) + c_offsets[None, :] + b_chunk
            ).ravel()   # (chunk*C,)

            if _use_gpu:
                fwd_idx_t = _torch.as_tensor(fwd_idx, dtype=_torch.int64, device=_dev)
                if _use_freq:
                    b_flat = b_chunk.ravel().astype(np.int32)
                    fwd_w  = inv_freq[b_flat]
                    fwd_w_t = _torch.as_tensor(fwd_w, dtype=_torch.float32, device=_dev)
                    fwd_acc.scatter_add_(0, fwd_idx_t, fwd_w_t)
                    del fwd_w_t, b_flat
                else:
                    ones_t = _torch.ones(len(fwd_idx), dtype=_torch.float32, device=_dev)
                    fwd_acc.scatter_add_(0, fwd_idx_t, ones_t)
                    del ones_t
                del fwd_idx_t
            else:
                if _use_freq:
                    b_flat = b_chunk.ravel().astype(np.int32)
                    fwd_w  = inv_freq[b_flat]
                    fwd_hist_flat += gpu_bincount_weighted(fwd_idx, fwd_w, V * C * V).astype(np.float32)
                else:
                    fwd_hist_flat += gpu_bincount_weighted(fwd_idx, None, V * C * V).astype(np.float32)

            total_pairs += chunk_len * C

            if verbose:
                elapsed = time.time() - t0
                pct     = 100.0 * end / shard_end
                print(f"[GoldenShiftDSV] {end:,}/{shard_end:,} ({pct:.1f}%) "
                      f"pairs={total_pairs:,}  elapsed={elapsed:.2f}s")

        # -- Distributed all-reduce: sum histograms across all ranks ----------
        if dist_world_size > 1:
            try:
                if _use_gpu and _td is not None and _td.is_available() and _td.is_initialized():
                    _td.all_reduce(fwd_acc, op=_td.ReduceOp.SUM)
                    t_p = _torch.tensor([total_pairs], dtype=_torch.int64, device=_dev)
                    _td.all_reduce(t_p, op=_td.ReduceOp.SUM)
                    total_pairs = int(t_p.item())
                    if verbose and dist_rank == 0:
                        print(f"[GoldenShiftDSV] All-reduce complete "
                              f"(world_size={dist_world_size}, total_pairs={total_pairs:,})")
                elif not _use_gpu and _td is not None and _td.is_available() and _td.is_initialized():
                    _fwd_t = _torch.from_numpy(fwd_hist_flat).to(_dev)
                    _td.all_reduce(_fwd_t, op=_td.ReduceOp.SUM)
                    fwd_hist_flat = _fwd_t.cpu().numpy()
                    del _fwd_t
                    t_p = _torch.tensor([total_pairs], dtype=_torch.int64, device=_dev)
                    _td.all_reduce(t_p, op=_td.ReduceOp.SUM)
                    total_pairs = int(t_p.item())
                    if verbose and dist_rank == 0:
                        print(f"[GoldenShiftDSV] All-reduce complete "
                              f"(world_size={dist_world_size}, total_pairs={total_pairs:,})")
            except Exception as _e:
                if verbose and dist_rank == 0:
                    print(f"[GoldenShiftDSV] All-reduce skipped ({_e})")

        # Materialise numpy array from GPU accumulator (D2H once)
        if _use_gpu:
            fwd_hist_flat = fwd_acc.cpu().numpy()
            del fwd_acc

        # Non-main ranks: histogram contributed -- matmul not needed
        if dist_rank > 0:
            return dict(sem_fwd_pm1=None, sem_fwd_u64=None,
                        sem_bwd_pm1=None, sem_bwd_u64=None,
                        total_pairs=total_pairs)

        # -- Step 2: C separate matmuls + output column rotation (rank 0) -----
        fwd_hist_2d = fwd_hist_flat.reshape(V, C * V)   # (V, C*V) float32

        if verbose:
            elapsed = time.time() - t0
            print(f"[GoldenShiftDSV] Computing sem_fwd: {C} x (V={V},V) @ (V,n_bits={D}) "
                  f"+ circular output shift per lag... elapsed={elapsed:.2f}s")

        t_mm = time.time()
        sem_fwd_spectrum = np.zeros((V, D), dtype=np.float32)

        for c_idx in range(C):
            fwd_block = fwd_hist_2d[:, c_idx * V : (c_idx + 1) * V]   # (V, V)

            # Matmul with unrotated CB_pm1: (V, V) @ (V, n_bits) -> (V, n_bits)
            result = gpu_matmul_f16(fwd_block, self.CB_pm1)   # (V, n_bits)

            # Circular output-column rotation for this lag's GoldenAxisShift
            bit_offset = lag_bit_offsets[c_idx]
            if bit_offset != 0:
                result = np.roll(result, bit_offset, axis=1)

            sem_fwd_spectrum += result

        sem_fwd_pm1 = gpu_sign_f32(sem_fwd_spectrum)   # (V, n_bits) float32 {-1,+1}
        sem_fwd_u64 = batch_pm1_to_uint64(sem_fwd_pm1)  # (V, n_words) uint64

        elapsed = time.time() - t0
        mm_t    = time.time() - t_mm
        if verbose:
            print(f"[GoldenShiftDSV] Matmul done in {mm_t:.2f}s")
            print(f"[GoldenShiftDSV] Total: {total_pairs:,} pairs in {elapsed:.2f}s")

        return dict(
            sem_fwd_pm1  = sem_fwd_pm1,
            sem_fwd_u64  = sem_fwd_u64,
            sem_bwd_pm1  = sem_fwd_pm1,   # alias: sem_bwd = sem_fwd for API compat
            sem_bwd_u64  = sem_fwd_u64,   # alias: sem_bwd = sem_fwd for API compat
            total_pairs  = total_pairs,
        )
