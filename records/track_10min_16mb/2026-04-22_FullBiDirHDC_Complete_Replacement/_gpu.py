"""GPU acceleration helpers for FullBiDirHDC.

Provides a thin torch-based GPU layer that is used by all hot-path functions
in _eigen_convergence.py, _spiral_dsv_lm.py, _bidi_hdc_engine.py, and
_bidi_train.py.

Design principles
─────────────────
1. **Graceful fallback** — every function works on CPU if CUDA is unavailable
   or if torch is not installed.
2. **Zero-copy where possible** — numpy arrays are wrapped with
   torch.as_tensor() (shares memory on CPU, copies to GPU only when needed).
3. **Local-rank aware** — reads LOCAL_RANK env var so each torchrun process
   uses its own GPU (rank 0 → cuda:0, rank 1 → cuda:1, …).
4. **Lazy init** — GPU device is resolved once on first use via _get_device().

Accelerated operations
──────────────────────
• matmul (float32 SGEMM / float16 HGEMM)  — EigenTrainer, EigenSpiralBuilder,
  HadamardEigenSolver, vote_scores_all_vocab
• bincount with weights                    — build_bigram_freq, absorb_bigrams
• sign / unpackbits equivalent             — batch_teleport, pm1 conversions
• softmax / log                            — bidi_bpb
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Torch availability
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None          # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Device resolution
# ─────────────────────────────────────────────────────────────────────────────

_DEVICE: Optional["torch.device"] = None   # type: ignore[name-defined]


def _get_device() -> "torch.device":       # type: ignore[name-defined]
    """Return the CUDA device for this process (or CPU if unavailable).

    Reads LOCAL_RANK to select the correct GPU in a torchrun multi-GPU setup.
    Result is cached after first call.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    if not _TORCH_AVAILABLE:
        import types
        # Return a dummy object that compares equal to "cpu"
        _DEVICE = "cpu"   # type: ignore[assignment]
        return _DEVICE    # type: ignore[return-value]
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        _DEVICE = torch.device(f"cuda:{local_rank}")
    else:
        _DEVICE = torch.device("cpu")
    return _DEVICE


def gpu_available() -> bool:
    """Return True if a CUDA GPU is available and torch is installed."""
    if not _TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


# ─────────────────────────────────────────────────────────────────────────────
# numpy ↔ torch helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_gpu_f32(arr: np.ndarray) -> "torch.Tensor":   # type: ignore[name-defined]
    """Copy a numpy float32 array to the GPU as a float32 tensor."""
    device = _get_device()
    return torch.as_tensor(arr.astype(np.float32, copy=False), device=device)


def to_gpu_f16(arr: np.ndarray) -> "torch.Tensor":   # type: ignore[name-defined]
    """Copy a numpy float32 array to the GPU as a float16 tensor (for HGEMM)."""
    device = _get_device()
    return torch.as_tensor(arr.astype(np.float32, copy=False), device=device).half()


def to_cpu_f32(t: "torch.Tensor") -> np.ndarray:     # type: ignore[name-defined]
    """Copy a GPU tensor back to a numpy float32 array."""
    return t.float().cpu().numpy()


def to_gpu_i64(arr: np.ndarray) -> "torch.Tensor":   # type: ignore[name-defined]
    """Copy a numpy int array to the GPU as int64."""
    device = _get_device()
    return torch.as_tensor(arr.astype(np.int64, copy=False), device=device)


# ─────────────────────────────────────────────────────────────────────────────
# GPU matmul
# ─────────────────────────────────────────────────────────────────────────────

def gpu_matmul_f32(
    a: np.ndarray,   # (..., K) float32
    b: np.ndarray,   # (K, M) float32
) -> np.ndarray:
    """Compute a @ b on GPU (float32 cuBLAS SGEMM), return numpy float32.

    Falls back to numpy if GPU unavailable.
    """
    if not gpu_available():
        return (a.astype(np.float32) @ b.astype(np.float32))
    a_t = to_gpu_f32(a)
    b_t = to_gpu_f32(b)
    out = torch.mm(a_t.reshape(-1, b_t.shape[0]), b_t)
    if a.ndim > 2:
        out = out.reshape(*a.shape[:-1], b.shape[-1])
    return to_cpu_f32(out)


def gpu_matmul_f16(
    a: np.ndarray,   # (..., K) float32 — will be cast to f16 on GPU
    b: np.ndarray,   # (K, M) float32 — will be cast to f16 on GPU
) -> np.ndarray:
    """Compute a @ b on GPU using float16 tensor cores (HGEMM), return numpy float32.

    ~2× faster than SGEMM on H100 for large matrices.
    Falls back to float32 GPU matmul if f16 is not supported.
    """
    if not gpu_available():
        return (a.astype(np.float32) @ b.astype(np.float32))
    try:
        a_t = to_gpu_f16(a)
        b_t = to_gpu_f16(b)
        out = torch.mm(a_t.reshape(-1, b_t.shape[0]), b_t).float()
        if a.ndim > 2:
            out = out.reshape(*a.shape[:-1], b.shape[-1])
        return to_cpu_f32(out)
    except Exception:
        return gpu_matmul_f32(a, b)


def gpu_batch_matmul_f32(
    a: np.ndarray,   # (B, K) float32
    b: np.ndarray,   # (K, M) float32
) -> np.ndarray:
    """Compute a @ b on GPU for a 2D batch matrix, return numpy float32.

    Equivalent to a @ b but dispatched to cuBLAS SGEMM.
    """
    if not gpu_available():
        return (a.astype(np.float32) @ b.astype(np.float32))
    a_t = to_gpu_f32(a)
    b_t = to_gpu_f32(b)
    out = torch.mm(a_t, b_t)
    return to_cpu_f32(out)


# ─────────────────────────────────────────────────────────────────────────────
# GPU bincount with weights
# ─────────────────────────────────────────────────────────────────────────────

def gpu_bincount_weighted(
    indices: np.ndarray,          # (N,) int
    weights,                       # (N,) float  OR  None for uniform count
    minlength: int,
    chunk_size: int = 50_000_000,  # 50 M per chunk → ≤600 MB GPU per chunk
) -> np.ndarray:
    """Weighted bincount on GPU, return numpy float64.

    Uses torch.zeros + scatter_add_ which is fully parallelised on CUDA.
    Falls back to numpy.bincount if GPU unavailable.

    Pass weights=None for a pure integer count (avoids allocating a large
    ones-array in the caller — the GPU chunk then creates a ones tensor of
    only chunk_size elements rather than N elements all at once).

    Processes in chunks to avoid GPU OOM for large inputs (e.g. 8 B tokens).
    """
    if not gpu_available():
        if weights is None:
            return np.bincount(
                indices.astype(np.int64), minlength=minlength
            ).astype(np.float64)
        return np.bincount(
            indices.astype(np.int64), weights=weights.astype(np.float64),
            minlength=minlength
        )
    device = _get_device()
    out = torch.zeros(minlength, dtype=torch.float32, device=device)
    N = len(indices)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        idx_chunk = torch.as_tensor(
            indices[start:end].astype(np.int64, copy=False), device=device
        )
        if weights is None:
            w_chunk = torch.ones(end - start, dtype=torch.float32, device=device)
        else:
            w_chunk = torch.as_tensor(
                weights[start:end].astype(np.float32, copy=False), device=device
            )
        out.scatter_add_(0, idx_chunk, w_chunk)
        del idx_chunk, w_chunk
    return out.double().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# GPU sign
# ─────────────────────────────────────────────────────────────────────────────

def gpu_sign_f32(arr: np.ndarray) -> np.ndarray:
    """Compute sign(arr) on GPU, ties → +1, return numpy float32.

    Falls back to numpy if GPU unavailable.
    """
    if not gpu_available():
        out = np.sign(arr).astype(np.float32)
        out[out == 0.0] = 1.0
        return out
    t = to_gpu_f32(arr)
    out = torch.sign(t)
    out[out == 0.0] = 1.0
    return to_cpu_f32(out)


# ─────────────────────────────────────────────────────────────────────────────
# GPU softmax + log
# ─────────────────────────────────────────────────────────────────────────────

def gpu_log_softmax_scores(
    scores: np.ndarray,   # (batch, vocab) float32
) -> np.ndarray:
    """Compute softmax(scores) on GPU, return (batch, vocab) float32 probabilities.

    Falls back to numpy if GPU unavailable.
    """
    if not gpu_available():
        scores = scores.astype(np.float32)
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs
    t = to_gpu_f32(scores)
    probs = torch.softmax(t, dim=1)
    return to_cpu_f32(probs)


# ─────────────────────────────────────────────────────────────────────────────
# GPU unpackbits equivalent (uint64 → pm1 float32)
# ─────────────────────────────────────────────────────────────────────────────

def gpu_uint64_batch_to_pm1(hvs: np.ndarray) -> np.ndarray:
    """(N, W) uint64 → (N, W×64) float32 in {-1, +1} on GPU.

    Uses torch bitwise ops to unpack bits without materialising a uint8 view.
    Falls back to numpy unpackbits if GPU unavailable.
    """
    if not gpu_available():
        N, W = hvs.shape
        bits = np.unpackbits(
            hvs.view(np.uint8).reshape(N, W * 8), axis=1, bitorder='little'
        )
        return bits.astype(np.float32) * 2.0 - 1.0

    device = _get_device()
    N, W = hvs.shape
    n_bits = W * 64

    # Reinterpret as uint8 on CPU (zero-copy view), then send to GPU
    u8 = hvs.view(np.uint8).reshape(N, W * 8)
    u8_t = torch.as_tensor(u8.copy(), device=device)  # (N, W*8) uint8

    # Unpack 8 bits per byte using bit shifts
    # bit_k of byte b = (byte >> k) & 1  for k in 0..7 (little-endian)
    shifts = torch.arange(8, device=device, dtype=torch.uint8)  # (8,)
    # u8_t: (N, W*8), shifts: (8,) → broadcast → (N, W*8, 8)
    bits = ((u8_t.unsqueeze(-1) >> shifts) & 1).reshape(N, n_bits)  # (N, n_bits)
    pm1 = bits.float() * 2.0 - 1.0
    return to_cpu_f32(pm1)


# ─────────────────────────────────────────────────────────────────────────────
# GPU abs-dot for bilateral confidence (vote_scores_all_vocab)
# ─────────────────────────────────────────────────────────────────────────────

def gpu_bilateral_confidence(
    sem_fwd_pm1 : np.ndarray,   # (vocab, n_bits) float32
    sem_bwd_pm1 : np.ndarray,   # (vocab, n_bits) float32
    codebook_pm1: np.ndarray,   # (vocab, n_bits) float32
    prev_tokens : np.ndarray,   # (batch,) int32
) -> np.ndarray:
    """Compute bilateral confidence scores on GPU.

    conf_fwd[b, v] = |sem_fwd_pm1[prev_t[b]] · codebook_pm1[v]| / n_bits
    conf_bwd[b, v] = |sem_bwd_pm1[v] · codebook_pm1[prev_t[b]]| / n_bits
    consistency    = (conf_fwd + conf_bwd) * 0.5

    Returns (batch, vocab) float32 clipped to [1e-30, 0.99].
    Falls back to numpy if GPU unavailable.
    """
    n_bits = codebook_pm1.shape[1]

    if not gpu_available():
        sf = sem_fwd_pm1[prev_tokens]
        conf_fwd = np.abs(sf @ codebook_pm1.T) / n_bits
        cb_prev  = codebook_pm1[prev_tokens]
        conf_bwd = np.abs(sem_bwd_pm1 @ cb_prev.T).T / n_bits
        consistency = (conf_fwd + conf_bwd) * 0.5
        return np.clip(0.5 + 0.49 * consistency, 1e-30, 0.99).astype(np.float32)

    device = _get_device()
    sf_t  = to_gpu_f16(sem_fwd_pm1)          # (vocab, n_bits) f16
    sb_t  = to_gpu_f16(sem_bwd_pm1)          # (vocab, n_bits) f16
    cb_t  = to_gpu_f16(codebook_pm1)         # (vocab, n_bits) f16
    idx_t = to_gpu_i64(prev_tokens)          # (batch,)

    sf_prev = sf_t[idx_t]                    # (batch, n_bits) f16
    conf_fwd = torch.mm(sf_prev, cb_t.T).abs().float() / n_bits   # (batch, vocab)

    cb_prev = cb_t[idx_t]                    # (batch, n_bits) f16
    conf_bwd = torch.mm(sb_t, cb_prev.T).abs().float().T / n_bits  # (batch, vocab)

    consistency = (conf_fwd + conf_bwd) * 0.5
    out = torch.clamp(0.5 + 0.49 * consistency, 1e-30, 0.99)
    return to_cpu_f32(out)


def gpu_bilateral_midpoint_scores(
    sem_fwd_pm1 : np.ndarray,   # (vocab, n_bits) float32
    sem_bwd_pm1 : np.ndarray,   # (vocab, n_bits) float32
    codebook_pm1: np.ndarray,   # (vocab, n_bits) float32
    prev_tokens : np.ndarray,   # (batch,) int32 — preceding token IDs
    next_tokens : np.ndarray,   # (batch,) int32 — following token IDs
) -> np.ndarray:
    """Bilateral midpoint convergence scores on GPU (cuBLAS HGEMM, fp16 tensor cores).

    Implements the closed-form bilateral fixed point:

        h*(b) = sign(sem_fwd_pm1[prev[b]] + sem_bwd_pm1[next[b]])
        score(b, v) = h*(b) · codebook_pm1[v] / n_bits

    h* is the unique token-HV equidistant between the forward trajectory of
    prev and the backward trajectory of next — no iteration needed.

    GPU path:
        Gather: sf[prev], sb[next]              ← torch index on CUDA
        Sum:    h_sum = sf_batch + sb_batch     ← elementwise, fp32
        Sign:   h_star = sign(h_sum)            ← fp32 (ties → +1)
        Matmul: score = h_star @ CB.T / n_bits  ← fp16 tensor cores (HGEMM)
        Normalise: row softmax → probability    ← fp32 clamp + divison

    Returns (batch, vocab) float32 row-normalised probability distribution.
    Falls back to numpy BLAS if GPU unavailable.
    """
    n_bits = codebook_pm1.shape[1]

    if not gpu_available():
        # NumPy BLAS fallback
        sf    = sem_fwd_pm1[prev_tokens]                              # (batch, n_bits)
        sb    = sem_bwd_pm1[next_tokens]                              # (batch, n_bits)
        h_sum = sf + sb
        h_star = np.sign(h_sum).astype(np.float32)
        h_star[h_star == 0.0] = 1.0
        raw   = (h_star @ codebook_pm1.T) / n_bits                   # (batch, vocab)
        probs = np.clip(0.5 + 0.49 * raw, 1e-30, 0.99).astype(np.float32)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    device   = _get_device()
    sf_t     = to_gpu_f16(sem_fwd_pm1)      # (vocab, n_bits) f16
    sb_t     = to_gpu_f16(sem_bwd_pm1)      # (vocab, n_bits) f16
    cb_t     = to_gpu_f16(codebook_pm1)     # (vocab, n_bits) f16
    prev_idx = to_gpu_i64(prev_tokens)      # (batch,)
    next_idx = to_gpu_i64(next_tokens)      # (batch,)

    # Gather bilateral pair rows from GPU tables
    sf_batch = sf_t[prev_idx]               # (batch, n_bits) f16 — forward context
    sb_batch = sb_t[next_idx]               # (batch, n_bits) f16 — backward context

    # Float32 sum then sign: avoids fp16 overflow on ±1 valued vectors
    h_sum  = sf_batch.float() + sb_batch.float()   # (batch, n_bits) f32
    h_star = torch.sign(h_sum)                       # (batch, n_bits) f32
    h_star[h_star == 0.0] = 1.0                      # ties → +1

    # fp16 HGEMM: (batch, n_bits) @ (n_bits, vocab) → (batch, vocab)
    raw = torch.mm(h_star.half(), cb_t.T).float() / n_bits   # (batch, vocab) f32

    # Row-normalise to probability distribution
    probs = torch.clamp(0.5 + 0.49 * raw, 1e-30, 0.99)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return to_cpu_f32(probs)


# ─────────────────────────────────────────────────────────────────────────────
# GPU bilateral scoring for FullBiDirHDC.vote_scores_vectorised
# ─────────────────────────────────────────────────────────────────────────────

def gpu_vote_scores_vectorised(
    codebook_vecs: np.ndarray,   # (vocab, n_words) uint64
    rule_bundle  : np.ndarray,   # (n_words,) uint64
    prev_tokens  : np.ndarray,   # (batch,) int32
) -> np.ndarray:
    """Compute bilateral softmax probabilities on GPU.

    Replaces the (batch, vocab, n_words) XOR + unpackbits in
    FullBiDirHDC.vote_scores_vectorised() with GPU matmul in pm1 space.

    Returns (batch, vocab) float32 probability distribution.
    Falls back to numpy if GPU unavailable.
    """
    vocab_size, n_words = codebook_vecs.shape
    n_bits = n_words * 64

    # Convert codebook and rule_bundle to pm1 on CPU (cheap for vocab=1024)
    cb_u8 = codebook_vecs.view(np.uint8).reshape(vocab_size, n_words * 8)
    cb_pm1 = (np.unpackbits(cb_u8, axis=1, bitorder='little').astype(np.float32) * 2.0 - 1.0)

    rb_u8 = rule_bundle.view(np.uint8).reshape(1, n_words * 8)
    rb_pm1 = (np.unpackbits(rb_u8, axis=1, bitorder='little').astype(np.float32) * 2.0 - 1.0)[0]

    # query_pm1[b] = cb_pm1[prev_t[b]] * rb_pm1  (element-wise product in pm1 = XOR in binary)
    # cosine(query, cb[v]) = (query · cb[v]) / n_bits
    query_pm1 = cb_pm1[prev_tokens] * rb_pm1[None, :]   # (batch, n_bits)

    if not gpu_available():
        fwd_scores = (query_pm1 @ cb_pm1.T) / n_bits    # (batch, vocab)
        bwd_hvs_pm1 = cb_pm1 * rb_pm1[None, :]          # (vocab, n_bits)
        bwd_scores = (bwd_hvs_pm1 @ cb_pm1[prev_tokens].T).T / n_bits  # (batch, vocab)
        consistency = (fwd_scores + bwd_scores) / 2.0
        consistency -= consistency.max(axis=1, keepdims=True)
        probs = np.exp(consistency)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

    device = _get_device()
    q_t   = to_gpu_f16(query_pm1)    # (batch, n_bits)
    cb_t  = to_gpu_f16(cb_pm1)       # (vocab, n_bits)
    rb_t  = to_gpu_f16(rb_pm1[None, :])  # (1, n_bits)

    fwd_scores = torch.mm(q_t, cb_t.T).float() / n_bits   # (batch, vocab)

    bwd_hvs = cb_t * rb_t                                  # (vocab, n_bits)
    cb_prev = cb_t[to_gpu_i64(prev_tokens)]                # (batch, n_bits)
    bwd_scores = torch.mm(bwd_hvs, cb_prev.T).float().T / n_bits  # (batch, vocab)

    consistency = (fwd_scores + bwd_scores) / 2.0
    probs = torch.softmax(consistency, dim=1)
    return to_cpu_f32(probs)


# ─────────────────────────────────────────────────────────────────────────────
# GPU batch_teleport shared_field + sign
# ─────────────────────────────────────────────────────────────────────────────

def gpu_batch_teleport(
    axes_pm1      : np.ndarray,   # (K, n_bits) float32
    axis_weights  : np.ndarray,   # (K,) float32
    goal_pm1      : np.ndarray,   # (n_bits,) float32
    goal_weight   : float,
    rule_pm1      : np.ndarray,   # (n_bits,) float32
    rule_weight   : float,
    manifold_fwd  : np.ndarray,   # (N, n_bits) float32
    manifold_bwd  : np.ndarray,   # (N, n_bits) float32
    inertia       : float,
    chain_pm1     = None,
    chain_weight  : float = 0.0,
    danger_pm1    = None,
    danger_weight : float = 0.0,
    oxytocin_pm1  = None,
    oxytocin_weight: float = 0.0,
    ego_pm1       = None,
    ego_weight    : float = 0.0,
    norm_pm1      = None,
    norm_weight   : float = 0.0,
    EPS           : float = 1e-8,
):
    """Compute h* = sign(spectrum) for all N hypotheses on GPU.

    Returns (h_star, goal_sims) as numpy arrays.
    Falls back to CPU numpy if GPU unavailable.
    """
    if not gpu_available():
        return None  # caller uses numpy path

    device = _get_device()

    # Shared field: axis_weights @ axes_pm1
    aw_t  = to_gpu_f32(axis_weights)   # (K,)
    ax_t  = to_gpu_f32(axes_pm1)       # (K, n_bits)
    shared = torch.mv(ax_t.T, aw_t)    # (n_bits,)  — equivalent to axis_weights @ axes_pm1

    if goal_weight > EPS:
        shared = shared + goal_weight * to_gpu_f32(goal_pm1)
    if rule_weight > EPS:
        shared = shared + rule_weight * to_gpu_f32(rule_pm1)
    if chain_weight > EPS and chain_pm1 is not None:
        shared = shared + chain_weight * to_gpu_f32(chain_pm1)
    if danger_weight > EPS and danger_pm1 is not None:
        shared = shared - danger_weight * to_gpu_f32(danger_pm1)
    if oxytocin_weight > EPS and oxytocin_pm1 is not None:
        shared = shared + oxytocin_weight * to_gpu_f32(oxytocin_pm1)
    if ego_weight > EPS and ego_pm1 is not None:
        shared = shared + ego_weight * to_gpu_f32(ego_pm1)
    if norm_weight > EPS and norm_pm1 is not None:
        shared = shared + norm_weight * to_gpu_f32(norm_pm1)

    # Per-hypothesis inertia
    fwd_t = to_gpu_f32(manifold_fwd)   # (N, n_bits)
    bwd_t = to_gpu_f32(manifold_bwd)   # (N, n_bits)
    inertia_half = inertia * 0.5
    spec = shared.unsqueeze(0) + inertia_half * (fwd_t + bwd_t)  # (N, n_bits)

    h_star = torch.sign(spec)
    h_star[h_star == 0.0] = 1.0        # ties → +1

    # Goal sims
    has_goal = goal_weight > EPS and np.any(goal_pm1 != 0.0)
    if has_goal:
        g_t = to_gpu_f32(goal_pm1)     # (n_bits,)
        goal_sims = (h_star * g_t.unsqueeze(0)).mean(dim=1)  # (N,)
        goal_sims_np = to_cpu_f32(goal_sims)
    else:
        N = manifold_fwd.shape[0]
        goal_sims_np = np.full(N, 0.5, dtype=np.float32)

    h_star_np = to_cpu_f32(h_star)
    return h_star_np, h_star_np, goal_sims_np
