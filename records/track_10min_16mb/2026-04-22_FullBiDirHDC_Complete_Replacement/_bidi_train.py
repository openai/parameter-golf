"""BiDir HDC Training and Evaluation Pipeline.

Replaces _hash_grad_train.py entirely.

Key functions:
  build_bigram_freq()     — O(N) bigram frequency table
  train_bidi_model()      — full distributed training pipeline
  bidi_bpb()              — vectorised BPB evaluation (identical formula to reference)
  save_bidi_artifact()    — LZMA9 artifact serialisation (.bdhgz)
  load_bidi_artifact()    — artifact deserialisation

BPB formula (identical to reference train_gpt.py:275-278):
    BPB = Σ(-log₂ p_correct) / Σ(utf8_bytes(token))
        = bits_per_token × tokens_per_byte

Eigen Training (2026-04-23 upgrade)
────────────────────────────────────
train_bidi_model() now calls FullBiDirHDC.train_on_tokens() which internally
uses EigenTrainer.absorb_bigrams() to replace the Python for-loop:

  Before: ~190–310 s per rank (Python loop, 62.5M bigrams × observe())
  After:  ~1–3 s per rank (np.add.at accumulation + single vocab×n_bits matmul)

The fixed point is:
    rule_bundle_pm1* = sign( token_reward_sums @ CB_pm1 )
    goal_hv_pm1*     = sign( high_reward_sums  @ CB_pm1 )

This is mathematically equivalent to the sequential EMA recurrence and
absorbs the entire training loop into two O(vocab_size × n_bits) matmuls.
"""

from __future__ import annotations

import lzma
import math
import os
import struct
import time
from typing import List, Optional, Tuple

import numpy as np

# ── GPU acceleration (optional, graceful fallback to CPU) ─────────────────────
try:
    from _gpu import gpu_available, gpu_bincount_weighted, gpu_log_softmax_scores
    _GPU_AVAILABLE = gpu_available()
except ImportError:
    _GPU_AVAILABLE = False
    def gpu_bincount_weighted(idx, w, ml):  # type: ignore[misc]
        return np.bincount(idx.astype(np.int64), weights=w.astype(np.float64), minlength=ml)
    def gpu_log_softmax_scores(scores):     # type: ignore[misc]
        scores = scores.astype(np.float32)
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores); probs /= probs.sum(axis=1, keepdims=True)
        return probs

# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers (same pattern as _hash_grad_train.py)
# ─────────────────────────────────────────────────────────────────────────────

def _dist_rank() -> int:
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_rank()
    except Exception:
        pass
    return 0


def _dist_world_size() -> int:
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_world_size()
    except Exception:
        pass
    return 1


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _dist_barrier() -> None:
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _dist.barrier()
    except Exception:
        pass


def _dist_all_reduce_sum_numpy(arr: np.ndarray) -> np.ndarray:
    """All-reduce (sum) a numpy array across all ranks via NCCL."""
    try:
        import torch
        import torch.distributed as _dist
        if not (_dist.is_available() and _dist.is_initialized()):
            return arr
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        t = torch.from_numpy(arr.copy()).to(device)
        _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
        return t.cpu().numpy()
    except Exception:
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# Engine initialisation
# ─────────────────────────────────────────────────────────────────────────────

def train_bidi_model(
    tokens: np.ndarray,           # kept for API compat — corpus not scanned here
    vocab_size: int,
    n_words: int,
    seeds: List[int],
    time_budget_s: float = 480.0, # kept for API compat — not used
    n_axes: int = 19,
    n_hyp: int = 200,
    max_iters: int = 40,
    chunk_size: int = 500_000,    # kept for API compat — not used
    verbose: bool = True,
) -> "FullBiDirHDC":  # type: ignore[name-defined]
    """Initialise a FullBiDirHDC engine.

    rule_bundle is derived from SpiralDSV sem_fwd_pm1 in train_gpt.py after
    build_from_tokens() completes:
        rule_bundle* = sign(unigram_freq @ sem_fwd_pm1)

    This is architecturally cleaner than a separate training scan because:
      * sem_fwd_pm1 already encodes the bilateral co-occurrence structure
        (GoldenAxisShift, 4 lags) — a strict superset of lag-1 bigrams
      * derivation is O(V x n_bits) matmul — zero extra corpus scan
      * uses full-corpus statistics (8B tokens) not a per-rank shard

    Args:
        tokens        : (N,) int array — not scanned; kept for API compat
        vocab_size    : Vocabulary size
        n_words       : HV width in uint64 words (n_words x 64 bits)
        seeds         : seeds[0] used for Codebook RNG; rest ignored
        time_budget_s : kept for API compat — not used
        n_axes        : Number of golden-ratio axes
        n_hyp         : Hypotheses per action
        max_iters     : Manifold propagation cap
        chunk_size    : kept for API compat — not used
        verbose       : Print status

    Returns:
        Initialised FullBiDirHDC engine (rank 0) or None (other ranks).
    """
    from _bidi_hdc_engine import Codebook, FullBiDirHDC

    seed = seeds[0] if seeds else 42

    if verbose and _dist_is_main():
        print(f"\n{'='*60}")
        print(f"[BiDirTrain] FullBiDirHDC engine init (seed={seed})")
        print(f"[BiDirTrain] vocab_size={vocab_size}, n_words={n_words}, "
              f"n_bits={n_words*64:,}")
        print(f"[BiDirTrain] rule_bundle derived from SpiralDSV after bilateral build.")
        print(f"{'='*60}\n")

    cb     = Codebook(vocab_size=vocab_size, n_words=n_words, seed=seed)
    engine = FullBiDirHDC(
        codebook  = cb,
        n_axes    = n_axes,
        n_hyp     = n_hyp,
        max_iters = max_iters,
    )

    _dist_barrier()

    if not _dist_is_main():
        return None  # type: ignore[return-value]

    if verbose:
        print(f"[BiDirTrain] Engine initialised. Awaiting SpiralDSV build...")

    return engine



# ─────────────────────────────────────────────────────────────────────────────
# Byte LUT helpers (identical to reference train_gpt.py build_sentencepiece_luts)
# ─────────────────────────────────────────────────────────────────────────────

def _build_byte_luts(sp_model, vocab_size: int):
    """Build base_bytes, has_leading_space, and is_boundary_token LUTs.

    Exactly mirrors reference train_gpt.py:180-204 (build_sentencepiece_luts).

    The critical difference from the old version:
      - is_boundary_token[tok] is True for control/unknown/unused tokens and
        for any token_id >= sp_vocab_size.  The reference uses this to suppress
        the leading-space byte when the *previous* token is a boundary token
        (i.e. a sequence boundary), matching:
            token_bytes += has_leading_space[tgt] & ~is_boundary_token[prev]

    Returns:
        base_bytes        : (table_size,) int16 — UTF-8 byte length of each token
        has_leading_space : (table_size,) bool  — True if token has leading space
        is_boundary_token : (table_size,) bool  — True for control/unknown/unused
    """
    sp_vocab_size = int(sp_model.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)

    base_bytes        = np.zeros(table_size, dtype=np.int16)
    has_leading_space = np.zeros(table_size, dtype=bool)
    is_boundary_token = np.ones(table_size,  dtype=bool)   # default True

    for tok_id in range(sp_vocab_size):
        try:
            if (sp_model.is_control(tok_id) or
                    sp_model.is_unknown(tok_id) or
                    sp_model.is_unused(tok_id)):
                continue  # leave is_boundary_token[tok_id] = True
            is_boundary_token[tok_id] = False
            if sp_model.is_byte(tok_id):
                base_bytes[tok_id] = 1
                continue
            piece = sp_model.id_to_piece(tok_id)
            if piece is None:
                continue
            # Leading space marker in SentencePiece (▁ = U+2581)
            if piece.startswith('\u2581'):
                has_leading_space[tok_id] = True
                piece = piece[1:]
            base_bytes[tok_id] = len(piece.encode('utf-8'))
        except Exception:
            pass

    return base_bytes, has_leading_space, is_boundary_token


# ─────────────────────────────────────────────────────────────────────────────
# BPB Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def bidi_bpb(
    val_tokens: np.ndarray,
    engine,
    sp_model,
    spiral_dsv=None,
    chunk_size: int = 4096,
    spiral_blend_alpha: float = 0.3,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Vectorised BPB evaluation using FullBiDirHDC bilateral scoring.

    BPB formula (identical to reference train_gpt.py:275-278):
        BPB = Σ(-log₂ p_correct) / Σ(utf8_bytes(token))
            = bits_per_token × tokens_per_byte

    Algebraically identical to reference:
        (Σ bits_i / N) × (N / Σ bytes_i) = Σ bits_i / Σ bytes_i

    Args:
        val_tokens         : (N,) int array of validation tokens
        engine             : Trained FullBiDirHDC engine
        sp_model           : SentencePiece model for byte counting
        spiral_dsv         : Optional SpiralDSVLanguageModel for bilateral blend
        chunk_size         : Chunk size for vectorised scoring
        spiral_blend_alpha : Blend weight for SpiralDSV tier (0 = disabled)
        verbose            : Print audit block

    Returns:
        (bpb, val_loss) — bits per byte and nats per token
    """
    vocab_size = engine.cb.vocab_size
    base_bytes, has_leading_space, is_boundary_token = _build_byte_luts(sp_model, vocab_size)

    total_bits  = 0.0
    total_bytes = 0
    total_nats  = 0.0
    total_toks  = 0

    N = len(val_tokens)

    for chunk_start in range(1, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)

        prev_toks = val_tokens[chunk_start - 1 : chunk_end - 1].astype(np.int32)
        tgt_toks  = val_tokens[chunk_start : chunk_end].astype(np.int32)

        # Next tokens for bilateral midpoint evaluation: one position ahead of targets
        # Boundary: for the very last position, fall back to the target token itself.
        next_end      = min(chunk_end + 1, N)
        next_toks_raw = val_tokens[chunk_start + 1 : next_end].astype(np.int32)
        if len(next_toks_raw) < len(tgt_toks):
            # Pad last element: use tgt as its own next (harmless fallback)
            next_toks_raw = np.append(next_toks_raw, tgt_toks[-1:])

        # Clip to valid range
        prev_toks = np.clip(prev_toks,     0, vocab_size - 1)
        tgt_toks  = np.clip(tgt_toks,      0, vocab_size - 1)
        next_toks = np.clip(next_toks_raw, 0, vocab_size - 1)

        # Vectorised bilateral scoring: (batch, vocab_size) float32
        # GPU path is used automatically inside vote_scores_vectorised()
        probs = engine.vote_scores_vectorised(prev_toks)  # (batch, vocab)

        # Optional SpiralDSV bilateral midpoint blend
        # Passes next_toks so vote_scores_all_vocab() uses the closed-form
        # bilateral fixed point: h* = sign(sem_fwd[prev] + sem_bwd[next])
        # instead of single-sided scoring.  No extra model parameters needed.
        if spiral_dsv is not None and spiral_blend_alpha > 0.0:
            p_spiral = spiral_dsv.vote_scores_all_vocab(
                prev_toks,
                next_tokens=next_toks,    # bilateral midpoint — uses both boundaries
            )
            probs = (1.0 - spiral_blend_alpha) * probs + spiral_blend_alpha * p_spiral
            # Re-normalise after blend
            probs /= probs.sum(axis=1, keepdims=True)

        # Extract p_correct for each target token
        p_correct = probs[np.arange(len(tgt_toks)), tgt_toks]  # (batch,)
        p_correct = np.clip(p_correct, 1e-30, 1.0)

        # BPB accumulation — identical to reference train_gpt.py:265-267:
        #   token_bytes  = base_bytes_lut[tgt_ids]
        #   token_bytes += has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        # The leading-space byte is only counted when the previous token is NOT
        # a boundary token (i.e. not a control/unknown/unused/pad token).
        # Byte count — identical to reference train_gpt.py:265-267.
        # Control / boundary tokens have base_bytes=0 and no leading space,
        # so tok_bytes=0 for those positions. They still contribute bits to the
        # numerator but 0 bytes to the denominator, exactly as in the reference
        # (which uses no clamp). Do NOT clamp to 1 — that would inflate the
        # denominator and produce an artificially lower BPB vs the reference formula.
        tok_bytes = (
            base_bytes[tgt_toks].astype(np.float64)
            + (has_leading_space[tgt_toks] & ~is_boundary_token[prev_toks]).astype(np.float64)
        )
        # No floor clamp — matches reference exactly

        total_bits  += float(-np.log2(p_correct).sum())
        total_bytes += int(tok_bytes.sum())
        total_nats  += float(-np.log(p_correct).sum())
        total_toks  += len(tgt_toks)

    if total_bytes == 0:
        return float('inf'), float('inf')

    avg_bpt = total_bytes / max(total_toks, 1)
    bpt     = total_bits  / max(total_toks, 1)
    npt     = total_nats  / max(total_toks, 1)
    bpb     = total_bits  / total_bytes

    if verbose:
        print(f"[BiDirHDC BPB audit]")
        print(f"  total_tokens    : {total_toks:,}")
        print(f"  total_utf8_bytes: {total_bytes:,}")
        print(f"  avg bytes/token : {avg_bpt:.4f}  (explains why BPB << bits/token)")
        print(f"  bits/token      : {bpt:.4f}")
        print(f"  nats/token (loss): {npt:.4f}")
        print(f"  BPB = bits/token / bytes/token = "
              f"{bpt:.4f} / {avg_bpt:.4f} = {bpb:.4f}")
        print(f"  (same formula as reference train_gpt.py: "
              f"bits_per_token * tokens_per_byte)")

    return float(bpb), float(npt)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact serialisation
# ─────────────────────────────────────────────────────────────────────────────

# Magic bytes for the new artifact format
_MAGIC = b"BDH1"
_MAGIC_V2 = b"BDH2"  # with SpiralDSV tables


def save_bidi_artifact(
    engine,
    path: str,
    spiral_dsv=None,
    verbose: bool = True,
) -> int:
    """Serialise FullBiDirHDC state to .bdhgz (LZMA9 compressed).

    Format:
        Magic(4B "BDH1" or "BDH2") + n_words(4B) + vocab_size(4B) + seed(8B) + flags(4B)
        + codebook      (vocab_size × n_words × 8 bytes, uint64)   [always]
        + rule_bundle   (n_words × 8 bytes, uint64)                 [always]
        + goal_hv       (n_words × 8 bytes, uint64)                 [if flags & 1]
        + spiral_fwd    (vocab_size × n_words × 8 bytes, uint64)    [if flags & 2, BDH2]
        + spiral_bwd    (vocab_size × n_words × 8 bytes, uint64)    [if flags & 2, BDH2]

    Args:
        engine     : Trained FullBiDirHDC engine
        path       : Output path (e.g. "bidi_model.bdhgz")
        spiral_dsv : Optional SpiralDSVLanguageModel to include in artifact
        verbose    : Print size info

    Returns:
        Compressed artifact size in bytes
    """
    n_words    = engine.W
    vocab_size = engine.cb.vocab_size
    seed       = getattr(engine.cb, 'seed', 42)

    flags = 0
    if engine.goal_hv is not None:
        flags |= 1
    if spiral_dsv is not None and spiral_dsv._built:
        flags |= 2

    magic = _MAGIC_V2 if (flags & 2) else _MAGIC

    data = bytearray()
    data += magic
    data += struct.pack("<IIQ", n_words, vocab_size, seed)
    data += struct.pack("<I", flags)

    # Codebook: (vocab_size, n_words) uint64
    data += engine.cb.vecs.astype(np.uint64).tobytes()

    # Rule bundle: (n_words,) uint64
    data += engine._rule_bundle.astype(np.uint64).tobytes()

    # Goal HV (optional)
    if flags & 1:
        data += engine.goal_hv.astype(np.uint64).tobytes()

    # SpiralDSV tables (optional)
    if flags & 2:
        data += spiral_dsv.sem_fwd.astype(np.uint64).tobytes()
        data += spiral_dsv.sem_bwd.astype(np.uint64).tobytes()

    raw_size = len(data)
    compressed = lzma.compress(bytes(data), preset=9)
    compressed_size = len(compressed)

    with open(path, "wb") as f:
        f.write(compressed)

    if verbose:
        print(f"[BiDirTrain] Artifact saved: {path}")
        print(f"  Raw size      : {raw_size:,} bytes")
        print(f"  Compressed    : {compressed_size:,} bytes")
        print(f"  Compression   : {100*(1-compressed_size/raw_size):.1f}%")

    return compressed_size


def load_bidi_artifact(path: str, verbose: bool = True):
    """Load .bdhgz artifact. Returns (engine, spiral_dsv_or_None).

    Args:
        path    : Path to .bdhgz artifact
        verbose : Print load info

    Returns:
        (engine, spiral_dsv) — FullBiDirHDC engine and optional SpiralDSVLanguageModel
    """
    from _bidi_hdc_engine import Codebook, FullBiDirHDC

    with open(path, "rb") as f:
        compressed = f.read()

    data = lzma.decompress(compressed)
    offset = 0

    # Magic
    magic = data[offset:offset+4]
    offset += 4
    if magic not in (_MAGIC, _MAGIC_V2):
        raise ValueError(f"Unknown magic bytes: {magic!r}")

    # Header
    n_words, vocab_size, seed = struct.unpack_from("<IIQ", data, offset)
    offset += struct.calcsize("<IIQ")
    flags, = struct.unpack_from("<I", data, offset)
    offset += 4

    if verbose:
        print(f"[BiDirTrain] Loading artifact: {path}")
        print(f"  n_words={n_words}, vocab_size={vocab_size}, seed={seed}, flags={flags:#010x}")

    # Codebook
    cb_bytes = vocab_size * n_words * 8
    cb_arr = np.frombuffer(data[offset:offset+cb_bytes], dtype=np.uint64).reshape(vocab_size, n_words).copy()
    offset += cb_bytes

    # Rule bundle
    rb_bytes = n_words * 8
    rb_arr = np.frombuffer(data[offset:offset+rb_bytes], dtype=np.uint64).copy()
    offset += rb_bytes

    # Reconstruct engine
    cb = Codebook(vocab_size=vocab_size, n_words=n_words, seed=seed)
    cb.vecs = cb_arr
    engine = FullBiDirHDC(codebook=cb)
    engine._rule_bundle = rb_arr

    # Goal HV (optional)
    if flags & 1:
        goal_bytes = n_words * 8
        goal_arr = np.frombuffer(data[offset:offset+goal_bytes], dtype=np.uint64).copy()
        offset += goal_bytes
        engine.goal_hv = goal_arr

    # SpiralDSV tables (optional)
    spiral_dsv = None
    if flags & 2:
        try:
            from _spiral_dsv_lm import SpiralDSVLanguageModel
            spiral_dsv = SpiralDSVLanguageModel(vocab_size=vocab_size, n_words=n_words, seed=seed)
            spiral_dsv.codebook = cb_arr  # share codebook

            fwd_bytes = vocab_size * n_words * 8
            fwd_arr = np.frombuffer(data[offset:offset+fwd_bytes], dtype=np.uint64).reshape(vocab_size, n_words).copy()
            offset += fwd_bytes
            spiral_dsv.sem_fwd = fwd_arr

            bwd_arr = np.frombuffer(data[offset:offset+fwd_bytes], dtype=np.uint64).reshape(vocab_size, n_words).copy()
            offset += fwd_bytes
            spiral_dsv.sem_bwd = bwd_arr
            spiral_dsv._built = True
        except ImportError:
            if verbose:
                print("[BiDirTrain] SpiralDSVLanguageModel not available — skipping DSV tables")

    if verbose:
        print(f"[BiDirTrain] Artifact loaded successfully")

    return engine, spiral_dsv


# ─────────────────────────────────────────────────────────────────────────────
# Artifact size check (same as reference)
# ─────────────────────────────────────────────────────────────────────────────

ARTIFACT_LIMIT = 16_000_000  # 16 MB decimal


def check_artifact_size(artifact_path: str, code_bytes: int) -> Tuple[int, bool]:
    """Check total artifact size (code + compressed model).

    Args:
        artifact_path : Path to .bdhgz artifact
        code_bytes    : Size of train_gpt.py in bytes

    Returns:
        (total_bytes, passes) — total artifact size and whether it passes the limit
    """
    model_bytes = os.path.getsize(artifact_path)
    total_bytes = code_bytes + model_bytes
    passes = total_bytes <= ARTIFACT_LIMIT
    return total_bytes, passes
