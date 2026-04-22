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
"""

from __future__ import annotations

import lzma
import math
import os
import struct
import time
from typing import List, Optional, Tuple

import numpy as np

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
# Phase 1: Bigram frequency table
# ─────────────────────────────────────────────────────────────────────────────

def build_bigram_freq(
    tokens: np.ndarray,
    vocab_size: int,
    verbose: bool = True,
) -> np.ndarray:
    """Build normalised bigram probability table P[a, b] = P(b | a).

    O(N) pass over tokens. Returns (vocab_size, vocab_size) float32.
    Row-normalised: each row sums to 1.0 (or 0.0 for unseen tokens).

    Args:
        tokens     : (N,) int array of token IDs
        vocab_size : Vocabulary size
        verbose    : Print progress

    Returns:
        (vocab_size, vocab_size) float32 — P(next_token | prev_token)
    """
    if verbose:
        print(f"[BiDirTrain] Building bigram freq table "
              f"(vocab={vocab_size}, N={len(tokens):,})...")
    t0 = time.time()

    freq = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    t_prev = tokens[:-1].astype(np.int32)
    t_next = tokens[1:].astype(np.int32)

    # Clip to valid range
    t_prev = np.clip(t_prev, 0, vocab_size - 1)
    t_next = np.clip(t_next, 0, vocab_size - 1)

    np.add.at(freq, (t_prev, t_next), 1.0)

    # Row-normalise
    row_sums = freq.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0)
    freq /= row_sums

    elapsed = time.time() - t0
    if verbose:
        print(f"[BiDirTrain] Bigram freq done in {elapsed:.1f}s")
    return freq


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Distributed training
# ─────────────────────────────────────────────────────────────────────────────

def train_bidi_model(
    tokens: np.ndarray,
    vocab_size: int,
    n_words: int,
    seeds: List[int],
    time_budget_s: float = 480.0,
    n_axes: int = 19,
    n_hyp: int = 200,
    max_iters: int = 40,
    chunk_size: int = 500_000,
    verbose: bool = True,
) -> "FullBiDirHDC":  # type: ignore[name-defined]
    """Full distributed training pipeline.

    Steps:
    1. Build bigram_freq (rank 0 only, ~2s for 500M tokens)
    2. Shard tokens across ranks
    3. Each rank: engine.train_on_tokens(shard, bigram_freq)
    4. dist.all_reduce(SUM) on rule_bundle
    5. Multi-seed merge (XOR majority vote across seeds)
    6. Return trained engine

    Args:
        tokens        : (N,) int array of training tokens
        vocab_size    : Vocabulary size
        n_words       : HV width in uint64 words (n_words × 64 bits)
        seeds         : List of random seeds for multi-seed merge
        time_budget_s : Total time budget in seconds
        n_axes        : Number of golden-ratio axes
        n_hyp         : Hypotheses per action
        max_iters     : Manifold propagation cap
        chunk_size    : Training chunk size for progress reporting
        verbose       : Print progress

    Returns:
        Trained FullBiDirHDC engine (on rank 0; other ranks return None)
    """
    from _bidi_hdc_engine import Codebook, FullBiDirHDC

    rank       = _dist_rank()
    world_size = _dist_world_size()
    t_start    = time.time()

    if verbose and _dist_is_main():
        print(f"\n{'='*60}")
        print(f"[BiDirTrain] FullBiDirHDC Training")
        print(f"[BiDirTrain] vocab_size={vocab_size}, n_words={n_words}")
        print(f"[BiDirTrain] seeds={seeds}, world_size={world_size}")
        print(f"[BiDirTrain] HV bits={n_words*64:,}, budget={time_budget_s:.0f}s")
        print(f"{'='*60}\n")

    # Phase 1: Build bigram freq (rank 0 only — fast, no need to distribute)
    bigram_freq = None
    if _dist_is_main():
        bigram_freq = build_bigram_freq(tokens, vocab_size, verbose=verbose)

    # Broadcast bigram_freq to all ranks
    if world_size > 1:
        try:
            import torch
            import torch.distributed as _dist_mod
            if _dist_mod.is_available() and _dist_mod.is_initialized():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
                if bigram_freq is None:
                    bigram_freq = np.zeros((vocab_size, vocab_size), dtype=np.float32)
                t = torch.from_numpy(bigram_freq).to(device)
                _dist_mod.broadcast(t, src=0)
                bigram_freq = t.cpu().numpy()
        except Exception as e:
            if verbose:
                print(f"[BiDirTrain] Bigram broadcast failed ({e}), using local copy")
            if bigram_freq is None:
                bigram_freq = build_bigram_freq(tokens, vocab_size, verbose=False)

    if bigram_freq is None:
        bigram_freq = build_bigram_freq(tokens, vocab_size, verbose=False)

    # Phase 2: Shard tokens across ranks
    N = len(tokens)
    shard_start = rank * N // world_size
    shard_end   = (rank + 1) * N // world_size
    shard = tokens[shard_start:shard_end]

    if verbose:
        print(f"[BiDirTrain] Rank {rank}: shard [{shard_start:,}, {shard_end:,}) "
              f"= {len(shard):,} tokens")

    # Per-seed training budget
    elapsed_so_far = time.time() - t_start
    per_seed_budget = max(30.0, (time_budget_s - elapsed_so_far - 30.0) / len(seeds))

    # Phase 3: Train each seed, merge rule bundles
    merged_rule_bundle = np.zeros(n_words, dtype=np.uint64)
    primary_engine = None

    for seed_idx, seed in enumerate(seeds):
        if time.time() - t_start > time_budget_s - 30:
            if verbose and _dist_is_main():
                print(f"[BiDirTrain] Time budget reached at seed {seed_idx}/{len(seeds)}")
            break

        if verbose and _dist_is_main():
            print(f"\n[BiDirTrain] Training seed {seed} ({seed_idx+1}/{len(seeds)})...")

        cb = Codebook(vocab_size=vocab_size, n_words=n_words, seed=seed)
        engine = FullBiDirHDC(
            codebook  = cb,
            n_axes    = n_axes,
            n_hyp     = n_hyp,
            max_iters = max_iters,
        )

        engine.train_on_tokens(
            tokens      = shard,
            bigram_freq = bigram_freq,
            chunk_size  = chunk_size,
            verbose     = verbose and _dist_is_main(),
        )

        # All-reduce rule_bundle across ranks
        local_bundle = engine._rule_bundle.copy()
        merged = _dist_all_reduce_sum_numpy(local_bundle.astype(np.int64))
        # Majority vote: bit is 1 if sum > world_size/2
        engine._rule_bundle = (merged > (world_size // 2)).astype(np.uint64)

        # XOR merge across seeds (majority vote)
        merged_rule_bundle = np.bitwise_xor(merged_rule_bundle, engine._rule_bundle)

        if seed_idx == 0:
            primary_engine = engine

        if verbose and _dist_is_main():
            elapsed = time.time() - t_start
            print(f"[BiDirTrain] Seed {seed} done. Elapsed: {elapsed:.1f}s")

    # Apply merged rule bundle to primary engine
    if primary_engine is not None:
        primary_engine._rule_bundle = merged_rule_bundle

    _dist_barrier()

    # Non-main ranks return None after barrier
    if not _dist_is_main():
        return None  # type: ignore[return-value]

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n[BiDirTrain] Training complete in {elapsed:.1f}s")

    return primary_engine


# ─────────────────────────────────────────────────────────────────────────────
# Byte LUT helpers (same as reference train_gpt.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_byte_luts(sp_model, vocab_size: int):
    """Build base_bytes and has_leading_space LUTs from SentencePiece model.

    Identical to the reference train_gpt.py byte counting logic.

    Returns:
        base_bytes        : (vocab_size,) int32 — UTF-8 byte length of each token
        has_leading_space : (vocab_size,) bool  — True if token has leading space
    """
    base_bytes        = np.ones(vocab_size, dtype=np.int32)
    has_leading_space = np.zeros(vocab_size, dtype=bool)

    for tok_id in range(vocab_size):
        try:
            piece = sp_model.id_to_piece(tok_id)
            if piece is None:
                continue
            # Leading space marker in SentencePiece
            if piece.startswith('\u2581'):
                has_leading_space[tok_id] = True
                piece = piece[1:]  # strip the marker for byte counting
            base_bytes[tok_id] = max(1, len(piece.encode('utf-8')))
        except Exception:
            pass

    return base_bytes, has_leading_space


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
    base_bytes, has_leading_space = _build_byte_luts(sp_model, vocab_size)

    total_bits  = 0.0
    total_bytes = 0
    total_nats  = 0.0
    total_toks  = 0

    N = len(val_tokens)

    for chunk_start in range(1, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)

        prev_toks = val_tokens[chunk_start - 1 : chunk_end - 1].astype(np.int32)
        tgt_toks  = val_tokens[chunk_start : chunk_end].astype(np.int32)

        # Clip to valid range
        prev_toks = np.clip(prev_toks, 0, vocab_size - 1)
        tgt_toks  = np.clip(tgt_toks,  0, vocab_size - 1)

        # Vectorised bilateral scoring: (batch, vocab_size) float32
        probs = engine.vote_scores_vectorised(prev_toks)  # (batch, vocab)

        # Optional SpiralDSV blend
        if spiral_dsv is not None and spiral_blend_alpha > 0.0:
            p_spiral = spiral_dsv.vote_scores_all_vocab(prev_toks)  # (batch, vocab)
            probs = (1.0 - spiral_blend_alpha) * probs + spiral_blend_alpha * p_spiral
            # Re-normalise after blend
            probs /= probs.sum(axis=1, keepdims=True)

        # Extract p_correct for each target token
        p_correct = probs[np.arange(len(tgt_toks)), tgt_toks]  # (batch,)
        p_correct = np.clip(p_correct, 1e-30, 1.0)

        # BPB accumulation (identical formula to reference)
        # has_leading_space: add 1 byte for the leading space
        tok_bytes = np.where(
            has_leading_space[tgt_toks],
            base_bytes[tgt_toks].astype(np.float64) + 1.0,
            base_bytes[tgt_toks].astype(np.float64)
        )
        tok_bytes = np.maximum(tok_bytes, 1.0)

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
