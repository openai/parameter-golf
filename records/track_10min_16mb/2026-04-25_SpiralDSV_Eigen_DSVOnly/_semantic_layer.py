"""Semantic Layer — DSV-Only Build + Eval.

Wraps SpiralDSVLanguageModel + EigenTrainer and exposes:
  - build_spiral_dsv()          : build bilateral tables from 500M training tokens
  - eval_spiral_dsv_bpb()       : compute BPB on validation tokens
  - save_spiral_dsv_artifact()  : save sem_fwd + sem_bwd to LZMA9-compressed .hgz
  - load_spiral_dsv_artifact()  : load sem_fwd + sem_bwd from .hgz artifact

Architecture (DSV-only, no NMF):
  - Removes all NMF phases (0–5, 8–9) from the 2026-04-07 pipeline
  - Removes Hadamard codebook (replaced by SpiralDSV internal GoldenAxisShift codebook)
  - Removes embed / W_out arrays (16 MB freed for DSV)
  - Keeps only Phase 6: EigenTrainer.build_bilateral_from_tokens()
    with GoldenAxisShift per-lag codebook rotation + PMI centering
  - Budget: 8 MB sem_fwd + 8 MB sem_bwd = 16 MB total (n_words=1024, 65,536 bits)

Artifact format (HGZ3):
  Magic(4B "HGZ3") + vocab_size(4B) + n_words(4B) + flags(4B)
  + sem_fwd bytes  (vocab_size × n_words × 8)   [8 MB]
  + sem_bwd bytes  (vocab_size × n_words × 8)   [8 MB]
  LZMA9 compressed → ~2–4 MB on disk

The SpiralDSVLanguageModel internal codebook (vocab × n_words uint64) is NOT
stored in the artifact — it is regenerated deterministically from seed=42 at
eval time, keeping the artifact within the 16 MB limit.

Coherence Gating (optional):
  A running document centroid that biases predictions toward tokens coherent
  with the document seen so far. Enabled by passing W_COHERENCE > 0.0 to
  eval_spiral_dsv_bpb(). Suggested range: 0.1–0.5. Start at 0.3.
"""

from __future__ import annotations

import lzma
import struct
import time
from typing import Optional, Tuple

import numpy as np

# ── Local imports (same directory) ───────────────────────────────────────────
from _spiral_dsv_lm import SpiralDSVLanguageModel, GOLDEN_AXES
from _eigen_convergence import EigenTrainer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE: int = 1024
N_WORDS_H100: int = 1024   # 65,536 bits — full 16 MB budget, for 8×H100 SXM
N_WORDS_4090: int = 128    # 8,192 bits  — fits RTX 4090 (24 GB VRAM)
N_WORDS_TEST: int = 16     # 1,024 bits  — smoke test / CPU
CTX_LEN: int = 4           # lags 1..4 with GoldenAxisShift per lag

_HGZ3_MAGIC = b'HGZ3'
_HGZ3_HEADER_FMT = '<4sIII'   # magic(4s) + vocab_size(I) + n_words(I) + flags(I)
_HGZ3_HEADER_SIZE = struct.calcsize(_HGZ3_HEADER_FMT)


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

def build_spiral_dsv(
    tokens: np.ndarray,
    vocab_size: int = VOCAB_SIZE,
    n_words: int = N_WORDS_H100,
    ctx_len: int = CTX_LEN,
    seed: int = 42,
    time_budget_s: float = 300.0,
    dist_rank: int = 0,
    dist_world_size: int = 1,
    verbose: bool = True,
) -> SpiralDSVLanguageModel:
    """Build bilateral DSV tables from token sequence.

    Uses EigenTrainer.build_bilateral_from_tokens() with:
      - GoldenAxisShift per-lag codebook rotation (lags 1..ctx_len)
      - PMI centering (encodes anti-correlations as negative entries)
      - GPU-accelerated dual HGEMM matmul (H100 tensor cores)
      - Distributed all-reduce across all ranks

    Memory budget (n_words=1024):
      sem_fwd : (1024, 1024) uint64 = 8 MB
      sem_bwd : (1024, 1024) uint64 = 8 MB
      Total   : 16 MB ✅

    Args:
        tokens          : (N,) int32/uint16 — training token sequence
        vocab_size      : Vocabulary size (default 1024)
        n_words         : HV width in uint64 words (default 1024 → 65,536 bits)
        ctx_len         : Context depth / number of lags (default 4)
        seed            : Random seed for codebook generation (default 42)
        time_budget_s   : Wall-clock budget in seconds (default 300)
        dist_rank       : This rank's index in distributed group (default 0)
        dist_world_size : Total number of ranks (default 1)
        verbose         : Print progress (default True)

    Returns:
        SpiralDSVLanguageModel with sem_fwd and sem_bwd populated.
        Non-zero ranks return a model with _built=False (tables not populated).
    """
    t0 = time.time()

    model = SpiralDSVLanguageModel(
        vocab_size=vocab_size,
        n_words=n_words,
        seed=seed,
    )

    if verbose and dist_rank == 0:
        n_bits = n_words * 64
        print(f"\n[SpiralDSV] Building bilateral tables")
        print(f"[SpiralDSV] vocab={vocab_size}, n_words={n_words}, "
              f"n_bits={n_bits:,}, ctx_len={ctx_len}")
        print(f"[SpiralDSV] sem_fwd+sem_bwd budget: "
              f"{2 * vocab_size * n_words * 8 / 1_000_000:.1f} MB")
        print(f"[SpiralDSV] dist: rank={dist_rank}/{dist_world_size}, "
              f"budget={time_budget_s:.0f}s")

    # Ensure GOLDEN_AXES has offsets computed for all ctx_len lags
    for c in range(1, ctx_len + 1):
        GOLDEN_AXES.offset(c)

    axis_word_shifts = [
        (GOLDEN_AXES._word_shifts[c], GOLDEN_AXES._bit_shifts[c])
        for c in range(1, ctx_len + 1)
    ]

    # EigenTrainer: frequency-weighted bilateral build with PMI centering
    trainer = EigenTrainer.from_codebook_uint64(
        codebook_vecs=model.codebook,   # (vocab, n_words) uint64
        goal_threshold=10.0,
    )

    result = trainer.build_bilateral_from_tokens(
        tokens=tokens,
        ctx_len=ctx_len,
        axis_word_shifts=axis_word_shifts,
        chunk_size=2_000_000,
        verbose=verbose,
        time_budget_s=time_budget_s,
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
    )

    # Non-zero ranks: histograms contributed via all-reduce, nothing to store
    if result.get('sem_fwd_u64') is None:
        if verbose:
            print(f"[SpiralDSV] rank={dist_rank}: histogram contributed, "
                  f"tables not stored (non-main rank)")
        return model  # _built=False

    # Apply fixed-point results
    model.sem_fwd = result['sem_fwd_u64']   # (vocab, n_words) uint64
    model.sem_bwd = result['sem_bwd_u64']   # (vocab, n_words) uint64
    model._built = True
    model._invalidate_pm1_cache()

    elapsed = time.time() - t0
    if verbose:
        total_pairs = result.get('total_pairs', 0)
        print(f"[SpiralDSV] Build complete: {total_pairs:,} pairs in {elapsed:.2f}s")
        print(f"[SpiralDSV] sem_fwd: {model.sem_fwd.nbytes:,} bytes  "
              f"sem_bwd: {model.sem_bwd.nbytes:,} bytes")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────────────

def eval_spiral_dsv_bpb(
    val_tokens: np.ndarray,
    model: SpiralDSVLanguageModel,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary_token: Optional[np.ndarray] = None,
    batch_size: int = 500_000,
    W_COHERENCE: float = 0.0,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Compute BPB using bilateral DSV scores for all positions.

    Uses model.vote_scores_all_vocab(prev_tokens) for all positions.
    GPU-accelerated via _gpu.py (cuBLAS HGEMM) when CUDA is available.

    BPB formula (identical to reference train_gpt.py):
        BPB = Σ(-log₂ p_correct) / Σ(utf8_bytes(token))

    Coherence gating (W_COHERENCE > 0):
        Maintains a running document centroid (mean of codebook vectors for
        all tokens seen so far in the current document). Biases predictions
        toward tokens coherent with the document topic.
        Reset on is_boundary_token boundaries.
        Suggested W_COHERENCE: 0.3 (tune on held-out val subset).

    Args:
        val_tokens        : (N,) int32/uint16 — validation token sequence
        model             : SpiralDSVLanguageModel with sem_fwd/sem_bwd built
        base_bytes        : (vocab_size,) int32 — UTF-8 byte count per token
        has_leading_space : (vocab_size,) bool — True if token has leading space
        is_boundary_token : (vocab_size,) bool — True if token is doc boundary
                            (used for coherence gating reset). Optional.
        batch_size        : Tokens per eval batch (default 500,000)
        W_COHERENCE       : Coherence gating weight (0.0 = disabled, 0.3 = default)
        verbose           : Print progress (default True)

    Returns:
        (bpb, val_loss) — bits-per-byte and mean cross-entropy loss
    """
    t0 = time.time()
    N = len(val_tokens)

    if verbose:
        print(f"\n[SpiralDSV-Eval] Evaluating BPB on {N:,} tokens "
              f"(batch_size={batch_size:,}, W_COHERENCE={W_COHERENCE})")

    # Build pm1 cache once (lazy, one-time cost)
    model._ensure_pm1_cache()
    n_bits = model.n_words * 64

    total_bits  = 0.0
    total_bytes = 0
    total_nats  = 0.0
    total_toks  = 0

    # Coherence gating state (per-document running centroid)
    use_coherence = (W_COHERENCE > 0.0 and
                     model._codebook_pm1 is not None and
                     is_boundary_token is not None)
    if use_coherence:
        coherence_pm1   = np.zeros(n_bits, dtype=np.float32)
        doc_token_count = 0

    for chunk_start in range(1, N, batch_size):
        chunk_end = min(chunk_start + batch_size, N)
        B = chunk_end - chunk_start

        prev_toks = np.clip(
            val_tokens[chunk_start - 1 : chunk_end - 1].astype(np.int32),
            0, model.vocab_size - 1,
        )
        tgt_toks = val_tokens[chunk_start:chunk_end].astype(np.int32)
        tgt_toks = np.clip(tgt_toks, 0, model.vocab_size - 1)

        if use_coherence:
            # Per-token coherence-augmented scoring (sequential — cannot batch)
            # This path is slower but provides the coherence signal.
            scores_batch = np.empty((B, model.vocab_size), dtype=np.float32)
            for i in range(B):
                prev_t = int(prev_toks[i])
                tgt_t  = int(tgt_toks[i])

                # Reset coherence on document boundary
                if is_boundary_token is not None and is_boundary_token[prev_t]:
                    coherence_pm1[:] = 0.0
                    doc_token_count  = 0

                # Update coherence with previous token's codebook vector
                coherence_pm1   += model._codebook_pm1[prev_t]
                doc_token_count += 1

                # Coherence-augmented query:
                # h*(b) = sign(sem_fwd_pm1[prev] + W_COHERENCE × coh_norm)
                coh_norm = coherence_pm1 / max(doc_token_count, 1)
                h_star   = np.sign(
                    model._sem_fwd_pm1[prev_t] + W_COHERENCE * coh_norm
                ).astype(np.float32)
                h_star[h_star == 0.0] = 1.0

                raw = (h_star @ model._codebook_pm1.T) / n_bits  # (vocab,)
                probs = np.clip(0.5 + 0.49 * raw, 1e-30, 0.99).astype(np.float32)
                probs /= probs.sum()
                scores_batch[i] = probs
        else:
            # Vectorised GPU-accelerated path (all positions in one matmul)
            scores_batch = model.vote_scores_all_vocab(
                prev_tokens=prev_toks,
                next_tokens=None,   # prev-only mode (next token unknown at eval)
            )  # (B, vocab_size) float32

        # Extract probability of correct token
        p_correct = scores_batch[np.arange(B), tgt_toks]
        p_correct = np.clip(p_correct, 1e-30, 1.0)

        # Byte count (same formula as reference train_gpt.py)
        prev_t_arr = np.clip(
            val_tokens[chunk_start - 1 : chunk_end - 1].astype(np.int32),
            0, base_bytes.shape[0] - 1,
        )
        # Exact official formula:
        #   tok_bytes = base_bytes[tgt] + (has_leading_space[tgt] & ~is_boundary_token[prev])
        # i.e. add 1 for leading space only when prev token is NOT a boundary token.
        tok_bytes = base_bytes[tgt_toks].astype(np.float64)
        if is_boundary_token is not None:
            tok_bytes += (
                has_leading_space[tgt_toks] & ~is_boundary_token[prev_t_arr]
            ).astype(np.float64)
        else:
            tok_bytes += has_leading_space[tgt_toks].astype(np.float64)

        total_bits  += float(-np.log2(p_correct).sum())
        total_bytes += int(tok_bytes.sum())
        total_nats  += float(-np.log(p_correct).sum())
        total_toks  += B

        if verbose:
            elapsed = time.time() - t0
            pct = 100.0 * chunk_end / N
            running_bpb = total_bits / max(total_bytes, 1)
            print(f"[SpiralDSV-Eval] {chunk_end:,}/{N:,} ({pct:.1f}%)  "
                  f"running_bpb={running_bpb:.4f}  elapsed={elapsed:.1f}s")

    bpb      = total_bits / max(total_bytes, 1)
    val_loss = total_nats / max(total_toks, 1)

    elapsed = time.time() - t0
    if verbose:
        print(f"[SpiralDSV-Eval] Done: BPB={bpb:.4f}  "
              f"val_loss={val_loss:.4f}  elapsed={elapsed:.1f}s")

    return float(bpb), float(val_loss)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_spiral_dsv_artifact(
    model: SpiralDSVLanguageModel,
    path: str,
    verbose: bool = True,
) -> int:
    """Save sem_fwd + sem_bwd to LZMA9-compressed .hgz artifact.

    Format (HGZ3):
        Magic(4B "HGZ3") + vocab_size(4B) + n_words(4B) + flags(4B)
        + sem_fwd bytes  (vocab_size × n_words × 8)   [8 MB for n_words=1024]
        + sem_bwd bytes  (vocab_size × n_words × 8)   [8 MB for n_words=1024]
        LZMA9 compressed → ~2–4 MB on disk

    The internal codebook is NOT stored — regenerated from seed at load time.

    Args:
        model   : SpiralDSVLanguageModel with sem_fwd/sem_bwd built
        path    : Output file path (e.g. "spiral_dsv_seed42.hgz")
        verbose : Print artifact size info

    Returns:
        Compressed artifact size in bytes.
    """
    if not getattr(model, '_built', False):
        raise ValueError("Model not built — call build_spiral_dsv() first")

    header = struct.pack(
        _HGZ3_HEADER_FMT,
        _HGZ3_MAGIC,
        model.vocab_size,
        model.n_words,
        0,  # flags (reserved)
    )
    payload = header + model.sem_fwd.tobytes() + model.sem_bwd.tobytes()

    uncompressed_mb = len(payload) / 1_000_000
    compressed = lzma.compress(payload, preset=9)
    compressed_mb = len(compressed) / 1_000_000

    with open(path, 'wb') as f:
        f.write(compressed)

    if verbose:
        print(f"[SpiralDSV] Artifact saved: {path}")
        print(f"[SpiralDSV] Uncompressed: {uncompressed_mb:.2f} MB  "
              f"Compressed (LZMA9): {compressed_mb:.2f} MB  "
              f"({len(compressed):,} bytes)")

    return len(compressed)


def load_spiral_dsv_artifact(
    path: str,
    seed: int = 42,
    verbose: bool = True,
) -> SpiralDSVLanguageModel:
    """Load sem_fwd + sem_bwd from .hgz artifact.

    Reconstructs the SpiralDSVLanguageModel with the saved tables.
    The internal codebook is regenerated from seed (deterministic).

    Args:
        path    : Path to .hgz artifact file
        seed    : Codebook seed (must match the seed used during build)
        verbose : Print load info

    Returns:
        SpiralDSVLanguageModel with sem_fwd/sem_bwd populated.
    """
    with open(path, 'rb') as f:
        data = lzma.decompress(f.read())

    magic, vocab_size, n_words, flags = struct.unpack_from(
        _HGZ3_HEADER_FMT, data, 0
    )
    if magic != _HGZ3_MAGIC:
        raise ValueError(f"Bad magic: {magic!r} (expected {_HGZ3_MAGIC!r})")

    table_bytes = vocab_size * n_words * 8  # uint64 = 8 bytes each

    sem_fwd = np.frombuffer(
        data, dtype=np.uint64,
        count=vocab_size * n_words,
        offset=_HGZ3_HEADER_SIZE,
    ).reshape(vocab_size, n_words).copy()

    sem_bwd = np.frombuffer(
        data, dtype=np.uint64,
        count=vocab_size * n_words,
        offset=_HGZ3_HEADER_SIZE + table_bytes,
    ).reshape(vocab_size, n_words).copy()

    model = SpiralDSVLanguageModel(
        vocab_size=vocab_size,
        n_words=n_words,
        seed=seed,
    )
    model.sem_fwd = sem_fwd
    model.sem_bwd = sem_bwd
    model._built  = True
    model._invalidate_pm1_cache()

    if verbose:
        print(f"[SpiralDSV] Artifact loaded: {path}")
        print(f"[SpiralDSV] vocab_size={vocab_size}, n_words={n_words}, "
              f"n_bits={n_words*64:,}, flags={flags:#010x}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser helpers (SentencePiece → base_bytes / has_leading_space)
# ─────────────────────────────────────────────────────────────────────────────

def build_token_byte_arrays(
    sp_model,
    vocab_size: int = VOCAB_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-token byte-count and leading-space arrays from SentencePiece model.

    Matches the official competition train_gpt.py formula exactly:

        is_boundary_token initialised to ALL True (ones).
        For each token:
          - byte tokens (sp.is_byte): base_bytes=1, is_boundary_token stays True, skip.
          - normal word-piece tokens: decode piece, set base_bytes=utf8_len,
            has_leading_space, then CLEAR is_boundary_token to False.
          - special/control/unknown tokens: base_bytes stays 0, is_boundary_token
            stays True (never cleared).

        Byte count per target token:
            tok_bytes = base_bytes[tgt] + (has_leading_space[tgt] & ~is_boundary_token[prev])
            i.e. add 1 for leading space ONLY when prev token is NOT a boundary token.

        BPB = Σ(-log₂ p_correct) / Σ(tok_bytes)

    Args:
        sp_model   : sentencepiece.SentencePieceProcessor instance
        vocab_size : Vocabulary size (default 1024)

    Returns:
        base_bytes        : (vocab_size,) int16 — UTF-8 byte count per token
                            (not counting leading space)
        has_leading_space : (vocab_size,) bool  — True if token has leading space
        is_boundary_token : (vocab_size,) bool  — True for byte tokens, BOS/EOS/PAD/UNK,
                            and any special/control piece. False only for normal word-pieces.
    """
    table_size = max(vocab_size, sp_model.GetPieceSize())
    base_bytes        = np.zeros(table_size, dtype=np.int16)
    has_leading_space = np.zeros(table_size, dtype=bool)
    # Official: initialise ALL tokens as boundary; only normal word-pieces are cleared.
    is_boundary_token = np.ones(table_size, dtype=bool)

    for tok_id in range(min(vocab_size, sp_model.GetPieceSize())):
        # Byte tokens: 1 byte, remain boundary (matches official is_byte branch)
        if sp_model.is_byte(tok_id):
            base_bytes[tok_id] = 1
            continue   # is_boundary_token stays True

        piece = sp_model.IdToPiece(tok_id)

        # Leading space (SentencePiece uses '▁' U+2581 for word-initial space)
        if piece.startswith('\u2581'):
            has_leading_space[tok_id] = True
            piece = piece[1:]   # strip the leading-space marker

        # UTF-8 byte count of the actual text (without leading space)
        try:
            base_bytes[tok_id] = len(piece.encode('utf-8'))
        except Exception:
            base_bytes[tok_id] = 1

        # Normal word-piece: clear boundary flag (matches official logic where
        # is_boundary_token is only True for byte/special tokens)
        is_boundary_token[tok_id] = False

    # Truncate to vocab_size (table_size may be larger than vocab_size)
    return (
        base_bytes[:vocab_size],
        has_leading_space[:vocab_size],
        is_boundary_token[:vocab_size],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Artifact size check
# ─────────────────────────────────────────────────────────────────────────────

ARTIFACT_LIMIT: int = 16_000_000   # 16 MB hard limit


def check_artifact_size(
    artifact_path: str,
    code_bytes: int = 0,
    verbose: bool = True,
) -> Tuple[int, bool]:
    """Check that artifact + code fits within the 16 MB limit.

    Args:
        artifact_path : Path to the .hgz artifact file
        code_bytes    : Total size of all Python source files in bytes
        verbose       : Print size report

    Returns:
        (total_bytes, passes) — total size and whether it passes the limit
    """
    import os
    artifact_bytes = os.path.getsize(artifact_path)
    total_bytes    = artifact_bytes + code_bytes
    passes         = total_bytes <= ARTIFACT_LIMIT

    if verbose:
        print(f"\n[SpiralDSV] Artifact size check:")
        print(f"  Artifact : {artifact_bytes:>12,} bytes  ({artifact_bytes/1e6:.3f} MB)")
        print(f"  Code     : {code_bytes:>12,} bytes  ({code_bytes/1e6:.3f} MB)")
        print(f"  Total    : {total_bytes:>12,} bytes  ({total_bytes/1e6:.3f} MB)")
        print(f"  Limit    : {ARTIFACT_LIMIT:>12,} bytes  ({ARTIFACT_LIMIT/1e6:.3f} MB)")
        print(f"  Result   : {'✅ PASS' if passes else '❌ FAIL'}")

    return total_bytes, passes
