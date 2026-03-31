#!/usr/bin/env python3
"""
Tier 6 — Stage 2a: Fixed-alpha mixture sweep.

Mixes neural model probabilities with PPM context mixer probabilities
at various alpha values and reports BPB for each.

  p_mix(token) = alpha * p_neural(token) + (1 - alpha) * p_mixer(token)
  bpb = sum(-log2(p_mix)) / total_bytes

Requires:
  - Pre-computed neural log-probs from dump_neural_logprobs.py
  - Validation data shards + tokenizer (for PPM mixer + byte accounting)

Usage:
  python3 experiments/tier6/stage2a_mixture_sweep.py \
      --neural-logprobs experiments/tier6/neural_logprobs.npz
"""
from __future__ import annotations

import argparse
import glob
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_shard(file: Path) -> np.ndarray:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    return np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)


def load_validation_tokens(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    return np.concatenate([load_data_shard(Path(f)) for f in files])


def build_byte_lut(sp: spm.SentencePieceProcessor, vocab_size: int):
    base_bytes = np.zeros(vocab_size, dtype=np.int16)
    has_leading_space = np.zeros(vocab_size, dtype=np.bool_)
    is_boundary = np.ones(vocab_size, dtype=np.bool_)
    for tid in range(min(int(sp.vocab_size()), vocab_size)):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary


# ---------------------------------------------------------------------------
# PPM Context Mixer (same as Stage 1)
# ---------------------------------------------------------------------------

class PPMContextMixer:
    def __init__(self, max_order: int, vocab_size: int):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.counts: list[dict] = [defaultdict(lambda: defaultdict(int)) for _ in range(max_order + 1)]
        self.totals: list[dict] = [defaultdict(int) for _ in range(max_order + 1)]
        self.uniques: list[dict] = [defaultdict(int) for _ in range(max_order + 1)]

    def reset(self):
        for k in range(self.max_order + 1):
            self.counts[k].clear()
            self.totals[k].clear()
            self.uniques[k].clear()

    def update(self, context: tuple, symbol: int):
        for k in range(min(len(context), self.max_order), -1, -1):
            ctx = context[-k:] if k > 0 else ()
            c = self.counts[k][ctx]
            if symbol not in c:
                self.uniques[k][ctx] += 1
            c[symbol] += 1
            self.totals[k][ctx] += 1

    def predict(self, context: tuple, symbol: int) -> float:
        prob = 0.0
        escape_product = 1.0
        for k in range(min(len(context), self.max_order), -1, -1):
            ctx = context[-k:] if k > 0 else ()
            total = self.totals[k].get(ctx, 0)
            unique = self.uniques[k].get(ctx, 0)
            if total == 0:
                continue
            denom = total + unique
            sym_count = self.counts[k][ctx].get(symbol, 0)
            if sym_count > 0:
                prob += escape_product * (sym_count / denom)
            escape_product *= unique / denom
        prob += escape_product * (1.0 / self.vocab_size)
        return prob


# ---------------------------------------------------------------------------
# Core: run mixer and compute per-token log-probs
# ---------------------------------------------------------------------------

def compute_mixer_logprobs(
    tokens: np.ndarray,
    seq_len: int,
    max_order: int,
    vocab_size: int,
    cumulative: bool,
) -> np.ndarray:
    """Return array of -log2(p_mixer) for each target token position."""
    mixer = PPMContextMixer(max_order, vocab_size)
    total_seqs = (len(tokens) - 1) // seq_len
    all_neg_log2_p = np.zeros(total_seqs * seq_len, dtype=np.float32)

    t0 = time.perf_counter()
    idx = 0
    for seq_idx in range(total_seqs):
        if not cumulative:
            mixer.reset()
        start = seq_idx * seq_len
        seq = tokens[start: start + seq_len + 1]
        for t in range(seq_len):
            target_id = int(seq[t + 1])
            ctx_start = max(0, t + 1 - max_order)
            context = tuple(int(seq[i]) for i in range(ctx_start, t + 1))
            p = mixer.predict(context, target_id)
            if p <= 0:
                p = 1e-12
            all_neg_log2_p[idx] = -math.log2(p)
            mixer.update(context, target_id)
            idx += 1
        if (seq_idx + 1) % 5000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  mixer: {seq_idx + 1}/{total_seqs} seqs  {elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"  mixer done: {total_seqs} seqs in {elapsed:.1f}s")
    return all_neg_log2_p


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_alpha(
    neural_neg_log2_p: np.ndarray,
    mixer_neg_log2_p: np.ndarray,
    target_ids: np.ndarray,
    prev_ids: np.ndarray,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary: np.ndarray,
    alphas: list[float],
) -> list[tuple[float, float]]:
    """
    Sweep mixture weights. Returns list of (alpha, bpb).

    p_mix = alpha * p_neural + (1-alpha) * p_mixer
    """
    # Precompute byte counts for all positions
    token_bytes = base_bytes[target_ids].astype(np.float64)
    # Add leading space byte when applicable
    space_mask = has_leading_space[target_ids] & ~is_boundary[prev_ids]
    token_bytes += space_mask.astype(np.float64)
    total_bytes = token_bytes.sum()

    # Convert -log2(p) back to probabilities
    # Clip to avoid overflow: -log2(p) > 30 means p < 1e-9
    neural_p = np.power(2.0, -np.clip(neural_neg_log2_p, 0, 50).astype(np.float64))
    mixer_p = np.power(2.0, -np.clip(mixer_neg_log2_p, 0, 50).astype(np.float64))

    results = []
    for alpha in alphas:
        mixed_p = alpha * neural_p + (1.0 - alpha) * mixer_p
        mixed_p = np.maximum(mixed_p, 1e-30)
        neg_log2_mixed = -np.log2(mixed_p)
        bpb = float((neg_log2_mixed * 1.0).sum() / total_bytes)
        # Also compute weighted version (proper BPB)
        # Wait - BPB = total_bits / total_bytes where total_bits = sum(-log2(p))
        # and total_bytes = sum(bytes_per_token). The bits are NOT weighted by bytes.
        bpb = float(neg_log2_mixed.sum() / total_bytes)
        results.append((alpha, bpb))
        print(f"  alpha={alpha:.3f}  bpb={bpb:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2a: Mixture sweep")
    parser.add_argument("--neural-logprobs", required=True,
                        help="Path to neural_logprobs.npz from dump script")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--max-seqs", type=int, default=0)
    args = parser.parse_args()

    print("=== Tier 6 Stage 2a: Fixed-alpha Mixture Sweep ===")

    # Load neural log-probs
    print("Loading neural log-probs...")
    data = np.load(args.neural_logprobs)
    neural_neg_log2_p = data["neg_log2_p"]
    target_ids = data["target_ids"]
    prev_ids = data["prev_ids"]
    saved_seq_len = int(data["seq_len"])
    total_tokens = len(neural_neg_log2_p)
    total_seqs = total_tokens // saved_seq_len
    print(f"  {total_tokens} tokens, {total_seqs} sequences, seq_len={saved_seq_len}")

    if saved_seq_len != args.seq_len:
        raise ValueError(
            f"Seq-len mismatch: NPZ was dumped with seq_len={saved_seq_len} "
            f"but --seq-len={args.seq_len}. These must match for token alignment."
        )

    if args.max_seqs > 0 and args.max_seqs < total_seqs:
        limit = args.max_seqs * saved_seq_len
        neural_neg_log2_p = neural_neg_log2_p[:limit]
        target_ids = target_ids[:limit]
        prev_ids = prev_ids[:limit]
        total_seqs = args.max_seqs
        total_tokens = limit
        print(f"  truncated to {total_seqs} sequences")

    # Neural-only BPB for reference
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes, has_leading_space, is_boundary = build_byte_lut(sp, args.vocab_size)

    token_byte_counts = base_bytes[target_ids].astype(np.float64)
    space_mask = has_leading_space[target_ids] & ~is_boundary[prev_ids]
    token_byte_counts += space_mask.astype(np.float64)
    total_bytes = token_byte_counts.sum()
    neural_bpb = float(neural_neg_log2_p.sum() / total_bytes)
    print(f"  neural-only BPB: {neural_bpb:.4f}")

    # Load validation tokens for mixer
    val_pattern = f"{args.data_path}/fineweb_val_*.bin"
    tokens = load_validation_tokens(val_pattern)
    if args.max_seqs > 0:
        tokens = tokens[:args.max_seqs * args.seq_len + 1]

    # Alpha grid
    alphas = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90, 0.85, 0.80, 0.70, 0.50, 0.0]

    # --- Per-document mixer ---
    print(f"\n--- Per-document PPM (order {args.max_order}) + neural mixture ---")
    mixer_pd = compute_mixer_logprobs(
        tokens, args.seq_len, args.max_order, args.vocab_size, cumulative=False
    )
    if len(mixer_pd) > total_tokens:
        mixer_pd = mixer_pd[:total_tokens]
    print("\nAlpha sweep (per-doc):")
    results_pd = sweep_alpha(
        neural_neg_log2_p, mixer_pd, target_ids, prev_ids,
        base_bytes, has_leading_space, is_boundary, alphas
    )

    # --- Cumulative mixer ---
    print(f"\n--- Cumulative PPM (order {args.max_order}) + neural mixture ---")
    mixer_cum = compute_mixer_logprobs(
        tokens, args.seq_len, args.max_order, args.vocab_size, cumulative=True
    )
    if len(mixer_cum) > total_tokens:
        mixer_cum = mixer_cum[:total_tokens]
    print("\nAlpha sweep (cumulative):")
    results_cum = sweep_alpha(
        neural_neg_log2_p, mixer_cum, target_ids, prev_ids,
        base_bytes, has_leading_space, is_boundary, alphas
    )

    # --- Summary ---
    best_pd = min(results_pd, key=lambda x: x[1])
    best_cum = min(results_cum, key=lambda x: x[1])

    print(f"\n{'='*60}")
    print(f"Neural only:         {neural_bpb:.4f} BPB  (alpha=1.0)")
    print(f"Best per-doc mix:    {best_pd[1]:.4f} BPB  (alpha={best_pd[0]:.3f})")
    print(f"Best cumulative mix: {best_cum[1]:.4f} BPB  (alpha={best_cum[0]:.3f})")
    print(f"Per-doc gain:        {neural_bpb - best_pd[1]:.4f} BPB")
    print(f"Cumulative gain:     {neural_bpb - best_cum[1]:.4f} BPB")
    print(f"{'='*60}")

    # Note: challenge acceptance threshold is 0.005 nats on val_loss, not BPB.
    # BPB and val_loss are related but not identical (BPB = val_loss / ln(2) * tokens/bytes).
    # We report BPB gain here for directional go/no-go; final submission must verify in nats.
    bpb_threshold = 0.003  # conservative directional threshold
    if best_pd[1] < neural_bpb - bpb_threshold:
        print(f"GO: Per-doc mixture beats neural by >{bpb_threshold} BPB. Proceed to Stage 2b.")
    elif best_cum[1] < neural_bpb - bpb_threshold:
        print(f"GO (cumulative only): Cumulative mixture beats neural by >{bpb_threshold} BPB.")
    else:
        print(f"MARGINAL: Mixture gain < {bpb_threshold} BPB. Consider Stage 2b gating or stop.")


if __name__ == "__main__":
    main()
