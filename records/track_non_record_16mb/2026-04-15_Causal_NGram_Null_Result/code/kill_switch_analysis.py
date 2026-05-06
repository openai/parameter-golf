"""
Kill-switch analysis for the causal n-gram approach.

Answers: does the sp1024 FineWeb val set have enough long-range n-gram
repetition to justify an eval-time cache, or should we pivot?

Numerical GO/NO-GO gates (my thresholds):
  GO: at least 5% of scored tokens are in positions where a confident
      order-4 match exists in history AND that match is > 2048 tokens back
      (i.e., outside the neural attention window).
  GO: theoretical BPB upper bound (assuming cache predicts those positions
      perfectly) > 0.003 nats.
  NO-GO: otherwise.

Reads sp1024 val shard directly; no torch, no model, no pod.
"""
from __future__ import annotations
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_val_tokens(path: Path) -> np.ndarray:
    """Load a challenge .bin shard: 256 int32 header + uint16 tokens."""
    header_bytes = 256 * 4
    hdr = np.fromfile(path, dtype='<i4', count=256)
    tokens = np.fromfile(path, dtype='<u2', offset=header_bytes)
    return tokens, hdr


def segment_documents(tokens: np.ndarray, eos_id: int) -> list[np.ndarray]:
    """Split the token stream by EOS. Returns list of document token arrays
    (without the trailing EOS)."""
    boundaries = np.nonzero(tokens == eos_id)[0]
    docs = []
    start = 0
    for b in boundaries:
        if b > start:
            docs.append(tokens[start:b])
        start = b + 1
    if start < len(tokens):
        docs.append(tokens[start:])
    return docs


def analyze_doc_lengths(docs: list[np.ndarray]) -> dict:
    lens = np.array([len(d) for d in docs])
    total_tokens = int(lens.sum())
    return {
        "num_docs": len(docs),
        "total_tokens": total_tokens,
        "mean_len": float(lens.mean()),
        "median_len": float(np.median(lens)),
        "p90_len": float(np.percentile(lens, 90)),
        "p99_len": float(np.percentile(lens, 99)),
        "max_len": int(lens.max()),
        "tokens_in_docs_gt_2048": int(lens[lens > 2048].sum()),
        "frac_tokens_in_docs_gt_2048": float(lens[lens > 2048].sum() / total_tokens),
        "tokens_in_docs_gt_4096": int(lens[lens > 4096].sum()),
        "frac_tokens_in_docs_gt_4096": float(lens[lens > 4096].sum() / total_tokens),
        "tokens_beyond_2048_in_long_docs": int(np.maximum(lens - 2048, 0).sum()),
        "frac_tokens_beyond_2048": float(np.maximum(lens - 2048, 0).sum() / total_tokens),
    }


def analyze_ngram_repetition(docs: list[np.ndarray], order: int,
                              max_docs: int | None = None,
                              report_every: int = 5000) -> dict:
    """For each scored token position (t) in each doc, check whether the order-K
    context (x_{t-K+1}..x_{t-1}) has been seen before at position p < t, and
    whether the (context, x_t) pair was observed (i.e., would the cache predict
    x_t exactly).

    Metrics:
      - hit_rate: % of positions where order-K context was seen earlier
      - correct_hit_rate: % of positions where order-K context was seen AND
        the majority predicted token matches x_t (cache would be "right")
      - longrange_hit_rate: % of positions where context was seen earlier
        AND the earliest match is > 2048 tokens back (outside neural window)
      - longrange_correct_rate: longrange hit AND majority matches x_t
      - mass_on_target: sum of p_cache(x_t) across all positions (a proxy
        for the theoretical BPB upper bound gain)
    """
    if max_docs is not None:
        docs = docs[:max_docs]

    positions_total = 0
    hit_positions = 0
    correct_hits = 0
    longrange_positions = 0
    longrange_hits = 0
    longrange_correct = 0
    # Sum of log p_cache(x_t) when cache had a hit (smoothed)
    sum_log_p_cache = 0.0
    # Sum of log p_uniform(x_t) across the same positions for baseline
    vocab_size_approx = 1024  # sp1024
    log_uniform = -math.log(vocab_size_approx)

    for d_idx, doc in enumerate(docs):
        # Per-doc cache: counts[context_tuple] -> Counter({next_token: count, ...})
        # Also store the FIRST position where the context was seen (for long-range check)
        cache: dict = {}
        first_pos: dict = {}
        dl = len(doc)
        for t in range(dl):
            positions_total += 1
            if t < order - 1:
                continue  # not enough history for the context
            ctx = tuple(int(x) for x in doc[t - (order - 1):t])
            tgt = int(doc[t])
            if ctx in cache:
                hit_positions += 1
                counter = cache[ctx]
                total = sum(counter.values())
                # MLE prediction with add-1 smoothing
                c = counter.get(tgt, 0)
                p_tgt = (c + 1) / (total + vocab_size_approx)
                sum_log_p_cache += math.log(p_tgt)

                most_common = counter.most_common(1)[0][0]
                if most_common == tgt:
                    correct_hits += 1

                # Long-range check: earliest observation of this context
                earliest = first_pos[ctx]
                if (t - earliest) > 2048:
                    longrange_positions += 1
                    if most_common == tgt:
                        longrange_correct += 1
            # Update the cache with this (context, token) observation
            if ctx not in cache:
                cache[ctx] = Counter()
                first_pos[ctx] = t
            cache[ctx][tgt] += 1

        if (d_idx + 1) % report_every == 0:
            print(f"  ... processed {d_idx + 1}/{len(docs)} docs "
                  f"({positions_total} positions, hits={hit_positions})",
                  file=sys.stderr)

    # Average log-prob for hit positions
    mean_log_p_cache_hits = (sum_log_p_cache / hit_positions) if hit_positions else 0.0

    # Theoretical BPB upper bound assuming cache always correct on "correct hits"
    # ... this is imprecise because we don't know the neural model's p at those
    # positions. Report the RAW entropy reduction available as a proxy:
    #   BPB saved ~= (hit_positions / positions_total) * (mean_log_p_cache_hits - log_uniform) / log(2)
    # This is nats converted to bits-per-token.

    if hit_positions:
        bpt_savings_vs_uniform = (
            (hit_positions / positions_total) *
            (mean_log_p_cache_hits - log_uniform) / math.log(2)
        )
    else:
        bpt_savings_vs_uniform = 0.0

    return {
        "order": order,
        "positions_total": positions_total,
        "hit_positions": hit_positions,
        "hit_rate": hit_positions / max(positions_total, 1),
        "correct_hits": correct_hits,
        "correct_hit_rate": correct_hits / max(positions_total, 1),
        "correct_rate_given_hit": correct_hits / max(hit_positions, 1),
        "longrange_positions": longrange_positions,
        "longrange_rate": longrange_positions / max(positions_total, 1),
        "longrange_correct": longrange_correct,
        "longrange_correct_rate": longrange_correct / max(positions_total, 1),
        "mean_log_p_cache_on_hit": mean_log_p_cache_hits,
        "bpt_upper_bound_vs_uniform_bits": bpt_savings_vs_uniform,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path,
                    default=Path("data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"))
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Limit docs for faster iteration")
    ap.add_argument("--orders", type=str, default="3,4,5",
                    help="Comma-separated orders to check")
    args = ap.parse_args()

    if not args.val.exists():
        print(f"Missing val shard: {args.val}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading val shard {args.val}...", file=sys.stderr)
    tokens, hdr = load_val_tokens(args.val)
    print(f"  {len(tokens):,} tokens", file=sys.stderr)

    # FineWeb .bin uses BOS (id 1) as the document separator — empirical check
    # showed id=1 appears ~870x per 1M tokens (~1148-token docs, matches FineWeb
    # median) while id=2 (</s>) has zero occurrences.
    bos = 1
    docs = segment_documents(tokens, eos_id=bos)
    print(f"  {len(docs):,} documents", file=sys.stderr)

    doc_stats = analyze_doc_lengths(docs)
    print("\n=== DOCUMENT LENGTH STATS ===")
    for k, v in doc_stats.items():
        if isinstance(v, float) and "frac" in k:
            print(f"  {k}: {v:.4%}")
        elif isinstance(v, float):
            print(f"  {k}: {v:,.1f}")
        else:
            print(f"  {k}: {v:,}")

    orders = [int(x) for x in args.orders.split(",")]
    for order in orders:
        print(f"\n=== ORDER-{order} REPETITION ANALYSIS (per-doc cache) ===")
        stats = analyze_ngram_repetition(docs, order=order, max_docs=args.max_docs)
        for k, v in stats.items():
            if "rate" in k or "frac" in k:
                print(f"  {k}: {v:.4%}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v:,}")

        # GO/NO-GO interpretation
        print()
        go_crit_1 = stats["longrange_rate"] >= 0.05
        go_crit_2 = stats["bpt_upper_bound_vs_uniform_bits"] >= 0.003
        print(f"  GO criterion 1 (longrange_rate >= 5%): "
              f"{'PASS' if go_crit_1 else 'FAIL'} ({stats['longrange_rate']:.2%})")
        print(f"  GO criterion 2 (bpt upper bound vs uniform >= 0.003): "
              f"{'PASS' if go_crit_2 else 'FAIL'} "
              f"({stats['bpt_upper_bound_vs_uniform_bits']:.6f} bits)")


if __name__ == "__main__":
    main()
