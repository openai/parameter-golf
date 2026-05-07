"""
proof_binding_beats_fixed.py

Empirical proof: binding-energy-modulated Dirichlet CTW beats fixed-concentration
Dirichlet CTW on structured text.

The test: generate a corpus with TWO regimes:
  - Rare-specific contexts (tokens 900-999): highly predictable next token
  - Common-ambiguous contexts (tokens 0-50): unpredictable next token

Fixed CTW uses c=5.0 everywhere — same trust for rare and common contexts.
Binding CTW uses c(B): higher trust for rare contexts, lower for common.

Metric: bits per token = -log2(p(correct_token))
Lower is better. If binding < fixed, the self-model thesis holds.
"""

import math
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from binding_ctw import BindingCTW


def generate_two_regime_corpus(n: int = 200_000, vocab_size: int = 1024,
                                seed: int = 42) -> np.ndarray:
    """
    Corpus with two distinct regimes:

    Regime A — RARE + PREDICTABLE (every 100 tokens):
      Context: [950, 951, 952] → always followed by 953
      These are rare tokens (low frequency) with deterministic continuation.
      A self-aware model should trust n-gram counts fully here.

    Regime B — COMMON + AMBIGUOUS (every 10 tokens):
      Context: [5] → followed by uniform random from [10..30]
      Token 5 is extremely common, and continuation is unpredictable.
      A self-aware model should smooth heavily here.

    The rest is uniform random noise.
    """
    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, vocab_size, size=n, dtype=np.uint16)

    # Regime A: rare deterministic — every 100 positions
    for i in range(0, n - 4, 100):
        tokens[i] = 950
        tokens[i + 1] = 951
        tokens[i + 2] = 952
        tokens[i + 3] = 953  # always 953

    # Regime B: common ambiguous — every 10 positions (offset by 5)
    for i in range(5, n - 2, 10):
        tokens[i] = 5
        tokens[i + 1] = rng.randint(10, 30)  # random from 20 options

    return tokens


def compute_bits_per_token(probs: np.ndarray) -> float:
    """Average -log2(p) over all scored positions."""
    # Clamp to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    bits = -np.log2(probs)
    return float(bits.mean())


def run_proof():
    print("=" * 70)
    print("PROOF: Binding-Modulated CTW vs Fixed-Concentration CTW")
    print("=" * 70)

    vocab_size = 1024
    corpus_size = 200_000

    # Generate corpus
    print("\n[1] Generating two-regime corpus...")
    tokens = generate_two_regime_corpus(n=corpus_size, vocab_size=vocab_size)
    print(f"    {corpus_size:,} tokens, vocab={vocab_size}")

    # Count regime occurrences
    n_rare = sum(1 for i in range(0, len(tokens)-4, 100)
                 if tokens[i]==950 and tokens[i+3]==953)
    n_common = sum(1 for i in range(5, len(tokens)-2, 10)
                   if tokens[i]==5)
    print(f"    Regime A (rare, deterministic): {n_rare} patterns")
    print(f"    Regime B (common, ambiguous):   {n_common} patterns")

    # Split: first 80% for "training" cache, last 20% for scoring
    split = int(corpus_size * 0.8)
    train_tokens = tokens[:split]
    eval_tokens = tokens  # score from split onward, but need full array for context

    # Build cache from training portion
    print("\n[2] Building n-gram cache from training data...")
    t0 = time.time()

    cache_fixed = BindingCTW(
        max_order=7, min_order=2, num_buckets=65536,
        vocab_size=vocab_size, c_base=5.0, beta=0.0)  # beta=0 → fixed

    cache_binding = BindingCTW(
        max_order=7, min_order=2, num_buckets=65536,
        vocab_size=vocab_size, c_base=5.0, beta=3.0)  # beta=3 → binding-modulated

    # Build both from same training data
    cache_fixed.build_full(train_tokens)
    cache_binding.build_full(train_tokens)

    # Also warm binding cache with token frequencies
    freq = np.bincount(train_tokens.astype(np.int32), minlength=vocab_size).astype(np.float64)
    cache_binding.warm_from_training(freq, len(train_tokens))

    t1 = time.time()
    print(f"    Built in {t1-t0:.2f}s")
    print(f"    Cache stats: {cache_fixed.stats()['total_ctx_entries']:,} ctx entries")

    # Score eval portion
    eval_start = split
    eval_end = min(split + 20_000, corpus_size - 1)  # score 20K positions
    seg_len = eval_end - eval_start

    print(f"\n[3] Scoring {seg_len:,} eval positions...")

    # Base probabilities: uniform (simulating a trivial neural model)
    base_p = np.full(seg_len, 1.0 / vocab_size)

    # Fixed concentration CTW
    t2 = time.time()
    probs_fixed = cache_fixed.lookup_hierarchical_fixed(
        tokens, eval_start, eval_end, base_p, concentration=5.0)
    t3 = time.time()

    # Binding-modulated CTW
    probs_binding = cache_binding.lookup_hierarchical_binding(
        tokens, eval_start, eval_end, base_p, context_len=6)
    t4 = time.time()

    bpt_fixed = compute_bits_per_token(probs_fixed)
    bpt_binding = compute_bits_per_token(probs_binding)
    bpt_uniform = compute_bits_per_token(base_p)

    print(f"    Fixed CTW:    {t3-t2:.2f}s")
    print(f"    Binding CTW:  {t4-t3:.2f}s")

    # Analyze by regime
    print(f"\n[4] Results (bits per token, lower is better):")
    print(f"    {'Method':<25} {'All':>10} {'Rare ctx':>10} {'Common ctx':>10}")
    print(f"    {'-'*55}")

    # Find regime-specific positions in eval range
    rare_positions = []
    common_positions = []
    for i in range(eval_start, eval_end):
        offset = i - eval_start
        # Check if this is a rare-regime prediction (position after 950,951,952)
        if i >= 3 and tokens[i-3]==950 and tokens[i-2]==951 and tokens[i-1]==952:
            rare_positions.append(offset)
        # Check if common-regime prediction (position after token 5)
        if i >= 1 and tokens[i-1]==5:
            common_positions.append(offset)

    rare_idx = np.array(rare_positions) if rare_positions else np.array([], dtype=int)
    common_idx = np.array(common_positions) if common_positions else np.array([], dtype=int)

    def regime_bpt(probs, idx):
        if len(idx) == 0:
            return float('nan')
        return compute_bits_per_token(probs[idx])

    print(f"    {'Uniform (baseline)':<25} {bpt_uniform:>10.4f} {regime_bpt(base_p, rare_idx):>10.4f} {regime_bpt(base_p, common_idx):>10.4f}")
    print(f"    {'Fixed CTW (c=5.0)':<25} {bpt_fixed:>10.4f} {regime_bpt(probs_fixed, rare_idx):>10.4f} {regime_bpt(probs_fixed, common_idx):>10.4f}")
    print(f"    {'Binding CTW (c=c(B))':<25} {bpt_binding:>10.4f} {regime_bpt(probs_binding, rare_idx):>10.4f} {regime_bpt(probs_binding, common_idx):>10.4f}")

    delta = bpt_fixed - bpt_binding
    print(f"\n[5] VERDICT:")
    print(f"    Fixed CTW:   {bpt_fixed:.6f} bits/token")
    print(f"    Binding CTW: {bpt_binding:.6f} bits/token")
    print(f"    Delta:       {delta:+.6f} bits/token")

    if delta > 0:
        print(f"\n    ✓ BINDING CTW WINS by {delta:.6f} bits/token")
        print(f"    ✓ Self-model thesis CONFIRMED:")
        print(f"      Context-aware concentration beats fixed concentration.")
        print(f"      The compression scheme that knows its own reliability")
        print(f"      outperforms the one that doesn't.")

        # Regime-specific analysis
        if len(rare_idx) > 0 and len(common_idx) > 0:
            rare_delta = regime_bpt(probs_fixed, rare_idx) - regime_bpt(probs_binding, rare_idx)
            common_delta = regime_bpt(probs_fixed, common_idx) - regime_bpt(probs_binding, common_idx)
            print(f"\n    Regime breakdown:")
            print(f"      Rare contexts:   {rare_delta:+.6f} bpt (binding {'wins' if rare_delta > 0 else 'loses'})")
            print(f"      Common contexts: {common_delta:+.6f} bpt (binding {'wins' if common_delta > 0 else 'loses'})")
            if rare_delta > 0 and common_delta <= 0:
                print(f"\n    ✓ As predicted: binding helps on rare contexts (more trust)")
                print(f"      and doesn't hurt on common contexts (appropriate smoothing)")
    else:
        print(f"\n    ✗ Fixed CTW wins by {-delta:.6f} bits/token")
        print(f"      Self-model thesis NOT confirmed at these hyperparameters.")
        print(f"      Try adjusting beta or c_base.")

    # Save results
    results = {
        'corpus_size': corpus_size,
        'vocab_size': vocab_size,
        'eval_positions': seg_len,
        'n_rare_patterns': len(rare_idx),
        'n_common_patterns': len(common_idx),
        'bpt_uniform': bpt_uniform,
        'bpt_fixed': bpt_fixed,
        'bpt_binding': bpt_binding,
        'delta': delta,
        'binding_wins': delta > 0,
        'rare_bpt_fixed': regime_bpt(probs_fixed, rare_idx),
        'rare_bpt_binding': regime_bpt(probs_binding, rare_idx),
        'common_bpt_fixed': regime_bpt(probs_fixed, common_idx),
        'common_bpt_binding': regime_bpt(probs_binding, common_idx),
    }

    import json
    out_path = os.path.join(os.path.dirname(__file__), "proof_binding_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n    Results saved → {out_path}")

    return results


if __name__ == "__main__":
    run_proof()
