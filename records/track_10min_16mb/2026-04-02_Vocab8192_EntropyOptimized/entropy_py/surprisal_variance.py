"""
Surprisal Variance Analysis
=============================

Measures the DISTRIBUTION of pointwise mutual information at each lag,
not just the average. This reveals whether dependencies at each distance
are consistent (every token pair somewhat correlated) or sparse (most
pairs uncorrelated but rare pairs strongly correlated).

Key insight: average MI at lag 100 might be 0.02 bits, suggesting
"don't bother with long range." But if the VARIANCE is high, it means
a few token pairs at lag 100 have PMI of 5+ bits (e.g., pronoun
resolution, topic callbacks). Those rare strong dependencies justify
keeping a global attention head even when the average MI is tiny.

USAGE:
  # Analyze a single shard:
  python surprisal_variance.py \
      --shard ./data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin \
      --vocab-size 8192 \
      --max-tokens 1000000 \
      --output surprisal_results.json

  # Compare both tokenizations:
  python surprisal_variance.py \
      --shard ./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin \
      --vocab-size 1024 \
      --shard-2 ./data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin \
      --vocab-size-2 8192 \
      --max-tokens 500000 \
      --output surprisal_comparison.json
"""

import argparse
import json
import math
import time

import numpy as np


def load_shard(file_path, max_tokens=None):
    """Load tokens from a FineWeb binary shard."""
    header = np.fromfile(file_path, dtype='<i4', count=256)
    magic, version, num_tokens = int(header[0]), int(header[1]), int(header[2])
    if magic != 20240520 or version != 1:
        raise ValueError(f"Bad shard header in {file_path}")
    header_bytes = 256 * np.dtype('<i4').itemsize
    tokens = np.fromfile(file_path, dtype='<u2', count=num_tokens, offset=header_bytes)
    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


def compute_pmi_distribution(tokens, lag, vocab_size, max_samples=500000):
    """
    Compute pointwise mutual information for each token pair at the given lag.
    
    PMI(x, y) = log2(P(x,y) / (P(x) * P(y)))
    
    Returns the array of PMI values and summary statistics.
    """
    n = len(tokens) - lag
    if n <= 0:
        return None
    
    x = tokens[lag:].astype(np.int64)
    y = tokens[:n].astype(np.int64)
    
    # Subsample if too many pairs
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
    else:
        x_sample = x
        y_sample = y
    
    # Build count tables from FULL data (for accurate probabilities)
    joint = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    np.add.at(joint, (x, y), 1)
    
    mx = joint.sum(axis=1).astype(np.float64)
    my = joint.sum(axis=0).astype(np.float64)
    
    # Compute PMI for each sampled pair
    n_total = float(n)
    pmi_values = np.zeros(len(x_sample), dtype=np.float64)
    
    for i in range(len(x_sample)):
        xi, yi = int(x_sample[i]), int(y_sample[i])
        p_xy = joint[xi, yi] / n_total
        p_x = mx[xi] / n_total
        p_y = my[yi] / n_total
        
        if p_xy > 0 and p_x > 0 and p_y > 0:
            pmi_values[i] = math.log2(p_xy / (p_x * p_y))
        else:
            pmi_values[i] = 0.0
    
    # Compute MI (the mean of PMI, weighted by joint probability)
    # Note: mean of PMI samples ≈ MI when samples are drawn from P(x,y)
    mi = float(np.mean(pmi_values))
    
    # Distribution statistics
    stats = {
        'lag': lag,
        'mi': round(mi, 4),
        'mean_pmi': round(float(np.mean(pmi_values)), 4),
        'std_pmi': round(float(np.std(pmi_values)), 4),
        'variance_pmi': round(float(np.var(pmi_values)), 4),
        'median_pmi': round(float(np.median(pmi_values)), 4),
        'skewness': round(float(skewness(pmi_values)), 4),
        'kurtosis': round(float(kurtosis(pmi_values)), 4),
        'pct_negative': round(float(np.mean(pmi_values < 0) * 100), 1),
        'pct_strong_pos': round(float(np.mean(pmi_values > 2.0) * 100), 2),
        'pct_very_strong': round(float(np.mean(pmi_values > 5.0) * 100), 3),
        'p99': round(float(np.percentile(pmi_values, 99)), 4),
        'p999': round(float(np.percentile(pmi_values, 99.9)), 4),
        'max_pmi': round(float(np.max(pmi_values)), 4),
        'min_pmi': round(float(np.min(pmi_values)), 4),
        'n_samples': len(pmi_values),
    }
    
    return stats


def skewness(x):
    """Compute skewness of an array."""
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def kurtosis(x):
    """Compute excess kurtosis of an array."""
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def run_analysis(tokens, vocab_size, label="", quick=False):
    """Run full surprisal variance analysis across lags."""
    
    if quick:
        lags = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    else:
        lags = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    
    lags = [l for l in lags if l < len(tokens) // 2]
    
    print(f"\n  Analyzing {len(tokens):,d} tokens at {len(lags)} lag distances...")
    print(f"\n  {'Lag':>5s}  {'MI':>7s}  {'Std':>7s}  {'Skew':>6s}  {'Kurt':>6s}  "
          f"{'%neg':>5s}  {'>2bit':>6s}  {'>5bit':>6s}  {'P99':>7s}  {'Max':>7s}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  "
          f"{'-'*5}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")
    
    all_stats = []
    t0 = time.time()
    
    for i, lag in enumerate(lags):
        stats = compute_pmi_distribution(tokens, lag, vocab_size)
        if stats is None:
            continue
        
        all_stats.append(stats)
        
        print(f"  {lag:>5d}  {stats['mi']:>7.4f}  {stats['std_pmi']:>7.4f}  "
              f"{stats['skewness']:>6.2f}  {stats['kurtosis']:>6.1f}  "
              f"{stats['pct_negative']:>4.1f}%  {stats['pct_strong_pos']:>5.2f}%  "
              f"{stats['pct_very_strong']:>5.3f}%  {stats['p99']:>7.3f}  "
              f"{stats['max_pmi']:>7.2f}")
        
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(lags) - i - 1)
            print(f"         [{i+1}/{len(lags)} done, ~{remaining:.0f}s remaining]")
    
    total_time = time.time() - t0
    print(f"\n  Analysis time: {total_time:.1f}s")
    
    return all_stats


def print_interpretation(all_stats, label=""):
    """Print interpretation of the surprisal variance results."""
    
    print(f"\n  {'='*60}")
    print(f"  INTERPRETATION: {label}")
    print(f"  {'='*60}")
    
    # Check if long-range dependencies are sparse-but-strong
    short_lags = [s for s in all_stats if s['lag'] <= 8]
    long_lags = [s for s in all_stats if s['lag'] >= 64]
    
    if short_lags and long_lags:
        avg_short_std = np.mean([s['std_pmi'] for s in short_lags])
        avg_long_std = np.mean([s['std_pmi'] for s in long_lags])
        avg_short_mi = np.mean([s['mi'] for s in short_lags])
        avg_long_mi = np.mean([s['mi'] for s in long_lags])
        
        # Coefficient of variation: std/mean — higher = more variable
        cv_short = avg_short_std / avg_short_mi if avg_short_mi > 0 else 0
        cv_long = avg_long_std / avg_long_mi if avg_long_mi > 0 else 0
        
        print(f"\n  Short-range (lag 1-8):")
        print(f"    Mean MI:  {avg_short_mi:.4f} bits")
        print(f"    Mean Std: {avg_short_std:.4f}")
        print(f"    CV (std/mean): {cv_short:.2f}")
        
        print(f"\n  Long-range (lag 64+):")
        print(f"    Mean MI:  {avg_long_mi:.4f} bits")
        print(f"    Mean Std: {avg_long_std:.4f}")
        print(f"    CV (std/mean): {cv_long:.2f}")
        
        if cv_long > cv_short * 1.5:
            print(f"\n  → Long-range dependencies are SPARSER but relatively STRONGER")
            print(f"    than average MI suggests. The CV ratio is {cv_long/cv_short:.1f}x.")
            print(f"    This supports keeping at least one global attention head.")
        elif cv_long > cv_short:
            print(f"\n  → Long-range dependencies are somewhat more variable.")
            print(f"    CV ratio: {cv_long/cv_short:.1f}x. Moderate case for global heads.")
        else:
            print(f"\n  → Dependencies are similarly distributed at all ranges.")
            print(f"    Head allocation can follow the MI histogram directly.")
    
    # Check tail behavior
    if long_lags:
        avg_p99 = np.mean([s['p99'] for s in long_lags])
        avg_p999 = np.mean([s['p999'] for s in long_lags])
        avg_strong = np.mean([s['pct_strong_pos'] for s in long_lags])
        
        print(f"\n  Tail behavior at long range:")
        print(f"    Average 99th percentile PMI: {avg_p99:.3f} bits")
        print(f"    Average 99.9th percentile PMI: {avg_p999:.3f} bits")
        print(f"    % of pairs with PMI > 2 bits: {avg_strong:.2f}%")
        
        if avg_p999 > 3.0:
            print(f"    → Heavy tail: rare but very informative long-range pairs exist.")
            print(f"       At least one global head is strongly justified.")
        elif avg_p999 > 1.5:
            print(f"    → Moderate tail: some informative long-range pairs.")
        else:
            print(f"    → Light tail: long-range dependencies are mostly weak.")


def main():
    parser = argparse.ArgumentParser(description="Surprisal variance analysis")
    parser.add_argument('--shard', type=str, required=True,
                       help='Path to primary binary shard')
    parser.add_argument('--vocab-size', type=int, required=True,
                       help='Vocabulary size for primary shard')
    parser.add_argument('--shard-2', type=str, default=None,
                       help='Path to second binary shard (for comparison)')
    parser.add_argument('--vocab-size-2', type=int, default=None,
                       help='Vocabulary size for second shard')
    parser.add_argument('--max-tokens', type=int, default=500_000,
                       help='Maximum tokens to analyze')
    parser.add_argument('--quick', action='store_true',
                       help='Fewer lag points for speed')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    print("=" * 65)
    print("SURPRISAL VARIANCE ANALYSIS")
    print("  Measuring the distribution of pointwise MI at each lag")
    print("=" * 65)
    
    all_results = {}
    
    # Primary shard
    for label, shard_path, vocab_size in [
        (f"vocab_{args.vocab_size}", args.shard, args.vocab_size),
        (f"vocab_{args.vocab_size_2}", args.shard_2, args.vocab_size_2),
    ]:
        if not shard_path:
            continue
        
        print(f"\n{'='*65}")
        print(f"  {label}: {shard_path}")
        print(f"{'='*65}")
        
        tokens = load_shard(shard_path, args.max_tokens)
        unique = len(np.unique(tokens))
        print(f"  Tokens: {len(tokens):,d}  Unique: {unique}  Vocab: {vocab_size}")
        
        stats = run_analysis(tokens, vocab_size, label, quick=args.quick)
        print_interpretation(stats, label)
        
        all_results[label] = stats
    
    # Side-by-side comparison
    keys = list(all_results.keys())
    if len(keys) == 2:
        print(f"\n{'='*65}")
        print(f"  COMPARISON: {keys[0]} vs {keys[1]}")
        print(f"{'='*65}")
        
        stats_a = {s['lag']: s for s in all_results[keys[0]]}
        stats_b = {s['lag']: s for s in all_results[keys[1]]}
        common_lags = sorted(set(stats_a.keys()) & set(stats_b.keys()))
        
        print(f"\n  {'Lag':>5s}  {'--- ' + keys[0] + ' ---':^24s}  {'--- ' + keys[1] + ' ---':^24s}")
        print(f"  {'':>5s}  {'MI':>7s} {'Std':>7s} {'CV':>6s} {'>2b%':>6s}  "
              f"{'MI':>7s} {'Std':>7s} {'CV':>6s} {'>2b%':>6s}")
        print(f"  {'-'*5}  {'-'*7} {'-'*7} {'-'*6} {'-'*6}  "
              f"{'-'*7} {'-'*7} {'-'*6} {'-'*6}")
        
        for lag in common_lags:
            a = stats_a[lag]
            b = stats_b[lag]
            cv_a = a['std_pmi'] / a['mi'] if a['mi'] > 0 else 0
            cv_b = b['std_pmi'] / b['mi'] if b['mi'] > 0 else 0
            
            print(f"  {lag:>5d}  {a['mi']:>7.4f} {a['std_pmi']:>7.4f} {cv_a:>6.2f} "
                  f"{a['pct_strong_pos']:>5.2f}%  "
                  f"{b['mi']:>7.4f} {b['std_pmi']:>7.4f} {cv_b:>6.2f} "
                  f"{b['pct_strong_pos']:>5.2f}%")
        
        print(f"\n  CV = coefficient of variation (std/mean)")
        print(f"  Higher CV at long range = sparser, burstier dependencies")
        print(f"  >2b% = percentage of token pairs with PMI above 2 bits")
    
    # Save
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
