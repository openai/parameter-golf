"""
Multi-Scale Entropy Analysis v3.1 — FineWeb Token-Level (Improved)
===================================================================

Changes from v3:
  1. SHUFFLED BASELINE: Measures MI on randomly shuffled tokens to find
     the "floor" MI caused by token frequency effects (not sequential 
     structure). Subtracts this floor so the histogram reflects only
     genuinely exploitable sequential information.
  
  2. EXTENDED LAG RANGE: Goes up to --max-lag 512 by default to properly
     capture the document-level band.
  
  3. DENSER LAG SAMPLING: More points in the 16-128 range where the 
     sentence/paragraph transition occurs.
  
  4. DECAY CURVE FITTING: Fits a power-law to the MI decay curve,
     which gives a single number characterizing how quickly information
     falls off with distance.
  
  5. OPTIMIZED FOR LONGER RUNS: Uses more tokens for tighter estimates.

USAGE (recommended for the real measurement, ~15-30 min on M1):
  python entropy_analysis_v3_1.py \
      --data-path ./data/datasets/fineweb10B_sp1024 \
      --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model \
      --val-only \
      --max-tokens 20000000 \
      --max-lag 512 \
      --output entropy_histogram_full.json

QUICK CHECK (~15 seconds):
  python entropy_analysis_v3_1.py \
      --data-path ./data/datasets/fineweb10B_sp1024 \
      --val-only \
      --max-tokens 500000 \
      --quick \
      --output entropy_histogram_quick.json
"""

import argparse
import collections
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import sentencepiece as spm
    HAS_SPM = True
except ImportError:
    HAS_SPM = False


# ============================================================
# DATA LOADING
# ============================================================

def load_shard(file_path):
    """Load a FineWeb binary shard (256-int32 header + uint16 tokens)."""
    header = np.fromfile(file_path, dtype='<i4', count=256)
    if header.size != 256:
        raise ValueError(f"Short header in {file_path}")
    
    magic, version, num_tokens = int(header[0]), int(header[1]), int(header[2])
    if magic != 20240520 or version != 1:
        raise ValueError(f"Bad header in {file_path}: magic={magic}, ver={version}")
    
    header_bytes = 256 * np.dtype('<i4').itemsize
    tokens = np.fromfile(file_path, dtype='<u2', count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read in {file_path}")
    
    return tokens


def load_tokens(data_path, pattern="fineweb_val_*.bin", max_tokens=None):
    """Load tokens from shards up to max_tokens."""
    files = sorted(glob.glob(os.path.join(data_path, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_path}")
    
    all_tokens = []
    total = 0
    for f in files:
        shard = load_shard(f)
        all_tokens.append(shard)
        total += len(shard)
        print(f"  Loaded {os.path.basename(f)}: {len(shard):,d} tokens (total: {total:,d})")
        if max_tokens and total >= max_tokens:
            break
    
    tokens = np.concatenate(all_tokens)
    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


# ============================================================
# MI COMPUTATION
# ============================================================

def compute_mi_fast(tokens, lag, vocab_size):
    """Compute I(X_t ; X_{t-lag}) using vectorized numpy."""
    n = len(tokens) - lag
    if n <= 0:
        return 0.0
    
    x = tokens[lag:].astype(np.int64)
    y = tokens[:n].astype(np.int64)
    
    joint = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    np.add.at(joint, (x, y), 1)
    
    mx = joint.sum(axis=1)
    my = joint.sum(axis=0)
    
    mask = joint > 0
    if not mask.any():
        return 0.0
    
    joint_nz = joint[mask].astype(np.float64)
    rows, cols = np.nonzero(joint)
    mx_nz = mx[rows].astype(np.float64)
    my_nz = my[cols].astype(np.float64)
    
    log_terms = np.log2(joint_nz * n / (mx_nz * my_nz))
    mi = np.sum((joint_nz / n) * log_terms)
    
    return float(mi)


def compute_shuffled_baseline_mi(tokens, lag, vocab_size, num_shuffles=3):
    """
    Compute MI on shuffled tokens to measure the "floor" MI that comes 
    from token frequency distributions rather than sequential structure.
    
    With a small vocabulary (1024), common tokens like space and 'e' create
    positive MI even between unrelated positions. This baseline captures that.
    """
    mi_values = []
    for i in range(num_shuffles):
        shuffled = tokens.copy()
        np.random.shuffle(shuffled)
        mi = compute_mi_fast(shuffled, lag, vocab_size)
        mi_values.append(mi)
    return float(np.mean(mi_values))


# ============================================================
# DECAY CURVE FITTING
# ============================================================

def fit_power_law(lags, mi_values):
    """
    Fit MI(lag) = a * lag^(-b) + c to the MI decay curve.
    
    The exponent b tells us how quickly information decays with distance:
      b ≈ 0.5: slow decay (lots of long-range structure)
      b ≈ 1.0: moderate decay
      b ≈ 2.0: fast decay (mostly local structure)
    
    The offset c represents the asymptotic MI floor.
    """
    # Filter to positive MI values and lags > 0
    valid = [(l, m) for l, m in zip(lags, mi_values) if l > 0 and m > 0]
    if len(valid) < 3:
        return None
    
    log_lags = np.array([math.log(l) for l, _ in valid])
    log_mi = np.array([math.log(m) for _, m in valid])
    
    # Simple linear regression in log-log space: log(MI) = log(a) - b*log(lag)
    # This ignores the offset c but gives a reasonable first estimate
    n = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_mi)
    sum_xy = np.sum(log_lags * log_mi)
    sum_xx = np.sum(log_lags * log_lags)
    
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-10:
        return None
    
    b = -(n * sum_xy - sum_x * sum_y) / denom
    log_a = (sum_y + b * sum_x) / n
    a = math.exp(log_a)
    
    # R² for fit quality
    y_pred = log_a - b * log_lags
    ss_res = np.sum((log_mi - y_pred) ** 2)
    ss_tot = np.sum((log_mi - np.mean(log_mi)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return {'a': a, 'b': b, 'r_squared': r_squared}


# ============================================================
# SCALE BANDS (adjusted based on v3 results)
# ============================================================

SCALE_BANDS = [
    #  name                      start  end   window  head_dim
    ("adjacent (lag 1-2)",          1,    3,     8,     32),
    ("local (lag 3-6)",             3,    7,    16,     48),
    ("phrasal (lag 7-16)",          7,   17,    32,     64),
    ("sentence (lag 17-48)",       17,   49,   128,     64),
    ("paragraph (lag 49-128)",     49,  129,   256,     64),
    ("document (lag 129-512)",    129,  513,  1024,     64),
]


def aggregate_into_bands(mi_results, baseline_mi=None):
    """
    Aggregate per-lag MI into scale bands.
    If baseline_mi is provided, subtract the shuffled floor from each lag.
    """
    band_totals = {}
    total_mi = 0.0
    
    for band_name, start, end, _, _ in SCALE_BANDS:
        band_mi = 0.0
        for lag, mi in mi_results:
            if start <= lag < end:
                # Subtract baseline if available
                if baseline_mi and lag in baseline_mi:
                    corrected = max(0.0, mi - baseline_mi[lag])
                else:
                    corrected = mi
                band_mi += corrected
        band_totals[band_name] = band_mi
        total_mi += band_mi
    
    histogram = {}
    for band_name in band_totals:
        pct = band_totals[band_name] / total_mi * 100 if total_mi > 0 else 0
        histogram[band_name] = {
            'mi_bits': band_totals[band_name],
            'percent': pct
        }
    
    return histogram, total_mi


def recommend_head_allocation(histogram, total_heads=8):
    """Produce attention head allocation from histogram."""
    band_lookup = {b[0]: (b[3], b[4]) for b in SCALE_BANDS}
    
    allocations = []
    remaining = total_heads
    
    sorted_bands = sorted(histogram.items(), key=lambda x: x[1]['percent'], reverse=True)
    
    for band_name, info in sorted_bands:
        pct = info['percent']
        if pct < 3.0 or remaining <= 0:
            continue
        
        heads = max(1, round(pct / 100 * total_heads))
        heads = min(heads, remaining)
        window, head_dim = band_lookup.get(band_name, (1024, 64))
        
        allocations.append({
            'band': band_name,
            'heads': heads,
            'window_size': window,
            'head_dim': head_dim,
            'info_percent': round(pct, 1)
        })
        remaining -= heads
    
    if remaining > 0 and allocations:
        allocations[0]['heads'] += remaining
    
    return allocations


# ============================================================
# MAIN ANALYSIS
# ============================================================

def build_lag_schedule(max_lag, quick=False):
    """Build the lag measurement schedule."""
    if quick:
        return [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
                192, 256, 384, 512]
    
    lags = (
        list(range(1, 21)) +                          # every 1 from 1-20
        list(range(22, 41, 2)) +                       # every 2 from 22-40
        list(range(44, 65, 4)) +                       # every 4 from 44-64
        list(range(72, 129, 8)) +                      # every 8 from 72-128
        list(range(144, 257, 16)) +                    # every 16 from 144-256
        list(range(288, min(max_lag + 1, 513), 32))    # every 32 from 288-512
    )
    return [l for l in lags if l <= max_lag]


def run_analysis(tokens, vocab_size, max_lag=512, quick=False):
    """Run the full analysis with shuffled baseline correction."""
    
    print(f"\nAnalyzing {len(tokens):,d} tokens (vocab_size={vocab_size})")
    
    lags = build_lag_schedule(max_lag, quick)
    lags = [l for l in lags if l < len(tokens) // 2]
    
    # --- Step 1: Compute MI at each lag ---
    print(f"\nStep 1: Computing MI at {len(lags)} lag distances...")
    print(f"{'Lag':>6s}  {'Raw MI':>9s}  {'Bar':>40s}")
    print(f"{'-'*6}  {'-'*9}  {'-'*40}")
    
    mi_results = []
    max_mi = 0
    t0 = time.time()
    
    for i, lag in enumerate(lags):
        mi = compute_mi_fast(tokens, lag, vocab_size)
        mi_results.append((lag, mi))
        max_mi = max(max_mi, mi) if mi > 0 else max_mi
        
        bar_len = int(mi / max(max_mi, 0.001) * 35)
        bar = '█' * bar_len
        print(f"{lag:>6d}  {mi:>9.4f}  {bar}")
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(lags) - i - 1)
            print(f"         [{i+1}/{len(lags)} done, ~{eta:.0f}s remaining]")
    
    mi_time = time.time() - t0
    print(f"\nMI computation: {mi_time:.1f}s")
    
    # --- Step 2: Shuffled baseline ---
    print(f"\nStep 2: Computing shuffled baseline (3 shuffles)...")
    print("  This measures MI from token frequencies, not sequential structure.")
    
    # Only compute baseline at a few representative lags (it's roughly constant)
    baseline_lags = [1, 4, 16, 64, 128, 256, 512]
    baseline_lags = [l for l in baseline_lags if l <= max_lag and l < len(tokens) // 2]
    
    baseline_mi = {}
    t0_baseline = time.time()
    
    for lag in baseline_lags:
        bmi = compute_shuffled_baseline_mi(tokens, lag, vocab_size, num_shuffles=3)
        baseline_mi[lag] = bmi
        print(f"  Lag {lag:>4d}: baseline MI = {bmi:.4f} bits")
    
    # Interpolate baseline for all lags (it should be roughly constant)
    avg_baseline = float(np.mean(list(baseline_mi.values())))
    print(f"\n  Average baseline MI: {avg_baseline:.4f} bits")
    print(f"  (This is subtracted from raw MI to get sequential MI)")
    
    # Fill in baseline for all measured lags using the average
    for lag, _ in mi_results:
        if lag not in baseline_mi:
            baseline_mi[lag] = avg_baseline
    
    baseline_time = time.time() - t0_baseline
    print(f"  Baseline computation: {baseline_time:.1f}s")
    
    # --- Step 3: Corrected MI and histogram ---
    print(f"\n{'='*65}")
    print("Step 3: CORRECTED MI (raw - baseline)")
    print(f"{'='*65}")
    
    corrected_results = []
    print(f"\n{'Lag':>6s}  {'Raw':>8s}  {'Base':>8s}  {'Corrected':>9s}  {'Bar':>30s}")
    print(f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*30}")
    
    max_corrected = 0
    for lag, mi in mi_results:
        base = baseline_mi.get(lag, avg_baseline)
        corrected = max(0.0, mi - base)
        corrected_results.append((lag, corrected))
        max_corrected = max(max_corrected, corrected)
        
        bar_len = int(corrected / max(max_corrected, 0.001) * 25) if max_corrected > 0 else 0
        bar = '█' * bar_len
        print(f"{lag:>6d}  {mi:>8.4f}  {base:>8.4f}  {corrected:>9.4f}  {bar}")
    
    # --- Step 4: Histogram ---
    print(f"\n{'='*65}")
    print("Step 4: ENTROPY HISTOGRAM (baseline-corrected)")
    print(f"{'='*65}")
    
    histogram, total_mi = aggregate_into_bands(mi_results, baseline_mi)
    
    print(f"\n{'Band':<32s} {'MI':>8s} {'%':>7s} {'':>25s}")
    print(f"{'-'*32} {'-'*8} {'-'*7} {'-'*25}")
    
    for band_name, _, _, _, _ in SCALE_BANDS:
        info = histogram.get(band_name, {'mi_bits': 0, 'percent': 0})
        bar = '█' * int(info['percent'] / 2)
        print(f"{band_name:<32s} {info['mi_bits']:>8.3f} {info['percent']:>6.1f}% {bar}")
    
    print(f"\n{'Total corrected MI:':<32s} {total_mi:>8.3f} bits")
    
    # --- Step 5: Decay curve fit ---
    print(f"\n{'='*65}")
    print("Step 5: DECAY CURVE ANALYSIS")
    print(f"{'='*65}")
    
    corrected_lags = [l for l, m in corrected_results if m > 0]
    corrected_mi = [m for l, m in corrected_results if m > 0]
    
    fit = fit_power_law(corrected_lags, corrected_mi)
    if fit:
        print(f"\n  Power-law fit: MI(lag) ≈ {fit['a']:.3f} × lag^(-{fit['b']:.3f})")
        print(f"  R² = {fit['r_squared']:.4f}")
        print(f"\n  Decay exponent b = {fit['b']:.3f}")
        if fit['b'] < 0.5:
            print("  → Very slow decay: substantial long-range structure")
            print("    Consider more global heads than the histogram suggests")
        elif fit['b'] < 1.0:
            print("  → Moderate decay: balanced local/global structure")
            print("    The histogram allocation should work well")
        else:
            print("  → Fast decay: mostly local structure")
            print("    Consider even more local heads than suggested")
    else:
        print("  Could not fit power-law (insufficient data)")
    
    # --- Step 6: Head allocation ---
    print(f"\n{'='*65}")
    print("Step 6: RECOMMENDED ATTENTION HEAD ALLOCATION")
    print(f"{'='*65}")
    
    allocations = recommend_head_allocation(histogram, total_heads=8)
    
    model_dim = 512
    print(f"\n{'Band':<32s} {'Heads':>5s} {'Window':>7s} {'HdDim':>6s} {'Info%':>6s}")
    print(f"{'-'*32} {'-'*5} {'-'*7} {'-'*6} {'-'*6}")
    
    total_attn_params = 0
    for alloc in allocations:
        print(f"{alloc['band']:<32s} {alloc['heads']:>5d} "
              f"{alloc['window_size']:>7d} {alloc['head_dim']:>6d} "
              f"{alloc['info_percent']:>5.1f}%")
        total_attn_params += alloc['heads'] * alloc['head_dim'] * model_dim * 4
    
    standard_params = 8 * 64 * model_dim * 4
    savings = standard_params - total_attn_params
    
    print(f"\n  Attention params/layer: {total_attn_params:,d} "
          f"(vs {standard_params:,d} standard)")
    print(f"  Savings: {savings:,d} params/layer ({savings/standard_params*100:.1f}%)")
    print(f"  Over 11 layers at int6: {int(savings * 11 * 0.75):,d} bytes freed")
    print(f"  That's {savings * 11 * 0.75 / 1_000_000:.2f} MB freed for other uses")
    
    return mi_results, corrected_results, baseline_mi, histogram, allocations, fit


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scale entropy analysis v3.1 for Parameter Golf"
    )
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--tokenizer-path', type=str, default=None)
    parser.add_argument('--val-only', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=10_000_000)
    parser.add_argument('--max-lag', type=int, default=512)
    parser.add_argument('--vocab-size', type=int, default=1024)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    print("=" * 65)
    print("PARAMETER GOLF: Multi-Scale Entropy Analysis v3.1")
    print("  (with baseline correction, extended range, decay fitting)")
    print("=" * 65)
    
    # Load tokens
    pattern = "fineweb_val_*.bin" if args.val_only else "fineweb_*_*.bin"
    print(f"\nLoading tokens from {args.data_path}")
    tokens = load_tokens(args.data_path, pattern=pattern, max_tokens=args.max_tokens)
    
    # Stats
    actual_max = int(tokens.max())
    actual_unique = len(np.unique(tokens))
    print(f"\nToken stats: min={int(tokens.min())}, max={actual_max}, "
          f"unique={actual_unique}/{args.vocab_size}")
    
    if actual_max >= args.vocab_size:
        args.vocab_size = actual_max + 1
        print(f"  Adjusted vocab_size to {args.vocab_size}")
    
    # Optional tokenizer info
    if args.tokenizer_path and HAS_SPM:
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        print(f"\nTokenizer: {sp.vocab_size()} tokens")
        # Show what typical tokens look like to give context for the lag distances
        # Estimate average characters per token
        sample_text = "The quick brown fox jumps over the lazy dog in the park."
        encoded = sp.encode(sample_text)
        chars_per_token = len(sample_text) / len(encoded)
        print(f"  Avg chars/token: ~{chars_per_token:.1f}")
        print(f"  So token lag 1 ≈ {chars_per_token:.0f} chars, "
              f"lag 10 ≈ {10*chars_per_token:.0f} chars, "
              f"lag 100 ≈ {100*chars_per_token:.0f} chars")
    
    # Run analysis
    (mi_results, corrected_results, baseline_mi, 
     histogram, allocations, fit) = run_analysis(
        tokens, args.vocab_size, max_lag=args.max_lag, quick=args.quick
    )
    
    # Save results
    if args.output:
        output = {
            'config': {
                'data_path': args.data_path,
                'tokens_analyzed': len(tokens),
                'vocab_size': args.vocab_size,
                'max_lag': args.max_lag,
            },
            'raw_mi': [{'lag': int(l), 'mi': float(m)} for l, m in mi_results],
            'corrected_mi': [{'lag': int(l), 'mi': float(m)} for l, m in corrected_results],
            'baseline_mi_avg': float(np.mean(list(baseline_mi.values()))),
            'histogram': histogram,
            'allocations': allocations,
            'decay_fit': fit,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    # Print ready-to-use config
    print(f"\n{'='*65}")
    print("READY-TO-USE HEAD CONFIGURATION")
    print(f"{'='*65}")
    print("\nPaste this into your modified CausalSelfAttention:\n")
    print("HEAD_CONFIG = [")
    for alloc in allocations:
        print(f"    {{'band': '{alloc['band']}', "
              f"'heads': {alloc['heads']}, "
              f"'window': {alloc['window_size']}, "
              f"'head_dim': {alloc['head_dim']}}},")
    print("]")
    
    total_time = time.time()
    print(f"\nDone. Run the full (non-quick) version for production-quality results.")


if __name__ == '__main__':
    main()
