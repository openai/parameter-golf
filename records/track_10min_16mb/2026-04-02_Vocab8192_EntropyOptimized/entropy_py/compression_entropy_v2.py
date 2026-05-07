"""
Compression-Based Entropy Scale Analysis
==========================================

Measures exploitable structure at different context distances by
compressing tokenized data in blocks of increasing size. The marginal
improvement in compression ratio as block size grows reveals how
much new structure becomes accessible at each scale.

This is an independent validation of the MI-based entropy histogram
using a completely different methodology (Kolmogorov complexity 
approximation via zlib).

USAGE:
  # Compare 1024 and 8192 tokenizations side by side:
  python compression_entropy.py \
      --shard-1024 ./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin \
      --shard-8192 ./data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin \
      --max-tokens 2000000 \
      --output compression_results.json

  # Single tokenization:
  python compression_entropy.py \
      --shard-1024 ./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin \
      --max-tokens 500000
"""

import argparse
import json
import math
import struct
import time
import zlib

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


def tokens_to_bytes(tokens):
    """Convert uint16 token array to raw bytes for compression."""
    return tokens.astype('<u2').tobytes()


def block_compression_analysis(tokens, block_sizes, label=""):
    """
    Compress token stream in non-overlapping blocks of each size.
    Returns the average compression ratio (compressed/raw) at each block size.
    
    Smaller blocks = compressor can only exploit local patterns.
    Larger blocks = compressor can exploit longer-range patterns.
    The marginal improvement from doubling block size = structure at that scale.
    """
    results = []
    raw_bytes_per_token = 2  # uint16
    
    print(f"\n  {'Block':>7s} {'Tokens':>8s}  {'Raw KB':>8s} {'Comp KB':>8s} "
          f"{'Ratio':>6s} {'BPT':>6s} {'Marginal':>8s}")
    print(f"  {'-'*7} {'-'*8}  {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*8}")
    
    prev_bpt = None
    
    for block_size in block_sizes:
        if block_size > len(tokens):
            break
        
        # Split into non-overlapping blocks
        n_blocks = len(tokens) // block_size
        if n_blocks == 0:
            continue
        
        total_raw = 0
        total_compressed = 0
        
        for i in range(n_blocks):
            block = tokens[i * block_size : (i + 1) * block_size]
            raw = tokens_to_bytes(block)
            compressed = zlib.compress(raw, level=9)
            total_raw += len(raw)
            total_compressed += len(compressed)
        
        ratio = total_compressed / total_raw
        # Bits per token: compressed_bits / num_tokens
        bpt = (total_compressed * 8) / (n_blocks * block_size)
        marginal = prev_bpt - bpt if prev_bpt is not None else 0
        
        results.append({
            'block_size': block_size,
            'n_blocks': n_blocks,
            'raw_bytes': total_raw,
            'compressed_bytes': total_compressed,
            'ratio': round(ratio, 4),
            'bits_per_token': round(bpt, 4),
            'marginal_bpt': round(marginal, 4),
        })
        
        marginal_str = f"+{marginal:.4f}" if marginal > 0 else f"{marginal:.4f}" if marginal else "    --"
        print(f"  {block_size:>7d} {n_blocks:>8d}  {total_raw/1024:>8.1f} {total_compressed/1024:>8.1f} "
              f"{ratio:>6.3f} {bpt:>6.2f} {marginal_str:>8s}")
        
        prev_bpt = bpt
    
    return results


def compute_scale_bands(results):
    """
    Aggregate marginal compression improvements into scale bands.
    This is analogous to the MI histogram but from compression data.
    """
    bands = [
        ("local (1-8 tokens)", 1, 9),
        ("phrasal (9-32 tokens)", 9, 33),
        ("sentence (33-128 tokens)", 33, 129),
        ("paragraph (129-512 tokens)", 129, 513),
        ("document (513+ tokens)", 513, 99999),
    ]
    
    total_marginal = sum(r['marginal_bpt'] for r in results if r['marginal_bpt'] > 0)
    
    if total_marginal <= 0:
        return {}
    
    histogram = {}
    for band_name, start, end in bands:
        band_marginal = sum(
            r['marginal_bpt'] for r in results
            if start <= r['block_size'] < end and r['marginal_bpt'] > 0
        )
        histogram[band_name] = {
            'marginal_bpt': round(band_marginal, 4),
            'percent': round(band_marginal / total_marginal * 100, 1)
        }
    
    return histogram


def sliding_window_analysis(tokens, sample_size=500000):
    """
    Alternative approach: compress the full stream with zlib at different
    window sizes (controlled via wbits parameter).
    
    zlib's wbits: 9 = 512 byte window, 15 = 32768 byte window.
    Each uint16 token = 2 bytes, so:
      wbits 9  = 256 token window
      wbits 10 = 512 token window
      wbits 15 = 16384 token window
    """
    sample = tokens[:sample_size]
    raw = tokens_to_bytes(sample)
    
    print(f"\n  {'Window':>8s} {'Tokens':>7s} {'Comp KB':>8s} {'Ratio':>6s} {'Δ ratio':>8s}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*6} {'-'*8}")
    
    results = []
    prev_ratio = None
    
    for wbits in range(9, 16):
        window_bytes = 2 ** wbits
        window_tokens = window_bytes // 2
        
        compressed = zlib.compress(raw, level=9, wbits=wbits)
        ratio = len(compressed) / len(raw)
        delta = prev_ratio - ratio if prev_ratio is not None else 0
        
        results.append({
            'wbits': wbits,
            'window_bytes': window_bytes,
            'window_tokens': window_tokens,
            'compressed_bytes': len(compressed),
            'ratio': round(ratio, 4),
            'delta_ratio': round(delta, 4),
        })
        
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}" if delta else "    --"
        print(f"  {window_bytes:>7d}B {window_tokens:>7d} {len(compressed)/1024:>8.1f} "
              f"{ratio:>6.3f} {delta_str:>8s}")
        
        prev_ratio = ratio
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compression-based entropy scale analysis")
    parser.add_argument('--shard-1024', type=str, default=None,
                       help='Path to a vocab-1024 binary shard')
    parser.add_argument('--shard-8192', type=str, default=None,
                       help='Path to a vocab-8192 binary shard')
    parser.add_argument('--max-tokens', type=int, default=1_000_000,
                       help='Maximum tokens to analyze per shard')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')
    args = parser.parse_args()
    
    if not args.shard_1024 and not args.shard_8192:
        parser.error("Provide at least one of --shard-1024 or --shard-8192")
    
    # Block sizes: powers of 2 plus some intermediate points
    block_sizes = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 
                   768, 1024, 1536, 2048, 4096, 8192]
    
    all_results = {}
    
    for label, shard_path, vocab_size in [
        ("vocab_1024", args.shard_1024, 1024),
        ("vocab_8192", args.shard_8192, 8192),
    ]:
        if not shard_path:
            continue
        
        print(f"\n{'='*65}")
        print(f"  COMPRESSION ANALYSIS: {label} ({shard_path})")
        print(f"{'='*65}")
        
        tokens = load_shard(shard_path, args.max_tokens)
        unique = len(np.unique(tokens))
        print(f"  Tokens: {len(tokens):,d}  Unique: {unique}  Vocab: {vocab_size}")
        
        # Method A: Block compression
        print(f"\n  --- Block Compression (non-overlapping blocks) ---")
        block_results = block_compression_analysis(tokens, block_sizes, label)
        
        # Compute scale bands
        histogram = compute_scale_bands(block_results)
        
        if histogram:
            print(f"\n  Compression-based entropy histogram:")
            print(f"  {'Band':<32s} {'Marginal':>8s} {'%':>6s} {'':>20s}")
            print(f"  {'-'*32} {'-'*8} {'-'*6} {'-'*20}")
            for band_name, info in histogram.items():
                bar = '█' * int(info['percent'] / 2.5)
                print(f"  {band_name:<32s} {info['marginal_bpt']:>8.4f} {info['percent']:>5.1f}% {bar}")
        
        # Method B: Sliding window via zlib wbits
        print(f"\n  --- Sliding Window (zlib wbits parameter) ---")
        window_results = sliding_window_analysis(tokens)
        
        all_results[label] = {
            'tokens': len(tokens),
            'vocab_size': vocab_size,
            'unique_tokens': unique,
            'block_analysis': block_results,
            'histogram': histogram,
            'window_analysis': window_results,
        }
    
    # Side-by-side comparison if both are present
    if 'vocab_1024' in all_results and 'vocab_8192' in all_results:
        print(f"\n{'='*65}")
        print(f"  SIDE-BY-SIDE COMPARISON")
        print(f"{'='*65}")
        
        r1 = all_results['vocab_1024']['block_analysis']
        r8 = all_results['vocab_8192']['block_analysis']
        
        # Match on block size
        sizes_1 = {r['block_size']: r for r in r1}
        sizes_8 = {r['block_size']: r for r in r8}
        common = sorted(set(sizes_1.keys()) & set(sizes_8.keys()))
        
        print(f"\n  {'Block':>7s}  {'--- Vocab 1024 ---':^22s}  {'--- Vocab 8192 ---':^22s}  {'Δ BPT':>6s}")
        print(f"  {'Size':>7s}  {'Ratio':>6s} {'BPT':>6s} {'Marginal':>8s}  {'Ratio':>6s} {'BPT':>6s} {'Marginal':>8s}")
        print(f"  {'-'*7}  {'-'*6} {'-'*6} {'-'*8}  {'-'*6} {'-'*6} {'-'*8}  {'-'*6}")
        
        for bs in common:
            s1 = sizes_1[bs]
            s8 = sizes_8[bs]
            delta = s8['bits_per_token'] - s1['bits_per_token']
            m1 = f"{s1['marginal_bpt']:+.4f}" if s1['marginal_bpt'] else "    --"
            m8 = f"{s8['marginal_bpt']:+.4f}" if s8['marginal_bpt'] else "    --"
            print(f"  {bs:>7d}  {s1['ratio']:>6.3f} {s1['bits_per_token']:>6.2f} {m1:>8s}"
                  f"  {s8['ratio']:>6.3f} {s8['bits_per_token']:>6.2f} {m8:>8s}"
                  f"  {delta:>+6.2f}")
        
        # Compare histograms
        h1 = all_results['vocab_1024']['histogram']
        h8 = all_results['vocab_8192']['histogram']
        
        if h1 and h8:
            print(f"\n  Entropy histogram comparison:")
            print(f"  {'Band':<32s} {'V=1024':>7s} {'V=8192':>7s} {'Shift':>7s}")
            print(f"  {'-'*32} {'-'*7} {'-'*7} {'-'*7}")
            for band_name in h1:
                p1 = h1.get(band_name, {}).get('percent', 0)
                p8 = h8.get(band_name, {}).get('percent', 0)
                shift = p8 - p1
                print(f"  {band_name:<32s} {p1:>6.1f}% {p8:>6.1f}% {shift:>+6.1f}%")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print(f"\n{'='*65}")
    print("INTERPRETATION")
    print(f"{'='*65}")
    print("""
  The 'bits per token' (BPT) at each block size shows how efficiently
  the data can be compressed when limited to that context window.
  
  The 'marginal' column shows how much BPT improved by doubling the
  block size. Large marginals = lots of exploitable structure at that
  scale. When marginals approach zero, you've captured most structure.
  
  Comparing vocab 1024 vs 8192:
  - If 8192 has smaller marginals at short range → the tokenizer has
    already absorbed local structure (confirming the filter hypothesis)
  - If 8192 has larger marginals at long range → longer-range structure
    is more visible with bigger tokens (supporting larger attention windows)
  - The block size where marginals flatten → maximum useful window size
""")


if __name__ == '__main__':
    main()
