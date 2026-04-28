"""
Vocabulary Size Sweep: How Tokenizer Size Filters Entropy
===========================================================

This tool investigates how vocabulary size redistributes information
across context distances by:

1. Decoding existing FineWeb tokens back to raw text
2. Training SentencePiece BPE tokenizers at each vocab size
3. Re-tokenizing the same text with each tokenizer  
4. Running MI analysis at each vocabulary size
5. Computing parameter budget implications
6. Producing a comparison table and JSON for visualization

USAGE (recommended, ~30-60 min on M1 8GB):
  python vocab_sweep.py \
      --data-path ./data/datasets/fineweb10B_sp1024 \
      --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model \
      --text-chars 2000000 \
      --mi-tokens 1000000 \
      --output vocab_sweep_results.json

QUICK TEST (~5 min):
  python vocab_sweep.py \
      --data-path ./data/datasets/fineweb10B_sp1024 \
      --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model \
      --text-chars 500000 \
      --mi-tokens 200000 \
      --quick \
      --output vocab_sweep_quick.json
"""

import argparse
import glob
import json
import math
import os
import tempfile
import time

import numpy as np

try:
    import sentencepiece as spm
except ImportError:
    print("ERROR: sentencepiece is required. Install with: pip install sentencepiece")
    raise


# ============================================================
# DATA LOADING & DECODING
# ============================================================

def load_shard(file_path):
    """Load a FineWeb binary shard."""
    header = np.fromfile(file_path, dtype='<i4', count=256)
    if header.size != 256:
        raise ValueError(f"Short header in {file_path}")
    magic, version, num_tokens = int(header[0]), int(header[1]), int(header[2])
    if magic != 20240520 or version != 1:
        raise ValueError(f"Bad header in {file_path}")
    header_bytes = 256 * np.dtype('<i4').itemsize
    tokens = np.fromfile(file_path, dtype='<u2', count=num_tokens, offset=header_bytes)
    return tokens


def decode_tokens_to_text(data_path, tokenizer_path, max_chars):
    """Decode tokenized FineWeb data back to raw text."""
    print(f"\n  Loading tokenizer: {tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    
    files = sorted(glob.glob(os.path.join(data_path, "fineweb_val_*.bin")))
    if not files:
        raise FileNotFoundError(f"No val files in {data_path}")
    
    print(f"  Loading shard: {os.path.basename(files[0])}")
    tokens = load_shard(files[0])
    
    # Decode in chunks to manage memory
    print(f"  Decoding tokens to text (target: {max_chars:,d} chars)...")
    chunk_size = 50000
    text_parts = []
    total_chars = 0
    
    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start:start + chunk_size].tolist()
        decoded = sp.decode(chunk)
        text_parts.append(decoded)
        total_chars += len(decoded)
        if total_chars >= max_chars:
            break
    
    text = ''.join(text_parts)[:max_chars]
    print(f"  Decoded {total_chars:,d} chars from {min(start + chunk_size, len(tokens)):,d} tokens")
    
    return text


# ============================================================
# SENTENCEPIECE TRAINING
# ============================================================

def train_sentencepiece(text, vocab_size, model_prefix, max_train_chars=2_000_000):
    """Train a SentencePiece BPE model on the given text."""
    # Write text to a temp file for SP training.
    # SentencePiece expects one sentence per line and skips lines longer
    # than max_sentence_length (default 16384). We split on sentence-ending
    # punctuation and newlines to create properly sized lines.
    sample = text[:max_train_chars]
    
    # Split into lines at sentence boundaries
    lines = []
    current = []
    for char in sample:
        current.append(char)
        # Split on sentence-ending punctuation followed by space, or on newlines
        if char in '.!?\n' and len(current) > 20:
            lines.append(''.join(current).strip())
            current = []
    if current:
        lines.append(''.join(current).strip())
    
    # Filter out empty lines and lines that are still too long
    lines = [line for line in lines if 10 < len(line) < 15000]
    
    if not lines:
        raise ValueError(f"No valid sentences found in text sample ({len(sample)} chars)")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, 
                                      encoding='utf-8') as f:
        f.write('\n'.join(lines))
        temp_path = f.name
    
    print(f"    ({len(lines)} sentences, avg {sum(len(l) for l in lines)//len(lines)} chars/line)")
    
    try:
        spm.SentencePieceTrainer.train(
            input=temp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=1.0,
            num_threads=4,
            max_sentence_length=16384,
            shuffle_input_sentence=True,
            # Match parameter-golf style: no special tokens beyond defaults
            pad_id=-1,
            bos_id=-1,
            eos_id=-1,
        )
    finally:
        os.unlink(temp_path)
    
    return f"{model_prefix}.model"


# ============================================================
# MI COMPUTATION (from v3.1)
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
    
    mi = np.sum((joint_nz / n) * np.log2(joint_nz * n / (mx_nz * my_nz)))
    return float(mi)


def compute_baseline_mi(tokens, vocab_size):
    """Compute shuffled baseline MI (average over a few lags)."""
    shuffled = tokens.copy()
    np.random.shuffle(shuffled)
    # Baseline is ~constant across lags, just measure at lag 1
    return compute_mi_fast(shuffled, 1, vocab_size)


# ============================================================
# SCALE BAND ANALYSIS
# ============================================================

# Bands defined relative to TOKEN positions
BANDS = [
    ("adjacent (1-2)",     1,   3),
    ("local (3-6)",        3,   7),
    ("phrasal (7-16)",     7,  17),
    ("sentence (17-48)",  17,  49),
    ("paragraph (49-128)",49, 129),
    ("document (129+)",  129, 999),
]

def compute_band_histogram(mi_results, baseline_mi):
    """Aggregate MI into bands with baseline correction."""
    band_totals = {}
    total = 0.0
    
    for name, start, end in BANDS:
        band_mi = sum(max(0, mi - baseline_mi) for lag, mi in mi_results if start <= lag < end)
        band_totals[name] = band_mi
        total += band_mi
    
    histogram = {}
    for name, mi in band_totals.items():
        histogram[name] = {
            'mi_bits': round(mi, 4),
            'percent': round(mi / total * 100, 1) if total > 0 else 0.0
        }
    
    return histogram, total


# ============================================================
# PARAMETER BUDGET CALCULATOR
# ============================================================

def compute_param_budget(vocab_size, model_dim, mlp_mult=3, num_kv_heads_ratio=0.5,
                          total_bytes=16_000_000, bytes_per_param=0.75):
    """
    Compute how many layers fit in the artifact given vocab size and model dim.
    
    bytes_per_param = 0.75 corresponds to int6 + zstd compression.
    """
    max_params = total_bytes / bytes_per_param
    
    # Embedding params (tied input/output)
    embed_params = vocab_size * model_dim
    
    # Per-layer params (approximate):
    # Attention: Q + K + V + O projections
    # With GQA: Q = dim*dim, K = dim*(dim*kv_ratio), V = same, O = dim*dim
    num_heads = max(1, model_dim // 64)
    num_kv_heads = max(1, int(num_heads * num_kv_heads_ratio))
    kv_dim = num_kv_heads * (model_dim // num_heads)
    attn_params = model_dim * model_dim + 2 * model_dim * kv_dim + model_dim * model_dim
    
    # MLP: up projection + down projection (with mlp_mult expansion)
    mlp_hidden = model_dim * mlp_mult
    mlp_params = model_dim * mlp_hidden + mlp_hidden * model_dim
    
    # Misc per layer: norms, scales, residual mix (~3 * model_dim)
    misc_params = 3 * model_dim
    
    layer_params = attn_params + mlp_params + misc_params
    
    # How many layers fit?
    available_for_layers = max_params - embed_params
    num_layers = max(1, int(available_for_layers / layer_params))
    
    total_params = embed_params + num_layers * layer_params
    artifact_mb = total_params * bytes_per_param / 1_000_000
    
    return {
        'vocab_size': vocab_size,
        'model_dim': model_dim,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'num_layers': num_layers,
        'embed_params': embed_params,
        'layer_params': layer_params,
        'total_params': int(total_params),
        'embed_pct': round(embed_params / total_params * 100, 1),
        'artifact_mb': round(artifact_mb, 2),
    }


# ============================================================
# MAIN SWEEP
# ============================================================

def run_sweep(text, vocab_sizes, mi_tokens, lags, temp_dir):
    """Run the full vocabulary sweep."""
    
    results = []
    
    for vi, vocab_size in enumerate(vocab_sizes):
        print(f"\n{'='*65}")
        print(f"  VOCAB SIZE: {vocab_size} ({vi+1}/{len(vocab_sizes)})")
        print(f"{'='*65}")
        
        t0 = time.time()
        
        # Train tokenizer
        model_prefix = os.path.join(temp_dir, f"sp_{vocab_size}")
        print(f"  Training SentencePiece (vocab={vocab_size})...")
        
        try:
            sp_model_path = train_sentencepiece(text, vocab_size, model_prefix)
        except Exception as e:
            print(f"  ERROR training tokenizer: {e}")
            continue
        
        # Load and tokenize
        sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        actual_vocab = sp.vocab_size()
        print(f"  Actual vocab size: {actual_vocab}")
        
        # Tokenize the text
        print(f"  Tokenizing text...")
        token_ids = sp.encode(text)
        tokens = np.array(token_ids, dtype=np.uint16)
        
        # Limit tokens for MI analysis
        if len(tokens) > mi_tokens:
            tokens = tokens[:mi_tokens]
        
        # Stats
        chars_per_token = len(text) / len(token_ids)
        unique_tokens = len(set(token_ids))
        print(f"  Tokens: {len(tokens):,d} ({chars_per_token:.1f} chars/token)")
        print(f"  Unique tokens used: {unique_tokens}/{actual_vocab}")
        
        # Compute MI at each lag
        print(f"  Computing MI at {len(lags)} lags...")
        baseline = compute_baseline_mi(tokens, actual_vocab)
        
        mi_results = []
        for lag in lags:
            if lag >= len(tokens) // 2:
                break
            mi = compute_mi_fast(tokens, lag, actual_vocab)
            mi_results.append((lag, mi))
        
        # Compute histogram
        histogram, total_mi = compute_band_histogram(mi_results, baseline)
        
        # Parameter budgets at different model dimensions
        budgets = []
        for dim in [256, 320, 384, 448, 512]:
            budget = compute_param_budget(actual_vocab, dim)
            budgets.append(budget)
        
        elapsed = time.time() - t0
        
        result = {
            'vocab_size': actual_vocab,
            'chars_per_token': round(chars_per_token, 2),
            'unique_tokens_used': unique_tokens,
            'total_tokens': len(tokens),
            'baseline_mi': round(baseline, 4),
            'total_corrected_mi': round(total_mi, 4),
            'histogram': histogram,
            'mi_curve': [{'lag': int(l), 'mi': round(m, 4)} for l, m in mi_results],
            'param_budgets': budgets,
            'time_seconds': round(elapsed, 1),
        }
        results.append(result)
        
        # Print summary for this vocab size
        print(f"\n  Entropy histogram:")
        for name, _, _ in BANDS:
            info = histogram.get(name, {'percent': 0})
            bar = '█' * int(info['percent'] / 2.5)
            print(f"    {name:<22s} {info['percent']:>5.1f}% {bar}")
        
        print(f"\n  Parameter budgets (16MB at int6):")
        print(f"    {'Dim':>4s} {'Layers':>6s} {'Params':>10s} {'Emb%':>5s} {'MB':>6s}")
        for b in budgets:
            print(f"    {b['model_dim']:>4d} {b['num_layers']:>6d} "
                  f"{b['total_params']:>10,d} {b['embed_pct']:>4.1f}% "
                  f"{b['artifact_mb']:>5.1f}")
        
        print(f"  Time: {elapsed:.1f}s")
        
        # Cleanup temp files
        for ext in ['.model', '.vocab']:
            path = f"{model_prefix}{ext}"
            if os.path.exists(path):
                os.unlink(path)
    
    return results


def print_comparison_table(results):
    """Print a comparison table across all vocabulary sizes."""
    
    print(f"\n{'='*100}")
    print("VOCABULARY SWEEP COMPARISON")
    print(f"{'='*100}")
    
    # Band distribution comparison
    print(f"\n{'Vocab':>6s} {'Chr/Tok':>7s} ", end='')
    for name, _, _ in BANDS:
        short_name = name.split('(')[0].strip()
        print(f"{'|':>2s} {short_name:>10s}", end='')
    print(f" {'|':>2s} {'MI Total':>8s}")
    
    print(f"{'-'*6} {'-'*7} ", end='')
    for _ in BANDS:
        print(f"{'|':>2s} {'-'*10}", end='')
    print(f" {'|':>2s} {'-'*8}")
    
    for r in results:
        print(f"{r['vocab_size']:>6d} {r['chars_per_token']:>7.1f} ", end='')
        for name, _, _ in BANDS:
            pct = r['histogram'].get(name, {}).get('percent', 0)
            bar = '█' * int(pct / 5)
            print(f"{'|':>2s} {pct:>5.1f}% {bar:<4s}", end='')
        print(f" {'|':>2s} {r['total_corrected_mi']:>7.3f}")
    
    # Optimal architecture suggestions
    print(f"\n{'='*100}")
    print("ARCHITECTURE SUGGESTIONS (16MB budget at int6)")
    print(f"{'='*100}")
    
    print(f"\n{'Vocab':>6s} {'Dim':>5s} {'Layers':>7s} {'Params':>10s} {'Emb%':>5s} "
          f"{'Adj%':>5s} {'Sent%':>5s} {'Suggested Heads':>20s}")
    print(f"{'-'*6} {'-'*5} {'-'*7} {'-'*10} {'-'*5} {'-'*5} {'-'*5} {'-'*20}")
    
    for r in results:
        adj_pct = r['histogram'].get('adjacent (1-2)', {}).get('percent', 0)
        sent_pct = r['histogram'].get('sentence (17-48)', {}).get('percent', 0)
        
        # Pick the "sweet spot" model dim for this vocab size
        # Target: embedding is 10-20% of total params
        best_budget = None
        for b in r['param_budgets']:
            if 8 <= b['embed_pct'] <= 25 and b['num_layers'] >= 6:
                if best_budget is None or b['num_layers'] > best_budget['num_layers']:
                    best_budget = b
        
        if best_budget is None:
            best_budget = r['param_budgets'][2]  # middle option
        
        # Suggest head split based on histogram
        n_heads = max(4, best_budget['model_dim'] // 64)
        adj_heads = max(1, round(adj_pct / 100 * n_heads))
        remaining = n_heads - adj_heads
        sent_heads = max(1, round(sent_pct / 100 * n_heads)) if remaining > 1 else 0
        remaining -= sent_heads
        mid_heads = max(1, remaining)
        head_str = f"{adj_heads}L/{mid_heads}M/{sent_heads}G"
        
        print(f"{r['vocab_size']:>6d} {best_budget['model_dim']:>5d} "
              f"{best_budget['num_layers']:>7d} "
              f"{best_budget['total_params']:>10,d} {best_budget['embed_pct']:>4.1f}% "
              f"{adj_pct:>4.1f}% {sent_pct:>4.1f}% {head_str:>20s}")


def main():
    parser = argparse.ArgumentParser(
        description="Vocabulary size sweep for entropy analysis"
    )
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--tokenizer-path', type=str, required=True,
                       help='Path to existing SP 1024 model for decoding')
    parser.add_argument('--vocab-sizes', type=str, 
                       default='512,1024,1536,2048,3072,4096,6144,8192',
                       help='Comma-separated vocab sizes to test')
    parser.add_argument('--text-chars', type=int, default=2_000_000,
                       help='Characters of text to decode and use')
    parser.add_argument('--mi-tokens', type=int, default=500_000,
                       help='Max tokens per vocab size for MI analysis')
    parser.add_argument('--quick', action='store_true',
                       help='Fewer lag points for speed')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    vocab_sizes = [int(v.strip()) for v in args.vocab_sizes.split(',')]
    
    print("=" * 65)
    print("VOCABULARY SIZE SWEEP: Tokenizer as Entropy Filter")
    print("=" * 65)
    print(f"\nVocab sizes to test: {vocab_sizes}")
    print(f"Text sample: {args.text_chars:,d} chars")
    print(f"MI tokens per vocab: {args.mi_tokens:,d}")
    
    # Step 1: Decode existing tokens to raw text
    print(f"\n{'='*65}")
    print("STEP 1: Decoding FineWeb tokens to raw text")
    print(f"{'='*65}")
    
    text = decode_tokens_to_text(args.data_path, args.tokenizer_path, args.text_chars)
    print(f"  Raw text: {len(text):,d} chars")
    print(f"  Sample: {text[:100]}...")
    
    # Step 2: Define lag schedule
    if args.quick:
        lags = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    else:
        lags = (
            list(range(1, 17)) + 
            list(range(18, 33, 2)) + 
            list(range(36, 65, 4)) + 
            list(range(72, 129, 8)) +
            list(range(144, 257, 16)) +
            list(range(288, 513, 32))
        )
    
    # Step 3: Run sweep
    print(f"\n{'='*65}")
    print("STEP 2: Running vocabulary sweep")
    print(f"{'='*65}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        results = run_sweep(text, vocab_sizes, args.mi_tokens, lags, temp_dir)
    
    # Step 4: Print comparison
    print_comparison_table(results)
    
    # Step 5: Save results
    if args.output:
        output = {
            'config': {
                'text_chars': len(text),
                'mi_tokens_per_vocab': args.mi_tokens,
                'vocab_sizes': vocab_sizes,
                'num_lags': len(lags),
            },
            'results': results,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print(f"\n{'='*65}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*65}")
    print("""
WHAT TO LOOK FOR:

1. ENTROPY REDISTRIBUTION: As vocab size increases, the 'adjacent'
   band should shrink and 'phrasal'/'sentence' bands should grow.
   This shows the tokenizer filtering effect you hypothesized.

2. SWEET SPOT: Find the vocab size where:
   - Adjacent band drops below ~40% (balanced attention allocation)
   - Embedding cost stays below ~20% of total params 
   - You still get 8+ layers at a reasonable model_dim

3. PARAMETER EFFICIENCY: The 'architecture suggestions' table shows
   the best model shape for each vocab size. Look for configs where
   the entropy is spread across bands AND you have enough layers
   for deep sequential processing.

4. NEXT STEP: Take the best vocab size and run the entropy_analysis
   v3.1 on it with more tokens to get the precise head allocation.
   Then implement multi-resolution attention with that allocation.
""")


if __name__ == '__main__':
    main()
