"""
Re-encode SP1024 training shards as SP4096.
Decodes each shard with SP1024, re-encodes with SP4096, saves as new shard.
CPU-only — runs in parallel with GPU training.

Usage: python reencode_sp4096.py [--shards 80]
"""
import os, sys, time, glob, struct, argparse
import numpy as np
from pathlib import Path
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument('--shards', type=int, default=80)
args = parser.parse_args()

sp1024_path = 'data/tokenizers/fineweb_1024_bpe.model'
sp4096_path = 'data/tokenizers/fineweb_4096_bpe.model'

if not os.path.exists(sp4096_path):
    print(f"ERROR: {sp4096_path} not found. Run train_sp4096_tokenizer.py first.")
    sys.exit(1)

sp1024 = spm.SentencePieceProcessor(model_file=sp1024_path)
sp4096 = spm.SentencePieceProcessor(model_file=sp4096_path)
print(f"SP1024 vocab: {sp1024.vocab_size()}", flush=True)
print(f"SP4096 vocab: {sp4096.vocab_size()}", flush=True)

# Create output directory
out_dir = Path('data/datasets/fineweb10B_sp4096')
out_dir.mkdir(parents=True, exist_ok=True)

# Process training shards
train_files = sorted(glob.glob('data/datasets/fineweb10B_sp1024/fineweb_train_*.bin'))
n_shards = min(args.shards, len(train_files))
print(f"Re-encoding {n_shards} training shards...", flush=True)

t0 = time.time()
for shard_idx in range(n_shards):
    shard_path = train_files[shard_idx]
    shard_name = Path(shard_path).name.replace('fineweb_train_', '')

    # Read SP1024 tokens
    h = np.fromfile(shard_path, dtype='<i4', count=256)
    n_tokens = int(h[2])
    tokens_1024 = np.fromfile(shard_path, dtype='<u2', count=n_tokens, offset=256*4)

    # Decode in chunks and re-encode with SP4096
    chunk_size = 500000  # 500K tokens per chunk
    all_tokens_4096 = []
    for i in range(0, n_tokens, chunk_size):
        chunk = tokens_1024[i:i+chunk_size].tolist()
        text = sp1024.decode(chunk)
        new_tokens = sp4096.encode(text)
        all_tokens_4096.extend(new_tokens)

    tokens_4096 = np.array(all_tokens_4096, dtype=np.uint16)

    # Write new shard (same header format as original)
    out_path = out_dir / f'fineweb_train_{shard_name}'
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1  # version
    header[2] = len(tokens_4096)  # n_tokens
    header[3] = sp4096.vocab_size()

    with open(out_path, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens_4096.tobytes())

    elapsed = time.time() - t0
    ratio = n_tokens / len(tokens_4096)
    eta = elapsed / (shard_idx + 1) * (n_shards - shard_idx - 1) / 60
    print(f"  Shard {shard_idx}/{n_shards}: {n_tokens/1e6:.0f}M→{len(tokens_4096)/1e6:.0f}M tokens (ratio {ratio:.2f}x) [{elapsed:.0f}s, ETA {eta:.0f}min]", flush=True)

# Also process val shards
val_files = sorted(glob.glob('data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
print(f"\nRe-encoding {len(val_files)} val shards...", flush=True)
for val_path in val_files:
    val_name = Path(val_path).name
    h = np.fromfile(val_path, dtype='<i4', count=256)
    n_tokens = int(h[2])
    tokens_1024 = np.fromfile(val_path, dtype='<u2', count=n_tokens, offset=256*4)

    all_tokens_4096 = []
    for i in range(0, n_tokens, 500000):
        chunk = tokens_1024[i:i+500000].tolist()
        text = sp1024.decode(chunk)
        new_tokens = sp4096.encode(text)
        all_tokens_4096.extend(new_tokens)

    tokens_4096 = np.array(all_tokens_4096, dtype=np.uint16)

    out_path = out_dir / val_name
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens_4096)
    header[3] = sp4096.vocab_size()

    with open(out_path, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens_4096.tobytes())

    print(f"  {val_name}: {n_tokens/1e6:.0f}M→{len(tokens_4096)/1e6:.0f}M tokens", flush=True)

total_time = time.time() - t0
print(f"\n=== Done in {total_time/60:.1f} min ===", flush=True)
print(f"Output: {out_dir}", flush=True)
