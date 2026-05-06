"""
Train a SP4096 tokenizer from our existing SP1024-encoded shards.
Step 1: Decode 5 shards to raw text
Step 2: Train SentencePiece BPE with vocab_size=4096
Step 3: Save tokenizer to data/tokenizers/fineweb_4096_bpe.model

This is CPU-only — won't interfere with GPU training.
"""
import os, sys, time, glob, numpy as np, tempfile
from pathlib import Path
import sentencepiece as spm

print("=== SP4096 Tokenizer Training ===", flush=True)
t0 = time.time()

# Step 1: Decode shards to raw text file
sp1024 = spm.SentencePieceProcessor(model_file='data/tokenizers/fineweb_1024_bpe.model')
train_files = sorted(glob.glob('data/datasets/fineweb10B_sp1024/fineweb_train_*.bin'))
N_SHARDS = 5  # Use 5 shards for tokenizer training (~500M tokens, ~1.2B chars)

text_file = 'data/tokenizer_training_text.txt'
print(f"Step 1: Decoding {N_SHARDS} shards to {text_file}...", flush=True)

with open(text_file, 'w', encoding='utf-8') as f:
    for shard_idx in range(min(N_SHARDS, len(train_files))):
        shard_path = train_files[shard_idx]
        h = np.fromfile(shard_path, dtype='<i4', count=256)
        n_tokens = int(h[2])
        tokens = np.fromfile(shard_path, dtype='<u2', count=n_tokens, offset=256*4)

        # Decode in chunks to avoid memory issues
        chunk_size = 1000000
        for i in range(0, n_tokens, chunk_size):
            chunk = tokens[i:i+chunk_size].tolist()
            text = sp1024.decode(chunk)
            # Write as lines (SentencePiece expects one sentence per line)
            for line in text.split('\n'):
                line = line.strip()
                if len(line) > 10:  # Skip very short lines
                    f.write(line + '\n')

        elapsed = time.time() - t0
        print(f"  Shard {shard_idx}: {n_tokens/1e6:.0f}M tokens decoded ({elapsed:.0f}s)", flush=True)

text_size = os.path.getsize(text_file)
print(f"Text file: {text_size/1e9:.2f} GB", flush=True)

# Step 2: Train SP4096 BPE tokenizer
print(f"\nStep 2: Training SP4096 tokenizer...", flush=True)
model_prefix = 'data/tokenizers/fineweb_4096_bpe'

spm.SentencePieceTrainer.train(
    input=text_file,
    model_prefix=model_prefix,
    vocab_size=4096,
    model_type='bpe',
    character_coverage=1.0,
    byte_fallback=True,
    num_threads=os.cpu_count(),
    input_sentence_size=10000000,  # Use 10M sentences for training
    shuffle_input_sentence=True,
    max_sentence_length=16384,
    train_extremely_large_corpus=True,
)

elapsed = time.time() - t0
print(f"Tokenizer trained in {elapsed:.0f}s", flush=True)
print(f"Model saved: {model_prefix}.model", flush=True)

# Step 3: Verify
sp4096 = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
print(f"Vocab size: {sp4096.vocab_size()}", flush=True)

# Test encoding
test_text = "The quick brown fox jumps over the lazy dog."
tokens_1024 = sp1024.encode(test_text)
tokens_4096 = sp4096.encode(test_text)
print(f"Test: '{test_text}'")
print(f"  SP1024: {len(tokens_1024)} tokens")
print(f"  SP4096: {len(tokens_4096)} tokens")
print(f"  Compression ratio: {len(tokens_1024)/len(tokens_4096):.2f}x")

# Cleanup
# os.remove(text_file)  # Keep for re-encoding
print(f"\n=== Done in {(time.time()-t0)/60:.1f} min ===", flush=True)
print(f"Next: Run re-encode script to create SP4096 shards", flush=True)
