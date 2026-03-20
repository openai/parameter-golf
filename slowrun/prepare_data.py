"""
Preprocess FineWeb into pre-tokenized train/val splits using the GPT-2 tokenizer.

Usage:
    python prepare_data.py
    python prepare_data.py --train_tokens 100_000_000 --val_tokens 10_000_000 --local_dir fineweb_data
"""

import os
import hashlib
import argparse
import numpy as np
import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Constants

SEQUENCE_LENGTH = 2048
SEQUENCE_SIZE = SEQUENCE_LENGTH + 1  # input + target
BATCH_SIZE = 16  # device batch size, used for chunking

# Expected SHA-256 hashes
EXPECTED_HASHES = {
    "fineweb_val.pt": "80e7e430d3a7d10892c2ff32579370c5b65fbe833579d7ea5d55cbd0504c8462",
    "fineweb_train.pt": "e7e089aedbccb6865ce76c78453fa473c823969846fccd4000f5f13aef54e70e",
}

# -----------------------------------------------------------------------------
# Helpers

def tokenize_documents(dataset_iter, encoder, total_tokens):
    """Tokenize documents from an iterator until we have total_tokens tokens."""
    eot = encoder._special_tokens['<|endoftext|>']
    tokens = []
    pbar = tqdm(total=total_tokens, unit="tok")
    for doc in dataset_iter:
        doc_tokens = [eot] + encoder.encode_ordinary(doc["text"])
        tokens.extend(doc_tokens)
        pbar.update(len(doc_tokens))
        if len(tokens) >= total_tokens:
            tokens = tokens[:total_tokens]
            break
    pbar.close()
    return tokens


def create_sequences(tokens, sequence_size):
    """Split a flat token list into fixed-size sequences, discarding any remainder."""
    tokens = np.array(tokens, dtype=np.uint16)
    num_sequences = len(tokens) // sequence_size
    tokens = tokens[:num_sequences * sequence_size]
    sequences = tokens.reshape(num_sequences, sequence_size)
    return sequences


def write_datafile(filename, sequences, batch_size):
    """
    Write sequences to a chunked .pt file with padding metadata.

    Format:
    {
        'chunks': List[Tensor],       # each chunk is batch_size * sequence_size tokens
        'valid_counts': List[int],     # real (non-padding) sequences per chunk
        'batch_size': int,
        'sequence_size': int,
    }
    """
    if len(sequences) == 0:
        print(f"Warning: no sequences to write to {filename}")
        return

    sequence_size = sequences.shape[1]
    num_sequences = len(sequences)
    num_full_batches = num_sequences // batch_size
    leftover = num_sequences % batch_size

    chunks = []
    valid_counts = []

    # Full batches
    for i in range(num_full_batches):
        start = i * batch_size
        chunk = sequences[start:start + batch_size].reshape(-1)
        chunks.append(chunk)
        valid_counts.append(batch_size)

    # Leftover with zero-padding
    if leftover > 0:
        leftover_data = sequences[num_full_batches * batch_size:]
        padding = np.zeros((batch_size - leftover, sequence_size), dtype=np.uint16)
        padded = np.concatenate([leftover_data, padding], axis=0).reshape(-1)
        chunks.append(padded)
        valid_counts.append(leftover)

    print(f"Writing {len(chunks):,} chunks to {filename}")
    print(f"  {num_sequences:,} sequences ({num_full_batches} full batches of {batch_size})")
    if leftover > 0:
        print(f"  Last chunk: {leftover}/{batch_size} valid, {batch_size - leftover} padded")

    data = {
        'chunks': [torch.from_numpy(chunk.copy()) for chunk in chunks],
        'valid_counts': valid_counts,
        'batch_size': batch_size,
        'sequence_size': sequence_size,
    }
    torch.save(data, filename)


def sha256_file(filepath):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(filepath):
    """Check file hash against expected value. Print actual hash if mismatch or unset."""
    basename = os.path.basename(filepath)
    actual = sha256_file(filepath)
    expected = EXPECTED_HASHES.get(basename)
    if expected is None:
        print(f"  Hash for {basename}: {actual}")
        print(f"  (no expected hash set — paste this value into EXPECTED_HASHES to lock it in)")
    elif actual == expected:
        print(f"  Hash OK for {basename}: {actual}")
    else:
        print(f"  HASH MISMATCH for {basename}!")
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")


# -----------------------------------------------------------------------------
# Main

def preprocess(train_tokens, val_tokens, local_dir):
    encoder = tiktoken.get_encoding("gpt2")

    val_seqs = val_tokens // SEQUENCE_SIZE
    train_seqs = train_tokens // SEQUENCE_SIZE

    print(f"{'='*60}")
    print(f"Preprocessing FineWeb with GPT-2 tokenizer")
    print(f"{'='*60}")
    print(f"Sequence length: {SEQUENCE_LENGTH} (size {SEQUENCE_SIZE})")
    print(f"Val:   {val_tokens:>13,} raw tokens -> {val_seqs:,} sequences")
    print(f"Train: {train_tokens:>13,} raw tokens -> {train_seqs:,} sequences")
    print(f"Output: {local_dir}/")
    print(f"{'='*60}")

    os.makedirs(local_dir, exist_ok=True)

    # Stream dataset
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    dataset_iter = iter(dataset)

    # Pool 1: val
    print(f"\nTokenizing val ({val_tokens:,} tokens)...")
    val_raw = tokenize_documents(dataset_iter, encoder, val_tokens)
    val_sequences = create_sequences(val_raw, SEQUENCE_SIZE)
    np.random.seed(42)
    np.random.shuffle(val_sequences)
    print(f"  {len(val_sequences):,} val sequences ({len(val_sequences) * SEQUENCE_LENGTH:,} trainable tokens)")

    # Pool 2: train (continues from same iterator, no document overlap)
    print(f"\nTokenizing train ({train_tokens:,} tokens)...")
    train_raw = tokenize_documents(dataset_iter, encoder, train_tokens)
    train_sequences = create_sequences(train_raw, SEQUENCE_SIZE)
    np.random.seed(43)
    np.random.shuffle(train_sequences)
    print(f"  {len(train_sequences):,} train sequences ({len(train_sequences) * SEQUENCE_LENGTH:,} trainable tokens)")

    # Write
    print()
    val_path = os.path.join(local_dir, "fineweb_val.pt")
    train_path = os.path.join(local_dir, "fineweb_train.pt")
    write_datafile(val_path, val_sequences, BATCH_SIZE)
    write_datafile(train_path, train_sequences, BATCH_SIZE)

    # Verify hashes
    print()
    verify_hash(val_path)
    verify_hash(train_path)

    print(f"\nDone! Files saved to {local_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FineWeb with GPT-2 tokenizer")
    parser.add_argument("--train_tokens", type=int, default=100_000_000)
    parser.add_argument("--val_tokens", type=int, default=10_000_000)
    parser.add_argument("--local_dir", type=str, default="fineweb_data")
    args = parser.parse_args()

    preprocess(
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        local_dir=args.local_dir,
    )