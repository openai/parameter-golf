"""
Re-tokenize FineWeb corpus with a gravity tokenizer.

Takes the raw text (decoded from existing shards) and re-encodes it using
a gravity-weighted SentencePiece model. Outputs binary shard files in the
same format the Parameter Golf training script expects.

Usage:
    python scripts/retokenize_corpus.py \
        --base-tokenizer ./parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
        --gravity-tokenizer data/tokenizers/gravity_beta_0.3.model \
        --data-dir ./parameter-golf/data/datasets/fineweb10B_sp1024 \
        --output-dir ./parameter-golf/data/datasets/fineweb_gravity_beta_0.3 \
        --max-shards 10
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
SHARD_SIZE = 100_000_000  # tokens per shard (same as parameter-golf default)


def load_shard_tokens(path: Path) -> np.ndarray:
    """Load tokens from a binary shard file."""
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != SHARD_MAGIC:
        raise ValueError(f"Bad shard header: {path}")
    num_tokens = int(header[2])
    header_bytes = HEADER_INTS * np.dtype("<i4").itemsize
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens


def write_shard(path: Path, tokens: np.ndarray):
    """Write tokens to a binary shard file in parameter-golf format."""
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = len(tokens)

    token_u16 = tokens.astype("<u2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(token_u16.tobytes())


def decode_shard_to_text(shard_path: Path, sp: spm.SentencePieceProcessor) -> str:
    """Decode a tokenized shard back to raw text."""
    token_ids = load_shard_tokens(shard_path)
    # Decode in chunks
    chunk_size = 100_000
    text_parts = []
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i + chunk_size].tolist()
        text_parts.append(sp.decode(chunk))
    return "".join(text_parts)


def encode_text_to_tokens(text: str, sp: spm.SentencePieceProcessor) -> np.ndarray:
    """Encode text to token IDs using SentencePiece."""
    token_ids = sp.encode(text)
    return np.array(token_ids, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser(description="Re-tokenize corpus")
    parser.add_argument("--base-tokenizer", type=str, required=True,
                        help="Path to base SentencePiece model (for decoding)")
    parser.add_argument("--gravity-tokenizer", type=str, required=True,
                        help="Path to gravity SentencePiece model (for encoding)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with original tokenized shards")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for re-tokenized shards")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="Max shards to process (0=all)")
    args = parser.parse_args()

    sp_base = spm.SentencePieceProcessor(model_file=args.base_tokenizer)
    sp_gravity = spm.SentencePieceProcessor(model_file=args.gravity_tokenizer)

    print(f"Base tokenizer: vocab={sp_base.vocab_size()}")
    print(f"Gravity tokenizer: vocab={sp_gravity.vocab_size()}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Process training shards
    train_shards = sorted(data_dir.glob("fineweb_train_*.bin"))
    val_shards = sorted(data_dir.glob("fineweb_val_*.bin"))

    if args.max_shards > 0:
        train_shards = train_shards[:args.max_shards]

    print(f"\nProcessing {len(train_shards)} training shards + {len(val_shards)} val shards")

    total_base_tokens = 0
    total_gravity_tokens = 0

    for shard_path in tqdm(train_shards + val_shards, desc="Re-tokenizing"):
        print(f"\n  {shard_path.name}:")

        # Decode to text
        text = decode_shard_to_text(shard_path, sp_base)
        base_tokens_count = len(load_shard_tokens(shard_path))

        # Re-encode with gravity tokenizer
        gravity_tokens = encode_text_to_tokens(text, sp_gravity)
        gravity_tokens_count = len(gravity_tokens)

        total_base_tokens += base_tokens_count
        total_gravity_tokens += gravity_tokens_count

        ratio = gravity_tokens_count / base_tokens_count if base_tokens_count > 0 else 0
        print(f"    Base tokens: {base_tokens_count:,}")
        print(f"    Gravity tokens: {gravity_tokens_count:,} ({ratio:.2f}x)")

        # Check that all token IDs fit in uint16
        if gravity_tokens.max() >= 65536:
            print(f"    WARNING: Token IDs exceed uint16 range!")

        # Write output shard
        output_path = output_dir / shard_path.name
        write_shard(output_path, gravity_tokens)

    # Summary
    overall_ratio = total_gravity_tokens / total_base_tokens if total_base_tokens > 0 else 0
    print(f"\n{'='*60}")
    print(f"Re-tokenization complete")
    print(f"  Total base tokens: {total_base_tokens:,}")
    print(f"  Total gravity tokens: {total_gravity_tokens:,}")
    print(f"  Sequence length ratio: {overall_ratio:.3f}x")
    print(f"  (>1.0 means gravity produces longer sequences)")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
