"""Train a custom SentencePiece tokenizer and retokenize docs_selected.jsonl into binary shards.

Usage:
    # Full pipeline: train tokenizer + retokenize
    uv run data/retokenize.py

    # Only train tokenizer
    uv run data/retokenize.py --skip-retokenize

    # Only retokenize (reuse existing .model)
    uv run data/retokenize.py --skip-train-tokenizer

    # Limit training shards for faster iteration
    uv run data/retokenize.py --train-shards 10
"""
import argparse
import functools
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

# Force unbuffered output so progress is visible when piped
print = functools.partial(print, flush=True)

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
DEFAULT_SHARD_SIZE = 100_000_000  # tokens per shard (from manifest)
DEFAULT_NUM_VAL_DOCS = 50_000
DEFAULT_TOKENIZER_TRAIN_DOCS = 5_000_000
DEFAULT_SHUFFLE_SEED = 1337

USER_DEFINED_SYMBOLS = [
    "http://", "https://", "www.",
    ".com", ".org", ".net",
    ".gov", ".html", ".edu", ".co.uk",
]


def iter_jsonl_texts(path: Path, limit: int | None = None):
    """Yield document texts from a JSONL file, one per line."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                if text:
                    yield text
                    count += 1
            except json.JSONDecodeError:
                continue


def train_tokenizer(
    input_path: Path,
    model_prefix: str,
    vocab_size: int,
    num_train_docs: int,
):
    """Train a SentencePiece BPE tokenizer on the first num_train_docs documents."""
    import sentencepiece as spm

    print(f"Training SentencePiece BPE tokenizer (vocab_size={vocab_size})...")
    print(f"  Training docs: {num_train_docs:,}")
    print(f"  user_defined_symbols: {USER_DEFINED_SYMBOLS}")
    print(f"  split_digits: False")
    print(f"  max_sentence_length: 16384")
    print(f"  Output: {model_prefix}.model")

    # Create an iterator that yields text strings for training
    doc_iter = iter_jsonl_texts(input_path, limit=num_train_docs)

    t0 = time.time()
    spm.SentencePieceTrainer.train(
        sentence_iterator=doc_iter,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        # User's requested settings
        split_digits=False,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        max_sentence_length=16384,
        # Required for byte-level coverage (BPB scoring uses sp.is_byte())
        byte_fallback=True,
        character_coverage=0.9995,
        # Match original tokenizer control token IDs
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        # Performance
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=True,
    )
    elapsed = time.time() - t0
    print(f"  Tokenizer trained in {elapsed:.1f}s")

    # Verify
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    actual_vocab = sp.vocab_size()
    print(f"  Actual vocab size: {actual_vocab}")
    if actual_vocab != vocab_size:
        print(f"  WARNING: vocab size mismatch! Expected {vocab_size}, got {actual_vocab}")
        sys.exit(1)

    # Verify user_defined_symbols are single tokens
    for sym in USER_DEFINED_SYMBOLS:
        tokens = sp.encode(sym, out_type=str)
        token_strs = [t.replace("\u2581", "_") for t in tokens]  # safe for Windows console
        if len(tokens) != 1 or tokens[0].replace("\u2581", "") != sym:
            encoded_ids = sp.encode(sym)
            print(f"  Note: '{sym}' encodes as {token_strs} (ids: {encoded_ids})")

    print("  Tokenizer training complete.")
    return sp


class ShardWriter:
    """Accumulates token IDs and writes shards in the competition binary format.

    Tracks true UTF-8 byte counts from original text and stores them in header[3].
    This enables exact BPB calculation regardless of tokenizer byte-counting quirks.
    """

    def __init__(self, output_dir: Path, prefix: str, shard_size: int):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.shard_idx = 0
        self.buffer = []
        self.buffer_len = 0
        self.total_tokens = 0
        self.total_true_bytes = 0
        # Track true bytes per unflushed buffer region
        self._pending_true_bytes = 0

    def add_tokens(self, ids: list[int], true_bytes: int = 0):
        """Add token IDs with their corresponding true UTF-8 byte count."""
        self.buffer.extend(ids)
        self.buffer_len += len(ids)
        self._pending_true_bytes += true_bytes
        while self.buffer_len >= self.shard_size:
            # Proportionally split true_bytes between this shard and remainder
            frac = self.shard_size / self.buffer_len
            shard_bytes = int(self._pending_true_bytes * frac)
            self._flush_shard(self.shard_size, shard_bytes)
            self._pending_true_bytes -= shard_bytes

    def _flush_shard(self, count: int, true_bytes: int = 0):
        tokens = np.array(self.buffer[:count], dtype="<u2")
        self.buffer = self.buffer[count:]
        self.buffer_len = len(self.buffer)
        self._write_shard(tokens, true_bytes)

    def _write_shard(self, tokens: np.ndarray, true_bytes: int = 0):
        path = self.output_dir / f"{self.prefix}_{self.shard_idx:06d}.bin"
        header = np.zeros(HEADER_INTS, dtype="<i4")
        header[0] = SHARD_MAGIC
        header[1] = SHARD_VERSION
        header[2] = len(tokens)
        header[3] = true_bytes  # true UTF-8 byte count for this shard
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens.tobytes())
        self.total_tokens += len(tokens)
        self.total_true_bytes += true_bytes
        self.shard_idx += 1
        if true_bytes > 0:
            print(f"    Shard {self.shard_idx - 1}: {len(tokens):,} tokens, {true_bytes:,} true_bytes")

    def close(self):
        """Flush any remaining tokens as a final shard."""
        if self.buffer_len > 0:
            tokens = np.array(self.buffer, dtype="<u2")
            self._write_shard(tokens, self._pending_true_bytes)
            self.buffer = []
            self.buffer_len = 0
            self._pending_true_bytes = 0


_DIGIT_RE = re.compile(r'[1-9]')

def _apply_normalize(text: str, normalize_mode: str) -> str:
    """Apply text normalization before tokenization.

    NFKC is applied first because SentencePiece (nmt_nfkc) applies it
    internally after we return.  NFKC decomposes characters like ½ → 1⁄2,
    ² → 2, Ĳ → IJ — reintroducing uppercase and non-zero digits.  By
    running NFKC ourselves first, our .lower() and digit regex see the
    decomposed forms, and SP's subsequent NFKC pass is a no-op.
    """
    import unicodedata
    if normalize_mode == "casefold":
        return unicodedata.normalize("NFKC", text).lower()
    elif normalize_mode == "casefold_digits":
        return _DIGIT_RE.sub('0', unicodedata.normalize("NFKC", text).lower())
    return text


def _encode_batch(args_tuple):
    """Worker function: encode a batch of texts, return [(ids, true_bytes), ...]."""
    texts_batch, tokenizer_path_str, normalize_mode = args_tuple
    import sentencepiece as _spm
    _sp = _spm.SentencePieceProcessor(model_file=tokenizer_path_str)
    results = []
    for text in texts_batch:
        text = _apply_normalize(text, normalize_mode)
        ids = _sp.encode(text)
        true_bytes = len(text.encode("utf-8"))
        results.append((ids, true_bytes))
    return results


def retokenize(
    input_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    shard_size: int,
    num_val_docs: int,
    shuffle_seed: int,
    max_train_shards: int | None,
    sequential_val: bool = False,
    casefold: bool = False,
    normalize: str = "",
):
    """Retokenize the full corpus into binary shards."""
    import sentencepiece as spm

    # Resolve normalize mode: --normalize takes precedence over --casefold
    normalize_mode = normalize or ("casefold" if casefold else "")

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    print(f"Retokenizing with {tokenizer_path} (vocab_size={sp.vocab_size()})")
    print(f"  Output: {output_dir}")
    print(f"  Shard size: {shard_size:,} tokens")
    print(f"  Val docs: {num_val_docs:,}")
    print(f"  Shuffle seed: {shuffle_seed}")
    if normalize_mode:
        print(f"  Normalize: {normalize_mode}")

    # Two-pass streaming approach: avoids loading all 15M+ docs into memory.
    # Pass 1: count docs + build val index set (shuffle integer indices, not text).
    # Pass 2: stream JSONL, tokenize each doc, route to val or train writer.

    print("  Pass 1: Counting documents...")
    t0 = time.time()
    num_docs = 0
    for _ in iter_jsonl_texts(input_path):
        num_docs += 1
    print(f"  Counted {num_docs:,} documents in {time.time() - t0:.1f}s")

    if sequential_val:
        # Match upstream convention: first num_val_docs = val, rest = train
        print(f"  Sequential val split: first {num_val_docs:,} docs = val")
        val_indices = set(range(num_val_docs))
    else:
        print(f"  Building val index set (shuffling {num_docs:,} indices with seed {shuffle_seed})...")
        indices = list(range(num_docs))
        random.seed(shuffle_seed)
        random.shuffle(indices)
        val_indices = set(indices[:num_val_docs])
        del indices  # free the list (~120MB), keep only the set (~2MB)
    print(f"  Val set: {len(val_indices):,} doc indices")

    output_dir.mkdir(parents=True, exist_ok=True)
    max_train_tokens = max_train_shards * shard_size if max_train_shards else None

    # Pass 2: Multiprocessing tokenization
    # SentencePiece encode does NOT release the GIL, so threads give zero
    # parallelism. Use process pool with chunked batches instead.
    from multiprocessing import Pool

    num_workers = os.cpu_count() or 4
    CHUNK_SIZE = 10_000  # docs per chunk submitted to pool
    print(f"\n  Pass 2: Multiprocess tokenization ({num_workers} workers, chunk={CHUNK_SIZE})...")
    if max_train_shards:
        print(f"    Limiting training to {max_train_shards} shards ({max_train_tokens:,} tokens)")
    val_writer = ShardWriter(output_dir, "fineweb_val", shard_size)
    train_writer = ShardWriter(output_dir, "fineweb_train", shard_size)
    val_count = 0
    train_count = 0
    train_stopped = False
    t0 = time.time()

    tokenizer_path_str = str(tokenizer_path)

    def _process_chunk(chunk):
        """Submit chunk to pool, collect ordered results, route to writers."""
        nonlocal doc_idx, val_count, train_count, train_stopped
        # Split into sub-batches for workers (each worker encodes ~chunk/workers docs)
        batch_size = max(len(chunk) // num_workers, 100)
        batches = []
        for i in range(0, len(chunk), batch_size):
            batches.append((chunk[i:i+batch_size], tokenizer_path_str, normalize_mode))
        # Map preserves order
        for batch_results in pool.imap(_encode_batch, batches):
            for ids, true_bytes in batch_results:
                if doc_idx in val_indices:
                    val_writer.add_tokens(ids, true_bytes=true_bytes)
                    val_count += 1
                    if val_count % 10_000 == 0:
                        print(f"    val: {val_count:,}/{num_val_docs:,} docs")
                elif not train_stopped:
                    train_writer.add_tokens(ids, true_bytes=true_bytes)
                    train_count += 1
                    if max_train_tokens and train_writer.total_tokens + train_writer.buffer_len >= max_train_tokens:
                        train_stopped = True
                        print(f"    train: reached {max_train_shards} shard limit at doc {train_count:,}")
                    if train_count % 100_000 == 0:
                        elapsed = time.time() - t0
                        total_done = val_count + train_count
                        rate = total_done / elapsed if elapsed > 0 else 0
                        remaining = (num_docs - doc_idx) / rate if rate > 0 else 0
                        print(f"    progress: {total_done:,}/{num_docs:,} docs "
                              f"(val:{val_count:,} train:{train_count:,}) "
                              f"{elapsed:.0f}s, ~{remaining:.0f}s remaining")
                doc_idx += 1

    doc_idx = 0
    chunk = []
    with Pool(processes=num_workers) as pool:
        for text in iter_jsonl_texts(input_path):
            chunk.append(text)
            if len(chunk) >= CHUNK_SIZE:
                _process_chunk(chunk)
                chunk = []
        if chunk:
            _process_chunk(chunk)

    val_writer.close()
    train_writer.close()
    elapsed = time.time() - t0
    print(f"\n  Tokenization complete in {elapsed:.0f}s")

    # Summary
    total = val_writer.total_tokens + train_writer.total_tokens
    print(f"\n  Summary:")
    print(f"    Total tokens: {total:,}")
    print(f"    Val tokens:   {val_writer.total_tokens:,} ({val_writer.shard_idx} shards)")
    print(f"    Val true_bytes: {val_writer.total_true_bytes:,}")
    print(f"    Train tokens: {train_writer.total_tokens:,} ({train_writer.shard_idx} shards)")
    print(f"    Avg tokens/doc (val): {val_writer.total_tokens / max(val_count, 1):,.1f}")
    print(f"    Val docs: {val_count:,}, Train docs: {train_count:,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train custom tokenizer and retokenize corpus")
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).parent / "docs_selected.jsonl",
        help="Path to docs_selected.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "datasets" / "fineweb10B_sp1024_custom",
        help="Output directory for binary shards",
    )
    parser.add_argument(
        "--tokenizer-prefix", type=str,
        default=str(Path(__file__).parent / "tokenizers" / "fineweb_1024_custom"),
        help="Output prefix for tokenizer model (without .model extension)",
    )
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--train-docs", type=int, default=DEFAULT_TOKENIZER_TRAIN_DOCS,
                        help="Number of docs for tokenizer training")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                        help="Tokens per shard")
    parser.add_argument("--num-val-docs", type=int, default=DEFAULT_NUM_VAL_DOCS)
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--train-shards", type=int, default=None,
                        help="Limit number of training shards to produce")
    parser.add_argument("--skip-train-tokenizer", action="store_true",
                        help="Skip tokenizer training, reuse existing .model")
    parser.add_argument("--skip-retokenize", action="store_true",
                        help="Only train tokenizer, skip retokenization")
    parser.add_argument("--sequential-val", action="store_true",
                        help="Use sequential val split (first N docs = val) to match upstream")
    parser.add_argument("--casefold", action="store_true",
                        help="Lowercase text before encoding (for case-folded tokenizers)")
    parser.add_argument("--normalize", type=str, default="",
                        choices=["", "casefold", "casefold_digits"],
                        help="Text normalization mode (overrides --casefold)")
    return parser


def main():
    args = build_parser().parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found. Run:")
        print(f"  uv run data/cached_challenge_fineweb.py --with-docs --train-shards 0")
        sys.exit(1)

    tokenizer_model_path = Path(f"{args.tokenizer_prefix}.model")

    # Phase A: Train tokenizer
    if not args.skip_train_tokenizer:
        Path(args.tokenizer_prefix).parent.mkdir(parents=True, exist_ok=True)
        train_tokenizer(
            input_path=args.input,
            model_prefix=args.tokenizer_prefix,
            vocab_size=args.vocab_size,
            num_train_docs=args.train_docs,
        )
    else:
        if not tokenizer_model_path.exists():
            print(f"ERROR: --skip-train-tokenizer but {tokenizer_model_path} not found")
            sys.exit(1)
        print(f"Skipping tokenizer training, using {tokenizer_model_path}")

    # Phase B: Retokenize
    if not args.skip_retokenize:
        retokenize(
            input_path=args.input,
            tokenizer_path=tokenizer_model_path,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            num_val_docs=args.num_val_docs,
            shuffle_seed=args.shuffle_seed,
            max_train_shards=args.train_shards,
            sequential_val=args.sequential_val,
            casefold=args.casefold,
            normalize=args.normalize,
        )
    else:
        print("Skipping retokenization.")

    print("\nDone! To train with the custom tokenizer:")
    print(f"  TOKENIZER_PATH={tokenizer_model_path} \\")
    print(f"  DATA_PATH={args.output_dir} \\")
    print(f"  torchrun --standalone --nproc_per_node=1 train_gpt.py")


if __name__ == "__main__":
    main()
