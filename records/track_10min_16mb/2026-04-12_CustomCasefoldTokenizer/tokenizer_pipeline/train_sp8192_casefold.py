"""Train case-folded SP8192 BPE tokenizer.

Identical to train_sp8192.py but lowercases all text before training.
This collapses case variants ("the", "The", "THE") into shared tokens,
freeing ~21% of learned vocab slots for other subwords.

Usage:
    uv run data/train_sp8192_casefold.py
    uv run data/train_sp8192_casefold.py --vocab-size 8192
"""
import argparse
import functools
import json
import time
import unicodedata
from pathlib import Path

import sentencepiece as spm

print = functools.partial(print, flush=True)

DATA_DIR = Path(__file__).parent
DOCS_PATH = DATA_DIR / "docs_selected.jsonl"
TOKENIZERS_DIR = DATA_DIR / "tokenizers"


def iter_docs_casefold(path: Path, max_docs: int = 0):
    """Yield lowercased text from JSONL docs.

    NFKC first because SP applies it internally — decompositions like
    ½ → 1⁄2, Ĳ → IJ must happen before our lower().
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                yield unicodedata.normalize("NFKC", text).lower()
                count += 1
                if max_docs > 0 and count >= max_docs:
                    break
    print(f"  Yielded {count:,} casefold docs to SentencePiece trainer")


def train_sp_casefold(
    vocab_size: int = 8192,
    byte_fallback: bool = True,
    train_docs: int = 0,
) -> Path:
    TOKENIZERS_DIR.mkdir(parents=True, exist_ok=True)

    bf_tag = "" if byte_fallback else "_nobf"
    output_prefix = f"fineweb_{vocab_size}_bpe_casefold_raw{bf_tag}"
    prefix = TOKENIZERS_DIR / output_prefix
    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")

    for p in (model_path, vocab_path):
        if p.exists():
            p.unlink()

    print(f"Training CASEFOLD SP BPE vocab_size={vocab_size} byte_fallback={byte_fallback}")
    print(f"  Input: {DOCS_PATH} ({DOCS_PATH.stat().st_size / 1e9:.1f} GB)")
    print(f"  Output: {model_path}")

    # Identical SP params to train_sp8192.py — only the text is lowercased
    kwargs = {
        "sentence_iterator": iter_docs_casefold(DOCS_PATH, max_docs=train_docs),
        "model_prefix": str(prefix),
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "character_coverage": 0.999,
        "byte_fallback": byte_fallback,
        "split_digits": True,
        "normalization_rule_name": "nmt_nfkc",
        "add_dummy_prefix": False,
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
        "hard_vocab_limit": False,
        "num_threads": 8,
        "input_sentence_size": 10_000_000 if train_docs == 0 else 0,
        "shuffle_input_sentence": True,
    }

    t0 = time.perf_counter()
    spm.SentencePieceTrainer.train(**kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  Training completed in {elapsed:.1f}s")

    # Verify
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    actual_vocab = sp.vocab_size()
    byte_tokens = sum(1 for i in range(actual_vocab) if sp.is_byte(i))
    special_tokens = sum(1 for i in range(actual_vocab) if sp.is_control(i) or sp.is_unknown(i))
    learned_tokens = actual_vocab - byte_tokens - special_tokens

    print(f"\n  Result:")
    print(f"    Actual vocab size: {actual_vocab}")
    print(f"    Byte fallback tokens: {byte_tokens}")
    print(f"    Special tokens: {special_tokens}")
    print(f"    Learned tokens: {learned_tokens}")
    print(f"    Model file: {model_path} ({model_path.stat().st_size:,} bytes)")

    # Compression test on lowercased sample
    test = "the quick brown fox jumps over the lazy dog."
    tokens = sp.encode(test)
    test_bytes = len(test.encode("utf-8"))
    print(f"    Sample tok/byte: {len(tokens)/test_bytes:.4f}")

    # Quick case-variant check: should have zero uppercase learned tokens
    upper_count = 0
    for i in range(actual_vocab):
        if sp.is_byte(i) or sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i):
            continue
        piece = sp.id_to_piece(i).replace("\u2581", "")
        if piece != piece.lower():
            upper_count += 1
    print(f"    Tokens with uppercase chars: {upper_count} (should be 0 or near-0)")

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train case-folded SP BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--train-docs", type=int, default=0)
    parser.add_argument("--no-byte-fallback", action="store_true")
    args = parser.parse_args()

    train_sp_casefold(
        vocab_size=args.vocab_size,
        byte_fallback=not args.no_byte_fallback,
        train_docs=args.train_docs,
    )


if __name__ == "__main__":
    main()
