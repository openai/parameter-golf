#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate mock tokenizer and shards for local smoke tests")
    parser.add_argument("--output-root", default="data", help="Root directory to place tokenizers/ and datasets/")
    parser.add_argument("--dataset-name", default="mock_sp1024", help="Dataset directory name under data/datasets/")
    parser.add_argument("--vocab-size", type=int, default=1024, help="SentencePiece vocab size")
    parser.add_argument("--train-tokens", type=int, default=400_000, help="Number of train tokens to emit")
    parser.add_argument("--val-tokens", type=int, default=200_000, help="Number of val tokens to emit")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for synthetic token generation")
    return parser


def write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = int(tokens.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        header.tofile(handle)
        tokens.astype("<u2", copy=False).tofile(handle)


def alpha_token(value: int) -> str:
    letters = []
    n = value
    while True:
        n, rem = divmod(n, 26)
        letters.append(chr(ord("a") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(letters))


def build_corpus(path: Path, *, lines: int = 200_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(lines):
            uniq_a = alpha_token(idx)
            uniq_b = alpha_token(idx * 7 + 11)
            uniq_c = alpha_token(idx * 17 + 23)
            handle.write(
                " ".join(
                    [
                        f"doc{idx}",
                        f"lex{uniq_a}",
                        f"morph{uniq_b}",
                        f"gram{uniq_c}",
                        "alpha",
                        "beta",
                        "gamma",
                        "delta",
                        "epsilon",
                        "zeta",
                        "eta",
                        "theta",
                        "iota",
                        "kappa",
                        "lambda",
                        "mu",
                        "nu",
                        "xi",
                        "omicron",
                        "pi",
                        "rho",
                        "sigma",
                        "tau",
                        "upsilon",
                        "phi",
                        "chi",
                        "psi",
                        "omega",
                        f"blend{uniq_a[:4]}{uniq_b[-4:]}",
                        f"swap{uniq_b[:3]}{uniq_c[-3:]}",
                        f"stem{uniq_c[:5]}",
                        f"n{idx % 997}",
                        f"m{idx % 149}",
                        f"k{idx % 53}",
                    ]
                )
                + "\n"
            )


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets" / args.dataset_name
    corpus_path = output_root / "mock_corpus.txt"
    model_prefix = tokenizers_dir / "mock_1024_bpe"

    build_corpus(corpus_path)
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=200_000,
        shuffle_input_sentence=True,
        split_digits=True,
    )

    rng = np.random.default_rng(args.seed)
    train_tokens = rng.integers(0, args.vocab_size, size=args.train_tokens, dtype=np.uint16)
    val_tokens = rng.integers(0, args.vocab_size, size=args.val_tokens, dtype=np.uint16)
    write_shard(datasets_dir / "fineweb_train_000000.bin", train_tokens)
    write_shard(datasets_dir / "fineweb_val_000000.bin", val_tokens)

    print(f"DATA_PATH={datasets_dir}")
    print(f"TOKENIZER_PATH={model_prefix.with_suffix('.model')}")


if __name__ == "__main__":
    main()