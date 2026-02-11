"""Train a 4k-vocab SentencePiece tokenizer on FineWeb text."""

import argparse
import os

import sentencepiece as spm
from datasets import load_dataset


def iter_text(dataset, limit: int):
    for i, row in enumerate(dataset):
        if i >= limit:
            return
        text = row["text"].replace("\x00", " ").strip()
        if text:
            yield text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer on FineWeb")
    parser.add_argument("-v", "--version", type=str, default="10B", choices=["10B", "100B"])
    parser.add_argument("--num_docs", type=int, default=2_000_000)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument(
        "--model_prefix",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "tokenizers", "fineweb_4k_bpe"),
    )
    args = parser.parse_args()

    remote_name = "sample-10BT" if args.version == "10B" else "sample-100BT"
    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)

    dataset = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter_text(dataset, args.num_docs),
        model_prefix=args.model_prefix,
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=0.999,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="nmt_nfkc",
        add_dummy_prefix=False,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
    )

    print(f"Wrote {args.model_prefix}.model and {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()
