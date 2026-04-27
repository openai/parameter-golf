#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontier_tokenizer import (
    DATA_MANIFEST_PATH,
    dataset_name_for_suffix,
    dataset_path_for_suffix,
    recommended_bigram_vocab_size,
)


def _upsert_named(items: list[dict[str, object]], item: dict[str, object]) -> list[dict[str, object]]:
    name = item.get("name")
    return [existing for existing in items if existing.get("name") != name] + [item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a manifest-backed dataset/tokenizer pair for a tokenizer variant.")
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--dataset-suffix", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--manifest-path", default=str(DATA_MANIFEST_PATH))
    parser.add_argument("--status", default="planned")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"tokenizers": [], "datasets": []}

    dataset_name = dataset_name_for_suffix(args.dataset_suffix)
    dataset_path = dataset_path_for_suffix(args.dataset_suffix)
    tokenizer_model = f"tokenizers/fineweb_{args.vocab_size}_bpe.model"
    tokenizer_vocab = f"tokenizers/fineweb_{args.vocab_size}_bpe.vocab"

    tokenizer_entry = {
        "name": args.tokenizer_name,
        "kind": "sentencepiece_bpe",
        "vocab_size": args.vocab_size,
        "bos_id": 1,
        "eos_id": 2,
        "recommended_bigram_vocab_size": recommended_bigram_vocab_size(args.vocab_size),
        "model_path": tokenizer_model,
        "vocab_path": tokenizer_vocab,
        "status": args.status,
    }
    dataset_entry = {
        "name": dataset_name,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_kind": "sentencepiece_bpe",
        "path": dataset_path,
        "train_glob": f"{dataset_path}/fineweb_train_*.bin",
        "val_glob": f"{dataset_path}/fineweb_val_*.bin",
        "vocab_size": args.vocab_size,
        "bos_id": 1,
        "eos_id": 2,
        "recommended_bigram_vocab_size": recommended_bigram_vocab_size(args.vocab_size),
        "status": args.status,
    }

    manifest["tokenizers"] = _upsert_named(list(manifest.get("tokenizers", [])), tokenizer_entry)
    manifest["datasets"] = _upsert_named(list(manifest.get("datasets", [])), dataset_entry)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "dataset_name": dataset_name}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

