#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontier_tokenizer import (
    TOKENIZER_SPECS_PATH,
    TokenizerVariantSpec,
    append_tokenizer_variant_spec,
    tokenizer_model_path_for_suffix,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a deterministic tokenizer variant spec for frontier experiments.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset-suffix", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--spec-path", default=str(TOKENIZER_SPECS_PATH))
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()

    spec_path = Path(args.spec_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.is_file() else {"tokenizers": []}
    spec = TokenizerVariantSpec(
        name=args.name,
        dataset_suffix=args.dataset_suffix,
        vocab_size=args.vocab_size,
        model_path=args.model_path or tokenizer_model_path_for_suffix(args.dataset_suffix, args.vocab_size),
        dataset_path=args.dataset_path,
        notes=tuple(args.note),
    )
    updated = append_tokenizer_variant_spec(payload, spec)
    spec_path.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"updated_spec_path": str(spec_path), "tokenizer_name": spec.name}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

