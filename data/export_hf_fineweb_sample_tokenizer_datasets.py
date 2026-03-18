"""Export matched tokenizer datasets from a Hugging Face FineWeb sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset
import tiktoken
from tqdm import tqdm

from export_matched_fineweb_tokenizer_datasets import DEMO_CONFIG, build_parser as build_matched_parser
from export_matched_fineweb_tokenizer_datasets import run_export as run_matched_export


DEFAULT_DATASET_NAME = "sample-350BT"
DEFAULT_OUTPUT_ROOT = "/tmp/fineweb_sample350BT_train10B"
DEFAULT_TARGET_TRAIN_TOKENS = 10_000_000_000
DEFAULT_NUM_VAL_DOCS = 50_000
DEFAULT_SP_VOCAB_SIZES = "512,1024,2048"
GPT2_COUNT_BATCH_SIZE = int(os.environ.get("MATCHED_FINEWEB_GPT2_COUNT_BATCH_SIZE", "256"))


def parse_vocab_sizes(value: str) -> list[int]:
    vocab_sizes = [int(piece) for piece in value.split(",") if piece]
    if not vocab_sizes:
        raise ValueError("--sp_vocab_sizes must specify at least one vocab size")
    return vocab_sizes


def batched_rows(ds, batch_size: int):
    batch = []
    for row in ds:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_tokenizer_config(output_root: Path, *, sp_vocab_sizes: list[int], skip_byte: bool) -> Path:
    payload = json.loads(DEMO_CONFIG.read_text(encoding="utf-8"))
    specs = payload["tokenizers"] if isinstance(payload, dict) else payload
    vocab_size_set = set(sp_vocab_sizes)
    selected_specs = []
    found_sp_vocab_sizes: set[int] = set()
    for spec in specs:
        if "vocab_size" in spec:
            vocab_size = int(spec["vocab_size"])
            if vocab_size in vocab_size_set:
                selected_specs.append(spec)
                found_sp_vocab_sizes.add(vocab_size)
        elif not skip_byte:
            selected_specs.append(spec)
    missing = vocab_size_set - found_sp_vocab_sizes
    if missing:
        raise ValueError(f"unknown SentencePiece vocab sizes requested: {sorted(missing)}")
    tokenizer_config_path = output_root / "tokenizer_config.export.json"
    tokenizer_config_path.write_text(json.dumps({"tokenizers": selected_specs}, indent=2) + "\n", encoding="utf-8")
    return tokenizer_config_path


def maybe_reuse_existing_docs_cache(
    *,
    docs_jsonl: Path,
    dataset_name: str,
    dataset_revision: str | None,
    target_train_tokens: int,
    num_val_docs: int,
) -> dict[str, Any] | None:
    sidecar_path = docs_jsonl.with_name(f"{docs_jsonl.stem}.source_manifest.json")
    if not docs_jsonl.is_file() or not sidecar_path.is_file():
        return None
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "target_train_tokens": target_train_tokens,
        "num_val_docs": num_val_docs,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"existing docs cache metadata mismatch for {key}: expected {value}, got {payload.get(key)}. "
                "Use --rebuild_docs_cache or a fresh --output_root."
            )
    return payload


def build_docs_cache_from_hf_sample(
    *,
    dataset_name: str,
    dataset_revision: str | None,
    docs_jsonl: Path,
    target_train_tokens: int,
    num_val_docs: int,
) -> dict[str, Any]:
    print(
        f"Loading dataset HuggingFaceFW/fineweb name={dataset_name}"
        + (f" revision={dataset_revision}" if dataset_revision else "")
    )
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name=dataset_name,
        split="train",
        revision=dataset_revision,
        streaming=True,
    )
    ds = ds.remove_columns([column for column in ds.features if column != "text"])
    encoder = tiktoken.get_encoding("gpt2")
    docs_jsonl.parent.mkdir(parents=True, exist_ok=True)

    docs_val = 0
    docs_train = 0
    raw_tokens_val = 0
    raw_tokens_train = 0
    first_doc_preview = None

    with docs_jsonl.open("w", encoding="utf-8") as f:
        pbar = tqdm(total=target_train_tokens, unit="raw_tok", desc="Caching train docs")
        stop = False
        for rows in batched_rows(ds, GPT2_COUNT_BATCH_SIZE):
            texts = []
            for row in rows:
                text = row.get("text", "")
                if not isinstance(text, str):
                    text = "" if text is None else str(text)
                texts.append(text)
            exact_raw_token_counts = [len(tokens) for tokens in encoder.encode_ordinary_batch(texts)]
            for text, raw_tokens in zip(texts, exact_raw_token_counts, strict=True):
                split = "val" if docs_val < num_val_docs else "train"
                if docs_val < num_val_docs:
                    docs_val += 1
                    raw_tokens_val += raw_tokens
                elif raw_tokens_train < target_train_tokens:
                    docs_train += 1
                    raw_tokens_train += raw_tokens
                else:
                    stop = True
                    break
                if first_doc_preview is None:
                    first_doc_preview = text[:200]
                f.write(json.dumps({"text": text}, ensure_ascii=False))
                f.write("\n")
                if split == "train":
                    pbar.update(raw_tokens)
            if stop:
                break
        pbar.close()

    if docs_val != num_val_docs:
        raise ValueError(f"expected {num_val_docs} val docs, wrote {docs_val}")
    if raw_tokens_train < target_train_tokens:
        raise ValueError(
            f"expected at least {target_train_tokens} raw train tokens, only collected {raw_tokens_train}"
        )

    metadata = {
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "target_train_tokens": target_train_tokens,
        "num_val_docs": num_val_docs,
        "docs_val": docs_val,
        "docs_train": docs_train,
        "num_docs": docs_val + docs_train,
        "raw_tokens_val": raw_tokens_val,
        "raw_tokens_train": raw_tokens_train,
        "raw_tokenizer": "gpt2",
        "raw_token_count_source": "tiktoken.encode_ordinary_batch",
        "val_policy": f"first_{num_val_docs}_docs_from_stream",
        "train_policy": "continue_stream_until_raw_gpt2_target_met",
        "sample_assumption": "sample-350BT is a random subset of FineWeb",
        "first_doc_preview": first_doc_preview,
    }
    sidecar_path = docs_jsonl.with_name(f"{docs_jsonl.stem}.source_manifest.json")
    sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def augment_manifest(output_root: Path, source_dataset: dict[str, Any]) -> None:
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_huggingface"] = source_dataset
    manifest["dataset_revision"] = source_dataset.get("dataset_revision")
    manifest["shuffle_seed"] = None
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export matched tokenizer datasets from a Hugging Face FineWeb sample")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target_train_tokens", type=int, default=DEFAULT_TARGET_TRAIN_TOKENS)
    parser.add_argument("--num_val_docs", type=int, default=DEFAULT_NUM_VAL_DOCS)
    parser.add_argument("--chunk_tokens", type=int, default=None)
    parser.add_argument("--sp_vocab_sizes", default=DEFAULT_SP_VOCAB_SIZES)
    parser.add_argument("--tokenizer_train_docs", type=int, default=None)
    parser.add_argument("--reuse_sp_model", action="append", default=[])
    parser.add_argument("--skip_byte", action="store_true")
    parser.add_argument("--docs_only", action="store_true")
    parser.add_argument("--rebuild_docs_cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.target_train_tokens <= 0:
        raise ValueError(f"--target_train_tokens must be positive, got {args.target_train_tokens}")
    if args.num_val_docs <= 0:
        raise ValueError(f"--num_val_docs must be positive, got {args.num_val_docs}")

    output_root = Path(args.output_root).expanduser().resolve()
    docs_jsonl = output_root / "docs_selected.jsonl"

    if args.rebuild_docs_cache:
        docs_meta = build_docs_cache_from_hf_sample(
            dataset_name=args.dataset_name,
            dataset_revision=args.dataset_revision,
            docs_jsonl=docs_jsonl,
            target_train_tokens=args.target_train_tokens,
            num_val_docs=args.num_val_docs,
        )
    else:
        docs_meta = maybe_reuse_existing_docs_cache(
            docs_jsonl=docs_jsonl,
            dataset_name=args.dataset_name,
            dataset_revision=args.dataset_revision,
            target_train_tokens=args.target_train_tokens,
            num_val_docs=args.num_val_docs,
        )
        if docs_meta is None:
            docs_meta = build_docs_cache_from_hf_sample(
                dataset_name=args.dataset_name,
                dataset_revision=args.dataset_revision,
                docs_jsonl=docs_jsonl,
                target_train_tokens=args.target_train_tokens,
                num_val_docs=args.num_val_docs,
            )

    if args.docs_only:
        return

    tokenizer_config = write_tokenizer_config(
        output_root,
        sp_vocab_sizes=parse_vocab_sizes(args.sp_vocab_sizes),
        skip_byte=args.skip_byte,
    )
    matched_args = build_matched_parser().parse_args(
        [
            "--tokenizer_config",
            str(tokenizer_config),
            "--trust_tokenizer_config_code",
            "--output_root",
            str(output_root),
            "--docs_jsonl",
            str(docs_jsonl),
            "--num_val_docs",
            str(int(docs_meta["docs_val"])),
            *(["--chunk_tokens", str(args.chunk_tokens)] if args.chunk_tokens is not None else []),
            *(["--tokenizer_train_docs", str(args.tokenizer_train_docs)] if args.tokenizer_train_docs is not None else []),
            *(["--skip_byte"] if args.skip_byte else []),
            *sum((["--reuse_sp_model", value] for value in args.reuse_sp_model), []),
        ]
    )
    run_matched_export(matched_args)
    augment_manifest(output_root, docs_meta)


if __name__ == "__main__":
    main()
