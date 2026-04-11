#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
HEADER_INTS = 256


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def read_shard_header(path: Path) -> dict[str, int]:
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS:
        raise ValueError(f"{path} does not contain the expected {HEADER_INTS}-int header")
    magic = int(header[0])
    version = int(header[1])
    num_tokens = int(header[2])
    header_bytes = HEADER_INTS * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    expected_size = header_bytes + num_tokens * token_bytes
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path} size mismatch: expected {expected_size} bytes, found {actual_size}")
    if magic != DATAFILE_MAGIC or version != DATAFILE_VERSION:
        raise ValueError(
            f"{path} header mismatch: expected magic/version {DATAFILE_MAGIC}/{DATAFILE_VERSION}, "
            f"found {magic}/{version}"
        )
    return {"magic": magic, "version": version, "num_tokens": num_tokens}


def check_data(
    data_path: str,
    tokenizer_path: str,
    *,
    min_train_shards: int,
    seq_len: int,
) -> dict[str, object]:
    dataset_dir = Path(data_path).resolve()
    tokenizer_file = Path(tokenizer_path).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

    dataset_name, train_shards, expected_train_shards = validate_dataset_tokenizer_pair(
        str(dataset_dir),
        str(tokenizer_file),
    )
    if train_shards < min_train_shards:
        raise RuntimeError(
            f"{dataset_name} has {train_shards} train shards, but at least {min_train_shards} are required"
        )

    train_files = sorted(dataset_dir.glob("fineweb_train_*.bin"))
    val_files = sorted(dataset_dir.glob("fineweb_val_*.bin"))
    if not val_files:
        raise RuntimeError(f"No validation shards found in {dataset_dir}")
    if min_train_shards > 0 and not train_files:
        raise RuntimeError(f"No training shards found in {dataset_dir}")

    checked_train = read_shard_header(train_files[0]) if train_files else None
    checked_val = read_shard_header(val_files[0])
    if checked_val["num_tokens"] <= seq_len:
        raise RuntimeError(
            f"Validation shard is too short for seq_len={seq_len}: {checked_val['num_tokens']} tokens"
        )

    warnings: list[str] = []
    if expected_train_shards is not None and train_shards < expected_train_shards:
        warnings.append(
            f"subset_only:{train_shards}/{expected_train_shards} train shards present; "
            "epochs will wrap sooner than the full export"
        )

    return {
        "dataset_path": str(dataset_dir),
        "tokenizer_path": str(tokenizer_file),
        "dataset_name": dataset_name,
        "train_shards": train_shards,
        "expected_train_shards": expected_train_shards,
        "val_shards": len(val_files),
        "train_header": checked_train,
        "val_header": checked_val,
        "warnings": warnings,
    }


def _print_summary(summary: dict[str, object]) -> None:
    print(f"dataset_name: {summary['dataset_name']}")
    print(f"dataset_path: {summary['dataset_path']}")
    print(f"tokenizer_path: {summary['tokenizer_path']}")
    print(f"train_shards: {summary['train_shards']}")
    if summary.get("expected_train_shards") is not None:
        print(f"expected_train_shards: {summary['expected_train_shards']}")
    print(f"val_shards: {summary['val_shards']}")
    train_header = summary.get("train_header")
    if train_header:
        print(f"train_header_tokens: {train_header['num_tokens']}")
    val_header = summary.get("val_header")
    if val_header:
        print(f"val_header_tokens: {val_header['num_tokens']}")
    for warning in summary.get("warnings", []):
        print(f"warning: {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset/tokenizer readiness for Parameter Golf.")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--min-train-shards", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    args = parser.parse_args()

    try:
        summary = check_data(
            args.data_path,
            args.tokenizer_path,
            min_train_shards=args.min_train_shards,
            seq_len=args.seq_len,
        )
    except Exception as exc:
        print(f"check_data failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)


if __name__ == "__main__":
    main()
