#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "romeerp/parameter-golf-caseops-v1"
TOKENIZER = "datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
DATASET_DIR = "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))


def validate(local_dir: Path, min_train_shards: int) -> None:
    root = local_dir / "datasets"
    tokenizer = root / "tokenizers" / "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
    dataset = root / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    missing = [p for p in (tokenizer, dataset) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CaseOps files: " + ", ".join(str(p) for p in missing))

    train = count_files(dataset, "fineweb_train_*.bin")
    val = count_files(dataset, "fineweb_val_*.bin")
    val_bytes = count_files(dataset, "fineweb_val_bytes_*.bin")
    if train < min_train_shards:
        raise RuntimeError(f"Expected at least {min_train_shards} train shards, found {train}")
    if val == 0:
        raise RuntimeError("No fineweb_val_*.bin shards found")
    if val_bytes == 0:
        raise RuntimeError("No fineweb_val_bytes_*.bin sidecar shards found")
    print(f"CaseOps data ready: train_shards={train} val_shards={val} val_byte_shards={val_bytes}")
    print(f"DATA_DIR={root}")
    print(f"DATA_PATH={dataset}")
    print(f"TOKENIZER_PATH={tokenizer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the CaseOps SP8192 dataset used by the #1953 lineage.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", default="/workspace/caseops_data", type=Path)
    parser.add_argument("--min-train-shards", default=80, type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.validate_only:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(args.local_dir),
            allow_patterns=[
                TOKENIZER,
                f"{DATASET_DIR}/fineweb_train_*.bin",
                f"{DATASET_DIR}/fineweb_val_*.bin",
                f"{DATASET_DIR}/fineweb_val_bytes_*.bin",
            ],
        )
    validate(args.local_dir, args.min_train_shards)


if __name__ == "__main__":
    main()
