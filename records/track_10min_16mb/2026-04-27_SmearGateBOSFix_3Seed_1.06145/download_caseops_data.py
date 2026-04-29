#!/usr/bin/env python3
"""Download the public CaseOps SP8192 dataset used by this record.

This script materializes the exact directory layout expected by train_gpt.py
when CASEOPS_ENABLED=1:

  data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/...
  data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_.../

It downloads pre-tokenized CaseOps shards from romeerp/parameter-golf-caseops-v1.
Do not run prepare_caseops_data.py unless rebuilding from docs_selected.jsonl.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


REPO_ID = "romeerp/parameter-golf-caseops-v1"
REMOTE_ROOT = "datasets"
DATASET_NAME = "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
TOKENIZER_NAME = "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"


def build_patterns(train_shards: int) -> list[str]:
    if train_shards < 0:
        raise ValueError("--train-shards must be non-negative")
    patterns = [
        f"{REMOTE_ROOT}/manifest.json",
        f"{REMOTE_ROOT}/tokenizers/*",
        f"{REMOTE_ROOT}/datasets/{DATASET_NAME}/fineweb_val_*.bin",
        f"{REMOTE_ROOT}/datasets/{DATASET_NAME}/fineweb_val_bytes_*.bin",
    ]
    patterns.extend(
        f"{REMOTE_ROOT}/datasets/{DATASET_NAME}/fineweb_train_{i:06d}.bin"
        for i in range(train_shards)
    )
    return patterns


def preflight(data_dir: Path, train_shards: int) -> None:
    root = data_dir / "datasets" / "fineweb10B_sp8192_caseops" / "datasets"
    paths = [
        root / "tokenizers" / TOKENIZER_NAME,
        root / "datasets" / DATASET_NAME / "fineweb_val_000000.bin",
        root / "datasets" / DATASET_NAME / "fineweb_val_bytes_000000.bin",
    ]
    if train_shards > 0:
        paths.append(root / "datasets" / DATASET_NAME / "fineweb_train_000000.bin")
    missing = [p for p in paths if not p.is_file()]
    for p in paths:
        print(("OK  " if p.is_file() else "MISS ") + str(p))
    if missing:
        raise FileNotFoundError("missing required CaseOps files")

    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["CASEOPS_ENABLED"] = "1"
    os.environ["VOCAB_SIZE"] = "8192"

    if "flash_attn_interface" not in sys.modules:
        mod = types.ModuleType("flash_attn_interface")

        def _unused_flash_attn(*args, **kwargs):
            raise RuntimeError("flash attention is not used during data preflight")

        mod.flash_attn_func = _unused_flash_attn
        mod.flash_attn_varlen_func = _unused_flash_attn
        sys.modules["flash_attn_interface"] = mod

    import importlib.util

    train_gpt = Path(__file__).with_name("train_gpt.py")
    spec = importlib.util.spec_from_file_location("caseops_train_gpt", train_gpt)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    h = module.Hyperparameters()
    val_data = module.ValidationData(h, torch.device("cpu"))
    print(f"datasets_dir={h.datasets_dir}")
    print(f"tokenizer_path={h.tokenizer_path}")
    print(f"train_files={h.train_files}")
    print(f"val_files={h.val_files}")
    print(f"val_bytes_files={h.val_bytes_files}")
    print(f"val_tokens={val_data.val_tokens.numel()}")
    print(f"val_bytes={val_data.val_bytes.numel() if val_data.val_bytes is not None else None}")
    print(f"sp_vocab={val_data.sp.vocab_size()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--train-shards", default=80, type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not args.check_only:
        local_dir = args.data_dir / "datasets" / "fineweb10B_sp8192_caseops"
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=build_patterns(args.train_shards),
        )
    preflight(args.data_dir, args.train_shards)


if __name__ == "__main__":
    main()
