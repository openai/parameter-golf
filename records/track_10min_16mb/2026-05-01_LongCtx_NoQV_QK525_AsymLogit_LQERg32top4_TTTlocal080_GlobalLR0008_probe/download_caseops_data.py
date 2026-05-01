#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET = "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"


def numbered_patterns(kind, count):
    if count < 0:
        return [f"datasets/datasets/{DATASET}/fineweb_{kind}_*.bin"]
    return [
        f"datasets/datasets/{DATASET}/fineweb_{kind}_{i:06d}.bin"
        for i in range(count)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default="/workspace/caseops_data")
    ap.add_argument("--repo-id", default="romeerp/parameter-golf-caseops-v1")
    ap.add_argument("--train-shards", type=int, default=-1, help="-1 downloads all train shards")
    ap.add_argument("--val-shards", type=int, default=-1, help="-1 downloads all val shards")
    args = ap.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "datasets/manifest.json",
        "datasets/tokenizers/*",
        *numbered_patterns("train", args.train_shards),
        *numbered_patterns("val", args.val_shards),
    ]
    if args.val_shards < 0:
        patterns.append(f"datasets/datasets/{DATASET}/fineweb_val_bytes_*.bin")
    else:
        patterns.extend(
            f"datasets/datasets/{DATASET}/fineweb_val_bytes_{i:06d}.bin"
            for i in range(args.val_shards)
        )
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=patterns,
    )
    print(f"Downloaded CaseOps data to {local_dir}")


if __name__ == "__main__":
    main()
