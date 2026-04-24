"""Upload tokenized SP8192 CaseOps shards to a private HF dataset.

Run on the EU-NL-1 Phase A pod after prepare_caseops_data_parallel.py finishes.
The uploaded dataset becomes the input for Phase B (8xH100 AP-IN-1 no-volume).

Usage (on pod):
    HF_TOKEN=$(cat /root/.hf_token) \
    python3 /workspace/repo/parameter-golf/runpod/upload_caseops_to_hf.py \
        --src  /workspace/data/datasets/fineweb10B_sp8192_caseops \
        --repo FijaEE/parameter-golf-sp8192-caseops
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--repo", required=True, type=str)
    ap.add_argument("--private", action="store_true", default=True)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    # Idempotent: create if absent.
    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    print(f"uploading {args.src} -> hf://datasets/{args.repo}")

    # upload_large_folder is HF's recommended path for multi-GB datasets.
    # It chunks, resumes, and runs multiple upload workers in parallel.
    api.upload_large_folder(
        folder_path=str(args.src),
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=["**/*.bin", "**/*.json", "**/*.txt"],
        num_workers=16,
        print_report=True,
    )
    print("done")


if __name__ == "__main__":
    main()
