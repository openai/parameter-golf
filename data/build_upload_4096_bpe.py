"""Build SentencePiece-4k BPE shards and upload to Hugging Face dataset."""

import argparse
import glob
import os
import subprocess
import sys

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "cocohearts/4096-bpe"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and upload 4k BPE FineWeb dataset")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--version", type=str, default="10B", choices=["10B", "100B"])
    parser.add_argument("--tokenizer_docs", type=int, default=2_000_000)
    parser.add_argument("--shard_docs", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=10**8)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    args = parser.parse_args()

    data_dir = os.path.dirname(os.path.abspath(__file__))
    model_prefix = os.path.join(data_dir, "tokenizers", "fineweb_4k_bpe")
    tokenizer_model = f"{model_prefix}.model"
    tokenizer_vocab = f"{model_prefix}.vocab"
    shard_dir_name = "fineweb10B_sp4k" if args.version == "10B" else "fineweb100B_sp4k"
    shard_dir = os.path.join(data_dir, shard_dir_name)

    if not args.skip_build:
        run(
            [
                sys.executable,
                os.path.join(data_dir, "train_sentencepiece_4k.py"),
                "--version",
                args.version,
                "--num_docs",
                str(args.tokenizer_docs),
                "--vocab_size",
                "4096",
                "--model_prefix",
                model_prefix,
            ]
        )
        run(
            [
                sys.executable,
                os.path.join(data_dir, "fineweb.py"),
                "--version",
                args.version,
                "--num_docs",
                str(args.shard_docs),
                "--shard_size",
                str(args.shard_size),
                "--tokenizer_model",
                tokenizer_model,
            ]
        )

    assert os.path.isfile(tokenizer_model), f"Missing tokenizer model: {tokenizer_model}"
    assert os.path.isfile(tokenizer_vocab), f"Missing tokenizer vocab: {tokenizer_vocab}"

    shard_files = sorted(glob.glob(os.path.join(shard_dir, "fineweb_*.bin")))
    assert shard_files, f"No shards found in {shard_dir}"
    assert any("_val_" in os.path.basename(path) for path in shard_files), "Missing validation shard"
    assert any("_train_" in os.path.basename(path) for path in shard_files), "Missing training shards"

    api = HfApi()
    repo = api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"Uploading to {repo.url}", flush=True)

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=shard_dir,
        path_in_repo=shard_dir_name,
        commit_message=f"Upload {shard_dir_name} shards",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=tokenizer_model,
        path_in_repo="tokenizers/fineweb_4k_bpe.model",
        commit_message="Upload tokenizer model",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=tokenizer_vocab,
        path_in_repo="tokenizers/fineweb_4k_bpe.vocab",
        commit_message="Upload tokenizer vocab",
    )
    print("Upload complete", flush=True)


if __name__ == "__main__":
    main()
