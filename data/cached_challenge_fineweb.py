import argparse
import os

from huggingface_hub import hf_hub_download


REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = "matched_10B_docs2m_seed1337"
LOCAL_ALIAS = os.environ.get("MATCHED_FINEWEB_LOCAL_ALIAS", "challenge_fineweb")

VARIANTS = {
    "byte260": {
        "dataset_dir": "fineweb10B_byte260",
        "tokenizer_files": ["fineweb_pure_byte_260.json"],
        "train_shards": 61,
        "val_shards": 2,
    },
    "sp512": {
        "dataset_dir": "fineweb10B_sp512",
        "tokenizer_files": ["fineweb_512_bpe.model", "fineweb_512_bpe.vocab"],
        "train_shards": 33,
        "val_shards": 1,
    },
    "sp1024": {
        "dataset_dir": "fineweb10B_sp1024",
        "tokenizer_files": ["fineweb_1024_bpe.model", "fineweb_1024_bpe.vocab"],
        "train_shards": 25,
        "val_shards": 1,
    },
    "sp2048": {
        "dataset_dir": "fineweb10B_sp2048",
        "tokenizer_files": ["fineweb_2048_bpe.model", "fineweb_2048_bpe.vocab"],
        "train_shards": 21,
        "val_shards": 1,
    },
    "sp4096": {
        "dataset_dir": "fineweb10B_sp4096",
        "tokenizer_files": ["fineweb_4096_bpe.model", "fineweb_4096_bpe.vocab"],
        "train_shards": 19,
        "val_shards": 1,
    },
}
def get(relative_path: str) -> None:
    local_dir = os.path.dirname(__file__)
    full_local_path = os.path.join(local_dir, relative_path)
    if not os.path.exists(full_local_path):
        hf_hub_download(
            repo_id=REPO_ID,
            filename=relative_path,
            repo_type="dataset",
            local_dir=local_dir,
        )


def ensure_local_alias() -> None:
    local_dir = os.path.dirname(__file__)
    remote_root = os.path.join(local_dir, REMOTE_ROOT_PREFIX)
    alias_root = os.path.join(local_dir, LOCAL_ALIAS)
    if os.path.exists(alias_root):
        if os.path.realpath(alias_root) != os.path.realpath(remote_root):
            raise FileExistsError(f"Local alias already exists and points elsewhere: {alias_root}")
        return
    alias_target = os.path.relpath(remote_root, os.path.dirname(alias_root))
    os.symlink(alias_target, alias_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download challenge FineWeb shards from Hugging Face")
    parser.add_argument(
        "train_shards",
        nargs="?",
        type=int,
        default=1,
        help="Number of training shards to download for the selected variant.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="sp2048",
        help="Tokenizer family to download.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip downloading manifest.json.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    variant = VARIANTS[args.variant]
    max_train_shards = variant["train_shards"]
    if args.train_shards < 0:
        raise ValueError("train_shards must be non-negative")
    if args.train_shards > max_train_shards:
        raise ValueError(
            f"{args.variant} only has {max_train_shards} training shards on {REPO_ID}, "
            f"requested {args.train_shards}"
        )

    ensure_local_alias()

    if not args.skip_manifest:
        get(f"{REMOTE_ROOT_PREFIX}/manifest.json")

    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{variant['dataset_dir']}"
    for i in range(variant["val_shards"]):
        get(f"{dataset_prefix}/fineweb_val_{i:06d}.bin")
    for i in range(args.train_shards):
        get(f"{dataset_prefix}/fineweb_train_{i:06d}.bin")

    tokenizer_prefix = f"{REMOTE_ROOT_PREFIX}/tokenizers"
    for tokenizer_file in variant["tokenizer_files"]:
        get(f"{tokenizer_prefix}/{tokenizer_file}")


if __name__ == "__main__":
    main()
