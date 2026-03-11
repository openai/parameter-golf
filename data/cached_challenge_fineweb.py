import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
# Preserve existing local exports under data/challenge_fineweb while exposing canonical data/* paths.
LEGACY_LOCAL_ROOT = os.environ.get("MATCHED_FINEWEB_LOCAL_ALIAS", "challenge_fineweb")
LOCAL_DATASETS_DIR = Path(os.environ.get("MATCHED_FINEWEB_LOCAL_DATASETS_DIR", "datasets"))
LOCAL_TOKENIZERS_DIR = Path(os.environ.get("MATCHED_FINEWEB_LOCAL_TOKENIZERS_DIR", "tokenizers"))

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


def local_dir() -> Path:
    return Path(__file__).resolve().parent


def local_path_for_remote(relative_path: str) -> Path:
    remote_path = Path(relative_path)
    if not remote_path.parts or remote_path.parts[0] != REMOTE_ROOT_PREFIX:
        return local_dir() / remote_path
    inner = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if inner.parts[:1] == ("datasets",):
        return local_dir() / LOCAL_DATASETS_DIR.joinpath(*inner.parts[1:])
    if inner.parts[:1] == ("tokenizers",):
        return local_dir() / LOCAL_TOKENIZERS_DIR.joinpath(*inner.parts[1:])
    return local_dir() / inner


def ensure_alias(path: Path, target: Path, *, bad_target: Path | None = None) -> None:
    if path.is_symlink():
        resolved = path.resolve()
        if resolved == target.resolve():
            return
        if bad_target is not None and bad_target.exists() and resolved == bad_target.resolve():
            path.unlink()
        else:
            return
    if path.exists() or not target.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent))


def get(relative_path: str) -> None:
    destination = local_path_for_remote(relative_path)
    if destination.exists():
        return

    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_path, destination)
    except OSError:
        shutil.copy2(cached_path, destination)
    ensure_local_layout()


def ensure_local_layout() -> None:
    root = local_dir()
    legacy_root = root / LEGACY_LOCAL_ROOT
    ensure_alias(root / LOCAL_DATASETS_DIR, legacy_root / "datasets", bad_target=legacy_root)
    ensure_alias(root / LOCAL_TOKENIZERS_DIR, legacy_root / "tokenizers")
    ensure_alias(root / "manifest.json", legacy_root / "manifest.json")
    ensure_alias(root / "docs_selected.jsonl", legacy_root / "docs_selected.jsonl")


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

    ensure_local_layout()

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
    ensure_local_layout()


if __name__ == "__main__":
    main()
