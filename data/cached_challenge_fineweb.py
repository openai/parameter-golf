import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get(
    "MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "matched_100B_train30Btok_even_seed1337"
)


def repo_relative_env_path(name: str, default: str) -> Path:
    path = Path(os.environ.get(name, default))
    if not path.parts or path.is_absolute():
        raise ValueError(f"{name} must be a non-empty repo-relative path under data/, got {path}")
    root = Path(__file__).resolve().parent
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{name} must stay under {root}, got {path}")
    return path


# Preserve existing local exports under data/challenge_fineweb while exposing canonical data/* paths.
LEGACY_LOCAL_ROOT = repo_relative_env_path("MATCHED_FINEWEB_LOCAL_ALIAS", "challenge_fineweb")
LOCAL_DATASETS_DIR = repo_relative_env_path("MATCHED_FINEWEB_LOCAL_DATASETS_DIR", "datasets")
LOCAL_TOKENIZERS_DIR = repo_relative_env_path("MATCHED_FINEWEB_LOCAL_TOKENIZERS_DIR", "tokenizers")

VARIANTS = {
    "byte260": {
        "dataset_dir": "fineweb10B_byte260",
    },
    "sp512": {
        "dataset_dir": "fineweb10B_sp512",
    },
    "sp1024": {
        "dataset_dir": "fineweb10B_sp1024",
    },
    "sp2048": {
        "dataset_dir": "fineweb10B_sp2048",
    },
    "sp4096": {
        "dataset_dir": "fineweb10B_sp4096",
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
    if destination.is_symlink():
        destination.unlink()

    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    # HF cache entries may be snapshot symlinks. Resolve to the underlying blob so we
    # always materialize a real file in data/, not a broken relative symlink.
    cached_source = cached_path.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)
    ensure_local_layout()


def ensure_local_layout() -> None:
    root = local_dir()
    legacy_root = root / LEGACY_LOCAL_ROOT
    ensure_alias(root / LOCAL_DATASETS_DIR, legacy_root / "datasets", bad_target=legacy_root)
    ensure_alias(root / LOCAL_TOKENIZERS_DIR, legacy_root / "tokenizers")
    ensure_alias(root / "manifest.json", legacy_root / "manifest.json")
    ensure_alias(root / "docs_selected.jsonl", legacy_root / "docs_selected.jsonl")


def manifest_path() -> Path:
    return local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/manifest.json")


def load_manifest(*, skip_manifest_download: bool) -> dict:
    path = manifest_path()
    if not path.is_file():
        if skip_manifest_download:
            raise FileNotFoundError(
                f"manifest.json is required for manifest-driven shard counts but is not present locally at {path}"
            )
        get(f"{REMOTE_ROOT_PREFIX}/manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_paths_for_tokenizer(tokenizer_entry: dict) -> list[str]:
    artifacts = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            artifacts.append(str(value))
    if not artifacts:
        raise ValueError(f"tokenizer entry is missing downloadable artifacts: {tokenizer_entry}")
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download challenge FineWeb shards from Hugging Face")
    parser.add_argument(
        "train_shards_positional",
        nargs="?",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=1,
        help="Number of training shards to download for the selected variant. Defaults to 1.",
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
    train_shards = args.train_shards_positional if args.train_shards_positional is not None else args.train_shards
    if train_shards < 0:
        raise ValueError("train_shards must be non-negative")

    ensure_local_layout()
    manifest = load_manifest(skip_manifest_download=args.skip_manifest)
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == variant["dataset_dir"]), None)
    if dataset_entry is None:
        raise ValueError(f"dataset {variant['dataset_dir']} not found in {REMOTE_ROOT_PREFIX}/manifest.json")
    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train"))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val"))
    if train_shards > max_train_shards:
        raise ValueError(
            f"{args.variant} only has {max_train_shards} training shards on {REPO_ID}, requested {train_shards}"
        )
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"tokenizer {tokenizer_name} not found in {REMOTE_ROOT_PREFIX}/manifest.json")

    if not args.skip_manifest:
        get(f"{REMOTE_ROOT_PREFIX}/manifest.json")

    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{variant['dataset_dir']}"
    for i in range(val_shards):
        get(f"{dataset_prefix}/fineweb_val_{i:06d}.bin")
    for i in range(train_shards):
        get(f"{dataset_prefix}/fineweb_train_{i:06d}.bin")

    for artifact_path in artifact_paths_for_tokenizer(tokenizer_entry):
        get(f"{REMOTE_ROOT_PREFIX}/{artifact_path}")
    ensure_local_layout()


if __name__ == "__main__":
    main()
