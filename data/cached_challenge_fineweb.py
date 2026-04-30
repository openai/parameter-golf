import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "willdepueoai/parameter-golf"
DEFAULT_REMOTE_ROOT_PREFIX = "datasets"
SP8192_REPO_ID = "Jaikirat/fineweb10B_sp8192"
SP8192_REMOTE_ROOT_PREFIX = ""
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
TOKENIZERS_DIR = ROOT / "tokenizers"


@dataclass(frozen=True)
class DatasetSource:
    repo_id: str
    remote_root_prefix: str


def source_for_variant(variant: str) -> DatasetSource:
    if variant == "sp8192":
        default_repo_id = SP8192_REPO_ID
        default_remote_root_prefix = SP8192_REMOTE_ROOT_PREFIX
    else:
        default_repo_id = DEFAULT_REPO_ID
        default_remote_root_prefix = DEFAULT_REMOTE_ROOT_PREFIX
    return DatasetSource(
        repo_id=os.environ.get("MATCHED_FINEWEB_REPO_ID", default_repo_id),
        remote_root_prefix=os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", default_remote_root_prefix),
    )


def remote_path(source: DatasetSource, *parts: str) -> str:
    path_parts = [source.remote_root_prefix, *parts]
    return "/".join(part.strip("/") for part in path_parts if part.strip("/"))


def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def local_path_for_remote(relative_path: str, source: DatasetSource) -> Path:
    remote_path = Path(relative_path)
    if source.remote_root_prefix and remote_path.parts[:1] == (source.remote_root_prefix,):
        remote_path = remote_path.relative_to(source.remote_root_prefix)
    if remote_path.parts[:1] == ("datasets",):
        return DATASETS_DIR.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return TOKENIZERS_DIR.joinpath(*remote_path.parts[1:])
    return ROOT / remote_path


def get(relative_path: str, source: DatasetSource, *, force: bool = False) -> None:
    destination = local_path_for_remote(relative_path, source)
    if destination.exists() and not force:
        return
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    remote_path = Path(relative_path)
    downloaded_path = Path(
        hf_hub_download(
            repo_id=source.repo_id,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
            local_dir=ROOT,
            force_download=force,
        )
    )
    if downloaded_path == destination:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path.replace(destination)


def manifest_path(source: DatasetSource) -> Path:
    return local_path_for_remote(remote_path(source, "manifest.json"), source)


def manifest_has_dataset(manifest: dict, dataset_dir: str) -> bool:
    return any(entry.get("name") == dataset_dir for entry in manifest.get("datasets", []))


def load_manifest(source: DatasetSource, dataset_dir: str, *, skip_manifest_download: bool) -> dict:
    path = manifest_path(source)
    if not path.is_file():
        if skip_manifest_download:
            raise FileNotFoundError(
                f"manifest.json is required for manifest-driven shard counts but is not present locally at {path}"
            )
        get(remote_path(source, "manifest.json"), source)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest_has_dataset(manifest, dataset_dir) or skip_manifest_download:
        return manifest
    get(remote_path(source, "manifest.json"), source, force=True)
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
        default=80,
        help="Number of training shards to download for the selected variant. Defaults to 80.",
    )
    parser.add_argument(
        "--variant",
        default="sp1024",
        help="Tokenizer family to download, for example sp1024, sp4096, or byte260.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip downloading manifest.json.",
    )
    parser.add_argument(
        "--with-docs",
        action="store_true",
        help="Also download docs_selected.jsonl and its sidecar for tokenizer retraining or dataset re-export.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = source_for_variant(args.variant)
    dataset_dir = dataset_dir_for_variant(args.variant)
    train_shards = args.train_shards_positional if args.train_shards_positional is not None else args.train_shards
    if train_shards < 0:
        raise ValueError("train_shards must be non-negative")

    manifest = load_manifest(source, dataset_dir, skip_manifest_download=args.skip_manifest)
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None)
    if dataset_entry is None:
        raise ValueError(f"dataset {dataset_dir} not found in {remote_path(source, 'manifest.json')}")
    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train"))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val"))
    if train_shards > max_train_shards:
        raise ValueError(
            f"{args.variant} only has {max_train_shards} training shards on {source.repo_id}, requested {train_shards}"
        )
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"tokenizer {tokenizer_name} not found in {remote_path(source, 'manifest.json')}")

    if args.with_docs:
        get(remote_path(source, "docs_selected.jsonl"), source)
        get(remote_path(source, "docs_selected.source_manifest.json"), source)

    dataset_prefix = remote_path(source, "datasets", dataset_dir)
    for i in range(val_shards):
        get(f"{dataset_prefix}/fineweb_val_{i:06d}.bin", source)
    for i in range(train_shards):
        get(f"{dataset_prefix}/fineweb_train_{i:06d}.bin", source)

    for artifact_path in artifact_paths_for_tokenizer(tokenizer_entry):
        get(remote_path(source, artifact_path), source)


if __name__ == "__main__":
    main()
