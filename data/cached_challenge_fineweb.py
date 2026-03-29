import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
TOKENIZERS_DIR = ROOT / "tokenizers"


def dataset_dir_for_variant(name: str) -> str:
    if name in {"bytes", "byte256", "rawbytes"}:
        return "fineweb10B_bytes"
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected bytes, byte260, or sp<VOCAB_SIZE>")


def normalize_remote_root(prefix: str | None) -> str:
    if prefix is None:
        return ""
    return prefix.strip().strip("/")


def join_remote_path(*parts: str) -> str:
    clean = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(clean)


def local_path_for_remote(relative_path: str, *, remote_root_prefix: str | None = None) -> Path:
    root_prefix = normalize_remote_root(REMOTE_ROOT_PREFIX if remote_root_prefix is None else remote_root_prefix)
    remote_path = Path(relative_path)
    if root_prefix and remote_path.parts[:1] == (root_prefix,):
        remote_path = remote_path.relative_to(root_prefix)
    if remote_path.parts[:1] == ("datasets",):
        return DATASETS_DIR.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return TOKENIZERS_DIR.joinpath(*remote_path.parts[1:])
    return ROOT / remote_path


def get(relative_path: str, *, repo_id: str, remote_root_prefix: str | None = None) -> bool:
    destination = local_path_for_remote(relative_path, remote_root_prefix=remote_root_prefix)
    if destination.exists():
        return True
    if destination.is_symlink():
        destination.unlink()

    remote_path = Path(relative_path)
    try:
        cached_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_path.name,
                subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
                repo_type="dataset",
            )
        )
    except EntryNotFoundError:
        return False
    # HF cache entries may be snapshot symlinks. Resolve to the underlying blob so we
    # always materialize a real file in data/, not a broken relative symlink.
    cached_source = cached_path.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)
    return True


def manifest_path(*, remote_root_prefix: str | None = None) -> Path:
    root_prefix = REMOTE_ROOT_PREFIX if remote_root_prefix is None else remote_root_prefix
    return local_path_for_remote(join_remote_path(root_prefix or "", "manifest.json"), remote_root_prefix=root_prefix)


def load_manifest(*, repo_id: str, remote_root_prefix: str, skip_manifest_download: bool) -> dict:
    path = manifest_path(remote_root_prefix=remote_root_prefix)
    if not path.is_file():
        if skip_manifest_download:
            raise FileNotFoundError(
                f"manifest.json is required for manifest-driven shard counts but is not present locally at {path}"
            )
        found = get(join_remote_path(remote_root_prefix, "manifest.json"), repo_id=repo_id, remote_root_prefix=remote_root_prefix)
        if not found:
            raise FileNotFoundError(
                f"manifest.json not found in repo {repo_id!r} under remote root {remote_root_prefix!r}"
            )
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
        help="Tokenizer family to download, for example bytes, byte260, sp1024, or sp4096.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help="Hugging Face dataset repo id. Defaults to MATCHED_FINEWEB_REPO_ID or willdepueoai/parameter-golf.",
    )
    parser.add_argument(
        "--remote-root",
        default=REMOTE_ROOT_PREFIX,
        help="Optional remote root inside the dataset repo. Use '' for flat shard repos at the repo root.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Override the local/remote dataset directory name. Defaults to the name implied by --variant.",
    )
    parser.add_argument(
        "--flat-repo",
        action="store_true",
        help="Treat the Hugging Face repo as a flat shard store rather than a manifest-driven export.",
    )
    parser.add_argument(
        "--val-shards",
        type=int,
        default=None,
        help="Number of validation shards to download in flat-repo mode. Defaults to 1.",
    )
    parser.add_argument(
        "--val-from-local",
        default=None,
        help="Local directory containing fineweb_val_*.bin shards to hardlink/copy into the downloaded dataset.",
    )
    parser.add_argument(
        "--val-repo-id",
        default=None,
        help="Optional separate Hugging Face dataset repo id for validation shards in flat-repo mode.",
    )
    parser.add_argument(
        "--val-remote-root",
        default=None,
        help="Optional separate remote root for validation shards in flat-repo mode.",
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


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_local_validation(*, source_dir: Path, destination_dir: Path) -> int:
    files = sorted(source_dir.glob("fineweb_val_*.bin"))
    if not files:
        raise FileNotFoundError(f"No validation shards found in {source_dir}")
    for src in files:
        dst = destination_dir / src.name
        if not dst.exists():
            link_or_copy(src, dst)
    return len(files)


def download_flat_split(
    *,
    repo_id: str,
    remote_root: str,
    destination_dir: Path,
    prefix: str,
    count: int,
) -> None:
    for i in range(count):
        filename = f"{prefix}_{i:06d}.bin"
        relative_path = join_remote_path(remote_root, filename)
        ok = get(relative_path, repo_id=repo_id, remote_root_prefix=remote_root)
        if not ok:
            raise FileNotFoundError(f"{filename} not found in repo {repo_id!r} under {remote_root!r}")
        downloaded = local_path_for_remote(relative_path, remote_root_prefix=remote_root)
        target = destination_dir / filename
        if downloaded != target and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), str(target))


def main() -> None:
    args = build_parser().parse_args()
    repo_id = args.repo_id
    remote_root = normalize_remote_root(args.remote_root)
    dataset_dir = args.dataset_dir or dataset_dir_for_variant(args.variant)
    train_shards = args.train_shards_positional if args.train_shards_positional is not None else args.train_shards
    if train_shards < 0:
        raise ValueError("train_shards must be non-negative")
    dataset_local_dir = DATASETS_DIR / dataset_dir

    if args.flat_repo:
        dataset_local_dir.mkdir(parents=True, exist_ok=True)
        download_flat_split(
            repo_id=repo_id,
            remote_root=remote_root,
            destination_dir=dataset_local_dir,
            prefix="fineweb_train",
            count=train_shards,
        )
        if args.val_from_local:
            copy_local_validation(source_dir=Path(args.val_from_local).expanduser().resolve(), destination_dir=dataset_local_dir)
        else:
            val_shards = 1 if args.val_shards is None else args.val_shards
            val_repo_id = args.val_repo_id or repo_id
            val_remote_root = normalize_remote_root(args.val_remote_root if args.val_remote_root is not None else remote_root)
            if val_shards > 0:
                download_flat_split(
                    repo_id=val_repo_id,
                    remote_root=val_remote_root,
                    destination_dir=dataset_local_dir,
                    prefix="fineweb_val",
                    count=val_shards,
                )
        return

    manifest = load_manifest(repo_id=repo_id, remote_root_prefix=remote_root, skip_manifest_download=args.skip_manifest)
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None)
    if dataset_entry is None:
        raise ValueError(f"dataset {dataset_dir} not found in {join_remote_path(remote_root, 'manifest.json')}")
    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train"))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val"))
    if train_shards > max_train_shards:
        raise ValueError(
            f"{args.variant} only has {max_train_shards} training shards on {repo_id}, requested {train_shards}"
        )
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"tokenizer {tokenizer_name} not found in {join_remote_path(remote_root, 'manifest.json')}")

    if args.with_docs:
        get(join_remote_path(remote_root, "docs_selected.jsonl"), repo_id=repo_id, remote_root_prefix=remote_root)
        get(
            join_remote_path(remote_root, "docs_selected.source_manifest.json"),
            repo_id=repo_id,
            remote_root_prefix=remote_root,
        )

    dataset_prefix = join_remote_path(remote_root, "datasets", dataset_dir)
    for i in range(val_shards):
        get(f"{dataset_prefix}/fineweb_val_{i:06d}.bin", repo_id=repo_id, remote_root_prefix=remote_root)
    for i in range(train_shards):
        get(f"{dataset_prefix}/fineweb_train_{i:06d}.bin", repo_id=repo_id, remote_root_prefix=remote_root)

    for artifact_path in artifact_paths_for_tokenizer(tokenizer_entry):
        get(join_remote_path(remote_root, artifact_path), repo_id=repo_id, remote_root_prefix=remote_root)


if __name__ == "__main__":
    main()
