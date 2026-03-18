#!/usr/bin/env python3
"""Replace challenge dataset artifacts in a Hugging Face dataset repo with a local export."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "willdepueoai/parameter-golf"
DEFAULT_PATH_IN_REPO = "datasets"
DATA_ARTIFACT_NAMES = {
    "datasets",
    "tokenizers",
    "manifest.json",
    "docs_selected.jsonl",
    "docs_selected.source_manifest.json",
    "tokenizer_config.export.json",
    "snapshot_meta.json",
}


def repo_path(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replace old dataset artifacts in a HF dataset repo with a local export")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-export-root", required=True)
    parser.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO, help="Subdirectory inside the dataset repo")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--commit-message", default="Replace dataset export")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api = HfApi()
    local_export_root = Path(args.local_export_root).expanduser().resolve()
    if not local_export_root.is_dir():
        raise FileNotFoundError(local_export_root)

    prefix = args.path_in_repo.strip("/")
    top_level_local = {path.name for path in local_export_root.iterdir()}
    delete_names = sorted(DATA_ARTIFACT_NAMES | top_level_local)
    root_entries = {
        entry.path: entry
        for entry in api.list_repo_tree(
            repo_id=args.repo_id,
            recursive=False,
            repo_type=args.repo_type,
            revision=args.revision,
        )
    }

    if prefix:
        if prefix in root_entries:
            print(f"delete {prefix}")
            if not args.dry_run:
                api.delete_folder(
                    prefix,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    revision=args.revision,
                    commit_message=f"Delete {prefix}",
                )

    remote_entries = root_entries if not prefix else {}

    for name in delete_names:
        if prefix:
            break
        remote_path = repo_path(prefix, name)
        entry = remote_entries.get(remote_path)
        if entry is None:
            continue
        print(f"delete {remote_path}")
        if args.dry_run:
            continue
        if entry.__class__.__name__ == "RepoFolder":
            api.delete_folder(
                remote_path,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                commit_message=f"Delete {remote_path}",
            )
        else:
            api.delete_file(
                remote_path,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                commit_message=f"Delete {remote_path}",
            )

    print(f"upload {local_export_root} -> {prefix or '/'}")
    if args.dry_run:
        return
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        folder_path=local_export_root,
        path_in_repo=prefix or None,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
