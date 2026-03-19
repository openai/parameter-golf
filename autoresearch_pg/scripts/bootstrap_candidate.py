#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.workspace import bootstrap_candidate, candidate_dir, default_candidate_id, repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a new Parameter Golf autoresearch candidate")
    parser.add_argument("--candidate", help="Candidate id. Defaults to a timestamped id.")
    parser.add_argument("--parent", help="Parent candidate id. If set, seed from that candidate's train_gpt.py.")
    parser.add_argument(
        "--source",
        help="Optional source entrypoint path. Defaults to the root train_gpt.py, or the parent candidate entrypoint.",
    )
    parser.add_argument("--note", help="Initial hypothesis note for notes.md.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidate_id = args.candidate or default_candidate_id()

    if args.source:
        source_train_gpt = Path(args.source).resolve()
    elif args.parent:
        parent_meta_path = candidate_dir(args.parent) / "meta.json"
        if parent_meta_path.is_file():
            import json

            parent_meta = json.loads(parent_meta_path.read_text(encoding="utf-8"))
            entrypoint_filename = parent_meta.get("entrypoint_filename", "train_gpt.py")
        else:
            entrypoint_filename = "train_gpt.py"
        source_train_gpt = candidate_dir(args.parent) / entrypoint_filename
    else:
        source_train_gpt = repo_root() / "train_gpt.py"

    out_dir = bootstrap_candidate(
        candidate_id=candidate_id,
        source_train_gpt=source_train_gpt,
        parent_candidate_id=args.parent,
        note=args.note,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
