#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.session_registry import get_codex_session, list_codex_sessions


TERMINAL_STATUSES = {"completed", "failed", "proposal_failed", "proposal_duplicate"}
PROPOSAL_STATUSES = {"proposing", "proposal_done", "proposed", "proposal_pending"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch one Codex session in real time")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate", help="Candidate id to watch.")
    group.add_argument("--latest", action="store_true", help="Watch the most recently updated session.")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Polling interval.")
    parser.add_argument("--no-follow-training", action="store_true", help="Stop after the proposal stream instead of switching to train.log.")
    return parser


def resolve_candidate_id(args: argparse.Namespace) -> str:
    if args.candidate:
        return args.candidate
    rows = list_codex_sessions(active_only=False)
    if not rows:
        raise SystemExit("no codex sessions found")
    return str(rows[0]["candidate_id"])


def stream_file(path: Path, position: int) -> int:
    if not path.is_file():
        return position
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(position)
        while True:
            chunk = handle.readline()
            if not chunk:
                break
            sys.stdout.write(chunk)
            sys.stdout.flush()
        return handle.tell()


def main() -> int:
    args = build_parser().parse_args()
    candidate_id = resolve_candidate_id(args)
    current_path: Path | None = None
    positions: dict[str, int] = {}
    printed_header_for: set[str] = set()

    while True:
        row = get_codex_session(candidate_id)
        if row is None:
            print(f"session not found for candidate {candidate_id}")
            return 1

        status = str(row.get("status"))
        stream_path = row.get("stdout_latest_path")
        train_path = row.get("train_log")

        desired_path: Path | None = None
        if not args.no_follow_training and status == "training" and train_path:
            desired_path = Path(train_path)
        elif stream_path:
            desired_path = Path(stream_path)
        elif train_path:
            desired_path = Path(train_path)

        if desired_path is not None:
            path_key = str(desired_path)
            if current_path != desired_path:
                current_path = desired_path
                if path_key not in printed_header_for:
                    print(f"--- watching {candidate_id} status={status} file={desired_path}")
                    printed_header_for.add(path_key)
            position = positions.get(path_key, 0)
            positions[path_key] = stream_file(desired_path, position)

        if status in TERMINAL_STATUSES:
            if not args.no_follow_training and train_path and current_path != Path(train_path):
                current_path = Path(train_path)
                continue
            # Give one final poll for trailing bytes.
            if desired_path is not None:
                path_key = str(desired_path)
                positions[path_key] = stream_file(desired_path, positions.get(path_key, 0))
            print(f"\n--- session ended status={status} candidate={candidate_id}")
            return 0

        time.sleep(max(args.poll_seconds, 0.1))


if __name__ == "__main__":
    raise SystemExit(main())
