#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.session_registry import list_codex_sessions, pid_is_alive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List Codex proposal/training sessions")
    parser.add_argument("--active-only", action="store_true", help="Show only active proposal/training sessions.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = list_codex_sessions(active_only=args.active_only)
    if not rows:
        print("no codex sessions found")
        return 0
    for row in rows:
        pid = row.get("pid")
        alive = pid_is_alive(int(pid)) if isinstance(pid, int) else False
        print(
            " ".join(
                [
                    f"candidate={row.get('candidate_id')}",
                    f"status={row.get('status')}",
                    f"family={row.get('family')}",
                    f"tier={row.get('tier')}",
                    f"mode={row.get('proposal_mode')}",
                    f"pid={pid}",
                    f"alive={str(alive).lower()}",
                ]
            )
        )
        print(f"  started={row.get('started_at')} updated={row.get('updated_at')}")
        print(f"  stream={row.get('stdout_latest_path')}")
        if row.get("train_log"):
            print(f"  train_log={row.get('train_log')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
