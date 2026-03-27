#!/usr/bin/env python3
"""
wait_for_result.py — Local blocking script.

Blocks until .gpu_state.json reaches 'completed' or 'failed'.
Prints live status updates while waiting.

Usage:
  python3 wait_for_result.py [--repo-dir .] [--poll-interval 5] [--timeout 900]

Exit codes:
  0  — experiment completed successfully (state=completed)
  1  — experiment failed (state=failed)
  2  — timed out waiting (local safety net)
  3  — unexpected error
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = ".gpu_state.json"
VALID_TERMINAL_STATES = {"completed", "failed"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_state(repo: Path) -> dict | None:
    """Read state file. Returns None if missing or unreadable."""
    path = repo / STATE_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block until the remote GPU experiment finishes."
    )
    parser.add_argument("--repo-dir", default=".", help="Shared repo path (default: .)")
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="Poll interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Local safety timeout in seconds (default: 1200 = 20 min)",
    )
    args = parser.parse_args()

    repo = Path(args.repo_dir).resolve()
    state_path = repo / STATE_FILE

    print(f"[wait]  Watching: {state_path}")
    print(f"[wait]  Timeout : {args.timeout}s  Poll: {args.poll_interval}s")
    print(f"[wait]  Started : {_utcnow_iso()}")
    print()

    start_time = time.monotonic()
    last_printed_state: str | None = None
    last_printed_elapsed: int = -1

    try:
        while True:
            elapsed = time.monotonic() - start_time

            # Local safety timeout
            if elapsed > args.timeout:
                print(
                    f"\n[wait]  TIMEOUT: waited {fmt_elapsed(elapsed)} "
                    f"(local limit={args.timeout}s). Giving up."
                )
                print("[wait]  The remote watcher may still be running. Check run.log manually.")
                sys.exit(2)

            state_data = read_state(repo)

            if state_data is None:
                if int(elapsed) // 15 != last_printed_elapsed // 15:
                    last_printed_elapsed = int(elapsed)
                    print(
                        f"[wait]  {fmt_elapsed(elapsed)}  state=<missing>  "
                        "(waiting for state file to appear...)"
                    )
                time.sleep(args.poll_interval)
                continue

            state_name = state_data.get("state", "unknown")
            commit = state_data.get("commit", "?")

            # Print a status line each time state changes, or every 30 s
            elapsed_bucket = int(elapsed) // 30
            state_changed = state_name != last_printed_state
            tick = elapsed_bucket != last_printed_elapsed // 30

            if state_changed or tick:
                last_printed_state = state_name
                last_printed_elapsed = int(elapsed)

                if state_name == "idle":
                    print(
                        f"[wait]  {fmt_elapsed(elapsed)}  state=idle  "
                        "(experiment not yet submitted — is local agent done committing?)"
                    )
                elif state_name == "pending":
                    print(
                        f"[wait]  {fmt_elapsed(elapsed)}  state=pending  commit={commit}  "
                        "(remote watcher picking up...)"
                    )
                elif state_name == "running":
                    started = state_data.get("started_at", "?")
                    pid = state_data.get("pid", "?")
                    print(
                        f"[wait]  {fmt_elapsed(elapsed)}  state=running  "
                        f"commit={commit}  pid={pid}  started={started}"
                    )
                elif state_name in VALID_TERMINAL_STATES:
                    pass  # handled below
                else:
                    print(f"[wait]  {fmt_elapsed(elapsed)}  state={state_name}  (unexpected)")

            # Terminal states
            if state_name == "completed":
                finished = state_data.get("finished_at", "?")
                print(
                    f"\n[wait]  ✓ COMPLETED  commit={commit}  "
                    f"finished={finished}  elapsed={fmt_elapsed(elapsed)}"
                )
                print("[wait]  run.log is ready. Proceeding to read results.")
                sys.exit(0)

            elif state_name == "failed":
                finished = state_data.get("finished_at", "?")
                error = state_data.get("error", "unknown")
                exit_code = state_data.get("exit_code", "?")
                print(
                    f"\n[wait]  ✗ FAILED  commit={commit}  "
                    f"error={error}  exit_code={exit_code}  "
                    f"finished={finished}  elapsed={fmt_elapsed(elapsed)}"
                )
                print("[wait]  Check run.log for details.")
                sys.exit(1)

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print(f"\n[wait]  Interrupted after {fmt_elapsed(time.monotonic() - start_time)}. Exiting.")
        sys.exit(3)


if __name__ == "__main__":
    main()
