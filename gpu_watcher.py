#!/usr/bin/env python3
"""
gpu_watcher.py — Remote GPU machine daemon.

Monitors .gpu_state.json for state transitions and runs experiments.

State machine:
  idle       → (local writes remote_gpu_run.sh + sets state=pending after git commit)
  pending    → (this daemon picks it up) → running
  running    → (experiment finishes)     → completed | failed
  completed  → (local reads results)     → idle
  failed     → (local reads results)     → idle

Usage (on remote GPU machine):
  python3 gpu_watcher.py [--repo-dir /path/to/shared/repo] [--poll-interval 5]
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────── constants ────────────────────────────────────────

STATE_FILE = ".gpu_state.json"
RUN_SCRIPT = "remote_gpu_run.sh"
RUN_LOG = "run.log"

VALID_STATES = {"idle", "pending", "running", "completed", "failed"}

# If a run exceeds this many seconds, it is killed and marked failed.
# 10 min training + 3 min overhead + 2 min buffer
HARD_TIMEOUT_SECONDS = 15 * 60

# How many seconds a state file can stay in "running" with no heartbeat
# before we consider the watcher crashed and recover.
STALE_RUNNING_TIMEOUT = 5 * 60

# ─────────────────────────── state file helpers ───────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_state(repo: Path) -> dict:
    """Read and return the state dict. Returns default idle state if missing."""
    path = repo / STATE_FILE
    if not path.exists():
        return _default_idle()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if data.get("state") not in VALID_STATES:
            print(f"[watcher] WARNING: unknown state '{data.get('state')}', treating as idle")
            return _default_idle()
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"[watcher] WARNING: could not read state file ({e}), treating as idle")
        return _default_idle()


def write_state(repo: Path, data: dict) -> None:
    """Atomically write state dict to STATE_FILE."""
    data = {**data, "updated_at": _utcnow_iso()}
    path = repo / STATE_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _default_idle() -> dict:
    return {
        "state": "idle",
        "commit": None,
        "run_script": None,
        "pid": None,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "error": None,
        "updated_at": _utcnow_iso(),
    }


# ─────────────────────────── git helpers ──────────────────────────────────────


def get_current_commit(repo: Path) -> str | None:
    """Return the short HEAD commit hash, or None on error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def git_pull(repo: Path) -> bool:
    """Run git pull (fast-forward only). Returns True on success."""
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True
        print(f"[watcher] git pull failed: {result.stderr.strip()}")
    except Exception as e:
        print(f"[watcher] git pull exception: {e}")
    return False


# ─────────────────────────── experiment runner ────────────────────────────────


def run_experiment(repo: Path, script_path: Path, state: dict) -> None:
    """
    Execute remote_gpu_run.sh, stream output to run.log, update state on finish.
    This function blocks until the process completes or is killed.
    """
    log_path = repo / RUN_LOG
    started_at = _utcnow_iso()

    # Transition: pending → running
    write_state(repo, {
        **state,
        "state": "running",
        "started_at": started_at,
        "finished_at": None,
        "exit_code": None,
        "error": None,
    })
    print(f"[watcher] [{started_at}] State → running  (commit={state.get('commit')})")

    proc = None
    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as log_fh:
            header = (
                f"# gpu_watcher: started at {started_at}\n"
                f"# commit: {state.get('commit')}\n"
                f"# script: {script_path}\n"
                "# " + "─" * 60 + "\n"
            )
            log_fh.write(header)
            log_fh.flush()

            proc = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=repo,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group → clean kill
            )

        # Update state with PID
        current = read_state(repo)
        write_state(repo, {**current, "pid": proc.pid})
        print(f"[watcher] Experiment PID={proc.pid}")

        # Wait with timeout, writing a heartbeat every 30 s so local can monitor
        deadline = time.monotonic() + HARD_TIMEOUT_SECONDS
        while True:
            try:
                proc.wait(timeout=30)
                break  # process finished
            except subprocess.TimeoutExpired:
                if time.monotonic() > deadline:
                    print(f"[watcher] HARD TIMEOUT reached ({HARD_TIMEOUT_SECONDS}s), killing PID={proc.pid}")
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    time.sleep(5)
                    if proc.poll() is None:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait()
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(f"\n# gpu_watcher: KILLED (hard timeout {HARD_TIMEOUT_SECONDS}s)\n")
                    finished_at = _utcnow_iso()
                    write_state(repo, {
                        **read_state(repo),
                        "state": "failed",
                        "finished_at": finished_at,
                        "exit_code": -1,
                        "error": f"hard_timeout_{HARD_TIMEOUT_SECONDS}s",
                        "pid": None,
                    })
                    print(f"[watcher] [{finished_at}] State → failed  (hard timeout)")
                    return
                else:
                    # Heartbeat: refresh updated_at so local knows we're alive
                    cur = read_state(repo)
                    if cur["state"] == "running":
                        write_state(repo, cur)

        exit_code = proc.returncode
        finished_at = _utcnow_iso()

        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n# gpu_watcher: finished at {finished_at}, exit_code={exit_code}\n")

        if exit_code == 0:
            new_state = "completed"
        else:
            new_state = "failed"

        write_state(repo, {
            **read_state(repo),
            "state": new_state,
            "finished_at": finished_at,
            "exit_code": exit_code,
            "error": None if exit_code == 0 else f"exit_code_{exit_code}",
            "pid": None,
        })
        print(f"[watcher] [{finished_at}] State → {new_state}  (exit_code={exit_code})")

    except Exception as e:
        finished_at = _utcnow_iso()
        error_msg = str(e)
        print(f"[watcher] Exception during experiment: {error_msg}")
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        write_state(repo, {
            **read_state(repo),
            "state": "failed",
            "finished_at": finished_at,
            "exit_code": -1,
            "error": f"watcher_exception: {error_msg}",
            "pid": None,
        })
        print(f"[watcher] [{finished_at}] State → failed  (exception)")


# ─────────────────────────── stale state recovery ────────────────────────────


def check_and_recover_stale(repo: Path, state: dict) -> dict:
    """
    If state is 'running' but updated_at is too old (watcher crashed),
    transition back to idle so the next experiment can proceed.
    """
    if state["state"] != "running":
        return state

    updated_at_str = state.get("updated_at")
    if not updated_at_str:
        return state

    try:
        updated_at = datetime.fromisoformat(updated_at_str)
        now = datetime.now(timezone.utc)
        # Ensure both are timezone-aware
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        age = (now - updated_at).total_seconds()
        if age > STALE_RUNNING_TIMEOUT:
            print(
                f"[watcher] STALE: state=running but last heartbeat was {age:.0f}s ago "
                f"(threshold={STALE_RUNNING_TIMEOUT}s). Recovering to idle."
            )
            new_state = {
                **state,
                "state": "idle",
                "error": f"recovered_from_stale_running_after_{age:.0f}s",
                "pid": None,
            }
            write_state(repo, new_state)
            return new_state
    except ValueError:
        pass
    return state


# ─────────────────────────── main loop ────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU watcher daemon for parameter-golf cowork")
    parser.add_argument(
        "--repo-dir",
        default=".",
        help="Path to the shared git repository (default: current directory)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between state-file polls (default: 5)",
    )
    args = parser.parse_args()

    repo = Path(args.repo_dir).resolve()
    if not (repo / ".git").exists():
        print(f"[watcher] ERROR: {repo} does not look like a git repository (no .git dir)")
        sys.exit(1)

    print(f"[watcher] Starting. repo={repo}  poll={args.poll_interval}s")
    print(f"[watcher] State file : {repo / STATE_FILE}")
    print(f"[watcher] Run script : {repo / RUN_SCRIPT}")
    print(f"[watcher] Run log    : {repo / RUN_LOG}")
    print(f"[watcher] Hard timeout: {HARD_TIMEOUT_SECONDS}s per experiment")
    print("[watcher] Press Ctrl-C to stop.\n")

    # Ensure we start with a valid state file
    current_state = read_state(repo)
    if current_state["state"] not in ("idle", "pending"):
        # On startup, if we were 'running' that process is definitely dead now
        if current_state["state"] == "running":
            print("[watcher] Found stale 'running' state on startup, resetting to idle.")
            current_state = _default_idle()
            write_state(repo, current_state)

    last_seen_commit: str | None = None

    try:
        while True:
            current_state = read_state(repo)
            current_state = check_and_recover_stale(repo, current_state)
            state_name = current_state["state"]

            if state_name == "idle":
                # Poll git for new commits
                commit = get_current_commit(repo)
                if commit and commit != last_seen_commit:
                    last_seen_commit = commit
                    print(f"[watcher] HEAD={commit}  state=idle  (watching for pending...)")

                time.sleep(args.poll_interval)

            elif state_name == "pending":
                # A new experiment has been queued by local
                queued_commit = current_state.get("commit")
                print(f"\n[watcher] ── New experiment queued ──────────────────────────────")
                print(f"[watcher]   commit    : {queued_commit}")
                print(f"[watcher]   run script: {current_state.get('run_script')}")

                # Pull latest code
                print("[watcher] Running git pull...")
                pulled = git_pull(repo)
                if not pulled:
                    print("[watcher] git pull failed — marking as failed")
                    write_state(repo, {
                        **current_state,
                        "state": "failed",
                        "finished_at": _utcnow_iso(),
                        "error": "git_pull_failed",
                    })
                    time.sleep(args.poll_interval)
                    continue

                # Verify run script exists and is executable
                script_path = repo / RUN_SCRIPT
                if not script_path.exists():
                    print(f"[watcher] ERROR: {RUN_SCRIPT} not found — marking as failed")
                    write_state(repo, {
                        **current_state,
                        "state": "failed",
                        "finished_at": _utcnow_iso(),
                        "error": f"{RUN_SCRIPT}_not_found",
                    })
                    time.sleep(args.poll_interval)
                    continue

                # Verify the commit in the state file matches HEAD (after pull)
                head_commit = get_current_commit(repo)
                if queued_commit and head_commit and queued_commit != head_commit:
                    print(
                        f"[watcher] WARNING: state commit={queued_commit} != HEAD={head_commit}. "
                        "Proceeding with HEAD."
                    )

                # Run the experiment (blocks until done)
                run_experiment(repo, script_path, {**current_state, "commit": head_commit or queued_commit})
                last_seen_commit = get_current_commit(repo)

            elif state_name in ("completed", "failed"):
                # Waiting for local agent to read results and reset to idle
                # Just sleep — local will flip back to idle when done
                time.sleep(args.poll_interval)

            else:
                print(f"[watcher] Unexpected state '{state_name}', sleeping...")
                time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\n[watcher] Interrupted by user. Exiting cleanly.")
        # If we were in the middle of something, leave the state as-is so local can see it
        sys.exit(0)


if __name__ == "__main__":
    main()
