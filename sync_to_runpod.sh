#!/usr/bin/env bash
# ============================================================
# Sync local changes to a RunPod pod via rsync over SSH.
# Usage: RUNPOD_HOST=user@host:port ./sync_to_runpod.sh [--watch]
#
# Set RUNPOD_HOST in your shell or .env before running.
# Example:
#   export RUNPOD_HOST="root@213.34.12.XX"
#   export RUNPOD_PORT="22204"
#   ./sync_to_runpod.sh --watch
# ============================================================
set -e

HOST="${RUNPOD_HOST:?Set RUNPOD_HOST=user@ip}"
PORT="${RUNPOD_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/parameter-golf}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

RSYNC_OPTS=(
  -avz --progress
  --exclude=".venv"
  --exclude=".git"
  --exclude="__pycache__"
  --exclude="*.pyc"
  --exclude="data/datasets"
  --exclude="data/tokenizers"
  -e "ssh -p $PORT"
)

do_sync() {
  rsync "${RSYNC_OPTS[@]}" "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/"
  echo "[$(date '+%H:%M:%S')] Synced to $HOST:$REMOTE_DIR"
}

if [ "${1}" = "--watch" ]; then
  echo "Watching for changes (Ctrl-C to stop)..."
  do_sync
  # Use inotifywait if available, else poll
  if command -v inotifywait &>/dev/null; then
    while inotifywait -r -e modify,create,delete,move \
        --exclude '\.git|\.venv|__pycache__|data/datasets|data/tokenizers' \
        "$LOCAL_DIR" 2>/dev/null; do
      sleep 0.5
      do_sync
    done
  else
    while sleep 5; do do_sync; done
  fi
else
  do_sync
fi
