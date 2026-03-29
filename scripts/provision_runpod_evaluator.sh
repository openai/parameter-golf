#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKSPACE_PARENT="$(dirname "$WORKSPACE_ROOT")"
WORKSPACE_NAME="$(basename "$WORKSPACE_ROOT")"
REMOTE_HOST="${RUNPOD_SSH_HOST:-103.207.149.123}"
REMOTE_PORT="${RUNPOD_SSH_PORT:-11530}"
REMOTE_USER="${RUNPOD_SSH_USER:-root}"
REMOTE_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/google_compute_engine}"
REMOTE_BASE="${RUNPOD_REMOTE_BASE:-/workspace/hyperactive-octupus}"

SSH_BASE=(ssh -o StrictHostKeyChecking=no -i "$REMOTE_KEY" -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST")

"${SSH_BASE[@]}" "mkdir -p /workspace"

tar \
  --exclude="$WORKSPACE_NAME/.git" \
  --exclude="$WORKSPACE_NAME/.venv" \
  --exclude="$WORKSPACE_NAME/venv" \
  --exclude="$WORKSPACE_NAME/**/.venv" \
  --exclude="$WORKSPACE_NAME/**/venv" \
  --exclude="$WORKSPACE_NAME/**/node_modules" \
  --exclude="$WORKSPACE_NAME/.pytest_cache" \
  --exclude="$WORKSPACE_NAME/.mypy_cache" \
  --exclude="$WORKSPACE_NAME/.ruff_cache" \
  --exclude="$WORKSPACE_NAME/**/.ruff_cache" \
  --exclude="$WORKSPACE_NAME/**/__pycache__" \
  --exclude="$WORKSPACE_NAME/**/*.pyc" \
  --exclude="$WORKSPACE_NAME/**/.DS_Store" \
  --exclude="$WORKSPACE_NAME/parameter-golf/skydiscover_runs" \
  --exclude="$WORKSPACE_NAME/parameter-golf/skydiscover_runtime" \
  --exclude="$WORKSPACE_NAME/parameter-golf/verification-runs" \
  --exclude="$WORKSPACE_NAME/parameter-golf/hypothesis-runs" \
  --exclude="$WORKSPACE_NAME/simplebackend/multi-agent/runs" \
  -C "$WORKSPACE_PARENT" \
  -cf - "$WORKSPACE_NAME" | "${SSH_BASE[@]}" "cd /workspace && tar -xf -"

"${SSH_BASE[@]}" "cd '$REMOTE_BASE' && python -m pip install --upgrade pip uv && uv pip install --system --upgrade torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && uv pip install --system sentencepiece"

echo "RunPod workspace synced to $REMOTE_BASE"
