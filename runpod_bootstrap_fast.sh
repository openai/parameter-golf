#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/openai/parameter-golf.git}"
REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"

if [[ ! -d "${REPO_DIR}" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git pull --ff-only

# Sweep stage
bash run_remote_fast.sh

# Submission stage (requires GITHUB_ID and push permissions).
if [[ -n "${GITHUB_ID:-}" ]]; then
  bash submit_remote_fast.sh
else
  echo "GITHUB_ID not set; skipped submit_remote_fast.sh"
  echo "To submit now: export GITHUB_ID=your_handle; bash submit_remote_fast.sh"
fi
