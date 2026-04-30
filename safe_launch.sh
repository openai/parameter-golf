#!/usr/bin/env bash
# Verify train_pr1493.py is the 74dc702 stacking version before launching torchrun.
# If the file regressed (FS layer rolled it back), restore from the local backup or origin.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET="$REPO_DIR/train_pr1493.py"
BACKUP="$REPO_DIR/train_pr1493.py.74dc702"
REQUIRED_SYMBOLS=(paired_head_muon_enabled fold_iha_mixes paired_head_zeropower 'gptq:all-rank Hessian' gptq_damp gptq_block_size)

cd "$REPO_DIR"

ok=1
for sym in "${REQUIRED_SYMBOLS[@]}"; do
  if ! grep -q "$sym" "$TARGET"; then
    ok=0; echo "[safe_launch] missing symbol: $sym" >&2
  fi
done

if [ "$ok" -ne 1 ]; then
  if [ -f "$BACKUP" ]; then
    echo "[safe_launch] restoring train_pr1493.py from local backup" >&2
    cp -p "$BACKUP" "$TARGET"
  else
    echo "[safe_launch] restoring train_pr1493.py from origin/shikhar" >&2
    git show "origin/shikhar:train_pr1493.py" > "$TARGET"
  fi
  for sym in "${REQUIRED_SYMBOLS[@]}"; do
    if ! grep -q "$sym" "$TARGET"; then
      echo "[safe_launch] FATAL: $sym still missing after restore" >&2
      exit 2
    fi
  done
fi

CURRENT_HEAD=$(git rev-parse --short HEAD)
echo "[safe_launch] HEAD=$CURRENT_HEAD file_md5=$(md5sum "$TARGET" | awk '{print $1}') symbols_ok" >&2
exec "$@"
