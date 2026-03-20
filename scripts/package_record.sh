#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -lt 2 ]; then
  echo "usage: bash scripts/package_record.sh <target-record-dir> <log1> [log2 ...]" >&2
  exit 1
fi

TARGET_DIR="$1"
shift

mkdir -p "$TARGET_DIR"
cp records/_template/README.md "$TARGET_DIR/README.md"
cp records/_template/submission.json "$TARGET_DIR/submission.json"
cp train_gpt.py "$TARGET_DIR/train_gpt.py"

for log_file in "$@"; do
  cp "$log_file" "$TARGET_DIR/"
done

echo "Packaged template into $TARGET_DIR"
echo "Next: fill README.md and submission.json with real metrics."
