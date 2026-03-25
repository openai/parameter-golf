#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEST="${DRIVE_SYNC_DEST:-/content/drive/MyDrive/parameter-golf-artifacts}"
INCLUDE_TIMESTAMP_DIR="${INCLUDE_TIMESTAMP_DIR:-1}"

usage() {
  cat <<'EOF'
Usage: bash scripts/sync_artifacts_to_drive.sh [--dest /content/drive/MyDrive/parameter-golf-artifacts]

Options:
  --dest   Destination root on mounted Drive
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument '$1'"
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "/content/drive/MyDrive" ]]; then
  echo "Drive is not mounted at /content/drive/MyDrive; skipping sync."
  exit 0
fi

if [[ "$DEST" != /content/drive/* ]]; then
  echo "ERROR: destination must be inside /content/drive/* to keep active repo/data local."
  exit 2
fi

mkdir -p "$DEST"

TARGET_ROOT="$DEST"
if [[ "$INCLUDE_TIMESTAMP_DIR" == "1" ]]; then
  TARGET_ROOT="$DEST/sync_$(date -u +"%Y%m%d_%H%M%S")"
  mkdir -p "$TARGET_ROOT"
fi

sync_dir() {
  local src="$1"
  local dst="$2"
  if [[ ! -d "$src" ]]; then
    echo "skip_missing_dir: $src"
    return 0
  fi
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src"/ "$dst"/
  else
    cp -a "$src"/. "$dst"/
  fi
  echo "synced_dir: $src -> $dst"
}

sync_file() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "skip_missing_file: $src"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
  echo "synced_file: $src -> $dst"
}

echo "sync_start_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "target_root: $TARGET_ROOT"
echo "note: active repo and dataset remain on /content local disk; only artifacts are copied."

sync_dir "$ROOT_DIR/logs/runs" "$TARGET_ROOT/logs/runs"
sync_dir "$ROOT_DIR/logs/downloads" "$TARGET_ROOT/logs/downloads"
sync_file "$ROOT_DIR/results/runs.csv" "$TARGET_ROOT/results/runs.csv"

echo "sync_complete_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "sync_output: $TARGET_ROOT"
