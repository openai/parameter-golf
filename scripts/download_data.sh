#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v "${PYTHON_BIN:-python3}" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python was not found. Install python3 and rerun."
  exit 1
fi

VARIANT="sp1024"
TRAIN_SHARDS="1"

usage() {
  cat <<'EOF'
Usage: bash scripts/download_data.sh [--variant sp1024] [--train-shards 1]

Options:
  --variant       Dataset/tokenizer variant to download (default: sp1024)
  --train-shards  Number of train shards to request (default: 1)
  -h, --help      Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="${2:-}"
      shift 2
      ;;
    --train-shards)
      TRAIN_SHARDS="${2:-}"
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

if ! [[ "$TRAIN_SHARDS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --train-shards must be a non-negative integer (received '$TRAIN_SHARDS')."
  exit 2
fi

case "$VARIANT" in
  byte260)
    DATASET_DIR_NAME="fineweb10B_byte260"
    ;;
  sp[0-9]*)
    DATASET_DIR_NAME="fineweb10B_${VARIANT}"
    ;;
  *)
    echo "ERROR: Unsupported variant '$VARIANT'. Expected byte260 or sp<VOCAB_SIZE>."
    exit 2
    ;;
esac

TIMESTAMP="$(date -u +"%Y%m%d_%H%M%S")"
RUN_DIR="$ROOT_DIR/logs/downloads/${TIMESTAMP}_${VARIANT}_train${TRAIN_SHARDS}"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/download.log"
FILES_BEFORE="$RUN_DIR/files_before.txt"
FILES_AFTER="$RUN_DIR/files_after.txt"
NEW_FILES="$RUN_DIR/new_files.txt"

snapshot_files() {
  {
    [[ -d data/datasets ]] && find data/datasets -type f
    [[ -d data/tokenizers ]] && find data/tokenizers -type f
    [[ -f data/manifest.json ]] && echo data/manifest.json
    [[ -f data/docs_selected.jsonl ]] && echo data/docs_selected.jsonl
  } | sort
}

snapshot_files > "$FILES_BEFORE"

echo "download_start_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee "$LOG_FILE"
echo "variant: $VARIANT" | tee -a "$LOG_FILE"
echo "train_shards: $TRAIN_SHARDS" | tee -a "$LOG_FILE"
echo "dataset_dir: data/datasets/$DATASET_DIR_NAME" | tee -a "$LOG_FILE"
echo "command: $PYTHON_BIN data/cached_challenge_fineweb.py --variant $VARIANT --train-shards $TRAIN_SHARDS" | tee -a "$LOG_FILE"

set +e
"$PYTHON_BIN" data/cached_challenge_fineweb.py --variant "$VARIANT" --train-shards "$TRAIN_SHARDS" 2>&1 | tee -a "$LOG_FILE"
DL_STATUS=${PIPESTATUS[0]}
set -e

if [[ "$DL_STATUS" -ne 0 ]]; then
  echo "ERROR: Downloader failed with exit code $DL_STATUS. See $LOG_FILE for details."
  exit "$DL_STATUS"
fi

snapshot_files > "$FILES_AFTER"
comm -13 "$FILES_BEFORE" "$FILES_AFTER" > "$NEW_FILES"

if [[ -s "$NEW_FILES" ]]; then
  echo "new_files_downloaded:" | tee -a "$LOG_FILE"
  cat "$NEW_FILES" | tee -a "$LOG_FILE"
else
  echo "new_files_downloaded: none (all requested artifacts already present)" | tee -a "$LOG_FILE"
fi

"$PYTHON_BIN" - "$VARIANT" "$TRAIN_SHARDS" <<'PY' | tee -a "$LOG_FILE"
import json
import sys
from pathlib import Path

variant = sys.argv[1]
train_shards_required = int(sys.argv[2])
root = Path(".")

if variant == "byte260":
    dataset_dir_name = "fineweb10B_byte260"
elif variant.startswith("sp") and variant[2:].isdigit():
    dataset_dir_name = f"fineweb10B_{variant}"
else:
    raise SystemExit(f"ERROR: Unsupported variant {variant!r}")

dataset_dir = root / "data" / "datasets" / dataset_dir_name
manifest_path = root / "data" / "manifest.json"

if not dataset_dir.is_dir():
    raise SystemExit(f"ERROR: dataset directory not found: {dataset_dir}")
if not manifest_path.is_file():
    raise SystemExit(f"ERROR: manifest not found at {manifest_path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir_name), None)
if dataset_entry is None:
    raise SystemExit(f"ERROR: dataset {dataset_dir_name} not found in manifest")

val_expected = int((dataset_entry.get("stats") or {}).get("files_val", 0))
if val_expected <= 0:
    raise SystemExit(f"ERROR: invalid files_val in manifest for {dataset_dir_name}: {val_expected}")

train_count = len(list(dataset_dir.glob("fineweb_train_*.bin")))
val_count = len(list(dataset_dir.glob("fineweb_val_*.bin")))

if train_count < train_shards_required:
    raise SystemExit(
        f"ERROR: expected at least {train_shards_required} train shards for {dataset_dir_name}, found {train_count}"
    )
if val_count < val_expected:
    raise SystemExit(
        f"ERROR: expected at least {val_expected} val shards for {dataset_dir_name}, found {val_count}"
    )

tokenizer_name = dataset_entry.get("tokenizer_name")
tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
if tokenizer_entry is None:
    raise SystemExit(f"ERROR: tokenizer {tokenizer_name!r} referenced by manifest is missing")

missing = []
for key in ("model_path", "vocab_path", "path"):
    value = tokenizer_entry.get(key)
    if value and not (root / "data" / value).is_file():
        missing.append(f"data/{value}")
if missing:
    raise SystemExit(f"ERROR: missing tokenizer artifacts after download: {missing}")

summary = {
    "dataset_dir": str(dataset_dir),
    "train_shards_found": train_count,
    "val_shards_found": val_count,
    "val_shards_expected": val_expected,
    "tokenizer_name": tokenizer_name,
}
print("verification_ok:", json.dumps(summary, sort_keys=True))
PY

echo "download_complete_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$LOG_FILE"
echo "log_file: $LOG_FILE"
