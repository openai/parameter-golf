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

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
if ! command -v "$TORCHRUN_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$TORCHRUN_BIN' is not available in PATH."
  exit 1
fi

RUN_ID="${RUN_ID:-baseline_sp1024}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH does not exist: $DATA_PATH"
  echo "Run: bash scripts/download_data.sh --variant sp1024 --train-shards 80"
  exit 2
fi

if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "ERROR: TOKENIZER_PATH does not exist: $TOKENIZER_PATH"
  echo "Run: bash scripts/download_data.sh --variant sp1024 --train-shards 80"
  exit 2
fi

mapfile -t TRAIN_FILES < <(find "$DATA_PATH" -maxdepth 1 -name 'fineweb_train_*.bin' | sort)
mapfile -t VAL_FILES < <(find "$DATA_PATH" -maxdepth 1 -name 'fineweb_val_*.bin' | sort)
if [[ "${#TRAIN_FILES[@]}" -lt 1 ]]; then
  echo "ERROR: No train shards found in $DATA_PATH"
  exit 2
fi
if [[ "${#VAL_FILES[@]}" -lt 1 ]]; then
  echo "ERROR: No val shards found in $DATA_PATH"
  exit 2
fi

DATA_BASENAME="$(basename "${DATA_PATH%/}")"
DATA_VARIANT="${DATA_BASENAME#fineweb10B_}"
if [[ "$DATA_BASENAME" == "$DATA_VARIANT" ]]; then
  DATA_VARIANT="unknown"
fi

TIMESTAMP="$(date -u +"%Y%m%d_%H%M%S")"
RUN_DIR="$ROOT_DIR/logs/runs/${TIMESTAMP}_${RUN_ID}"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/console.log"

GIT_COMMIT="$(git rev-parse HEAD)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
GPU_NAME="unknown"
GPU_MEMORY_MIB="unknown"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')"
  GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d '\r')"
fi
TIMESTAMP_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
TRAIN_SHARDS_FOUND="$(find "$DATA_PATH" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l | tr -d ' ')"

cat > "$RUN_DIR/run_env.txt" <<EOF
RUN_ID=$RUN_ID
DATA_PATH=$DATA_PATH
TOKENIZER_PATH=$TOKENIZER_PATH
VOCAB_SIZE=$VOCAB_SIZE
SEED=$SEED
NPROC_PER_NODE=$NPROC_PER_NODE
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-}
ITERATIONS=${ITERATIONS:-}
TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-}
VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-}
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-}
EOF

export META_RUN_ID="$RUN_ID"
export META_TIMESTAMP_UTC="$TIMESTAMP_UTC"
export META_GIT_COMMIT="$GIT_COMMIT"
export META_GIT_BRANCH="$GIT_BRANCH"
export META_SEED="$SEED"
export META_GPU_NAME="$GPU_NAME"
export META_GPU_MEMORY_MIB="$GPU_MEMORY_MIB"
export META_DATA_VARIANT="$DATA_VARIANT"
export META_TRAIN_SHARDS="$TRAIN_SHARDS_FOUND"
export META_DATA_PATH="$DATA_PATH"
export META_TOKENIZER_PATH="$TOKENIZER_PATH"
export META_VOCAB_SIZE="$VOCAB_SIZE"

"$PYTHON_BIN" - <<'PY' > "$RUN_DIR/run_metadata.json"
import json
import os

meta = {
    "run_id": os.environ.get("META_RUN_ID", ""),
    "timestamp_utc": os.environ.get("META_TIMESTAMP_UTC", ""),
    "git_commit": os.environ.get("META_GIT_COMMIT", ""),
    "branch": os.environ.get("META_GIT_BRANCH", ""),
    "seed": os.environ.get("META_SEED", ""),
    "gpu_name": os.environ.get("META_GPU_NAME", ""),
    "gpu_memory_mib": os.environ.get("META_GPU_MEMORY_MIB", ""),
    "data_variant": os.environ.get("META_DATA_VARIANT", ""),
    "train_shards": os.environ.get("META_TRAIN_SHARDS", ""),
    "data_path": os.environ.get("META_DATA_PATH", ""),
    "tokenizer_path": os.environ.get("META_TOKENIZER_PATH", ""),
    "vocab_size": os.environ.get("META_VOCAB_SIZE", ""),
}
print(json.dumps(meta, sort_keys=True, indent=2))
PY

{
  echo "HARNESS_META timestamp_utc=$TIMESTAMP_UTC"
  echo "HARNESS_META run_dir=$RUN_DIR"
  echo "HARNESS_META run_id=$RUN_ID"
  echo "HARNESS_META git_commit=$GIT_COMMIT"
  echo "HARNESS_META branch=$GIT_BRANCH"
  echo "HARNESS_META seed=$SEED"
  echo "HARNESS_META gpu_name=$GPU_NAME"
  echo "HARNESS_META gpu_memory_mib=$GPU_MEMORY_MIB"
  echo "HARNESS_META data_variant=$DATA_VARIANT"
  echo "HARNESS_META train_shards=$TRAIN_SHARDS_FOUND"
  echo "HARNESS_META data_path=$DATA_PATH"
  echo "HARNESS_META tokenizer_path=$TOKENIZER_PATH"
  echo "HARNESS_META vocab_size=$VOCAB_SIZE"
  echo "HARNESS_META command=$TORCHRUN_BIN --standalone --nproc_per_node=$NPROC_PER_NODE train_gpt.py"
} | tee "$LOG_FILE"

START_EPOCH="$(date +%s)"
set +e
(
  export PYTHONUNBUFFERED=1
  export RUN_ID="$RUN_ID"
  export DATA_PATH="$DATA_PATH"
  export TOKENIZER_PATH="$TOKENIZER_PATH"
  export VOCAB_SIZE="$VOCAB_SIZE"
  export SEED="$SEED"
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py
) 2>&1 | tee -a "$LOG_FILE"
RUN_STATUS=${PIPESTATUS[0]}
set -e
END_EPOCH="$(date +%s)"
ELAPSED_SECONDS="$((END_EPOCH - START_EPOCH))"
echo "HARNESS_META elapsed_wallclock_seconds=$ELAPSED_SECONDS" | tee -a "$LOG_FILE"

SUMMARY="$(
"$PYTHON_BIN" - "$LOG_FILE" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read()
final = re.findall(r"final_int8_zlib_roundtrip(?:_exact)? val_loss:([0-9.]+) val_bpb:([0-9.]+)", text)
if final:
    vloss, vbpb = final[-1]
    print(f"val_loss={vloss} val_bpb={vbpb} source=final_int8_zlib_roundtrip")
    raise SystemExit(0)

steps = re.findall(r"step:\d+/\d+ val_loss:([0-9.]+) val_bpb:([0-9.]+)", text)
if steps:
    vloss, vbpb = steps[-1]
    print(f"val_loss={vloss} val_bpb={vbpb} source=step_validation")
PY
)"

if [[ -n "$SUMMARY" ]]; then
  echo "BASELINE_SUMMARY $SUMMARY" | tee -a "$LOG_FILE"
else
  echo "BASELINE_SUMMARY unavailable (no val summary line found)" | tee -a "$LOG_FILE"
fi

if [[ "$RUN_STATUS" -ne 0 ]]; then
  echo "ERROR: Baseline run failed with exit code $RUN_STATUS. See $LOG_FILE."
  exit "$RUN_STATUS"
fi

"$PYTHON_BIN" scripts/parse_logs.py --log "$LOG_FILE" --quiet || true
echo "baseline_run_ok: $RUN_DIR"
