#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 {core_discovery|core_promotion|eval_time_discovery|storage_discovery|representation_discovery}"
  exit 1
fi

LANE_KEY="$1"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

SEED_BEST_SOURCE="${SEED_BEST_SOURCE:-$ROOT_DIR/autoresearch/train_gpt.best.py}"
SEED_HISTORY_SOURCE="${SEED_HISTORY_SOURCE:-$ROOT_DIR/autoresearch/history.jsonl}"
BACKGROUND="${BACKGROUND:-0}"

COMMON_ENV=(
  "GPUS=1"
  "VAL_LOSS_EVERY=0"
  "AUTORESEARCH_MODEL=${AUTORESEARCH_MODEL:-opus}"
  "CLAUDE_EFFORT=${CLAUDE_EFFORT:-high}"
)

case "$LANE_KEY" in
  core_discovery)
    ENV_VARS=(
      "AUTORESEARCH_LANE=core"
      "AUTORESEARCH_STAGE=discovery"
      "AUTORESEARCH_NAMESPACE=core_discovery"
      "EXPERIMENT_SECONDS=${EXPERIMENT_SECONDS:-180}"
      "MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-100}"
    )
    ;;
  core_promotion)
    ENV_VARS=(
      "AUTORESEARCH_LANE=core"
      "AUTORESEARCH_STAGE=promotion"
      "AUTORESEARCH_NAMESPACE=core_promotion"
      "EXPERIMENT_SECONDS=${EXPERIMENT_SECONDS:-600}"
      "MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-20}"
    )
    ;;
  eval_time_discovery)
    ENV_VARS=(
      "AUTORESEARCH_LANE=eval_time"
      "AUTORESEARCH_STAGE=discovery"
      "AUTORESEARCH_NAMESPACE=eval_time_discovery"
      "EXPERIMENT_SECONDS=${EXPERIMENT_SECONDS:-180}"
      "MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-50}"
      "MAX_EVAL_TIME_MS=${MAX_EVAL_TIME_MS:-60000}"
    )
    ;;
  storage_discovery)
    ENV_VARS=(
      "AUTORESEARCH_LANE=storage"
      "AUTORESEARCH_STAGE=discovery"
      "AUTORESEARCH_NAMESPACE=storage_discovery"
      "EXPERIMENT_SECONDS=${EXPERIMENT_SECONDS:-180}"
      "MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-50}"
      "MAX_QUANTIZATION_GAP=${MAX_QUANTIZATION_GAP:-0.08}"
      "STORAGE_MAX_REGRESSION=${STORAGE_MAX_REGRESSION:-0.003}"
      "STORAGE_MIN_SIZE_IMPROVEMENT=${STORAGE_MIN_SIZE_IMPROVEMENT:-250000}"
    )
    ;;
  representation_discovery)
    ENV_VARS=(
      "AUTORESEARCH_LANE=representation"
      "AUTORESEARCH_STAGE=discovery"
      "AUTORESEARCH_NAMESPACE=representation_discovery"
      "EXPERIMENT_SECONDS=${EXPERIMENT_SECONDS:-180}"
      "MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-30}"
      "REPRESENTATION_VERIFIED=${REPRESENTATION_VERIFIED:-0}"
    )
    ;;
  *)
    echo "Unknown lane key: $LANE_KEY"
    exit 1
    ;;
esac

NAMESPACE=""
for kv in "${ENV_VARS[@]}"; do
  case "$kv" in
    AUTORESEARCH_NAMESPACE=*)
      NAMESPACE="${kv#AUTORESEARCH_NAMESPACE=}"
      ;;
  esac
done

if [ -z "$NAMESPACE" ]; then
  echo "Failed to derive AUTORESEARCH_NAMESPACE for $LANE_KEY"
  exit 1
fi

mkdir -p "$ROOT_DIR/autoresearch/$NAMESPACE"

if [ ! -f "$ROOT_DIR/autoresearch/$NAMESPACE/train_gpt.best.py" ] && [ -f "$SEED_BEST_SOURCE" ]; then
  cp "$SEED_BEST_SOURCE" "$ROOT_DIR/autoresearch/$NAMESPACE/train_gpt.best.py"
fi

if [ -z "${BASELINE_BPB:-}" ] && [ -f "$SEED_HISTORY_SOURCE" ]; then
  BEST_FROM_HISTORY="$(python3 - "$SEED_HISTORY_SOURCE" <<'PY'
import json, sys
best = None
path = sys.argv[1]
with open(path) as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kept") and row.get("val_bpb") is not None:
            val = row["val_bpb"]
            if best is None or val < best:
                best = val
print("" if best is None else best)
PY
)"
  if [ -n "$BEST_FROM_HISTORY" ]; then
    ENV_VARS+=("BASELINE_BPB=$BEST_FROM_HISTORY")
  fi
fi

echo "Launching $LANE_KEY"
printf '  %s\n' "${COMMON_ENV[@]}" "${ENV_VARS[@]}"

if [ "$BACKGROUND" = "1" ]; then
  LOG_FILE="$ROOT_DIR/autoresearch/$NAMESPACE/autoresearch.out"
  env "${COMMON_ENV[@]}" "${ENV_VARS[@]}" \
    nohup python3 autoresearch.py >"$LOG_FILE" 2>&1 &
  echo "Started in background: $!"
  echo "Log: $LOG_FILE"
else
  exec env "${COMMON_ENV[@]}" "${ENV_VARS[@]}" python3 autoresearch.py
fi
