#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${CASTOR_TRAIN_ENV:-$ROOT/configs/train/l7grow_v4_castor.env}"
PRETRAIN_CONFIG="${CASTOR_PRETRAIN_CONFIG:-$ROOT/configs/datasets/castor_pretrain_mix_v0.yaml}"
TRAIN_MODE="${CASTOR_TRAIN_MODE:-resume}"
CONTINUE_WITH_TRAIN_LOADER_STATE="${CASTOR_CONTINUE_WITH_TRAIN_LOADER_STATE:-1}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

resolve_path() {
  local value="$1"
  if [[ -z "$value" ]]; then
    return 0
  fi
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$ROOT/${value#./}"
  fi
}

DATASETS_DIR_ABS="$(resolve_path "${DATASETS_DIR:-./data/datasets/castor_pretrain_sp8192_v0}")"
CHECKPOINT_PATH_ABS="$(resolve_path "${RESUME_CHECKPOINT:-./runs/${RUN_ID:-castor_l7grow_v4_seed1337}/checkpoints/latest.pt}")"

mkdir -p "$ROOT/logs"

if ! compgen -G "$DATASETS_DIR_ABS/fineweb_train_*.bin" > /dev/null; then
  echo "pretokenized shards not found in $DATASETS_DIR_ABS; preparing them now"
  "$ROOT/scripts/prepare_l7grow_data.sh"
fi

if ! compgen -G "$DATASETS_DIR_ABS/fineweb_val_*.bin" > /dev/null; then
  echo "validation shards are not ready yet in $DATASETS_DIR_ABS" >&2
  echo "wait for scripts/prepare_l7grow_data.sh to finish so summary.json and fineweb_val_*.bin are written" >&2
  exit 1
fi

if [[ "$TRAIN_MODE" == "continue_phase" ]]; then
  if [[ ! -f "$CHECKPOINT_PATH_ABS" ]]; then
    echo "continue_phase requested but checkpoint is missing: $CHECKPOINT_PATH_ABS" >&2
    exit 1
  fi

  BASE_RUN_ID="${RUN_ID:-castor_l7grow_v4_seed1337}"
  NEW_RUN_ID="${CASTOR_NEW_RUN_ID:-${BASE_RUN_ID}_phase_$(date +%Y%m%d_%H%M%S)}"
  NEW_RUN_DIR="$ROOT/runs/$NEW_RUN_ID"
  EXPORTED_MODEL="$NEW_RUN_DIR/init_from_${BASE_RUN_ID}.pt"
  EXPORTED_TRAIN_LOADER="$NEW_RUN_DIR/init_train_loader_state.pt"
  METADATA_JSON="$NEW_RUN_DIR/continue_from.json"
  mkdir -p "$NEW_RUN_DIR"

  ./.venv/bin/python "$ROOT/scripts/export_training_checkpoint.py" \
    --checkpoint "$CHECKPOINT_PATH_ABS" \
    --output "$EXPORTED_MODEL" \
    --train-loader-output "$EXPORTED_TRAIN_LOADER" \
    --metadata-output "$METADATA_JSON" >/dev/null

  ACTIVE_SEQ_LEN="$(
    ./.venv/bin/python - <<'PY' "$METADATA_JSON"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(int(payload.get("active_train_seq_len", 0) or 0))
PY
  )"
  if [[ "$ACTIVE_SEQ_LEN" -le 0 ]]; then
    echo "could not determine active_train_seq_len from $METADATA_JSON" >&2
    exit 1
  fi

  START_LOOPING_ACTIVE="$(
    ./.venv/bin/python - <<'PY' "$METADATA_JSON"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(1 if payload.get("looping_active", False) else 0)
PY
  )"

  export RUN_ID="$NEW_RUN_ID"
  export RUN_DIR="$NEW_RUN_DIR"
  export CHECKPOINT_DIR="$NEW_RUN_DIR/checkpoints"
  export RESUME_CHECKPOINT="$NEW_RUN_DIR/checkpoints/latest.pt"
  export INIT_MODEL_PATH="$EXPORTED_MODEL"
  export MODEL_PATH="$NEW_RUN_DIR/final_model.pt"
  export BEST_MODEL_PATH="$NEW_RUN_DIR/best_model.pt"
  export BEST_MODEL_METADATA_PATH="$NEW_RUN_DIR/best_model.json"
  export QUANTIZED_MODEL_PATH="$NEW_RUN_DIR/final_model.int6.ptz"
  export LOGFILE="$ROOT/logs/$NEW_RUN_ID.txt"
  export TRAIN_SEQ_LEN="$ACTIVE_SEQ_LEN"
  export TRAIN_SEQ_SCHEDULE="${ACTIVE_SEQ_LEN}@1.000"
  export START_LOOPING_ACTIVE="$START_LOOPING_ACTIVE"
  if [[ "$CONTINUE_WITH_TRAIN_LOADER_STATE" != "0" && -f "$EXPORTED_TRAIN_LOADER" ]]; then
    export INIT_TRAIN_LOADER_STATE_PATH="$EXPORTED_TRAIN_LOADER"
  else
    unset INIT_TRAIN_LOADER_STATE_PATH || true
  fi

  echo "continue_phase: starting new run $NEW_RUN_ID from $CHECKPOINT_PATH_ABS at seq_len=$ACTIVE_SEQ_LEN looping_active=$START_LOOPING_ACTIVE"
elif [[ -f "$CHECKPOINT_PATH_ABS" ]]; then
  echo "resume checkpoint found: $CHECKPOINT_PATH_ABS"
else
  echo "no resume checkpoint found; trainer will warm-start from INIT_MODEL_PATH if available"
fi

LOGFILE_ABS="$(resolve_path "${LOGFILE:-./logs/${RUN_ID:-castor_l7grow_v4_seed1337}.txt}")"
CONSOLE_LOG="${LOGFILE_ABS%.txt}.console.txt"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
  SIMON_ENV_FILE="$ENV_FILE" ./.venv/bin/torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    trainers/l7_grow/train_gpt.py |& tee -a "$CONSOLE_LOG"
else
  SIMON_ENV_FILE="$ENV_FILE" ./.venv/bin/python -u trainers/l7_grow/train_gpt.py |& tee -a "$CONSOLE_LOG"
fi
