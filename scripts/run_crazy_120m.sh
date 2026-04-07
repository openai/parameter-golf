#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-120}"
MIN_SECONDS_LEFT_TO_START="${MIN_SECONDS_LEFT_TO_START:-900}"
AUTO_STOP_STEP="${AUTO_STOP_STEP:-1000}"
AUTO_STOP_MAX_VAL_BPB="${AUTO_STOP_MAX_VAL_BPB:-1.395}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"

BASE_ENV=(
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  VOCAB_SIZE=1024
  MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}
  VAL_LOSS_EVERY=${VAL_LOSS_EVERY}
  TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY}
  TRAIN_BATCH_TOKENS=524288
  TRAIN_SEQ_LEN=1024
  ROUNDTRIP_EVAL_SEQ_LEN=2048
  ROUNDTRIP_EVAL_STRIDE=64
  TTT_EVAL_SEQ_LEN=1024
  NUM_LAYERS=11
  NUM_HEADS=8
  NUM_KV_HEADS=4
  MODEL_DIM=512
  MUON_WEIGHT_DECAY=0.04
  ATTN_TWICE_ALPHA=0.05
  OVERTONE_INIT_POWER=0.5
)

RUN_NAMES=(
  crazy_fullheads_untied
  crazy_fullheads_shared
  crazy_fullheads_mtp
  crazy_fullheads_hybrid
  crazy_kv2
  crazy_high_twice
  crazy_layerwise_high
  crazy_shared_aggressive
  crazy_overtone_low
  crazy_overtone_hybrid
)

RUN_ENVS=(
  "NUM_KV_HEADS=8 TIE_EMBEDDINGS=0 HEAD_LR=0.008"
  "NUM_KV_HEADS=8 SHARED_DEPTH_N=5 SHARED_DEPTH_EDGE_UNIQUE=2 SHARED_DEPTH_GAIN=0.05"
  "NUM_KV_HEADS=8 MTP_DEPTH=2 MTP_LOSS_WEIGHT=0.05"
  "NUM_KV_HEADS=8 HYBRID_DELTA_EVERY=4"
  "NUM_KV_HEADS=2"
  "ATTN_TWICE_ALPHA=0.12"
  "ATTN_TWICE_ALPHA=0.08 ATTN_TWICE_ALPHA_SLOPE=0.8"
  "SHARED_DEPTH_N=4 SHARED_DEPTH_EDGE_UNIQUE=1 SHARED_DEPTH_GAIN=0.08"
  "OVERTONE_INIT_POWER=0.25"
  "OVERTONE_INIT_POWER=0.75 HYBRID_DELTA_EVERY=4"
)

start_ts="$(date +%s)"
budget_seconds="$((TOTAL_BUDGET_MINUTES * 60))"

for idx in "${!RUN_NAMES[@]}"; do
  name="${RUN_NAMES[$idx]}"
  extra="${RUN_ENVS[$idx]}"
  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  remaining="$((budget_seconds - elapsed))"

  if [ "$remaining" -lt "$MIN_SECONDS_LEFT_TO_START" ]; then
    echo "=== Budget stop: remaining=${remaining}s is below minimum start window ${MIN_SECONDS_LEFT_TO_START}s ==="
    exit 0
  fi

  echo
  echo "=== Crazy run ${name} elapsed=${elapsed}s remaining=${remaining}s ==="

  env \
    "${BASE_ENV[@]}" \
    RUN_ID="$name" \
    AUTO_STOP_STEP="$AUTO_STOP_STEP" \
    AUTO_STOP_MAX_VAL_BPB="$AUTO_STOP_MAX_VAL_BPB" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    ${extra} \
    bash scripts/run_remote_experiment.sh

  tail -n 12 "logs/${name}.txt"
done
