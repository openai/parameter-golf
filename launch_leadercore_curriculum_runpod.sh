#!/bin/bash
set -euo pipefail

VARIANT="${1:-base}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"

cd /workspace/parameter-golf

SCRIPT_PATH="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_BoundaryCurriculumLRFloor/train_gpt.py"
RECORD_ROOT="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_BoundaryCurriculumLRFloor"

case "$DATA_ROOT_MODE" in
  workspace)
    export DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
    export TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
    ;;
  tmp)
    export DATA_PATH=/tmp/parameter-golf-data/datasets/fineweb10B_sp1024
    export TOKENIZER_PATH=/tmp/parameter-golf-data/tokenizers/fineweb_1024_bpe.model
    ;;
  *)
    echo "DATA_ROOT_MODE must be one of: workspace, tmp"
    exit 1
    ;;
esac

export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=200
export VOCAB_SIZE=1024
export INT8_KEEP_TOK_EMB_FP16=1
export SHORT_SEQ_LEN=512
export CURRICULUM_SWITCH_FRACTION=0.30
export CURRICULUM_BOUNDARY_ALIGN=1
export LR_FLOOR_SCALE=0.15

case "$VARIANT" in
  base)
    export RUN_ID=leadercore10l_curriculum_base
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_base"
    ;;
  no_curriculum)
    export RUN_ID=leadercore10l_curriculum_no_curriculum
    export CURRICULUM_SWITCH_FRACTION=0
    export SHORT_SEQ_LEN=1024
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_no_curriculum"
    ;;
  no_lrfloor)
    export RUN_ID=leadercore10l_curriculum_no_lrfloor
    export LR_FLOOR_SCALE=0.0
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_no_lrfloor"
    ;;
  floor010)
    export RUN_ID=leadercore10l_curriculum_floor010
    export LR_FLOOR_SCALE=0.10
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_floor010"
    ;;
  switch035)
    export RUN_ID=leadercore10l_curriculum_switch035
    export CURRICULUM_SWITCH_FRACTION=0.35
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_switch035"
    ;;
  boundary_off)
    export RUN_ID=leadercore10l_curriculum_boundary_off
    export CURRICULUM_BOUNDARY_ALIGN=0
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_boundary_off"
    ;;
  *)
    echo "Usage: $0 {base|no_curriculum|no_lrfloor|floor010|switch035|boundary_off}"
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

nohup python3 -m torch.distributed.run --standalone --nproc_per_node=8 "$SCRIPT_PATH" \
  > "$OUT_DIR/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo "started $RUN_ID data_root_mode=$DATA_ROOT_MODE pid=$PID log=$OUT_DIR/train.log"
