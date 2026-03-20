#!/bin/bash
set -euo pipefail

VARIANT="${1:-base}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"

cd /workspace/parameter-golf

SCRIPT_PATH="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_ValidEval_TempOnly_Int8Search/train_gpt.py"
RECORD_ROOT="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_ValidEval_TempOnly_Int8Search"

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

case "$VARIANT" in
  base)
    export RUN_ID=leadercore10l_valid_base
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_base"
    ;;
  embedlr08)
    export RUN_ID=leadercore10l_valid_embedlr08
    export TIED_EMBED_LR=0.08
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_embedlr08"
    ;;
  matrixlr005)
    export RUN_ID=leadercore10l_valid_matrixlr005
    export MATRIX_LR=0.05
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_matrixlr005"
    ;;
  warmdown1800)
    export RUN_ID=leadercore10l_valid_warmdown1800
    export WARMDOWN_ITERS=1800
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_warmdown1800"
    ;;
  tokemb_int8)
    export RUN_ID=leadercore10l_valid_tokemb_int8
    export INT8_KEEP_TOK_EMB_FP16=0
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_tokemb_int8"
    ;;
  *)
    echo "Usage: $0 {base|embedlr08|matrixlr005|warmdown1800|tokemb_int8}"
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

nohup python3 -m torch.distributed.run --standalone --nproc_per_node=8 "$SCRIPT_PATH" \
  > "$OUT_DIR/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo "started $RUN_ID data_root_mode=$DATA_ROOT_MODE pid=$PID log=$OUT_DIR/train.log"
