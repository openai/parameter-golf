#!/bin/bash
set -euo pipefail

VARIANT="${1:-base}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"

cd /workspace/parameter-golf

SCRIPT_PATH="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_11L_WD04_524K_SWA_ValidEval/train_gpt.py"
RECORD_ROOT="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_11L_WD04_524K_SWA_ValidEval"

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
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=524288
export NUM_LAYERS=11
export MLP_MULT=3
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_WD=0.04
export ADAM_WD=0.04
export SWA_ENABLED=1
export SWA_EVERY=200
export SWA_START_SCALE=0.5
export WARMDOWN_ITERS=3000
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500

case "$VARIANT" in
  base)
    export RUN_ID=frontier11l_wd04_524k_base
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_base"
    ;;
  wd038)
    export RUN_ID=frontier11l_wd038_524k
    export MUON_WD=0.038
    export ADAM_WD=0.038
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_wd038"
    ;;
  no_swa)
    export RUN_ID=frontier11l_wd04_524k_no_swa
    export SWA_ENABLED=0
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_no_swa"
    ;;
  matrixlr002)
    export RUN_ID=frontier11l_wd04_524k_matrixlr002
    export MATRIX_LR=0.02
    export SCALAR_LR=0.02
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_matrixlr002"
    ;;
  tokemb_int8)
    export RUN_ID=frontier11l_wd04_524k_tokemb_int8
    export INT8_KEEP_TOK_EMB_FP16=0
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_tokemb_int8"
    ;;
  *)
    echo "Usage: $0 {base|wd038|no_swa|matrixlr002|tokemb_int8}"
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

nohup python3 -m torch.distributed.run --standalone --nproc_per_node=8 "$SCRIPT_PATH" \
  > "$OUT_DIR/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo "started $RUN_ID data_root_mode=$DATA_ROOT_MODE pid=$PID log=$OUT_DIR/train.log"
