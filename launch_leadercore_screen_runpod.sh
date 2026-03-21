#!/bin/bash
set -euo pipefail

VARIANT="${1:-base}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-tmp}"
SCREEN_SECONDS="${SCREEN_SECONDS:-180}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

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

export MAX_WALLCLOCK_SECONDS="$SCREEN_SECONDS"
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=25
export VOCAB_SIZE=1024
export INT8_KEEP_TOK_EMB_FP16=1
export PROXY_SKIP_EXPORT=1

case "$VARIANT" in
  base)
    export RUN_ID=leadercore10l_screen_base
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_base"
    ;;
  embedlr08)
    export RUN_ID=leadercore10l_screen_embedlr08
    export TIED_EMBED_LR=0.08
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_embedlr08"
    ;;
  matrixlr005)
    export RUN_ID=leadercore10l_screen_matrixlr005
    export MATRIX_LR=0.05
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_matrixlr005"
    ;;
  matrixlr006)
    export RUN_ID=leadercore10l_screen_matrixlr006
    export MATRIX_LR=0.06
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_matrixlr006"
    ;;
  warmdown1800)
    export RUN_ID=leadercore10l_screen_warmdown1800
    export WARMDOWN_ITERS=1800
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_warmdown1800"
    ;;
  warmdown800)
    export RUN_ID=leadercore10l_screen_warmdown800
    export WARMDOWN_ITERS=800
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_warmdown800"
    ;;
  gradclip03)
    export RUN_ID=leadercore10l_screen_gradclip03
    export GRAD_CLIP_NORM=0.3
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_gradclip03"
    ;;
  muon099)
    export RUN_ID=leadercore10l_screen_muon099
    export MUON_MOMENTUM=0.99
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_muon099"
    ;;
  warmdown800_matrixlr006)
    export RUN_ID=leadercore10l_screen_warmdown800_matrixlr006
    export WARMDOWN_ITERS=800
    export MATRIX_LR=0.06
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_warmdown800_matrixlr006"
    ;;
  muonwarm300)
    export RUN_ID=leadercore10l_screen_muonwarm300
    export MUON_MOMENTUM_WARMUP_STEPS=300
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_muonwarm300"
    ;;
  qkgain17)
    export RUN_ID=leadercore10l_screen_qkgain17
    export QK_GAIN_INIT=1.7
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_qkgain17"
    ;;
  width520)
    export RUN_ID=leadercore10l_screen_width520
    export MODEL_DIM=520
    OUT_DIR="$RECORD_ROOT/screen_${DATA_ROOT_MODE}_width520"
    ;;
  *)
    echo "Usage: $0 {base|embedlr08|matrixlr005|matrixlr006|warmdown1800|warmdown800|warmdown800_matrixlr006|gradclip03|muon099|muonwarm300|qkgain17|width520}"
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

nohup python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_PATH" \
  > "$OUT_DIR/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo "started $RUN_ID data_root_mode=$DATA_ROOT_MODE nproc_per_node=$NPROC_PER_NODE screen_seconds=$SCREEN_SECONDS pid=$PID log=$OUT_DIR/train.log"
