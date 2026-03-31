#!/bin/bash
set -euo pipefail

VARIANT="${1:-base}"
DATA_ROOT_MODE="${DATA_ROOT_MODE:-workspace}"

cd /workspace/parameter-golf

SCRIPT_PATH="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_MixedQuant_MicroTTT/train_gpt.py"
RECORD_ROOT="/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_MixedQuant_MicroTTT"

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
export MIXED_QUANT_ENABLED=1
export MLP_INT_BITS=5
export ATTN_INT_BITS=6
export MICRO_TTT_ENABLED=1
export MICRO_TTT_LR=0.002
export MICRO_TTT_MOMENTUM=0.9
export MICRO_TTT_STEPS=32
export MICRO_TTT_BATCH_SEQS=32

case "$VARIANT" in
  base)
    export RUN_ID=leadercore10l_mixedquant_microttt_base
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_base"
    ;;
  no_microttt)
    export RUN_ID=leadercore10l_mixedquant_no_microttt
    export MICRO_TTT_ENABLED=0
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_no_microttt"
    ;;
  no_mixedquant)
    export RUN_ID=leadercore10l_mixedquant_off
    export MIXED_QUANT_ENABLED=0
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_no_mixedquant"
    ;;
  mlp6_attn8)
    export RUN_ID=leadercore10l_mixedquant_mlp6_attn8
    export MLP_INT_BITS=6
    export ATTN_INT_BITS=8
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_mlp6_attn8"
    ;;
  microttt64)
    export RUN_ID=leadercore10l_mixedquant_microttt64
    export MICRO_TTT_STEPS=64
    OUT_DIR="$RECORD_ROOT/runpod_${DATA_ROOT_MODE}_microttt64"
    ;;
  *)
    echo "Usage: $0 {base|no_microttt|no_mixedquant|mlp6_attn8|microttt64}"
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

nohup python3 -m torch.distributed.run --standalone --nproc_per_node=8 "$SCRIPT_PATH" \
  > "$OUT_DIR/train.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT_DIR/train.pid"
echo "started $RUN_ID data_root_mode=$DATA_ROOT_MODE pid=$PID log=$OUT_DIR/train.log"
