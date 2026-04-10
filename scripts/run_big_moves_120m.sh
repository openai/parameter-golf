#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-120}"
MIN_SECONDS_LEFT_TO_START="${MIN_SECONDS_LEFT_TO_START:-480}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-540}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
AUTO_STOP_STEP="${AUTO_STOP_STEP:-0}"
AUTO_STOP_MAX_VAL_BPB="${AUTO_STOP_MAX_VAL_BPB:-0}"

BASE_ENV=(
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  VOCAB_SIZE=1024
  MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS}
  VAL_LOSS_EVERY=${VAL_LOSS_EVERY}
  TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY}
  SKIP_FINAL_EVAL=${SKIP_FINAL_EVAL}
  TRAIN_BATCH_TOKENS=524288
  TRAIN_SEQ_LEN=1024
  ROUNDTRIP_EVAL_SEQ_LEN=2048
  ROUNDTRIP_EVAL_STRIDE=64
  TTT_EVAL_SEQ_LEN=1024
  NUM_LAYERS=11
  NUM_HEADS=8
  NUM_KV_HEADS=4
  MODEL_DIM=512
  MLP_MULT=3
  TIE_EMBEDDINGS=1
  MATRIX_LR=0.025
  SCALAR_LR=0.025
  TIED_EMBED_LR=0.035
  WARMDOWN_ITERS=3500
  MUON_MOMENTUM=0.99
  MUON_MOMENTUM_WARMUP_START=0.92
  MUON_MOMENTUM_WARMUP_STEPS=1500
  MUON_WEIGHT_DECAY=0.04
  GRAD_CLIP_NORM=0.3
  EMA_DECAY=0.0
  SWA_START_FRAC=0.0
  SWA_STRIDE=50
  MLP_ACT=relu2
  LEAKY_RELU_SLOPE=0.5
  BIGRAM_VOCAB_SIZE=2048
  BIGRAM_DIM=128
  Z_LOSS_COEF=0.0
  ATTN_TWICE_ALPHA=0.0
  ATTN_TWICE_ALPHA_SLOPE=0.0
  MTP_DEPTH=0
  MTP_LOSS_WEIGHT=0.1
  HYBRID_DELTA_EVERY=0
  SHARED_DEPTH_N=0
  SHARED_DEPTH_GAIN=0.0
  SHARED_DEPTH_EDGE_UNIQUE=0
  ROPE_DIMS=0
  LN_SCALE=0
  HEAD_LR=0.008
)

RUN_NAMES=(
  bigmove_bigram_relu
  bigmove_bigram_leaky
  bigmove_bigram3072_leaky
  bigmove_bigram_fullheads
  bigmove_bigram_fullheads_leaky
  bigmove_bigram3072_fullheads_leaky
  bigmove_bigram_shared_mid
  bigmove_bigram_hybrid
  bigmove_bigram_mtp
  bigmove_bigram_untied
)

RUN_ENVS=(
  ""
  "MLP_ACT=leaky2"
  "MLP_ACT=leaky2 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112"
  "NUM_KV_HEADS=8"
  "NUM_KV_HEADS=8 MLP_ACT=leaky2"
  "NUM_KV_HEADS=8 MLP_ACT=leaky2 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112"
  "MLP_ACT=leaky2 SHARED_DEPTH_N=5 SHARED_DEPTH_EDGE_UNIQUE=2 SHARED_DEPTH_GAIN=0.05"
  "MLP_ACT=leaky2 HYBRID_DELTA_EVERY=4"
  "MLP_ACT=leaky2 MTP_DEPTH=2"
  "MLP_ACT=leaky2 TIE_EMBEDDINGS=0"
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
  echo "=== Big-move run ${name} elapsed=${elapsed}s remaining=${remaining}s ==="

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
