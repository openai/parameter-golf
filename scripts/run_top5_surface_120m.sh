#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL_BUDGET_MINUTES="${TOTAL_BUDGET_MINUTES:-120}"
MIN_SECONDS_LEFT_TO_START="${MIN_SECONDS_LEFT_TO_START:-600}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"

# Common surface distilled from current top-5 winners, with EMA left configurable.
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
  MLP_MULT=3
  TIE_EMBEDDINGS=1
  MATRIX_LR=0.025
  SCALAR_LR=0.025
  TIED_EMBED_LR=0.035
  MUON_MOMENTUM=0.99
  MUON_MOMENTUM_WARMUP_START=0.92
  MUON_MOMENTUM_WARMUP_STEPS=1500
  MUON_WEIGHT_DECAY=0.04
  WARMDOWN_ITERS=3500
  GRAD_CLIP_NORM=0.3
  OVERTONE_INIT_POWER=0.5
)

RUN_NAMES=(
  top5surf_relu_ema
  top5surf_relu_noema
  top5surf_leaky_noema
  top5surf_fullheads_noema
  top5surf_layerwise_noema
)

RUN_ENVS=(
  "EMA_DECAY=0.997 MLP_ACT=relu2"
  "EMA_DECAY=0.0 MLP_ACT=relu2"
  "EMA_DECAY=0.0 MLP_ACT=leaky2 LEAKY_RELU_SLOPE=0.5"
  "EMA_DECAY=0.0 MLP_ACT=relu2 NUM_KV_HEADS=8"
  "EMA_DECAY=0.0 MLP_ACT=relu2 ATTN_TWICE_ALPHA=0.05 ATTN_TWICE_ALPHA_SLOPE=0.5"
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
  echo "=== Top5 surface run ${name} elapsed=${elapsed}s remaining=${remaining}s ==="

  env \
    "${BASE_ENV[@]}" \
    RUN_ID="$name" \
    SKIP_FINAL_EVAL="$SKIP_FINAL_EVAL" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    ${extra} \
    bash scripts/run_remote_experiment.sh

  tail -n 12 "logs/${name}.txt"
done
