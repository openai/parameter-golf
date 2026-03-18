#!/usr/bin/env bash
set -euo pipefail

profile="${1:-base}"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export RUN_ID="${RUN_ID:-$profile}"

case "$profile" in
  base)
    ;;
  abl_no_lora)
    export LORA_RANK=0
    ;;
  abl_no_mtp)
    export MTP_HEADS=0
    export MTP_WEIGHT=0.0
    ;;
  abl_no_ema)
    export USE_EMA=0
    ;;
  abl_no_rec_scale)
    export USE_RECURRENCE_SCALES=0
    ;;
  abl_relu2)
    export USE_SWIGLU=0
    ;;
  rank8)
    export LORA_RANK=8
    ;;
  rank16)
    export LORA_RANK=16
    ;;
  rank32)
    export LORA_RANK=32
    ;;
  mtp_w005)
    export MTP_WEIGHT=0.05
    ;;
  mtp_w010)
    export MTP_WEIGHT=0.10
    ;;
  mtp_w015)
    export MTP_WEIGHT=0.15
    ;;
  mtp_w025)
    export MTP_WEIGHT=0.25
    ;;
  mtp1)
    export MTP_HEADS=1
    ;;
  mtp2)
    export MTP_HEADS=2
    ;;
  mtp4)
    export MTP_HEADS=4
    ;;
  ema_099_0)
    export EMA_DECAY=0.99
    export EMA_START_STEP=0
    ;;
  ema_0995_50)
    export EMA_DECAY=0.995
    export EMA_START_STEP=50
    ;;
  ema_0998_100)
    export EMA_DECAY=0.998
    export EMA_START_STEP=100
    ;;
  dim768)
    export MODEL_DIM=768
    ;;
  dim832)
    export MODEL_DIM=832
    ;;
  layout_4x3)
    export NUM_UNIQUE_LAYERS=4
    export NUM_RECURRENCE=3
    ;;
  layout_3x4)
    export NUM_UNIQUE_LAYERS=3
    export NUM_RECURRENCE=4
    ;;
  layout_3x5)
    export NUM_UNIQUE_LAYERS=3
    export NUM_RECURRENCE=5
    ;;
  *)
    echo "Unknown profile: $profile" >&2
    exit 1
    ;;
esac

cd "$ROOT_DIR"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" train_gpt.py "$@"
