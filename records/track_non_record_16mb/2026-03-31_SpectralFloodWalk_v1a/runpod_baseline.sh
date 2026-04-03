#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runpod_common.sh"

SEED="${SFW_SEED:-1337}"

SFW_USE_SEMANTIC_MEMORY=false \
SFW_SEMANTIC_LAYERS="${SFW_BASELINE_SEMANTIC_LAYERS:-${SFW_SEMANTIC_LAYERS:-}}" \
sfw_run_profile baseline "${SEED}" \
  --train-tokens "${SFW_BASELINE_TRAIN_TOKENS:-${SFW_TRAIN_TOKENS:-262144}}" \
  --val-tokens "${SFW_BASELINE_VAL_TOKENS:-${SFW_VAL_TOKENS:-65536}}" \
  --train-steps "${SFW_BASELINE_TRAIN_STEPS:-${SFW_TRAIN_STEPS:-400}}" \
  --eval-batches "${SFW_BASELINE_EVAL_BATCHES:-${SFW_EVAL_BATCHES:-128}}" \
  --batch-size "${SFW_BASELINE_BATCH_SIZE:-${SFW_BATCH_SIZE:-8}}" \
  --seq-len "${SFW_BASELINE_SEQ_LEN:-${SFW_SEQ_LEN:-128}}" \
  --stride "${SFW_BASELINE_STRIDE:-${SFW_STRIDE:-64}}" \
  --report-every "${SFW_BASELINE_REPORT_EVERY:-${SFW_REPORT_EVERY:-10}}" \
  --embed-dim "${SFW_BASELINE_EMBED_DIM:-${SFW_EMBED_DIM:-256}}" \
  --num-layers "${SFW_BASELINE_NUM_LAYERS:-${SFW_NUM_LAYERS:-6}}" \
  --num-heads "${SFW_BASELINE_NUM_HEADS:-${SFW_NUM_HEADS:-8}}" \
  --ff-mult "${SFW_BASELINE_FF_MULT:-${SFW_FF_MULT:-4}}" \
  --pos-buckets "${SFW_BASELINE_POS_BUCKETS:-${SFW_POS_BUCKETS:-256}}" \
  "$@"
