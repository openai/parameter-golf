#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runpod_common.sh"

SEED="${SFW_SEED:-1337}"

SFW_USE_SEMANTIC_MEMORY=true \
SFW_SEMANTIC_LAYERS="${SFW_FULL_SEMANTIC_LAYERS:-${SFW_SEMANTIC_LAYERS:-2,4}}" \
sfw_run_profile full "${SEED}" \
  --train-tokens "${SFW_FULL_TRAIN_TOKENS:-${SFW_TRAIN_TOKENS:-262144}}" \
  --val-tokens "${SFW_FULL_VAL_TOKENS:-${SFW_VAL_TOKENS:-65536}}" \
  --train-steps "${SFW_FULL_TRAIN_STEPS:-${SFW_TRAIN_STEPS:-400}}" \
  --eval-batches "${SFW_FULL_EVAL_BATCHES:-${SFW_EVAL_BATCHES:-128}}" \
  --batch-size "${SFW_FULL_BATCH_SIZE:-${SFW_BATCH_SIZE:-8}}" \
  --seq-len "${SFW_FULL_SEQ_LEN:-${SFW_SEQ_LEN:-128}}" \
  --stride "${SFW_FULL_STRIDE:-${SFW_STRIDE:-64}}" \
  --report-every "${SFW_FULL_REPORT_EVERY:-${SFW_REPORT_EVERY:-10}}" \
  --embed-dim "${SFW_FULL_EMBED_DIM:-${SFW_EMBED_DIM:-256}}" \
  --num-layers "${SFW_FULL_NUM_LAYERS:-${SFW_NUM_LAYERS:-6}}" \
  --num-heads "${SFW_FULL_NUM_HEADS:-${SFW_NUM_HEADS:-8}}" \
  --ff-mult "${SFW_FULL_FF_MULT:-${SFW_FF_MULT:-4}}" \
  --pos-buckets "${SFW_FULL_POS_BUCKETS:-${SFW_POS_BUCKETS:-256}}" \
  --pk-num-subkeys "${SFW_FULL_PK_NUM_SUBKEYS:-${SFW_PK_NUM_SUBKEYS:-64}}" \
  --pk-key-dim "${SFW_FULL_PK_KEY_DIM:-${SFW_PK_KEY_DIM:-16}}" \
  --pk-topk-sub "${SFW_FULL_PK_TOPK_SUB:-${SFW_PK_TOPK_SUB:-4}}" \
  --pk-topk-final "${SFW_FULL_PK_TOPK_FINAL:-${SFW_PK_TOPK_FINAL:-8}}" \
  --pk-code-dim "${SFW_FULL_PK_CODE_DIM:-${SFW_PK_CODE_DIM:-64}}" \
  "$@"
