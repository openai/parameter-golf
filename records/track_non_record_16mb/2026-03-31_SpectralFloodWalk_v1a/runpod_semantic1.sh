#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runpod_common.sh"

SEED="${SFW_SEED:-1337}"

SFW_USE_SEMANTIC_MEMORY=true \
SFW_SEMANTIC_LAYERS="${SFW_SEMANTIC1_LAYERS:-${SFW_SEMANTIC_LAYERS:-2}}" \
sfw_run_profile semantic1 "${SEED}" \
  --train-tokens "${SFW_SEMANTIC1_TRAIN_TOKENS:-${SFW_TRAIN_TOKENS:-262144}}" \
  --val-tokens "${SFW_SEMANTIC1_VAL_TOKENS:-${SFW_VAL_TOKENS:-65536}}" \
  --train-steps "${SFW_SEMANTIC1_TRAIN_STEPS:-${SFW_TRAIN_STEPS:-400}}" \
  --eval-batches "${SFW_SEMANTIC1_EVAL_BATCHES:-${SFW_EVAL_BATCHES:-128}}" \
  --batch-size "${SFW_SEMANTIC1_BATCH_SIZE:-${SFW_BATCH_SIZE:-8}}" \
  --seq-len "${SFW_SEMANTIC1_SEQ_LEN:-${SFW_SEQ_LEN:-128}}" \
  --stride "${SFW_SEMANTIC1_STRIDE:-${SFW_STRIDE:-64}}" \
  --report-every "${SFW_SEMANTIC1_REPORT_EVERY:-${SFW_REPORT_EVERY:-10}}" \
  --embed-dim "${SFW_SEMANTIC1_EMBED_DIM:-${SFW_EMBED_DIM:-256}}" \
  --num-layers "${SFW_SEMANTIC1_NUM_LAYERS:-${SFW_NUM_LAYERS:-6}}" \
  --num-heads "${SFW_SEMANTIC1_NUM_HEADS:-${SFW_NUM_HEADS:-8}}" \
  --ff-mult "${SFW_SEMANTIC1_FF_MULT:-${SFW_FF_MULT:-4}}" \
  --pos-buckets "${SFW_SEMANTIC1_POS_BUCKETS:-${SFW_POS_BUCKETS:-256}}" \
  --pk-num-subkeys "${SFW_SEMANTIC1_PK_NUM_SUBKEYS:-${SFW_PK_NUM_SUBKEYS:-64}}" \
  --pk-key-dim "${SFW_SEMANTIC1_PK_KEY_DIM:-${SFW_PK_KEY_DIM:-16}}" \
  --pk-topk-sub "${SFW_SEMANTIC1_PK_TOPK_SUB:-${SFW_PK_TOPK_SUB:-4}}" \
  --pk-topk-final "${SFW_SEMANTIC1_PK_TOPK_FINAL:-${SFW_PK_TOPK_FINAL:-8}}" \
  --pk-code-dim "${SFW_SEMANTIC1_PK_CODE_DIM:-${SFW_PK_CODE_DIM:-64}}" \
  "$@"
