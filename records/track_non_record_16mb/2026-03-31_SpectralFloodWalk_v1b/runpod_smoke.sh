#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/runpod_common.sh"

SEED="${SFW_SEED:-1337}"

SFW_USE_SEMANTIC_MEMORY="${SFW_SMOKE_USE_SEMANTIC_MEMORY:-${SFW_USE_SEMANTIC_MEMORY}}" \
SFW_SEMANTIC_LAYERS="${SFW_SMOKE_SEMANTIC_LAYERS:-${SFW_SEMANTIC_LAYERS:-}}" \
SFW_EVAL_MODES="${SFW_SMOKE_EVAL_MODES:-${SFW_EVAL_MODES:-controller,raw,refined}}" \
sfw_run_profile smoke "${SEED}" \
  --train-tokens "${SFW_SMOKE_TRAIN_TOKENS:-${SFW_TRAIN_TOKENS:-65536}}" \
  --val-tokens "${SFW_SMOKE_VAL_TOKENS:-${SFW_VAL_TOKENS:-16384}}" \
  --train-steps "${SFW_SMOKE_TRAIN_STEPS:-${SFW_TRAIN_STEPS:-32}}" \
  --eval-batches "${SFW_SMOKE_EVAL_BATCHES:-${SFW_EVAL_BATCHES:-64}}" \
  --batch-size "${SFW_SMOKE_BATCH_SIZE:-${SFW_BATCH_SIZE:-8}}" \
  --seq-len "${SFW_SMOKE_SEQ_LEN:-${SFW_SEQ_LEN:-128}}" \
  --stride "${SFW_SMOKE_STRIDE:-${SFW_STRIDE:-64}}" \
  --report-every "${SFW_SMOKE_REPORT_EVERY:-${SFW_REPORT_EVERY:-4}}" \
  --embed-dim "${SFW_SMOKE_EMBED_DIM:-${SFW_EMBED_DIM:-192}}" \
  --num-layers "${SFW_SMOKE_NUM_LAYERS:-${SFW_NUM_LAYERS:-4}}" \
  --num-heads "${SFW_SMOKE_NUM_HEADS:-${SFW_NUM_HEADS:-6}}" \
  --ff-mult "${SFW_SMOKE_FF_MULT:-${SFW_FF_MULT:-3}}" \
  --pos-buckets "${SFW_SMOKE_POS_BUCKETS:-${SFW_POS_BUCKETS:-256}}" \
  --pk-num-subkeys "${SFW_SMOKE_PK_NUM_SUBKEYS:-${SFW_PK_NUM_SUBKEYS:-32}}" \
  --pk-key-dim "${SFW_SMOKE_PK_KEY_DIM:-${SFW_PK_KEY_DIM:-12}}" \
  --pk-topk-sub "${SFW_SMOKE_PK_TOPK_SUB:-${SFW_PK_TOPK_SUB:-4}}" \
  --pk-topk-final "${SFW_SMOKE_PK_TOPK_FINAL:-${SFW_PK_TOPK_FINAL:-8}}" \
  --pk-code-dim "${SFW_SMOKE_PK_CODE_DIM:-${SFW_PK_CODE_DIM:-32}}" \
  "$@"
