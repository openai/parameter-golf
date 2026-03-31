#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

env \
  SEED="${SEED:-1337}" \
  MLP_ACT=leaky_relu_sq \
  MLP_LEAKY_SLOPE=0.5 \
  XSA_LAST_N=4 \
  BIGRAM_VOCAB_SIZE=1536 \
  ROPE_DIMS=24 \
  TTT_EVAL_ENABLED=0 \
  COMPILE_ENABLED=1 \
  COMPILE_FULLGRAPH=1 \
  NGRAM_EVAL_ORDER=7 \
  NGRAM_EVAL_ADAPTIVE=1 \
  NGRAM_EVAL_ALPHA=0.30 \
  NGRAM_EVAL_MIN_COUNT=2 \
  NGRAM_EVAL_BUCKETS=4194304 \
  NGRAM_EVAL_ALPHA_MIN=0.05 \
  NGRAM_EVAL_ALPHA_MAX=0.60 \
  CUBRIC_CADENCE=4 \
  CUBRIC_COUNT_DECAY=0.02 \
  CUBRIC_BOOST_CONFIDENT=1 \
  CUBRIC_PRUNE_NOISY=1 \
  CUBRIC_REWEIGHT_ORDERS=1 \
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
    "${SCRIPT_DIR}/train_gpt_cadence4.py"
