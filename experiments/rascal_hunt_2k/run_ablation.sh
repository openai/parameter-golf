#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${REPO_ROOT}"

CASE="${1:-ctrl}"
SEED="${SEED:-444}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/experiments/rascal_hunt_2k/logs"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="/usr/local/lib/python3.12/dist-packages:${REPO_ROOT}/flash-attention/hopper${PYTHONPATH:+:${PYTHONPATH}}"

export DATA_PATH="${REPO_ROOT}/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model"
export RUN_ID="hunt2k_${CASE}_s${SEED}_${TS}"
export SEED

export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=200
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048

export COMPILE_ENABLED=1
export COMPILE_FULLGRAPH=1
export LOADER_MODE=coprime
export COPRIME_MAX_LOADED_SHARDS=4
export COPRIME_SHARDS_PER_BATCH=1
export COPRIME_SHARD_HOLD_STEPS=64

export SKIP_GPTQ=1
export NGRAM_EVAL_ORDER=0
export POST_EMA_DIAGNOSTIC=1

case "${CASE}" in
  ctrl)
    ;;
  gptq)
    export SKIP_GPTQ=0
    ;;
  ngram5)
    export NGRAM_EVAL_ORDER=5
    ;;
  ngram7)
    export NGRAM_EVAL_ORDER=7
    ;;
  gptq_ngram7)
    export SKIP_GPTQ=0
    export NGRAM_EVAL_ORDER=7
    ;;
  qkgain4)
    export QK_GAIN_INIT=4
    ;;
  mlp35)
    export MLP_MULT=3.5
    ;;
  bigram3072)
    export BIGRAM_VOCAB_SIZE=3072
    ;;
  combo_qk_gptq)
    export SKIP_GPTQ=0
    export QK_GAIN_INIT=4
    ;;
  *)
    echo "Unknown case: ${CASE}" >&2
    echo "Cases: ctrl gptq ngram5 ngram7 gptq_ngram7 qkgain4 mlp35 bigram3072 combo_qk_gptq" >&2
    exit 1
    ;;
esac

echo "CASE=${CASE} SEED=${SEED} RUN_ID=${RUN_ID}"
echo "LOG=${LOG_DIR}/${RUN_ID}.log"

python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${REPO_ROOT}/experiments/rascal_hunt_2k/train_gpt.py" \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
