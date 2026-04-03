#!/bin/bash
set -euo pipefail
# Single-H100 signal run for Rascal_Turbo.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
SEED="${SEED:-444}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

command -v "${TORCHRUN_BIN}" >/dev/null 2>&1 || { echo "ERROR: TORCHRUN_BIN not found: ${TORCHRUN_BIN}"; exit 1; }
[[ -f "${TOKENIZER_PATH}" ]] || { echo "ERROR: tokenizer not found: ${TOKENIZER_PATH}"; exit 1; }
[[ -d "${DATA_PATH}" ]] || { echo "ERROR: data path not found: ${DATA_PATH}"; exit 1; }

mkdir -p logs

SEED="${SEED}" \
ITERATIONS="${ITERATIONS:-2000}" \
WARMDOWN_ITERS="${WARMDOWN_ITERS:-0}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}" \
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-1}" \
POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-1}" \
COMPILE_ENABLED="${COMPILE_ENABLED:-0}" \
SKIP_GPTQ=1 \
LOADER_MODE=coprime \
COPRIME_MAX_LOADED_SHARDS=1 \
COPRIME_SHARDS_PER_BATCH=1 \
COPRIME_SHARD_HOLD_STEPS=64 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
NGRAM_EVAL_ORDER=0 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=0 \
MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-4}" \
MUON_POST_NORM="${MUON_POST_NORM:-row_col}" \
"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/rascal_turbo_h100_s${SEED}_$(date +%Y%m%d_%H%M%S).log"
