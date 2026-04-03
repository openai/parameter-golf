#!/usr/bin/env bash
# Context-Only SLOT Legal ablation — 1-GPU proxy, seed=444, 1200 steps
# Usage: bash gate.sh
# One variable: SLOT_ENABLED (0=baseline, 1=legal context-only SLOT)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_GPT="${SCRIPT_DIR}/train_gpt.py"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

NPROC=1
SEED="${SEED:-444}"
SLOT_MAX_WINDOWS=512
TORCHRUN="${TORCHRUN:-$(find /venv /usr /opt -name torchrun -type f 2>/dev/null | head -1)}"

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
PYTHONPATH_EXTRA=""
if [[ -d "${REPO_ROOT}/flash-attention/hopper" ]]; then
    PYTHONPATH_EXTRA="${REPO_ROOT}/flash-attention/hopper:"
fi

echo "=== QK_Gain_SLOT_Legal gate  seed=${SEED}  nproc=${NPROC}  windows=${SLOT_MAX_WINDOWS} ==="
echo "Torchrun: ${TORCHRUN}"
echo "Data:     ${DATA_PATH}"

run_arm() {
    local name="$1"
    local slot="$2"
    local log="${LOG_DIR}/${name}_s${SEED}.log"
    echo ""
    echo "=============================="
    echo "ARM: ${name}  SLOT_ENABLED=${slot}"
    echo "log: ${log}"
    echo "=============================="
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
    LATE_QAT_THRESHOLD=0.15 \
    POST_EMA_DIAGNOSTIC=1 \
    EVAL_STRIDE=64 \
    SKIP_GPTQ=1 \
    MAX_WALLCLOCK_SECONDS=0 \
    ITERATIONS=1200 \
    SLOT_ENABLED="${slot}" \
    SLOT_MAX_WINDOWS="${SLOT_MAX_WINDOWS}" \
    SEED="${SEED}" \
    DATA_PATH="${DATA_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    PYTHONPATH="${PYTHONPATH_EXTRA}${PYTHONPATH:-}" \
    "${TORCHRUN}" --standalone "--nproc_per_node=${NPROC}" "${TRAIN_GPT}" \
        2>&1 | tee "${log}"
    echo "[${name}] done"
}

run_arm baseline    0
run_arm slot_legal  1

echo ""
echo "=== RESULTS ==="
echo "baseline:"
grep "final_sliding_window" "${LOG_DIR}/baseline_s${SEED}.log"   | tail -1
echo "slot_legal:"
grep "final_sliding_window" "${LOG_DIR}/slot_legal_s${SEED}.log" | tail -1
