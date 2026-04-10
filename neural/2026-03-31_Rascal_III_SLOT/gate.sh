#!/usr/bin/env bash
# Rascal_III_SLOT gate — 1-GPU, 2000 steps, paired A/B (baseline vs slot_legal)
# Usage: bash gate.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SEED="${SEED:-444}"
NPROC=1
TORCHRUN="${TORCHRUN:-$(find /venv /usr /opt -name torchrun -type f 2>/dev/null | head -1)}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
PYTHONPATH_EXTRA=""
if [[ -d "${REPO_ROOT}/flash-attention/hopper" ]]; then
    PYTHONPATH_EXTRA="${REPO_ROOT}/flash-attention/hopper:"
fi

echo "=== Rascal_III_SLOT gate  seed=${SEED}  nproc=${NPROC} ==="
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
    ITERATIONS=2000 \
    SLOT_ENABLED="${slot}" \
    SLOT_STEPS=8 \
    SLOT_LR=0.005 \
    SLOT_MAX_WINDOWS=512 \
    SEED="${SEED}" \
    DATA_PATH="${DATA_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    PYTHONPATH="${PYTHONPATH_EXTRA}${PYTHONPATH:-}" \
    "${TORCHRUN}" --standalone "--nproc_per_node=${NPROC}" "${SCRIPT_DIR}/train_gpt_slot.py" \
        2>&1 | tee "${log}"
    echo "[${name}] done"
}

run_arm baseline  0
run_arm slot_legal 1

echo ""
echo "=== RESULTS ==="
echo "baseline:"
grep "final_sliding_window_exact" "${LOG_DIR}/baseline_s${SEED}.log"  | tail -1
echo "slot_legal:"
grep "final_sliding_window" "${LOG_DIR}/slot_legal_s${SEED}.log" | tail -1
echo ""
echo "Gate passes if slot_legal delta vs baseline < -0.003"
