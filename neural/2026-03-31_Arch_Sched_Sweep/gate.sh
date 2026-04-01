#!/usr/bin/env bash
# Arch+Sched Sweep — smoke test then full 6-case sweep
# Usage: bash gate.sh [--dry-run] [--cases case1 case2 ...]
# Requires: 4×H100 pod, env vars DATA_PATH and TOKENIZER_PATH set, or defaults will be used.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_GPT="${SCRIPT_DIR}/train_gpt.py"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

NPROC="${NPROC:-4}"
SEED="${SEED:-444}"
TORCHRUN="${TORCHRUN:-torchrun}"

DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=1
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# Expected step time: 91ms × (8/NPROC); threshold = 2.5×
THRESHOLD=$(( 91 * 8 / NPROC * 5 / 2 ))
echo "=== Arch+Sched Sweep gate.sh ==="
echo "NPROC=${NPROC}  SEED=${SEED}  THRESHOLD=${THRESHOLD}ms"
echo "Repo root: ${REPO_ROOT}"

# ── Smoke test (20 steps) ──────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 0 ]]; then
    echo ""
    echo "--- SMOKE TEST (20 steps) ---"
    SMOKE_LOG="${LOG_DIR}/smoke_s${SEED}.log"

    DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
    TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
    PYTHONPATH_EXTRA=""
    if [[ -d "${REPO_ROOT}/flash-attention/hopper" ]]; then
        PYTHONPATH_EXTRA="${REPO_ROOT}/flash-attention/hopper"
    fi

    PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}" \
    MAX_WALLCLOCK_SECONDS=20 \
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
    LATE_QAT_THRESHOLD=0.15 \
    POST_EMA_DIAGNOSTIC=1 \
    EVAL_STRIDE=64 \
    SKIP_FINAL_EVAL=1 \
    DATA_PATH="${DATA_PATH}" \
    TOKENIZER_PATH="${TOKENIZER_PATH}" \
    SEED="${SEED}" \
    "${TORCHRUN}" --standalone "--nproc_per_node=${NPROC}" "${TRAIN_GPT}" \
        2>&1 | tee "${SMOKE_LOG}" || true

    # Extract step_avg from step:500 line, or fallback to any step line
    STEP_AVG=$(grep -oP 'step_avg:\K[0-9]+' "${SMOKE_LOG}" | tail -1 || true)
    if [[ -z "${STEP_AVG}" ]]; then
        # try any step timing pattern
        STEP_AVG=$(grep -oP '\b\d+ms\b' "${SMOKE_LOG}" | grep -oP '\d+' | tail -1 || echo "0")
    fi

    echo ""
    echo "Smoke: step_avg=${STEP_AVG}ms  threshold=${THRESHOLD}ms"
    if [[ "${STEP_AVG}" -gt "${THRESHOLD}" ]]; then
        echo "SMOKE TEST FAILED: ${STEP_AVG}ms > ${THRESHOLD}ms — pod too slow, aborting"
        exit 1
    fi
    echo "SMOKE TEST PASSED"
fi

# ── Full sweep ─────────────────────────────────────────────────────────────
echo ""
echo "--- LAUNCHING SWEEP ---"
SWEEP_CMD=("python3" "${SCRIPT_DIR}/run_sweep.py"
           "--seed" "${SEED}"
           "--nproc" "${NPROC}"
           "--torchrun" "${TORCHRUN}")
if [[ "${DRY_RUN}" -eq 1 ]]; then
    SWEEP_CMD+=("--dry-run")
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    SWEEP_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running: ${SWEEP_CMD[*]}"
"${SWEEP_CMD[@]}"
