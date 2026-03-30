#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PROFILE="${PROFILE:-smoke}"             # smoke | full
NPROC="${NPROC:-8}"
SEEDS_STR="${SEEDS:-42}"
TARGET_BPB="${TARGET_BPB:-1.10200000}"
RUN_TAG="${RUN_TAG:-rascal_ab_${PROFILE}_$(date +%Y%m%d_%H%M%S)}"

export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

LOG_DIR="${SCRIPT_DIR}/logs/${RUN_TAG}"
mkdir -p "${LOG_DIR}"
CSV="${LOG_DIR}/summary.csv"

variants=(baseline turbomuon engramlite combo)
declare -A script_by_variant=(
    [baseline]="${SCRIPT_DIR}/train_gpt_baseline.py"
    [turbomuon]="${SCRIPT_DIR}/train_gpt_turbomuon.py"
    [engramlite]="${SCRIPT_DIR}/train_gpt_engramlite.py"
    [combo]="${SCRIPT_DIR}/train_gpt_combo.py"
)

declare -A base_by_seed

echo "profile,seed,variant,val_bpb_exact,delta_vs_baseline,gap_vs_target,logfile" > "${CSV}"

echo "============================================================"
echo "RASCAL A/B MATRIX"
echo "profile=${PROFILE}  nproc=${NPROC}  seeds=${SEEDS_STR}"
echo "target_bpb=${TARGET_BPB}"
echo "log_dir=${LOG_DIR}"
echo "============================================================"

if [[ ! -d "${DATA_PATH}" ]]; then
    echo "ERROR: DATA_PATH does not exist: ${DATA_PATH}"
    exit 1
fi
if [[ ! -f "${TOKENIZER_PATH}" ]]; then
    echo "ERROR: TOKENIZER_PATH does not exist: ${TOKENIZER_PATH}"
    exit 1
fi

extract_metric() {
    local log_file="$1"
    local m
    m=$(grep 'final_sliding_window_exact' "${log_file}" 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || true)
    if [[ -z "${m}" ]]; then
        m=$(grep 'final_sliding_window_s64_exact' "${log_file}" 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || true)
    fi
    if [[ -z "${m}" ]]; then
        m="N/A"
    fi
    printf "%s" "${m}"
}

is_num() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

calc_delta() {
    local cur="$1"
    local base="$2"
    if is_num "${cur}" && is_num "${base}"; then
        awk -v a="${cur}" -v b="${base}" 'BEGIN { printf "%+.8f", (a-b) }'
    else
        printf "N/A"
    fi
}

calc_gap() {
    local cur="$1"
    if is_num "${cur}"; then
        awk -v a="${cur}" -v t="${TARGET_BPB}" 'BEGIN { printf "%+.8f", (a-t) }'
    else
        printf "N/A"
    fi
}

run_one() {
    local seed="$1"
    local variant="$2"
    local script="${script_by_variant[$variant]}"
    local log_file="${LOG_DIR}/${variant}_seed${seed}.log"

    if [[ ! -f "${script}" ]]; then
        echo "ERROR: missing script for ${variant}: ${script}"
        exit 1
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "RUN: seed=${seed}  variant=${variant}  profile=${PROFILE}"
    echo "script=${script}"
    echo "log=${log_file}"
    echo "------------------------------------------------------------"

    if [[ "${PROFILE}" == "smoke" ]]; then
        SEED="${seed}" \
        RUN_ID="ab_${PROFILE}_${variant}_s${seed}" \
        ITERATIONS="${ITERATIONS:-3200}" \
        WARMDOWN_ITERS="${WARMDOWN_ITERS:-800}" \
        TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}" \
        VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-300}" \
        MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}" \
        EVAL_STRIDE="${EVAL_STRIDE:-64}" \
        SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}" \
        POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}" \
        LOADER_MODE="${LOADER_MODE:-coprime}" \
        COPRIME_MAX_LOADED_SHARDS="${COPRIME_MAX_LOADED_SHARDS:-1}" \
        COPRIME_SHARDS_PER_BATCH="${COPRIME_SHARDS_PER_BATCH:-1}" \
        COPRIME_SHARD_HOLD_STEPS="${COPRIME_SHARD_HOLD_STEPS:-64}" \
        XSA_LAST_N="${XSA_LAST_N:-11}" \
        ROPE_DIMS="${ROPE_DIMS:-16}" \
        BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}" \
        TRIGRAM="${TRIGRAM:-0}" \
        torchrun --standalone --nproc_per_node="${NPROC}" "${script}" 2>&1 | tee "${log_file}"
    elif [[ "${PROFILE}" == "full" ]]; then
        SEED="${seed}" \
        RUN_ID="ab_${PROFILE}_${variant}_s${seed}" \
        MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
        EVAL_STRIDE="${EVAL_STRIDE:-64}" \
        SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}" \
        POST_EMA_DIAGNOSTIC="${POST_EMA_DIAGNOSTIC:-0}" \
        LOADER_MODE="${LOADER_MODE:-coprime}" \
        COPRIME_MAX_LOADED_SHARDS="${COPRIME_MAX_LOADED_SHARDS:-1}" \
        COPRIME_SHARDS_PER_BATCH="${COPRIME_SHARDS_PER_BATCH:-1}" \
        COPRIME_SHARD_HOLD_STEPS="${COPRIME_SHARD_HOLD_STEPS:-64}" \
        COMPLEMENT_ALPHA="${COMPLEMENT_ALPHA:-0}" \
        XSA_LAST_N="${XSA_LAST_N:-11}" \
        BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}" \
        ROPE_DIMS="${ROPE_DIMS:-16}" \
        SWA_EVERY="${SWA_EVERY:-50}" \
        MTP_NUM_HEADS="${MTP_NUM_HEADS:-0}" \
        TRIGRAM="${TRIGRAM:-0}" \
        NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-0}" \
        CUBRIC_CADENCE="${CUBRIC_CADENCE:-0}" \
        NGRAM_ENTROPY_SHIFT="${NGRAM_ENTROPY_SHIFT:-0}" \
        torchrun --standalone --nproc_per_node="${NPROC}" "${script}" 2>&1 | tee "${log_file}"
    else
        echo "ERROR: PROFILE must be smoke or full (got: ${PROFILE})"
        exit 1
    fi

    local metric base delta gap
    metric="$(extract_metric "${log_file}")"

    if [[ "${variant}" == "baseline" ]]; then
        base_by_seed["${seed}"]="${metric}"
    fi
    base="${base_by_seed[$seed]:-N/A}"
    delta="$(calc_delta "${metric}" "${base}")"
    gap="$(calc_gap "${metric}")"

    printf "%s,%s,%s,%s,%s,%s,%s\n" \
        "${PROFILE}" "${seed}" "${variant}" "${metric}" "${delta}" "${gap}" "${log_file}" >> "${CSV}"

    printf "RESULT seed=%s variant=%-10s val_bpb=%-12s delta_vs_base=%-12s gap_vs_1.102=%s\n" \
        "${seed}" "${variant}" "${metric}" "${delta}" "${gap}"
}

for seed in ${SEEDS_STR}; do
    for variant in "${variants[@]}"; do
        run_one "${seed}" "${variant}"
    done

    echo ""
    echo "Seed ${seed} summary:"
    if command -v column >/dev/null 2>&1; then
        awk -F, -v s="${seed}" 'NR==1 || $2==s {print}' "${CSV}" \
            | column -t -s ','
    else
        awk -F, -v s="${seed}" 'NR==1 || $2==s {print}' "${CSV}"
    fi
done

echo ""
echo "============================================================"
echo "A/B COMPLETE"
echo "CSV: ${CSV}"
echo "============================================================"

if command -v column >/dev/null 2>&1; then
    column -t -s ',' "${CSV}"
else
    cat "${CSV}"
fi
