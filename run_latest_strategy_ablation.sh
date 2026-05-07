#!/usr/bin/env bash
set -euo pipefail

cd /v/ai/nobackup/xma/openai/parameter-golf

DEVICE="${DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
ITERATIONS="${ITERATIONS:-1500}"
WARMDOWN_ITERS="${WARMDOWN_ITERS:-1500}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
EVAL_STRIDE="${EVAL_STRIDE:-0}"
TTT_ENABLED="${TTT_ENABLED:-0}"
SEEDS="${SEEDS:-1337}"
ABLATION_SET="${ABLATION_SET:-quick}"

mkdir -p logs

run_experiment() {
    local run_id="$1"
    local script="$2"
    local seed="$3"
    shift 3

    local log_path="logs/${run_id}.txt"
    if [[ -f "${log_path}" ]] && rg -q "^(final_int6_roundtrip_exact|legal_ttt_exact) val_loss:" "${log_path}"; then
        echo "================================================================================"
        echo "Skipping ${run_id}"
        echo "Reason: found completion marker in ${log_path}"
        echo "================================================================================"
        return 0
    fi

    echo "================================================================================"
    echo "Starting ${run_id}"
    echo "Script: ${script}"
    echo "Seed: ${seed}"
    echo "Log: ${log_path}"
    echo "================================================================================"

    (
        export CUDA_VISIBLE_DEVICES="${DEVICE}"
        export RUN_ID="${run_id}"
        export SEED="${seed}"
        export DATA_PATH="${DATA_PATH}"
        export TOKENIZER_PATH="${TOKENIZER_PATH}"
        export VOCAB_SIZE="${VOCAB_SIZE}"
        export ITERATIONS="${ITERATIONS}"
        export WARMDOWN_ITERS="${WARMDOWN_ITERS}"
        export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}"
        export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY}"
        export VAL_LOSS_EVERY="${VAL_LOSS_EVERY}"
        export EVAL_STRIDE="${EVAL_STRIDE}"
        export TTT_ENABLED="${TTT_ENABLED}"

        for arg in "$@"; do
            export "$arg"
        done

        "${PYTHON_BIN}" -u "${script}"
    ) 2>&1 | tee "${log_path}"

    local status=${PIPESTATUS[0]}
    echo "Finished ${run_id} with status ${status}"
    return "${status}"
}

run_quick_set() {
    local seed="$1"

    # run_experiment "ablation_top_disabled_seed${seed}_iter${ITERATIONS}" \
    #     "train_gpt_style_top.py" "${seed}" \
    #     STYLE_ENABLED="0"

    # run_experiment "ablation_top_film_w128_d0_seed${seed}_iter${ITERATIONS}" \
    #     "train_gpt_style_top.py" "${seed}" \
    #     STYLE_ENABLED="1" \
    #     STYLE_WINDOW="128" \
    #     STYLE_MODE="film" \
    #     STYLE_DIM="0"

    run_experiment "ablation_v2_dual_gate_seed${seed}_iter${ITERATIONS}" \
        "train_gpt_style_v2.py" "${seed}" \
        STYLE_ENABLED="1" \
        STYLE_WINDOW="128" \
        STYLE_MODE="film" \
        STYLE_DIM="0" \
        STYLE_SUMMARY_MODE="dual" \
        STYLE_SHORT_WINDOW="16" \
        STYLE_USE_NORMED_INPUT="1" \
        STYLE_SAMPLE_GATE="1" \
        STYLE_BIAS_TANH="0"

    run_experiment "ablation_v2_dual_nogate_seed${seed}_iter${ITERATIONS}" \
        "train_gpt_style_v2.py" "${seed}" \
        STYLE_ENABLED="1" \
        STYLE_WINDOW="128" \
        STYLE_MODE="film" \
        STYLE_DIM="0" \
        STYLE_SUMMARY_MODE="dual" \
        STYLE_SHORT_WINDOW="16" \
        STYLE_USE_NORMED_INPUT="1" \
        STYLE_SAMPLE_GATE="0" \
        STYLE_BIAS_TANH="0"

    run_experiment "ablation_freq_mid_seed${seed}_iter${ITERATIONS}" \
        "train_gpt_style_freq.py" "${seed}" \
        STYLE_ENABLED="1" \
        STYLE_WINDOW="128" \
        STYLE_MODE="film" \
        STYLE_DIM="0" \
        FREQ_LOSS_ENABLED="1" \
        FREQ_LOSS_MID_LOW="0.2" \
        FREQ_LOSS_MID_HIGH="0.8" \
        FREQ_LOSS_WEIGHT="1.1"

    run_experiment "ablation_localmem_seed${seed}_iter${ITERATIONS}" \
        "train_gpt_style_localmem.py" "${seed}" \
        STYLE_ENABLED="1" \
        STYLE_WINDOW="128" \
        STYLE_MODE="film" \
        STYLE_DIM="0" \
        LOCAL_ATTN_WINDOW="128" \
        LOCAL_ATTN_LAYERS="0,1,2,3" \
        DEPTH_MEMORY_ENABLED="1" \
        DEPTH_MEMORY_DIM="64" \
        DEPTH_MEMORY_TOKENS="128" \
        DEPTH_MEMORY_LAYERS="all" \
        DEPTH_MEMORY_GATE_INIT="-3.0"
}

run_full_set() {
    local seed="$1"

    run_quick_set "${seed}"

    run_experiment "ablation_v2_mean_gate_seed${seed}_iter${ITERATIONS}" \
        "train_gpt_style_v2.py" "${seed}" \
        STYLE_ENABLED="1" \
        STYLE_WINDOW="128" \
        STYLE_MODE="film" \
        STYLE_DIM="0" \
        STYLE_SUMMARY_MODE="mean" \
        STYLE_SHORT_WINDOW="16" \
        STYLE_USE_NORMED_INPUT="1" \
        STYLE_SAMPLE_GATE="1" \
        STYLE_BIAS_TANH="0"

    run_experiment "ablation_v2_dual_gate_tanhbias_seed${seed}_iter${ITERATIONS}" \
        "train_gpt_style_v2.py" "${seed}" \
        STYLE_ENABLED="1" \
        STYLE_WINDOW="128" \
        STYLE_MODE="film" \
        STYLE_DIM="0" \
        STYLE_SUMMARY_MODE="dual" \
        STYLE_SHORT_WINDOW="16" \
        STYLE_USE_NORMED_INPUT="1" \
        STYLE_SAMPLE_GATE="1" \
        STYLE_BIAS_TANH="1"

    run_experiment "ablation_local_only_seed${seed}_iter${ITERATIONS}" \
        "train_gpt_style_localmem.py" "${seed}" \
        STYLE_ENABLED="1" \
        STYLE_WINDOW="128" \
        STYLE_MODE="film" \
        STYLE_DIM="0" \
        LOCAL_ATTN_WINDOW="128" \
        LOCAL_ATTN_LAYERS="0,1,2,3" \
        DEPTH_MEMORY_ENABLED="0"

    run_experiment "ablation_depth_memory_only_seed${seed}_iter${ITERATIONS}" \
        "train_gpt_style_localmem.py" "${seed}" \
        STYLE_ENABLED="1" \
        STYLE_WINDOW="128" \
        STYLE_MODE="film" \
        STYLE_DIM="0" \
        LOCAL_ATTN_WINDOW="0" \
        DEPTH_MEMORY_ENABLED="1" \
        DEPTH_MEMORY_DIM="64" \
        DEPTH_MEMORY_TOKENS="128" \
        DEPTH_MEMORY_LAYERS="all" \
        DEPTH_MEMORY_GATE_INIT="-3.0"
}

for seed in ${SEEDS}; do
    case "${ABLATION_SET}" in
        quick)
            run_quick_set "${seed}"
            ;;
        full)
            run_full_set "${seed}"
            ;;
        *)
            echo "Unsupported ABLATION_SET=${ABLATION_SET}. Use quick or full." >&2
            exit 1
            ;;
    esac
done
