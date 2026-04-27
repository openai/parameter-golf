#!/usr/bin/env bash
set -euo pipefail

cd /v/ai/nobackup/xma/openai/parameter-golf

DEVICE="${DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SCRIPT="${SCRIPT:-train_gpt_style_top.py}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
ITERATIONS="${ITERATIONS:-3500}"
WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
EVAL_STRIDE="${EVAL_STRIDE:-0}"
TTT_ENABLED="${TTT_ENABLED:-0}"
STYLE_GATE_ENABLED="${STYLE_GATE_ENABLED:-1}"
STYLE_GATE_INIT="${STYLE_GATE_INIT:--4.0}"

mkdir -p logs

run_experiment() {
    local base_run_id="$1"
    local seed="$2"
    shift 2

    local run_id="${base_run_id}_seed${seed}_iter${ITERATIONS}"
    local log_path="logs/${run_id}.txt"

    echo "================================================================================"
    echo "Starting ${run_id}"
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
        export STYLE_GATE_ENABLED="${STYLE_GATE_ENABLED}"
        export STYLE_GATE_INIT="${STYLE_GATE_INIT}"

        for arg in "$@"; do
            export "$arg"
        done

        "${PYTHON_BIN}" -u "${SCRIPT}"
    ) 2>&1 | tee "${log_path}"

    local status=${PIPESTATUS[0]}
    echo "Finished ${run_id} with status ${status}"
    return "${status}"
}

for seed in 42 2025 1337; do
    run_experiment "champion_style_film_w128_d0" "${seed}" \
        STYLE_ENABLED="1" \
        STYLE_WINDOW="128" \
        STYLE_MODE="film" \
        STYLE_DIM="0"

    run_experiment "champion_style_disabled" "${seed}" \
        STYLE_ENABLED="0"
done
