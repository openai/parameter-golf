#!/bin/bash
set -euo pipefail

# Multi-order backoff (2-7) + entropy-adaptive alpha.
# Baseline is fixed to current best CAR02 lane (t2 rope24 profile).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODE="${1:-all}"

run_case() {
    local case_id="$1"
    local ngram_order="$2"
    local hypothesis="$3"
    local run_id="f1_car02_iso_${case_id}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
    local log_path="logs/${run_id}.log"

    echo "============================================"
    echo "  F1 CAR02 ISO TEST :: ${case_id}"
    echo "  Seed: ${SEED}"
    echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
    echo "  RUN_ID: ${run_id}"
    echo "  Hypothesis: ${hypothesis}"
    echo "============================================"

    env \
    SEED="${SEED}" \
    RUN_ID="${run_id}" \
    F1_CORR_RANK="${F1_CORR_RANK:-0}" \
    DISTILL_ENABLED="${DISTILL_ENABLED:-0}" \
    MLP_ACT="${MLP_ACT:-leaky_relu_sq}" \
    MLP_LEAKY_SLOPE="${MLP_LEAKY_SLOPE:-0.5}" \
    XSA_LAST_N="${XSA_LAST_N:-4}" \
    BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}" \
    TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-0}" \
    TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-0.8}" \
    TTT_EVAL_ENABLED="${TTT_EVAL_ENABLED:-0}" \
    ROPE_DIMS="${ROPE_DIMS:-24}" \
    COMPILE_ENABLED="${COMPILE_ENABLED:-1}" \
    COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-0}" \
    NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-7}" \
    NGRAM_EVAL_MIN_ORDER="${NGRAM_EVAL_MIN_ORDER:-2}" \
    NGRAM_EVAL_ADAPTIVE="${NGRAM_EVAL_ADAPTIVE:-1}" \
    NGRAM_EVAL_ALPHA="${NGRAM_EVAL_ALPHA:-0.30}" \
    NGRAM_EVAL_ALPHA_MIN="${NGRAM_EVAL_ALPHA_MIN:-0.05}" \
    NGRAM_EVAL_ALPHA_MAX="${NGRAM_EVAL_ALPHA_MAX:-0.60}" \
    NGRAM_EVAL_ENTROPY_CENTER="${NGRAM_EVAL_ENTROPY_CENTER:-4.0}" \
    NGRAM_EVAL_ENTROPY_SCALE="${NGRAM_EVAL_ENTROPY_SCALE:-2.0}" \
    NGRAM_EVAL_MIN_COUNT="${NGRAM_EVAL_MIN_COUNT:-2}" \
    NGRAM_EVAL_BUCKETS="${NGRAM_EVAL_BUCKETS:-4194304}" \
    NGRAM_EVAL_MAX_SECONDS="${NGRAM_EVAL_MAX_SECONDS:-300}" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
        "${SCRIPT_DIR}/train_gpt.py" \
        2>&1 | tee "${log_path}"

    echo ""
    echo "--- ${case_id} summary (${log_path}) ---"
    if command -v rg >/dev/null 2>&1; then
        rg -n \
          "model_params:|DIAGNOSTIC post_ema|final_int6_sliding_window_exact|final_int6_sliding_window_ngram|ngram_eval:cutoff|legal_ttt_exact|Total submission size int6\\+zstd|step:[0-9]+/20000 val_loss:" \
          "${log_path}" || true
    else
        grep -nE \
          "model_params:|DIAGNOSTIC post_ema|final_int6_sliding_window_exact|final_int6_sliding_window_ngram|ngram_eval:cutoff|legal_ttt_exact|Total submission size int6\\+zstd|step:[0-9]+/20000 val_loss:" \
          "${log_path}" || true
    fi
    echo "-----------------------------------------"
}

run_case \
  "backoff_7gram_adaptive" \
  "7" \
  "Multi-order backoff (2-7) + entropy-adaptive alpha. Legal: alpha from model entropy only."

echo ""
echo "============================================"
echo "  DONE — CAR02 isolated n-gram A/B complete"
echo "============================================"
