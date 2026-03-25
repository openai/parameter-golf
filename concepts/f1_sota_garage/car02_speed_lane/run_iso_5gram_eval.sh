#!/bin/bash
set -euo pipefail

# Isolated-variable A/B for eval-time legal hashed n-gram interpolation.
# Baseline is fixed to current best CAR02 lane (t2 rope24 profile).
# Single variable under test: NGRAM_EVAL_ORDER (0 -> 5).

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
    ROPE_DIMS="${ROPE_DIMS:-24}" \
    NGRAM_EVAL_ORDER="${ngram_order}" \
    NGRAM_EVAL_ALPHA="${NGRAM_EVAL_ALPHA:-0.20}" \
    NGRAM_EVAL_MIN_COUNT="${NGRAM_EVAL_MIN_COUNT:-2}" \
    NGRAM_EVAL_BUCKETS="${NGRAM_EVAL_BUCKETS:-4194304}" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
        "${SCRIPT_DIR}/train_gpt.py" \
        2>&1 | tee "${log_path}"

    echo ""
    echo "--- ${case_id} summary (${log_path}) ---"
    if command -v rg >/dev/null 2>&1; then
        rg -n \
          "model_params:|DIAGNOSTIC post_ema|final_int6_sliding_window_exact|final_int6_sliding_window_ngram|legal_ttt_exact|Total submission size int6\\+zstd|step:[0-9]+/20000 val_loss:" \
          "${log_path}" || true
    else
        grep -nE \
          "model_params:|DIAGNOSTIC post_ema|final_int6_sliding_window_exact|final_int6_sliding_window_ngram|legal_ttt_exact|Total submission size int6\\+zstd|step:[0-9]+/20000 val_loss:" \
          "${log_path}" || true
    fi
    echo "-----------------------------------------"
}

case "${MODE}" in
    control)
        run_case \
          "control_t2_rope24_ngram_off" \
          "0" \
          "Turning n-gram interpolation off should reproduce rope24 baseline metrics within run noise."
        ;;
    v5)
        run_case \
          "var_t2_rope24_ngram5" \
          "5" \
          "Enabling fixed-weight legal 5-gram interpolation improves sliding-window BPB by exploiting local token patterns without label-aware gating."
        ;;
    all)
        run_case \
          "control_t2_rope24_ngram_off" \
          "0" \
          "Turning n-gram interpolation off should reproduce rope24 baseline metrics within run noise."
        run_case \
          "var_t2_rope24_ngram5" \
          "5" \
          "Enabling fixed-weight legal 5-gram interpolation improves sliding-window BPB by exploiting local token patterns without label-aware gating."
        ;;
    *)
        echo "Usage: $0 [all|control|v5]"
        exit 2
        ;;
esac

echo ""
echo "============================================"
echo "  DONE — CAR02 isolated n-gram A/B complete"
echo "============================================"
