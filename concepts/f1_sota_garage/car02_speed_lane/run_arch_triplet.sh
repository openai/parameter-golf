#!/bin/bash
set -euo pipefail

# F1 car02 architecture suite (non-TTT tuning):
# - same legal-LB baseline knobs
# - only architecture knobs change across tests

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
    shift
    local run_id="f1_car02_${case_id}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
    local log_path="logs/${run_id}.log"

    echo "============================================"
    echo "  F1 CAR02 ARCH TEST :: ${case_id}"
    echo "  Seed: ${SEED}"
    echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
    echo "  RUN_ID: ${run_id}"
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
    "$@" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
        "${SCRIPT_DIR}/train_gpt.py" \
        2>&1 | tee "${log_path}"

    echo ""
    echo "--- ${case_id} summary (${log_path}) ---"
    rg -n \
      "model_params:|step:[0-9]+/20000 val_loss:|DIAGNOSTIC post_ema|final_int6_sliding_window_exact|legal_ttt_exact|Total submission size int6\\+zstd" \
      "${log_path}" || true
    echo "-----------------------------------------"
}

case "${MODE}" in
    baseline)
        run_case "baseline_control"
        ;;
    t1)
        # T1: spread value-embedding signal one block earlier.
        run_case "t1_ve_spread" VE_LAYERS=8,9,10
        ;;
    t2)
        # T2: more rotary dimensions for stronger positional anchoring.
        run_case "t2_rope24" ROPE_DIMS=24
        ;;
    t3)
        # T3: speed-lean architecture variant (shallower XSA scope).
        run_case "t3_xsa3_speed" XSA_LAST_N=3
        ;;
    all)
        run_case "baseline_control"
        run_case "t1_ve_spread" VE_LAYERS=8,9,10
        run_case "t2_rope24" ROPE_DIMS=24
        run_case "t3_xsa3_speed" XSA_LAST_N=3
        ;;
    *)
        echo "Usage: $0 [all|baseline|t1|t2|t3]"
        exit 2
        ;;
esac

echo ""
echo "============================================"
echo "  DONE — CAR02 architecture suite complete"
echo "============================================"
