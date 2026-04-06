#!/usr/bin/env bash
set -euo pipefail

# Repro runner for post-1.1147 ablations.
# Usage:
#   chmod +x run_next_step_experiments.sh
#   ./run_next_step_experiments.sh <experiment_key>
#
# Example:
#   ./run_next_step_experiments.sh delta_full_stack
#
# Notes:
# - All runs stay on the same 10min/16MB track defaults unless overridden here.
# - Use a single seed first for fast triage, then promote winners to 3-seed runs.

if [[ $# -lt 1 ]]; then
  echo "missing experiment key"
  exit 1
fi

EXPERIMENT_KEY="$1"
SEED="${SEED:-314}"
NPROC="${NPROC:-8}"
COMMON_ENV=(
  "TARGET_MB=15.9"
  "WARMDOWN_ITERS=4000"
  "SEED=${SEED}"
)

run_case() {
  local name="$1"
  shift
  echo "=== running ${name} ==="
  env "${COMMON_ENV[@]}" "$@" torchrun --standalone --nproc_per_node="${NPROC}" train_gpt.py
}

case "${EXPERIMENT_KEY}" in
  # 1) Confirm contribution split: Full GPTQ vs GPTQ-lite, and XSA scope.
  delta_full_stack)
    run_case "delta_full_stack" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
    ;;
  delta_no_full_gptq)
    run_case "delta_no_full_gptq" \
      GPTQ_METHOD=lite GPTQ_CALIB_MODE=none XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
    ;;
  delta_xsa_last4)
    run_case "delta_xsa_last4" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen XSA_LAST_N=4 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
    ;;

  # 2) AR calibration distribution ensembles (temp + top-p sweep).
  calib_ensemble_lowdiv)
    run_case "calib_ensemble_lowdiv" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen \
      GPTQ_CALIB_NUM_SEQS=64 GPTQ_CALIB_TEMP_LIST=0.75,0.8 GPTQ_CALIB_TOP_P_LIST=0.0,0.9 \
      XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
    ;;
  calib_ensemble_widediv)
    run_case "calib_ensemble_widediv" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen \
      GPTQ_CALIB_NUM_SEQS=64 GPTQ_CALIB_TEMP_LIST=0.7,0.8,0.9 GPTQ_CALIB_TOP_P_LIST=0.0,0.92 \
      XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
    ;;

  # 3) Layer-adaptive GPTQ settings.
  gptq_layer_adapt_v1)
    run_case "gptq_layer_adapt_v1" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen \
      GPTQ_DAMP_ATTN=0.008 GPTQ_DAMP_MLP=0.014 GPTQ_BLOCK_SIZE_ATTN=96 GPTQ_BLOCK_SIZE_MLP=160 \
      XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
    ;;
  gptq_layer_adapt_v2)
    run_case "gptq_layer_adapt_v2" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen \
      GPTQ_DAMP_ATTN=0.006 GPTQ_DAMP_MLP=0.018 GPTQ_BLOCK_SIZE_ATTN=128 GPTQ_BLOCK_SIZE_MLP=192 \
      XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
    ;;

  # 4) Sensitivity-aware pruning score + precision map variants.
  prune_hdiag_only)
    run_case "prune_hdiag_only" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen PRUNE_SCORE_MODE=hdiag \
      XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
    ;;
  prune_hdiag_plus_precision)
    run_case "prune_hdiag_plus_precision" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen PRUNE_SCORE_MODE=hdiag \
      INT8_FORCE_LAST_K=1 INT8_FORCE_PATTERNS=tok_emb.weight,lm_head.weight \
      XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
    ;;

  # 5) Bigram neighborhood search around 3072x112.
  bigram_2816x128)
    run_case "bigram_2816x128" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=128
    ;;
  bigram_3072x112)
    run_case "bigram_3072x112" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112
    ;;
  bigram_3328x96)
    run_case "bigram_3328x96" \
      GPTQ_METHOD=full GPTQ_CALIB_MODE=ar_selfgen XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=3328 BIGRAM_DIM=96
    ;;

  *)
    echo "unknown experiment key: ${EXPERIMENT_KEY}"
    exit 1
    ;;
esac
