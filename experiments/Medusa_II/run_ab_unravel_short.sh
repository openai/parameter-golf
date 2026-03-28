#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
AB_GPTQ_CALIB_SAMPLES="${AB_GPTQ_CALIB_SAMPLES:-64}"
AB_GPTQ_CALIB_SEQ_LEN="${AB_GPTQ_CALIB_SEQ_LEN:-1024}"
AB_VAL_TOKEN_CAP="${AB_VAL_TOKEN_CAP:-62021632}"
AB_INCLUDE_NO_LOOP_AWARE="${AB_INCLUDE_NO_LOOP_AWARE:-0}"
AB_INIT_MODEL_PATH="${AB_INIT_MODEL_PATH:-}"
AB_CASES="${AB_CASES:-A,B,C}"

mkdir -p logs
ts="$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo "  MEDUSA_II finish-only A/B for unravel debugging"
echo "  seed=${SEED} nproc=${NPROC_PER_NODE}"
echo "  gptq_calib_samples=${AB_GPTQ_CALIB_SAMPLES} gptq_calib_seq_len=${AB_GPTQ_CALIB_SEQ_LEN}"
echo "  val_token_cap=${AB_VAL_TOKEN_CAP}"
echo "  exit_only=1 init_model_path=${AB_INIT_MODEL_PATH:-<none>}"
echo "  cases=${AB_CASES}"
echo "============================================"

if [[ -z "${AB_INIT_MODEL_PATH}" ]]; then
  echo "AB_INIT_MODEL_PATH is required (finish-only mode)."
  exit 2
fi

run_case() {
  local case_name="$1"
  shift
  local log="logs/medusa2_ab_finish_${case_name}_s${SEED}_${ts}.log"
  local summary_pattern="gptq:calibration config|DIAGNOSTIC post_ema|final_int6_roundtrip_exact|final_int6_sliding_window_exact|final_int8_zlib_roundtrip_exact"
  echo
  echo ">>> CASE ${case_name}"
  echo ">>> LOG  ${log}"

  env \
    SEED="${SEED}" \
    MAX_WALLCLOCK_SECONDS=0 \
    WARMDOWN_ITERS=0 \
    COMPLEMENT_ALPHA=0 \
    XSA_LAST_N=11 \
    BIGRAM_VOCAB_SIZE=2048 \
    ROPE_DIMS=16 \
    SWA_EVERY=50 \
    MTP_NUM_HEADS=0 \
    LATE_QAT_THRESHOLD=0 \
    MATRIX_LR=0.03 \
    TORCHDYNAMO_OPTIMIZE_DDP=0 \
    COMPILE_FULLGRAPH=0 \
    NGRAM_EVAL_ORDER=0 \
    USE_CRAWLER=1 \
    NUM_FLAT_LAYERS=4 \
    NUM_CRAWLER_LAYERS=1 \
    CRAWLER_LOOPS=4 \
    INST_DIM=32 \
    CRAWLER_QUANT_INT8=1 \
    DELTA_NET_HEADS=4 \
    EMA_START_STEP=0 \
    EMA_DECAY=0.99 \
    LOOP_AWARE_GPTQ=1 \
    GPTQ_CALIB_SAMPLES="${AB_GPTQ_CALIB_SAMPLES}" \
    GPTQ_CALIB_SEQ_LEN="${AB_GPTQ_CALIB_SEQ_LEN}" \
    VAL_TOKEN_CAP="${AB_VAL_TOKEN_CAP}" \
    EXIT_ONLY=1 \
    INIT_MODEL_PATH="${AB_INIT_MODEL_PATH}" \
    "$@" \
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      "${SCRIPT_DIR}/train_gpt.py" \
      2>&1 | tee "${log}"

  echo ">>> SUMMARY ${case_name}"
  if command -v rg >/dev/null 2>&1; then
    rg -n "${summary_pattern}" "${log}" || true
  else
    grep -nE "${summary_pattern}" "${log}" || true
  fi
}

if [[ "${AB_CASES}" == *"A"* ]]; then
  # A: current fix path (post-EMA calibration)
  run_case "A_post_ema" GPTQ_POST_EMA=1 DELTA_NET_QUANT_INT8=0
fi

if [[ "${AB_CASES}" == *"B"* ]]; then
  # B: control (pre-EMA calibration, old behavior)
  run_case "B_pre_ema_control" GPTQ_POST_EMA=0 DELTA_NET_QUANT_INT8=0
fi

if [[ "${AB_CASES}" == *"C"* ]]; then
  # C: add delta-net int8 protection
  run_case "C_post_ema_delta_int8" GPTQ_POST_EMA=1 DELTA_NET_QUANT_INT8=1
fi

if [[ "${AB_INCLUDE_NO_LOOP_AWARE}" == "1" && "${AB_CASES}" == *"D"* ]]; then
  # D: disable loop-aware GPTQ
  run_case "D_post_ema_no_loopaware" GPTQ_POST_EMA=1 LOOP_AWARE_GPTQ=0 DELTA_NET_QUANT_INT8=0
fi

echo
echo "============================================"
echo "  FINISH-ONLY A/B DONE"
echo "============================================"
