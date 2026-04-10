#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "${ROOT}"

MODE="${1:-all}"
SEED="${SEED:-444}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/neural/experiments/QK4_Warmdown/logs"
SUMMARY_FILE="${LOG_DIR}/warmdown_kit_${SEED}_${TS}.tsv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PROXY_ITERATIONS="${PROXY_ITERATIONS:-2000}"
PROXY_WARMDOWN_ITERS="${PROXY_WARMDOWN_ITERS:-200}"

mkdir -p "${LOG_DIR}"

export PYTHONPATH="${ROOT}/flash-attention/hopper${PYTHONPATH:+:${PYTHONPATH}}"
export DATA_PATH="${ROOT}/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="${ROOT}/data/tokenizers/fineweb_1024_bpe.model"
export SEED

base_env() {
  export ITERATIONS="${PROXY_ITERATIONS}"
  export MAX_WALLCLOCK_SECONDS=0
  export VAL_LOSS_EVERY=0
  export TRAIN_LOG_EVERY=200
  export TRAIN_BATCH_TOKENS=786432
  export TRAIN_SEQ_LEN=2048
  export EVAL_SEQ_LEN=2048
  export COMPILE_ENABLED=1
  export COMPILE_FULLGRAPH=1
  export LOADER_MODE=coprime
  export COPRIME_MAX_LOADED_SHARDS=4
  export COPRIME_SHARDS_PER_BATCH=1
  export COPRIME_SHARD_HOLD_STEPS=64
  export POST_EMA_DIAGNOSTIC=1
  export SKIP_FINAL_EVAL=1
  export NGRAM_EVAL_ORDER=0
  export SKIP_GPTQ=1
  export QK_GAIN_INIT=4
  export WARMDOWN_ITERS="${PROXY_WARMDOWN_ITERS}"
  export WARMDOWN_MODE=linear
  export WARMDOWN_JITTER_SIGMA=0.3
  export WARMDOWN_SWIRL_CYCLES=4
  export WARMDOWN_SWIRL_AMP=0.3
  export WARMDOWN_SCALAR_MULT=1.5
  export WARMDOWN_BANK_MULT=0.7
}

apply_case() {
  local case_name="$1"
  base_env
  case "${case_name}" in
    linear)
      ;;
    jitter)
      export WARMDOWN_MODE=jitter
      ;;
    swirl3)
      export WARMDOWN_MODE=swirl
      export WARMDOWN_SWIRL_CYCLES=3
      export WARMDOWN_SWIRL_AMP=0.3
      ;;
    swirl5)
      export WARMDOWN_MODE=swirl
      export WARMDOWN_SWIRL_CYCLES=5
      export WARMDOWN_SWIRL_AMP=0.2
      ;;
    cascade)
      export WARMDOWN_MODE=cascade
      ;;
    *)
      echo "Unknown case: ${case_name}" >&2
      return 1
      ;;
  esac
}

append_summary() {
  local case_name="$1"
  local logfile="$2"
  local post_ema roundtrip model_bytes total_bytes
  post_ema="$(grep -F 'DIAGNOSTIC post_ema' "${logfile}" | tail -1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || true)"
  roundtrip="$(grep -F 'final_int6_roundtrip_exact' "${logfile}" | tail -1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || true)"
  model_bytes="$(grep -F 'Serialized model int6+' "${logfile}" | tail -1 | sed -E 's/.*: ([0-9]+) bytes/\1/' || true)"
  total_bytes="$(grep -F 'Total submission size int6+' "${logfile}" | tail -1 | sed -E 's/.*: ([0-9]+) bytes/\1/' || true)"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${case_name}" "${post_ema:-}" "${roundtrip:-}" "${model_bytes:-}" "${total_bytes:-}" >> "${SUMMARY_FILE}"
}

run_one() {
  local case_name="$1"
  local run_id logfile
  apply_case "${case_name}"
  run_id="qk4warm_${case_name}_s${SEED}_${TS}"
  logfile="${LOG_DIR}/${run_id}.log"
  export RUN_ID="${run_id}"
  echo "CASE=${case_name} SEED=${SEED} NPROC=${NPROC_PER_NODE} ITER=${ITERATIONS} WD=${WARMDOWN_ITERS} MODE=${WARMDOWN_MODE}"
  echo "LOG=${logfile}"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${ROOT}/neural/experiments/QK4_Warmdown/train_gpt.py" \
    2>&1 | tee "${logfile}"
  append_summary "${case_name}" "${logfile}"
}

printf "case\tpost_ema_bpb\troundtrip_bpb\tmodel_bytes\ttotal_bytes\n" > "${SUMMARY_FILE}"

case "${MODE}" in
  fast)
    for c in linear swirl3 cascade; do
      run_one "${c}"
    done
    ;;
  all)
    for c in linear jitter swirl3 swirl5 cascade; do
      run_one "${c}"
    done
    ;;
  *)
    run_one "${MODE}"
    ;;
esac

echo
echo "SUMMARY=${SUMMARY_FILE}"
column -t -s $'\t' "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"
