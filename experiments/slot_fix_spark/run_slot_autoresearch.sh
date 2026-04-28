#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

MODE="${1:-short}"
SEED="${SEED:-300}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/experiments/slot_fix_spark/logs"
SUMMARY_FILE="${LOG_DIR}/slot_autoresearch_${SEED}_${TS}.tsv"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)}"
if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" == "0" ]]; then
  NPROC_PER_NODE=1
fi

mkdir -p "${LOG_DIR}"

base_env() {
  export SEED
  export NPROC_PER_NODE
  export ITERATIONS="${ITERATIONS:-64}"
  export MAX_WALLCLOCK_SECONDS=0
  export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-16}"
  export POST_EMA_DIAGNOSTIC=0
  export SKIP_FINAL_EVAL=1
  export EXIT_AFTER_SIZE_ONLY=1
  export COMPILE_ENABLED="${COMPILE_ENABLED:-0}"
  export COMPILE_FULLGRAPH=0
  export SKIP_GPTQ=1
  export SLOT_ENABLED=0
  export SLOT_STEPS=1
  export SLOT_LR=1e-2
  export SLOT_POWER=0.30
}

apply_case() {
  local case_name="$1"
  base_env
  case "${case_name}" in
    ctrl)
      ;;
    slot_p10)
      export SLOT_ENABLED=1
      export SLOT_POWER=0.10
      export SLOT_STEPS=1
      export SLOT_LR=0.005
      ;;
    slot_p30)
      export SLOT_ENABLED=1
      export SLOT_POWER=0.30
      export SLOT_STEPS=1
      export SLOT_LR=0.01
      ;;
    slot_p30_s2)
      export SLOT_ENABLED=1
      export SLOT_POWER=0.30
      export SLOT_STEPS=2
      export SLOT_LR=0.01
      ;;
    slot_p50_s2)
      export SLOT_ENABLED=1
      export SLOT_POWER=0.50
      export SLOT_STEPS=2
      export SLOT_LR=0.01
      ;;
    *)
      echo "Unknown case: ${case_name}" >&2
      return 1
      ;;
  esac
}

extract_field() {
  local pattern="$1"
  local logfile="$2"
  grep -F "${pattern}" "${logfile}" | tail -1 | sed -E 's/.*: ([0-9]+) bytes/\1/' || true
}

append_summary() {
  local case_name="$1"
  local logfile="$2"
  local model_bytes total_bytes code_bytes
  model_bytes="$(extract_field "Serialized model int6+" "${logfile}")"
  total_bytes="$(extract_field "Total submission size int6+" "${logfile}")"
  code_bytes="$(extract_field "Code size" "${logfile}")"
  printf "%s\t%s\t%s\t%s\n" \
    "${case_name}" "${model_bytes:-}" "${total_bytes:-}" "${code_bytes:-}" >> "${SUMMARY_FILE}"
}

run_one() {
  local case_name="$1"
  local run_id logfile
  apply_case "${case_name}"
  run_id="slot_auto_${case_name}_s${SEED}_${TS}"
  logfile="${LOG_DIR}/${run_id}.log"
  export RUN_ID="${run_id}"
  echo "CASE=${case_name} SEED=${SEED} NPROC=${NPROC_PER_NODE} ITER=${ITERATIONS} SLOT=${SLOT_ENABLED} STEPS=${SLOT_STEPS} LR=${SLOT_LR} POWER=${SLOT_POWER}"
  echo "LOG=${logfile}"
  bash "${ROOT}/experiments/slot_fix_spark/run_slot_fix.sh" 2>&1 | tee "${logfile}"
  append_summary "${case_name}" "${logfile}"
}

printf "case\tmodel_bytes\ttotal_bytes\tcode_bytes\n" > "${SUMMARY_FILE}"

case "${MODE}" in
  short)
    for c in ctrl slot_p30 slot_p30_s2; do
      run_one "${c}"
    done
    ;;
  all)
    for c in ctrl slot_p10 slot_p30 slot_p30_s2 slot_p50_s2; do
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
