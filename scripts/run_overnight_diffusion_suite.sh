#!/usr/bin/env zsh
set -u

SCRIPT_DIR=${0:A:h}
ROOT_DIR=${SCRIPT_DIR:h}
MANIFEST=${MANIFEST:-${ROOT_DIR}/configs/overnight/manifest.txt}
PYTHON_BIN=${ROOT_DIR}/.venv/bin/python
TRAIN_SCRIPT=${ROOT_DIR}/train_diffusion.py
EXPERIMENT_LOG=${ROOT_DIR}/EXPERIMENT_LOG.md
SEARCH_BIN=$(command -v rg 2>/dev/null || true)
if [[ -z "${SEARCH_BIN}" ]]; then
  SEARCH_BIN=$(command -v grep 2>/dev/null || true)
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -z "${SEARCH_BIN}" ]]; then
  echo "Missing a search tool: neither rg nor grep is available" >&2
  exit 1
fi

SUITE_ID=${1:-overnight_diffusion_$(date +%Y%m%d_%H%M%S)}
SUITE_DIR=${ROOT_DIR}/logs/${SUITE_ID}
RUNNER_LOG=${SUITE_DIR}/runner.txt
SUMMARY_TSV=${SUITE_DIR}/summary.tsv

mkdir -p "${SUITE_DIR}"

{
  echo "suite_id=${SUITE_ID}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "manifest=${MANIFEST}"
  echo "suite_dir=${SUITE_DIR}"
} | tee -a "${RUNNER_LOG}"

if [[ ! -f "${SUMMARY_TSV}" ]]; then
  printf 'run_id\tstatus\tconfig\ttrain_loss\tval_loss\tmodel_bytes\tdiffusion_log\tconsole_log\n' > "${SUMMARY_TSV}"
fi

append_experiment_log() {
  local result_status=$1
  local run_id=$2
  local config_rel=$3
  local goal=$4
  local settings=$5
  local train_loss=$6
  local val_loss=$7
  local model_bytes=$8
  local diffusion_log=$9
  local console_log=${10}
  local model_path=${11}

  {
    echo ""
    echo "### $(date +%F) - ${run_id}"
    echo "- Status: ${result_status}"
    echo "- Script: [\`train_diffusion.py\`](${ROOT_DIR}/train_diffusion.py)"
    echo "- Config: [\`${config_rel}\`](${ROOT_DIR}/${config_rel})"
    echo "- Dataset: FineWeb \`sp1024\`, local overnight suite"
    echo "- Goal: ${goal}"
    echo "- Key settings: ${settings}"
    echo "- Result summary: overnight suite auto-entry"
    echo "- Metrics: train_loss=\`${train_loss}\`, val_loss=\`${val_loss}\`, model_bytes=\`${model_bytes}\`"
    echo "- Artifacts: [\`${diffusion_log:t}\`](${diffusion_log}), [\`${console_log:t}\`](${console_log}), [\`${model_path:t}\`](${model_path})"
    echo "- Findings: pending morning review"
    echo "- Next step: compare against the rest of the overnight suite"
  } >> "${EXPERIMENT_LOG}"
}

last_matching_line() {
  local pattern=$1
  local file_path=$2
  if [[ ! -f "${file_path}" ]]; then
    return 0
  fi
  if [[ ${SEARCH_BIN:t} == "rg" ]]; then
    "${SEARCH_BIN}" "${pattern}" "${file_path}" | tail -n 1
  else
    "${SEARCH_BIN}" -E "${pattern}" "${file_path}" | tail -n 1
  fi
}

experiment_already_completed() {
  local config_rel=$1
  if [[ ! -f "${SUMMARY_TSV}" ]]; then
    return 1
  fi
  awk -v cfg="${config_rel}" '
    NR == 1 { next }
    {
      line = $0
      gsub(/\\t/, "\t", line)
      n = split(line, fields, "\t")
      if (n >= 3 && fields[2] == "completed" && fields[3] == cfg) {
        found = 1
      }
    }
    END { exit(found ? 0 : 1) }
  ' "${SUMMARY_TSV}"
}

while IFS= read -r config_rel || [[ -n "${config_rel}" ]]; do
  [[ -z "${config_rel}" || "${config_rel}" == \#* ]] && continue
  config_path=${ROOT_DIR}/${config_rel}
  if [[ ! -f "${config_path}" ]]; then
    echo "missing_config=${config_rel}" | tee -a "${RUNNER_LOG}"
    continue
  fi
  if experiment_already_completed "${config_rel}"; then
    {
      echo ""
      echo "===== SKIP ${config_rel} ====="
      echo "reason=already_completed_in_summary"
      echo "skipped_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    } | tee -a "${RUNNER_LOG}"
    continue
  fi

  unset EXPERIMENT_NAME EXPERIMENT_GOAL RUN_ID OUT_DIR DATA_PATH TOKENIZER_PATH VOCAB_SIZE TRAIN_SHARDS TRAIN_SEQ_LEN
  unset TRAIN_BATCH_TOKENS VAL_BATCH_TOKENS VAL_MAX_TOKENS ITERATIONS TRAIN_LOG_EVERY VAL_LOSS_EVERY SAMPLE_EVERY
  unset NUM_LAYERS MODEL_DIM NUM_HEADS MLP_MULT NUM_DIFFUSION_STEPS MASK_SCHEDULE MAX_MASK_RATE LEARNING_RATE
  unset GRAD_ACCUM_STEPS MLX_MAX_MICROBATCH_TOKENS MAX_WALLCLOCK_SECONDS SAMPLE_PROMPT

  set -a
  source "${config_path}"
  set +a

  experiment_name=${EXPERIMENT_NAME:-${config_path:t:r}}
  experiment_goal=${EXPERIMENT_GOAL:-Overnight diffusion experiment}
  run_id=${SUITE_ID}_${experiment_name}
  export RUN_ID=${run_id}
  export OUT_DIR=${SUITE_DIR}

  console_log=${SUITE_DIR}/${run_id}_console.txt
  diffusion_log=${SUITE_DIR}/${run_id}_diffusion.txt
  model_path=${SUITE_DIR}/${run_id}_diffusion_mlx.npz

  settings="seq=${TRAIN_SEQ_LEN} dim=${MODEL_DIM} layers=${NUM_LAYERS} heads=${NUM_HEADS} batch_tokens=${TRAIN_BATCH_TOKENS} diff_steps=${NUM_DIFFUSION_STEPS} schedule=${MASK_SCHEDULE} max_mask=${MAX_MASK_RATE:-1.0} lr=${LEARNING_RATE} iterations=${ITERATIONS}"

  {
    echo ""
    echo "===== START ${run_id} ====="
    echo "config=${config_rel}"
    echo "goal=${experiment_goal}"
    echo "settings=${settings}"
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  } | tee -a "${RUNNER_LOG}"

  set +e
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" 2>&1 | tee "${console_log}"
  cmd_status=${pipestatus[1]}
  set -e

  train_line=$(last_matching_line '^step:[0-9]+/[0-9]+ train_loss:' "${diffusion_log}")
  val_line=$(last_matching_line '^step:[0-9]+/[0-9]+ val_loss:' "${diffusion_log}")
  saved_line=$(last_matching_line '^saved_model:' "${diffusion_log}")

  train_loss=$(echo "${train_line}" | sed -n 's/.*train_loss:\([0-9.]*\).*/\1/p')
  val_loss=$(echo "${val_line}" | sed -n 's/.*val_loss:\([0-9.]*\).*/\1/p')
  model_bytes=$(echo "${saved_line}" | sed -n 's/.*bytes:\([0-9]*\).*/\1/p')

  [[ -z "${train_loss}" ]] && train_loss="n/a"
  [[ -z "${val_loss}" ]] && val_loss="n/a"
  [[ -z "${model_bytes}" ]] && model_bytes="n/a"

  if [[ ${cmd_status} -eq 0 ]]; then
    status_label="completed"
  else
    status_label="failed(${cmd_status})"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${run_id}" "${status_label}" "${config_rel}" "${train_loss}" "${val_loss}" "${model_bytes}" "${diffusion_log}" "${console_log}" >> "${SUMMARY_TSV}"

  {
    echo "ended_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "status=${status_label}"
    echo "train_loss=${train_loss}"
    echo "val_loss=${val_loss}"
    echo "model_bytes=${model_bytes}"
    echo "diffusion_log=${diffusion_log}"
    echo "console_log=${console_log}"
    echo "===== END ${run_id} ====="
  } | tee -a "${RUNNER_LOG}"

  append_experiment_log "${status_label}" "${run_id}" "${config_rel}" "${experiment_goal}" "${settings}" "${train_loss}" "${val_loss}" "${model_bytes}" "${diffusion_log}" "${console_log}" "${model_path}"
done < "${MANIFEST}"

{
  echo ""
  echo "suite_completed_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "summary_tsv=${SUMMARY_TSV}"
} | tee -a "${RUNNER_LOG}"
