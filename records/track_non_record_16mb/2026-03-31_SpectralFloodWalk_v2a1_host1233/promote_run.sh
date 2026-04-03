#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 runs_host1233/<timestamp>_<profile>_seed<seed>" >&2
  exit 2
fi

RECORD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PATH="$1"
if [[ ! -d "${RUN_PATH}" ]]; then
  echo "missing run directory: ${RUN_PATH}" >&2
  exit 1
fi

cp "${RUN_PATH}/train.log" "${RECORD_DIR}/train.log"
cp "${RUN_PATH}/result.json" "${RECORD_DIR}/result.json"
cp "${RUN_PATH}/final_model.int6.ptz" "${RECORD_DIR}/final_model.int6.ptz"
cp "${RUN_PATH}/residual_artifact.npz" "${RECORD_DIR}/residual_artifact.npz"
echo "Promoted ${RUN_PATH} into ${RECORD_DIR}"
