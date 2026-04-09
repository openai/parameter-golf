#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 runs/<timestamp>_<profile>_seed<seed>" >&2
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
cp "${RUN_PATH}/model_int8.npz" "${RECORD_DIR}/model_int8.npz"
cp "${RUN_PATH}/residual_tables.npz" "${RECORD_DIR}/residual_tables.npz"
echo "Promoted ${RUN_PATH} into ${RECORD_DIR}"
