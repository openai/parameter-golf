#!/usr/bin/env bash
set -euo pipefail

RECORD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 1
fi

RUN_DIR="$1"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

for required in train.log result.json model_int8.npz; do
  if [[ ! -f "${RUN_DIR}/${required}" ]]; then
    echo "missing required artifact: ${RUN_DIR}/${required}" >&2
    exit 1
  fi
done

for name in train.log result.json model_int8.npz command.sh notes.txt; do
  if [[ -f "${RUN_DIR}/${name}" ]]; then
    cp "${RUN_DIR}/${name}" "${RECORD_DIR}/${name}"
  fi
done

echo "Promoted run into ${RECORD_DIR}:"
echo "  train.log"
echo "  result.json"
echo "  model_int8.npz"
