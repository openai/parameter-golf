#!/usr/bin/env bash
set -euo pipefail

# NON-CLAIMING scaffold commands.
# Replace TBD commands with real benchmark pipeline commands before execution.

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVIDENCE_DIR="${RUN_DIR}/evidence"
mkdir -p "${EVIDENCE_DIR}"

echo "[info] benchmark scaffold only; no benchmark run executed yet"

echo "TODO: run benchmark training pipeline and tee logs to ${EVIDENCE_DIR}/train.log"
echo "TODO: run benchmark eval pipeline and tee logs to ${EVIDENCE_DIR}/eval.log"
echo "TODO: produce ${EVIDENCE_DIR}/submission.json and ${EVIDENCE_DIR}/artifact_sizes.json"
echo "TODO: record environment details to ${EVIDENCE_DIR}/environment.json"
