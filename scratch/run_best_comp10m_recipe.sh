#!/usr/bin/env bash
set -euo pipefail

# Reproducible best-known 10-minute competition recipe (2x3090)
# - Diagnostics OFF
# - Full export/eval ON
# - Roundtrip audit ON

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_god_tier_skc.sh" ]]; then
  ROOT_DIR="${SCRIPT_DIR}"
else
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-best_comp10m_$(date +%Y%m%d_%H%M%S)}"
NPROC="${NPROC:-2}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-540}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-3000}"

env \
  RUN_ID="$RUN_ID" \
  AUTO_TUNE=0 \
  NPROC="$NPROC" \
  DDP_FIND_UNUSED_PARAMETERS=1 \
  DIAGNOSTICS_ENABLED=0 \
  TRAINING_DYNAMICS_ONLY=0 \
  COMPILE_MODE=max-autotune-no-cudagraphs \
  COMPILE_TARGET=full \
  TRAIN_BATCH_TOKENS=32768 \
  MATRIX_LOCK_BATCH_TOKENS=32768 \
  MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
  RUN_TIMEOUT_SECONDS="$RUN_TIMEOUT_SECONDS" \
  ROUNDTRIP_LOGIT_AUDIT=1 \
  ROUNDTRIP_LOGIT_AUDIT_TOKENS=1024 \
  ROUNDTRIP_LOGIT_AUDIT_ENFORCE=0 \
  EXPORT_PARITY_HARNESS=1 \
  bash "$ROOT_DIR/run_god_tier_skc.sh"
