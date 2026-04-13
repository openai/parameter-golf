#!/usr/bin/env bash
# Phase 1a: QK_GAIN_INIT sweep (5.0 .. 6.0). Usage: ./scripts/bpb_sweep_qk.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
for QK in 5.0 5.25 5.5 5.75 6.0; do
  echo "========== QK_GAIN_INIT=$QK =========="
  QK_GAIN_INIT="$QK" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" "$TRAIN" || true
done
