#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Easy launcher for stripped RASCAL skip-gram signal tests on 1x H100.
# Default: calibration sweep (baseline + low + high).
#
# Quick start:
#   bash experiments/Rascal_Stripper_Skipgram_2200/run_calibration.sh
#
# Optional overrides:
#   MODE=ab SEEDS=444 bash experiments/Rascal_Stripper_Skipgram_2200/run_calibration.sh
#   MODE=calibrate SEEDS=42,300,444 NPROC=1 ITERATIONS=2200 bash experiments/Rascal_Stripper_Skipgram_2200/run_calibration.sh

MODE="${MODE:-calibrate}"              # calibrate | ab
SEEDS="${SEEDS:-444}"                  # comma-separated, e.g. 42,300,444
NPROC="${NPROC:-1}"
ITERATIONS="${ITERATIONS:-2200}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-131072}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

# Low/high calibration defaults (only used when MODE=calibrate).
LOW_PATTERNS="${LOW_PATTERNS:-1,3}"
LOW_MIX="${LOW_MIX:-0.5}"
HIGH_PATTERNS="${HIGH_PATTERNS:-1,3,5;1,2,4;1,4,8}"
HIGH_MIX="${HIGH_MIX:-1.5}"

echo "============================================================"
echo "RASCAL STRIPPED SKIPGRAM LAUNCHER"
echo "mode=${MODE} seeds=${SEEDS} nproc=${NPROC} iterations=${ITERATIONS}"
echo "============================================================"

python3 "${SCRIPT_DIR}/run.py" \
  --nproc-per-node "${NPROC}" \
  --seeds "${SEEDS}" \
  --iterations "${ITERATIONS}" \
  --train-batch-tokens "${TRAIN_BATCH_TOKENS}" \
  --mode "${MODE}" \
  --low-patterns "${LOW_PATTERNS}" \
  --low-mix "${LOW_MIX}" \
  --high-patterns "${HIGH_PATTERNS}" \
  --high-mix "${HIGH_MIX}" \
  --torchrun-bin "${TORCHRUN_BIN}"
