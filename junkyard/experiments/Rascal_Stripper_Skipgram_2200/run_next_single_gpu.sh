#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-1}"
SEED="${SEED:-444}"
ITERATIONS="${ITERATIONS:-1200}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
MIN_TRAIN_SHARDS="${MIN_TRAIN_SHARDS:-4}"

echo "============================================================"
echo "RASCAL NEXT SINGLE-GPU LAUNCH"
echo "seed=${SEED} nproc=${NPROC} iterations=${ITERATIONS}"
echo "train_batch_tokens=${TRAIN_BATCH_TOKENS} min_train_shards=${MIN_TRAIN_SHARDS}"
echo "============================================================"

python3 "${SCRIPT_DIR}/run_next_single_gpu.py" \
  --nproc-per-node "${NPROC}" \
  --seed "${SEED}" \
  --iterations "${ITERATIONS}" \
  --train-batch-tokens "${TRAIN_BATCH_TOKENS}" \
  --min-train-shards "${MIN_TRAIN_SHARDS}" \
  --torchrun-bin "${TORCHRUN_BIN}"

