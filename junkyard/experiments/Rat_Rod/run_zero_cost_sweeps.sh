#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for remaining zero-overhead Rat Rod sweeps.
# Runs WARMDOWN_ITERS sweep, then SWA_EVERY sweep, forwarding runtime overrides.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WALLCLOCK_SECONDS="${WALLCLOCK_SECONDS:-200}"

echo "============================================"
echo "  Rat Rod Zero-Cost Sweep Bundle"
echo "  Seeds: ${SEEDS}"
echo "  Wallclock per run: ${WALLCLOCK_SECONDS}s"
echo "  NPROC: ${NPROC_PER_NODE}"
echo "============================================"

SEEDS="${SEEDS}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
WALLCLOCK_SECONDS="${WALLCLOCK_SECONDS}" \
COMPILE_ENABLED="${COMPILE_ENABLED:-1}" \
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-1}" \
TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-0}" \
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}" \
NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-9}" \
bash "${REPO_ROOT}/experiments/Rat_Rod/sweep_warmdown_200s.sh"

SEEDS="${SEEDS}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
WALLCLOCK_SECONDS="${WALLCLOCK_SECONDS}" \
COMPILE_ENABLED="${COMPILE_ENABLED:-1}" \
COMPILE_FULLGRAPH="${COMPILE_FULLGRAPH:-1}" \
TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-0}" \
SKIP_FINAL_EVAL="${SKIP_FINAL_EVAL:-0}" \
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}" \
NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-9}" \
bash "${REPO_ROOT}/experiments/Rat_Rod/sweep_swa_200s.sh"

echo "============================================"
echo "  Done"
echo "============================================"
