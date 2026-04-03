#!/bin/bash
set -euo pipefail
# BW20_Brotli_2k — 1k step gate (1-GPU, seed=444)
# ONE variable: zstd → brotli compression
# Quick sanity: no blowups, check artifact size delta

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="BW20BR"
LOGDIR="${SCRIPT_DIR}/results"
mkdir -p "${LOGDIR}"

pip install brotli -q 2>/dev/null || true

export SEED="${SEED}"
export ITERATIONS=1000
export MAX_WALLCLOCK_SECONDS=0
export COMPILE_FULLGRAPH=1
export CRAWLER_MLP_CHOKE_DIM=0
export CRAWLER_LOOP_ROPE_SCALES=9,1,1
export SKIP_GPTQ=1

echo "=== BW20_Brotli gate — seed=${SEED} �� $(date) ==="

python "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "${LOGDIR}/${TAG}_s${SEED}_${TS}.log"

echo "=== BW20_Brotli gate finished — exit=$? — $(date) ==="
