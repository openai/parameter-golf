#!/bin/bash
set -euo pipefail
# Helix — 1k step gate (1-GPU, seed=444)
# TWO arms: stride=1 (9 crawler passes) and stride=3 (3 passes)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${SCRIPT_DIR}/results"
mkdir -p "${LOGDIR}"

pip install brotli -q 2>/dev/null || true

# Shared config (Ouroboros base + helix)
BASE_ENV=(
    SEED="${SEED}"
    ITERATIONS=1000
    MAX_WALLCLOCK_SECONDS=0
    COMPILE_FULLGRAPH=1
    USE_CRAWLER=1
    NUM_FLAT_LAYERS=9
    NUM_CRAWLER_LAYERS=1
    CRAWLER_LOOPS=2
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    SKIP_GPTQ=1
    SKIP_EMA=1
    MODEL_DIM=512
    INST_DIM=32
    CRAWLER_MLP_MULT=6.0
    CRAWLER_TAP_DIM=0
    ANCHOR_DIM=0
    QK_GAIN_INIT=4.0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    MATRIX_LR=0.03
    HELIX_DIM=32
)

echo "=== Helix Control (no helix) — seed=${SEED} — $(date) ==="
env "${BASE_ENV[@]}" HELIX=0 \
    python "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "${LOGDIR}/helix_ctrl_s${SEED}_${TS}.log"

echo "=== Helix Stride=3 (3 crawler passes) — seed=${SEED} — $(date) ==="
env "${BASE_ENV[@]}" HELIX=1 HELIX_STRIDE=3 \
    python "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "${LOGDIR}/helix_s3_s${SEED}_${TS}.log"

echo "=== Helix Stride=1 (9 crawler passes) — seed=${SEED} — $(date) ==="
env "${BASE_ENV[@]}" HELIX=1 HELIX_STRIDE=1 \
    python "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "${LOGDIR}/helix_s1_s${SEED}_${TS}.log"

echo "=== Helix gate complete — $(date) ==="
