#!/bin/bash
set -euo pipefail
# ================================================================
# Helix — DGX Spark Micro Suite 2
#
# Following up on Phase 1 findings:
#   - dim=64 was breakout signal (-0.0464 vs ctrl)
#   - stride=5 (rare fire) beat stride=1 (frequent)
#   - Marco-Polo cross-attn not yet tested
#
# This suite: push dim higher, combine best stride+dim,
# test marco-polo, find the 7F control, and hunt the ceiling.
#
# Usage:
#   cd ~/parameter-golf-lab && bash crawler/2026-04-03_Helix/run_spark_micro2.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${SCRIPT_DIR}/results/spark_micro2"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/micro2_summary_${TS}.tsv"

pip install brotli -q 2>/dev/null || true

MICRO_ENV=(
    SEED="${SEED}"
    ITERATIONS=200
    MAX_WALLCLOCK_SECONDS=0
    WARMDOWN_ITERS=50
    TRAIN_BATCH_TOKENS=131072
    VAL_BATCH_SIZE=131072
    EVAL_STRIDE=2048
    TRAIN_SEQ_LEN=512
    EVAL_SEQ_LEN=512
    COMPILE_ENABLED=0
    COMPILE_FULLGRAPH=0
    USE_CRAWLER=1
    NUM_CRAWLER_LAYERS=1
    CRAWLER_MLP_MULT=4.0
    MODEL_DIM=256
    NUM_HEADS=4
    NUM_KV_HEADS=2
    INST_DIM=16
    BIGRAM_VOCAB_SIZE=512
    BIGRAM_DIM=64
    XSA_LAST_N=0
    ROPE_DIMS=8
    SWA_EVERY=20
    SKIP_GPTQ=1
    SKIP_EMA=1
    QK_GAIN_INIT=4.0
    MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_LEAKY_SLOPE=0.5
    CRAWLER_MLP_CHOKE_DIM=0
    CRAWLER_LOOP_ROPE_SCALES=9,1,1
    CRAWLER_TAP_DIM=0
    ANCHOR_DIM=0
    MATRIX_LR=0.03
    TRAIN_LOG_EVERY=50
    VAL_LOSS_EVERY=200
)

run_arm() {
    local tag="$1"; shift
    local logfile="${RESULTS_DIR}/${tag}_s${SEED}_${TS}.log"
    echo ""
    echo "================================================================"
    echo "  ARM: ${tag} — $(date)"
    echo "================================================================"
    env "${MICRO_ENV[@]}" "$@" \
        python "${TRAIN_PY}" 2>&1 | tee "${logfile}"
    local bpb=$(grep -oP 'step:200/200 val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local int6=$(grep -oP 'final_int6_sliding_window_exact.*val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local step_ms=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 2>/dev/null || echo "?")
    local params=$(grep -oP 'model_params:\K[0-9]+' "${logfile}" 2>/dev/null || echo "?")
    echo -e "${tag}\t${params}\t${bpb}\t${int6}\t${step_ms}" >> "${SUMMARY}"
    echo "  >>> ${tag}: bpb=${bpb} int6=${int6} step_ms=${step_ms} params=${params}"
}

echo -e "arm\tparams\traw_bpb\tint6_sw_bpb\tstep_ms" > "${SUMMARY}"

echo ""
echo "================================================================"
echo "  HELIX MICRO SUITE 2 — Pushing the ceiling"
echo "  200 steps, dim=256, seq=512, compile=off"
echo "================================================================"

# ----------------------------------------------------------------
# CONTROLS: Missing from suite 1
# ----------------------------------------------------------------

echo ""
echo "==== CONTROLS — Fill the gaps ===="

# X0: 7F control (no helix) — needed to isolate helix vs depth
run_arm "X0_ctrl_7flat_2loop" \
    NUM_FLAT_LAYERS=7 CRAWLER_LOOPS=2 HELIX=0

# X1: 5F control 1loop (no helix) — match C-series loop count
run_arm "X1_ctrl_5flat_1loop" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=0

# ----------------------------------------------------------------
# DIM CEILING: How far does the bridge width go?
# ----------------------------------------------------------------

echo ""
echo "==== DIM CEILING — Does dim keep scaling? ===="

# H0: Rebase dim=64 at stride=1 1loop (reference from suite 1 = 1.8616)
run_arm "H0_dim64_s1" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64

# H1: dim=96
run_arm "H1_dim96_s1" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=96

# H2: dim=128
run_arm "H2_dim128_s1" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=128

# H3: dim=192 (3/4 of model_dim — very fat pipe)
run_arm "H3_dim192_s1" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=192

# ----------------------------------------------------------------
# BEST COMBO: stride=5 + fat dim
# ----------------------------------------------------------------

echo ""
echo "==== COMBO — Rare fire + fat pipe ===="

# J0: stride=5 + dim=64 (combine two best findings)
run_arm "J0_s5_dim64" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=5 HELIX_DIM=64

# J1: stride=5 + dim=128
run_arm "J1_s5_dim128" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=5 HELIX_DIM=128

# J2: stride=3 + dim=64 (slightly more frequent)
run_arm "J2_s3_dim64" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=3 HELIX_DIM=64

# J3: stride=3 + dim=128
run_arm "J3_s3_dim128" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=3 HELIX_DIM=128

# ----------------------------------------------------------------
# MARCO-POLO: Cross-attention vs linear at fat dims
# ----------------------------------------------------------------

echo ""
echo "==== MARCO-POLO — Content-addressed routing ===="

# M0: Marco-Polo dim=16 (baseline comparison)
run_arm "M0_marco_dim16" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=16 HELIX_CROSS_ATTN=1

# M1: Marco-Polo dim=32
run_arm "M1_marco_dim32" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=32 HELIX_CROSS_ATTN=1

# M2: Marco-Polo dim=64 (head-to-head with D3 linear)
run_arm "M2_marco_dim64" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 HELIX_CROSS_ATTN=1

# M3: Marco-Polo dim=64 stride=5 (content-addressed + rare fire)
run_arm "M3_marco_dim64_s5" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=5 HELIX_DIM=64 HELIX_CROSS_ATTN=1

# M4: Marco-Polo dim=128 stride=1
run_arm "M4_marco_dim128" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=128 HELIX_CROSS_ATTN=1

# ----------------------------------------------------------------
# DEPTH + FAT DIM: Scale the winner to 7F and 9F
# ----------------------------------------------------------------

echo ""
echo "==== DEPTH SCALING — Fat pipe at real depths ===="

# K0: 7F + dim=64 stride=1 linear
run_arm "K0_7f_dim64" \
    NUM_FLAT_LAYERS=7 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64

# K1: 7F + dim=64 stride=3 linear
run_arm "K1_7f_dim64_s3" \
    NUM_FLAT_LAYERS=7 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=3 HELIX_DIM=64

# K2: 7F + dim=64 marco-polo
run_arm "K2_7f_marco_dim64" \
    NUM_FLAT_LAYERS=7 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 HELIX_CROSS_ATTN=1

# K3: 9F + dim=64 linear (production-scale depth)
run_arm "K3_9f_dim64" \
    NUM_FLAT_LAYERS=9 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64

# K4: 9F + dim=64 marco-polo
run_arm "K4_9f_marco_dim64" \
    NUM_FLAT_LAYERS=9 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64 HELIX_CROSS_ATTN=1

# ----------------------------------------------------------------
# DONE
# ----------------------------------------------------------------

echo ""
echo "================================================================"
echo "  MICRO SUITE 2 COMPLETE — $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""
cat "${SUMMARY}" | column -t -s$'\t'
