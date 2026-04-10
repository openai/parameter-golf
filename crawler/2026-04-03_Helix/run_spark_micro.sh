#!/bin/bash
set -euo pipefail
# ================================================================
# Helix — DGX Spark Micro Test Suite
#
# Tiny model, tiny data, fast iterations. Testing that the math works
# and cross-injection signals flow correctly before committing to
# big runs. Each arm runs ~60-120s on DGX Spark.
#
# Usage:
#   bash crawler/2026-04-03_Helix/run_spark_micro.sh
#   NPROC_PER_NODE=2 bash crawler/2026-04-03_Helix/run_spark_micro.sh
# ================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-1}"
TRAIN_PY="${SCRIPT_DIR}/train_gpt.py"
TS="$(date +%Y%m%d_%H%M%S)"

RESULTS_DIR="${SCRIPT_DIR}/results/spark_micro"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/micro_summary_${TS}.tsv"

pip install brotli -q 2>/dev/null || true

# Micro config: tiny everything, just testing math
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
    HELIX_DIM=16
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
    # Extract key metrics
    local bpb=$(grep -oP 'step:200/200 val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local int6=$(grep -oP 'final_int6_sliding_window_exact.*val_bpb:\K[0-9.]+' "${logfile}" 2>/dev/null || echo "?")
    local step_ms=$(grep -oP 'step_avg:\K[0-9.]+' "${logfile}" | tail -1 2>/dev/null || echo "?")
    local params=$(grep -oP 'model_params:\K[0-9]+' "${logfile}" 2>/dev/null || echo "?")
    echo -e "${tag}\t${params}\t${bpb}\t${int6}\t${step_ms}" >> "${SUMMARY}"
    echo "  >>> ${tag}: bpb=${bpb} int6=${int6} step_ms=${step_ms} params=${params}"
}

# Header
echo -e "arm\tparams\traw_bpb\tint6_sw_bpb\tstep_ms" > "${SUMMARY}"

echo ""
echo "================================================================"
echo "  HELIX MICRO TEST SUITE — DGX Spark"
echo "  200 steps, dim=256, seq=512, compile=off"
echo "  Testing architecture math, not absolute quality"
echo "================================================================"

# ----------------------------------------------------------------
# FOUNDATION TESTS: Does helix work at all?
# ----------------------------------------------------------------

echo ""
echo "==== PHASE 1: FOUNDATION — Does helix work? ===="

# A0: Control — standard sequential (no helix)
run_arm "A0_ctrl_3flat_2loop" \
    NUM_FLAT_LAYERS=3 CRAWLER_LOOPS=2 HELIX=0

# A1: Helix stride=1 on tiny 3F model (crawler fires 3x)
run_arm "A1_helix_s1_3flat" \
    NUM_FLAT_LAYERS=3 CRAWLER_LOOPS=2 HELIX=1 HELIX_STRIDE=1

# A2: Helix stride=1, no loops (CRAWLER_LOOPS=1, pure helix co-fire)
run_arm "A2_helix_s1_3flat_1loop" \
    NUM_FLAT_LAYERS=3 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1

# ----------------------------------------------------------------
# DEPTH TESTS: How does helix scale with flat depth?
# ----------------------------------------------------------------

echo ""
echo "==== PHASE 2: DEPTH — Helix at different flat depths ===="

# B0: Control 5F sequential
run_arm "B0_ctrl_5flat_2loop" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=2 HELIX=0

# B1: Helix 5F stride=1 (5 crawler passes)
run_arm "B1_helix_s1_5flat" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=2 HELIX=1 HELIX_STRIDE=1

# B2: Helix 7F stride=1 (7 crawler passes)
run_arm "B2_helix_s1_7flat" \
    NUM_FLAT_LAYERS=7 CRAWLER_LOOPS=2 HELIX=1 HELIX_STRIDE=1

# ----------------------------------------------------------------
# STRIDE TESTS: How much crawler co-firing do we need?
# ----------------------------------------------------------------

echo ""
echo "==== PHASE 3: STRIDE — How often should crawler fire? ===="

# C0: 5F helix stride=1 (every layer, 5 passes)
# (reuses B1 if already run — skip or re-run for consistency)
run_arm "C0_helix_5f_s1" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1

# C1: 5F helix stride=2 (layers 2,4 — 2 passes)
run_arm "C1_helix_5f_s2" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=2

# C2: 5F helix stride=5 (only final layer — 1 pass)
run_arm "C2_helix_5f_s5" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=5

# ----------------------------------------------------------------
# CROSS-INJECTION DIM TESTS: How wide should the bridge be?
# ----------------------------------------------------------------

echo ""
echo "==== PHASE 4: BRIDGE WIDTH — Cross-injection dimension ===="

# D0: helix_dim=8 (very narrow bridge)
run_arm "D0_helix_dim8" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=8

# D1: helix_dim=16 (default for micro)
run_arm "D1_helix_dim16" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=16

# D2: helix_dim=32 (wider bridge)
run_arm "D2_helix_dim32" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=32

# D3: helix_dim=64 (wide bridge — does it overfit at this scale?)
run_arm "D3_helix_dim64" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=64

# ----------------------------------------------------------------
# MARCO-POLO TESTS: Cross-attention vs linear projection
# ----------------------------------------------------------------

echo ""
echo "==== PHASE 5: MARCO-POLO — Cross-attention vs linear projection ===="

# F0: Linear projection baseline (same as B1 but explicit for comparison)
run_arm "F0_linear_5f_dim16" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=16 HELIX_CROSS_ATTN=0

# F1: Marco-Polo cross-attention dim=16
run_arm "F1_marco_5f_dim16" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=16 HELIX_CROSS_ATTN=1

# F2: Marco-Polo dim=32
run_arm "F2_marco_5f_dim32" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=32 HELIX_CROSS_ATTN=1

# F3: Marco-Polo dim=8 (very cheap — is it enough?)
run_arm "F3_marco_5f_dim8" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=8 HELIX_CROSS_ATTN=1

# F4: Marco-Polo at stride=3 (less frequent but content-addressed)
run_arm "F4_marco_5f_s3_dim16" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=3 HELIX_DIM=16 HELIX_CROSS_ATTN=1

# F5: Marco-Polo on deeper model (7F)
run_arm "F5_marco_7f_dim16" \
    NUM_FLAT_LAYERS=7 CRAWLER_LOOPS=1 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=16 HELIX_CROSS_ATTN=1

# F6: Marco-Polo + 2 loops (does loop recurrence help cross-attn?)
run_arm "F6_marco_5f_2loop" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=2 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=16 HELIX_CROSS_ATTN=1

# ----------------------------------------------------------------
# INTERACTION TESTS: Helix + other crawler features
# ----------------------------------------------------------------

echo ""
echo "==== PHASE 6: INTERACTIONS — Helix + existing features ===="

# G0: Helix + INST_DIM (do instructions help or conflict?)
run_arm "G0_helix_inst32" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=2 HELIX=1 HELIX_STRIDE=1 INST_DIM=32

# G1: Helix + anchor (does anchor help in helix mode?)
run_arm "G1_helix_anchor16" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=2 HELIX=1 HELIX_STRIDE=1 ANCHOR_DIM=16

# G2: Marco-Polo + INST_DIM (does content-addressed + instruction stack?)
run_arm "G2_marco_inst32" \
    NUM_FLAT_LAYERS=5 CRAWLER_LOOPS=2 HELIX=1 HELIX_STRIDE=1 HELIX_DIM=16 HELIX_CROSS_ATTN=1 INST_DIM=32

# ----------------------------------------------------------------
# DONE
# ----------------------------------------------------------------

echo ""
echo "================================================================"
echo "  ALL MICRO TESTS COMPLETE — $(date)"
echo "  Summary: ${SUMMARY}"
echo "================================================================"
echo ""
cat "${SUMMARY}" | column -t -s$'\t'
