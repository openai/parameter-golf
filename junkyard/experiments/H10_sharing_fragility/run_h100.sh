#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# H10: Sharing Fragility — Does SWA Kill the Sharing Signal?
# Scale: 0.25 (150s wallclock)    GPU: 1x H100
#
# Hypothesis: Weight sharing provides ~0.01 BPB via implicit
#             regularization, but SWA weight averaging destroys
#             the convergent structure that shared weights learn.
#
# Arm A: 6 flat + 1 crawler x2, SWA ON   (current default)
# Arm B: 6 flat + 1 crawler x2, SWA OFF  (preserve sharing structure)
# Arm C: 8 flat + 0 crawler,    SWA ON   (flat control)
# Arm D: 8 flat + 0 crawler,    SWA OFF  (flat without SWA)
#
# If sharing + SWA_OFF beats sharing + SWA_ON by more than
# flat + SWA_OFF beats flat + SWA_ON, then SWA specifically
# damages the sharing signal.
#
# Interaction test: (B-A) vs (D-C). If (B-A) > (D-C), SWA hurts
# shared models more than flat models → fragility confirmed.
# ══════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    if [ -d "flash-attention/hopper" ]; then
        export PYTHONPATH="$(pwd)/flash-attention/hopper:${PYTHONPATH:-}"
    else
        echo "ERROR: flash_attn_interface not found." && exit 1
    fi
fi

NPROC=1
SEED="${SEED:-1337}"
RESULTS_DIR="experiments/H10_sharing_fragility/results"
mkdir -p "$RESULTS_DIR" checkpoints

# ── Shared config ─────────────────────────────────────────────────
BASE_ENV=(
    SEED="$SEED"
    MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3 MLP_MULT=3 VOCAB_SIZE=1024
    CRAWLER_MLP_MULT=3
    TRIGRAM_VOCAB_SIZE=1024 TRIGRAM_DIM=128
    XSA_LAST_N=2 ROPE_DIMS=16 LN_SCALE=1
    TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0
    TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432
    ITERATIONS=20000 WARMUP_STEPS=20 GRAD_CLIP_NORM=0.3
    MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=625
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 TIED_EMBED_INIT_STD=0.005
    MUON_MOMENTUM=0.99 MUON_BACKEND_STEPS=5 MUON_WD=0.04 ADAM_WD=0.04 MUON_BETA2=0.95
    MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
    QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15
    EVAL_STRIDE=64 VAL_LOSS_EVERY=25 VAL_BATCH_SIZE=524288
    DIAG_FIXED_CADENCE=0 DIAG_FAST_VAL=1
    VE_ENABLED=0 TTT_BURST_ENABLED=0 DISTILL_ENABLED=0 POLAR_ENABLED=0 DTG_ENABLED=0
    TS_PD_ENABLED=0
)

run_arm() {
    local ARM_NAME="$1" FLAT="$2" CRAWL="$3" LOOPS="$4" SWA="$5"
    local RUN_ID="h10_${ARM_NAME}_$(date +%Y%m%d_%H%M%S)"

    echo ""
    echo "================================================================"
    echo "  ARM ${ARM_NAME}: ${FLAT}flat+${CRAWL}cx${LOOPS} SWA=${SWA}"
    echo "  RUN_ID=$RUN_ID"
    echo "================================================================"
    echo ""

    mkdir -p "$RESULTS_DIR/$RUN_ID"
    env \
      "${BASE_ENV[@]}" \
      RUN_ID="$RUN_ID" \
      NUM_FLAT_LAYERS="$FLAT" NUM_CRAWLER_LAYERS="$CRAWL" CRAWLER_LOOPS="$LOOPS" \
      CRAWLER_CADENCE_EARLY=2 CRAWLER_CADENCE_MAIN=2 CRAWLER_CADENCE_LATE=2 \
      SWA_ENABLED="$SWA" SWA_EVERY=50 \
      VE_LAYERS="" \
      DIAG_CSV_PATH="$RESULTS_DIR/${RUN_ID}/diag.csv" \
      torchrun --standalone --nproc_per_node="$NPROC" train_gpt_h4_bottleneck_crawler.py

    cp final_model.pt "checkpoints/${RUN_ID}_final.pt" 2>/dev/null || true
    cp final_model.int6.ptz "checkpoints/${RUN_ID}_final.int6.ptz" 2>/dev/null || true
    echo "ARM ${ARM_NAME} done: $RESULTS_DIR/${RUN_ID}/diag.csv"
}

# Arm A: shared + SWA on
run_arm "A_shared_swa" 6 1 2 1

# Arm B: shared + SWA off
run_arm "B_shared_noswa" 6 1 2 0

# Arm C: flat + SWA on
run_arm "C_flat_swa" 8 0 1 1

# Arm D: flat + SWA off
run_arm "D_flat_noswa" 8 0 1 0

echo ""
echo "================================================================"
echo "  H10 SHARING FRAGILITY — COMPLETE"
echo "================================================================"
echo "  Compare: (B_noswa - A_swa) vs (D_noswa - C_swa)"
echo "  If SWA damages sharing more: (B-A) > (D-C)"
echo "================================================================"
