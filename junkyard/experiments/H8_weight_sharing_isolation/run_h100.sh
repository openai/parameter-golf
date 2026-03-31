#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# H8: Weight-Sharing Isolation Test
# Scale: 0.25 (150s wallclock)    GPU: 1x H100    Cadence: 0 (no C-steps)
#
# Question: Does weight-shared depth (crawler looping 2x) improve BPB
#           over equivalent unique layers at the same effective depth?
#
# Arm A: 8 flat layers, 0 crawler  (8 unique blocks, 8 effective depth)
# Arm B: 6 flat + 1 crawler x2     (7 unique blocks, 8 effective depth)
#
# All extras disabled (VE, TTT, distill, polar, DTG) to isolate
# the weight-sharing signal.
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
RESULTS_DIR="experiments/H8_weight_sharing_isolation/results"
mkdir -p "$RESULTS_DIR" checkpoints

# ── Shared config (small fast model, all extras off) ─────────────
SHARED_ENV=(
    SEED="$SEED"
    MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3 MLP_MULT=3 VOCAB_SIZE=1024
    CRAWLER_MLP_MULT=3
    TRIGRAM_VOCAB_SIZE=1024 TRIGRAM_DIM=128
    XSA_LAST_N=2 ROPE_DIMS=16 LN_SCALE=1
    TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0
    TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432
    ITERATIONS=20000 WARMUP_STEPS=20 GRAD_CLIP_NORM=0.3
    MAX_WALLCLOCK_SECONDS=150 WARMDOWN_ITERS=625
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 TIED_EMBED_INIT_STD=0.005
    MUON_MOMENTUM=0.99 MUON_BACKEND_STEPS=5 MUON_WD=0.04 ADAM_WD=0.04 MUON_BETA2=0.95
    MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
    SWA_ENABLED=1 SWA_EVERY=50 QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15
    EVAL_STRIDE=64 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=524288
    DIAG_FIXED_CADENCE=0 DIAG_FAST_VAL=1
    VE_ENABLED=0 TTT_BURST_ENABLED=0 DISTILL_ENABLED=0 POLAR_ENABLED=0 DTG_ENABLED=0
    TS_PD_ENABLED=0
)

# ══════════════════════════════════════════════════════════════════
# ARM A: 8 flat layers, 0 crawler  (8 unique, 8 effective)
# ══════════════════════════════════════════════════════════════════
RUN_A="h8_armA_8flat_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "================================================================"
echo "  ARM A: 8 flat layers, 0 crawler"
echo "  (8 unique blocks, 8 effective depth, no weight sharing)"
echo "  RUN_ID=$RUN_A"
echo "================================================================"
echo ""

mkdir -p "$RESULTS_DIR/$RUN_A"
env \
  "${SHARED_ENV[@]}" \
  RUN_ID="$RUN_A" \
  NUM_FLAT_LAYERS=8 NUM_CRAWLER_LAYERS=0 CRAWLER_LOOPS=1 \
  CRAWLER_CADENCE_EARLY=2 CRAWLER_CADENCE_MAIN=2 CRAWLER_CADENCE_LATE=2 \
  VE_LAYERS="" \
  DIAG_CSV_PATH="$RESULTS_DIR/${RUN_A}/diag.csv" \
  torchrun --standalone --nproc_per_node="$NPROC" train_gpt_h4_bottleneck_crawler.py

cp final_model.pt "checkpoints/${RUN_A}_final.pt" 2>/dev/null || true
cp final_model.int6.ptz "checkpoints/${RUN_A}_final.int6.ptz" 2>/dev/null || true
echo "ARM A done: $RESULTS_DIR/${RUN_A}/diag.csv"

# ══════════════════════════════════════════════════════════════════
# ARM B: 6 flat + 1 crawler x2  (7 unique, 8 effective)
# ══════════════════════════════════════════════════════════════════
RUN_B="h8_armB_6f1cx2_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "================================================================"
echo "  ARM B: 6 flat layers + 1 crawler x2 at bottleneck"
echo "  (7 unique blocks, 8 effective depth, weight sharing)"
echo "  RUN_ID=$RUN_B"
echo "================================================================"
echo ""

mkdir -p "$RESULTS_DIR/$RUN_B"
env \
  "${SHARED_ENV[@]}" \
  RUN_ID="$RUN_B" \
  NUM_FLAT_LAYERS=6 NUM_CRAWLER_LAYERS=1 CRAWLER_LOOPS=2 \
  CRAWLER_CADENCE_EARLY=2 CRAWLER_CADENCE_MAIN=2 CRAWLER_CADENCE_LATE=2 \
  VE_LAYERS="" \
  DIAG_CSV_PATH="$RESULTS_DIR/${RUN_B}/diag.csv" \
  torchrun --standalone --nproc_per_node="$NPROC" train_gpt_h4_bottleneck_crawler.py

cp final_model.pt "checkpoints/${RUN_B}_final.pt" 2>/dev/null || true
cp final_model.int6.ptz "checkpoints/${RUN_B}_final.int6.ptz" 2>/dev/null || true
echo "ARM B done: $RESULTS_DIR/${RUN_B}/diag.csv"

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  H8 WEIGHT SHARING ISOLATION — COMPLETE"
echo "================================================================"
echo "  Arm A (8 flat, 0 crawler):     $RESULTS_DIR/${RUN_A}/diag.csv"
echo "  Arm B (6 flat, 1 crawler x2):  $RESULTS_DIR/${RUN_B}/diag.csv"
echo ""
echo "  Compare final BPB from each diag.csv to determine if"
echo "  weight-shared depth helps, hurts, or is neutral."
echo "================================================================"
