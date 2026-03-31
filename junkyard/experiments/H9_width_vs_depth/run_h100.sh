#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# H9: Width vs Depth at Fixed Parameter Budget
# Scale: 0.25 (150s wallclock)    GPU: 1x H100    No sharing.
#
# Hypothesis: At fixed param count, fewer wider layers beat more
#             narrow layers. The crawler's advantage was WIDTH,
#             not recursion.
#
# Arm A: 6 flat layers, dim=444  (wide, shallow)  ~14.8M params
# Arm B: 8 flat layers, dim=384  (baseline)       ~14.7M params
# Arm C: 10 flat layers, dim=342 (deep, narrow)   ~14.7M params
#
# All flat. No crawlers. No sharing. Isolates width vs depth.
# All extras disabled (same as H8) to keep clean.
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
RESULTS_DIR="experiments/H9_width_vs_depth/results"
mkdir -p "$RESULTS_DIR" checkpoints

# ── Shared config (all extras off, no crawlers anywhere) ──────────
BASE_ENV=(
    SEED="$SEED"
    NUM_KV_HEADS=3 MLP_MULT=3 VOCAB_SIZE=1024
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
    SWA_ENABLED=1 SWA_EVERY=50 QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15
    EVAL_STRIDE=64 VAL_LOSS_EVERY=25 VAL_BATCH_SIZE=524288
    DIAG_FIXED_CADENCE=0 DIAG_FAST_VAL=1
    VE_ENABLED=0 TTT_BURST_ENABLED=0 DISTILL_ENABLED=0 POLAR_ENABLED=0 DTG_ENABLED=0
    TS_PD_ENABLED=0
    NUM_CRAWLER_LAYERS=0 CRAWLER_LOOPS=1
    CRAWLER_CADENCE_EARLY=2 CRAWLER_CADENCE_MAIN=2 CRAWLER_CADENCE_LATE=2
    VE_LAYERS=""
)

# ══════════════════════════════════════════════════════════════════
# ARM A: 6 flat layers, dim=444  (wide, shallow)
# heads=6, head_dim=74
# ══════════════════════════════════════════════════════════════════
RUN_A="h9_armA_6L_dim444_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "================================================================"
echo "  ARM A: 6 flat layers, dim=444 (WIDE)"
echo "  RUN_ID=$RUN_A"
echo "================================================================"
echo ""

mkdir -p "$RESULTS_DIR/$RUN_A"
env \
  "${BASE_ENV[@]}" \
  RUN_ID="$RUN_A" \
  NUM_FLAT_LAYERS=6 MODEL_DIM=444 NUM_HEADS=6 \
  DIAG_CSV_PATH="$RESULTS_DIR/${RUN_A}/diag.csv" \
  torchrun --standalone --nproc_per_node="$NPROC" train_gpt_h4_bottleneck_crawler.py

cp final_model.pt "checkpoints/${RUN_A}_final.pt" 2>/dev/null || true
cp final_model.int6.ptz "checkpoints/${RUN_A}_final.int6.ptz" 2>/dev/null || true
echo "ARM A done: $RESULTS_DIR/${RUN_A}/diag.csv"

# ══════════════════════════════════════════════════════════════════
# ARM B: 8 flat layers, dim=384  (baseline, matches H8 arm A)
# heads=6, head_dim=64
# ══════════════════════════════════════════════════════════════════
RUN_B="h9_armB_8L_dim384_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "================================================================"
echo "  ARM B: 8 flat layers, dim=384 (BASELINE)"
echo "  RUN_ID=$RUN_B"
echo "================================================================"
echo ""

mkdir -p "$RESULTS_DIR/$RUN_B"
env \
  "${BASE_ENV[@]}" \
  RUN_ID="$RUN_B" \
  NUM_FLAT_LAYERS=8 MODEL_DIM=384 NUM_HEADS=6 \
  DIAG_CSV_PATH="$RESULTS_DIR/${RUN_B}/diag.csv" \
  torchrun --standalone --nproc_per_node="$NPROC" train_gpt_h4_bottleneck_crawler.py

cp final_model.pt "checkpoints/${RUN_B}_final.pt" 2>/dev/null || true
cp final_model.int6.ptz "checkpoints/${RUN_B}_final.int6.ptz" 2>/dev/null || true
echo "ARM B done: $RESULTS_DIR/${RUN_B}/diag.csv"

# ══════════════════════════════════════════════════════════════════
# ARM C: 10 flat layers, dim=342  (deep, narrow)
# heads=6, head_dim=57
# ══════════════════════════════════════════════════════════════════
RUN_C="h9_armC_10L_dim342_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "================================================================"
echo "  ARM C: 10 flat layers, dim=342 (DEEP)"
echo "  RUN_ID=$RUN_C"
echo "================================================================"
echo ""

mkdir -p "$RESULTS_DIR/$RUN_C"
env \
  "${BASE_ENV[@]}" \
  RUN_ID="$RUN_C" \
  NUM_FLAT_LAYERS=10 MODEL_DIM=342 NUM_HEADS=6 \
  DIAG_CSV_PATH="$RESULTS_DIR/${RUN_C}/diag.csv" \
  torchrun --standalone --nproc_per_node="$NPROC" train_gpt_h4_bottleneck_crawler.py

cp final_model.pt "checkpoints/${RUN_C}_final.pt" 2>/dev/null || true
cp final_model.int6.ptz "checkpoints/${RUN_C}_final.int6.ptz" 2>/dev/null || true
echo "ARM C done: $RESULTS_DIR/${RUN_C}/diag.csv"

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  H9 WIDTH vs DEPTH — COMPLETE"
echo "================================================================"
echo "  Arm A (6L wide,  dim=444):  $RESULTS_DIR/${RUN_A}/diag.csv"
echo "  Arm B (8L base,  dim=384):  $RESULTS_DIR/${RUN_B}/diag.csv"
echo "  Arm C (10L deep, dim=342):  $RESULTS_DIR/${RUN_C}/diag.csv"
echo ""
echo "  If A < B < C: WIDTH IS THE LEVER (ship wider SOTA)"
echo "  If C < B < A: DEPTH IS THE LEVER (add more layers)"
echo "  If B best:    SWEET SPOT exists (current config is optimal)"
echo "================================================================"
