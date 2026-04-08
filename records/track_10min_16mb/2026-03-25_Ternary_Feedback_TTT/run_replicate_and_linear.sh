#!/bin/bash
# Replicate 1.6552 baseline + linear rank=128 variant
# BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128 (original proven config)
set -euo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"; cd "$EXPDIR"
TS=$(date +%s); mkdir -p logs

run() {
    local NAME=$1; shift
    local LOG="logs/rep_${NAME}_${TS}.log"
    echo ""; echo "━━━ $NAME ━━━"
    RUN_ID="rep_${NAME}_${TS}" \
    ARCHITECTURE=skc \
    NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    VOCAB_SIZE=1024 \
    SKC_BLOCK_SIZE=16 SKC_NUM_CAPSULES=16 SKC_CAPSULE_DIM=64 SKC_CONV_KERNEL=4 \
    XSA_START_LAYER=999 \
    BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128 \
    ENGRAM_NUM_HEADS=4 ENGRAM_NUM_ORDERS=3 ENGRAM_INJECT_LAYER=1 \
    PARTIAL_ROPE_DIMS=16 LN_SCALE_DAMPING=1 \
    TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=4 \
    MLX_MAX_MICROBATCH_TOKENS=8192 MLX_EAGER_EVAL=1 \
    MAX_WALLCLOCK_SECONDS=600 ITERATIONS=100000 \
    WARMUP_STEPS=5 WARMDOWN_FRACTION=0.5 \
    CURRICULUM_ENABLED=1 \
    CURRICULUM_PHASE1_SEQ=64 CURRICULUM_PHASE2_SEQ=256 \
    CURRICULUM_PHASE1_FRAC=0.05 CURRICULUM_PHASE2_FRAC=0.20 \
    STOCHASTIC_DEPTH_PROB=0 \
    MATRIX_LR=0.02 SCALAR_LR=0.015 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.95 MUON_MOMENTUM_WARMUP_STEPS=0 MUON_BACKEND_STEPS=5 \
    MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3 \
    LAWA_ENABLED=1 LAWA_K=5 LAWA_FREQ=100 \
    SWA_ENABLED=1 SWA_EVERY=50 SMEARGATE_ENABLED=1 TKO_ENABLED=0 \
    FEEDBACK_ENABLED=0 VRL_ENABLED=0 TTT_ENABLED=0 EMA_ENABLED=0 MOE_ENABLED=0 \
    GPTQ_LITE_ENABLED=1 TURBO_QUANT_EXPORT=1 TURBO_QUANT_TRAIN=0 TURBO_QUANT_KV=1 \
    NGRAM_CACHE_ENABLED=1 NGRAM_MAX_ORDER=5 \
    NGRAM_ALPHA_BASE=0.05 NGRAM_ALPHA_SCALE=0.55 NGRAM_ENTROPY_CENTER=4.0 \
    SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=32 TEMP_SCALING=0 \
    TRAIN_LOG_EVERY=50 VAL_BATCH_SIZE=65536 VAL_LOSS_EVERY=0 \
    SEED=42 \
    env "$@" bash run_mlx_reasoner.sh 2>&1 | tee "$LOG"

    BPB=$(grep "ngram_cache" "$LOG" | grep -o 'val_bpb:[0-9.]*' | tail -1 | cut -d: -f2)
    STEPS=$(grep "^step:" "$LOG" | grep -v val_loss | tail -1 | sed 's/step:\([0-9]*\)\/.*/\1/')
    MS=$(grep "^step:" "$LOG" | grep -v val_loss | awk -F'step:' '{print $NF}' | sed 's/ms//' | tail -20 | awk '{s+=$1;n++} END{printf "%.0f",s/n}')
    echo "  ✓ $NAME → bpb=${BPB:-?}  steps=${STEPS:-?}  ms/step=${MS:-?}"
    echo "$NAME|${BPB:-?}|${STEPS:-?}|${MS:-?}"
}

echo "════════════════════════════════════════════"
echo "  Replicate 1.6552 + Linear Bottleneck r128"
echo "  bigram=4096×128 | seq=256 | 10min each"
echo "════════════════════════════════════════════"

run "A_baseline"    CAPSULE_ENABLED=0
run "B_linear_r128" CAPSULE_ENABLED=1 LINEAR_BOTTLENECK=1 LINEAR_BOTTLENECK_RANK=128

echo ""
echo "════════════════════════════════════════════"
echo "  FINAL COMPARISON"
echo "════════════════════════════════════════════"
echo "  Original 1.6552 run:    1.6552 BPB (reference)"
grep "^[AB]_" /dev/stdin <<'EOF' || true
EOF
grep -h "ngram_cache" logs/rep_A_baseline_${TS}.log logs/rep_B_linear_r128_${TS}.log 2>/dev/null | \
  grep -o 'val_bpb:[0-9.]*' | awk -F: 'NR==1{print "  A baseline:    "$2" BPB"} NR==2{print "  B linear r128: "$2" BPB"}'
