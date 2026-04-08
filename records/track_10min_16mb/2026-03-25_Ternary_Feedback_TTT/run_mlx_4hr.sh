#!/bin/bash
# ============================================================================
# MLX SCALE RUN — 4 HOURS
# Architecture: SKC  8L  dim=256  vocab=1024  (~4M params)
#
# Same exact config as the best 10-min result (8L dim=256 → 1.6552 BPB ngram)
# scaled to a 4-hour window for maximum token throughput on Apple Silicon.
#
# Estimates at ~500ms/step:
#   Steps  : ~28,800
#   Tokens : 28,800 × 16384 = ~472M
#   Memory : ~1GB peak (model + activations + optimizer) — minimal
#
# Memory minimization:
#   - TRAIN_BATCH_TOKENS=16384 (not larger — avoids OOM and keeps step<1s)
#   - MLX_MAX_MICROBATCH_TOKENS=8192
#   - No feedback, no capsule, no VRL, no EMA, no MoE
#   - TURBO_QUANT_TRAIN=0 (quantize export only, not training weights)
#
# Curriculum: 256→512→1024 finishing at 35%/65% — same as 10-min
# Warmdown: starts at 50% of budget (longer run benefits from late warmdown)
# ============================================================================
set -euo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR"

TS=$(date +%s)
export RUN_ID="mlx_4hr_8L256_${TS}"
mkdir -p logs

echo "============================================================"
echo "  MLX 4-HOUR SCALE RUN"
echo "  Model: SKC 8L dim=256  Batch: 16384  Seq: 1024 (final)"
echo "  Budget: 14400s  ~28800 steps  ~472M tokens"
echo "============================================================"

LOG="logs/${RUN_ID}.log"

RUN_ID=${RUN_ID} \
ARCHITECTURE=skc \
NUM_LAYERS=8    MODEL_DIM=256  NUM_HEADS=4  NUM_KV_HEADS=2  MLP_MULT=4 \
VOCAB_SIZE=1024 \
\
SKC_BLOCK_SIZE=16  SKC_NUM_CAPSULES=16  SKC_CAPSULE_DIM=64  SKC_CONV_KERNEL=4 \
\
XSA_START_LAYER=999 \
BIGRAM_HASH_ENABLED=1  BIGRAM_HASH_BUCKETS=3072  BIGRAM_HASH_DIM=112 \
ENGRAM_NUM_HEADS=4  ENGRAM_NUM_ORDERS=3  ENGRAM_INJECT_LAYER=1 \
PARTIAL_ROPE_DIMS=16  LN_SCALE_DAMPING=1 \
\
TRAIN_SEQ_LEN=1024  TRAIN_BATCH_TOKENS=16384  GRAD_ACCUM_STEPS=4 \
MLX_MAX_MICROBATCH_TOKENS=8192  MLX_EAGER_EVAL=1 \
MAX_WALLCLOCK_SECONDS=14400  ITERATIONS=1000000 \
WARMUP_STEPS=5  WARMDOWN_FRACTION=0.5 \
\
CURRICULUM_ENABLED=1 \
CURRICULUM_PHASE1_SEQ=64  CURRICULUM_PHASE2_SEQ=256 \
CURRICULUM_PHASE1_FRAC=0.05  CURRICULUM_PHASE2_FRAC=0.20 \
STOCHASTIC_DEPTH_PROB=0 \
\
MATRIX_LR=0.02  SCALAR_LR=0.015  TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.95  MUON_MOMENTUM_WARMUP_STEPS=0  MUON_BACKEND_STEPS=5 \
MUON_WD=0.04  ADAM_WD=0.04  GRAD_CLIP_NORM=0.3 \
\
LAWA_ENABLED=1  LAWA_K=5  LAWA_FREQ=100 \
SWA_ENABLED=1   SWA_EVERY=50  SMEARGATE_ENABLED=1 \
TKO_ENABLED=0 \
\
FEEDBACK_ENABLED=0  CAPSULE_ENABLED=0  VRL_ENABLED=0 \
TTT_ENABLED=0  EMA_ENABLED=0  MOE_ENABLED=0 \
\
GPTQ_LITE_ENABLED=1  TURBO_QUANT_EXPORT=1  TURBO_QUANT_TRAIN=0  TURBO_QUANT_KV=1 \
NGRAM_CACHE_ENABLED=1  NGRAM_MAX_ORDER=5 \
NGRAM_ALPHA_BASE=0.05  NGRAM_ALPHA_SCALE=0.55  NGRAM_ENTROPY_CENTER=4.0 \
SLIDING_EVAL=1  SLIDING_EVAL_STRIDE=64  TEMP_SCALING=1 \
TRAIN_LOG_EVERY=500  VAL_BATCH_SIZE=65536  VAL_LOSS_EVERY=5000 \
SEED=42 \
bash run_mlx_reasoner.sh 2>&1 | tee "$LOG"

echo ""
echo "===  4-HOUR RESULT  ==="
python3 -c "
import re, math
with open('${LOG}') as f: c = f.read()
steps  = re.findall(r'step:(\d+)/', c)
bpbs   = re.findall(r'ngram_cache.*?val_bpb:([\d.]+)', c) or re.findall(r'final_sliding.*?val_bpb:([\d.]+)', c) or re.findall(r'val_bpb:([\d.]+)', c)
sizes  = re.findall(r'artifact:([\d.]+MB)', c)
toks   = int(steps[-1]) * 16384 if steps else 0
print(f'  steps   : {steps[-1] if steps else \"?\"}')
print(f'  tokens  : {toks/1e6:.1f}M')
print(f'  bpb     : {bpbs[-1] if bpbs else \"?\"}')
print(f'  artifact: {sizes[-1] if sizes else \"?\"}')
if steps and bpbs:
    b = float(bpbs[-1])
    print()
    print(f'  10-min anchor (8L dim=256): 1.6552 BPB')
    print(f'  4-hour result             : {b:.4f} BPB')
    print(f'  Delta vs 10-min           : {b - 1.6552:+.4f}')
" 2>/dev/null || true
echo "  log: $LOG"
