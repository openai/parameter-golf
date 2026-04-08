#!/bin/bash
# ============================================================================
# MLX SCALE RUN — 10 MINUTES
# Goal: maximum data throughput × maximum parameters in a 10-min window
#
# Config:
#   Model  : SKC  8L  dim=256  vocab=1024  (~4.2M params)
#   Seq    : curriculum 256→512→1024 (final)
#   Batch  : 16384 tokens/step  → ~1.0s/step on M-series → ~580 steps
#   Tokens : 580 × 16384 = 9.5M total
#
# Why these numbers:
#   dim=256 is the proven sweet spot for 10-min window (best BPB in ablations)
#   batch=16384 + seq=1024 keeps step time ~1s → ~580 gradient updates in 600s
#   More gradient updates > larger batch for convergence at this token budget
#   Curriculum (256→512→1024) accelerates early convergence before final seq_len
#
# Expected: ~1.65 BPB (ngram) — reproducing the ablation best
# Calibration use: anchor for H100 scaling projection
# ============================================================================
set -euo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR"

TS=$(date +%s)
export RUN_ID="mlx_10min_s42_${TS}"
mkdir -p logs

echo "============================================================"
echo "  MLX 10-MIN SCALE RUN"
echo "  Model: SKC 8L dim=256  Batch: 16384  Seq: 1024"
echo "  Expected: ~580 steps / ~9.5M tokens / ~1.65 BPB"
echo "============================================================"

LOG="logs/${RUN_ID}.log"

RUN_ID=${RUN_ID} \
ARCHITECTURE=skc \
NUM_LAYERS=8    MODEL_DIM=256  NUM_HEADS=4  NUM_KV_HEADS=2  MLP_MULT=4 \
VOCAB_SIZE=1024 \
\
SKC_BLOCK_SIZE=16  SKC_NUM_CAPSULES=16  SKC_CAPSULE_DIM=64  SKC_CONV_KERNEL=4 \
\
XSA_START_LAYER=0 \
BIGRAM_HASH_ENABLED=1  BIGRAM_HASH_BUCKETS=3072  BIGRAM_HASH_DIM=112 \
ENGRAM_NUM_HEADS=4  ENGRAM_NUM_ORDERS=3  ENGRAM_INJECT_LAYER=1 \
PARTIAL_ROPE_DIMS=16  LN_SCALE_DAMPING=1 \
\
TRAIN_SEQ_LEN=1024  TRAIN_BATCH_TOKENS=16384  GRAD_ACCUM_STEPS=4 \
MLX_MAX_MICROBATCH_TOKENS=8192  MLX_EAGER_EVAL=1 \
MAX_WALLCLOCK_SECONDS=600  ITERATIONS=100000 \
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
TRAIN_LOG_EVERY=50  VAL_BATCH_SIZE=65536  SEED=42 \
bash run_mlx_reasoner.sh 2>&1 | tee "$LOG"

echo ""
echo "===  10-MIN RESULT  ==="
python3 -c "
import re
with open('${LOG}') as f: c = f.read()
steps  = re.findall(r'step:(\d+)/', c)
bpbs   = re.findall(r'ngram_cache.*?val_bpb:([\d.]+)', c) or re.findall(r'final_sliding.*?val_bpb:([\d.]+)', c) or re.findall(r'val_bpb:([\d.]+)', c)
sizes  = re.findall(r'artifact:([\d.]+MB)', c)
toks   = int(steps[-1]) * 16384 if steps else 0
print(f'  steps   : {steps[-1] if steps else \"?\"}')
print(f'  tokens  : {toks/1e6:.1f}M')
print(f'  bpb     : {bpbs[-1] if bpbs else \"?\"}')
print(f'  artifact: {sizes[-1] if sizes else \"?\"}')
" 2>/dev/null || true
echo "  log: $LOG"
