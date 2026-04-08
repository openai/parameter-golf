#!/bin/bash
# ============================================================================
# MLX SCALE RUN — 1 HOUR
# Goal: maximum data throughput × maximum parameters in a 1-hour window
#
# Config:
#   Model  : SKC  12L  dim=512  vocab=1024  (~20M params)
#   Seq    : curriculum 256→512→2048 (full competition seq_len)
#   Batch  : 65536 tokens/step  → ~3.5s/step on M-series → ~1000 steps
#   Tokens : 1000 × 65536 = 65M total
#
# Why these numbers:
#   12L dim=512: largest model that comfortably fits in 16GB with seq=512 (curriculum)
#     At peak seq=2048: batch=65536/2048=32 seqs × 512 dim → ~10GB activation mem → safe
#   batch=65536: fills GPU memory efficiently; 8 seqs per microbatch (8192/1024)
#   1-hour window: enough steps to see the full scaling curve + warmdown plateau
#   Curriculum 256→512→2048: lets model see short patterns early, then full context
#
# Expected: ~1.45–1.55 BPB (ngram) — best possible on this hardware
# Calibration use: this is the closest approximation to what H100 will do at scale
#   (12L dim=512 here → 24L dim=512 on H100 with 10× more tokens = projection anchor)
#
# OOM safety:
#   Peak mem = model_params + activations + optimizer state
#   12L dim=512 in bf16: ~20M × 2B = 40MB (tiny)
#   Activations at batch=65536 seq=2048: ~(65536/2048) × 2048 × 512 × bf16 × layers
#     = 32 × 2048 × 512 × 2B × 12 ≈ 768MB (well within 16GB)
#   Optimizer (Adam, SWA buffers): ~3× model = ~120MB
#   Total estimate: ~2GB — safe
# ============================================================================
set -euo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR"

TS=$(date +%s)
export RUN_ID="mlx_1hr_s42_${TS}"
mkdir -p logs

echo "============================================================"
echo "  MLX 1-HOUR SCALE RUN"
echo "  Model: SKC 12L dim=512  Batch: 65536  Seq: 2048 (final)"
echo "  Expected: ~1000 steps / ~65M tokens / ~1.45-1.55 BPB"
echo "============================================================"

LOG="logs/${RUN_ID}.log"

RUN_ID=${RUN_ID} \
ARCHITECTURE=skc \
NUM_LAYERS=12   MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4  MLP_MULT=4 \
VOCAB_SIZE=1024 \
\
SKC_BLOCK_SIZE=16  SKC_NUM_CAPSULES=16  SKC_CAPSULE_DIM=128  SKC_CONV_KERNEL=4 \
\
XSA_START_LAYER=0 \
BIGRAM_HASH_ENABLED=1  BIGRAM_HASH_BUCKETS=3072  BIGRAM_HASH_DIM=112 \
ENGRAM_NUM_HEADS=4  ENGRAM_NUM_ORDERS=3  ENGRAM_INJECT_LAYER=1 \
PARTIAL_ROPE_DIMS=16  LN_SCALE_DAMPING=1 \
\
TRAIN_SEQ_LEN=2048  TRAIN_BATCH_TOKENS=65536  GRAD_ACCUM_STEPS=4 \
MLX_MAX_MICROBATCH_TOKENS=8192  MLX_EAGER_EVAL=1 \
MAX_WALLCLOCK_SECONDS=3600  ITERATIONS=500000 \
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
TRAIN_LOG_EVERY=100  VAL_BATCH_SIZE=65536  VAL_LOSS_EVERY=2000 \
SEED=42 \
bash run_mlx_reasoner.sh 2>&1 | tee "$LOG"

echo ""
echo "===  1-HOUR RESULT  ==="
python3 -c "
import re
with open('${LOG}') as f: c = f.read()
steps  = re.findall(r'step:(\d+)/', c)
bpbs   = re.findall(r'ngram_cache.*?val_bpb:([\d.]+)', c) or re.findall(r'final_sliding.*?val_bpb:([\d.]+)', c) or re.findall(r'val_bpb:([\d.]+)', c)
sizes  = re.findall(r'artifact:([\d.]+MB)', c)
toks   = int(steps[-1]) * 65536 if steps else 0
print(f'  steps   : {steps[-1] if steps else \"?\"}')
print(f'  tokens  : {toks/1e6:.1f}M')
print(f'  bpb     : {bpbs[-1] if bpbs else \"?\"}')
print(f'  artifact: {sizes[-1] if sizes else \"?\"}')
print()
# Scaling projection to H100
if steps and bpbs:
    s = int(steps[-1]); b = float(bpbs[-1]); tk = s * 65536
    # H100: 12K steps × 786K tokens = 9.4B tokens (144× more)
    import math
    h100_tokens = 12000 * 786432
    scale_factor = math.log(h100_tokens / tk) / math.log(2)
    # Chinchilla: BPB improves ~0.05 per doubling of tokens at this scale
    improvement_per_doubling = 0.05
    projected = b - scale_factor * improvement_per_doubling
    print(f'  --- H100 PROJECTION ---')
    print(f'  Mac tokens : {tk/1e6:.0f}M')
    print(f'  H100 tokens: {h100_tokens/1e6:.0f}M  ({h100_tokens/tk:.0f}× more)')
    print(f'  Scale doublings: {scale_factor:.1f}')
    print(f'  Projected H100 BPB: ~{projected:.3f}  (rough estimate)')
" 2>/dev/null || true
echo "  log: $LOG"
