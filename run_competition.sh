#!/bin/bash
# RUN_COMPETITION: Full competition run with per-layer cosine TTT
#
# Training: standard 11L stack (600s wallclock)
# TTT: AdamW lr=5e-4, 30 epochs, cosine decay, per-layer lr (3x proj, 0.5x fc)
# Eval: sliding window stride=64
#
# Tested: 34 TTT configs on 4080. This is the best.
# Target: sub-1.10 on fast pod
# Kill if: step_avg@200 > 85ms

set -e
cd /workspace/parameter-golf
git fetch origin && git checkout next-gen && git reset --hard origin/next-gen

# Training config
export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export QAT=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=0

# Per-layer cosine TTT (our finding: +23.5% over flat AdamW)
export TTT_OPTIMIZER=adamw
export TTT_LR=0.0005
export TTT_EPOCHS=30
export TTT_COSINE=1
export TTT_PERLAYER=1
export TTT_FREEZE_BLOCKS=0
export TTT_BATCH_SEQS=64
export TTT_MAX_STEPS=9999

# Seed from argument or default 1337
export SEED=${1:-1337}

unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  WARMDOWN_ITERS BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 TRAIN_BATCH_TOKENS EMA_ENABLED SWA PRUNE_PCT \
  REPTILE_TTT VE_ENABLED TTT_TWO_PHASE

echo "=== COMPETITION RUN ==="
echo "SEED=$SEED TTT: AdamW lr=$TTT_LR ${TTT_EPOCHS}ep cosine perlayer"
echo "======================="

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
