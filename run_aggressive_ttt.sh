#!/bin/bash
# RUN_AGGRESSIVE_TTT: Two-phase DDP TTT (matches PR #415/#417 approach)
# Branch: next-gen (has DDP TTT rewrite)
#
# Key changes from baseline:
#   TTT_TWO_PHASE=1: two-phase norm-then-blocks TTT
#   TTT_P1_EPOCHS=50: phase 1 norm-only recalibration (Adam, 22K params)
#   TTT_P2_EPOCHS=10: phase 2 selective block adaptation (SGD, 7.6M params)
#   TTT_BATCH_SEQS=64: 64 seqs/GPU = 512 total (DDP sharded)
#   XSA_LAST_N=0: remove XSA (saves ~1.4ms/step)
#   EVAL_STRIDE=64: 2x faster eval
#   FP16_EMBED_EXPORT=0 LATE_K_FP16=0: artifact fits under 16MB
#
# Expected timing: training 600s + TTT ~350s + eval ~200s = ~1150s total
# Expected score: ~1.12x on a fast pod (sub-75ms)
# Kill if: step_avg@200 > 85ms

set -e
cd /workspace/parameter-golf
git fetch origin && git checkout next-gen && git reset --hard origin/next-gen

export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export QAT=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=0

# Two-phase DDP TTT
export TTT_TWO_PHASE=1
export TTT_P1_EPOCHS=50 TTT_P1_LR=0.01
export TTT_P2_EPOCHS=10 TTT_P2_LR=0.005 TTT_P2_UNFREEZE_BLOCKS=3
export TTT_BATCH_SEQS=64
export TTT_MAX_STEPS=9999

# Seed from argument or default 1337
export SEED=${1:-1337}

unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  WARMDOWN_ITERS BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 TRAIN_BATCH_TOKENS EMA_ENABLED SWA PRUNE_PCT \
  TTT_EPOCHS TTT_LR TTT_FREEZE_BLOCKS REPTILE_TTT VE_ENABLED

echo "=== TWO-PHASE DDP TTT RUN ==="
echo "SEED=$SEED P1=${TTT_P1_EPOCHS}ep@${TTT_P1_LR} P2=${TTT_P2_EPOCHS}ep@${TTT_P2_LR} BATCH=${TTT_BATCH_SEQS}/GPU XSA=0 STRIDE=64"
echo "=============================="

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
