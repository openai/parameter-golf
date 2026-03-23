#!/bin/bash
# RUN_NO_TTT: Standard sliding window eval, no TTT
#
# Same training as causal TTT run but evaluated with sliding window
# stride=64 instead of test-time training. Clean, no legality questions.

set -e
cd /workspace/parameter-golf
git fetch origin && git checkout causal-ttt && git reset --hard origin/causal-ttt

export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=4
export VE_ENABLED=1
export WARMDOWN_ITERS=3500

# No TTT
export TTT_ENABLED=0
export TTT_CAUSAL=0

export SEED=${1:-1337}

unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 TRAIN_BATCH_TOKENS EMA_ENABLED SWA PRUNE_PCT \
  REPTILE_TTT TTT_TWO_PHASE TTT_EPOCHS TTT_MAX_STEPS

echo "=== NO TTT (SLIDING WINDOW EVAL) ==="
echo "SEED=$SEED stride=64"
echo "====================================="

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
