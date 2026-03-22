#!/bin/bash
# RUN_AGGRESSIVE_TTT: Matches PR #398 (1.1213) strategy
# Key insight: aggressive TTT with all blocks unfrozen closes the gap
# Changes from baseline:
#   TTT_EPOCHS=20: 20 epochs vs our 3 (the breakthrough)
#   TTT_LR=0.008: 4x higher TTT learning rate
#   TTT_FREEZE_BLOCKS=0: all blocks unfrozen (freezing causes internal inconsistency)
#   TTT_MAX_STEPS=9999: no step cap
#   XSA_LAST_N=0: remove XSA (saves ~1.4ms/step = ~130 more steps)
#   FP16_EMBED_EXPORT=0: artifact fits under 16MB
#   LATE_K_FP16=0: smaller artifact
# Expected: ~1.12x on a fast pod (sub-75ms)
# Kill if: step_avg@200 > 85ms

set -e
cd /workspace/parameter-golf
git fetch origin && git checkout int6-3xMLP-pr && git reset --hard origin/int6-3xMLP-pr

export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=32 DOC_ISOLATED_EVAL=0
export QAT=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=0

# Aggressive TTT — the key change
export TTT_EPOCHS=20 TTT_LR=0.008
export TTT_FREEZE_BLOCKS=0 TTT_MAX_STEPS=9999

# Seed from argument or default 1337
export SEED=${1:-1337}

unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  WARMDOWN_ITERS BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 TRAIN_BATCH_TOKENS EMA_ENABLED SWA PRUNE_PCT

echo "=== AGGRESSIVE TTT RUN ==="
echo "SEED=$SEED TTT_EPOCHS=20 TTT_LR=0.008 TTT_FREEZE_BLOCKS=0 XSA_LAST_N=0"
echo "=========================="

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
