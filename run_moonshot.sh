#!/bin/bash
# RUN_MOONSHOT: Novel techniques that nobody else has combined
# Branch: next-gen (has Reptile TTT, Shared VE, GPTQ-lite, aggressive TTT)
#
# What's novel (not in any other submission):
#   REPTILE_TTT=1: Reptile meta-learning makes TTT 10x more effective
#   VE_ENABLED=1: Shared Value Embedding on layers 9,10 (only #374 has this)
#   Combined with aggressive TTT (7 epochs, all blocks unfrozen, lr=0.008)
#
# What we learned from analysis:
#   - 524K batch (786K too slow on our pods, breakeven at 132ms)
#   - EMA not SWA (EMA is better for TTT-based runs per #398)
#   - WD=3000 not 20000 (WD20K worse pre-quant at end)
#   - No pruning (3% pruning makes artifact LARGER due to zstd interaction)
#   - XSA_LAST_N=0 (saves ~1.4ms/step = ~130 more training steps, per #398)
#   - EVAL_STRIDE=64 (2x faster eval, leaves time for TTT)
#   - QAT=0 (dead code under torch.compile)
#
# Expected: ~1.115-1.125 (speculative — Reptile + VE untested together)
# Kill if: step_avg@200 > 85ms

set -e
cd /workspace/parameter-golf

export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export QAT=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=0

# Aggressive TTT (matches #398 strategy, capped at 7 for our single-GPU TTT)
export TTT_EPOCHS=7 TTT_LR=0.008
export TTT_FREEZE_BLOCKS=0 TTT_MAX_STEPS=9999

# Novel techniques
export REPTILE_TTT=1
export VE_ENABLED=1
export GPTQ_LITE=1

# Seed from argument or default 1337
export SEED=${1:-1337}

unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  WARMDOWN_ITERS BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 TRAIN_BATCH_TOKENS EMA_ENABLED SWA PRUNE_PCT

echo "=== MOONSHOT RUN ==="
echo "SEED=$SEED REPTILE_TTT=1 VE_ENABLED=1 TTT_EPOCHS=7 TTT_LR=0.008 TTT_FREEZE_BLOCKS=0 XSA=0 STRIDE=64"
echo "===================="

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
