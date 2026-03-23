#!/bin/bash
# RUN_NO_TTT: Standard sliding window eval, no TTT
#
# Matches PR #414 approach: EMA + Tight SWA, no QAT, sliding window eval.
# Clean, no legality questions.

set -e
cd /workspace/parameter-golf

export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=4
export VE_ENABLED=1
export WARMDOWN_ITERS=3500
export VALUE_RESIDUAL=1
export GATED_ATTENTION=1
export PERLAYER_TRAIN_LR=1
export PROJ_LR_MULT=1.5
export FC_LR_MULT=0.7
export STAR_RELU=1
export TRIGRAM_HASH=1
export BIGRAM_HASH_BUCKETS=8192
export TRAIN_BATCH_TOKENS=786432
export GRAD_CLIP_NORM=0.0

# Match PR #414: EMA + Tight SWA, no QAT
export EMA_ENABLED=1
export SWA=1
export QAT=0

# No TTT
export TTT_ENABLED=0
export TTT_CAUSAL=0

export SEED=${1:-1337}
export RUN_TAG="no_ttt_$(date +%Y%m%d_%H%M%S)"

# Clean env — only unset things we DON'T explicitly set above
unset QUANT_BITS RUN_ID TIER2_MODE \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 PRUNE_PCT \
  REPTILE_TTT TTT_TWO_PHASE TTT_EPOCHS TTT_MAX_STEPS

echo "=== NO TTT (SLIDING WINDOW EVAL) ==="
echo "SEED=$SEED stride=64 EMA=1 SWA=1 QAT=0 VR=1 GA=1 PERLAYER_LR=1 STAR_RELU=1 TRIGRAM=1 MLP=1792 batch=786K bigram=8192 noclip"
echo "====================================="

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
