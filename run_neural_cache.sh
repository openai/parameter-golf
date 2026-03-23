#!/bin/bash
# RUN_NEURAL_CACHE: Same training as run_no_ttt.sh but eval with Neural Cache
#
# Neural Cache persists KV across sliding windows — each token sees
# more context than standard sliding window eval. Zero artifact cost.
# Compare val_bpb against run_no_ttt.sh to measure cache benefit.

set -e
cd /workspace/parameter-golf
git fetch origin && git checkout perlayer-lr-stack && git reset --hard origin/perlayer-lr-stack

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

# Neural Cache eval (the difference vs run_no_ttt.sh)
export NEURAL_CACHE=1
export NEURAL_CACHE_MAX_LEN=8192

export SEED=${1:-1337}

# Clean env
unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 TRAIN_BATCH_TOKENS PRUNE_PCT \
  REPTILE_TTT TTT_TWO_PHASE TTT_EPOCHS TTT_MAX_STEPS

echo "=== NEURAL CACHE EVAL ==="
echo "SEED=$SEED stride=64 EMA=1 SWA=1 QAT=0 VR=1 GA=1 PERLAYER_LR=1 NEURAL_CACHE=1 (max_len=8192)"
echo "========================="

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
