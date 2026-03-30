#!/bin/bash
# RUN_ONESHOT: Best shot at 1.12x — safe improvements at 524K batch
# Branch: int6-3xMLP-pr (proven, no experimental features)
# Changes from baseline:
#   524K batch (default): 786K was too slow (133ms → fewer steps → worse score)
#   Tight SWA: saves ~3ms/step vs EMA (~180 more steps)
#   WD=20000: smoother weights → better score + compression
#   PRUNE_PCT=3: magnitude pruning for artifact size
#   GPTQ_LITE=1: optimal per-row clip search (default ON)
# Expected: ~1.125-1.135 on a fast pod (sub-75ms)
# Kill if: step_avg@200 > 85ms

set -e
cd /workspace/parameter-golf

export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=32 DOC_ISOLATED_EVAL=0 SEED=1337
export QAT=0 TTT_MAX_STEPS=500 TTT_FREEZE_BLOCKS=1
export EMA_ENABLED=0 SWA=1
export WARMDOWN_ITERS=20000
export PRUNE_PCT=3.0

unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE MLP_QUANT_BITS \
  USE_FA3 LATE_K_FP16 TRAIN_BATCH_TOKENS REPTILE_TTT VE_ENABLED

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
