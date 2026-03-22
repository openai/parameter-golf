#!/bin/bash
# RUN_BASELINE: Proven 1.1375 config — exactly what scored 1.1375 on Pod 3
# Branch: int6-3xMLP-pr
# Expected: ~1.137-1.140 depending on pod speed
# Artifact: ~16.6 MB (over 16MB limit — needs PRUNE_PCT or MLP_QUANT_BITS to fix)

set -e
cd /workspace/parameter-golf
# Save this script before checkout (it doesn't exist on int6-3xMLP-pr)
cp -f run_baseline.sh /tmp/run_baseline.sh 2>/dev/null || true
git fetch origin && git checkout int6-3xMLP-pr && git reset --hard origin/int6-3xMLP-pr

export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=32 DOC_ISOLATED_EVAL=0 SEED=1337
export QAT=0 TTT_MAX_STEPS=500 TTT_FREEZE_BLOCKS=1

unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  WARMDOWN_ITERS BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 LATE_K_FP16 TRAIN_BATCH_TOKENS \
  EMA_ENABLED SWA PRUNE_PCT GPTQ_LITE REPTILE_TTT VE_ENABLED

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
