#!/bin/bash
# v13h: Strip both + 12th layer — trade size for depth
#
# Base: v13g (strip both VE+bigram, 14.82 MiB)
# Change: NUM_LAYERS=12 (was 11), TARGET_MB=15.25 (max safe)
# Estimate: ~16.17 MiB unpruned → needs ~6% pruning to fit
# Risk: pruning killed v13b at 2.9%, but here we gain real capacity
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v13h_12layer"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# Architecture — strip both, add 12th layer
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export MATRIX_CLIP_SIGMAS=12.85
export PARTIAL_KEY_OFFSET=1
export RECUR_LAYERS="4,5"
export RECUR_START_FRAC=0.5
export WARMDOWN_ITERS=4000
export VE_ENABLED=0
export BIGRAM_VOCAB_SIZE=0
export BIGRAM_DIM=112

# v13h change: 12 layers
export NUM_LAYERS=12

# Use maximum safe TARGET_MB
export TARGET_MB=15.25

# No TTT, no parallel
export TTT_ENABLED=0
export PARALLEL_START_LAYER=-1

echo "========================================"
echo "  v13h: 12 layers (strip both VE+bigram)"
echo "  NUM_LAYERS=12, TARGET_MB=15.25"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_v13.py 2>&1 | tee "${LOGDIR}/v13h_12layer.log"
