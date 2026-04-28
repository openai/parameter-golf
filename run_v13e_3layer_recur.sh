#!/bin/bash
# v13e: 3-layer recurrence (L3-5) matching PR #1493
#
# Change from v13a: RECUR_LAYERS="3,4,5" (was "4,5")
# More virtual depth: encoder sees [0,1,2,3,4,5,3,4,5] decoder sees [3,4,5,6,7,8,9,10]
# Risk: more recurred layers = more params quantized twice = potential quant gap increase
#
# Purpose: test if 3-layer recurrence closes gap to PR #1493 (1.0810)
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v13e_3layer_recur"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# Architecture — same as v13a baseline
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export MATRIX_CLIP_SIGMAS=12.85
export PARTIAL_KEY_OFFSET=1
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export RECUR_START_FRAC=0.5
export WARMDOWN_ITERS=4000
export VE_ENABLED=0
export TARGET_MB=15.2

# v13e change: 3-layer recurrence matching PR #1493
export RECUR_LAYERS="3,4,5"

# No TTT, no parallel (isolate recurrence effect)
export TTT_ENABLED=0
export PARALLEL_START_LAYER=-1

echo "========================================"
echo "  v13e: 3-layer recurrence (L3-5)"
echo "  RECUR_LAYERS=3,4,5 (was 4,5)"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_v13.py 2>&1 | tee "${LOGDIR}/v13e_3layer_recur.log"
