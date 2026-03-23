#!/bin/bash
# RUN_BESTSHOT: Everything that works, stacked
#
# Run3 base (1.1496) + all confirmed/expected improvements:
#   1. MATRIX_LR=0.03 (local ablation: -0.06 BPB at 500 steps)
#   2. fp32 attn_gate (automatic, preserves GA quant correction)
#   3. CK_LR_MULT=1.5 (quant-gate analysis: c_k has 2x damage)
#   4. Late Training Replay (PR #445: 200 extra steps at 10% LR)
#   5. VALUE_RESIDUAL=0 (analysis showed near-zero v0 usage)

set -e
cd /workspace/parameter-golf

# Architecture (run3 base)
export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0
export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
export XSA_LAST_N=4
export VE_ENABLED=1
export WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=524288
export GRAD_CLIP_NORM=0.3
export BIGRAM_HASH_BUCKETS=4096

# 1. MATRIX_LR=0.03
export MATRIX_LR=0.03

# Our innovations
export GATED_ATTENTION=1
export STAR_RELU=1
export LEAKY_RELU=0
export PERLAYER_TRAIN_LR=1
export PROJ_LR_MULT=1.5
export FC_LR_MULT=0.7
export TRIGRAM_HASH=1

# 3. Per-key LR
export CK_LR_MULT=1.5

# 4. Late Training Replay
export LATE_REPLAY=1
export LATE_REPLAY_EPOCHS=2
export LATE_REPLAY_LR_FACTOR=0.1
export LATE_REPLAY_BUFFER_SIZE=100

# 5. Value Residual OFF (analysis showed near-zero usage)
export VALUE_RESIDUAL=0

# No run6 additions
export SIGMOID_SKIP_GATES=0
export DECODER_LR_MULT=1.0

# EMA + SWA, no QAT
export EMA_ENABLED=1
export SWA=1
export QAT=0
export TTT_ENABLED=0
export TTT_CAUSAL=0

export SEED=${1:-1337}
export RUN_TAG="bestshot_$(date +%Y%m%d_%H%M%S)"

# Clean env
unset QUANT_BITS RUN_ID TIER2_MODE MLP_HIDDEN \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 PRUNE_PCT \
  REPTILE_TTT TTT_TWO_PHASE TTT_EPOCHS TTT_MAX_STEPS

echo "=== BESTSHOT: LR03 + CK1.5 + REPLAY + NO_VR ==="
echo "SEED=$SEED MATRIX_LR=0.03 CK=1.5x REPLAY=1 VR=0"
echo "================================================="

NGPU=${NGPU:-8}
if [ "$NGPU" = "1" ]; then
    python3 records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
else
    torchrun --standalone --nproc_per_node=$NGPU \
      records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
fi
