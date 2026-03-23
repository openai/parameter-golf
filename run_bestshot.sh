#!/bin/bash
# RUN_BESTSHOT: Run3 config + MATRIX_LR=0.03 + QAT=1
#
# The key insight: higher LR trains better pre-quant but weights are quant-fragile.
# QAT teaches the model to survive int6 quantization at the higher LR.
#
# Run11 proved: LR=0.03 pre-quant 1.1520 but quant gap +0.014 → 1.1664 (worse)
# Run5 proved: QAT makes quant gap negative (-0.009)
# Combined: better pre-quant + protected quant gap → projected 1.140-1.145
#
# Only TWO changes from run3 (1.1496): MATRIX_LR and QAT. Everything else identical.

set -e
cd /workspace/parameter-golf

# Exact run3 config
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

# Our innovations (same as run3)
export VALUE_RESIDUAL=1
export GATED_ATTENTION=1
export STAR_RELU=1
export LEAKY_RELU=0
export PERLAYER_TRAIN_LR=1
export PROJ_LR_MULT=1.5
export FC_LR_MULT=0.7
export TRIGRAM_HASH=1

# No run6 additions
export SIGMOID_SKIP_GATES=0
export DECODER_LR_MULT=1.0

# THE TWO CHANGES from run3:
export MATRIX_LR=0.03    # Better pre-quant (run11: 1.1520 vs run3: 1.1499)
export QAT=1             # Protects quant gap (run5: -0.009 vs run3: -0.0003)

# Same as run3
export EMA_ENABLED=1
export SWA=1
export TTT_ENABLED=0
export TTT_CAUSAL=0

export SEED=${1:-1337}
export RUN_TAG="bestshot_$(date +%Y%m%d_%H%M%S)"

# Clean env
unset QUANT_BITS RUN_ID TIER2_MODE MLP_HIDDEN \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 PRUNE_PCT \
  REPTILE_TTT TTT_TWO_PHASE TTT_EPOCHS TTT_MAX_STEPS

echo "=== BESTSHOT: MATRIX_LR=0.03 + QAT=1 ==="
echo "SEED=$SEED MATRIX_LR=0.03 QAT=1 (2 changes from run3)"
echo "============================================"

NGPU=${NGPU:-8}
if [ "$NGPU" = "1" ]; then
    python3 records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
else
    torchrun --standalone --nproc_per_node=$NGPU \
      records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
fi
