#!/bin/bash
# ABLATION: Novel innovations (F-H) for closing the gap to PR #505 (1.1181)
#
# Usage: bash run_ablation_innovations.sh <F|G|H|I> [seed]
#
#   F = Progressive Layer Freezing (freeze encoder during warmdown)
#   G = Hyper-Connections scalar (learned mixing of all prior layers)
#   H = Hyper-Connections vector (per-dim mixing weights)
#   I = Logit Ensemble N=2 (average logits from EMA + raw checkpoint)
#
# Tier2 mode (1 GPU, 3 min): export TIER2_MODE=1 before running
# Full mode (8 GPU, 10 min): default

set -e
cd /workspace/parameter-golf

EXPERIMENT="${1:?Usage: bash run_ablation_innovations.sh <F|G|H|I> [seed]}"
export SEED=${2:-1337}

# Base config (matches run_no_ttt.sh)
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
export TRAIN_BATCH_TOKENS=524288
export GRAD_CLIP_NORM=0.0
export EMA_ENABLED=1
export SWA=1
export QAT=0
export TTT_ENABLED=0
export TTT_CAUSAL=0

# Clean env
unset QUANT_BITS RUN_ID TIER2_MODE MLP_HIDDEN \
  BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 PRUNE_PCT \
  REPTILE_TTT TTT_TWO_PHASE TTT_EPOCHS TTT_MAX_STEPS \
  PROGRESSIVE_FREEZE HYPER_CONNECTIONS HYPER_CONN_MODE LOGIT_ENSEMBLE

# Apply tier2 overrides if requested
if [ "${TIER2:-0}" = "1" ]; then
    export TIER2_MODE=1
    export MAX_WALLCLOCK_SECONDS=180
    export EMA_ENABLED=0
    export SWA=0
fi

case "$EXPERIMENT" in
    F)
        export PROGRESSIVE_FREEZE=1
        export PROGRESSIVE_FREEZE_THRESHOLD=0.3
        export RUN_TAG="freeze_${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)"
        echo "=== ABLATION F: Progressive Layer Freezing (threshold=0.3) ==="
        ;;
    G)
        export HYPER_CONNECTIONS=1
        export HYPER_CONN_MODE=scalar
        export UNET_SKIPS=0
        export RUN_TAG="hyper_${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)"
        echo "=== ABLATION G: Hyper-Connections (scalar) ==="
        ;;
    H)
        export HYPER_CONNECTIONS=1
        export HYPER_CONN_MODE=vector
        export UNET_SKIPS=0
        export RUN_TAG="hyper_${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)"
        echo "=== ABLATION H: Hyper-Connections (vector) ==="
        ;;
    I)
        export LOGIT_ENSEMBLE=1
        export LOGIT_ENSEMBLE_N=2
        export LOGIT_ENSEMBLE_STRIDE=128
        export RUN_TAG="ensemble_${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)"
        echo "=== ABLATION I: Logit Ensemble N=2 (stride=128) ==="
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT (use F, G, H, or I)"
        exit 1
        ;;
esac

echo "SEED=$SEED RUN_TAG=$RUN_TAG"

NGPU=${NGPU:-8}
if [ "$NGPU" = "1" ]; then
    python3 records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
else
    torchrun --standalone --nproc_per_node=$NGPU \
      records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
fi
