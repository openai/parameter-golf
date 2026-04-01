#!/bin/bash
# Ablation script: run multiple configs to isolate contributions
# Usage: bash run_ablation.sh <config_name> [ngpu]
#
# Configs:
#   baseline         - SOTA without our changes (control)
#   depth_only       - SOTA + depth recurrence (no enhanced TTT)
#   recovery_only    - SOTA + cosine recovery TTT (no depth recurrence)
#   combined         - SOTA + both improvements
#   recovery_30ep    - SOTA + 30 epoch recovery (higher budget)

CONFIG=${1:-combined}
NGPU=${2:-8}
SEED=${SEED:-1337}

# Common settings
COMMON="NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64"

# TTT common
TTT_COMMON="TTT_ENABLED=1 TTT_LR=0.002 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0"

case $CONFIG in
  baseline)
    echo "Running: baseline (SOTA control)"
    eval "$COMMON $TTT_COMMON TTT_EPOCHS=3 TTT_RECOVERY_EPOCHS=0 \
      DEPTH_RECURRENCE= SEED=$SEED RUN_ID=ablation_baseline_s${SEED} \
      torchrun --standalone --nproc_per_node=$NGPU train_gpt.py"
    ;;
  depth_only)
    echo "Running: SOTA + depth recurrence only"
    eval "$COMMON $TTT_COMMON TTT_EPOCHS=3 TTT_RECOVERY_EPOCHS=0 \
      DEPTH_RECURRENCE=4,5 SEED=$SEED RUN_ID=ablation_depth_s${SEED} \
      torchrun --standalone --nproc_per_node=$NGPU train_gpt.py"
    ;;
  recovery_only)
    echo "Running: SOTA + cosine recovery TTT only"
    eval "$COMMON $TTT_COMMON TTT_EPOCHS=5 TTT_RECOVERY_EPOCHS=20 TTT_RECOVERY_LR=0.001 \
      DEPTH_RECURRENCE= SEED=$SEED RUN_ID=ablation_recovery_s${SEED} \
      torchrun --standalone --nproc_per_node=$NGPU train_gpt.py"
    ;;
  combined)
    echo "Running: SOTA + depth recurrence + cosine recovery TTT"
    eval "$COMMON $TTT_COMMON TTT_EPOCHS=5 TTT_RECOVERY_EPOCHS=20 TTT_RECOVERY_LR=0.001 \
      DEPTH_RECURRENCE=4,5 SEED=$SEED RUN_ID=ablation_combined_s${SEED} \
      torchrun --standalone --nproc_per_node=$NGPU train_gpt.py"
    ;;
  recovery_30ep)
    echo "Running: SOTA + 30-epoch cosine recovery TTT"
    eval "$COMMON $TTT_COMMON TTT_EPOCHS=5 TTT_RECOVERY_EPOCHS=30 TTT_RECOVERY_LR=0.001 \
      DEPTH_RECURRENCE= SEED=$SEED RUN_ID=ablation_recovery30_s${SEED} \
      torchrun --standalone --nproc_per_node=$NGPU train_gpt.py"
    ;;
  *)
    echo "Unknown config: $CONFIG"
    echo "Available: baseline, depth_only, recovery_only, combined, recovery_30ep"
    exit 1
    ;;
esac
