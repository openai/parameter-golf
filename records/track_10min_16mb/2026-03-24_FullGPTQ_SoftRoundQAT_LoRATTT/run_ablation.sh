#!/bin/bash
set -euo pipefail

# Ablation study: test each improvement individually on 1xH100
# Usage: bash run_ablation.sh <ablation_name>
#   ablation_name: baseline | full_gptq | soft_round | lora_ttt | all

ABLATION=${1:-baseline}
SEED=${2:-42}

# Base hyperparams (same architecture for all)
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=2
export MLP_MULT=3
export TIE_EMBEDDINGS=1
export TIED_EMBED_INIT_STD=0.02
export LOGIT_SOFTCAP=30.0
export ROPE_BASE=10000
export QK_GAIN_INIT=1.0
export LN_SCALE=1
export XSA_LAST_N=4
export ROPE_DIMS=16
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=128
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export GATED_ATTENTION=0
export VALUE_RESIDUAL=1

# Shorter run for ablations (500 steps)
export ITERATIONS=500
export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_SIZE=64
export WARMUP_ITERS=50
export WARMDOWN_FRACTION=0.26
export LEARNING_RATE=0.0036
export MUON_LR=0.0126
export WEIGHT_DECAY=0.0
export ADAM_BETA2=0.95
export MIN_LR_RATIO=0.0
export EMA_DECAY=0.95
export EMA_START_STEP=100
export RANDOM_SEED=$SEED

# Defaults: all improvements OFF
export LATE_QAT_THRESHOLD=0.0
export TTT_ENABLED=0
export TTT_LORA=0

case "$ABLATION" in
  baseline)
    echo "=== ABLATION: Baseline (GPTQ-lite, no QAT, no TTT) ==="
    ;;
  full_gptq)
    echo "=== ABLATION: Full GPTQ only ==="
    # Full GPTQ is always on (Hessian collection happens automatically)
    ;;
  soft_round)
    echo "=== ABLATION: Soft-Round QAT ==="
    export LATE_QAT_THRESHOLD=0.18
    ;;
  lora_ttt)
    echo "=== ABLATION: LoRA TTT ==="
    export TTT_ENABLED=1
    export TTT_LORA=1
    export TTT_LORA_RANK=8
    export TTT_LORA_LR=0.005
    export TTT_LORA_ALPHA=16.0
    export TTT_EPOCHS=3
    export TTT_CHUNK_TOKENS=32768
    export TTT_BATCH_SEQS=32
    export TTT_GRAD_CLIP=1.0
    ;;
  all)
    echo "=== ABLATION: All improvements ==="
    export LATE_QAT_THRESHOLD=0.18
    export TTT_ENABLED=1
    export TTT_LORA=1
    export TTT_LORA_RANK=8
    export TTT_LORA_LR=0.005
    export TTT_LORA_ALPHA=16.0
    export TTT_EPOCHS=3
    export TTT_CHUNK_TOKENS=32768
    export TTT_BATCH_SEQS=32
    export TTT_GRAD_CLIP=1.0
    ;;
  *)
    echo "Unknown ablation: $ABLATION"
    echo "Valid options: baseline, full_gptq, soft_round, lora_ttt, all"
    exit 1
    ;;
esac

python3 train_gpt.py 2>&1 | tee "ablation_${ABLATION}_seed${SEED}.log"

echo "=== Ablation $ABLATION complete ==="
