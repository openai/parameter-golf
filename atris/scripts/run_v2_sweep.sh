#!/bin/bash
# Atris v2 SWEEP: Test architectural changes on 1 GPU
# Finds which direction to push before committing to 8xH100
#
# Each run: ~2 min, ~$0.05 on 1xA100

set -euo pipefail

cd "$(dirname "$0")/../.."

NPROC=1
WALLCLOCK=120

run() {
    local name=$1; shift
    echo ""
    echo "=== $name ==="
    RUN_ID="sweep_${name}_$(date +%s)" \
    MAX_WALLCLOCK_SECONDS=$WALLCLOCK \
    VAL_LOSS_EVERY=0 \
    TRAIN_LOG_EVERY=100 \
    "$@" \
    torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | \
        grep -E "(val_bpb|val_loss|final_int8|model_params|submission size)" | \
        tee -a atris/logs/sweep_v2.log
}

echo "=== Atris v2 Sweep: Architecture Search ==="
echo "Base: NUM_LAYERS=10, MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03"
echo ""

# Base (v1 config on 1 GPU for comparison)
run "base_10L" \
    NUM_LAYERS=10 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

# More layers (can we fit 11 or 12?)
run "11_layers" \
    NUM_LAYERS=11 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

run "12_layers" \
    NUM_LAYERS=12 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

# Wider model (needs to fit in 16MB after quant)
run "wider_576" \
    NUM_LAYERS=10 MODEL_DIM=576 NUM_HEADS=8 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

run "wider_640" \
    NUM_LAYERS=10 MODEL_DIM=640 NUM_HEADS=8 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

# MLP multiplier
run "mlp_3x" \
    NUM_LAYERS=10 MLP_MULT=3 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

# More KV heads
run "kv8" \
    NUM_LAYERS=10 NUM_KV_HEADS=8 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

# Fewer KV heads (saves params)
run "kv2" \
    NUM_LAYERS=10 NUM_KV_HEADS=2 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

# LR fine-tuning around 0.02
run "lr_015" \
    NUM_LAYERS=10 MATRIX_LR=0.015 SCALAR_LR=0.015 TIED_EMBED_LR=0.025

run "lr_025" \
    NUM_LAYERS=10 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035

# Vocab size
run "vocab_2048" \
    NUM_LAYERS=10 VOCAB_SIZE=2048 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

# RoPE base
run "rope_50k" \
    NUM_LAYERS=10 ROPE_BASE=50000 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

# Logit softcap
run "softcap_50" \
    NUM_LAYERS=10 LOGIT_SOFTCAP=50.0 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03

echo ""
echo "=== Sweep complete. Results in atris/logs/sweep_v2.log ==="
echo "Compare val_bpb across runs to find best config for v2."
