#!/bin/bash
# ==============================================================================
# Parameter Golf — Experiment Runner
# Usage: bash run_experiment.sh [experiment_name]
# Example: bash run_experiment.sh exp1_more_layers
#
# Edit the variables below to change what you're testing.
# After the run, compare the final val_bpb to your baseline (mlx_smoke = 2.3555)
# ==============================================================================

# ---- PICK A NAME FOR THIS RUN (no spaces) ----
EXPERIMENT_NAME=${1:-"my_experiment"}

# ---- TRAINING SETTINGS ----
# Increase ITERATIONS for longer (better) training. 500 is fast on Mac (~5 min).
# For a serious Mac run, try 1000. For a GPU run, use 10000+.
ITERATIONS=500

# ---- MODEL ARCHITECTURE (these are the knobs you'll tweak) ----
# Baseline: NUM_LAYERS=9, MODEL_DIM=512, NUM_HEADS=8, MLP_MULT=2
# Try changing ONE at a time and see if val_bpb improves!

NUM_LAYERS=10        # 10 layers — best legal score so far, now adding MLP3+SwiGLU
MODEL_DIM=512        # Was 512. Try 640 or 768 for a wider model.
NUM_HEADS=8          # Keep divisible into MODEL_DIM (MODEL_DIM / NUM_HEADS = head size)
NUM_KV_HEADS=4       # Keep <= NUM_HEADS
MLP_MULT=3           # 3x MLP — more capacity in the feed-forward network

# SwiGLU: Gated activation used in Llama/PaLM. Better than relu^2 per parameter.
# Set to 1 to enable, 0 to use the original relu^2 baseline.
USE_SWIGLU=1

# ---- DO NOT EDIT BELOW THIS LINE ----
echo "======================================================"
echo "  Starting Experiment: $EXPERIMENT_NAME"
echo "  NUM_LAYERS=$NUM_LAYERS | MODEL_DIM=$MODEL_DIM"
echo "  NUM_HEADS=$NUM_HEADS | MLP_MULT=$MLP_MULT | USE_SWIGLU=$USE_SWIGLU"
echo "  ITERATIONS=$ITERATIONS"
echo "======================================================"

RUN_ID="$EXPERIMENT_NAME" \
ITERATIONS=$ITERATIONS \
NUM_LAYERS=$NUM_LAYERS \
MODEL_DIM=$MODEL_DIM \
NUM_HEADS=$NUM_HEADS \
NUM_KV_HEADS=$NUM_KV_HEADS \
MLP_MULT=$MLP_MULT \
USE_SWIGLU=$USE_SWIGLU \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py 2>&1 | tee "logs/${EXPERIMENT_NAME}.log"

echo ""
echo "======================================================"
echo "  DONE! Log saved to: logs/${EXPERIMENT_NAME}.log"
echo "  Compare your final val_bpb to baseline: 2.3555"
echo "  Check compressed size stays under 16,000,000 bytes"
echo "======================================================"
