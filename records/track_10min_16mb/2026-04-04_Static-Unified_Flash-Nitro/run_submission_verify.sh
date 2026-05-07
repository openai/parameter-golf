#!/bin/bash
# Paradox-Golf Submission Verification Script 🏆
# Runs the optimized script 3 times to generate statistically significant logs for SOTA records.

export TORCH_COMPILE=1
export EVAL_STRIDE=512
export TTT_STEPS=1
export VAL_LOSS_EVERY=1300
export TRAIN_BATCH_TOKENS=1048576 
export GRAD_ACCUM_PHASE2=1
export QAT_ENABLED=0
export ITERATIONS=2600

# Ensure logs directory exists
mkdir -p logs

for i in {0..2}
do
   echo "==========================================="
   echo "Starting verification run $i..."
   echo "==========================================="
   torchrun --standalone --nproc_per_node=8 train_gpt.py > logs/train_v$i.txt 2>&1
   echo "Run $i finished. Log saved to logs/train_v$i.txt"
done

echo "Double-check your logs/ directory for all 3 files before submitting the PR!"
