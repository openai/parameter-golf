#!/bin/bash
# Full training on 8xH100 RunPod pod
# All v7 bugfixes applied on top of v3 baseline (NO N-gram for compliance)
# Goal: beat v3 (0.65802 BPB on 8xH100)

set -e
SEED=${SEED:-42}

cd /workspace

# Clone parameter-golf if needed
if [ ! -d "/workspace/pgolf" ]; then
    git clone https://github.com/openai/parameter-golf.git /workspace/pgolf
fi

cd /workspace/pgolf

# Prepare data
if [ ! -d "/workspace/pgolf/data/datasets/fineweb10B_sp1024" ]; then
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
fi

# Install flash-attn for speed
pip install flash-attn --no-build-isolation 2>&1 | tail -3 || echo "FA install failed, continuing"

# Copy our train_gpt.py
cp /workspace/train_gpt.py /workspace/pgolf/train_gpt.py

# Run training with v7 bugfixes but NO N-gram (compliance safe)
export SEED=${SEED}
export RUN_ID="trinity_v3_bugfixes_s${SEED}"

# TTT params (v3 stack, now with proper batch size)
export TTT_ENABLED=1 TTT_LR=0.001 TTT_EPOCHS=1
export TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=10 TTT_BATCH_SEQS=32

# SLOT params — PR #1430 aggressive (v7 bugfix: batch=128 works now!)
export SLOT_LR=0.432 SLOT_STEPS=24 SLOT_STRIDE=64
export SLOT_BETA1=0.6 SLOT_BETA2=0.5 SLOT_BATCH_SEQS=128
export SLOT_OPTIMIZER=adamw  # Lion was worse

# N-GRAM DISABLED (compliance)
export NGRAM_ENABLED=0

# Quantization: FP16 embed + per-row clip (v7 bugfixes, legal)
export EMBED_QUANT=fp16
export GPTQ_PER_ROW_CLIP=1
export GPTQ_DAMP_FACTOR=0.005 GPTQ_CALIB_VAL=1 GPTQ_CALIB_BATCHES=256

# Model params
export QK_GAIN_INIT=4.0 MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.1
export MAX_WALLCLOCK_SECONDS=600

# Count GPUs
NPROC=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Running on $NPROC GPUs, seed=$SEED"

# Train + TTT + SLOT eval
torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /workspace/result_seed${SEED}.log

# Extract final BPB
grep "final_slot_exact" /workspace/result_seed${SEED}.log | tail -1
echo "Training done for seed $SEED"
