#!/bin/bash
# Hyperparameter sweep — run overnight on 3060
# Each run: 2000 steps, batch 8K, no TTT

export ITERATIONS=2000
export TRAIN_BATCH_TOKENS=8192
export VAL_LOSS_EVERY=0
export VAL_BATCH_SIZE=8192
export MAX_WALLCLOCK_SECONDS=0
export TTT_STEPS=0

echo "=== Starting sweep at $(date) ==="

# 1. Baseline (current defaults: matrix_lr=0.04, embed_lr=0.05, scalar_lr=0.04)
echo "--- Run 1: baseline ---"
RUN_ID=sweep_baseline torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | grep -E "(model_params|step:2000|final_int8_zlib_roundtrip_exact)"

# 2. All lr x1.5
echo "--- Run 2: lr x1.5 ---"
RUN_ID=sweep_lr15 MATRIX_LR=0.06 TIED_EMBED_LR=0.075 SCALAR_LR=0.06 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | grep -E "(step:2000|final_int8_zlib_roundtrip_exact)"

# 3. All lr x2.0
echo "--- Run 3: lr x2.0 ---"
RUN_ID=sweep_lr20 MATRIX_LR=0.08 TIED_EMBED_LR=0.1 SCALAR_LR=0.08 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | grep -E "(step:2000|final_int8_zlib_roundtrip_exact)"

# 4. All lr x0.5
echo "--- Run 4: lr x0.5 ---"
RUN_ID=sweep_lr05 MATRIX_LR=0.02 TIED_EMBED_LR=0.025 SCALAR_LR=0.02 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | grep -E "(step:2000|final_int8_zlib_roundtrip_exact)"

# 5. Lower embed_lr ratio (embed_lr = 0.3x matrix_lr)
echo "--- Run 5: low embed_lr ---"
RUN_ID=sweep_lowemb TIED_EMBED_LR=0.012 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | grep -E "(step:2000|final_int8_zlib_roundtrip_exact)"

# 6. Longer warmdown (2400 iters)
echo "--- Run 6: warmdown_iters=2400 ---"
RUN_ID=sweep_wd2400 WARMDOWN_ITERS=2400 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | grep -E "(step:2000|final_int8_zlib_roundtrip_exact)"

# 7. Higher muon momentum
echo "--- Run 7: muon_momentum=0.98 ---"
RUN_ID=sweep_mom98 MUON_MOMENTUM=0.98 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | grep -E "(step:2000|final_int8_zlib_roundtrip_exact)"

# 8. Matrix lr x1.5 + lower embed
echo "--- Run 8: matrix_lr=0.06 + embed_lr=0.02 ---"
RUN_ID=sweep_combo MATRIX_LR=0.06 TIED_EMBED_LR=0.02 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | grep -E "(step:2000|final_int8_zlib_roundtrip_exact)"

echo "=== Sweep done at $(date) ==="
