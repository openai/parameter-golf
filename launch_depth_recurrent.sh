#!/bin/bash
# Launch depth-recurrent training with optimal settings
# Based on PR #1331 analysis + architecture search (#36) + novel ideas #22-37
#
# Best configs (from architecture search):
#   8L 3xMLP loop[2:5]x8: 19.4M params, 32 eff depth, 14.68 MB int6 — RECOMMENDED
#   8L 2xMLP loop[2:5]x8: 15.2M params, 32 eff depth, 11.52 MB int6 — safe fallback
#   11L 2xMLP loop[3:6]x2: 20.7M params, 17 eff depth, 15.67 MB int6 — PR #1331 style
#
# Usage:
#   GPU 0: CUDA_VISIBLE_DEVICES=0 bash launch_depth_recurrent.sh
#   GPU 1: CUDA_VISIBLE_DEVICES=1 bash launch_depth_recurrent.sh

# Config: 8L 3xMLP + loop3x8 (best quality/MB from architecture search)
# At V=4096: 21.0M params, 15.87 MB int6 (130KB headroom!)
# At V=1024: 19.4M params, 14.68 MB int6 (1.3MB headroom)
# WARNING: 11L configs DON'T FIT with V=4096!
export N_LAYERS=8
export MLP_MULT=3
export LOOP_START=2
export LOOP_END=5
export LOOP_ITERS=8
export RECUR_STEP=3000      # Activate recurrence at step 3000 (PR #1331)
export RECUR_WARMUP=20      # 20-step warmup for recurrence gates

# Training settings (from PR #1331 + our research + novel #43: longer warmdown)
export STEPS=50000
export WD=0.095             # Weight decay (PR #1331: 0.095)
export QAT_START=0.15       # Int6 QAT activates when LR frac < 0.15
export BYTE_WEIGHTED=1      # Focus on high-byte tokens (novel #25)
export FOCAL_GAMMA=0.0      # Focal loss (0=off, try 1.0 if needed)
# Novel #43: 30% warmdown is theoretically better than 20% (more settling time)
# Override in train_depth_recurrent.py by setting WARMDOWN_FRAC env var
export WARMDOWN_FRAC=0.30

# Vocab: SP4096 (all top PRs use it — ~28% fewer tokens = lower BPB)
export VOCAB_SIZE=4096

echo "=== Depth-Recurrent Training ==="
echo "Config: ${N_LAYERS}L ${MLP_MULT}xMLP, loop[${LOOP_START}:${LOOP_END}]x${LOOP_ITERS}"
echo "Effective depth: $((N_LAYERS + (LOOP_END - LOOP_START) * LOOP_ITERS))"
echo "Vocab: SP${VOCAB_SIZE}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

# Check if SP4096 data exists
if [ ! -d "data/datasets/fineweb10B_sp4096" ] || [ $(ls data/datasets/fineweb10B_sp4096/fineweb_train_*.bin 2>/dev/null | wc -l) -lt 10 ]; then
    echo "WARNING: SP4096 data not ready. Falling back to SP1024."
    export VOCAB_SIZE=1024
fi

python train_depth_recurrent.py 2>&1 | tee train_depth_recurrent_$(date +%Y%m%d_%H%M).log
