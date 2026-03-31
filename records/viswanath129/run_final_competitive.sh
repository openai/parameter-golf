#!/bin/bash
# COMPETITIVE FINAL RUN
# TRUE entropy SENT-lite + loss-variance guided TTT
# Expected: ~1.13 BPB

set -euo pipefail

echo "════════════════════════════════════════════"
echo "  FINAL COMPETITIVE RUN"
echo "════════════════════════════════════════════"
echo ""

# Verify environment
nvidia-smi --query-gpu=count --format=csv,noheader | grep -q "8" || {
    echo "ERROR: Need 8 GPUs"
    exit 1
}

# Prepare data if needed
if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "Downloading FineWeb data..."
    python data/cached_challenge_fineweb.py --variant sp1024
fi

# Competitive config
export SEED=1337
export ITERATIONS=20000
export USE_SENT_LITE=1
export SENT_LITE_ALPHA=0.15  # Optimized alpha for entropy
export USE_TTT_LORA=1
export WARMDOWN_ITERS=1200
export MAX_WALLCLOCK_SECONDS=600

# Run final training
echo "Starting final competitive run..."
echo "Configuration:"
echo "  - TRUE entropy SENT-lite"
echo "  - Loss-variance guided TTT"
echo "  - Tight training for <1.14 BPB"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/final_competitive_${TIMESTAMP}.log"
mkdir -p logs

torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "════════════════════════════════════════════"
echo "  FINAL RUN COMPLETE"
echo "════════════════════════════════════════════"

# Extract final metrics
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "FINAL METRICS:"
    grep -i "final\|bpb\|loss" "$LOG_FILE" | tail -5
fi

# Check artifact
if [ -f "final_model.int8.ptz" ]; then
    SIZE=$(stat -c%s final_model.int8.ptz 2>/dev/null || stat -f%z final_model.int8.ptz)
    SIZE_MB=$((SIZE / 1048576))
    echo ""
    echo "Model: $SIZE_MB MB (limit: 16 MB)"
fi
