#!/bin/bash
# =============================================================================
# ANS Design Space Sweep
#
# 5 runs on 8×H100. Maps the design space that only ANS can reach.
# Total time: ~1 hour. Cost: ~$25.
#
# Usage: bash run_sweep.sh [run_number]
#   run_sweep.sh 1  → SP1024 baseline + ANS (control)
#   run_sweep.sh 2  → SP1024 wider MLP + ANS
#   run_sweep.sh 3  → SP1024 deeper (11 layers) + ANS
#   run_sweep.sh 4  → SP1024 wider + deeper + ANS
#   run_sweep.sh 5  → SP1024 max params + ANS
#   run_sweep.sh all → run all 5
#
# We use SP1024 because it's the only cached tokenizer.
# SP4096/5120/6144 need custom tokenization (future work with more GPU time).
# =============================================================================

set -e

RUN=${1:-all}

# Setup
cd /workspace
if [ ! -d parameter-golf ]; then
    git clone https://github.com/OE-GOD/parameter-golf.git
    cd parameter-golf
    git checkout ans-compression
else
    cd parameter-golf
    git fetch origin 2>/dev/null || true
fi

# Install deps
pip install sentencepiece numpy tqdm huggingface_hub datasets brotli 2>/dev/null

# Download data if needed
if [ ! -f data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin ]; then
    echo "Downloading SP1024 data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 20
fi

# Get ANS compressor
cp records/track_non_record_16mb/2026-04-09_ANS_Compression/ans_compress.py . 2>/dev/null || true

# Helper function
run_experiment() {
    local NAME=$1
    local EXTRA_ARGS=$2

    echo ""
    echo "=========================================="
    echo "Run: $NAME"
    echo "Args: $EXTRA_ARGS"
    echo "=========================================="

    # Train
    eval "RUN_ID=$NAME $EXTRA_ARGS torchrun --standalone --nproc_per_node=8 train_gpt.py"

    # Convert to npz
    python3 -c "
import torch, numpy as np
state = torch.load('final_model.pt', map_location='cpu', weights_only=True)
np_state = {k: v.float().numpy() for k, v in state.items()}
np.savez('final_model_${NAME}.npz', **np_state)
print('Converted $NAME')
"

    # ANS analysis
    echo "--- ANS Analysis for $NAME ---"
    python3 ans_compress.py --input final_model_${NAME}.npz --analyze --bits 6

    # Log the quantized artifact size
    QUANT_FILE=$(ls -1 logs/${NAME}_quantized.pt 2>/dev/null || echo "")
    if [ -n "$QUANT_FILE" ]; then
        BROTLI_SZ=$(stat -c%s "$QUANT_FILE" 2>/dev/null || stat -f%z "$QUANT_FILE" 2>/dev/null)
        echo "Brotli artifact: $BROTLI_SZ bytes"
    fi

    # Get val_bpb from log
    grep "val_bpb\|val_loss" logs/${NAME}.txt | tail -3
    echo ""
}

# =============================================================================
# Run 1: Baseline SP1024 (control — matches what everyone else has)
# =============================================================================
if [ "$RUN" = "1" ] || [ "$RUN" = "all" ]; then
    run_experiment "sweep_baseline" ""
fi

# =============================================================================
# Run 2: Wider MLP (3x → 4x, uses the extra params ANS frees)
# This adds ~2M params. With ANS saving 1.8MB, we can afford 2.4M more at int6.
# =============================================================================
if [ "$RUN" = "2" ] || [ "$RUN" = "all" ]; then
    run_experiment "sweep_wide_mlp" "MLP_MULT=4"
fi

# =============================================================================
# Run 3: More layers (9 → 11)
# More depth = more representational power. Fits because ANS saves space.
# =============================================================================
if [ "$RUN" = "3" ] || [ "$RUN" = "all" ]; then
    run_experiment "sweep_deep" "NUM_LAYERS=11"
fi

# =============================================================================
# Run 4: Wider MLP + More layers
# Combining width and depth. This wouldn't fit under 16MB with Brotli.
# =============================================================================
if [ "$RUN" = "4" ] || [ "$RUN" = "all" ]; then
    run_experiment "sweep_wide_deep" "MLP_MULT=4 NUM_LAYERS=11"
fi

# =============================================================================
# Run 5: Maximum params that fit with ANS
# Push model dim wider too (512 → 576 or 640)
# =============================================================================
if [ "$RUN" = "5" ] || [ "$RUN" = "all" ]; then
    run_experiment "sweep_max" "MLP_MULT=4 NUM_LAYERS=11 MODEL_DIM=576"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "SWEEP RESULTS SUMMARY"
echo "=========================================="
echo ""
for name in sweep_baseline sweep_wide_mlp sweep_deep sweep_wide_deep sweep_max; do
    if [ -f logs/${name}.txt ]; then
        BPB=$(grep "final_int8_zlib_roundtrip_exact" logs/${name}.txt 2>/dev/null | grep -o "val_bpb:[0-9.]*" | cut -d: -f2)
        PARAMS=$(grep "model_params:" logs/${name}.txt | head -1 | grep -o "model_params:[0-9]*" | cut -d: -f2)
        echo "$name: BPB=$BPB params=$PARAMS"
    fi
done

echo ""
echo "Done. Pick the best BPB and do 3-seed validation."
echo "Next: RUN_ID=final SEED=42 torchrun ... && RUN_ID=final SEED=314 torchrun ... && RUN_ID=final SEED=999 torchrun ..."
