#!/bin/bash
set -euo pipefail

# Frugendorff SwiGLU sweep — find optimal sharing config
# Run as: bash run_frug_sweep.sh [1|2|3]
#   Batch 1: Size frontier (find what fits 16MB)
#   Batch 2: Quality frontier (depth vs sharing tradeoffs)
#   Batch 3: Compression levers (bigram, MLP tuning)
#
# Each test: ~600s train + ~150s eval ≈ 13 min on 8xGPU, ~25 min on 2xGPU
# 2xGPU: NPROC=2 bash run_frug_sweep.sh 1

cd /workspace/parameter-golf
export PYTHONPATH="/workspace/parameter-golf/flash-attention/hopper:${PYTHONPATH:-}"
mkdir -p logs/frug_sweep

python3 -c "import zstandard; print('deps OK')"

BATCH="${1:-1}"
NPROC="${NPROC:-8}"
SEED="${SEED:-42}"

run_test() {
    local NAME="$1"
    shift
    local LOGFILE="logs/frug_sweep/${NAME}.log"
    echo ""
    echo "========== TEST: $NAME =========="
    echo "  Config: $@"
    env "$@" SEED="$SEED" \
        torchrun --standalone --nproc_per_node="$NPROC" \
        train_gpt_swiglu_frugendorff.py 2>&1 | tee "$LOGFILE"
    echo ""
    echo "--- $NAME results ---"
    grep -oP "(model_params|Total submission size int6|final_int6_zstd.*roundtrip_exact|Serialized model int6).*" "$LOGFILE" 2>/dev/null || true
    echo "====================="
}

if [ "$BATCH" = "1" ]; then
    echo "=== BATCH 1: Size frontier — what fits 16MB? ==="
    echo "Known: 11L/SHARE4/LOOPS3 = 27.6M params, 16.68MB ❌ (over by 680KB)"

    # Baseline we know works on SwiGLU: no sharing, naive int6 = ~15.7MB
    # Our GPTQ adds ~3MB. Sharing reduces unique params to compensate.

    # F1: More sharing — loop 4x instead of 3x (fewer unique params)
    run_test "F1_11L_share4_loop4" \
        NUM_LAYERS=11 SHARE_START=4 SHARE_LOOPS=4

    # F2: Earlier sharing — start at layer 3 (more layers shared)
    run_test "F2_11L_share3_loop3" \
        NUM_LAYERS=11 SHARE_START=3 SHARE_LOOPS=3

    # F3: Fewer layers — 9 unique with 3x loop
    run_test "F3_9L_share3_loop3" \
        NUM_LAYERS=9 SHARE_START=3 SHARE_LOOPS=3

elif [ "$BATCH" = "2" ]; then
    echo "=== BATCH 2: Quality frontier — maximize BPB within 16MB ==="

    # F4: 10 layers, share from 4, loop 3x (eff depth = 12)
    run_test "F4_10L_share4_loop3" \
        NUM_LAYERS=10 SHARE_START=4 SHARE_LOOPS=3

    # F5: 10 layers, share from 3, loop 4x (eff depth = 13, fewer unique)
    run_test "F5_10L_share3_loop4" \
        NUM_LAYERS=10 SHARE_START=3 SHARE_LOOPS=4

    # F6: 11 layers, share from 3, loop 4x (max depth = 14, aggressive sharing)
    run_test "F6_11L_share3_loop4" \
        NUM_LAYERS=11 SHARE_START=3 SHARE_LOOPS=4

elif [ "$BATCH" = "3" ]; then
    echo "=== BATCH 3: Compression levers on best config ==="
    # Run after batch 1+2 identify the best config
    # Replace NUM_LAYERS/SHARE_START/SHARE_LOOPS with winner

    # F7: Halve bigram buckets (saves ~0.5MB compressed)
    run_test "F7_best_bigram4096" \
        NUM_LAYERS=11 SHARE_START=4 SHARE_LOOPS=4 \
        BIGRAM_BUCKETS=4096

    # F8: Smaller MLP (1536 instead of 1792)
    run_test "F8_best_mlp1536" \
        NUM_LAYERS=11 SHARE_START=4 SHARE_LOOPS=4 \
        MLP_HIDDEN=1536

    # F9: Both compression levers
    run_test "F9_best_both" \
        NUM_LAYERS=11 SHARE_START=4 SHARE_LOOPS=4 \
        BIGRAM_BUCKETS=4096 MLP_HIDDEN=1536

else
    echo "Usage: bash run_frug_sweep.sh [1|2|3]"
    exit 1
fi

echo ""
echo "=== BATCH $BATCH COMPLETE ==="
echo "Summary in logs/frug_sweep/"
echo ""
echo "Quick compare:"
for f in logs/frug_sweep/F*.log; do
    name=$(basename "$f" .log)
    params=$(grep -oP 'model_params:\K\d+' "$f" 2>/dev/null | head -1)
    size=$(grep -oP 'Total submission size int6\+zstd-22: \K\d+' "$f" 2>/dev/null | head -1)
    bpb=$(grep -oP 'final_int6_zstd-22_roundtrip_exact val_bpb:\K\S+' "$f" 2>/dev/null | head -1)
    printf "  %-25s params=%-10s size=%-12s bpb=%s\n" "$name" "${params:-?}" "${size:-?}" "${bpb:-?}"
done
