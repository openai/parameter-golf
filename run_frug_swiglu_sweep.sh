#!/bin/bash
set -euo pipefail

# Frugendorff SwiGLU compression sweep — find optimal loops/sharing
# The question: how many loops before quality falls off vs size saved?
# Run: NPROC=1 bash run_frug_swiglu_sweep.sh
# ~12 min per test, 6 tests = ~72 min total

cd /workspace/parameter-golf
mkdir -p logs/frug_swiglu_sweep

python3 -c "import zstandard; print('deps OK')"

NPROC="${NPROC:-1}"
SEED="${SEED:-42}"

run_test() {
    local NAME="$1"
    shift
    local LOGFILE="logs/frug_swiglu_sweep/${NAME}.log"
    echo ""
    echo "=========================================="
    echo "  TEST: $NAME"
    echo "  Config: $@"
    echo "=========================================="
    env "$@" SEED="$SEED" \
        torchrun --standalone --nproc_per_node="$NPROC" \
        train_gpt_swiglu_frugendorff.py 2>&1 | tee "$LOGFILE"
    echo ""
    echo "--- $NAME results ---"
    grep -oP "(model_params|Serialized model int6|Total submission size int6|final_int6.*roundtrip_exact|payload_ratio).*" "$LOGFILE" 2>/dev/null || true
    echo "====================="
}

echo "=== FRUGENDORFF SWIGLU COMPRESSION SWEEP ==="
echo "=== 6 tests: loop count, share position, compression levers ==="
echo ""

# --- CORE SWEEP: how many loops? ---
# Hold: 11L, SHARE_START=4. Vary: SHARE_LOOPS 3,4,5

# S1: Baseline (known: 27.6M, 16.68MB, over limit)
run_test "S1_loops3_baseline" \
    NUM_LAYERS=11 SHARE_START=4 SHARE_LOOPS=3

# S2: One more loop (est: ~24.7M, ~14.9MB, should fit with margin)
run_test "S2_loops4" \
    NUM_LAYERS=11 SHARE_START=4 SHARE_LOOPS=4

# S3: Aggressive (est: ~21.8M, ~13.2MB, lots of margin)
run_test "S3_loops5" \
    NUM_LAYERS=11 SHARE_START=4 SHARE_LOOPS=5

# --- SHARE POSITION: does sharing earlier vs later matter? ---

# S4: Share from layer 3 instead of 4 (same param count as S2 but different layers shared)
run_test "S4_share3_loops4" \
    NUM_LAYERS=11 SHARE_START=3 SHARE_LOOPS=4

# --- COMPRESSION LEVERS on the likely winner (loops=4) ---

# S5: loops=4 + half bigrams (if S2 is tight on size)
run_test "S5_loops4_bigram4096" \
    NUM_LAYERS=11 SHARE_START=4 SHARE_LOOPS=4 BIGRAM_BUCKETS=4096

# S6: loops=4 + smaller MLP (quality vs size tradeoff)
run_test "S6_loops4_mlp1536" \
    NUM_LAYERS=11 SHARE_START=4 SHARE_LOOPS=4 MLP_HIDDEN=1536

echo ""
echo "=== SWEEP COMPLETE ==="
echo ""
echo "Summary:"
printf "  %-25s %12s %15s %15s\n" "Test" "Params" "Size (bytes)" "BPB"
echo "  -----------------------------------------------------------------------"
for f in logs/frug_swiglu_sweep/S*.log; do
    name=$(basename "$f" .log)
    params=$(grep -oP 'model_params:\K\d+' "$f" 2>/dev/null | head -1)
    size=$(grep -oP 'Total submission size int6\+zstd-22: \K\d+' "$f" 2>/dev/null | head -1)
    bpb=$(grep -oP 'final_int6_zstd-22_roundtrip_exact val_bpb:\K\S+' "$f" 2>/dev/null | head -1)
    printf "  %-25s %12s %15s %15s\n" "$name" "${params:-?}" "${size:-?}" "${bpb:-?}"
done
echo ""
echo "16MB limit = 16000000 bytes. Pick the config with best BPB that fits."
