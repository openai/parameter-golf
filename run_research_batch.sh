#!/bin/bash
set -euo pipefail

# Research batch — 4 tests on 2xGPU
# Run as: bash run_research_batch.sh [1|2]
#   Batch 1: tests A and B
#   Batch 2: tests C and D
# Each test takes ~600s training + ~100s eval ≈ 12 min
# Total per batch: ~25 min

cd /workspace/parameter-golf
export PYTHONPATH="/workspace/parameter-golf/flash-attention/hopper:${PYTHONPATH:-}"
mkdir -p logs/research

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

BATCH="${1:-1}"
NPROC="${NPROC:-2}"

run_test() {
    local NAME="$1"
    shift
    local LOGFILE="logs/research/${NAME}.log"
    echo ""
    echo "========== TEST: $NAME =========="
    env "$@" SEED=1337 \
        torchrun --standalone --nproc_per_node="$NPROC" \
        train_gpt_v7.py 2>&1 | tee "$LOGFILE"
    echo ""
    echo "--- $NAME results ---"
    grep -oP "(DIAGNOSTIC|final_int6_sliding|final_int6_roundtrip|Total submission size int6).*" "$LOGFILE" 2>/dev/null || true
    echo "====================="
}

if [ "$BATCH" = "1" ]; then
    echo "=== BATCH 1: Bigram size + GPTQ percdamp ==="

    # TEST A: Bigram 1536 + XSA-11 (size fit test)
    # Does reducing bigrams to 1536 fit under 16MB with XSA-11?
    run_test "A_bigram1536_xsa11" \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=1536 \
        INT8_SENSITIVE="" \
        TTT_EVAL_ENABLED=0

    # TEST B: Bigram 1024 + XSA-11 (aggressive size reduction)
    # How much quality do we lose with half the bigram table?
    run_test "B_bigram1024_xsa11" \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=1024 \
        INT8_SENSITIVE="" \
        TTT_EVAL_ENABLED=0

elif [ "$BATCH" = "2" ]; then
    echo "=== BATCH 2: GPTQ tuning ==="

    # TEST C: GPTQ percdamp=0.05 (more damping = more conservative = better compression?)
    # Default is 0.01. Higher percdamp regularizes the Hessian inverse,
    # producing less extreme error corrections = potentially more compressible values
    run_test "C_percdamp005_xsa11" \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=1536 \
        GPTQ_PERCDAMP=0.05 \
        INT8_SENSITIVE="" \
        TTT_EVAL_ENABLED=0

    # TEST D: GPTQ block_size=64 (smaller blocks = less error accumulation)
    # Default is 128. Smaller blocks limit how far errors propagate,
    # might produce more compressible values at slight quality cost
    run_test "D_block64_xsa11" \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=1536 \
        GPTQ_BLOCK_SIZE=64 \
        INT8_SENSITIVE="" \
        TTT_EVAL_ENABLED=0

else
    echo "Usage: bash run_research_batch.sh [1|2]"
    exit 1
fi

echo ""
echo "=== BATCH $BATCH COMPLETE ==="
echo "Results in logs/research/"
