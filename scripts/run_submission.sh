#!/bin/bash
# Run 3-seed submission on 8xH100 SXM.
# Usage: bash scripts/run_submission.sh
set -e
cd /workspace/parameter-golf

SUBMISSION_DIR="records/track_10min_16mb/2026-04-06_CorruptedContext_R2-11"
SCRIPT="${SUBMISSION_DIR}/train_gpt.py"
NGPU=8

echo "=========================================="
echo "SUBMISSION RUN: Corrupted Context R2-11"
echo "Script: ${SCRIPT}"
echo "GPUs: ${NGPU}"
echo "Start: $(date)"
echo "=========================================="

for SEED in 42 137 256; do
    echo ""
    echo "=========================================="
    echo "SEED ${SEED} — $(date)"
    echo "=========================================="

    RUN_ID="submission_seed${SEED}" \
    SEED=${SEED} \
    torchrun --standalone --nproc_per_node=${NGPU} ${SCRIPT} \
        2>&1 | tee "${SUBMISSION_DIR}/train_seed${SEED}.log"

    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "FAILED: seed ${SEED} (exit code ${EXIT_CODE})"
    else
        echo "COMPLETED: seed ${SEED} — $(date)"
    fi

    # Extract key result
    RESULT=$(grep "final_int8_zlib_roundtrip_exact" "${SUBMISSION_DIR}/train_seed${SEED}.log" 2>/dev/null | tail -1)
    if [ -n "$RESULT" ]; then
        echo "RESULT seed ${SEED}: ${RESULT}"
    else
        echo "WARNING: No final result found for seed ${SEED}"
    fi
done

echo ""
echo "=========================================="
echo "ALL SEEDS COMPLETE — $(date)"
echo "=========================================="
echo ""
echo "RESULTS SUMMARY:"
for SEED in 42 137 256; do
    RESULT=$(grep "final_int8_zlib_roundtrip_exact" "${SUBMISSION_DIR}/train_seed${SEED}.log" 2>/dev/null | tail -1)
    echo "  Seed ${SEED}: ${RESULT:-FAILED}"
done

echo ""
echo "ARTIFACT SIZES:"
for SEED in 42 137 256; do
    SIZE=$(grep "Total submission size int8+zlib" "${SUBMISSION_DIR}/train_seed${SEED}.log" 2>/dev/null | tail -1)
    echo "  Seed ${SEED}: ${SIZE:-UNKNOWN}"
done
