#!/usr/bin/env bash
# ============================================================
# Session 05c-plus SMOKE TEST
# Run on ANY single Pegasus GPU before committing $25 RunPod.
# ============================================================
#
# What this validates:
#   1. Script boots, torch.compile succeeds
#   2. VE128 params exist, get gradients, optimizer grouping is correct
#   3. Forward/backward pass works (training + EMA)
#   4. Export path (int6 + zstd) produces valid artifact under 16MB
#   5. Roundtrip eval runs (no shape mismatches)
#   6. Sliding window eval runs
#   7. No leftover state from prior runs contaminates results
#
# What this does NOT validate:
#   - BPB quality (meaningless at 1xGPU with 50 steps)
#   - Multi-GPU DDP correctness (need 2+ GPUs for that)
#   - Exact step timing (different GPU type)
#
# Usage (from repo root on Pegasus):
#   # Option A: Any available GPU partition
#   srun -K -p <PARTITION> --nodes=1 --ntasks=1 --gpus-per-task=1 \
#     --cpus-per-task=6 --mem=80G --time=00:10:00 \
#     --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
#     --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
#     bash records/track_non_record_16mb/2026-03-30_training_bundle_plus/smoke_test.sh
#
#   # Option B: Specific partition examples
#   -p H100    # best match for final run
#   -p A100    # fallback, still validates everything
#   -p RTXA6000  # if nothing else available
#
# ============================================================

set -euo pipefail

echo "=========================================="
echo " 05c-plus SMOKE TEST"
echo "=========================================="
echo "Date: $(date -Iseconds)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=========================================="

# ---------- CLEAN STATE ----------
# Remove any leftover artifacts from prior runs
rm -f final_model.pt final_model.int6.ptz
rm -rf logs/

# ---------- DEPS ----------
export PYTHONUNBUFFERED=1
pip install --no-cache-dir sentencepiece zstandard 2>/dev/null

# ---------- 1xGPU SMOKE CONFIG ----------
# Override via env vars: short run, skip the long sliding eval
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export ITERATIONS=50
export MAX_WALLCLOCK_SECONDS=120
export VAL_LOSS_EVERY=25
export TRAIN_LOG_EVERY=10

SCRIPT=records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py

echo ""
echo ">>> Phase 1: Syntax check"
python3 -m py_compile "$SCRIPT"
echo "PASS: py_compile"

echo ""
echo ">>> Phase 2: Training + Export (50 steps, 1xGPU)"
python -u "$SCRIPT" 2>&1 | tee /tmp/smoke_output.txt
echo ""
echo "PASS: Training + export completed"

# ---------- POST-RUN CHECKS ----------
echo ""
echo ">>> Phase 3: Post-run validation"

# Check 1: Artifact exists and is under cap
if [ -f final_model.int6.ptz ]; then
    ARTIFACT_SIZE=$(stat -c%s final_model.int6.ptz)
    CODE_SIZE=$(stat -c%s "$SCRIPT")
    TOTAL=$((ARTIFACT_SIZE + CODE_SIZE))
    echo "Artifact: $ARTIFACT_SIZE bytes"
    echo "Code:     $CODE_SIZE bytes"
    echo "Total:    $TOTAL bytes"
    if [ "$TOTAL" -le 16000000 ]; then
        echo "PASS: Under 16MB cap ($TOTAL <= 16000000)"
    else
        echo "FAIL: Over 16MB cap ($TOTAL > 16000000)"
        exit 1
    fi
else
    echo "FAIL: final_model.int6.ptz not found"
    exit 1
fi

# Check 2: Key log lines present
echo ""
for pattern in "model_params:" "anchor:05c_plus" "ve=128" "leaky_relu_sq=0.5" \
               "pre_quant_ema" "final_int6_roundtrip" "final_int6_sliding_window"; do
    if grep -q "$pattern" /tmp/smoke_output.txt; then
        echo "PASS: Found '$pattern' in output"
    else
        echo "FAIL: Missing '$pattern' in output"
        exit 1
    fi
done

# Check 3: Parameter count is sane (should be ~27M with VE128)
PARAM_COUNT=$(grep "model_params:" /tmp/smoke_output.txt | grep -oP '\d+')
echo ""
echo "Parameter count: $PARAM_COUNT"
if [ "$PARAM_COUNT" -gt 26000000 ] && [ "$PARAM_COUNT" -lt 28000000 ]; then
    echo "PASS: Param count in expected range"
else
    echo "FAIL: Param count $PARAM_COUNT outside [26M, 28M]"
    exit 1
fi

# Check 4: No NaN/Inf in training loss
if grep -qP "train_loss:(nan|inf|-inf)" /tmp/smoke_output.txt; then
    echo "FAIL: NaN/Inf detected in training loss"
    exit 1
else
    echo "PASS: No NaN/Inf in training loss"
fi

# Check 5: Roundtrip eval produced a number (not crash)
if grep -qP "final_int6_roundtrip val_loss:\d" /tmp/smoke_output.txt; then
    echo "PASS: Roundtrip eval produced valid loss"
else
    echo "FAIL: Roundtrip eval did not produce valid loss"
    exit 1
fi

# Check 6: Sliding window eval produced a number
if grep -qP "final_int6_sliding_window val_loss:\d" /tmp/smoke_output.txt; then
    echo "PASS: Sliding window eval produced valid loss"
else
    echo "FAIL: Sliding window eval did not produce valid loss"
    exit 1
fi

# ---------- CLEANUP ----------
rm -f final_model.pt final_model.int6.ptz

echo ""
echo "=========================================="
echo " ALL SMOKE CHECKS PASSED"
echo "=========================================="
echo ""
echo "Next: Run full 8xH100 with confidence."
echo "See README.md for launch command."
