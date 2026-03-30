#!/usr/bin/env bash
# ============================================================
# Session 05c-plus SMOKE TEST
# Run on ANY single Pegasus GPU before committing $25 RunPod.
# ============================================================
#
# What this validates:
#   1. Script boots, torch.compile succeeds
#   2. Forward/backward pass completes (training + EMA + export)
#   3. VE128 params exist and changed from their init values (parameter update proof)
#   4. Artifact size under 16MB cap
#   5. Roundtrip eval runs without shape mismatches
#
# What this does NOT validate:
#   - BPB quality (meaningless at 1xGPU with 50 steps)
#   - Multi-GPU DDP correctness (need 2+ GPUs)
#   - Exact step timing (different GPU type)
#   - Sliding window eval (skipped to keep smoke fast on slow GPUs)
#   - Correct optimizer group assignment (would need loss-contribution test)
#
# Usage (from repo root on Pegasus):
#   srun -K -p <PARTITION> --nodes=1 --ntasks=1 --gpus-per-task=1 \
#     --cpus-per-task=6 --mem=80G --time=00:10:00 \
#     --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
#     --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
#     bash records/track_non_record_16mb/2026-03-30_training_bundle_plus/smoke_test.sh
#
#   Pegasus partition names:
#     -p H100        # best match for final run
#     -p A100-80GB   # fallback
#     -p A100-40GB   # fallback
#     -p RTXA6000    # if nothing else available
#
# ============================================================

# No set -e: we want to run all checks and report, not bail on first failure.
set -uo pipefail

SMOKE_DIR="$(mktemp -d /tmp/smoke_05c_XXXXXX)"
echo "=========================================="
echo " 05c-plus SMOKE TEST"
echo "=========================================="
echo "Date:      $(date -Iseconds)"
echo "Host:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Smoke dir: $SMOKE_DIR (kept on failure, cleaned on success)"
echo "=========================================="

# ---------- DEPS ----------
export PYTHONUNBUFFERED=1
pip install --no-cache-dir sentencepiece zstandard 2>/dev/null

# ---------- 1xGPU SMOKE CONFIG ----------
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export ITERATIONS=50
export MAX_WALLCLOCK_SECONDS=120
export VAL_LOSS_EVERY=25
export TRAIN_LOG_EVERY=10
export EVAL_STRIDE=0
export SMOKE_SAVE_VE_INIT=1

SCRIPT=records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
OUTPUT="$SMOKE_DIR/smoke_output.txt"

# ---------- CLEAN STALE ARTIFACTS ----------
# Delete any leftover files from prior runs so Phase 3/4 cannot read stale data.
for f in final_model.pt final_model.int6.ptz ve_init_snapshot.pt; do
    if [ -f "$f" ]; then
        echo "Removing stale artifact: $f"
        rm -f "$f"
    fi
done

FAIL=0
check() {
    local name="$1" result="$2"
    if [ "$result" = "1" ]; then
        echo "PASS: $name"
    else
        echo "FAIL: $name"
        FAIL=1
    fi
}

# ---------- Phase 1: Syntax ----------
echo ""
echo ">>> Phase 1: Syntax check"
if python3 -m py_compile "$SCRIPT"; then
    echo "PASS: py_compile"
else
    echo "FAIL: py_compile"
    FAIL=1
fi

# ---------- Phase 2: Training + Export ----------
echo ""
echo ">>> Phase 2: Training + Export (50 steps, 1xGPU)"
if python -u "$SCRIPT" 2>&1 | tee "$OUTPUT"; then
    echo ""
    echo "PASS: Training + export completed"
else
    echo ""
    echo "FAIL: Training script exited with error (exit code $?)"
    FAIL=1
    # Copy output for debugging before any cleanup
    cp "$OUTPUT" /tmp/smoke_05c_failed_output.txt 2>/dev/null || true
    echo "Output saved to: /tmp/smoke_05c_failed_output.txt"
fi

# ---------- Phase 3: Post-run checks ----------
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
    check "Under 16MB cap ($TOTAL <= 16000000)" "$([ "$TOTAL" -le 16000000 ] && echo 1 || echo 0)"
else
    check "Artifact file exists" "0"
fi

# Check 2: Key log lines present (features active)
echo ""
if [ -f "$OUTPUT" ]; then
    for pattern in "model_params:" "anchor:05c_plus" "ve=128" "leaky_relu_sq=0.5" \
                   "pre_quant_ema" "final_int6_roundtrip"; do
        check "Log contains '$pattern'" "$(grep -q "$pattern" "$OUTPUT" && echo 1 || echo 0)"
    done
else
    check "Output file exists" "0"
fi

# Check 3: Parameter count in expected range
if [ -f "$OUTPUT" ]; then
    PARAM_COUNT=$(grep "model_params:" "$OUTPUT" | grep -oP '\d+' || echo "0")
    echo ""
    echo "Parameter count: $PARAM_COUNT"
    check "Param count in [26M, 28M]" "$([ "$PARAM_COUNT" -gt 26000000 ] && [ "$PARAM_COUNT" -lt 28000000 ] && echo 1 || echo 0)"
fi

# Check 4: No NaN/Inf in training loss
if [ -f "$OUTPUT" ]; then
    check "No NaN/Inf in training loss" "$(grep -qP 'train_loss:(nan|inf|-inf)' "$OUTPUT" && echo 0 || echo 1)"
fi

# Check 5: Roundtrip eval produced a valid number
if [ -f "$OUTPUT" ]; then
    check "Roundtrip eval produced valid loss" "$(grep -qP 'final_int6_roundtrip val_loss:\d' "$OUTPUT" && echo 1 || echo 0)"
fi

# ---------- Phase 4: VE128 weight-change validation ----------
# Compares post-training checkpoint against pre-training init snapshot.
# If VE weights are identical to init, they didn't receive gradient updates.
echo ""
echo ">>> Phase 4: VE128 weight-change validation"
if [ -f final_model.pt ] && [ -f ve_init_snapshot.pt ]; then
    python3 -c "
import torch, sys

init = torch.load('ve_init_snapshot.pt', map_location='cpu')
final = torch.load('final_model.pt', map_location='cpu')

ve_keys = sorted(init.keys())
if not ve_keys:
    print('FAIL: No VE keys in init snapshot')
    sys.exit(1)

all_ok = True
for k in ve_keys:
    if k not in final:
        print(f'FAIL: {k} missing from final checkpoint')
        all_ok = False
        continue
    t_init = init[k].float()
    t_final = final[k].float()
    delta = (t_final - t_init).norm().item()
    init_norm = t_init.norm().item()
    print(f'  {k}: init_norm={init_norm:.6f} delta_norm={delta:.6f}')
    if delta < 1e-8:
        print(f'    FAIL: {k} unchanged from init — no gradient flow')
        all_ok = False
    else:
        print(f'    PASS: weight changed (delta/init = {delta/max(init_norm,1e-10):.4f})')

if all_ok:
    print('PASS: All VE parameters received gradient updates')
else:
    print('FAIL: Some VE parameters did not change from init')
    sys.exit(1)
" 2>&1 | tee -a "$OUTPUT"
    VE_EXIT=$?
    check "VE128 weights changed from init" "$([ $VE_EXIT -eq 0 ] && echo 1 || echo 0)"
else
    if [ ! -f final_model.pt ]; then
        check "final_model.pt exists for VE check" "0"
    fi
    if [ ! -f ve_init_snapshot.pt ]; then
        check "ve_init_snapshot.pt exists for VE check" "0"
    fi
fi

# ---------- CLEANUP ----------
rm -f final_model.pt final_model.int6.ptz ve_init_snapshot.pt

echo ""
if [ "$FAIL" -eq 0 ]; then
    # Clean up tmpdir only on success
    rm -rf "$SMOKE_DIR"
    echo "=========================================="
    echo " ALL SMOKE CHECKS PASSED"
    echo "=========================================="
    echo ""
    echo "Next: Run full 8xH100 with confidence."
    echo "See README.md for launch command."
else
    echo "=========================================="
    echo " SMOKE FAILED — see above for details"
    echo "=========================================="
    echo "Output preserved at: $OUTPUT"
    exit 1
fi
