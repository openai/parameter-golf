#!/bin/bash
# TTT variant sweep on PR #1493's quantized model
# Each run loads fresh from final_model.int6.ptz — no retraining
#
# Estimated time: ~6 min per run, ~20 runs = ~2 hours total
# Results logged to logs/ttt_sweep.txt and stdout
set -e

LOGFILE="/workspace/parameter-golf/logs/ttt_sweep.txt"
echo "========================================" | tee -a "$LOGFILE"
echo "  TTT Variant Sweep — $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

run_variant() {
    local mode="$1"
    local lr="$2"
    local epochs="$3"
    local extra_env="$4"
    local label="${mode}_lr${lr}_ep${epochs}"
    if [ -n "$extra_env" ]; then
        label="${label}_adaptive"
    fi

    echo "" | tee -a "$LOGFILE"
    echo "--- $label ---" | tee -a "$LOGFILE"

    # Clear dynamo cache between runs
    rm -rf ~/.cache/torch_extensions 2>/dev/null || true

    env TTT_PARAM_MODE="$mode" TTT_LR="$lr" TTT_EPOCHS="$epochs" \
        TTT_ENABLED=1 TTT_MOMENTUM=0.9 TTT_CHUNK_TOKENS=32768 \
        EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 RUN_ID="ttt_${label}" \
        QK_GAIN_INIT=5.25 SEED=42 \
        $extra_env \
        torchrun --standalone --nproc_per_node=8 eval_ttt_variants.py 2>&1 \
        | grep -E "RESULT|ttt_variant|ttt_adaptive|val_bpb|error|Error|Traceback" \
        | tee -a "$LOGFILE"
}

# =============================================
# Phase 1: Baseline reproduction + LR sweep
# =============================================
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 1: all params, LR sweep ===" | tee -a "$LOGFILE"

run_variant "all" "0.003" "3"
run_variant "all" "0.005" "3"    # baseline (should match 1.0810)
run_variant "all" "0.007" "3"

# Epoch sweep at best LR (0.005 baseline)
run_variant "all" "0.005" "2"
run_variant "all" "0.005" "4"

# =============================================
# Phase 2: late_blocks (layers 7-10)
# Research says last 25% of blocks is optimal
# =============================================
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 2: late_blocks (L7-10) ===" | tee -a "$LOGFILE"

run_variant "late_blocks" "0.005" "3"
run_variant "late_blocks" "0.01" "3"
run_variant "late_blocks" "0.005" "4"
run_variant "late_blocks" "0.01" "4"

# =============================================
# Phase 3: no_embeddings
# Literature strongly warns against updating embeddings
# =============================================
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 3: no_embeddings ===" | tee -a "$LOGFILE"

run_variant "no_embeddings" "0.005" "3"
run_variant "no_embeddings" "0.003" "3"

# =============================================
# Phase 4: loop_blocks (L3-5)
# Risky — these params are used 3x in virtual depth
# =============================================
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 4: loop_blocks (L3-5) ===" | tee -a "$LOGFILE"

run_variant "loop_blocks" "0.002" "3"
run_variant "loop_blocks" "0.005" "3"

# =============================================
# Phase 5: control_only (scales, gains, skip, resid_mix)
# Very few params — needs higher LR
# =============================================
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 5: control_only ===" | tee -a "$LOGFILE"

run_variant "control_only" "0.02" "3"
run_variant "control_only" "0.05" "3"
run_variant "control_only" "0.02" "5"

# =============================================
# Phase 6: norm_scale_qgain_skip (control minus resid_mix)
# =============================================
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 6: norm_scale_qgain_skip ===" | tee -a "$LOGFILE"

run_variant "norm_scale_qgain_skip" "0.02" "3"
run_variant "norm_scale_qgain_skip" "0.05" "3"

# =============================================
# Phase 7: Entropy-adaptive TTT (best mode from above)
# Using "all" mode as default; re-run with winner later
# =============================================
echo "" | tee -a "$LOGFILE"
echo "=== PHASE 7: Entropy-adaptive ===" | tee -a "$LOGFILE"

run_variant "all" "0.005" "3" \
    "TTT_ADAPTIVE=1 TTT_ADAPTIVE_EASY_NLL=2.5 TTT_ADAPTIVE_HARD_NLL=3.0 TTT_ADAPTIVE_EASY_EP=1 TTT_ADAPTIVE_HARD_EP=5"

run_variant "no_embeddings" "0.005" "3" \
    "TTT_ADAPTIVE=1 TTT_ADAPTIVE_EASY_NLL=2.5 TTT_ADAPTIVE_HARD_NLL=3.0 TTT_ADAPTIVE_EASY_EP=1 TTT_ADAPTIVE_HARD_EP=5"

echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "  Sweep complete — $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
