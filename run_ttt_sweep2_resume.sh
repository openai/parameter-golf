#!/bin/bash
set -e

LOGFILE="/workspace/parameter-golf/logs/ttt_sweep.txt"

run_variant() {
    local mode="$1"
    local lr="$2"
    local epochs="$3"
    local label="${mode}_lr${lr}_ep${epochs}"

    echo "" | tee -a "$LOGFILE"
    echo "--- $label ---" | tee -a "$LOGFILE"
    rm -rf ~/.cache/torch_extensions 2>/dev/null || true

    env TTT_PARAM_MODE="$mode" TTT_LR="$lr" TTT_EPOCHS="$epochs" \
        TTT_ENABLED=1 TTT_MOMENTUM=0.9 TTT_CHUNK_TOKENS=32768 \
        EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 RUN_ID="ttt_${label}" \
        QK_GAIN_INIT=5.25 SEED=42 \
        torchrun --standalone --nproc_per_node=8 eval_ttt_variants.py 2>&1 \
        | grep -E "RESULT|ttt_variant|ttt_adaptive|val_bpb|error|Error|Traceback" \
        | tee -a "$LOGFILE"
}

echo "" | tee -a "$LOGFILE"
echo "=== PHASE 8 (resumed): late_blocks_6 ===" | tee -a "$LOGFILE"
run_variant "late_blocks_6" "0.01" "3"
run_variant "late_blocks_6" "0.007" "4"

echo "" | tee -a "$LOGFILE"
echo "=== PHASE 9: mlp_proj_only (ByteDance In-Place TTT) ===" | tee -a "$LOGFILE"
run_variant "mlp_proj_only" "0.005" "3"
run_variant "mlp_proj_only" "0.007" "3"
run_variant "mlp_proj_only" "0.01" "3"
run_variant "mlp_proj_only" "0.005" "4"
run_variant "mlp_proj_only" "0.01" "4"

echo "" | tee -a "$LOGFILE"
echo "=== PHASE 10: all lr=0.007 ep=4/5 (combined best) ===" | tee -a "$LOGFILE"
run_variant "all" "0.007" "4"
run_variant "all" "0.007" "5"
run_variant "all" "0.01" "3"

echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "  Sweep 2 complete — $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
