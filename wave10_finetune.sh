#!/bin/bash
# Wave 10: Fine-grained tuning around best config from Wave 9
# This script assumes 1/8 asymmetric is confirmed best.
# Fine-tune softcap and matrix LR with small increments.
cd /workspace/parameter-golf
LOG="/workspace/wave10_results.log"

echo "=== WAVE 10: FINE-TUNE $(date) ===" > $LOG
echo "Building on best from Wave 9" >> $LOG
echo "" >> $LOG

grab() {
    local name="$1"
    local logfile=$(ls -t /workspace/parameter-golf/logs/*.txt | head -1)
    local result=$(grep "^step:.*val_bpb" "$logfile" | tail -1)
    echo "$result" >> $LOG
    echo "END: $(date)" >> $LOG
    echo "" >> $LOG
    sleep 2
    pkill -9 -f train_gpt_focal 2>/dev/null
    sleep 3
}

run() {
    local name="$1"
    shift
    echo "--- $name ---" >> $LOG
    echo "START: $(date)" >> $LOG
    env ITERATIONS=400 "$@" python train_gpt_focal_fixed.py > "/workspace/${name}.txt" 2>&1
    grab "$name"
}

# ============================================
# PHASE 1: FINE-GRAINED SOFTCAP SWEEP ON 1/8
# If SC15 won Wave 9, sweep 13-17 in steps of 1
# ============================================
echo "========== SOFTCAP FINE SWEEP ===========" >> $LOG

# Sweep softcap around SC15 with MatLR=0.10 (E8 base)
run "F1_SC13_MatLR10" TIDAL_LR=1 LOGIT_SOFTCAP=13.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1
run "F2_SC14_MatLR10" TIDAL_LR=1 LOGIT_SOFTCAP=14.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1
run "F3_SC16_MatLR10" TIDAL_LR=1 LOGIT_SOFTCAP=16.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1
run "F4_SC17_MatLR10" TIDAL_LR=1 LOGIT_SOFTCAP=17.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1

# ============================================
# PHASE 2: FINE-GRAINED MATRIX LR SWEEP ON 1/8
# ============================================
echo "========== MATRIX LR FINE SWEEP ===========" >> $LOG

# E8 showed SC15+MatLR0.10 = 1.5354. Sweep around MatLR0.10 with SC15.
run "F5_SC15_MatLR009" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.09 ENCODER_LAYERS=1
run "F6_SC15_MatLR011" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.11 ENCODER_LAYERS=1

# ============================================
# PHASE 3: GQA ON 1/8
# 2 KV heads was decent before (A17 = 1.5761)
# With asymmetric it might be different.
# ============================================
echo "========== GQA ON 1/8 ===========" >> $LOG

run "F7_GQA2" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1 NUM_KV_HEADS=2

# ============================================
# PHASE 4: TIDAL WARMUP RATIO ON 1/8
# Default Tidal = 38.2% warmup. With more decoder
# layers and faster steps, maybe different ratio helps.
# ============================================
echo "========== TIDAL VARIANT ===========" >> $LOG

# Try 30% warmup (shorter warmup, more time at high LR)
run "F8_Tidal30" TIDAL_LR=1 TIDAL_WARMUP=0.30 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1

# ============================================
# PHASE 5: STACK BEST COMBO
# Combine the best softcap + best LR from above
# ============================================
echo "========== FINAL STACK ===========" >> $LOG

# Stack: E8 config (SC15+MatLR0.10) + QK2.0
run "F9_E8_QK2" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1 QK_GAIN_INIT=2.0

# Rerun E8 config for confidence
run "F10_E8_Rerun" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1

echo "" >> $LOG
echo "=== WAVE 10 COMPLETE $(date) ===" >> $LOG
cat $LOG
