#!/bin/bash
# Wave 12: Aggressive experiments — pure decoder, wider MLP, model scaling
# Uses findings from Waves 9-11 + 3.1MB artifact headroom
cd /workspace/parameter-golf
LOG="/workspace/wave12_results.log"

echo "=== WAVE 12: AGGRESSIVE $(date) ===" > $LOG
echo "Pure decoder, wider MLP, scaling experiments" >> $LOG
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
# PHASE 1: PURE DECODER (ENCODER_LAYERS=0)
# Bug fixed: default=-1, so ENCODER_LAYERS=0 now works
# All 9 layers as decoder, no encoder skip connections
# ============================================
echo "========== PURE DECODER ===========" >> $LOG

# H1: Pure decoder with best config (E8 base)
run "H1_PureDecoder" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=0

# ============================================
# PHASE 2: WIDER MLP (3x)
# Top competition entries use 3x MLP width
# MLP_MULT=3 + SiLU² → hidden=1024 (vs current 682)
# More params but potentially much better quality
# Will be slower per step but might make up in quality
# ============================================
echo "========== WIDER MLP ===========" >> $LOG

# H2: 3x MLP on best config (1/8 split)
run "H2_MLP3x" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1 MLP_MULT=3

# H3: 3x MLP + pure decoder
run "H3_MLP3x_PureDec" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=0 MLP_MULT=3

# ============================================
# PHASE 3: WD SCHEDULE (only env var that exists)
# WD_SCHEDULE ramps weight decay from 0 to target
# Competition winners use WD=0.04
# ============================================
echo "========== WEIGHT DECAY SCHEDULE ===========" >> $LOG

# H4: WD schedule ramping to 0.04
run "H4_WDSched04" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1 WD_SCHEDULE=0.04

# ============================================
# PHASE 4: STACK WINNERS FROM ABOVE
# ============================================
echo "========== STACK ===========" >> $LOG

# H5: Pure decoder + 3x MLP + WD (aggressive combo)
run "H5_AllStack" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=0 MLP_MULT=3 WD_SCHEDULE=0.04

# H6: Best config rerun for final confidence
run "H6_BestRerun" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1

echo "" >> $LOG
echo "=== WAVE 12 COMPLETE $(date) ===" >> $LOG
cat $LOG
