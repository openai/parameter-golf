#!/bin/bash
# Wave 11: Final experiments — novel ideas + confidence runs
# By this point we should have a well-tuned best config.
# This wave tries a few novel things and does confidence reruns.
cd /workspace/parameter-golf
LOG="/workspace/wave11_results.log"

echo "=== WAVE 11: FINAL $(date) ===" > $LOG
echo "Final experiments + confidence runs" >> $LOG
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
# PHASE 1: NOVEL ARCHITECTURE IDEAS ON BEST CONFIG
# Things we haven't tried at all
# ============================================
echo "========== NOVEL IDEAS ===========" >> $LOG

# G1: Untied embeddings on best config
# Separate input/output embeddings — more params but potentially better
run "G1_Untied" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1 TIE_EMBEDDINGS=0

# G2: Embed scale sqrt(d) on best config
run "G2_EmbScale" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1 EMBED_SCALE=22.6

# G3: Weight decay schedule (ramp 0 to 0.01) on best config
run "G3_WDSched" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1 WD_SCHEDULE=0.01

# G4: Muon momentum 0.98 (default 0.95)
run "G4_Muon098" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1 MUON_MOMENTUM=0.98

# G5: RoPE 4000 on 1/8 (midpoint between 3k and 5k)
run "G5_RoPE4k" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=4000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1

# ============================================
# PHASE 2: EMBED LR FINE-TUNE
# 0.8 was good, 1.0 was bad. Try 0.9 and 0.7.
# ============================================
echo "========== EMBED LR ===========" >> $LOG

run "G6_EmbLR09" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.9 MATRIX_LR=0.10 ENCODER_LAYERS=1
run "G7_EmbLR07" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.7 MATRIX_LR=0.10 ENCODER_LAYERS=1

# ============================================
# PHASE 3: CONFIDENCE RUNS
# Multiple reruns of absolute best to measure variance
# ============================================
echo "========== CONFIDENCE RUNS ===========" >> $LOG

run "G8_Best_Run1" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1
run "G9_Best_Run2" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1
run "G10_Best_Run3" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1

echo "" >> $LOG
echo "=== WAVE 11 COMPLETE $(date) ===" >> $LOG
cat $LOG
