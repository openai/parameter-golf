#!/bin/bash
# Wave 8: Asymmetric splits + stacking Wave 7 wins
# Best: C10 = 1.5501 (Parallel+SiLU²+HD+MatLR0.08+Tidal+SC20+RoPE5k)
# New base = B4 config but with MATRIX_LR=0.08
cd /workspace/parameter-golf
LOG="/workspace/wave8_results.log"

echo "=== WAVE 8: ASYMMETRY $(date) ===" > $LOG
echo "BASELINE: 1.6117 | Best: C10 = 1.5501 (MatLR0.08)" >> $LOG
echo "Default split: 4 encoder / 5 decoder (9 layers)" >> $LOG
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
# PHASE 1: ESTABLISH NEW BEST (stack C10 wins)
# ============================================
echo "========== STACKING ===========" >> $LOG

# D1: C10 config + SC18 (both were individual wins)
# Hypothesis: SC18 tied SC15 for best softcap, MatLR0.08 was best LR.
# Stacking should compound.
run "D1_MatLR08_SC18" TIDAL_LR=1 LOGIT_SOFTCAP=18.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08

# D2: C10 config + SC15
run "D2_MatLR08_SC15" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08

# D3: C10 config but NO HeadDiv (C14 showed it barely matters)
# If this ties C10, we simplify the config.
run "D3_MatLR08_NoHD" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 EMBED_LR=0.8 MATRIX_LR=0.08

# ============================================
# PHASE 2: ASYMMETRIC ENCODER/DECODER SPLIT
# Default is 4/5. More decoder layers = more generation capacity.
# ============================================
echo "========== ASYMMETRIC SPLITS ===========" >> $LOG

# D4: 3 encoder / 6 decoder on C10 config
# Hypothesis: More decoder capacity helps generation quality.
run "D4_Asym36" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=3

# D5: 2 encoder / 7 decoder on C10 config
# Hypothesis: Push even harder toward decoder.
run "D5_Asym27" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=2

# D6: 1 encoder / 8 decoder (extreme)
# Hypothesis: If 2/7 helps, push to the limit.
run "D6_Asym18" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=1

# D7: 5 encoder / 4 decoder (opposite direction — more encoder)
# Hypothesis: Control experiment. If more encoder helps, our assumption is wrong.
run "D7_Asym54" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=5

# ============================================
# PHASE 3: BEST ASYMMETRIC + BEST SOFTCAP
# Stack if both asymmetry and softcap help
# ============================================
echo "========== STACK BEST ===========" >> $LOG

# D8: Best asymmetric (3/6) + SC18 + MatLR0.08
# Only run if D4 shows improvement over D1
run "D8_Asym36_SC18" TIDAL_LR=1 LOGIT_SOFTCAP=18.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=3

# D9: Best asymmetric (2/7) + SC18 + MatLR0.08
run "D9_Asym27_SC18" TIDAL_LR=1 LOGIT_SOFTCAP=18.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=2

# D10: Rerun C10 (confirm 1.5501 is real, not noise)
run "D10_C10_Rerun" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08

echo "" >> $LOG
echo "=== WAVE 8 COMPLETE $(date) ===" >> $LOG
cat $LOG
