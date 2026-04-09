#!/bin/bash
# Wave 6: Focused experiments — building on Parallel blocks win
# Each experiment has a clear hypothesis
cd /workspace/parameter-golf
LOG="/workspace/wave6_results.log"

echo "=== WAVE 6: FOCUSED $(date) ===" > $LOG
echo "BASELINE: Cosine = 1.6117 | Best: A14 = 1.5586 (Parallel+Leaky+HD+Aggr)" >> $LOG
echo "Best Parallel alone: A7 = 1.5600" >> $LOG
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
# B1: Parallel + relu² + HeadDiv + AggrLR
# Hypothesis: A14 used LeakyReLU² which hurts.
# Plain relu² parallel + full extras should beat 1.5586.
# ============================================
run "B1_ParallelBest" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# ============================================
# B2: Parallel + 10 layers (9L default)
# Hypothesis: Parallel saves ~200ms/step (2315 vs 2510).
# 10L parallel should be ~2570ms → ~234 steps.
# More capacity at similar step count = lower BPB.
# ============================================
run "B2_Parallel10L" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 NUM_LAYERS=10

# ============================================
# B3: Parallel + 10L + HeadDiv + AggrLR
# Hypothesis: Stack the best extras on 10L parallel.
# ============================================
run "B3_Parallel10LBest" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 NUM_LAYERS=10 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# ============================================
# B4: Parallel + SiLU² + HeadDiv + AggrLR
# Hypothesis: SiLU² was best activation (1.5743) at same
# speed as relu². With parallel it won't lose steps.
# Tests if SiLU² + parallel + extras can beat B1.
# ============================================
run "B4_ParallelSiLU2Best" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# ============================================
# B5: Parallel + QK gain 2.0
# Hypothesis: Default qk_gain_init=1.5. Parallel shares
# normed input between attn+MLP, so stronger attention
# signal (higher gain) might help differentiate.
# ============================================
run "B5_ParallelQK2" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06 QK_GAIN_INIT=2.0

echo "" >> $LOG
echo "=== WAVE 6 COMPLETE $(date) ===" >> $LOG
cat $LOG
