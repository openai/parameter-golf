#!/bin/bash
# Wave 9: Optimize around asymmetric 1/8 split
# Best so far: D5 = 1.5412 (Asym 1/8 + Parallel + SiLU² + HD + MatLR0.08 + SC20)
# D2 showed SC15 helps. D8/D9 test SC18 on asymmetric.
# This wave: fine-tune everything around 1/8.
cd /workspace/parameter-golf
LOG="/workspace/wave9_results.log"

echo "=== WAVE 9: OPTIMIZE $(date) ===" > $LOG
echo "BASELINE: 1.6117 | Best: D6 = 1.5377 (Asym 1/8)" >> $LOG
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
# PHASE 1: SOFTCAP ON 1/8
# SC15 helped on 4/5 split. Test on 1/8.
# ============================================
echo "========== SOFTCAP ON 1/8 ===========" >> $LOG

# E1: 1/8 + SC15 (best softcap from Wave 7)
run "E1_Asym27_SC15" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=1

# E2: 1/8 + SC18
run "E2_Asym27_SC18" TIDAL_LR=1 LOGIT_SOFTCAP=18.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=1

# E3: 1/8 + SC12 (push tighter)
run "E3_Asym27_SC12" TIDAL_LR=1 LOGIT_SOFTCAP=12.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=1

# ============================================
# PHASE 2: LR TUNING ON 1/8
# MatLR0.08 was better than 0.06. Try more.
# ============================================
echo "========== LR ON 1/8 ===========" >> $LOG

# E4: 1/8 + Matrix LR 0.10
run "E4_Asym27_MatLR10" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1

# E5: 1/8 + Matrix LR 0.12
run "E5_Asym27_MatLR12" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.12 ENCODER_LAYERS=1

# ============================================
# PHASE 3: ACTIVATION ON 1/8
# SiLU² won on 4/5. Does it still win on 1/8?
# ============================================
echo "========== ACTIVATION ON 1/8 ===========" >> $LOG

# E6: 1/8 with plain relu² (ablation)
run "E6_Asym27_ReLU2" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=1

# ============================================
# PHASE 4: QK GAIN ON 1/8
# B5 nearly tied best on 4/5. Try on 1/8.
# ============================================
echo "========== QK GAIN ON 1/8 ===========" >> $LOG

# E7: 1/8 + QK gain 2.0
run "E7_Asym27_QK2" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=1 QK_GAIN_INIT=2.0

# ============================================
# PHASE 5: STACK ALL WINS ON 1/8
# Combine best softcap + best LR + best QK
# ============================================
echo "========== STACK BEST ON 1/8 ===========" >> $LOG

# E8: 1/8 + SC15 + MatLR0.10 (if both help individually)
run "E8_Asym27_SC15_MatLR10" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.10 ENCODER_LAYERS=1

# E9: 1/8 + SC15 + QK2.0
run "E9_Asym27_SC15_QK2" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=1 QK_GAIN_INIT=2.0

# E10: Rerun D5 to confirm 1.5412
run "E10_D5_Rerun" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08 ENCODER_LAYERS=1

echo "" >> $LOG
echo "=== WAVE 9 COMPLETE $(date) ===" >> $LOG
cat $LOG
