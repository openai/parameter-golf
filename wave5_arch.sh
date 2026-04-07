#!/bin/bash
# Wave 5: Architecture experiments — static changes, torch.compile safe
cd /workspace/parameter-golf
LOG="/workspace/wave5_results.log"

echo "=== WAVE 5: ARCHITECTURE $(date) ===" > $LOG
echo "BASELINE: Cosine = 1.6117 | Best: 1.5744 (Tidal+SC20+RoPE5k+HD+AggrLR)" >> $LOG
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
# PHASE 0: ACTIVATION FUNCTIONS (on best config)
# ============================================
echo "========== ACTIVATION FUNCTIONS ==========" >> $LOG

# A1: LeakyReLU² — on the leaderboard! (#2 entry uses this)
run "A1_LeakyReLU2" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 MLP_ACT=leaky_relu2

# A2: LeakyReLU² on cosine baseline (to isolate activation effect)
run "A2_LeakyReLU2_cosine" COSINE_LR=1 MLP_ACT=leaky_relu2

# A3: SwiGLU — standard in Llama/Mistral
run "A3_SwiGLU" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 MLP_ACT=swiglu

# A4: SwiGLU on cosine baseline
run "A4_SwiGLU_cosine" COSINE_LR=1 MLP_ACT=swiglu

# A5: GELU² — smoother than ReLU²
run "A5_GELU2" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 MLP_ACT=gelu2

# A6: SiLU² — like SwiGLU but without gate
run "A6_SiLU2" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 MLP_ACT=silu2

# ============================================
# PHASE 1: BLOCK STRUCTURE
# ============================================
echo "========== BLOCK STRUCTURE ==========" >> $LOG

# A7: Parallel attention + MLP (PaLM-style)
run "A7_Parallel" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1

# A8: Parallel on cosine
run "A8_Parallel_cosine" COSINE_LR=1 PARALLEL_BLOCK=1

# A9: Sandwich norm (extra norm after attention)
run "A9_Sandwich" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 SANDWICH_NORM=1

# ============================================
# PHASE 2: ACTIVATION + STRUCTURE COMBOS
# ============================================
echo "========== COMBOS ==========" >> $LOG

# A10: LeakyReLU² + Parallel (combine two arch changes)
run "A10_LeakyParallel" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 MLP_ACT=leaky_relu2 PARALLEL_BLOCK=1

# A11: SwiGLU + Parallel
run "A11_SwiGLUParallel" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 MLP_ACT=swiglu PARALLEL_BLOCK=1

# A12: LeakyReLU² + full best config
run "A12_LeakyBest" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06 MLP_ACT=leaky_relu2

# A13: SwiGLU + full best config
run "A13_SwiGLUBest" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06 MLP_ACT=swiglu

# A14: LeakyReLU² + Parallel + full best (EVERYTHING)
run "A14_EVERYTHING" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06 MLP_ACT=leaky_relu2 PARALLEL_BLOCK=1

# ============================================
# PHASE 3: WIDER/DEEPER WITH ARCH CHANGES
# ============================================
echo "========== SCALE WITH ARCH ==========" >> $LOG

# A15: LeakyReLU² + wider MLP (3x)
run "A15_LeakyMLP3" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 MLP_ACT=leaky_relu2 MLP_MULT=3

# A16: SwiGLU + wider dim (since SwiGLU has fewer params per layer)
run "A16_SwiGLU_wide" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 MLP_ACT=swiglu MODEL_DIM=576 NUM_LAYERS=8

# A17: LeakyReLU² + 2 KV heads (more aggressive GQA)
run "A17_LeakyGQA2" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 MLP_ACT=leaky_relu2 NUM_KV_HEADS=2

# A18: LeakyReLU² + 1 KV head (MQA)
run "A18_LeakyMQA" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 MLP_ACT=leaky_relu2 NUM_KV_HEADS=1

echo "" >> $LOG
echo "=== WAVE 5 COMPLETE $(date) ===" >> $LOG
cat $LOG
