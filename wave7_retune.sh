#!/bin/bash
# Wave 7: Re-tune hyperparameters around Parallel+SiLU² architecture
# Best config: PARALLEL_BLOCK=1 MLP_ACT=silu2 + Tidal+SC20+RoPE5k+HD+AggrLR = 1.5527
# Hypothesis: hyperparams were tuned for sequential blocks. Parallel changes
# gradient flow, so optimal softcap/RoPE/LR/etc may have shifted.
cd /workspace/parameter-golf
LOG="/workspace/wave7_results.log"

echo "=== WAVE 7: RETUNE $(date) ===" > $LOG
echo "BASELINE: Cosine = 1.6117 | Best: B4 = 1.5527 (Parallel+SiLU2+HD+Aggr)" >> $LOG
echo "" >> $LOG

# Base config for all experiments (B4 winner)
# TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

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
# PHASE 1: MQA + PARALLEL (untested combo)
# A18 showed MQA is fast (2379ms) + good (1.5740)
# Parallel is fast (2315ms) + good (1.5600)
# Together = even faster = even more steps?
# ============================================
echo "========== MQA + PARALLEL ===========" >> $LOG

# C1: MQA + Parallel (basic)
run "C1_MQA_Parallel" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 NUM_KV_HEADS=1

# C2: MQA + Parallel + SiLU² + full extras
run "C2_MQA_ParSiLU2Best" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 NUM_KV_HEADS=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# C3: MQA + Parallel + more heads (16 query heads, 1 KV head)
# More query heads = more capacity without KV cost
run "C3_MQA_16H" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 NUM_KV_HEADS=1 NUM_HEADS=16 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# ============================================
# PHASE 2: SOFTCAP RE-TUNE FOR PARALLEL
# SC20 was optimal for sequential. Parallel shares
# normed input → logit distribution may differ.
# ============================================
echo "========== SOFTCAP RETUNE ===========" >> $LOG

# C4: Softcap 15 (tighter) on best parallel config
run "C4_SC15" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# C5: Softcap 25 (looser) on best parallel config
run "C5_SC25" TIDAL_LR=1 LOGIT_SOFTCAP=25.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# C6: Softcap 18 on best parallel config
run "C6_SC18" TIDAL_LR=1 LOGIT_SOFTCAP=18.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# ============================================
# PHASE 3: ROPE RE-TUNE FOR PARALLEL
# ============================================
echo "========== ROPE RETUNE ===========" >> $LOG

# C7: RoPE 3000 (tighter positional attention)
run "C7_RoPE3k" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=3000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# C8: RoPE 7500
run "C8_RoPE7500" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=7500 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# ============================================
# PHASE 4: LR TUNING FOR PARALLEL
# Parallel changes gradient flow — may want different LRs
# ============================================
echo "========== LR RETUNE ===========" >> $LOG

# C9: More aggressive embed LR
run "C9_EmbLR1" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=1.0 MATRIX_LR=0.06

# C10: More aggressive matrix LR
run "C10_MatLR008" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.08

# C11: Both more aggressive
run "C11_AggrLR2" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=1.0 MATRIX_LR=0.08

# ============================================
# PHASE 5: TIDAL WARMUP RE-TUNE
# 38.2% warmup was optimal for sequential.
# Parallel converges differently — try other ratios.
# ============================================
echo "========== WARMUP RETUNE ===========" >> $LOG

# C12: Breathing LR instead of Tidal (4-7-8 pattern)
run "C12_Breathing" BREATHING_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.06

# ============================================
# PHASE 6: HEAD DIVERSITY STRENGTH
# ============================================
echo "========== HEAD DIV TUNE ===========" >> $LOG

# C13: Stronger head diversity
run "C13_HD1e3" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-3 EMBED_LR=0.8 MATRIX_LR=0.06

# C14: No head diversity (to confirm it still helps with parallel)
run "C14_NoHD" TIDAL_LR=1 LOGIT_SOFTCAP=20.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 EMBED_LR=0.8 MATRIX_LR=0.06

echo "" >> $LOG
echo "=== WAVE 7 COMPLETE $(date) ===" >> $LOG
cat $LOG
