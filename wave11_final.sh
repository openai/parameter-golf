#!/bin/bash
# Wave 11: Trimmed to 2 key experiments only
cd /workspace/parameter-golf
LOG="/workspace/wave11_results.log"

echo "=== WAVE 11: TRIMMED $(date) ===" > $LOG
echo "Only 2 key experiments" >> $LOG
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

# G1: Untied embeddings — novel, could be big win
run "G1_Untied" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.11 ENCODER_LAYERS=1 TIE_EMBEDDINGS=0

# G3: WD Schedule — competition winners use this
run "G3_WDSched" TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.11 ENCODER_LAYERS=1 WD_SCHEDULE=0.01

echo "" >> $LOG
echo "=== WAVE 11 COMPLETE $(date) ===" >> $LOG
cat $LOG
