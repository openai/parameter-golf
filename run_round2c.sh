#!/bin/bash
# Round 2c — Stack the two biggest winners from round 2b
# 12 layers (-0.019 bpb) + bigram 20K (-0.011 bpb)

COMMON="CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=2400 ITERATIONS=1000 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 SWA_ENABLED=0 EVAL_STRIDE=0 WARMUP_STEPS=5"

cd /c/Users/deepc/parameter-golf

run_experiment() {
    local name=$1
    local extra_env=$2
    local script=$3
    echo ""
    echo ">>> STARTING: $name at $(date)"
    eval "$COMMON RUN_ID=exp_${name} $extra_env python $script" 2>&1 | grep -E "^(step:(1|2|3|4|5|6|7|8|9|10)/|step:100/|step:200/|step:500/|step:1000/|model_params|stopping|peak|final_int8|val_loss|val_bpb)"
    echo ">>> DONE: $name at $(date)"
    echo "============================================"
}

echo "============================================"
echo "ROUND 2c — COMBINED WINNERS — started $(date)"
echo "============================================"

# Combined v2: 12 layers + bigram 20K (top 2 winners)
run_experiment "combined_v2" "NUM_LAYERS=12 BIGRAM_VOCAB_SIZE=20480" "records/track_10min_16mb/our_submission/train_gpt.py"

# Combined v3: 12 layers + bigram 20K + MLP 3.5x (top 3 winners)
run_experiment "combined_v3" "NUM_LAYERS=12 BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5" "records/track_10min_16mb/our_submission/train_gpt.py"

echo ""
echo "ALL ROUND 2c EXPERIMENTS COMPLETE at $(date)"
