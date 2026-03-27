#!/bin/bash
# Round 2b — restart from 12_layers_v2 onward
# bigger_batch already showed val_bpb=1.6879 (worse), skip it

COMMON="CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=1800 ITERATIONS=1000 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 SWA_ENABLED=0 EVAL_STRIDE=0 WARMUP_STEPS=5"

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
echo "ROUND 2b EXPERIMENTS — started $(date)"
echo "============================================"

# 1. 12 layers (more wallclock for eval)
run_experiment "12_layers_v2" "NUM_LAYERS=12 MAX_WALLCLOCK_SECONDS=2400" "records/track_10min_16mb/our_submission/train_gpt.py"

# 2. Larger BigramHash vocab
run_experiment "bigram_20k" "BIGRAM_VOCAB_SIZE=20480" "records/track_10min_16mb/our_submission/train_gpt.py"

# 3. Higher bigram dim
run_experiment "bigram_dim256" "BIGRAM_DIM=256" "records/track_10min_16mb/our_submission/train_gpt.py"

# 4. More MLP capacity
run_experiment "mlp_3.5x" "MLP_MULT=3.5" "records/track_10min_16mb/our_submission/train_gpt.py"

# 5. Muon momentum 0.95
run_experiment "muon_095" "MUON_MOMENTUM=0.95" "records/track_10min_16mb/our_submission/train_gpt.py"

# 6. Lower weight decay
run_experiment "wd_002" "WEIGHT_DECAY=0.02" "records/track_10min_16mb/our_submission/train_gpt.py"

# 7. 2048 seq len
run_experiment "seq2048" "TRAIN_SEQ_LEN=2048" "records/track_10min_16mb/our_submission/train_gpt.py"

# 8. Combined best guesses
run_experiment "combined_v1" "BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5 TRAIN_SEQ_LEN=2048" "records/track_10min_16mb/our_submission/train_gpt.py"

echo ""
echo "ALL ROUND 2b EXPERIMENTS COMPLETE at $(date)"
