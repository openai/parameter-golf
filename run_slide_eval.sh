#!/bin/bash
# Sliding eval only — no SWA (our actual submission config)
BASE="CUDA_VISIBLE_DEVICES=0 NO_COMPILE=1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=5400 ITERATIONS=3000 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 WARMUP_STEPS=5 NUM_LAYERS=12 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5 SWA_ENABLED=0 WARMDOWN_ITERS=3000"

cd /c/Users/deepc/parameter-golf

run_experiment() {
    local name=$1
    local extra_env=$2
    echo ""
    echo ">>> STARTING: $name at $(date)"
    eval "$BASE RUN_ID=exp_${name} $extra_env python records/track_10min_16mb/our_submission/train_gpt.py" 2>&1 | grep -E "^(step:(1|2|3|4|5|6|7|8|9|10)/|step:100/|step:500/|step:1000/|step:1500/|step:2000/|step:2500/|step:3000/|model_params|stopping|peak|final_int8|val_loss|val_bpb|sliding_eval)"
    echo ">>> DONE: $name at $(date)"
    echo "============================================"
}

echo "============================================"
echo "SLIDING EVAL EXPERIMENTS (no SWA) — started $(date)"
echo "============================================"

# Stride 64 (no SWA)
run_experiment "slide64_noswa" "EVAL_STRIDE=64 EVAL_BATCH_SEQS=16"

# Stride 32 (no SWA, smaller batch to avoid OOM)
run_experiment "slide32_noswa" "EVAL_STRIDE=32 EVAL_BATCH_SEQS=8"

echo "ALL SLIDING EVAL EXPERIMENTS COMPLETE at $(date)"
