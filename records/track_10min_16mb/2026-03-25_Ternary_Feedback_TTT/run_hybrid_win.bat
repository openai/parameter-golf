@echo off
set "ARCHITECTURE=hybrid"
set "COMPILE_MODE=none"
set "DATA_PATH=../../../data/datasets/fineweb10B_sp1024"
set "TOKENIZER_PATH=fineweb_8192_bpe.model"

REM === Model Architecture ===
set "NUM_LAYERS=8"
set "MODEL_DIM=256"
set "NUM_HEADS=4"
set "NUM_KV_HEADS=2"
set "MLP_MULT=4"
set "EMBED_DIM=256"
set "SHARED_BLOCKS=2"

REM === Innovations ===
set "FEEDBACK_ENABLED=1"
set "CAPSULE_ENABLED=1"
set "CAPSULE_NUM=16"
set "CAPSULE_DIM=128"
set "KOOPMAN_ENABLED=1"
set "KOOPMAN_SPECULATOR_ENABLED=1"
set "KOOPMAN_STATE_DIM=128"
set "KOOPMAN_MIXER_RANK=4"
set "BIGRAM_HASH_ENABLED=1"
set "BIGRAM_HASH_BUCKETS=8192"
set "BIGRAM_HASH_DIM=128"
set "ENGRAM_NUM_ORDERS=3"
set "TTT_ENABLED=1"
set "TTT_LR=0.002"
set "TTT_EPOCHS=1"
set "TTT_SCOPE=shared"
set "TURBO_QUANT_KV=1"
set "TURBO_QUANT_TRAIN=1"

REM === GTX 1650 Ti Memory-Safe Training Config ===
set "NPROC_PER_NODE=1"
set "TRAIN_SEQ_LEN=512"
set "TRAIN_BATCH_TOKENS=4096"
set "GRAD_ACCUM_STEPS=4"
set "VAL_BATCH_SIZE=4096"

REM === Chinchilla Scaling Run: 60 minutes ===
set "MAX_WALLCLOCK_SECONDS=3600"
set "ITERATIONS=10000"
set "VAL_LOSS_EVERY=100"
set "TRAIN_LOG_EVERY=10"
set "SLIDING_EVAL=1"
set "SLIDING_EVAL_STRIDE=64"
set "SLIDING_BATCH_SIZE=64"
set "TEMP_SCALING=1"
set "SEED=42"

echo "==========================================================================" > hybrid_cuda_benchmark.log
echo "RUNNING: TKA-Hybrid-CUDA Chinchilla Scaling (60min, Seed: 42)" >> hybrid_cuda_benchmark.log
echo "==========================================================================" >> hybrid_cuda_benchmark.log

C:\Python312\python.exe -u train_gpt.py >> hybrid_cuda_benchmark.log 2>&1
