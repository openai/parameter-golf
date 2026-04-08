@echo off
set "RUN_ID=cuda_marathon_sp1024_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%"
set "DATA_PATH=C:/Users/Public/parameter-golf/data/datasets/fineweb10B_sp1024"
set "TOKENIZER_PATH=C:/Users/Public/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
set "VOCAB_SIZE=1024"

REM --- Architecture (Stable Baseline) ---
set "ARCHITECTURE=hybrid"
set "MODEL_DIM=128"
set "NUM_LAYERS=8"
set "SHARED_BLOCKS=2"
set "MOE_ENABLED=1"
set "MOE_NUM_EXPERTS=3"
set "MOE_TOP_K=1"

REM --- Performance Scaling (OOM-Safe for 4GB 1650 Ti) ---
set "MAX_WALLCLOCK_SECONDS=21600"
set "TRAIN_BATCH_TOKENS=4096"
set "GRAD_ACCUM_STEPS=8"
set "VAL_BATCH_SIZE=4096"
set "ITERATIONS=100000"
set "COMPILE_MODE=none"

REM --- Optimization (AdamW Baseline) ---
set "ADAM_LR=0.04"
set "BETA1=0.9"
set "BETA2=0.95"
set "FEEDBACK_ENABLED=1"
set "FEEDBACK_PASSES=1"

echo "=========================================================================="
echo "LAUNCHING: 6-Hour CUDA Marathon (SP1024 | Native Hybrid)"
echo "RUN ID: %RUN_ID%"
echo "=========================================================================="

C:\Python312\python.exe -u train_gpt.py 2>&1
