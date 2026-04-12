@echo off
setlocal
cd /d "%~dp0"
:: Final 10-Minute Championship Run
:: Incorporates Polar Express, 256-Seq Curriculum, and Warp Speed TTT

:: --- Training Window (10 Minutes) ---
set MAX_WALLCLOCK_SECONDS=600
set ITERATIONS=55
set WARMUP_STEPS=20
set VAL_LOSS_EVERY=10


:: --- Dataset & Throughput ---
set DATA_PATH=..\..\..\data\datasets\fineweb10B_sp1024
set TOKENIZER_PATH=..\..\..\data\tokenizers\fineweb_1024_bpe.model
set TRAIN_BATCH_TOKENS=524288
set GRAD_ACCUM_STEPS=16
set TRAIN_SEQ_LEN=256
set LORA_RANK=16

:: --- Optimizer (Elite 22.8 "Safe-Stability") ---
set MUON_BACKEND_STEPS=5
set MATRIX_LR=0.009
set GRAD_CLIP_NORM=1.0

:: --- TTT (Test-Time Training) - TTT Cooling ---
set TTT_ENABLED=1
set TTT_LR=4e-4
set EVAL_STRIDE=64

echo =======================================================
echo Launching Final 10-Minute Championship Run
echo Mode: Polar Express + Warp Speed TTT
echo Data: 256-1024 Seq Curriculum (32x Throughput)
echo Limit: 600 Seconds (10 Minutes)
echo =======================================================

.\venv\Scripts\python train_gpt_windows.py

echo.
echo =======================================================
echo Training Window Complete. Generated final_model.int8.ptz
echo =======================================================
pause
