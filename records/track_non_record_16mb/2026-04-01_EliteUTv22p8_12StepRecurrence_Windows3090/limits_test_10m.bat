@echo off
setlocal
cd /d "%~dp0"
:: Elite Universal Transformer: Final 10-Minute Wallclock Optimization
:: Standard: 524,288 Token Global Batch | 12-Step Reasoning Depth
:: Optimization: Safe-Speed (0.012 Muon LR)

:: --- Training Window (10-Minute Wall) ---
set MAX_WALLCLOCK_SECONDS=600
set ITERATIONS=55
set WARMUP_STEPS=20
set VAL_LOSS_EVERY=10


:: --- Elite Standard 22.8 (Efficient Frontier / v22.8) ---
set DATA_PATH=..\..\..\data\datasets\fineweb10B_sp1024
set TOKENIZER_PATH=..\..\..\data\tokenizers\fineweb_1024_bpe.model
set TRAIN_BATCH_TOKENS=524288
set GRAD_ACCUM_STEPS=16
set TRAIN_SEQ_LEN=256
set LORA_RANK=16

:: --- Optimization ---
set MUON_BACKEND_STEPS=5
set MATRIX_LR=0.009
set GRAD_CLIP_NORM=1.0

echo =======================================================
echo Elite Universal Transformer: Final Submission Window
echo Standard: 524k Global Batch ^| 12-Step Reasoning
echo Mode: Safe-Stability (0.009 Muon LR)
echo Limit: 600 Seconds
echo =======================================================

.\venv\Scripts\python train_gpt_windows.py

echo.
echo =======================================================
echo Training Period Complete. Checking model.pt for BPB.
echo =======================================================
pause
