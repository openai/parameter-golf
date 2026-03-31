@echo off
REM COMPLETE AUTOMATED TRAINING SCRIPT (Windows)
REM Run this on your 8xH100 machine
REM Usage: train_automated.bat

setlocal enabledelayedexpansion
cls

echo.
echo ============================================================
echo  OpenAI Parameter Golf - Automated Training Script (Windows)
echo  Requires: 8x H100 GPUs, PyTorch 2.4+
echo ============================================================
echo.

REM Step 1: Verify Environment
echo [STEP 1/6] Verifying environment...

REM Check for nvidia-smi
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ERROR: CUDA not found. Install NVIDIA CUDA drivers.
    exit /b 1
)

echo ✓ CUDA available
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

REM Step 2: Install Dependencies
echo.
echo [STEP 2/6] Installing dependencies...
pip install -q torch sentencepiece numpy
echo ✓ Dependencies installed

REM Step 3: Clone and Prepare Data
echo.
echo [STEP 3/6] Preparing FineWeb dataset...

if not exist "parameter-golf" (
    echo  Cloning official repository...
    git clone --depth 1 https://github.com/openai/parameter-golf parameter-golf
)

cd parameter-golf

if not exist "data\datasets\fineweb10B_sp1024\fineweb_train_0001.bin" (
    echo  Downloading FineWeb data (this takes 20-30 minutes)...
    python data\cached_challenge_fineweb.py --variant sp1024
) else (
    echo ✓ FineWeb data already present
)

if not exist "data\tokenizers\fineweb_1024_bpe.model" (
    echo ERROR: Tokenizer not found
    exit /b 1
)

echo ✓ Data ready

REM Step 4: Copy Training Code
echo.
echo [STEP 4/6] Setting up training code...

if not exist "train_gpt.py" (
    echo ERROR: train_gpt.py not found
    exit /b 1
)

python -m py_compile train_gpt.py
echo ✓ Training code verified

REM Step 5: Run Training
echo.
echo [STEP 5/6] Starting training (max 600 seconds)...
echo  Training will complete in ~10 minutes on 8xH100
echo.

REM Create output directory
if not exist "logs" mkdir logs
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set LOG_FILE=logs\training_!mydate!_!mytime!.log

REM Run training
torchrun --standalone --nproc_per_node=8 train_gpt.py > !LOG_FILE! 2>&1

REM Step 6: Verify Results
echo.
echo [STEP 6/6] Verifying results...

if exist "final_model.int8.ptz" (
    for %%A in (final_model.int8.ptz) do (
        set SIZE=%%~zA
        set SIZE_MB=!SIZE:~0,-6!
        if !SIZE! lss 16000000 (
            echo ✓ Model artifact: !SIZE_MB! MB ^(limit: 16 MB^)
        ) else (
            echo ✗ Model too large: !SIZE_MB! MB
            exit /b 1
        )
    )
) else (
    echo ERROR: Model artifact not created
    exit /b 1
)

echo.
echo ============================================================
echo               ^!^!^! TRAINING COMPLETE ^!^!^!
echo ============================================================
echo Results:
echo  Model created: final_model.int8.ptz
echo  Log file: !LOG_FILE!
echo.
echo NEXT STEPS:
echo 1. Update submission.json with metrics
echo 2. Create GitHub repository
echo 3. Fork official repository
echo 4. Submit pull request
echo.
echo See FINAL_CHECKLIST.md for detailed instructions
echo ============================================================
echo.

endlocal
