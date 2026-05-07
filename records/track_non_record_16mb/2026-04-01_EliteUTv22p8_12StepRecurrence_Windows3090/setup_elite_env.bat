@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [Elite Setup] Starting environment installation for RTX 3090/Windows...
echo.

:: 1. Create Virtual Environment
echo [Elite Setup] Creating Virtual Environment (venv)...
python -m venv venv
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create venv. Ensure Python 3.10+ is in your PATH.
    pause
    exit /b 1
)

:: 2. Activate Environment
echo [Elite Setup] Activating Environment...
call venv\Scripts\activate

:: 3. Upgrade Pip
echo [Elite Setup] Upgrading Pip...
python -m pip install --upgrade pip

:: 4. Install Optimized PyTorch (CUDA 12.4)
echo [Elite Setup] Installing PyTorch 2.6.0 (CUDA 12.4)...
:: This specific index is required for high-performance Ampere/Hopper support on Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

:: 5. Apply Elite Bug Fix 2 (Triton Compatibility)
echo [Elite Setup] Applying 'Elite' Bug Fix 2: Triton-Windows...
:: COMMUNITY FIX: PyTorch 2.6 requires Triton < 3.3 on Windows for AttrsDescriptor stability.
pip install "triton-windows<3.3"

:: 6. Install Remaining Dependencies
echo [Elite Setup] Installing regular dependencies from requirements.txt...
pip install -r requirements.txt

:: 7. Data Preparation (Elite Standard: sp1024, 80 shards)
echo [Elite Setup] Downloading FineWeb-Edu-10B Shards (sp1024)...
:: This will materialize the training data and tokenizer in the ./data folder.
python ..\..\..\data\cached_challenge_fineweb.py --variant sp1024 --train-shards 80

:: 8. Final Sanity Check
echo [Elite Setup] Running final verification...
python -c "import torch; import triton; print(f'--- VERIFICATION SUCCESS ---\nTorch: {torch.__version__}\nCUDA: {torch.version.cuda}\nTriton: Available\n----------------------------')"

echo.
echo ============================================================
echo [SUCCESS] Elite Universal Transformer environment is READY.
echo Instructions:
echo 1. Run 'venv\Scripts\activate'
echo 2. Execute 'limits_test_10m.bat' for a 10-minute stability test.
echo 3. Execute 'final_run_10m.bat' for the championship result.
echo ============================================================
pause
