#!/bin/bash
# COMPLETE AUTOMATED TRAINING SCRIPT
# Run this on your 8xH100 machine
# Usage: bash train_automated.sh

set -euo pipefail

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  OpenAI Parameter Golf - Automated Training Script            ║"
echo "║  Requires: 8x H100 GPUs, PyTorch 2.4+, ~30 min (data + train) ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Verify Environment
echo -e "${YELLOW}[STEP 1/5]${NC} Verifying environment..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: CUDA not found. Install NVIDIA CUDA drivers.${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 8 ]; then
    echo -e "${RED}ERROR: Need 8 GPUs, found $GPU_COUNT${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found $GPU_COUNT GPUs${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Step 2: Install Dependencies
echo ""
echo -e "${YELLOW}[STEP 2/5]${NC} Installing dependencies..."
pip install -q torch sentencepiece numpy 2>&1 | tail -1
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Clone and Prepare Data
echo ""
echo -e "${YELLOW}[STEP 3/5]${NC} Preparing FineWeb dataset..."
if [ ! -d "parameter-golf" ]; then
    echo "  Cloning official repository..."
    git clone --depth 1 https://github.com/openai/parameter-golf parameter-golf 2>&1 | tail -1
fi

cd parameter-golf

if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "  Downloading FineWeb data (this takes ~20-30 minutes)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo -e "${GREEN}✓ FineWeb data already present${NC}"
fi

# Verify data
if [ ! -f "data/tokenizers/fineweb_1024_bpe.model" ]; then
    echo -e "${RED}ERROR: Tokenizer not found${NC}"
    exit 1
fi

TRAIN_FILES=$(find data/datasets/fineweb10B_sp1024 -name "fineweb_train_*.bin" | wc -l)
VAL_FILES=$(find data/datasets/fineweb10B_sp1024 -name "fineweb_val_*.bin" | wc -l)
echo -e "${GREEN}✓ Data ready: $TRAIN_FILES train files, $VAL_FILES val files${NC}"

# Step 4: Copy Training Code
echo ""
echo -e "${YELLOW}[STEP 4/5]${NC} Setting up training code..."
if [ ! -f "train_gpt.py" ]; then
    echo -e "${RED}ERROR: train_gpt.py not found in current directory${NC}"
    exit 1
fi

# Verify syntax
python -m py_compile train_gpt.py
echo -e "${GREEN}✓ Training code verified${NC}"

# Step 5: Run Training
echo ""
echo -e "${YELLOW}[STEP 5/5]${NC} Starting training (max 600 seconds)..."
echo "  Training will complete in ~10 minutes on 8xH100"
echo ""

# Create output directory
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Run training with logging
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$LOG_FILE"

# Step 6: Verify Results
echo ""
echo -e "${YELLOW}[STEP 6/6]${NC} Verifying results..."

if [ -f "final_model.int8.ptz" ]; then
    SIZE=$(stat -c%s final_model.int8.ptz 2>/dev/null || stat -f%z final_model.int8.ptz)
    SIZE_MB=$((SIZE / 1048576))

    if [ "$SIZE" -le 16000000 ]; then
        echo -e "${GREEN}✓ Model artifact: ${SIZE_MB} MB (limit: 16 MB)${NC}"
    else
        echo -e "${RED}✗ Model too large: ${SIZE_MB} MB (limit: 16 MB)${NC}"
        exit 1
    fi
else
    echo -e "${RED}ERROR: Model artifact not created${NC}"
    exit 1
fi

# Extract BPB score
BPB=$(grep -i "final bpb" "$LOG_FILE" | tail -1 || echo "Not found")
WALLCLOCK=$(grep -i "wallclock" "$LOG_FILE" | tail -1 || echo "Not found")

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║               🎉 TRAINING COMPLETE! 🎉                   ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║ Results:                                                 ║${NC}"
echo -e "${GREEN}║  Model Size: ${SIZE_MB} MB / 16 MB ✅                           ║${NC}"
echo -e "${GREEN}║  Score: $BPB                               ║${NC}"
echo -e "${GREEN}║  Time: $WALLCLOCK                                 ║${NC}"
echo -e "${GREEN}║  Log: $LOG_FILE                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo "NEXT STEPS:"
echo "1. Update submission.json with metrics"
echo "2. Create GitHub repository"
echo "3. Fork official repository"
echo "4. Submit pull request"
echo ""
echo "See FINAL_CHECKLIST.md for detailed instructions"
