#!/bin/bash
# Deploy and run on 8xH100 RunPod
# Usage: Run this script on the RunPod machine after cloning the repo

set -e

echo "=== Setting up Parameter Golf submission ==="

# Clone repo if not already present
cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/openai/parameter-golf.git
fi
cd parameter-golf

# Download sp4096 data (full 80 shards + validation)
echo "=== Downloading sp4096 dataset ==="
python3 data/cached_challenge_fineweb.py --variant sp4096

# Copy the QAT training script
echo "=== QAT script should be at /workspace/parameter-golf/train_gpt_qat.py ==="
echo "=== If not, copy it from your local machine ==="

# Verify setup
echo "=== Verifying GPU setup ==="
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "=== Ready! Run the submission with: ==="
echo ""
echo "  RUN_ID=submission_6bit_v4096_rec5x2_d768 \\"
echo "  VOCAB_SIZE=4096 \\"
echo "  DATA_PATH=./data/datasets/fineweb10B_sp4096 \\"
echo "  TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \\"
echo "  QAT_BITS=6 \\"
echo "  QAT_START_FRAC=0.0 \\"
echo "  NUM_LAYERS=10 \\"
echo "  NUM_UNIQUE_LAYERS=5 \\"
echo "  MODEL_DIM=768 \\"
echo "  NUM_HEADS=12 \\"
echo "  NUM_KV_HEADS=4 \\"
echo "  MLP_MULT=2 \\"
echo "  TRAIN_BATCH_TOKENS=524288 \\"
echo "  VAL_LOSS_EVERY=1000 \\"
echo "  WARMDOWN_ITERS=1200 \\"
echo "  torchrun --standalone --nproc_per_node=8 train_gpt_qat.py"
