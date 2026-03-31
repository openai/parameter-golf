#!/usr/bin/env bash
# =================================================================
# FINAL SUBMISSION RUN — Parameter Golf
# This script is tested and validated. Run EXACTLY this on H100.
# =================================================================
set -euo pipefail

echo "===== PARAMETER GOLF — FINAL SUBMISSION ====="
echo "Time: $(date)"
echo "=============================================="

cd /workspace
if [ ! -d parameter-golf/.git ]; then
    rm -rf parameter-golf
    git clone https://github.com/Omrigotlieb/parameter-golf.git
fi
cd parameter-golf
git fetch origin main
git reset --hard origin/main
pip install -q zstandard

echo "[1/5] Verifying setup..."
nvidia-smi -L | head -2
python3 -c "import zstandard; print('zstandard OK')"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.device_count()} GPUs')"
ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l | xargs echo "Train shards:"
ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin | wc -l | xargs echo "Val shards:"

echo "[2/5] Smoke test (2 GPU, 3 iters)..."
NCCL_IB_DISABLE=1 ITERATIONS=3 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 \
MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=1024 WARMUP_STEPS=1 \
torchrun --standalone --nproc_per_node=2 train_gpt.py 2>&1 | \
grep -E "step:|final|int6|GPTQ|submission|Error" | head -15

# Check smoke passed
if [ $? -ne 0 ]; then
    echo "SMOKE TEST FAILED — aborting"
    exit 1
fi
echo "Smoke test PASSED"

echo "[3/5] Downloading data (if needed)..."
SHARD_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -lt 10 ]; then
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
fi

echo "[4/5] TRAINING (8×H100, 600s wallclock)..."
NCCL_IB_DISABLE=1 \
RUN_ID="FINAL_$(date +%Y%m%d_%H%M%S)" \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=200 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/FINAL.log

echo "[5/5] Checking results..."
echo ""
echo "===== SUBMISSION RESULTS ====="
grep -E "val_bpb|stopping|final|int6|GPTQ|submission|peak" /workspace/FINAL.log
echo ""

# Verify submission is valid
INT6_SIZE=$(grep "total_submission:" /workspace/FINAL.log | tail -1 | grep -oP 'total_submission:\K[0-9]+')
INT6_BPB=$(grep "final_int6_gptq_roundtrip_exact" /workspace/FINAL.log | grep -oP 'val_bpb:\K[0-9.]+')

echo "Artifact + code: ${INT6_SIZE:-UNKNOWN} bytes (limit: 16,000,000)"
echo "Post-quant bpb: ${INT6_BPB:-UNKNOWN} (baseline: 1.2244)"

if [ -n "$INT6_SIZE" ] && [ "$INT6_SIZE" -lt 16000000 ]; then
    echo "SIZE: PASS"
else
    echo "SIZE: FAIL or UNKNOWN"
fi

echo ""
echo "===== DONE ====="
echo "Artifacts saved: final_model.pt, final_model.int8.ptz, final_model.int6.ptz"
echo "Log saved: /workspace/FINAL.log"
