#!/bin/bash
# Capacity Sweep Phase A: Size checks (9 configs)
# 1 step each, just build + quantize + compress with rANS
set -e
cd /workspace/parameter-golf
mkdir -p logs

BASE_SIZE="DATA_PATH=/workspace/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
NUM_LAYERS=12 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0 \
TIE_EMBEDDINGS=1 USE_FUSED_QKV=1 BIGRAM_VOCAB_SIZE=0 \
LEAKY_RELU2=1 LEAKY_SLOPE=0.95 USE_RANS=1 \
ITERATIONS=1 WARMUP_STEPS=0 VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=262144 TRAIN_SEQ_LEN=2048 \
SWA_ENABLED=0 TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=0 SEED=42"

echo "=== SZ-A: 12L dim512 mlp3.0 kv4 (baseline) ==="
eval "$BASE_SIZE" python -u train_gpt.py 2>&1 | tee logs/sz_A_baseline.log
echo ""

echo "=== SZ-B: 12L dim512 mlp3.25 kv4 ==="
eval "$BASE_SIZE MLP_MULT=3.25" python -u train_gpt.py 2>&1 | tee logs/sz_B_mlp325.log
echo ""

echo "=== SZ-C: 12L dim512 mlp3.5 kv4 ==="
eval "$BASE_SIZE MLP_MULT=3.5" python -u train_gpt.py 2>&1 | tee logs/sz_C_mlp350.log
echo ""

echo "=== SZ-D: 12L dim512 mlp3.75 kv4 ==="
eval "$BASE_SIZE MLP_MULT=3.75" python -u train_gpt.py 2>&1 | tee logs/sz_D_mlp375.log
echo ""

echo "=== SZ-E: 12L dim512 mlp3.0 kv8 ==="
eval "$BASE_SIZE NUM_KV_HEADS=8" python -u train_gpt.py 2>&1 | tee logs/sz_E_kv8.log
echo ""

echo "=== SZ-F: 12L dim512 mlp3.25 kv8 ==="
eval "$BASE_SIZE MLP_MULT=3.25 NUM_KV_HEADS=8" python -u train_gpt.py 2>&1 | tee logs/sz_F_mlp325_kv8.log
echo ""

echo "=== SZ-G: 12L dim576 mlp3.0 kv4 ==="
eval "$BASE_SIZE MODEL_DIM=576" python -u train_gpt.py 2>&1 | tee logs/sz_G_dim576.log
echo ""

echo "=== SZ-H: 13L dim512 mlp2.5 kv4 ==="
eval "$BASE_SIZE NUM_LAYERS=13 MLP_MULT=2.5" python -u train_gpt.py 2>&1 | tee logs/sz_H_13L_mlp25.log
echo ""

echo "=== SZ-I: 12L dim576 mlp3.25 kv4 ==="
eval "$BASE_SIZE MODEL_DIM=576 MLP_MULT=3.25" python -u train_gpt.py 2>&1 | tee logs/sz_I_dim576_mlp325.log
echo ""

echo "=== ALL SIZE CHECKS COMPLETE ==="
echo "Finished: $(date)"
