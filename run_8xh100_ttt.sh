#!/bin/bash
set -e
cd /workspace/parameter-golf

echo "=== Installing FlashAttention 3 ==="
pip install flash-attn --no-build-isolation -q 2>/dev/null || echo "flash-attn pip install failed, trying from source..."
# FA3 needs the flash_attn_interface module
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')" 2>/dev/null || {
    echo "FA3 not available, will use SDPA fallback"
}

echo "=== Downloading data ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

echo "=== Checking PyTorch and GPU ==="
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

for SEED in 1337 42 2024; do
    echo "======================================================================"
    echo "=== SEED $SEED | $(date) ==="
    echo "======================================================================"
    NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
    EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
    ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 \
    MUON_WD=0.04 ADAM_WD=0.04 \
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
    MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
    ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
    SEED=$SEED RUN_ID=pr315_ttt_s${SEED} \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee run_seed${SEED}.txt
    echo "--- Seed $SEED complete ---"
done

echo "======================================================================"
echo "=== ALL SEEDS COMPLETE ==="
echo "======================================================================"
for f in run_seed*.txt; do
    echo "--- $f ---"
    grep -E "final_int6.*exact|sliding.*exact|ttt:|Total submission" "$f" | tail -5
done
echo "End: $(date)"
