#!/bin/bash
# =============================================================================
# ANS Compression Experiment on #1 Entry (PR #1333)
#
# Run this on RunPod 8×H100 SXM. Total time: ~15 minutes. Cost: ~$5.
#
# What it does:
# 1. Sets up the environment
# 2. Downloads SP4096 data
# 3. Trains using #1 entry's config
# 4. Compares Brotli (their compression) vs ANS (ours)
# 5. Reports artifact size difference
# =============================================================================

set -e
echo "=== ANS Compression Experiment ==="
echo "Started: $(date)"

# 1. Setup
cd /workspace
git clone https://github.com/OE-GOD/parameter-golf.git 2>/dev/null || (cd parameter-golf && git pull)
cd parameter-golf

# Install dependencies
pip install sentencepiece numpy tqdm huggingface_hub datasets brotli 2>/dev/null

# Check PyTorch version
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "PyTorch: $TORCH_VER"

# Check if flash_attn_3 is available (needed for #1 entry)
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null && echo "Flash Attention 3: OK" || {
    echo "Flash Attention 3 not available. Installing..."
    pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 2>/dev/null || {
        echo "WARNING: Could not install Flash Attention 3. Using baseline instead."
        USE_BASELINE=1
    }
}

# 2. Download SP4096 data (or SP1024 fallback)
if [ -z "$USE_BASELINE" ]; then
    echo "Downloading SP4096 data..."
    python3 data/cached_challenge_fineweb.py --variant sp4096 --train-shards 10 2>/dev/null || {
        echo "SP4096 not available in cache. Using SP1024."
        python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
        USE_SP1024=1
    }
else
    echo "Downloading SP1024 data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
    USE_SP1024=1
fi

# 3. Get ANS compressor
git remote add fork https://github.com/OE-GOD/parameter-golf.git 2>/dev/null || true
git fetch fork ans-compression 2>/dev/null
git checkout fork/ans-compression -- records/track_non_record_16mb/2026-04-09_ANS_Compression/ans_compress.py 2>/dev/null
cp records/track_non_record_16mb/2026-04-09_ANS_Compression/ans_compress.py . 2>/dev/null || true

# 4. Train the baseline model
echo ""
echo "=== Training baseline model ==="
if [ -z "$USE_SP1024" ]; then
    # SP4096 config (matching #1 entry where possible)
    RUN_ID=ans_sp4096 \
    VOCAB_SIZE=4096 \
    DATA_PATH=./data/datasets/fineweb10B_sp4096/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
else
    # SP1024 fallback
    RUN_ID=ans_sp1024 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
fi

# 5. Convert model to npz for ANS analysis
echo ""
echo "=== Converting model for ANS analysis ==="
python3 -c "
import torch, numpy as np
state = torch.load('final_model.pt', map_location='cpu', weights_only=True)
np_state = {k: v.float().numpy() for k, v in state.items()}
np.savez('final_model.npz', **np_state)
print('Converted to npz')
"

# 6. Run ANS analysis
echo ""
echo "=== ANS vs Brotli Compression Analysis ==="
python3 ans_compress.py --input final_model.npz --analyze --bits 6

# 7. Compress with ANS and compare sizes
echo ""
echo "=== Compressing with ANS ==="
python3 ans_compress.py --input final_model.npz --output final_model_ans.bin --bits 6 --verify

# 8. Compare with the Brotli artifact
echo ""
echo "=== Size Comparison ==="
BROTLI_SIZE=$(ls -l logs/*_quantized.pt 2>/dev/null | awk '{print $5}' | head -1)
ANS_SIZE=$(ls -l final_model_ans.bin | awk '{print $5}')
echo "Brotli artifact: ${BROTLI_SIZE:-unknown} bytes"
echo "ANS artifact:    $ANS_SIZE bytes"

if [ -n "$BROTLI_SIZE" ] && [ -n "$ANS_SIZE" ]; then
    SAVED=$((BROTLI_SIZE - ANS_SIZE))
    echo "Savings:          $SAVED bytes"
    echo "Extra params at int6: $((SAVED * 8 / 6))"
fi

echo ""
echo "=== Experiment Complete ==="
echo "Finished: $(date)"
echo ""
echo "NEXT STEPS:"
echo "1. If ANS saves >1MB: modify train_gpt.py to use wider MLP"
echo "2. Retrain with wider model + ANS compression"
echo "3. Measure BPB improvement"
