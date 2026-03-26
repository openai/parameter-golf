#!/usr/bin/env bash
# setup_pod_ttt_sweep.sh — Prepare RunPod for TTT calibration sweep
#
# Upload to pod and run:
#   scp -i ~/.ssh/id_ed25519_apollo scripts/setup_pod_ttt_sweep.sh root@POD:/workspace/
#   ssh -i ~/.ssh/id_ed25519_apollo root@POD "bash /workspace/setup_pod_ttt_sweep.sh"
#
# Or pipe via SSH:
#   cat scripts/setup_pod_ttt_sweep.sh | ssh -tt -i ~/.ssh/id_ed25519_apollo root@POD

set -euo pipefail

WORKSPACE="/workspace/parameter-golf"
FLASH_ATTN_DIR="/workspace/parameter-golf/flash-attention/hopper"
CHECKPOINT="final_model.int6.ptz"

echo "============================================"
echo "  RunPod TTT Sweep Setup"
echo "  $(date)"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Get into workspace
# ---------------------------------------------------------------------------
cd /workspace
if [ ! -d "$WORKSPACE" ]; then
    echo "ERROR: $WORKSPACE not found"
    exit 1
fi
cd "$WORKSPACE"
echo "==> Working directory: $(pwd)"

# ---------------------------------------------------------------------------
# Step 2: Git pull latest (get sweep scripts)
# ---------------------------------------------------------------------------
echo ""
echo "==> Pulling latest from experiments/pr374-edge..."
git fetch origin 2>&1 | tail -3
git checkout experiments/pr374-edge 2>&1 | tail -3
git pull origin experiments/pr374-edge 2>&1 | tail -3
echo "    HEAD: $(git log --oneline -1)"

# ---------------------------------------------------------------------------
# Step 3: Verify flash-attention hopper build
# ---------------------------------------------------------------------------
echo ""
echo "==> Checking flash-attn (Hopper)..."
if [ -d "$FLASH_ATTN_DIR" ]; then
    export PYTHONPATH="${FLASH_ATTN_DIR}:${PYTHONPATH:-}"
    python3 -c "from flash_attn_interface import flash_attn_func; print('  flash_attn_interface OK')" 2>&1 || {
        echo "  WARNING: flash_attn_interface import failed"
        echo "  Trying to rebuild..."
        cd "$FLASH_ATTN_DIR" && pip install -e . 2>&1 | tail -3
        cd "$WORKSPACE"
    }
else
    echo "  WARNING: $FLASH_ATTN_DIR not found"
    echo "  Trying pip install flash-attn..."
    pip install flash-attn --no-build-isolation 2>&1 | tail -5
    # Create shim for flash_attn_interface
    python3 -c "
import sys, os
shim = 'from flash_attn.flash_attn_interface import flash_attn_func\n'
site = [p for p in sys.path if 'site-packages' in p and os.path.isdir(p)][0]
with open(os.path.join(site, 'flash_attn_interface.py'), 'w') as f:
    f.write(shim)
print('  flash_attn_interface shim created')
"
fi

# ---------------------------------------------------------------------------
# Step 4: Verify deps
# ---------------------------------------------------------------------------
echo ""
echo "==> Checking Python deps..."
python3 -c "
import torch, sentencepiece, zstandard, numpy
print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')
print(f'  sentencepiece OK, zstandard OK, numpy OK')
" 2>&1

# ---------------------------------------------------------------------------
# Step 5: Verify checkpoint
# ---------------------------------------------------------------------------
echo ""
echo "==> Checking checkpoint..."
if [ -f "$WORKSPACE/$CHECKPOINT" ]; then
    SIZE=$(ls -lh "$WORKSPACE/$CHECKPOINT" | awk '{print $5}')
    echo "  $CHECKPOINT: $SIZE"
else
    echo "  WARNING: $CHECKPOINT not found!"
    echo "  Available model files:"
    ls -lh "$WORKSPACE"/final_model* 2>/dev/null || echo "    (none)"
    echo ""
    echo "  If checkpoint is missing, you need to re-run training or restore from backup."
    echo "  Check: ls -lh checkpoints/ or look for .pt / .ptz files"
fi

# ---------------------------------------------------------------------------
# Step 6: Verify val data
# ---------------------------------------------------------------------------
echo ""
echo "==> Checking val data..."
VAL_PATTERN="$WORKSPACE/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"
VAL_COUNT=$(ls $VAL_PATTERN 2>/dev/null | wc -l)
if [ "$VAL_COUNT" -gt 0 ]; then
    VAL_SIZE=$(du -sh "$WORKSPACE/data/datasets/fineweb10B_sp1024/" | awk '{print $1}')
    echo "  Val shards: $VAL_COUNT files ($VAL_SIZE)"
else
    echo "  WARNING: No val data found at $VAL_PATTERN"
fi

TOK="$WORKSPACE/data/tokenizers/fineweb_1024_bpe.model"
if [ -f "$TOK" ]; then
    echo "  Tokenizer: OK"
else
    echo "  WARNING: Tokenizer not found at $TOK"
fi

# ---------------------------------------------------------------------------
# Step 7: Dry-run import test
# ---------------------------------------------------------------------------
echo ""
echo "==> Import test..."
cd "$WORKSPACE"
PYTHONPATH="${FLASH_ATTN_DIR}:${PYTHONPATH:-}" python3 -c "
import ttt_eval_runner
print('  ttt_eval_runner.py imports OK')
print(f'  Model: {ttt_eval_runner.Hyperparameters.num_layers}L {ttt_eval_runner.Hyperparameters.model_dim}d')
" 2>&1 || echo "  WARNING: import test failed (may need PYTHONPATH fix)"

# ---------------------------------------------------------------------------
# Step 8: Create logs dir + show run command
# ---------------------------------------------------------------------------
echo ""
mkdir -p "$WORKSPACE/logs"

echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo ""
echo "  To run the 11-config TTT sweep (~45 min):"
echo ""
echo "    cd $WORKSPACE"
echo "    export PYTHONPATH=\"${FLASH_ATTN_DIR}:\${PYTHONPATH:-}\""
echo "    bash sweep_ttt_calibration.sh"
echo ""
echo "  Or run a single config manually:"
echo ""
echo "    EVAL_ONLY=1 CHECKPOINT_PATH=final_model.int6.ptz \\"
echo "    TTT_MAX_TRAIN_CHUNKS=40 TTT_EMA_DECAY=0 TTT_FREEZE_BLOCKS=2 \\"
echo "    torchrun --standalone --nproc_per_node=8 ttt_eval_runner.py"
echo ""
echo "  Results will be in: logs/ttt_sweep_*/results.csv"
echo ""
echo "  When done, pull results locally:"
echo "    ./scripts/pull_from_pod.sh root@POD_IP ttt_sweep"
echo ""
