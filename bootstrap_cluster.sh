#!/bin/bash
set -euo pipefail
# Bootstrap a fresh H100 cluster for parameter-golf.
# Usage: ssh ubuntu@<host> 'bash -s' < bootstrap_cluster.sh
# Or:    scp bootstrap_cluster.sh ubuntu@<host>: && ssh ubuntu@<host> bash bootstrap_cluster.sh
#
# Assumes: Ubuntu, NVIDIA drivers already installed, NFS volume at /data or mountable.

VENV=/data/pgolf_venv
REPO=/data/parameter-golf
DATA_DIR=$REPO/data/datasets/fineweb10B_sp1024
TOK_DIR=$REPO/data/tokenizers
UV="${UV:-$HOME/.local/bin/uv}"

echo "=== Step 0: Check GPUs ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "FATAL: no GPUs found"; exit 1; }
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $GPU_COUNT GPUs"

echo "=== Step 1: Mount NFS (if not already mounted) ==="
if ! mountpoint -q /data; then
    NFS_IP="${NFS_IP:-10.15.69.43}"
    echo "Mounting NFS from $NFS_IP..."
    sudo apt-get install -y -qq nfs-common > /dev/null 2>&1
    echo "${NFS_IP}:/data /data nfs rw,nconnect=16,nfsvers=3 0 0" | sudo tee -a /etc/fstab
    sudo mount -a
    echo "Mounted /data"
else
    echo "/data already mounted"
fi

echo "=== Step 2: Install uv (if needed) ==="
if ! command -v "$UV" &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
fi
echo "uv: $($UV --version)"

echo "=== Step 3: Create venv + install deps ==="
if [ ! -f "$VENV/bin/python" ]; then
    echo "Creating venv at $VENV..."
    $UV venv "$VENV" --python 3.10
fi
echo "Installing PyTorch + deps..."
$UV pip install --python "$VENV/bin/python" \
    torch --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || \
$UV pip install --python "$VENV/bin/python" torch
$UV pip install --python "$VENV/bin/python" \
    numpy tqdm huggingface-hub datasets tiktoken sentencepiece kernels "typing-extensions==4.15.0"

echo "=== Step 4: Verify torch + CUDA ==="
$VENV/bin/python -c "
import torch
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
assert torch.cuda.is_available(), 'CUDA not available!'
assert torch.cuda.device_count() > 0, 'No GPUs detected by torch!'
"

echo "=== Step 5: Clone/update repo ==="
if [ ! -d "$REPO/.git" ]; then
    echo "Cloning parameter-golf..."
    git clone https://github.com/openai/parameter-golf.git "$REPO"
else
    echo "Repo exists, pulling latest..."
    cd "$REPO" && git pull --ff-only || echo "Pull failed (local changes?), continuing..."
fi

echo "=== Step 6: Download dataset ==="
if [ -d "$DATA_DIR" ] && [ "$(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)" -ge 10 ]; then
    echo "Dataset already present ($DATA_DIR)"
else
    echo "Downloading fineweb10B_sp1024 dataset (full 80 shards)..."
    cd "$REPO"
    $VENV/bin/python data/cached_challenge_fineweb.py --variant sp1024
fi

echo "=== Step 7: Verify key paths ==="
ALL_OK=true
for f in "$REPO/train_gpt.py" "$DATA_DIR/fineweb_train_000000.bin" "$TOK_DIR/fineweb_1024_bpe.model"; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        ALL_OK=false
    fi
done

echo ""
echo "============================================================"
echo "Bootstrap complete."
echo "  Python:  $VENV/bin/python"
echo "  uv:      $UV"
echo "  Repo:    $REPO"
echo "  Data:    $DATA_DIR"
echo "  GPUs:    $GPU_COUNT"
if [ "$ALL_OK" = false ]; then
    echo "  WARNING: Some files missing, check above."
fi
echo ""
echo "Quick smoke test (1 GPU, 30s):"
echo "  cd $REPO && CUDA_VISIBLE_DEVICES=0 MAX_WALLCLOCK_SECONDS=30 \\"
echo "    $VENV/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py"
echo ""
echo "Launch stage3:"
echo "  cd $REPO && PGOLF_PYTHON=$VENV/bin/python nohup $VENV/bin/python stage3/orchestrate_stage3.py --phase all --label <name> > stage3/run.log 2>&1 & disown"
echo "============================================================"
