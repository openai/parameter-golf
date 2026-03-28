#!/usr/bin/env bash
# Pegasus environment diagnostic for parameter-golf
# Run this from a Pegasus login node BEFORE allocating GPUs.
# Usage: bash scripts/pegasus_diagnostic.sh

set -euo pipefail

echo "=== Pegasus Parameter-Golf Diagnostic ==="
echo "Date: $(date -Iseconds)"
echo "User: $USER"
echo "Hostname: $(hostname)"
echo ""

# 1. NGC container images available
echo "--- Available NGC PyTorch images ---"
if [ -d /enroot ]; then
    ls -1 /enroot/nvcr.io_nvidia_pytorch_*.sqsh 2>/dev/null | sort -V | tail -10
    echo ""
    echo "Latest image:"
    ls -1 /enroot/nvcr.io_nvidia_pytorch_*.sqsh 2>/dev/null | sort -V | tail -1
else
    echo "WARNING: /enroot not found — container images not available"
fi
echo ""

# 2. Check CUDA/PyTorch versions in bare-metal env
echo "--- Bare-metal Python/PyTorch/CUDA ---"
python3 -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
        print(f'GPU count: {torch.cuda.device_count()}')
except ImportError:
    print('PyTorch not importable in bare-metal env')
try:
    import zstandard
    print(f'zstandard: {zstandard.__version__}')
except ImportError:
    print('zstandard: NOT INSTALLED')
" 2>&1 || echo "Python check failed"
echo ""

# 3. Storage paths
echo "--- Storage availability ---"
echo "/netscratch/$USER exists: $([ -d /netscratch/$USER ] && echo YES || echo NO)"
echo "/fscratch/$USER exists: $([ -d /fscratch/$USER ] && echo YES || echo NO)"
if [ -d /netscratch/$USER ]; then
    echo "/netscratch/$USER usage:"
    du -sh /netscratch/$USER 2>/dev/null || echo "  (could not check)"
fi
if [ -d /fscratch/$USER ]; then
    echo "/fscratch/$USER usage:"
    du -sh /fscratch/$USER 2>/dev/null || echo "  (could not check)"
fi
echo ""

# 4. Check data availability
echo "--- Parameter-golf data ---"
REPO_NETSCRATCH="/netscratch/$USER/parameter-golf"
if [ -d "$REPO_NETSCRATCH" ]; then
    echo "Repo on netscratch: YES ($REPO_NETSCRATCH)"
    echo "  Data shards: $(ls $REPO_NETSCRATCH/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l) files"
    echo "  Val data: $(ls $REPO_NETSCRATCH/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l) files"
    echo "  Tokenizer: $([ -f $REPO_NETSCRATCH/data/tokenizers/fineweb_1024_bpe.model ] && echo YES || echo NO)"
else
    echo "Repo on netscratch: NO — run: cd /netscratch/$USER && git clone <repo-url> parameter-golf"
fi
echo ""

# 5. Check fscratch for potential data copy
FSCRATCH_DATA="/fscratch/$USER/parameter-golf-data"
if [ -d "$FSCRATCH_DATA" ]; then
    echo "Fast data on fscratch: YES ($FSCRATCH_DATA)"
    echo "  Shards: $(ls $FSCRATCH_DATA/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l) files"
else
    echo "Fast data on fscratch: NO"
    echo "  To set up: mkdir -p $FSCRATCH_DATA && cp -r $REPO_NETSCRATCH/data/datasets $FSCRATCH_DATA/ && cp -r $REPO_NETSCRATCH/data/tokenizers $FSCRATCH_DATA/"
fi
echo ""

# 6. PyPI cache
echo "--- Pegasus PyPI cache ---"
curl -s --connect-timeout 3 http://pypi-cache/index/ >/dev/null 2>&1 && echo "PyPI cache: REACHABLE" || echo "PyPI cache: NOT REACHABLE (use regular pip)"
echo ""

# 7. Partition availability
echo "--- H100/H200 partition status ---"
sinfo -p H100 --noheader -o "%P %a %D %T %N" 2>/dev/null || echo "sinfo failed (may need to be on login node)"
echo ""
sinfo -p H200 --noheader -o "%P %a %D %T %N" 2>/dev/null || echo "H200 partition not available or sinfo failed"
echo ""

# 8. Check if container launch works (dry run)
echo "--- Container launch check ---"
if [ -f /enroot/nvcr.io_nvidia_pytorch_24.05-py3.sqsh ]; then
    echo "NGC PyTorch 24.05 image: AVAILABLE"
elif ls /enroot/nvcr.io_nvidia_pytorch_*.sqsh >/dev/null 2>&1; then
    LATEST=$(ls -1 /enroot/nvcr.io_nvidia_pytorch_*.sqsh | sort -V | tail -1)
    echo "NGC PyTorch latest available: $LATEST"
else
    echo "No NGC PyTorch images found"
fi
echo ""

echo "=== Diagnostic complete ==="
echo ""
echo "RECOMMENDED NEXT STEPS:"
echo "1. If NGC images exist, use the latest one with --container-image"
echo "2. If /fscratch is available, copy data there for lower latency"
echo "3. Run: pip install zstandard (via PyPI cache if reachable)"
echo "4. Then run: bash scripts/pegasus_optimized_launcher.sh"
