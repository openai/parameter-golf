#!/usr/bin/env bash
# Set up /fscratch with parameter-golf data for low-latency I/O
# Run ONCE from a Pegasus login node before training.
#
# /fscratch is a fast local-ish filesystem on Pegasus.
# /netscratch is BeeGFS (high throughput, higher latency).
# RunPod uses local NVMe which is similar to /fscratch.
# This helps close the Pegasus-vs-RunPod performance gap.

set -euo pipefail

SRC="/netscratch/${USER}/parameter-golf/data"
DST="/fscratch/${USER}/parameter-golf-data"

echo "=== Setting up /fscratch for parameter-golf ==="

# Check source exists
if [ ! -d "${SRC}/datasets/fineweb10B_sp1024" ]; then
    echo "ERROR: Source data not found at ${SRC}/datasets/fineweb10B_sp1024"
    echo "First download it: cd /netscratch/${USER}/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10"
    exit 1
fi

# Check /fscratch exists
if [ ! -d "/fscratch/${USER}" ]; then
    echo "ERROR: /fscratch/${USER} does not exist. /fscratch may not be available on this cluster."
    exit 1
fi

# Create destination
mkdir -p "${DST}"

# Copy datasets
echo "Copying datasets to /fscratch (this may take a few minutes)..."
rsync -av --progress "${SRC}/datasets/" "${DST}/datasets/"

# Copy tokenizers
echo "Copying tokenizers..."
rsync -av --progress "${SRC}/tokenizers/" "${DST}/tokenizers/"

# Verify
echo ""
echo "=== Verification ==="
echo "Train shards: $(ls ${DST}/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)"
echo "Val files:    $(ls ${DST}/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)"
echo "Tokenizer:    $([ -f ${DST}/tokenizers/fineweb_1024_bpe.model ] && echo OK || echo MISSING)"
echo ""
echo "Total size:"
du -sh "${DST}"
echo ""
echo "Done. The optimized launcher will auto-detect this path."
