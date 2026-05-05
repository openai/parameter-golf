#!/bin/bash
# Run this once when deploying a new pod.
# Downloads SP4096 data if not already present, installs deps.

set -euo pipefail

echo "=== Installing dependencies ==="
pip install -q sentencepiece huggingface_hub numpy torch==2.5.1 brotli

echo "=== Checking SP1024 data ==="
if [ -d /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 ]; then
    echo "SP1024 data: OK"
else
    echo "SP1024 data: MISSING — downloading..."
    cd /workspace/parameter-golf
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

echo "=== Checking SP4096 data ==="
if [ -d /workspace/parameter-golf/data/datasets/fineweb10B_sp4096 ]; then
    echo "SP4096 data: OK"
else
    echo "SP4096 data: MISSING — downloading..."
    cd /workspace/parameter-golf
    python3 data/cached_challenge_fineweb.py --variant sp4096
    if [ $? -ne 0 ]; then
        echo "Default repo failed, trying kevclark/parameter-golf..."
        MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096
    fi
fi

echo "=== Setup complete ==="
echo "Datasets available:"
ls /workspace/parameter-golf/data/datasets/
echo "Tokenizers available:"
ls /workspace/parameter-golf/data/tokenizers/
