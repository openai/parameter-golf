#!/bin/bash
# Setup script for 8xH100 production instance
set -e

# Fix Thunder GPU access
echo '12b0cf078a6ab489354a33bcc714dec1401bf56a42c211c68cd7d86e06581cda' > ~/.thunder/token
ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so 2>/dev/null
ldconfig 2>/dev/null

# Clone repo and install deps
cd /home/ubuntu
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
pip install sentencepiece numpy tqdm huggingface-hub datasets wandb 2>&1 | tail -3

# Download full data (80 shards for competition-accurate results)
python3 data/cached_challenge_fineweb.py --variant sp1024 2>&1 | tail -3

echo "=== SETUP COMPLETE ==="
nvidia-smi
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.version.cuda}')"
