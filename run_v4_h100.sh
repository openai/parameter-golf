#!/bin/bash
# Run v4 looped architecture on 8xH100
# Usage: ssh into RunPod pod, then run this script

cd /workspace/parameter-golf
git pull -q

echo "$(date) SETUP"
pip install -q --upgrade torch zstandard huggingface-hub datasets sentencepiece tqdm 2>&1 | tail -1

echo "$(date) DATA" 
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 2>&1 | tail -1

echo "$(date) TRAIN"
RUN_ID=v4_looped SEED=42 \
NUM_LAYERS=8 LOOP_COUNT=2 USE_INPUT_FEATURES=1 \
XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3000 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 experiments/v4_looped_features.py 2>&1 | tee /workspace/v4_run.log

echo "$(date) DONE"
echo "=== RESULTS ==="
grep -E "final_|stopping|Serialized|Total" /workspace/parameter-golf/logs/v4_looped.txt 2>/dev/null
