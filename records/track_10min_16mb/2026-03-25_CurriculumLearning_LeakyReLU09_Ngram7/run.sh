#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SEED="${SEED:-42}"

# Production-ready: PR #753 base + curriculum learning
export SEED
export SHARD_ORDER="${SHARD_ORDER:-44,63,65,42,18,67,30,69,61,3,13,19,50,49,56,45,73,79,57,32,28,68,66,34,46,38,17,77,0,14,26,74,59,62,41,9,58,22,78,4,48,8,12,27,75,36,16,43,52,15,33,47,25,55,54,23,37,51,31,21,60,1,20,72,24,53,39,35,71,76,40,5,10,2,7,6,70,11,64,29}"
# N-gram backoff defaults from PR #753
export NGRAM_EVAL_ORDER="${NGRAM_EVAL_ORDER:-7}"
# LeakyReLU slope 0.9 > 0.5 (MatoTeziTanka sweep, -0.013 BPP)
export MLP_LEAKY_SLOPE="${MLP_LEAKY_SLOPE:-0.9}"

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "GPUs: $NGPU | Seed: $SEED | Ngram: $NGRAM_EVAL_ORDER | Shard order: ${SHARD_ORDER:+yes}"

if [ "$NGPU" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt.py
else
    python train_gpt.py
fi
