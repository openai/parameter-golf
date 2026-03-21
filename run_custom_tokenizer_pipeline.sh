#!/bin/bash
# =============================================================================
# Parameter Golf: Custom Tokenizer + Competitive Run (All-in-One)
# =============================================================================
#
# Steps:
#   1. Download docs_selected.jsonl (~45GB, 10-30 min)
#   2. Train unigram tokenizer (5-10 min)
#   3. Export binary shards (30-60 min)
#   4. Run SOTA training with custom tokenizer (10 min)
#
# Usage (paste into RunPod terminal):
#
#   git clone https://github.com/User123331/parameter-golf.git
#   cd parameter-golf
#   git pull
#   bash run_custom_tokenizer_pipeline.sh
#
# =============================================================================

set -e

VOCAB_SIZE=1024
MODEL_TYPE=unigram
MAX_TRAIN_DOCS=200000
EVAL_DOCS=10000
HF_TOKEN="${HF_TOKEN:-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}"

DATA_DIR="./data/datasets"
TOKENIZER_DIR="./data/tokenizers_custom"
DOCS_JSONL="${DATA_DIR}/docs_selected.jsonl"
CUSTOM_SHARDS="${DATA_DIR}/fineweb10B_custom_${MODEL_TYPE}${VOCAB_SIZE}"
CUSTOM_MODEL="${TOKENIZER_DIR}/spm_${MODEL_TYPE}_${VOCAB_SIZE}.model"

GREEN='\033[0;32m'
NC='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }

# =============================================================================
# Step 1: Download docs_selected.jsonl
# =============================================================================
log "Step 1: Downloading docs_selected.jsonl (~45GB)..."

mkdir -p "${DATA_DIR}"

if [ ! -f "${DOCS_JSONL}" ]; then
    pip install --quiet huggingface_hub

    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
cached = hf_hub_download(
    repo_id='willdepueoai/parameter-golf',
    filename='docs_selected.jsonl',
    subfolder='datasets',
    repo_type='dataset',
)
src = os.path.realpath(cached)
dst = '${DOCS_JSONL}'
print(f'Copying to {dst}')
try:
    os.link(src, dst)
except OSError:
    shutil.copy2(src, dst)
"
fi

log "Docs ready: $(du -h "${DOCS_JSONL}" 2>/dev/null | cut -f1)"

# =============================================================================
# Step 2: Train custom tokenizer
# =============================================================================
log "Step 2: Training ${MODEL_TYPE} tokenizer..."

mkdir -p "${TOKENIZER_DIR}"
pip install --quiet sentencepiece numpy

python3 data/train_tokenizer.py \
    --vocab-size ${VOCAB_SIZE} \
    --model-type ${MODEL_TYPE} \
    --docs-path "${DOCS_JSONL}" \
    --max-docs ${MAX_TRAIN_DOCS} \
    --eval-docs ${EVAL_DOCS} \
    --character-coverage 0.995

log "Tokenizer ready: ${CUSTOM_MODEL}"

# =============================================================================
# Step 3: Export binary shards
# =============================================================================
log "Step 3: Exporting binary shards (30-60 min)..."

python3 data/train_tokenizer.py \
    --vocab-size ${VOCAB_SIZE} \
    --model-type ${MODEL_TYPE} \
    --docs-path "${DOCS_JSONL}" \
    --export-shards \
    --shard-output-dir "${CUSTOM_SHARDS}"

log "Shards ready: ${CUSTOM_SHARDS}"
log "Train shards: $(ls ${CUSTOM_SHARDS}/fineweb_train_*.bin 2>/dev/null | wc -l | tr -d ' ')"
log "Val shards:   $(ls ${CUSTOM_SHARDS}/fineweb_val_*.bin 2>/dev/null | wc -l | tr -d ' ')"

# =============================================================================
# Step 4: Run training with custom tokenizer
# =============================================================================
log "Step 4: Running SOTA training..."

pip install --quiet zstandard

NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
[ -z "$NPROC" ] || [ "$NPROC" -lt 1 ] && NPROC=1

RUN_ID="custom_${MODEL_TYPE}${VOCAB_SIZE}_$(date +%Y%m%d_%H%M%S)" \
DATA_PATH="${CUSTOM_SHARDS}" \
TOKENIZER_PATH="${CUSTOM_MODEL}" \
VOCAB_SIZE=${VOCAB_SIZE} \
SEED=42 \
NUM_LAYERS=10 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3.0 \
TIE_EMBEDDINGS=1 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
WEIGHT_DECAY=0.04 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
GRAD_CLIP_NORM=0.3 \
BIGRAM_VOCAB_SIZE=10240 \
BIGRAM_DIM=128 \
SWA_ENABLED=1 \
SWA_START_FRAC=0.4 \
SWA_EVERY=50 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=32 \
torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | tee /workspace/custom_tok_train.log

log "Done!"
grep -E 'val_bpb|final_int8' /workspace/custom_tok_train.log | tail -5