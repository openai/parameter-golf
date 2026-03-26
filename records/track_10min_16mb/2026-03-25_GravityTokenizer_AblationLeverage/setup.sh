#!/bin/bash
# -------------------------------------------------------------------------------
# Gravity Tokenizer — Setup Script
# Downloads stock FineWeb, then retokenizes with the gravity vocabulary.
# -------------------------------------------------------------------------------

set -e

echo "----------------------------------------------"
echo " Gravity Tokenizer — Setup"
echo "----------------------------------------------"

# -------------------------------------------------------------------------------
# 1. Dependencies
# -------------------------------------------------------------------------------
echo ""
echo "[1/3] Checking dependencies..."

pip install --quiet sentencepiece numpy huggingface-hub tqdm

echo "    Done."

# -------------------------------------------------------------------------------
# 2. Download stock BPE FineWeb (competition data)
# -------------------------------------------------------------------------------
echo ""
echo "[2/3] Downloading stock FineWeb (sp1024)..."

STOCK_DIR=./data/datasets/fineweb10B_sp1024
STOCK_TOK=./data/tokenizers/fineweb_1024_bpe.model

if [ -d "$STOCK_DIR" ] && ls "$STOCK_DIR"/fineweb_train_*.bin 1>/dev/null 2>&1; then
    TRAIN_COUNT=$(ls "$STOCK_DIR"/fineweb_train_*.bin | wc -l)
    echo "    Found $TRAIN_COUNT existing train shards — skipping download."
else
    python3 data/cached_challenge_fineweb.py --variant sp1024
    echo "    Downloaded."
fi

# -------------------------------------------------------------------------------
# 3. Retokenize with gravity vocabulary
# -------------------------------------------------------------------------------
echo ""
echo "[3/3] Retokenizing with gravity tokenizer..."

GRAVITY_DIR=./data/datasets/fineweb_gravity_beta_1.0
GRAVITY_TOK=./data/tokenizers/gravity_beta_1.0.model

# Copy gravity tokenizer model into expected location
mkdir -p ./data/tokenizers
if [ ! -f "$GRAVITY_TOK" ]; then
    # Download from HuggingFace if not bundled
    if [ -f "$(dirname "$0")/gravity_beta_1.0.model" ]; then
        cp "$(dirname "$0")/gravity_beta_1.0.model" "$GRAVITY_TOK"
    else
        huggingface-cli download dcrow85/gravity-tokenizer-fineweb \
            tokenizers/gravity_beta_1.0.model \
            --repo-type dataset \
            --local-dir ./data
    fi
fi

if [ -d "$GRAVITY_DIR" ] && ls "$GRAVITY_DIR"/fineweb_val_*.bin 1>/dev/null 2>&1; then
    TRAIN_COUNT=$(ls "$GRAVITY_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
    echo "    Found $TRAIN_COUNT existing gravity shards — skipping retokenization."
else
    python3 retokenize_corpus.py \
        --base-tokenizer "$STOCK_TOK" \
        --gravity-tokenizer "$GRAVITY_TOK" \
        --data-dir "$STOCK_DIR" \
        --output-dir "$GRAVITY_DIR"
    echo "    Retokenization complete."
fi

# -------------------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------------------
echo ""
echo "----------------------------------------------"
echo " Verification"
echo "----------------------------------------------"

python3 -c "
import os, glob, numpy as np, sentencepiece as spm

train = sorted(glob.glob('$GRAVITY_DIR/fineweb_train_*.bin'))
val = sorted(glob.glob('$GRAVITY_DIR/fineweb_val_*.bin'))
print(f'Gravity train shards : {len(train)}')
print(f'Gravity val shards   : {len(val)}')

total_tokens = 0
for f in train + val:
    header = np.fromfile(f, dtype='<i4', count=256)
    tokens = int(header[2])
    total_tokens += tokens

print(f'Total tokens         : {total_tokens:,}')

sp = spm.SentencePieceProcessor(model_file='$GRAVITY_TOK')
print(f'Tokenizer vocab      : {sp.vocab_size()}')

test = 'The water because caused the damage'
ids = sp.encode(test)
pieces = [sp.id_to_piece(i) for i in ids]
crystals = [p for p in pieces if not p.startswith('<0x')]
print(f'Crystal tokens in test: {crystals}')
"

echo ""
echo "----------------------------------------------"
echo " Done. Run training with:"
echo ""
echo "   MODEL_DIM=384 NUM_LAYERS=12 NUM_HEADS=6 NUM_KV_HEADS=2 MLP_MULT=3 \\"
echo "   TRAIN_SEQ_LEN=2048 VOCAB_SIZE=1024 \\"
echo "   DATA_PATH=./data/datasets/fineweb_gravity_beta_1.0 \\"
echo "   TOKENIZER_PATH=./data/tokenizers/gravity_beta_1.0.model \\"
echo "   ITERATIONS=11000 WARMUP_STEPS=50 WARMDOWN_ITERS=2500 \\"
echo "   MAX_WALLCLOCK_SECONDS=600 SEED=1337 \\"
echo "   torchrun --standalone --nproc_per_node=8 train_gpt.py"
echo "----------------------------------------------"
