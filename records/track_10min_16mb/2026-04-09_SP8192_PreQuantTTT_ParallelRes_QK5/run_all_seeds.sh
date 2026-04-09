#!/bin/bash
# Run 011: SP8192 + Pre-quant TTT + Parallel Residuals + QK5
# 3 seeds for statistical significance

set -e

RUN_DIR="records/track_10min_16mb/2026-04-09_SP8192_PreQuantTTT_ParallelRes_QK5"
mkdir -p $RUN_DIR

echo "=== Run 011: SP8192 + Pre-quant TTT + Parallel Residuals ==="
echo "Starting 3-seed run at $(date)"

# Seed 42
echo "=== Seed 42 ==="
RUN_ID=run011_s42 \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
TRAIN_SEQ_LEN=2048 \
QK_GAIN_INIT=5.0 \
PREQUANT_TTT_ENABLED=1 \
PREQUANT_TTT_LR=0.0005 \
PREQUANT_TTT_EPOCHS=6 \
PREQUANT_TTT_FREEZE_BLOCKS=2 \
EMA_DECAY=0.9965 \
GPTQ_ENABLED=1 \
SLIDING_WINDOW_ENABLED=1 \
ETLB_ENABLED=1 \
MAX_WALLCLOCK_SECONDS=588 \
python3 train_gpt.py 2>&1 | tee $RUN_DIR/train_s42.log

# Seed 314
echo "=== Seed 314 ==="
RUN_ID=run011_s314 \
SEED=314 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
TRAIN_SEQ_LEN=2048 \
QK_GAIN_INIT=5.0 \
PREQUANT_TTT_ENABLED=1 \
PREQUANT_TTT_LR=0.0005 \
PREQUANT_TTT_EPOCHS=6 \
PREQUANT_TTT_FREEZE_BLOCKS=2 \
EMA_DECAY=0.9965 \
GPTQ_ENABLED=1 \
SLIDING_WINDOW_ENABLED=1 \
ETLB_ENABLED=1 \
MAX_WALLCLOCK_SECONDS=588 \
python3 train_gpt.py 2>&1 | tee $RUN_DIR/train_s314.log

# Seed 999
echo "=== Seed 999 ==="
RUN_ID=run011_s999 \
SEED=999 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
TRAIN_SEQ_LEN=2048 \
QK_GAIN_INIT=5.0 \
PREQUANT_TTT_ENABLED=1 \
PREQUANT_TTT_LR=0.0005 \
PREQUANT_TTT_EPOCHS=6 \
PREQUANT_TTT_FREEZE_BLOCKS=2 \
EMA_DECAY=0.9965 \
GPTQ_ENABLED=1 \
SLIDING_WINDOW_ENABLED=1 \
ETLB_ENABLED=1 \
MAX_WALLCLOCK_SECONDS=588 \
python3 train_gpt.py 2>&1 | tee $RUN_DIR/train_s999.log

echo "=== All seeds completed at $(date) ==="
