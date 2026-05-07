# R8 10L QK35 SW64 Mixed Int6

Best run:

- Seed: `42`
- `final_mixed_roundtrip_exact val_bpb`: **1.18275114**
- Total submission size mixed: **15,554,319 bytes**
- Training time: **587,611 ms**
- Hardware: **8x H100**

Three-seed summary:

| Seed | final_mixed_roundtrip_exact val_bpb | Total submission size mixed |
|---:|---:|---:|
| 1337 | 1.18440304 | 15,528,382 bytes |
| 42 | 1.18275114 | 15,554,319 bytes |
| 2024 | 1.18398094 | 15,528,395 bytes |

Mean over 3 seeds: **1.18371171**.

## Method

This submission combines a 10-layer compact transformer, GQA attention, tied embeddings, bigram hash embedding, U-Net-style skip connections, late-layer MLP factorization, dynamic optimizer scheduling, a short healing phase, QK gain initialization set to `3.5`, sliding-window evaluation with `VAL_STRIDE=64`, and mixed int6/int8 export compressed with zstd.

## Reproduction

```bash
RUN_ID=FULL_R8_10L_MIXED_INT6_QK35_SW64_11000_SEED42 \
SEED=42 \
EXPORT_FORMAT=mixed_int6 \
USE_SMEAR=0 \
USE_BIGRAM=1 \
QK_GAIN_INIT=3.5 \
NUM_LAYERS=10 \
VAL_STRIDE=64 \
MID_STEP=1600 \
MID_WEIGHT_DECAY=0.008 \
MID_MUON_WEIGHT_DECAY=0.008 \
MID_MATRIX_LR_SCALE=0.50 \
HEALING_STEPS=2500 \
HEALING_LR_SCALE=0.12 \
HEALING_WD_SCALE=0.25 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=11000 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=524288 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
WEIGHT_DECAY=0.04 \
MUON_WEIGHT_DECAY=0.04 \
FACTORIZED_LATE_LAYERS=2 \
FACTORIZED_RANK=192 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
