# 11L XSA11 + BigramHash3072 + AdamW Legal TTT

**val_bpb: 1.1157** (single-seed legal TTT) | **15,983,339 bytes** | **2xH100, 2400s**

## Results

| Seed | Steps | ms/step | Roundtrip bpb | Sliding bpb (s64) | Legal TTT bpb | Artifact |
|------|-------|---------|---------------|-------------------|---------------|----------|
| 1337 | 7,287 | 329.4 | 1.1412 | 1.1176 | **1.1157** | 15,983,339 bytes |

## Model

- 11 layers, 512 dim, 8 heads, 4 KV heads
- 3x MLP with LeakyReLU(0.5)^2
- BigramHash with 3072 buckets and dim 112
- XSA enabled on all 11 layers
- Tied embeddings, partial RoPE, LN scale, VE128 on layers 9-10
- Parameter banks with Parallel Muon for matrix weights
- EMA + tight SWA + late QAT

## Training

- 2xH100 run with 2400s wallclock cap
- Global batch 786,432 tokens, seq_len 2048
- Muon momentum 0.99 with warmup, matrix lr 0.025
- AdamW for embeddings/scalars, tied embed lr 0.035, scalar lr 0.025
- FlashAttention-3 on Hopper (`flash_attn3:sm90`)
- Training stopped at step 7,287 by wallclock cap

## Evaluation

- Int6 + lzma export with `TARGET_MB=15.90`
- Sliding window eval at stride 64: `1.11762571`
- Score-first legal TTT with AdamW
- Chunk size 131,072 tokens, 3 epochs per chunk
- Freeze first 8 blocks during TTT
- Final legal TTT score: `1.11565196`

## Setup

```bash
pip install -r requirements.txt --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2100
```

## Run Command

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
RUN_ID=xsa11_bigram3072_adamwttt_seed1337 \
SEED=1337 \
BIGRAM_VOCAB_SIZE=3072 \
BIGRAM_DIM=112 \
XSA_LAST_N=11 \
TARGET_MB=15.90 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=2400 \
TRAIN_BATCH_TOKENS=786432 \
VAL_LOSS_EVERY=999999 \
EVAL_STRIDE=64 \
TTT_ENABLED=1 \
TTT_OPTIMIZER=adamw \
TTT_LR=0.0001 \
TTT_MOMENTUM=0.9 \
TTT_WEIGHT_DECAY=0.01 \
TTT_EPOCHS=3 \
TTT_FREEZE_BLOCKS=8 \
TTT_BATCH_SEQS=32 \
TTT_CHUNK_TOKENS=131072 \
TTT_GRAD_CLIP=1.0 \
FLASH_ATTN3_MODE=auto \
torchrun --standalone --nproc_per_node=2 train_gpt.py
```
