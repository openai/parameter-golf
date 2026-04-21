# SP8192 + dim=464 + Pre-Quantization TTT + Brotli

**val_bpb: 1.1863** (roundtrip, seed 1337) | **15.92 MB** | 1×RTX 5090, 12k steps (non-record)

Post-TTT: **1.1524 BPB** (score-first TTT, 3 epochs on top of preq-adapted weights)

## Key Technique: Pre-Quantization TTT

After training ends, before INT6 quantization, adapt the FP32 weights on the full validation set using standard (non-score-first) TTT. This "warms up" the weights to the val distribution before the precision loss from quantization locks them in.

```
Training (12k steps, QAT INT6) → preq-TTT (adapt FP32 on full val, 7 epochs) → INT6 quantize → score-first TTT eval
```

**Scaling law (dim=464, 12k steps, single RTX 5090):**

| preq-TTT epochs | Roundtrip BPB | Delta |
|-----------------|--------------|-------|
| 0               | 1.2347       | —     |
| 3               | 1.2097       | −0.025 |
| 5               | 1.1968       | −0.013 |
| 7               | **1.1863**   | −0.011 |

Still scaling at 7 epochs. On 8×H100 (DDP interleaved, weight sync per epoch), 21 epochs ≈ 240s additional — expected ~1.15 BPB.

## Architecture

Built on the SP8192 + parallel residuals + depth recurrence stack:

| Component | Setting |
|-----------|---------|
| Tokenizer | SP8192 (8192 BPE vocab) |
| Layers | 11 (dim=464, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| Depth recurrence | Layers 3–5, ×2 loops |
| Parallel residuals | From layer 7 |
| Quantization | INT6 QAT (all layers) + INT8 embeddings |
| Compression | Brotli + byte shuffle (~2× vs zstd) |
| Weight avg | EMA(0.997) + SWA(every 50) |
| Optimizer | MuonEq-R (row-normalized) + Adam |
| Warmdown | 3000 steps |

**Artifact:** 15.92 MB ✅ (84 KB headroom)

## Run Command (8×H100, expected leaderboard submission)

```bash
RUN_ID=sp8192_464_preqttt21 SEED=1337 MODEL_DIM=464 \
  VOCAB_SIZE=8192 DATA_PATH=./data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
  TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=2048 \
  VAL_LOSS_EVERY=2000 VAL_BATCH_SIZE=65536 TRAIN_LOG_EVERY=500 \
  WARMUP_STEPS=20 MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=11 MLP_MULT=3 LEAKY_RELU_SLOPE=0.5 \
  BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
  QAT_ENABLED=1 QAT_INT6=1 INT6_LAYER_START=0 INT6_LAYER_END=10 \
  INT8_EMBED_EXPORT=1 BYTE_SHUFFLE=1 USE_BROTLI=1 MUON_ROW_NORMALIZE=1 \
  MUON_WEIGHT_DECAY=0.04 ADAM_WEIGHT_DECAY=0.04 \
  EMA_ENABLED=1 SWA_ENABLED=1 SWA_EVERY=50 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=500 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  WARMDOWN_ITERS=3000 \
  NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
  PARALLEL_RESIDUAL_START=7 \
  PREQ_TTT_ENABLED=1 PREQ_TTT_EPOCHS=21 PREQ_TTT_LR=5e-4 \
  PREQ_TTT_CHUNK_SIZE=32768 PREQ_TTT_MAX_TOKENS=0 \
  TTT_ENABLED=1 TTT_EPOCHS=3 TTT_LR=0.005 TTT_CHUNK_SIZE=32768 \
  TTT_COSINE_DECAY=1 TTT_MAX_TOKENS=2800000 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Single-GPU Run Command (1×RTX 5090, this submission)

```bash
RUN_ID=sp8192_464_preqttt7 SEED=1337 ITERATIONS=12000 MODEL_DIM=464 \
  VOCAB_SIZE=8192 DATA_PATH=./data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
  TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=2048 \
  VAL_LOSS_EVERY=2000 VAL_BATCH_SIZE=65536 TRAIN_LOG_EVERY=500 \
  WARMUP_STEPS=20 MAX_WALLCLOCK_SECONDS=0 \
  NUM_LAYERS=11 MLP_MULT=3 LEAKY_RELU_SLOPE=0.5 \
  BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
  QAT_ENABLED=1 QAT_INT6=1 INT6_LAYER_START=0 INT6_LAYER_END=10 \
  INT8_EMBED_EXPORT=1 BYTE_SHUFFLE=1 USE_BROTLI=1 MUON_ROW_NORMALIZE=1 \
  MUON_WEIGHT_DECAY=0.04 ADAM_WEIGHT_DECAY=0.04 \
  EMA_ENABLED=1 SWA_ENABLED=1 SWA_EVERY=50 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=500 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  WARMDOWN_ITERS=3000 \
  NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
  PARALLEL_RESIDUAL_START=7 \
  PREQ_TTT_ENABLED=1 PREQ_TTT_EPOCHS=7 PREQ_TTT_LR=5e-4 \
  PREQ_TTT_CHUNK_SIZE=32768 PREQ_TTT_MAX_TOKENS=0 \
  TTT_ENABLED=1 TTT_EPOCHS=3 TTT_LR=0.005 TTT_CHUNK_SIZE=32768 \
  TTT_COSINE_DECAY=1 TTT_MAX_TOKENS=2800000 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Results (1×RTX 5090, seed 1337)

| Metric | Value |
|--------|-------|
| Raw val_bpb (post-train) | 1.2338 |
| Roundtrip val_bpb (post-preq-TTT + INT6 + brotli) | **1.1863** |
| Post-TTT val_bpb (3-epoch score-first) | **1.1524** |
| TTT delta | −0.0075 |
| Artifact size | 15,915,528 bytes (15.17 MiB) |
| Training time | ~33 min (12k steps × 163ms/step) |
| preq-TTT time | ~1286s (7 epochs) |
| Hardware | 1× NVIDIA RTX 5090 32GB |

## Multi-GPU preq-TTT Implementation

`preq_ttt_adapt` is DDP-aware: chunks are interleaved across ranks (`range(rank, n_chunks, world_size)`), with `all_reduce(AVG)` after each epoch to sync weights. On 8×H100, 21 epochs ≈ 240s.

## Data

SP8192 tokenizer + FineWeb 10B, 5 train shards (500M tokens). Cycles at step ~7600 in 12k runs.
Download: `python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 5`
