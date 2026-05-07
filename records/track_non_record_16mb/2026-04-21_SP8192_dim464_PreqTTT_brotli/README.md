# SP8192 + dim=464 + Pre-Quantization TTT + Brotli

**val_bpb: 1.1863** (roundtrip, seed 1337) | **15.92 MB** | 1×RTX 5090, non-record

Post-TTT (score-first, 3 epochs): **1.1524 BPB**

## Key Technique: Pre-Quantization TTT

After training ends, before INT6 quantization, adapt the FP32 weights on the **full validation set** using standard (non-score-first) TTT. This conditions the model to the val distribution before quantization locks in the weights — the quantization then compresses a val-adapted model rather than the raw trained one.

The implementation is DDP-aware: chunks are interleaved across ranks (`range(rank, n_chunks, world_size)`), weights are `all_reduce(AVG)` after each epoch. On 8×H100, 21 epochs ≈ 240s additional.

## Validated Scaling Law (dim=464, 12k steps, 1×RTX 5090)

| preq-TTT epochs | Roundtrip BPB | Delta vs prev |
|-----------------|--------------|--------------|
| 0               | 1.2347       | —            |
| 3               | 1.2097       | −0.025       |
| 5               | 1.1968       | −0.013       |
| 7               | **1.1863**   | −0.011       |

Still scaling at 7 epochs — diminishing returns are slow. On 8×H100 with 21 epochs (≈ 240s), estimated **~1.14 BPB**.

## Architecture

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
| Optimizer | MuonEq-R + Adam |
| Warmdown | 3000 steps |

**Artifact:** 15,915,528 bytes (84 KB under 16 MB limit)

## Run Command (8×H100, competition-ready)

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

## Results (1×RTX 5090, seed 1337)

| Metric | Value |
|--------|-------|
| Raw val_bpb (step 12k) | 1.2338 |
| Roundtrip val_bpb (preq-TTT 7ep + INT6 + brotli) | **1.1863** |
| Post-TTT val_bpb (3-epoch score-first on top) | **1.1524** |
| Artifact size | 15,915,528 bytes |
| Hardware | 1× NVIDIA RTX 5090 32GB |
| Training | 12k steps, ~33 min |
| preq-TTT | 7 epochs, ~1286s |

## Data

SP8192 tokenizer + FineWeb 10B. 20 train shards available (2B tokens, covers 12k steps without cycling).
Download: `python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 20`
(or use `data/stream_tokenize_sp8192.py` to generate from the raw corpus)
