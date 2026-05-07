# 11L MLP2x + LeakyReLU² + Legal TTT (val_bpb=1.2201, 3-seed mean)

**val_bpb: 1.2201** (3-seed mean, std 0.0015) | **~15.0 MB** | 8×H100 SXM

## Seeds

| Seed | Steps | Pre-TTT BPB | Post-TTT BPB | Artifact |
|------|-------|-------------|--------------|----------|
| 1337 | 7,821 | 1.3772 | 1.2184 | 14,986,599 |
| 42 | 7,833 | 1.4170 | 1.2212 | 15,010,826 |
| 2025 | 7,863 | 1.3899 | 1.2207 | 14,980,707 |
| **Mean** | **7,839** | **1.3947** | **1.2201 (std 0.0015)** | **14,992,711** |

All 3 artifacts under 16,000,000 bytes. ✅

## Architecture

| Setting | Value |
|---------|-------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 2× with LeakyReLU(0.5)² |
| Sequence length | 2048 (train + eval) |
| BigramHash | 4096 buckets |
| SmearGate | Enabled |
| U-Net skips | Enabled |
| XSA | Last 4 layers |
| LN Scale | 1/√(layer+1) |
| RoPE | Full (no partial) |
| Embeddings | Tied input/output |

## Training

| Setting | Value |
|---------|-------|
| Optimizer | Muon (lr=0.025) + AdamW |
| Muon momentum | 0.95 |
| Weight averaging | EMA(0.997) |
| Warmup | 20 steps |
| Warmdown | 3500 iterations |
| Batch tokens | 524,288 |
| Wallclock cap | 600s |
| QAT | Int6 STE (full training) |

## Quantization

| Setting | Value |
|---------|-------|
| Precision | Int6 per-row with GPTQ-lite (5-percentile clip search) |
| Compression | zstd level 22, multi-threaded |
| Model bytes | 14,885,322 |
| Code bytes | 90,451 |
| **Total** | **14,975,773** |

## Evaluation Protocol

**Sliding window eval** (stride=64) + **legal score-first TTT**:

1. Val tokens split into 1,893 non-overlapping 32K-token chunks
2. For each chunk (except last):
   - **SCORE:** Sliding window eval under `torch.inference_mode()` — no gradients, no weight mutation
   - **TRAIN:** SGD(lr=0.002, momentum=0.9) on the already-scored chunk. 7 epochs, cosine LR decay, grad clip 1.0, all blocks unfrozen
3. Last chunk scored but never trained on
4. Chunk N scored by model adapted only on chunks 0..N-1

Same protocol as PR #549 (current SOTA), which uses 3 epochs. We use 7 epochs on already-scored data.

| TTT Parameter | Value |
|---------------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | SGD + momentum(0.9) |
| Learning rate | 0.002 (cosine decay across chunks) |
| Epochs per chunk | 7 |
| Frozen blocks | None (all blocks adapt) |
| Gradient clip | 1.0 |

## Timing

| Phase | Time |
|-------|------|
| Training (7,630 steps) | 600s |
| Quant eval | ~3s |
| Sliding window eval | ~2s |
| Legal TTT | ~583s |
| **Total eval** | **~588s** (< 10 min) |

## Reproduction

```bash
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
TRAIN_SEQ_LEN=2048 MLP_MULT=2 \
BIGRAMHASH_BUCKETS=4096 SMEARGATE=1 UNET_SKIPS=1 \
INT6_QAT=1 TIE_EMBEDDINGS=1 \
ROPE_PARTIAL_DIMS=0 LN_SCALE=1 XSA_LAYERS=4 EMA_DECAY=0.997 \
ITERATIONS=12000 WARMDOWN_ITERS=3500 WARMUP_STEPS=20 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.95 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
TTT_ENABLED=1 TTT_EPOCHS=7 TTT_LR=0.002 \
TTT_CHUNK_TOKENS=32768 TTT_BATCH_SEQS=32 \
TTT_MOMENTUM=0.9 TTT_GRAD_CLIP=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Attribution

Built on techniques from the community:
- LeakyReLU(0.5)² activation (PR #493 by @parinzee)
- Legal score-first TTT (PR #461 by @Christopher-Lee-McClendon)
- EMA + GPTQ-lite + warmdown stack (PR #414 by @signalrush)
- XSA (PR #198), SmearGate, BigramHash, U-Net skips from community contributions

## Platform

RunPod 8×H100 SXM, PyTorch 2.9.1+cu128
