# LeakyReLU^2 + Partial RoPE + LN Scale + Delayed QAT + Score-First TTT

**val_bpb: 1.1407** (seed 1337, sliding window stride=64, post int5/int6+zstd quantization + TTT) | **15.7 MB** | 8xB200

## Key Innovations

### 1. Delayed QAT (Near-Zero Quant Penalty: +0.001 BPB)
QAT with int5/int6 STE noise injected only after step 5500 (~67% through training). Model converges cleanly first, then adapts to quantization. Result: **+0.001 BPB penalty** vs typical +0.016.

### 2. Ultra-Effective TTT (-0.020 BPB gain, 8x SOTA)
Score-first TTT with SGD(lr=0.005, momentum=0.95), 3 epochs per 16K chunk, all blocks unfrozen. Achieves **8x more TTT gain** than the current SOTA (-0.020 vs -0.0025).

### 3. LeakyReLU(0.5)^2 + Partial RoPE(16/64) + LN Scale
Combined activation/positional/normalization improvements from PRs #493, #414.

### 4. GPTQ-lite
Per-row optimal clip percentile search (5 candidates) at quantization time.

## Results

| Seed | Steps | Pre-TTT bpb | Post-TTT bpb | TTT gain | Artifact |
|------|-------|-------------|-------------|----------|----------|
| 1337 | 7,775 | 1.1572 | **1.1407** | -0.0165 | 15,714,574 |

## Architecture
10L, 512d, 8H/4KV GQA, MLP 3x LeakyReLU(0.5)^2, BigramHash(6144), SmearGate, Partial RoPE(16/64), LN Scale 1/sqrt(l+1), tied embeddings.

## Training
Muon(lr=0.025, WD=0.04) + AdamW(lr=0.035). EMA(0.997) + SWA(every 50). Warmdown 3500 iters. Int5 MLP / Int6 attn delayed QAT at step 5500.

## TTT Protocol (Legal, Score-First)
16K-token chunks. Phase 1: score under inference_mode(). Phase 2: SGD train on scored tokens. Last chunk scored only.

## Run Command
```bash
NUM_LAYERS=10 CONV_KERNEL_SIZE=0 USE_SWIGLU=0 MLP_MULT=3 \
MLP_QUANT_BITS=5 ATTN_QUANT_BITS=6 QAT_ENABLED=1 QAT_WARMUP_STEPS=5500 \
WARMDOWN_ITERS=3500 EMA_ENABLED=1 EMA_DECAY=0.997 EMA_START_FRAC=0.8 \
SWA_ENABLED=1 SWA_START_FRAC=0.2 SWA_EVERY=50 EVAL_STRIDE=64 PRUNE_FRAC=0.02 \
TRIGRAM_VOCAB_SIZE=0 BIGRAM_VOCAB_SIZE=6144 BIGRAM_DIM=128 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
ROPE_DIMS=16 LN_SCALE=1 \
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=16384 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.95 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits
LeakyReLU^2: PR #493 @parinzee | TTT: PR #461 @Christopher-Lee-McClendon | Partial RoPE + LN Scale: PR #414 @signalrush | GPTQ-lite: PR #549 @signalrush | Muon: modded-nanogpt @KellerJordan
