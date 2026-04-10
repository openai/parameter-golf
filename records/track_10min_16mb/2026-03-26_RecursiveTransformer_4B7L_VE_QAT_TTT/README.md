# Recursive Transformer 4B/7L + ValueEmbedding + QAT Int6 + TTT

**val_bpb: 1.1696** (3-seed mean) | **~15.76 MB** | 8xH100 SXM, 600s | Score-First TTT + Sliding Window

## Results (3-seed)

| Seed | Steps | ms/step | **TTT+Sliding BPB** | Post-quant BPB | Artifact |
|------|-------|---------|---------------------|----------------|----------|
| 1337 | 5,963 | 100.7 | **1.1698** | 1.1952 | 15,749,104 |
| 42 | 5,967 | 100.7 | **1.1697** | 1.1949 | 15,778,257 |
| 2024 | 5,965 | 100.7 | **1.1693** | 1.1947 | 15,750,116 |
| **Mean** | 5,965 | | **1.1696** | 1.1949 | |

## Method

### Recursive Transformer Architecture

Unlike all other submissions which use standard 10-11 layer transformers, this submission uses a **recursive/looped transformer**: a small number of shared transformer blocks applied repeatedly in a loop. This dramatically reduces unique parameters, allowing a much wider model (dim=1024 vs the standard 512) while staying under 16MB.

**Core idea:** Instead of 10 unique blocks with 10 sets of weights, use 4 shared blocks applied 7 times (4B/7L). The model sees "depth" through weight reuse, trading unique parameters for effective depth. This is inspired by [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (Samsung SAIL Montreal).

### U-Net Skip Connections

The 7 loop iterations are split into encoder (first 3) and decoder (last 4) phases. Encoder loops store activations as skip connections, consumed in reverse order by decoder loops. Per-loop learnable skip weights modulate the skip contribution.

### Key Components

| Component | Setting | Notes |
|-----------|---------|-------|
| **Architecture** | 4 shared blocks, 7 loops (dim=1024) | Recursive transformer, ~30M params |
| Attention | 32 heads, 8 KV heads (GQA) | |
| MLP | 2x multiplier | |
| XSA | Last 4 of 7 loops | Cross-Sequence Attention, zero extra params |
| ValueEmbedding | dim=128, last 2 loops | Reinjects token identity into value projection |
| SmearGate | Learned per-dim gate | Blends current token with previous |
| BigramHash | 10240 buckets, dim=128 | Hash(prev, cur) embedding |
| U-Net Skips | Encoder-decoder with learnable weights | |
| RoPE | Standard | |

### Quantization

**Int6 QAT from step 0** using Straight-Through Estimator (STE). During training, all large weight matrices are fake-quantized to 6-bit precision in the forward pass. The model learns to be robust to quantization noise throughout training, not just during warmdown.

The final artifact uses **int8 quantization + GPTQ-lite** (5 clip percentiles per row) compressed with **zstd level 22**. Despite training with int6 QAT, int8 serialization produces better BPB at similar compressed size due to byte-aligned data compressing well with zstd.

| Stage | BPB (3-seed mean) |
|-------|-----|
| Pre-quant (SWA) | 1.1820 |
| Post-quant int8+zstd | 1.1949 (+0.013) |
| TTT+Sliding eval | **1.1696** |

### Evaluation: Score-First TTT + Sliding Window

At eval time, the validation data is processed in chunks (32768 tokens each). For each chunk:
1. **Score** the chunk using sliding windows (stride=64) — pure inference, no training
2. **Train** on the scored chunk with SGD (lr=0.002, momentum=0.9, 3 epochs) — adapts the model for subsequent chunks

This is "score-first" TTT: you never train on tokens before scoring them, satisfying the competition's constraint that validation data can only be used for TTT on already-evaluated tokens.

### Optimizer

Muon optimizer (momentum=0.99) with:
- Matrix LR: 0.02
- Scalar LR: 0.01
- Tied embedding LR: 0.02
- Weight decay: 0.04
- Grad clip: 0.3
- Warmdown: 3500 iterations
- Warmup: 100 steps
- SWA: start at 20% of warmdown, every 50 steps

### Why Recursive?

The recursive architecture is a genuinely novel contribution to this competition. While it may not match the absolute BPB of heavily-optimized standard transformers, it demonstrates that:

1. **Weight sharing works at small scale** — 4 blocks doing 7 loops matches or approaches 10-11 unique blocks
2. **Width over depth** — dim=1024 (2x wider than standard) compensates for fewer unique parameters
3. **QAT is essential for recursive models** — without it, quantization error compounds through shared-weight loops (+0.56 BPB degradation without QAT vs +0.012 with QAT from step 0)
4. **U-Net skip connections** stabilize deep recursive loops and enable gradient flow

### Negative Results

- **8192 BPE vocabulary**: The -0.42 BPB gain reported for standard transformers does not transfer to recursive models. Factored embeddings add complexity without benefit.
- **EMA**: Incompatible with QAT — averages non-QAT-trained weights, producing mixed weight distributions that quantize poorly.
- **Int5/Int6 post-training quantization**: Without QAT, error compounds through recursive loops. +0.56 BPB degradation vs +0.012 with full QAT.
- **Bitpacking int6**: 6-bit packed data has higher entropy density, compresses worse than int8 with zstd. Net file size increase.

## Run Command

## Acknowledgments

Training hyperparameters (Muon LR, weight decay, BigramHash 10240x128, SWA, grad clip) adapted from [PR #414](https://github.com/openai/parameter-golf/pull/414) by @thwu1. 
Recursive architecture inspired by [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (Samsung SAIL Montreal).

## Run Command

```bash
NUM_BLOCKS=4 MODEL_DIM=1024 NUM_HEADS=32 NUM_KV_HEADS=8 \
NUM_LOOPS=7 TRAIN_SEQ_LEN=2048 VOCAB_SIZE=1024 \
USE_SMEAR_GATE=1 BIGRAM_BUCKETS=10240 BIGRAM_DIM=128 \
QAT_ENABLED=1 QAT_BITS=6 QAT_MLP_BITS=0 LATE_QAT_THRESHOLD=1.0 \
EMA_DECAY=0 SWA_START_FRAC=0.2 SWA_EVERY=50 \
SLIDING_WINDOW_STRIDE=64 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=1 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
XSA_LAST_N=4 VE_ENABLED=1 VE_DIM=128 VE_LAST_N=2 \
MATRIX_LR=0.02 SCALAR_LR=0.01 TIED_EMBED_LR=0.02 \
MUON_MOMENTUM=0.99 WEIGHT_DECAY=0.04 GRAD_CLIP_NORM=0.3 \
WARMDOWN_ITERS=3500 WARMUP_STEPS=100 \
SEED=$SEED \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
