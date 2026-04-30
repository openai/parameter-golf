# Record: Vocab 4096 + MLP 4.0x + High WD + Simplifications

**val_bpb: 1.1048** (3-seed mean, std 0.0008) | **~15.95 MB** | 8xH100 SXM, 590s | No TTT, No SLOT

## Results (8xH100 80GB SXM, Montreal)

| Seed | Steps | Pre-quant BPB | Post-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------------|----------------|-----------------|----------|
| 42 | 4,807 | 1.1109 | 1.1223 | **1.1039** | 15,946,451 |
| 1337 | 4,701 | 1.1127 | 1.1238 | **1.1054** | 15,929,221 |
| 2025 | 4,758 | 1.1124 | 1.1234 | **1.1052** | 15,959,609 |
| **Mean** | **4,755** | **1.1120** | **1.1232** | **1.1048** | |
| **Std** | | | | **0.0008** | |

Merged SOTA (PR #1019, @abaybektursun): **1.1147 BPB** (1.8822 nats).
This submission: **1.1048 BPB** (~1.8656 nats).
Delta: **-0.0166 nats** (-0.0099 BPB). Clears the 0.005-nat threshold by 3.3x.

## Overview

This submission builds on PR #1218 (@clarkkev) with the same architecture run on our hardware. The key insight: a wider model (MLP 4.0x) with a larger vocabulary (4096) and aggressive weight decay (0.085) produces a more compressible model that fits under 16MB via brotli-11, while delivering better training quality per step than the narrower 1024-vocab architecture.

## Architecture

- 11 transformer layers, d=512, 8 attention heads, 4 KV heads (GQA)
- MLP expansion 4.0x (up from 3.0x in SOTA)
- Vocabulary size 4096 (up from 1024)
- XSA (cross-sequence attention) on all 11 layers
- QK_GAIN_INIT=4.0
- EMA with decay 0.997
- Sigmoid-gated U-Net skip connections
- Coprime-stride data loader for better data diversity
- 34.4M parameters (vs 27M for #1019)

## What was removed (vs #1019 SOTA)

- BigramHash embeddings
- SmearGate
- Value residuals
- Gated attention
- Quantization-aware training (QAT)
- Test-time training (TTT)
- Parameter banking
- Distributed Muon (replaced with simple DDP Muon)

## Training

- Muon optimizer with weight decay 0.085 (up from 0.04)
- Embeddings weight decay 0.085 (was 0)
- Adam weight decay 0.02 (down from 0.04)
- Learning rate 0.02 (down from 0.025)
- Dynamic warmdown: 66.7% of actual training steps
- Max wallclock: 600s (GPTQ reserves 10s, effective 590s)

## Quantization and Compression

- Full Hessian GPTQ with AR self-generated calibration data (no val or train data access)
- Int6 quantization
- Byte shuffle + brotli-11 compression (saves ~400KB vs LZMA)
- All artifacts under 16,000,000 bytes

## Causality and Legality

- **No test-time training (TTT)**: No parameter updates during evaluation
- **No SLOT**: No eval-time delta optimization
- **No n-gram cache**: No eval-time frequency table construction
- **No pre-eval adaptation**: GPTQ calibration uses AR self-generated tokens only
- Standard sliding window evaluation with stride 64
- F.cross_entropy scoring produces full normalized probability distributions

## Key Insight: Weight Decay and Compressibility

The compressibility of a weight matrix (quantized-and-compressed size / raw size) correlates with the matrix's root-mean-square value with R^2 near 0.99 (credit: @clarkkev PR #1218). Higher weight decay produces lower-magnitude weights that compress better, allowing a wider model to fit under the 16MB cap. This is why MLP 4.0x + WD 0.085 works where MLP 3.0x + WD 0.04 would not.

## Tokenizer

Uses the sp4096 SentencePiece tokenizer from kevclark/parameter-golf on HuggingFace. Larger vocab means more context per sequence and more training data processed per step, partially compensating for slower per-step throughput.

## Reproduction

```bash
pip install sentencepiece zstandard brotli
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

# Download sp4096 data
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp4096 --train-shards 143

# Run (each seed)
for SEED in 42 1337 2025; do
  SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Credits

- PR #1218 (@clarkkev) for the architecture and key insights
- PR #1019 (@abaybektursun) for the merged SOTA baseline
- PR #1089 for sigmoid-gated U-Net skips and brotli compression
- PR #1125 for QK_GAIN=4.0 sweep
- PR #726 for coprime-stride data loader

## Test Plan

- [x] 3-seed verification (std 0.0008, p < 0.01 vs SOTA)
- [x] All artifacts under 16,000,000 bytes
- [x] Training under 600s per seed
- [x] Evaluation under 600s per seed
- [x] No TTT, no SLOT, no n-gram cache
- [x] GPTQ calibration within training budget (AR self-gen)
- [x] Standard F.cross_entropy scoring (full normalized distributions)
