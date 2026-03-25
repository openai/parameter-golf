# Full GPTQ + XSA-4 + SWA/EMA + Score-First TTT

**val_bpb: 1.1198** (3-seed mean, std 0.0006) | **~15.9 MB** | 8xH100 SXM

## Results (8xH100 SXM, PyTorch 2.9.1+cu128)

| Seed | Steps | ms/step | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|-------|---------|-------------|-----------------|----------|----------|----------|
| 1337 | 6,461 | 86.67 | 1.1193 | **1.1193** | -0.00004 | ~236s | 15,899,061 |
| 42 | 6,457 | 86.73 | 1.1197 | **1.1196** | -0.00004 | ~236s | 15,954,941 |
| 2025 | 6,457 | 86.74 | 1.1206 | **1.1205** | -0.00006 | ~236s | 15,907,769 |

**Mean: 1.1198 | Std: 0.0006**

## Timing Budget

| Phase | Time | Notes |
|-------|------|-------|
| Training loop | 560s | Main training (Muon + Adam, ~6,460 steps) |
| GPTQ calibration + quantization | ~40s | Hessian calibration on training data, within training budget |
| **Total artifact construction** | **~600s** | **Within 10-min training limit** |
| Standard eval (roundtrip + sliding) | ~82s | No training data access |
| Score-first TTT | ~236s | Legal: score chunk, then adapt, never re-score |
| **Total eval** | **~318s** | **Within 10-min eval limit** |

Note: GPTQ calibration uses training data and is counted as part of the training/artifact construction phase, not the eval phase. No training data is accessed during evaluation.

## Key Techniques

### Full Hessian GPTQ
- 256-batch calibration from training data for per-layer Hessian approximation
- Column-wise int6 quantization with Cholesky error compensation, block size 128
- Percentile clip search over [0.999, 0.9995, 0.9999, 0.99999, 1.0] per layer
- Act-order column permutation (quantize most-activated columns first)

### Legal Score-First TTT (PR #461/#549 recipe)
Backward-looking, score-first TTT:
1. Val tokens processed in non-overlapping 128K-token chunks
2. **For each chunk**:
   - **SCORE**: Sliding window eval under `torch.inference_mode()` — no gradients, no weight mutation
   - **TRAIN**: AdamW(lr=1e-4, wd=0) on the already-scored chunk. 3 epochs, first 9/11 blocks frozen, grad clip 1.0
3. Last chunk scored but never trained on
4. Chunk N scored by model adapted only on chunks 0..N-1

`inference_mode()` provides a hard guarantee that scoring is stateless.

### XSA on Last 4 Layers
Cross-Sequence Attention on transformer layers 7-10. Extended context beyond training sequence length at eval time.

### SWA/EMA Weight Blending
EMA (decay=0.997) + Stochastic Weight Averaging (every 50 steps during warmdown), blended 50/50. Smooths weight landscape before quantization.

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| BigramHash | 3072 buckets, 128-dim |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | Full GPTQ int6 + LZMA |
| Optimizer | Muon (matrices) + AdamW (scalars/embeddings) |

## Training

- Muon optimizer (matrices): lr=0.025, momentum=0.99, WD=0.04, 5 Newton-Schulz steps
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 4,000 iters (wallclock-based)
- Late QAT: disabled (LATE_QAT_THRESHOLD=0)

## Run Command

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=560 \
XSA_LAST_N=4 WARMDOWN_ITERS=4000 \
CLIP_RANGE=31 COMPRESSOR=lzma \
LATE_QAT_THRESHOLD=0 PRUNE_PCT=0.005 \
GPTQ_ENABLED=1 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 \
SWA_ENABLED=1 \
TTT_ENABLED=1 TTT_ADAMW=1 TTT_LR=0.0001 TTT_EPOCHS=3 TTT_WD=0 \
TTT_FREEZE_BLOCKS=9 TTT_FREEZE_EMBEDDINGS=0 TTT_CHUNK_TOKENS=131072 \
EVAL_STRIDE=64 NUM_KV_HEADS=4 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Requirements

```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install zstandard sentencepiece lzma
```

## Credits

- **Base architecture**: PR #589 by @RoyiRa (11L GEPA, GQA, VE128, Late QAT)
- **GPTQ reference**: PR #609 by @saml212, PR #626 by @kshitizz36
- **Score-first TTT**: PR #461 by @Christopher-Lee-McClendon
- **XSA, BigramHash, SmearGate**: Various community contributors
