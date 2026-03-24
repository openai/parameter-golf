# Full GPTQ + LeakyReLU² + Parallel Muon + BigramHash 3072

**val_bpb: 1.1163** (3-seed mean, std 0.0012) | **~15.90 MB** | 8×H100 SXM, 600s | No TTT

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-quant bpb | **Post-GPTQ sliding bpb** | Artifact |
|------|----------|-------|---------------|--------------------------|----------|
| 42 | 83.4ms | 7,192 | 1.1349 | **1.1149** | 15,895,636 |
| 1337 | 83.4ms | 7,195 | 1.1370 | **1.1172** | 15,899,284 |
| 2024 | 83.5ms | 7,190 | 1.1367 | **1.1167** | 15,904,036 |
| **Mean** | **83.4ms** | **7,192** | **1.1362** | **1.1163 (std 0.0012)** | |

GPTQ improves post-quantization BPB by **0.0199** vs pre-quantization (1.1362 → 1.1163). Standard GPTQ-lite gives only 1.1218 from the same pre-quant model — Full GPTQ is 0.0055 better.

## Key Innovation: Full Hessian GPTQ

Standard GPTQ-lite searches for the best per-row clip percentile — a greedy row-wise optimization. Full GPTQ uses second-order information (the Hessian H = X^T X) to compensate for quantization error across columns:

1. **Hessian collection**: 256 calibration batches through a non-banked model replica, accumulating H = X^T X per linear layer via forward hooks
2. **Column reordering (actorder)**: Quantize columns in order of descending Hessian diagonal (most important first)
3. **Cholesky error compensation**: For each column block, propagate quantization error to remaining columns using H^{-1}, minimizing total reconstruction loss
4. **Per-row scale search**: Same 5-percentile search as GPTQ-lite, but applied within the Cholesky framework

Based on IST-DASLab/gptq (ICLR 2023). Adapted for banked weights by unbanking to a temporary non-banked model for Hessian collection.

## Training Architecture

PR #414 stack with Parameter Banking + Parallel Muon ([PR #399](https://github.com/openai/parameter-golf/pull/399)):

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with **LeakyReLU(0.5)²** |
| BigramHash | **3072 buckets, dim=80** |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | **Full Hessian GPTQ int6** + lzma(9) |
| Optimizer | Parameter Banking + Parallel Muon |

## Hardware-Aligned BigramHash Configuration

The change from BigramHash 1536×128 to **3072×80** is a budget-optimal reallocation of the 16MB artifact limit, informed by H100 roofline analysis:

**Coverage beats fidelity.** Each bigram embedding passes through a learned 80→512 projection before entering the model. The projection has enough capacity to reconstruct useful features from a narrower input — the information loss from 128→80 dim is small. But a bigram pattern that hashes to a collision (because the table is too small) gets zero representation. Doubling buckets from 1536→3072 halves hash collisions, capturing more unique bigram patterns.

**Quantized embeddings are the most expensive bytes in the artifact.** Random-looking embedding vectors have high entropy and compress poorly under GPTQ+lzma. Each additional embedding dimension costs disproportionately in the compressed artifact. Narrower embeddings (dim=80) give better bits-per-information-bit in the compressed output, freeing bytes for more buckets.

**GPTQ memory fix.** The training model is freed from GPU memory (`base_model.cpu()`) before GPTQ calibration, preventing OOM when the Hessian collection model is loaded. This was necessary because the larger BigramHash increases optimizer state size, leaving insufficient headroom for the second model. Previous configurations OOM'd during GPTQ without this fix.

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=80 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Eval Timing

| Phase | Time |
|-------|------|
| Training | 600s |
| Free training model + Hessian collection (256 batches) | ~30s |
| GPTQ quantization + lzma(9) compression | ~90s |
| Sliding window eval (stride=64) | ~100s |
| **Total eval** | **~220s (< 10 min)** |

No TTT needed — Full GPTQ alone beats all prior TTT-based submissions.

## Credits

- **Full GPTQ**: PR #569 by @abaybektursun (Hessian-aware quantization implementation)
- **LeakyReLU²**: PR #493, PR #518
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
- **GPTQ algorithm**: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (ICLR 2023)
