# 12L XSA-all + Partial RoPE + Batch 786K

**val_bpb: TBD (8xH100 3-seed mean)** | ~13.5 MB | 8xH100 SXM

Dev result (1xH100, 5K steps): **1.1412 BPB**, 13.49 MB artifact

## Results (8xH100 80GB SXM)

| Seed | step_avg | steps | val_bpb | Artifact |
|------|----------|-------|---------|----------|
| 1337 | TBD | TBD | TBD | TBD |
| 42 | TBD | TBD | TBD | TBD |
| 2024 | TBD | TBD | TBD | TBD |
| **Mean** | — | — | **TBD** | — |

## Key Innovation: 12-Layer Architecture

Most competitive submissions use 11 layers. We demonstrate that **12 layers fits comfortably under the 16 MB limit** (13.5 MB with 2.5 MB headroom) while providing consistent BPB improvements. The extra layer adds ~2.6M parameters but GPTQ-lite int6 quantization + zstd-22 compression keeps the artifact well within budget.

## Systematic Ablation (1xH100, 5K steps)

We ran 11 isolated experiments, each changing one variable vs the 12L baseline (L = 1.1564 BPB):

### Hyperparameter Tuning (M1-M4)

| Exp | Change | BPB | Delta | Verdict |
|-----|--------|-----|-------|---------|
| **M1** | Batch 786K (from 524K) | **1.1461** | **-0.0103** | **Winner** |
| M2 | BigramHash 3072/112 | 1.1580 | +0.0016 | Dropped |
| M3 | Momentum 0.99 | 1.1461 | -0.0103 | Dropped (artifact 17.4 MB) |
| M4 | Lower LRs 0.025/0.035 | 1.1618 | +0.0054 | Dropped |

### Architecture Experiments (M6-M9)

| Exp | Change | BPB | Delta | Verdict |
|-----|--------|-----|-------|---------|
| **M6** | Partial RoPE 16/64 dims | **1.1552** | **-0.0012** | **Winner** |
| M7 | LN Scale 1/sqrt(layer+1) | 1.1590 | +0.0026 | Dropped |
| **M8** | XSA all 12 layers | **1.1540** | **-0.0024** | **Winner** |
| M9 | Value Embedding dim=128 | 1.1561 | -0.0003 | Dropped (neutral) |

### Combined Experiments

| Exp | Config | BPB | Delta | Artifact |
|-----|--------|-----|-------|----------|
| M10 | M1+M3+M6+M8 (all 4) | 1.1273 | -0.0291 | 17.7 MB (over limit!) |
| **M11** | **M1+M6+M8 (drop M3)** | **1.1412** | **-0.0152** | **13.5 MB** |

### Interesting Finding: Momentum vs Compression

M3 (Muon momentum 0.99) delivered -0.0103 BPB but inflated the artifact from 13.4 to 17.4 MB despite identical model architecture. Higher momentum produces weight distributions with higher entropy that compress poorly under zstd-22. When combined (M10), all four winners achieved our best BPB (1.1273) but exceeded the 16 MB limit. Dropping M3 (M11) sacrifices BPB but keeps the artifact at 13.5 MB.

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | **12** (512d, 8H, 4KV) |
| MLP | 3x with LeakyReLU |
| XSA | **All 12 layers** |
| RoPE | **Partial (16/64 dims)** |
| Batch tokens | **786,432** |
| BigramHash | 2048 buckets, dim 128 |
| SmearGate | Enabled |
| Weight avg | EMA(0.997) |
| Quantization | GPTQ-lite int6 + zstd-22 |
| QAT | Late QAT (threshold 0.15) |
| Optimizer | Muon + Adam (WD 0.04) |
| Warmdown | 3500 iters |
| Grad clip | 0.3 |

## Run Command

```bash
NUM_LAYERS=12 MLP_MULT=3 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 \
SMEARGATE=1 BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
EMA_ENABLED=1 EMA_DECAY=0.997 EVAL_STRIDE=64 \
LATE_QAT=1 QAT_THRESHOLD=0.15 \
MUON_WD=0.04 ADAM_WD=0.04 \
WARMDOWN_ITERS=3500 GRAD_CLIP_NORM=0.3 \
XSA_LAST_N=12 ROPE_DIMS=16 \
TRAIN_BATCH_TOKENS=786432 \
MAX_WALLCLOCK_SECONDS=600 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base model architecture: modded-nanogpt / parameter-golf baseline
- XSA (Exclusive Self Attention): PR #414 by @signalrush
- Partial RoPE: Adapted from #4 submission (PR #518 by @sofiabod)
- SmearGate, BigramHash, EMA, GPTQ-lite: community contributions
