Non-record submission run on 1xH100 and 2xH100 SXM 80GB (10-minute wallclock each).

## Approach

Adopting the competitive stack: deeper architecture with int6 quantization + zstd compression to maximize parameters within the 16MB artifact budget.

### Architecture
- **11 layers x 512 dim**, 8 heads, 4 KV heads (GQA), 3x MLP (hidden=1536), relu-squared
- **SmearGate** — learned sigmoid gate blending each token with previous token's embedding
- **BigramHash(2048)** — hash-based bigram embeddings projected to model dim
- **OrthoInit + muP** — orthogonal weight init, output projections scaled by 1/sqrt(2*num_layers)
- **U-Net skip connections** — encoder-decoder with learned skip weights
- Tied embeddings, logit softcap=30

### Training
- **Muon WD=0.04** + AdamW WD=0.04 — weight decay for quantization-friendly weights
- **LRs:** matrix=0.025, scalar=0.025, tied_embed=0.035
- **Muon momentum warmup** 0.92 → 0.95 over 150-1500 steps (scaled to step budget)
- **SWA every 50 steps** from warmdown phase
- **Seq len 2048** (train + eval), sliding window eval stride=64
- **Batch 786K tokens**, grad clip 0.3

### Quantization
- Int6 per-row quantization ([-32, 31]) for MLP and attention weights
- FP16 passthrough for tied embeddings
- zstd-22 compression

### Methodology
- 55+ experiments across A40 and H100 GPUs
- v1 (abandoned): wider-shallower 4x768, int8 QAT — peaked at 1.3043 bpb
- v2 (current): adopted competitive 11L int6 stack, validated on 1xH100 and 2xH100

## Results

### v2 Core Stack (1xH100 SXM 80GB, 10 min)
| Metric | Value |
|--------|-------|
| Steps | 822 |
| ms/step | 724 |
| Pre-quant val_bpb | 1.2885 |
| **Int6 roundtrip val_bpb** | **1.3133** |
| Artifact size | 12.38 MB |
| Model params | 26.8M |
| Peak memory | 19.9 GB |

### v2 Core Stack (2xH100 SXM 80GB, 10 min)
| Metric | Value |
|--------|-------|
| Steps | 1646 |
| ms/step | 365 |
| Pre-quant val_bpb | 1.2216 |
| **Int6 roundtrip val_bpb** | **1.2441** |
| Artifact size | 13.24 MB |

### Encoder Recurrence Ablation (1xH100, 10 min)
| Metric | Without | With (2x encoder) |
|--------|---------|-------------------|
| Steps | 822 | 580 |
| ms/step | 724 | 1035 |
| Pre-quant val_bpb | 1.2885 | 1.3512 |
| Verdict | — | Not worth it at low step counts |

## 8xH100 Projection
At 1646 steps on 2xH100, extrapolating to ~6400 steps on 8xH100:
- Estimated pre-quant val_bpb: ~1.14-1.15
- With FA3 (~7000+ steps): ~1.13-1.14
- With TTT at eval: potentially ~1.12
- Pending compute grant for official runs

## Configuration
```bash
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
MUON_WD=0.04 ADAM_WD=0.04 SWA_ENABLED=1 SWA_EVERY=50 \
EVAL_STRIDE=64 BIGRAM_VOCAB_SIZE=2048 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.95 MUON_MOMENTUM_WARMUP_START=0.92 \
WARMDOWN_ITERS=300 GRAD_CLIP_NORM=0.3 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Included Files
- `train_gpt.py` — v2 training script (11L int6 core stack)
- `submission.json` — metadata
