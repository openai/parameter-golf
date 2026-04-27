## Non-record: Byte-level transformer + JEPA auxiliary loss (val_bpb: 1.1903)

**val_bpb = 1.1903** (sliding window, stride=512) | **14.4 MB** artifact | 8xH100 SXM, 600s

Byte-level autoregressive transformer operating directly on raw UTF-8 bytes (vocab 260, no tokenizer). Beats the sp1024 baseline (1.2244) by 0.034 BPB.

### What this is

A 13-layer byte-level transformer where the primary training objective is standard next-byte CE loss. A lightweight JEPA (Joint Embedding Predictive Architecture) module adds an auxiliary chunk-level latent prediction signal, contributing ~0.1% of gradient at peak (λ_max=0.001). Chunk prediction concept inspired by LeWM.

The heavy lifting is the AR transformer + technique stack (Muon, EMA, XSA, Partial RoPE, LN Scale, SmearGate, BigramHash, OrthoInit). JEPA adds a modest 0.01 BPB improvement (1.2006 → 1.1905) at 5% overhead.

### Ablation

| | Without JEPA | With JEPA | Delta |
|---|---|---|---|
| Int6 sliding s512 | 1.2006 | **1.1905** | **-0.0101** |
| Step time | 60ms | 63ms | +3ms |
| Params | 24.2M | 24.6M | +459K |
| Artifact | 14.0 MB | 14.2 MB | +0.2 MB |

### Architecture

| Component | Detail |
|-----------|--------|
| Backbone | 13L transformer, dim=512, 8H/4KV GQA, MLP 2x (LeakyReLU(0.5)²), U-Net skips |
| JEPA projector | Linear(512,256) → RMSNorm → SiLU → Linear(256,256) |
| JEPA predictor | 2-layer MLP, 256d, causal shift with learned start token |
| JEPA injection | Linear(256,512), zero-init, adds predicted latents to residual stream |
| SIGReg | Epps-Pulley regularization (256 projections, 17 knots) prevents collapse |
| Training schedule | Phased: 30% pure AR → 50% AR+JEPA ramp (λ: 0→0.001) → 20% pure AR |

Full stack: Muon+WD=0.04, EMA 0.997, XSA last 4 layers, Partial RoPE 16 dims, LN Scale (1/√(l+1)), SmearGate, BigramHash(4096,32), OrthoInit+muP, int6 MLP+attn + int8 embed + zstd-22, FA3.

### Key Metrics

- 9,000 steps in 568s (63ms/step)
- ~3.5B train bytes (9,000 steps × 393,216 bytes/step)
- Peak memory: ~10,800 MiB per GPU

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.2293 |
| Int6 roundtrip val_bpb | 1.2184 |
| **Int6 sliding val_bpb (s512)** | **1.1905** |
| Compressed artifact (int6+zstd) | 14,111,704 bytes |
| Code size | 71,203 bytes |
| **Total submission size** | **14,182,907 bytes** |

### Reproducibility

| Seed | Steps | Sliding s512 | Artifact |
|------|-------|-------------|----------|
| **2025** | **9,000** | **1.1903** | **14,369,791** |
| 42 | 9,000 | 1.1905 | 14,182,907 |
| 7 | 9,000 | 1.1915 | 14,445,175 |

Mean val_bpb: **1.1908**. Submitted: seed 2025 (best). Inter-seed range: 0.0012.

### Data

`fineweb10B_byte260` — raw UTF-8 bytes from FineWeb validation set. Token IDs 0-3 are special (PAD/BOS/EOS/UNK), IDs 4-259 map to byte values 0-255. BPB = loss / ln(2), no tokens-per-byte correction. Converted from sp1024 shards via precomputed byte lookup table.

### Configuration

```bash
NUM_LAYERS=13 VOCAB_SIZE=260 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2.0 \
TRAIN_SEQ_LEN=4096 TRAIN_BATCH_TOKENS=393216 BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=32 \
XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=512 JEPA_CHUNK_SIZE=8 JEPA_LATENT_DIM=256 JEPA_PROJ_HIDDEN=256 \
JEPA_LAMBDA_MAX=0.001 JEPA_SIGREG_WEIGHT=0.02 JEPA_LR=0.001 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Included files

- `train_gpt.py` — full training + quantization + evaluation script
- `train.log` — training log from best seed (42)
- `train_seed42.log`, `train_seed2025.log`, `train_seed7.log` — all seed logs
- `submission.json` — leaderboard metadata
