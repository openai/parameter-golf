## 11L EMA + LeakyReLU² + LZMA + Int6 GPTQ-lite (val_bpb: 1.1303)

**val_bpb: 1.1303** (2-seed mean, sliding window stride=16) | **15.87 MB** (mean) | 8×H100 SXM, 600s

### Summary

Non-record submission combining several architectural optimizations with LZMA compression and temperature-scaled evaluation. Built on the established 11-layer Transformer stack with key modifications: LeakyReLU(0.5)² activation, LZMA extreme compression replacing zlib/zstd, and evaluation-time temperature scaling.

### Key Changes

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **MLP Activation** | LeakyReLU(0.5)² | ~0.005 BPB improvement over ReLU² at this scale |
| **Compression** | LZMA preset 9 + extreme | ~3MB smaller than zlib, enables int6-only quantization |
| **Eval Temperature** | T=0.90 | Free ~0.005 BPB for LeakyReLU² models |
| **BigramHash** | 2048 buckets (128d) | Reduced from 4096; minimal BPB impact, saves ~0.3MB |
| **TrigramHash** | Disabled | Saves ~590K params (~0.44MB) with no proven BPB benefit |
| **Depth Recurrence** | Disabled | Ineffective per multiple independent reproductions |
| **Gated Attention** | Disabled | No measurable BPB benefit at this scale |
| **Value Residual** | Disabled | Incompatible with optimal XSA stack |

### Results (2 seeds, 8×H100 SXM)

| Seed | Steps | val_loss | Sliding BPB (s16) | Artifact |
|------|-------|----------|-------------------|----------|
| **1337** | 7188 | 1.9083 | **1.1302** | 15.86 MB |
| **42** | 7187 | 1.9086 | **1.1304** | 15.88 MB |

**Mean: 1.1303 | Std: 0.0001** | Submitted: seed 1337 (best)

### Model Selection (seed 1337)

| Stage | val_bpb |
|-------|---------|
| Raw model | 1.1389 |
| **EMA (selected)** | **1.1378** |
| SWA (n=14) | 1.1389 |
| After int6 quant | 1.1462 |
| **Sliding eval (stride=16)** | **1.1302** |

### Architecture

- 11 Transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- **LeakyReLU(0.5)² MLP** (3× expansion = 1536 hidden)
- U-Net skip connections (5 encoder, 6 decoder)
- Exclusive Self-Attention (XSA) on last 4 layers (GQA-aware, zero-alloc)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/√(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10)
- SmearGate + BigramHash (2048 buckets, 128d)
- Parameter Banking (weight sharing via banked tensors)
- Tied embeddings, logit softcap=30.0
- **27.0M parameters**

### Training

- FlashAttention 3 (Hopper-optimized)
- Muon optimizer (matrices): lr=0.025, momentum 0.92→0.99 (1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step (gas=1 on 8 GPUs), seq_len=2048
- Warmdown: 3500 iterations (wallclock-based cosine)
- EMA: decay=0.997, late start at 40% training
- Tight SWA: every 50 steps when LR scale < 0.2
- Late QAT: STE int6 fake-quantization when LR scale < 0.15
- OrthoInit + muP-scaled projections and MLP down weights

### Quantization & Compression

- **GPTQ-lite**: Per-row optimal clip percentile search (5 candidates: .999, .9995, .9999, .99999, 1.0) for int6
- Int6 per-row for all large weight matrices
- Control tensors in fp32, small tensors in fp16
- **LZMA compression** (preset 9, extreme mode) — saves ~3MB vs zlib/zstd
- Int8 format exceeds 16MB (22.3MB); int6 selected automatically

### Evaluation

- Sliding window evaluation at stride=16 (2048 seq_len)
- **Temperature scaling T=0.90** — empirically optimal for LeakyReLU² activations
- Peak VRAM: 21,533 MiB per GPU

### Run Command

```bash
# 8×H100 SXM (10 minute wallclock)
SEED=1337 \
ITERATIONS=12000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=786432 \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 TRIGRAM_VOCAB_SIZE=0 \
XSA_LAST_N=4 ROPE_DIMS=16 GATED_ATTENTION=0 VALUE_RESIDUAL=0 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10" \
DEPTH_RECUR_LAYERS="" DEPTH_RECUR_PASSES=1 \
LATE_QAT_THRESHOLD=0.15 WARMDOWN_ITERS=3500 \
SWA_ENABLED=1 SWA_EVERY=50 \
EMA_START_FRAC=0.4 LR_WARMUP_STEPS=50 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3 \
TORCH_COMPILE=1 TTT_ENABLED=0 \
EVAL_STRIDE=16 EVAL_TEMPERATURE=0.90 \
TRAIN_LOG_EVERY=200 VAL_LOSS_EVERY=2000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Reproducibility

Trained and evaluated on RunPod 8×H100 SXM using `runpod/parameter-golf:latest` image (PyTorch 2.9.1, CUDA 12.8). Training completes in exactly 10 minutes (600s wallclock). Both seeds produce consistent results (std=0.0001 BPB) with artifacts well within 16MB.
