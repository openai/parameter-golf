# LeakyReLU² + BigramHash + Mixed Int5/Int6 + Sliding Window + EMA + AdamW TTT

**Non-record submission** — tested on 1×H100 only (credit-limited). Seeking 8×H100 for official record attempt.

## Results (1×H100 80GB, 10 minutes)

| Metric | Value |
|--------|-------|
| **Sliding Window val_bpb** | **1.3036** |
| Standard val_bpb | 1.3373 |
| Int8 roundtrip val_bpb | 1.3373 |
| Quant degradation | 0.0005 |
| Steps | 1,090 |
| Step avg | 553ms |
| Peak memory | 11,396 MiB |
| Compressed size | 15,893,048 bytes |

## 8×H100 Projection

Based on throughput scaling (grad_accum 8→1), expect ~8,500-9,500 steps:
- **Projected standard val_bpb: ~1.20-1.23**
- **Projected sliding val_bpb: ~1.17-1.19**

## Key Techniques

### 1. LeakyReLU(0.5)² Activation
One-line change from relu² that preserves negative gradient flow. Ablated at -0.003 BPB by PR #493.

```python
F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```

### 2. Mixed Int5/Int6 Quantization + FP16 Embeddings
- **Int5 [-16,15]** for MLP weights (highly compressible, ~1.88x zstd ratio)
- **Int6 [-32,31]** for attention weights (precision-sensitive)
- **FP16** for tied embeddings (errors compound through both input/output paths)

### 3. BigramHash(1536, dim=128)
Hash consecutive token pairs into 1536-bucket embedding table, projected to model_dim via learned linear. Provides cheap n-gram features.

### 4. EMA(0.997)
Exponential Moving Average of weights during training for smoother convergence and better quantization robustness.

### 5. AdamW Test-Time Training (Pre-Quantization)
Score-first legal TTT protocol:
- Val tokens split into 32K-token chunks
- For each chunk: SCORE under `torch.inference_mode()`, then TRAIN with AdamW
- TTT runs **before** quantization on full-precision weights (key insight from PR #1006)
- AdamW instead of SGD (SGD fails on CastedLinear architectures)

### 6. LZMA Compression
Uses LZMA (preset=extreme) instead of zlib for ~280KB savings in artifact size.

### 7. Sliding Window Evaluation (stride=64)
Every token scored with ~960+ context tokens instead of the variable 0-1023 average. Free ~0.03 BPB improvement.

### 8. Muon Optimizer + Weight Decay
- Matrix params: Muon(lr=0.02, momentum=0.99, WD=0.04)
- Scalar/embed params: AdamW(WD=0.04)
- Gradient clipping at 0.3

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 3× expansion with LeakyReLU(0.5)² |
| BigramHash | 1536 buckets, dim=128 |
| Embeddings | Tied input/output, FP16 export |
| Skip connections | U-Net style |
| Logit cap | softcap=30 |
| Seq length | 1024 |

## Training

- Warmup: 20 compilation steps (state restored)
- LR schedule: Wallclock-aligned warmdown (iterations=1050, warmdown=150)
- SWA: Last 40% of warmdown, every 10 steps
- Batch: 524,288 tokens per step (8× grad accumulation on 1 GPU)

## Run Command

```bash
RUN_ID=submission \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt_v2.py
```

## Acknowledgments

Built on insights from the Parameter Golf community, particularly:
- PR #493 (LeakyReLU²)
- PR #414 (BigramHash, XSA)
- PR #1006 (AdamW TTT, pre-quant TTT, LZMA)
- PR #162 (SmearGate, OrthoInit)
