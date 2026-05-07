## 11L XSA6 + Warmdown3000 + QAT@0.30 (val_bpb: 1.1352)

**val_bpb: 1.1360** (sliding window stride=64, 2-seed mean) | **15.88 MB** (best seed) | 8xH100 SXM, 600s

### Changes from SOTA (PR #414, 1.1228 BPB)

Three targeted hyperparameter changes identified through 37 local ablation experiments on an RTX 4060 Ti:

| Change | SOTA (PR #414) | Ours | Rationale |
|--------|---------------|------|-----------|
| **XSA layers** | last 4 | last 6 | More context-only attention layers |
| **Warmdown** | 3500 iters | 3000 iters | Shorter cooldown preserves more full-LR training |
| **Late QAT threshold** | 0.15 | 0.30 | Earlier QAT gives more steps to adapt to int6 |

### Results (2 seeds, 8xH100 SXM)

| Seed | Steps | ms/step | val_loss | Sliding BPB (s64) | Artifact |
|------|-------|---------|----------|-------------------|----------|
| **42** | 5,447 | 110.2 | 1.9168 | **1.1352** | 15,883,805 bytes |
| 1337 | 5,448 | 110.1 | 1.9192 | 1.1367 | 15,730,868 bytes |

**Mean: 1.1360 | Std: 0.0008** | Submitted: seed 42 (best)

### Note on Step Count

This submission achieves ~5,400 steps in 600s (110ms/step) compared to SOTA's ~7,100 steps (85ms/step). The difference is due to PyTorch SDPA attention fallback — FlashAttention 3 (Hopper) was not available in our deployment environment. With FA3, we would expect ~7,000 steps and correspondingly lower BPB.

### Architecture (from SOTA PR #414)

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), relu-squared activation
- U-Net skip connections (5 encoder, 6 decoder)
- **Efficient Partial XSA on last 6 layers** (extended from 4)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- Shared Value Embedding (dim=128, layers 9,10)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training

- PyTorch SDPA attention (FA3 unavailable)
- Muon optimizer (matrices): lr=0.025, momentum=0.99 (warmup 0.92->0.99 over 1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- **Warmdown: 3000 iterations** (shortened from 3500)
- EMA: decay=0.997, every step
- **Late QAT: STE int6 when LR scale < 0.30** (raised from 0.15)
- OrthoInit + muP-scaled output projections

### Quantization

- GPTQ-lite: Per-row optimal clip percentile search (5 candidates) for int6
- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

### Run Command

```bash
BIGRAM_VOCAB_SIZE=2048 WARMDOWN_ITERS=3000 LATE_QAT_THRESHOLD=0.30 XSA_LAST_N=6 \
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Local Ablation Methodology

We conducted 37 experiments across 8 hyperparameter dimensions on a single RTX 4060 Ti (500 steps, ~5 min/run):

| Dimension | Configs Tested | Best (Local) | Transfer to H100? |
|-----------|---------------|--------------|-------------------|
| BigramHash buckets | 2048, 3072, 4096, 5120 | 5120 (-0.003) | Size-limited on H100 |
| EMA decay | 0.995, 0.997, 0.998, 0.999 | 0.995 (-0.163) | No (step-dependent) |
| Warmdown ratio | 42%, 49%, 56%, 63% | 42% (-0.004) | **Yes** |
| Matrix LR | 0.015–0.050 | 0.040 (-0.058) | No (step-dependent) |
| Gradient clip | 0.1, 0.3, 0.5, 1.0 | 0.3 (confirmed) | N/A |
| QAT threshold | 0.05–0.30 | 0.30 (-0.009) | **Yes** |
| XSA layers | 0, 2, 4, 6 | 6 (-0.001) | **Yes** |
| Muon momentum | 0.95, 0.97, 0.99, 0.995 | 0.995 (-0.004) | Uncertain |

Key finding: EMA decay and learning rate are strongly step-count-dependent. Optimal values for 500-step local runs do not transfer to 5,400-step H100 runs. Only proportional (warmdown, QAT threshold) and architectural (XSA layers) changes transfer reliably.

### Compliance

- [x] 2 seeds run on 8xH100 SXM
- [x] Both seeds train in <=600s
- [x] Both seeds artifact <=16,000,000 bytes (max: 15,883,805)
- [x] Sliding window eval stride=64
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] Self-contained train_gpt.py
