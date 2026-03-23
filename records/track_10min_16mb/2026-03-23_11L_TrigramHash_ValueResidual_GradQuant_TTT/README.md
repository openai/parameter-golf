## Record: 11L TrigramHash + Value Residual + Gradient-Guided Quant + AdamW TTT

**val_bpb = 1.1101** (sliding window stride=64, best seed 2024) | **15.34 MB** artifact | 8xH100 SXM, 600s

### Three Novel Contributions

Built on the PR #398/#442 baseline (11L EMA + AdamW TTT), this submission adds three techniques that improve quality-per-step, compensating for their compute overhead:

1. **TrigramHash Embedding (4096 buckets, dim=128).** Extends BigramHash to 3-token context. Hash function: `xor(36313*t[i], 27191*t[i-1], 51497*t[i-2]) % 4095`. Added to token embedding before transformer blocks. Cost: ~0.5M params, ~10ms/step.

2. **Value Residual (ResFormer, arXiv:2410.17897).** Cache V vectors from the first attention layer, blend into all subsequent layers via learned scalars: `v = lambda[0] * v0 + lambda[1] * v`. Just 2 params per layer (22 total). Provides cross-layer value coherence.

3. **Gradient-Guided Adaptive Quantization.** During the last 10% of warmdown, accumulate per-tensor squared gradient magnitudes (zero throughput cost — gradients already computed). At quantization time, rank tensors by sensitivity:
   - Top 10%: Int7 (63 levels) — still learning, need precision
   - Middle 70%: Int6 (31 levels) — standard
   - Bottom 20%: Int5 (15 levels) — converged, tolerate compression

### Results (3-seed, 8xH100 SXM)

| Seed | Steps | Sliding BPB (s64) | Artifact |
|------|-------|-------------------|----------|
| 42 | 5,190 | 1.1177 | 15.54 MB |
| 1337 | 5,925 | 1.1118 | 15.76 MB |
| **2024** | **5,930** | **1.1101** | **15.34 MB** |

**Mean: 1.1132 | Std: 0.0040**

### Ablation: Additions vs Baseline

Same hardware (8xH100 SXM, 600s), same TTT config:

| Config | Steps | BPB | Delta |
|--------|-------|-----|-------|
| Baseline (PR #398 stack, no additions) | 6,613 | 1.1403 | — |
| **+ TrigramHash + ValueResidual + GradQuant** | **5,190** | **1.1177** | **-0.023** |

Despite 22% fewer training steps (5,190 vs 6,613) due to compute overhead, the three additions improve BPB by -0.023. Quality-per-step is significantly higher.

### Architecture

- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA)
- 3x MLP relu-squared + SmearGate + BigramHash(4096) + TrigramHash(4096)
- Value Residual across all layers
- EMA (decay=0.997), OrthoInit
- Partial RoPE (16/64 dims), LN Scale
- Int6 QAT + Gradient-Guided adaptive Int5/6/7 + zstd-22

### TTT Configuration

- AdamW (lr=0.0005, weight_decay=0.0), 10 epochs
- All blocks unfrozen (freeze_blocks=0)
- TTT time: ~154s on 8xH100

### Training Configuration

- Muon WD=0.04, Adam WD=0.04
- Matrix LR=0.025, Scalar LR=0.025, Embed LR=0.035
- Muon momentum: 0.92 → 0.99 (1500 step warmup)
- Warmdown: 3000 iterations
- Batch: 786,432 tokens, seq_len=2048

### Run Command

```bash
SEED=2024 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are set as defaults in train_gpt.py — no environment variables needed.
