## Record: 11L Depth Recurrence + EMA Tuning (0.9965) (val_bpb: 1.0925)

**val_bpb: 1.0925** (sliding window stride=64, 3-seed mean) | **15.95 MB** (mean) | 8xH100 SXM, 590s

### Key Innovation Over PR #1334

Hyperparameter refinement on the EMA decay constant, built on PR #1334's (@aryanbhosale) depth recurrence architecture:

| Change | PR #1334 | This | Impact |
|--------|----------|------|--------|
| **EMA decay** | 0.997 | 0.9965 | Stabilized post-quantization performance, reduced destructive pruning |

### EMA Decay Tuning

By lowering the EMA decay from 0.997 to 0.9965, the exponential moving average assigns slightly more weight to recent training steps. This produces a final checkpoint that quantizes more cleanly under GPTQ int6, reducing the number of values requiring selective pruning (~290K vs baseline).

### Results (3 seeds, 8xH100 SXM)

| Seed | Pre-quant BPB | Sliding BPB (s64) | Artifact |
|------|---------------|-------------------|----------|
| 42 | 1.0965 | **1.0921** | 15,954,858 B |
| 1337 | 1.0973 | **1.0928** | 15,959,674 B |
| 2024 | 1.0969 | **1.0926** | 15,948,766 B |

**Mean: 1.0925 | Std: 0.0004** | All artifacts under 16,000,000 bytes

### Architecture (from PR #1334)

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- Depth recurrence: layers 4,5 repeat (virtual 13 layers), activated at step 3000
- Skip gates (learnable residual gating)
- Shared Value Embedding (dim=128, layers 9,10)
- Tied embeddings, logit softcap=30.0
- SP4096 tokenizer (SentencePiece BPE)

### Training

- FlashAttention 3 (Hopper-optimized)
- Muon optimizer (matrices): lr=0.02, momentum=0.99, WD=0.09, backend_steps=5
- Adam (head params): lr=0.008, fused=True
- AdamW (embeddings): lr=0.6, WD=0.09, fused=True
- AdamW (scalars): lr=0.02, WD=0.02, fused=True
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 66.7% of training
- **EMA**: decay=0.9965, every step
- Wallclock cap: 600s (590s effective, 10s reserved for GPTQ)

### Quantization

- GPTQ int6 with percdamp=0.05, 64 calibration batches
- Selective pruning of lowest-error values to fit 16MB
- Brotli compression
- ~290K values pruned (minimal impact)

### Reproducibility

All 3 seeds produce valid artifacts under 16MB with tight variance (std=0.0004 BPB). Training completes in ~590s with ~5200-5400 steps depending on seed.

### Attribution

Base architecture and training recipe from PR #1334 by @aryanbhosale.
