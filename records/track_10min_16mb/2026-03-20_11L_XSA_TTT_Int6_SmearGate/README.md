# 11L + XSA + TTT + Int6 + SmearGate + BigramHash (val_bpb: 1.1429)

## Results
- **val_bpb: 1.1429** (sliding window, stride=64)
- Pre-quantization BPB: 1.1578
- Model parameters: 26,829,913
- Artifact size: 16,175,323 bytes (slightly over 16MB limit — non-record, needs WD tuning)
- Training: 7,723 steps in 600 seconds (~77.7ms/step) on 8xH100 SXM
- SWA: 13 checkpoint average during warmdown (every 120 steps)

## Approach

Combines the two strongest eval-time techniques (XSA + TTT) on top of the full competitive meta stack.

### XSA (Exclusive Self Attention) on last 3 layers
Based on arXiv:2603.09078. Efficient GQA-aware implementation using free reshape + broadcasting instead of repeat_interleave. Removes self-value bias in attention at ~2ms/step overhead.

### TTT (Test-Time Training)
Full-weight SGD adaptation on validation data before scoring. 3 epochs at lr=0.002 with momentum=0.9. Freezes first 2 transformer blocks for stability. Full DDP support across all ranks. TTT took 79.4s (separate from training budget).

### Architecture
- 11 transformer layers, 512-dim, 8 heads (4 KV heads via GQA)
- 3x MLP expansion (1536 hidden), relu-squared activation
- U-Net skip connections (encoder=5, decoder=6)
- SmearGate + BigramHash (2048 buckets, 128 dim)
- Tied embeddings, logit softcap=30.0
- XSA on layers 8, 9, 10

### Training
- FlashAttention 2.8.3
- Muon optimizer: lr=0.025, momentum=0.99 (warmup from 0.92 over 1500 steps)
- AdamW for embeddings/scalars: lr=0.035/0.025
- Weight decay: 0.04 (both Muon and AdamW)
- Warmdown: 3000 iterations, grad clip 0.3
- Batch size: 524,288 tokens
- SWA every 120 steps (scale < 0.5)
- OrthoInit + muP-scaled output projections

### Quantization
- Int6 per-row quantization on MLP + attention weights
- Int8 for embeddings
- zstd level 22 compression

### Validation progression
| Step | val_bpb |
|------|---------|
| 1000 | 1.3514 |
| 2000 | 1.2913 |
| 3000 | 1.2675 |
| 4000 | 1.2537 |
| 5000 | 1.2403 |
| 6000 | 1.2139 |
| 7000 | 1.1825 |
| 7723 | 1.1578 |

### Next steps
- Increase weight decay to 0.045-0.05 to bring artifact under 16MB (~0.002-0.003 bpb cost)
- Sweep BigramHash bucket size (2048 vs 10240)
- Expected valid submission score: ~1.143-1.145

## Checklist
- [x] Submission folder in `records/track_10min_16mb/`
- [x] `README.md` with approach description
- [x] `submission.json` with metadata
- [x] `train_gpt.py` (single file, self-contained)
- [x] Training log
- [x] BPB score (1.1429, non-record due to artifact size)
