# 11L + XSA + TTT + Int6 + SmearGate + BigramHash

## Results
- **val_bpb:** Pending 8xH100 validation (applying for compute grant)
- Model parameters: ~27M
- Target artifact size: <16MB (int6 + zstd-22)
- Training: 8xH100 SXM, 600s

## Approach

Combines the two strongest eval-time techniques (XSA + TTT) on top of the full competitive meta stack. Neither technique costs training time — XSA adds ~2ms/step overhead, and TTT runs entirely during eval.

### Novel Combination: XSA + TTT Stacking

No existing submission combines both:
- **XSA** (Exclusive Self Attention) on last 3 layers removes self-value bias (~0.002 bpb)
- **TTT** (Test-Time Training) adapts weights via SGD on val data before scoring (~0.005 bpb)

These are complementary — XSA improves the model's attention mechanism, TTT adapts the full model to the eval distribution.

### Architecture
- 11 transformer layers, 512-dim, 8 heads (4 KV heads via GQA)
- 3x MLP expansion (1536 hidden), relu-squared activation
- U-Net skip connections
- SmearGate + BigramHash (2048 buckets, 128 dim)
- Tied embeddings, logit softcap=30.0
- XSA on layers 8, 9, 10

### Training
- FlashAttention 3
- Muon optimizer: lr=0.025, momentum=0.99 (warmup from 0.92 over 1500 steps)
- AdamW for embeddings/scalars: lr=0.035/0.025
- Weight decay: 0.04 (both Muon and AdamW)
- Warmdown: 3000 iterations, grad clip 0.3
- Batch size: 524,288 tokens (optimized per saml212's finding)
- SWA every 120 steps during warmdown
- OrthoInit + muP-scaled output projections

### TTT Configuration
- SGD with momentum=0.9, lr=0.002, 3 epochs
- Freezes first 2 transformer blocks for stability
- Full DDP support across all ranks
- Gradient clipping at 1.0

### Quantization
- Int6 per-row quantization on MLP + attention weights
- Int8 for embeddings
- zstd level 22 compression

## Checklist
- [x] Submission folder in `records/track_10min_16mb/`
- [x] `README.md` with approach description
- [x] `submission.json` with metadata
- [x] `train_gpt.py` (single file, self-contained)
- [ ] Training log (pending compute)
- [ ] BPB score (pending compute)
