# 10L Int6 QAT + BigramHash + Muon WD + Sliding Window

## Results

| Seed | Steps | val_bpb (standard) | val_bpb (sliding) | Artifact size |
|------|-------|--------------------|--------------------|---------------|
| 1337 | ~8,500 | ~1.1807 | 1.1572 | ~14.2MB |
| 42 | ~8,500 | ~1.1810 | 1.1581 | ~14.2MB |
| 3 | ~8,500 | ~1.1812 | 1.1578 | ~14.2MB |

**Mean val_bpb (sliding): 1.1577** (std: 0.00047)

Statistical significance vs baseline (2.0727 val_loss):
- Improvement: 0.1180 nats, t=-245.7, p << 0.01

## Techniques

1. **10 transformer layers** (from 9 baseline)
2. **STE int6 QAT** — fake quantization during training eliminates quant gap
3. **Full int6 quantization** [-31,31] + **zstd-22** compression
4. **MLP hidden 1344** (2.625x model_dim)
5. **BigramHash** — hash-based bigram embedding (4096 buckets, 128-dim)
6. **FP16 tied embedding passthrough**
7. **Sequence length 2048**
8. **Muon momentum 0.99** with **weight decay 0.02**
9. **Lower LRs**: MATRIX_LR=0.02, SCALAR_LR=0.02
10. **Gradient clipping** 0.3
11. **Warmdown 3000**
12. **Sliding window evaluation** stride=64

## Hardware

8xH100 80GB HBM3, PyTorch 2.8.0+cu128, ~71ms/step.
Requires: `pip install zstandard`
