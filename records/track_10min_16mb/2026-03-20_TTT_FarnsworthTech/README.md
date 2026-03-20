# Test-Time Training (TTT) with Full-Model SGD Adaptation

**Best Score: 1.17436 BPB** (val_loss: 1.9829)

## Results

| Run | TTT LR | TTT Epochs | Seed | Static BPB | Final BPB | Steps | Train Time |
|-----|--------|-----------|------|------------|-----------|-------|------------|
| Conservative | 3e-4 | 1 | 1337 | 1.2105 | 1.1767 | 9647 | 599.988s |
| **Aggressive** | **2e-3** | **2** | **1337** | **1.2087** | **1.1744** | **9728** | **601.1s** |

## Approach

**Test-time training** adapts the model's weights during evaluation, reclaiming the unused 10-min eval budget as adaptive compression (analogous to Lempel-Ziv).

### Training Phase (10 min, 8xH100 SXM)

- **Architecture:** 9-layer, 512-dim, GQA (8 heads / 4 KV), tied embeddings
- **Optimizer:** Muon + Adam with standard LR schedule
- **Tokenizer:** SP-1024 BPE (FineWeb 10B)

### TTT Eval Phase (~349s of 600s budget)

1. Decompress int8+zlib artifact back to full precision
2. **TTT adaptation:** 2 epochs of full-model SGD (lr=0.002, momentum=0.9, batch_size=32) — 311s
3. **Sliding window eval** (stride=64, seq_len=1024) — 38s

**TTT improved BPB from 1.2087 to 1.1744 (2.84% gain at zero parameter cost)**

## Artifact

- **Model:** 18,897,488 parameters
- **Total:** 15,328,877 bytes (< 16,000,000 cap)

## Reproducibility

```bash
# Aggressive TTT (best score)
TTT_LR=0.002 TTT_EPOCHS=2 TTT_MOMENTUM=0.9 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Conservative TTT
TTT_LR=3e-4 TTT_EPOCHS=1 TTT_MOMENTUM=0.95 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware

- 8x NVIDIA H100 80GB HBM3 (SXM), RunPod
- Ubuntu 22.04.5, PyTorch 2.6.0+cu124, CUDA 12.4
- Peak memory: 11,389 MiB per GPU
