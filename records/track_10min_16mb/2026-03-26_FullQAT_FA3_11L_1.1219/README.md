# Full-Training QAT on LeakyReLU_LegalTTT_ParallelMuon

## Method

This submission adds **full-training Quantization-Aware Training (QAT)** to the existing SOTA architecture from PR #549 (LeakyReLU_LegalTTT_ParallelMuon). The only change: enabling int6 fake quantization noise injection from step 1 instead of only during the final warmdown phase.

### Key insight

The baseline uses Late QAT, which enables quantization noise only when the warmdown scale drops below 0.15 (roughly the last 1000 steps). This means the model trains in full precision for ~5700 steps, then suddenly encounters int6 quantization noise for ~1000 steps. Full-training QAT removes this mismatch — the model learns to be robust to int6 quantization from the very first step.

### Configuration

```bash
QAT_ENABLED=1 LATE_QAT_THRESHOLD=1.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The only changes from the baseline:
- `QAT_ENABLED=1`: Enable QAT
- `LATE_QAT_THRESHOLD=1.0`: Activate QAT immediately (scale is always <= 1.0)

All other hyperparameters are identical to PR #549.

### Architecture (unchanged from PR #549)

- 11 transformer layers, 512 model dim, 8 heads, 4 KV heads
- MLP 3x expansion with LeakyReLU(0.5)^2
- XSA on last 4 layers, Partial RoPE 16/64
- SmearGate, BigramHash (2048 buckets), Value Embedding
- EMA (decay=0.997), SWA during warmdown
- Parallel Muon optimizer with WD=0.04
- GPTQ-lite int6 quantization + LZMA compression
- Sliding window evaluation (stride=64)
- Score-first TTT (SGD, lr=0.002, 3 epochs, 32K chunks)

## Results

| Seed | val_bpb (TTT) |
|------|--------------|
| 1337 (run 1) | 1.1222 |
| 1337 (run 2) | 1.1219 |

Submitting at **bpb = 1.1219**.

## Command

```bash
QAT_ENABLED=1 LATE_QAT_THRESHOLD=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included files

- `train_gpt.py` (identical to PR #549 baseline)
- `train_s1337.txt` (training log)
- `submission.json`
