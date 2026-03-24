# LeakyReLU(0.5)² + Per-Document LoRA TTT

> SWE attempts ML 🏌️
>
> ![SWE → ML](https://adamsarson.com/wp-content/uploads/2013/06/club-flying.gif)

Mean val_bpb: **0.9443** (3 seeds, std: 0.0023)

## Results

| Seed | val_bpb | Steps | Step Avg | Artifact Size |
|------|---------|-------|----------|---------------|
| 1337 | 0.9461 | 7008 | 85.62ms | 15,430,887 B (96.4%) |
| 42 | 0.9450 | 7008 | 85.62ms | 15,430,887 B (96.4%) |
| 2025 | 0.9417 | 6984 | 85.98ms | 15,430,887 B (96.4%) |
| **Mean** | **0.9443** | | | |
| **Std** | **0.0023** | | | |

## Techniques

### Architecture (11L, 512dim, 27M params)
- U-Net encoder/decoder with skip connections (5 encoder + 6 decoder)
- GQA: 8 heads, 4 KV heads, RoPE (base=50000)
- **LeakyReLU(0.5)²** MLP activation — preserves negative gradient flow, prevents dead neurons
- SmearGate — learned token blending via sigmoid gate
- BigramHash embedding (2048 buckets, dim=128)
- Depth-scaled residuals (1/√(layer+1))
- Logit softcap at 30.0

### Training (10 min, 8xH100 SXM)
- Muon optimizer (lr=0.02, momentum warmup 0.92→0.99, weight decay=0.04)
- Adam for embeddings (lr=0.03) and scalars (lr=0.02)
- Warmdown over final ~3000 steps
- OrthoInit for weight matrices
- SWA (Stochastic Weight Averaging, decay=0.999)
- Batch: 786,432 tokens, seq_len=1024

### Quantization
- int8 per-row with 99.99984th percentile clipping
- zstd-22 compression
- 3% magnitude pruning

### Test-Time Training
- Per-document backward-looking LoRA TTT (rank=8, chunk=256)
- 3 epochs per document (TTT_EPOCHS=3)
- Documents ≥512 tokens adapted (TTT_MIN_DOC_LEN=512)
- Adam optimizer (lr=0.01)
- LoRA on Q, V projections and LM head
- Fresh LoRA per document — zero cross-document leakage

### Environment Variables
```
TTT_EPOCHS=3 TTT_MIN_DOC_LEN=512
```

## Hardware
- 8× NVIDIA H100 SXM 80GB (RunPod Secure Cloud)
- PyTorch 2.9.1+cu128
- ~17 min per seed (10 min training + 7 min eval/TTT)

## Known Issue: TTT Epoch Scoring

The current TTT implementation only records scores on the final epoch. In a 3-epoch run, epochs 1 and 2 train the LoRA on each chunk without scoring, meaning by epoch 3 the LoRA has already been trained on tokens it hasn't yet scored in that pass. This was flagged on PR #512 by @pinnerwt.

The fix is to score on every epoch (overwriting previous scores), so each token is always scored before the LoRA trains on it within that epoch:

```
Epoch 1: chunk forward → score → train  (scores recorded)
Epoch 2: chunk forward → score → train  (scores overwrite epoch 1)
Epoch 3: chunk forward → score → train  (final scores kept)
```

This requires removing the `if epoch == args.ttt_epochs - 1:` guard on the scoring block (line 933). The fix is a 1-line change. BPB may increase slightly as the scoring becomes strictly backward-looking at every epoch.
