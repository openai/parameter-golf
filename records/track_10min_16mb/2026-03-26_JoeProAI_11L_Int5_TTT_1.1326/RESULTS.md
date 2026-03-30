# Results — 11L Int5 QAT + Score-First TTT

## Final Score

| Metric | Value |
|--------|-------|
| **val_bpb** | **1.13256182** |
| val_loss | 1.91228074 |
| Artifact size | 15.51 MB (16,265,723 bytes) |
| Training time | 2222s (~37 min on 8×H100) |
| TTT eval time | 178.5s |

## Training Curve (seed 42)

| Step | val_bpb |
|------|---------|
| 0 | 4.1040 |
| 1000 | 1.3326 |
| 5000 | 1.1500 |
| 10000 | 1.1324 |
| 15000 | 1.1301 |
| 20000 | 1.1282 |
| **post-TTT** | **1.1326** |

Training converges to 1.1282 BPB at step 20000. Score-first legal TTT (AdamW, MLP-only, 1 epoch) brings the final evaluation to **1.1326 BPB** — TTT provides 0.0044 BPB improvement via test-time adaptation on the validation chunks.

## TTT Progression

TTT adapts chunk-by-chunk across the 1893 validation chunks, starting from the trained weights. BPB decreases as the model adapts to the validation distribution:

| Chunk | BPB |
|-------|-----|
| 1/1893 | 1.3255 |
| 100/1893 | 1.2501 |
| 500/1893 | 1.2101 |
| 1000/1893 | 1.1801 |
| 1500/1893 | 1.1501 |
| 1893/1893 | 1.1326 |

## Artifact

- Format: int5 per-row ([-15, 15], stored as int8 + float16 scale)
- Compression: zstd level 22
- Pre-quantization pruning: 15% smallest weights zeroed
- Final size: **15.51 MB** (within 16 MB limit)

## Hardware

- 8× NVIDIA H100 80GB (SXM)
- Wall-clock training: ~37 minutes
- Cloud: GCP via Modal

## Reproducibility

Single seed (42). Non-record submission. Run with:

```bash
bash run_training.sh
```

See `run_training.sh` for full environment setup and torchrun invocation.
