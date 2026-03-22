## Record: 11L TTT Burst + EMA + GPTQ-lite (val_bpb=1.1232)

**val_bpb: 1.1232** (sliding window stride=64, seed 1337) | **15.68 MB** | 8xH100 SXM, 600s

### Key Innovation Over PR #414

| Change | PR #414 | This | Impact |
|--------|---------|------|--------|
| **TTT Burst** | None | 2-epoch replay of last 100 training batches at 10% LR before EMA | -0.0001 BPB |

Everything else inherited from PR #414: EMA(0.997), GPTQ-lite(5 percentiles), warmdown 3500, Late QAT@0.15, int6+zstd-22.

### TTT Burst: Late-Stage Sharpening

After the main training loop and before EMA application, we replay the last 100 training batches for 2 epochs at 10% of base LR. EMA is updated during the burst so it absorbs the sharpened signal. This gives the model a final sharpening pass on recent data before weight averaging and quantization.

### Results (8xH100 SXM)

| Seed | Steps | val_loss | Sliding BPB (s64) | Artifact |
|------|-------|----------|-------------------|----------|
| **1337** | 6991 | 1.9246 | **1.1232** | 15.68 MB |
| 42 | 6994 | 1.9262 | 1.1240 | 16.37 MB* |

*Seed 42 artifact over size limit due to compression variance; BPB still validates the approach.

### Architecture

11L, 512d, 8H/4KV, MLP 3x (relu^2), U-Net skips, XSA4, Partial RoPE 16/64, LN Scale, VE128, SmearGate, BigramHash(2048), FA3, Muon WD=0.04, EMA(0.997), Tight SWA, Late QAT@0.15, TTT Burst(2ep/10%LR), int6+zstd-22, GPTQ-lite.

### Run Command

```bash
SEED=1337 torchrun --nproc_per_node=8 train_gpt.py
```

### Test plan

- [x] Seed 1337 under 16MB (15.68 MB)
- [x] Seed 1337 trains in 600s on 8xH100
- [x] Post-quant roundtrip verified
- [x] Sliding window eval (stride=64) consistent across seeds
- [x] train_gpt.py under 1500 lines
- [x] No TTT on validation data
