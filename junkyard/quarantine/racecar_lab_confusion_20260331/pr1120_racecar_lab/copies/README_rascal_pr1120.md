# Rascal — val_bpb 1.1099 (3-seed mean)

**Junkyard Rat Rascal II**: 11L XSA-all + Parallel Muon + Coprime loader, no GPTQ, naive int6 + zstd (~15.5MB).

## Results

| Seed | val_bpb (sliding window) | Steps | Size |
|------|--------------------------|-------|------|
| 42   | 1.11018163               | 6593  | 15,540,001 bytes |
| 300  | 1.10979099               | 6593  | 15,542,719 bytes |
| 444  | 1.10986874               | 6593  | 15,554,053 bytes |
| **mean** | **1.1099**           |       | **15,554,053 bytes (max)** |

Hardware: 8×H100 SXM, 600s wallclock cap.

## Config

- 11 layers, XSA-all (all layers use cross-shard attention)
- GQA: 8 heads, 4 KV heads
- Bigram hash table: 2048
- RoPE: 16
- Coprime loader (batch_stride=47 for seeds 42/444, 63 for seed 300)
- SWA starting ~step 5900
- Late QAT at ~step 6070 (scale=0.15)
- Parallel Muon optimizer
- SKIP_GPTQ=1 — naive int6 quantization (5 layers + embed), zstd compressed
- 26.99M parameters

## Reproduce

```bash
# Set env and run from repo root
SKIP_GPTQ=1 torchrun --nproc_per_node=8 records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py \
    --seed 42
```

See `train_seed42.log`, `train_seed300.log`, `train_seed444.log` for full run output.
