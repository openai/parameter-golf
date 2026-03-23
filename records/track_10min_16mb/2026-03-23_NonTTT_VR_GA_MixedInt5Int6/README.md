# NonTTT VR GA MixedInt5Int6

**Mean val_bpb: 1.1428** (3 seeds: 1337, 42, 2025)

## Key Techniques

1. **Non-TTT route**: No test-time training; standard train + quantize + compress pipeline.
2. **Mixed Int5/Int6 quantization** with zstd-22 compression.
3. **GQA attention** (8 heads, 4 KV heads) with tied embeddings.
4. **Sliding window evaluation** (stride=64) for final scoring.
5. **EMA weights** applied before export.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact Bytes |
|------|----------|---------|-------|---------|----------------|
| 1337 | 1.9296 | 1.14280 | 4622 | 129.82 | 16,026,184 |
| 42 | 1.9296 | 1.14281 | 4623 | 129.79 | 16,339,774 |
| 2025 | 1.9297 | 1.14287 | 4632 | 129.55 | 16,244,044 |
| **Mean** | **1.9296** | **1.14283** | | | |

## Notes

- All seeds trained for exactly 600s wallclock (8xH100 SXM).
- Peak memory: ~27 GiB allocated.
- **Warning**: All 3 seeds produce artifacts exceeding the 16MB limit (by 26KB–340KB).
- Eval method: `final_int6_sliding_window_exact`.
