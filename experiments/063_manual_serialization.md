# Experiment 063: Manual Serialization Test

## Status: COMPLETED

## Results
| Metric | Value |
|--------|-------|
| Sliding eval | **1.1477 BPB** (matches 060) |
| torch.save + zstd artifact | 17,979,878 bytes |
| **manual + zstd artifact** | **17,639,406 bytes (340KB / 1.9% smaller)** |

## Finding
Manual serialization saves 340KB — helpful but not enough (still 1.64MB over 16MB).
The dtype-grouped version (int8 together, fp16 together) should compress better.
Need ~1.6MB more savings from: dtype grouping + outlier splitting + testing on Runpod.
