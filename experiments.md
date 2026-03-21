# Experiment Log

## 1xH100 SXM Runs (10 min cap, relative comparison only)

| # | Date | Config | Steps | val_bpb (std) | val_bpb (slide) | Size (MB) | Notes |
|---|------|--------|-------|---------------|-----------------|-----------|-------|
| 1 | 03-19 | Baseline + MTP(2) | 1078 | 1.3500 | — | — | MTP only, no memory tokens |
| 2 | 03-19 | MTP(2) + compile fix | 1124 | 1.3415 | — | — | Fixed torch.compile recompilation |
| 3 | 03-19 | Memory tokens(32) + seq1024 | 1336 | 1.3333 | — | — | Memory tokens beat MTP |
| 4 | 03-19 | Memory tokens(32) + seq2048 | 1379 | 1.3080 | — | — | Best single-feature result |
| 5 | 03-19 | Mem(32) + MTP(2) + seq2048 + 9L | 1087 | 1.3263 | broken | 13.0 | Sliding window was overwriting, not prepending |
| 6 | 03-19 | Mem(32) + MTP(2) + seq2048 + 10L | 1069 | 1.3220 | 1.3102 | 14.3 | 10 layers helped; sliding window fixed + batched |
| 7 | 03-19 | Mem(32) + seq2048 + 10L (no MTP) | 1079 | 1.3227 | 1.3110 | 14.3 | MTP removal = wash on 1xH100 |
| 8 | 03-20 | Mem(32) + seq2048 + 10L + spectral init + WD (no MTP) | 1076 | 1.3369 | — | 12.5 | Init changes hurt — reverted |
| 9 | 03-20 | Mem(64) + MTP(2) + seq2048 + 10L + WD (no init) | 1056 | 1.3173 | 1.3006 | 13.8 | 64 mem tokens + WD + no WD on mem |
| 10 | 03-20 | Mem(64) + MTP(2) + 10L + bigram + smear + 3xMLP + WD | 1053 | 1.3060 | 1.2898 | 17.3* | *Over 16MB with int8+zlib, needs int6+zstd |

## 8xH100 SXM Runs (submission quality)

| # | Date | Config | Steps | val_bpb (std) | val_bpb (slide) | Size (MB) | Notes |
|---|------|--------|-------|---------------|-----------------|-----------|-------|
| 1 | 03-21 | Mem(64) + MTP(2) + 10L + bigram + smear + 3xMLP + WD + int6+zstd | 9403 | 1.1832 | 1.1670 | 17.3* | *Over 16MB! int6+zstd not enough, need int5 MLP |

## Key Findings

- Memory tokens (32) > MTP auxiliary heads on 1xH100
- MTP costs ~20% of steps due to overhead, net negative on 1xH100 (wash — may help at 8xH100 scale)
- Memory tokens 64 > 32 (1.3006 vs 1.3102 sliding)
- seq2048 gives significant improvement over seq1024
- 10 layers > 9 layers, fits in 16MB (~14.3MB with int8)
- Bigram + SmearGate + 3xMLP: 1.3006 → 1.2898 sliding (big gain)
- 3xMLP pushes past 16MB with int8+zlib (17.3MB), needs int6+zstd
- Sliding window eval must prepend (not overwrite) memory tokens
- Batched + compiled sliding window: ~47s vs ~8min unbatched
- Spectral init + higher LR hurt — reverted, keep defaults
- Memory tokens should be exempt from weight decay (learned scratchpad)
