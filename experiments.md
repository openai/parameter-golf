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
| 11 | 03-21 | Run 10 + int5/int6+zstd + SWA(3) + WD=0.04 + warmdown=300 | 1207 | 1.2912 | 1.2748 | 15.5 | Under 16MB! Pre-EMA code |
| 12 | 03-21 | Run 11 + partial RoPE + LN scale (EMA crashed) | 979 | 1.3118 | — | 15.1 | EMA broke quant (+0.12 BPB), disabled |
| 13a | 03-21 | Full stack, Mem(64), no EMA/QAT | 1194 | 1.2950 | 1.2787 | 15.1 | A/B test: WITH memory tokens |
| 13b | 03-21 | Full stack, Mem(0), no EMA/QAT | 1051 | 1.3093 | 1.2928 | 15.0 | A/B test: WITHOUT memory tokens |

## 8xH100 SXM Runs (submission quality)

| # | Date | Config | Steps | val_bpb (std) | val_bpb (slide) | Size (MB) | Notes |
|---|------|--------|-------|---------------|-----------------|-----------|-------|
| 1 | 03-21 | Mem(64) + MTP(2) + 10L + bigram + smear + 3xMLP + WD + int6+zstd | 9403 | 1.1832 | 1.1670 | 17.3* | *Over 16MB w/ int6, need int5 MLP |
| 2 | 03-21 | Full stack + int5 MLP + EMA + partial RoPE + LN scale + late QAT | 4440 | 1.1895 | 1.1735 | 15.0 | Under 16MB! EMA+batch=786K cut steps in half |

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
- A/B confirmed: Mem(64) gives -0.014 BPB vs no memory tokens (1.2787 vs 1.2928 sliding)
- Memory tokens also speed up step time (502ms vs 571ms) — fewer real tokens to process
- EMA destroys quantization on 1xH100 (~1000 steps) — only use on 8xH100 (9000+ steps)
- Partial RoPE (16/64 dims) and LN Scale added, untested in isolation
