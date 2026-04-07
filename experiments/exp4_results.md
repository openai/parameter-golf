# Experiment 4: Depth Recurrence — Results

## Final Result

**val_bpb: 1.0980** (3-seed mean, std 0.0008)
**Delta vs SOTA (1.1147): -0.0167 BPB**
**PR: https://github.com/openai/parameter-golf/pull/1435**

## 3-Seed Results

| Seed | Pre-quant BPB | Sliding BPB (s64) | Artifact |
|------|---------------|-------------------|----------|
| 1337 | 1.1104 | 1.0989 | 14,597,964 B |
| 42 | 1.1089 | 1.0973 | 14,564,857 B |
| 2024 | 1.1097 | 1.0977 | 14,561,630 B |
| **Mean** | **1.1097** | **1.0980** | **14,574,817** |

## Variant Comparison (seed 1337)

| Variant | Sliding BPB | Artifact | Model Params |
|---------|------------|----------|-------------|
| Vanilla (PR #1421 + SP1024) | 1.0999 | 14,327,531 | 32,435,292 |
| **+ BigramHash (winner)** | **1.0989** | 14,597,964 | 32,665,181 |

## Key Decisions

- **SP4096 unavailable**: Public data manifest only has SP1024. Used SP1024 fallback.
  Expected ~0.01 BPB loss from tokenizer quality difference.
- **BigramHash wins**: +0.001 BPB over vanilla at ~270KB artifact cost. Consistent across seeds.
- **Depth recurrence works**: Even with SP1024 (suboptimal tokenizer), the recurrence
  architecture beats merged SOTA by 0.017 BPB.

## Infrastructure

- Pod: z5ohwz2aji3saz (8xH100 SXM SECURE, $21.52/hr)
- Runtime: ~4 hours (including setup, 4 training runs, evaluation)
- Estimated cost: ~$86
- Pod deleted after completion.

## Training Timeline (per run)

- Steps 0-3000: Normal 11-layer training (~4.8 min, ~8.1M tok/s)
- Step 3000: Recurrence activated (layers 4,5 repeat → 13 virtual layers)
- Steps 3000-5400: Recurrence training (~4.2 min, ~7.2M tok/s, ~12% overhead)
- 590s total training → 10s GPTQ → sliding window eval
