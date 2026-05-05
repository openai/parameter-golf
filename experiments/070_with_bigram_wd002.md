# Experiment 070: Full config WITH BigramHash + WD=0.02

## Status: COMPLETED

## Results
| Metric | Value |
|--------|-------|
| Steps | 7,327 @ 82.0ms/step |
| Model params | 22,368,841 |
| Standard eval | **1.1669 BPB** |
| **Sliding eval** | **1.1458 BPB** ← ALL-TIME BEST |
| FLAT+zstd | **17,383,391 bytes ❌ (1.38MB over)** |

## BigramHash helps BPP but kills artifact size
| | 069 (no bigram) | 070 (with bigram) |
|--|----------------|-------------------|
| Sliding | 1.1495 | **1.1458 (-0.004)** |
| FLAT+zstd | 15.34MB ✅ | 17.38MB ❌ |

BigramHash weights (590K params, mostly fp16/int8) don't compress as well as MLP int6 weights.
Compression benchmark overestimated savings because it tested on a DIFFERENT checkpoint (068, no bigram).

## Conclusion
BigramHash is worth +0.004 BPB but costs 2MB in artifact.
Can't fit under 16MB on this platform. Need to test on Runpod or find better compression.
069 remains best submission-ready config.
