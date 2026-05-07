# ParGolf-Zero v2 — 6-Layer Joint Optimization

## Status
Non-record submission. Awaiting H100 compute grant for final scored run.
Pipeline confirmed on Kaggle T4 — artifact 5.52MB under 16MB limit.

## Unique Contribution
Joint compression-aware training. Per-row weight range penalty minimizes
int8 quantization error during every gradient step. Nobody else is doing this.

## Layers
- L1 COMPAT: Auto-detects GPU platform
- L2 TRAIN: FP16 embed + Muon WD + warmdown + SWA
- L3 COMPRESS: Weight range penalty + QAT + zstd-22
- L4 EVAL: Sliding window stride=64, 960 token context
- L5 ADAPT: Low-rank Q/V regularization for TTT
- L6 BIGRAM: BigramHash(10240) learned bigram table

## Results (T4 smoke test, 200 steps)
- Artifact: 5.52MB ✅
- val_bpb: 3.19 (smoke test only, not real score)
- roundtrip val_bpb: 3.23 ✅

## Author
Sanjith G — github.com/sanjith3057
