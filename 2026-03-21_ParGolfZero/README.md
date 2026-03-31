# ParGolf-Zero v2 — 6-Layer Joint Optimization

## Status
Non-record submission. Awaiting H100 compute grant for final scored run.
Pipeline fully confirmed on Kaggle T4 — artifact 5.52MB under 16MB limit.

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

## Confirmed Results (T4 smoke test)
- Artifact size: 5.52MB ✅
- Pipeline: train → quantize → zstd → roundtrip ✅
- val_bpb (200 steps, tiny batch): 3.19 (not real score)

## Author
Sanjith G — github.com/sanjith3057
