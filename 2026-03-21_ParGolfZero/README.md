# ParGolf-Zero — 5-Layer Joint Optimization System

## Unique Contribution
Unlike other submissions that train normally and compress post-hoc,
ParGolf-Zero treats the 16MB budget as a first-class training objective.
A per-row weight range penalty directly minimizes int8 quantization error
during every gradient step. QAT activates in the final 500 steps.

## Layers
- Layer 1 COMPAT: Auto-detects GPU platform, runs anywhere
- Layer 2 TRAIN: FP16 embeddings + Muon WD + warmdown=20000
- Layer 3 COMPRESS: Weight range penalty + QAT final steps
- Layer 4 EVAL: Sliding window stride=64, 960 token context
- Layer 5 ADAPT: Low-rank Q/V regularization for TTT readiness

## Status
Non-record submission. Pipeline confirmed on Kaggle T4.
Awaiting H100 compute grant for final scored run.

## Author
Sanjith G — sanjith3057
