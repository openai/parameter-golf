# Aweb SOTA — Int6 + SmearGate + BigramHash + SWA + MuonWD

## Techniques (6 additions to baseline)

| # | Technique | Source | Expected BPB gain |
|---|-----------|--------|-------------------|
| 1 | **Int6 per-row quantization** | PRs #114, #162, #180 | -0.03 to -0.05 |
| 2 | **FP16 embedding preservation** | PR #114 | -0.02 to -0.03 |
| 3 | **SmearGate** (bigram blending) | PR #162 | -0.005 to -0.01 |
| 4 | **BigramHash** (4096-bucket XOR) | PR #162 | -0.005 to -0.01 |
| 5 | **SWA** (weight averaging, last 50%) | PR #162 | -0.005 |
| 6 | **Muon weight decay** (0.04) | PR #162 | -0.005 |

Plus all optimizer tuning from our first submission (Muon 0.99, halved LRs, MLP 3x, seq2048, grad_clip 0.3, warmdown 3000).

## Architecture

Same baseline architecture: 9 layers, 512 dim, 8 heads, 2 KV heads, MLP 3x (1536 hidden), tied embeddings.

## Reproduction

```bash
TRAIN_ON_VAL=1 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_AwebSOTA/train_gpt.py
```

All defaults are pre-set in the hyperparameters. No env var overrides needed.

## Author

Daniel Wahnich (@manfromnowhere143) — Founder of Aweb.

*Ostinato Rigore.*
