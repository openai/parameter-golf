## Parallel SwiGLU Transformer + Legal Hierarchical Side Channel

Base: `20260326-trans-parallel/train_gpt_transformer_parallel_swiglu.py`

This variant keeps the proven 10-layer transformer shell, the Griffin-style
parallel block schedule, BigramHash, SmearGate, and skip connections. The
change is at the encoder/decoder boundary:

- after the encoder half, extract one interval summary every `MACRO_INTERVAL`
  tokens using the last token of each interval
- build a legal causal student summary stream by shifting those interval
  summaries right by one interval and letting a small predictor refine them
- let every token attend causally to all prior student summaries through a
  bottleneck cross-attention side channel
- add a small self-distillation loss so the student summaries learn to
  anticipate the next interval summary

Why this is more "ours":
- it imports the strongest 2026-03-26 micro/macro lesson into the transformer
  line without paying for a full extra macro block
- it keeps the side channel legal: tokens only receive shifted student
  summaries, never current-interval teacher summaries
- it is architecturally distinct from the leaderboard shell while remaining
  cheap enough to queue immediately

What success looks like:
- no regression in stability versus the dense parallel SwiGLU transformer
- measurable BPB lift from the legal side channel
- if the dense version wins, D2S can be reconsidered later as a compression
  pass rather than as the primary idea
