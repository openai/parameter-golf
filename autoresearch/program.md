# autoresearch — parameter-golf edition

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for the OpenAI Parameter Golf competition.

## Competition Constraints

- **Artifact size**: code + compressed model ≤ 16,000,000 bytes (16MB decimal)
- **Training time**: ≤ 10 minutes on 8xH100 SXM GPUs (official), but we iterate locally on a single RTX 3070 (8GB VRAM)
- **Metric**: val_bpb (bits per byte) on FineWeb validation set — lower is better
- **New records must beat SOTA by ≥ 0.005 nats** with p < 0.01

## Local Iteration Setup

- **Hardware**: RTX 3070 8GB VRAM
- **Iteration cap**: ~500 training steps per experiment (~2-5 minutes)
- **Evaluation**: Use a small validation subset for the loop; full eval only on promising candidates
- **File to modify**: `train_gpt.py` (in the repo root)
- **Do NOT modify**: data loading, evaluation harness, tokenizer

## What's Known to Work (the "meta stack")

These techniques are proven on the leaderboard — use them as the baseline:
- 11 layers / 512d / 3x MLP expansion (hidden=1536)
- Int6 QAT with straight-through estimator
- SmearGate + BigramHash embeddings (10240 buckets)
- EMA weight averaging (decay=0.997)
- Muon optimizer (momentum=0.99, weight decay=0.04)
- LeakyReLU(0.5)^2 activation
- XSA (cross-sequence attention) on last 3-4 layers
- Partial RoPE (16/64 dims)
- Sliding window eval (stride=64)
- U-Net skip connections
- zstd-22 compression
- Orthogonal weight initialization

## What's Known NOT to Work

Avoid these — they've been tested and fail:
- SwiGLU (better per-step but 45% slower, net negative)
- Depth recurrence / looped transformers (+0.025 BPB worse)
- MoE at this scale (0.06-0.08 BPB worse)
- LZMA for int8 data (worse than zlib)
- Factored embeddings with small vocab (1024)
- Full-weight TTT with AdamW (massive overfitting)

## Exploration Directions

Try things not yet on the leaderboard:
- Novel activation functions beyond LeakyReLU^2
- Attention pattern variations (sparse, linear, local+global hybrid)
- Better weight initialization schemes
- Training schedule innovations (cyclic LR, progressive resizing)
- Loss function modifications (label smoothing, focal loss)
- Normalization alternatives (LayerNorm variants, no-norm)
- Embedding compression techniques
- Novel quantization-aware training approaches
- Data ordering / curriculum learning strategies

## VRAM Constraint

The GPU has only 8GB VRAM. If a configuration OOMs, reduce batch size or model width. Don't waste time on configs that obviously won't fit.

## The Experiment Loop

Same as the original autoresearch: modify train_gpt.py → commit → run ~500 iterations → check val_bpb → keep or revert. Never stop. Be autonomous.

## Output

After each run, extract:
```
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

Log results to `results.tsv` (tab-separated):
```
commit	val_bpb	memory_gb	status	description
```
