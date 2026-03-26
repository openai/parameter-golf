# Podracing II: Electric Bugaloo

## Results

| Seed | Sliding BPB | 7-gram Backoff BPB | Artifact |
|------|-------------|-------------------|----------|
| 1337 | 1.1195 | 1.0217 | 15.59 MB |
| 42 | 1.1210 | **0.9631** | 15.59 MB |
| 2045 | 1.1196 | **0.9620** | 15.71 MB |
| **Mean** | **1.1200** | **0.9823** | — |

## What Changed vs Podracing I (#706)

Two eval-time improvements, no training changes:

1. **Multi-order backoff (orders 2-7)**: try longest context first, cascade down on miss
2. **Entropy-adaptive alpha**: `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))` where H = model entropy. Trust n-gram more when model is uncertain.

## Compliance

- Score-first, backward-looking: cache built from already-scored tokens only
- Alpha depends solely on model's own softmax entropy — no target/label access
- No oracle selection, no min-NLL comparison
- GPTQ calibration runs inside training phase (before wallclock stop)

## Credits

- N-gram eval cache concept: @deanbrr (PR #659)
- Multi-order backoff + adaptive alpha inspiration: @Asukabot0 (PR #727)
- Base architecture: @signalrush (PR #414)

## Reproduce

```bash
SEED=2045 MLP_ACT=leaky_relu_sq MLP_LEAKY_SLOPE=0.5 XSA_LAST_N=4 BIGRAM_VOCAB_SIZE=1536 ROPE_DIMS=24 NGRAM_EVAL_ORDER=7 NGRAM_EVAL_ADAPTIVE=1 NGRAM_EVAL_ALPHA=0.30 NGRAM_EVAL_MIN_COUNT=2 NGRAM_EVAL_BUCKETS=4194304 TTT_EVAL_ENABLED=0 torchrun --nproc_per_node=8 train_gpt.py
```

8xH100 SXM, 600s training + ~140s eval.
