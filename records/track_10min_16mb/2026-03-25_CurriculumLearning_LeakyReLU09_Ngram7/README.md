# Record: Curriculum Learning + LeakyReLU(0.9)^2 + 7-gram Backoff (val_bpb=0.9633)

**val_bpb = 0.9633** (seed 42, additional seeds pending compute grant) | **15.56 MB** | 8xH100 SXM, 600s

## Approach

Built on PR #753 (Podracing II) with two additions:

### 1. Curriculum Learning (Shard Reordering)

Training shards reordered by model perplexity — hardest shards first. Based on PR #650 by @abaybektursun which demonstrated -0.003 BPB from shard ordering alone. Zero code change, just environment variable.

### 2. LeakyReLU(0.9)^2 Slope Optimization

Following @MatoTeziTanka's controlled slope sweep on issue #140, replaced standard slope=0.5 with slope=0.9. The sweep showed monotonic improvement from 0.1 to 0.9, with 0.9 giving -0.013 BPB vs 0.5 on the same stack.

## Results

| Metric | Value |
|--------|-------|
| Sliding window (stride=64) | 1.1216 |
| **Sliding + 7-gram backoff** | **0.9633** |
| Legal TTT (score-first, 3ep) | 1.1216 |
| Artifact | 15,560,351 bytes |
| Steps | 6,647 at 90.3ms/step |
| Training time | 600s |

## Architecture (from PR #753)

- 11L, 512d, GQA 8/4, MLP 3x
- LeakyReLU(0.9)^2 activation
- XSA on all 11 layers
- BigramHash, SmearGate, SWA, EMA
- Int6 QAT + GPTQ (within training budget, issue #677 compliant)
- 7-gram backoff eval cache (backward-looking, no weight updates)

## Eval-time Techniques

**7-gram backoff cache** (from PR #753): Multi-order n-gram model built from already-scored tokens. Linear interpolation with entropy-adaptive alpha. Fully backward-looking — each token scored before its statistics enter the cache.

**Legal score-first TTT** (from PR #753): SGD with 3 epochs, freeze last 2 blocks. Every token scored under inference_mode before any weight update.

## Reproduction

```bash
SEED=42 bash run.sh
```

Environment variables set in run.sh:
- `SHARD_ORDER=44,63,65,42,...` (curriculum learning)
- `MLP_LEAKY_SLOPE=0.9`
- `NGRAM_EVAL_ORDER=7`

## Acknowledgments

- @newjordan (PR #753, Podracing II base)
- @abaybektursun (PR #650, curriculum learning / shard reordering)
- @MatoTeziTanka (LeakyReLU slope sweep, issue #140)
- @Asukabot0 (PR #715/#727, n-gram backoff technique)

## Status

1 seed submitted. 2 additional seeds pending OpenAI compute grant ($1000 applied).
Previously PR #486 (formerly #2 on leaderboard, TrigramHash originator). $339 personal compute spent.
