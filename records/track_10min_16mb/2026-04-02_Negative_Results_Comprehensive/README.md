## What this is

A collection of things that don't work on well-trained GPTQ'd models.
My coding agents and I ran ~30 experiments over the past two weeks trying
to push below the 1.11 BPP frontier. Most of them failed. This documents
the failures so others don't repeat them.

Updates and extends my earlier negative results PR (#1186).

## Eval-time techniques that don't work on strong models

These all work on weak/undertrained models but provide zero benefit once
your base model is well-trained with Full Hessian GPTQ + sliding window
eval:

| Technique | BPP delta | Why it fails |
|-----------|:---------:|:-------------|
| Properly normalized n-gram (Kneser-Ney, exact trie) | +0.001 to -0.003 | Model is 100x better than n-gram at predicting the correct token. Mixing at any alpha dilutes model confidence. Confirms PR #511 (-0.001) and PR #1145 (-0.003). |
| Online Logit Bias (per-token SGD on logit bias vector) | +0.003 (hurts) | GPTQ'd model is already well-calibrated. No systematic bias to correct. Also takes 1229s (way over eval budget). |
| Prime MLP Adapters (zero-init rank-64, PR #1222 approach) | -0.00009 | PR #1222 got -0.073 but on a 1.50 BPP baseline. Our 1.11 baseline leaves no room — sliding window context already provides everything adapters would learn. |
| Complementary Training (down-weight n-gram-predictable tokens during training) | -0.0004 (noise) | Doesn't change model behavior enough. By the time the model converges, it already knows everything the bigram knows. |
| Score-first chunked TTT (PR #549 approach) | -0.003 | Works but the gain is tiny on GPTQ'd models. PR #1184 also found TTT "neutral" on their stack. |

## The n-gram normalization proof

I built the best possible legal n-gram cache: Kneser-Ney smoothing with
exact trie counts (zero hashing, zero collisions), order 7, full
normalized distribution over all 1024 tokens at every position.

Results on 500K positions:
- Max normalization error: **1.78e-15** (distributions are perfect)
- Zero normalization violations across all positions
- N-gram avg NLL: **5.40** vs model avg NLL: **0.79** (n-gram is 6.8x worse)
- Mixing at ANY alpha hurts on average

The entire 0.09-0.97 BPP improvement from hashed n-gram caches was a
measurement artifact from unnormalized distributions. The real signal from
properly normalized n-grams is 0.001-0.003 BPP, so it's not worth the complexity.

## SLOT violates causal dependence

Detailed in my PR #1240. 100% violation rate across 240 tested pairs.
Self-prediction advantage: +0.24 nats (shared delta), +0.73 nats
(per-sample). Every SLOT-based result on the leaderboard is suspect.

## Scylla tokenizer doesn't help (with correct accounting)

Covered in my other PR. With corrected byte accounting, Scylla gets 1.1289
BPP, the same as SP1024 at 1.1157. The entire sub-1.0 claim was a byte
accounting bug in `candidate.meta.npz`.

## What actually matters

After all these experiments, the model quality is dominated by:
1. **Training data volume** (194+ shards > 80 shards)
2. **Full Hessian GPTQ** (Cholesky + actorder, ~0.005 BPP over naive int6)
3. **Coprime-stride data loader** (batch diversity)
4. **XSA on all layers** (small but consistent gain with coprime loader)

## Files included

- `ngram_test.py` — Kneser-Ney trie with full normalization proof
- `online_logit_bias.py` — Online logit bias implementation + synthetic test
- `correct_meta.npz` — Corrected Scylla byte accounting
- `retokenize_proper.py` — Proper retokenization with official train/val split
