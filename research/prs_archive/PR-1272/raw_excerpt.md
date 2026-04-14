# PR 1272 — Comprehensive Negative Results on Strong Models

**Author:** (same author as PR 1271 — negative results PR series, updates PR #1186)
**Claimed BPB:** N/A (negative results compilation). Techniques tested on ~1.11 BPB baseline.
**Seeds:** not stated
**Hardware:** not stated

## Files retrieved
- `records__track_10min_16mb__2026-04-02_Negative_Results_Comprehensive__README.md`
- `records__track_10min_16mb__2026-04-02_Negative_Results_Comprehensive__ngram_test.py`
- `records__track_10min_16mb__2026-04-02_Negative_Results_Comprehensive__online_logit_bias.py`
- `records__track_10min_16mb__2026-04-02_Negative_Results_Comprehensive__retokenize_proper.py`

Note: `correct_meta.npz` is binary and was not extracted to the archive.

## Claimed changes (from README, verbatim)

Collection of ~30 failed experiments on well-trained GPTQ'd models. Updates PR #1186.

Eval-time techniques that don't work on strong models (baseline ~1.11 BPB):

| Technique | BPP delta | Why |
|---|---|---|
| Properly normalized n-gram (Kneser-Ney, exact trie) | +0.001 to -0.003 | Model 100x better than n-gram |
| Online Logit Bias (per-token SGD on logit bias) | +0.003 (hurts) | GPTQ'd model already well-calibrated; 1229s over budget |
| Prime MLP Adapters (zero-init rank-64, PR #1222 approach) | -0.00009 | No room at 1.11 baseline; sliding context already provides it |
| Complementary Training (down-weight n-gram-predictable tokens) | -0.0004 (noise) | |
| Score-first chunked TTT (PR #549 approach) | -0.003 | Works but tiny gain on GPTQ'd models |

N-gram normalization proof: Kneser-Ney smoothing with exact trie counts, order 7, full normalized distribution. Max normalization error 1.78e-15. N-gram avg NLL 5.40 vs model avg NLL 0.79 (n-gram 6.8x worse). Mixing at any alpha hurts on average. "The entire 0.09-0.97 BPP improvement from hashed n-gram caches was a measurement artifact from unnormalized distributions."

SLOT violates causal dependence (cross-refs PR #1240): 100% violation rate across 240 tested pairs.

Scylla tokenizer (cross-refs PR #1271): Corrected accounting -> 1.1289 BPB, same as SP1024.

"What actually matters":
1. Training data volume (194+ shards > 80)
2. Full Hessian GPTQ (Cholesky + actorder, ~0.005 BPP over naive int6)
3. Coprime-stride data loader (batch diversity)
4. XSA on all layers (small but consistent gain with coprime loader)
