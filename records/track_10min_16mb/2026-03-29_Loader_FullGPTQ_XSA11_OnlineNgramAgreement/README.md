# Loader FullGPTQ XSA11 + Online Ngram Agreement

**val_bpb: 1.11085863** (4-seed mean, std 0.00030217) | **15,953,221 bytes worst case** | 8xH100 SXM

Improves the current README leader at `1.1194` by **0.00592043 nats/byte** and **0.00854137 bpb** on the bundled 4-seed subset (`42`, `1337`, `2025`, `15`).

All four bundled seed logs and the included code files correspond to the packaged submission in this folder.

## Results (8xH100 80GB SXM)

| Seed | step_avg | steps | Standard sliding bpb | Online-pass LLM bpb | **Online best-agree bpb** | Online gain | Eval time | Total bytes |
|------|----------:|------:|---------------------:|--------------------:|--------------------------:|------------:|----------:|------------:|
| 42 | 91.59ms | 6443 | 1.11343872 | 1.11372806 | **1.11058356** | 0.00314451 | 481.04s | 15817813 |
| 1337 | 91.40ms | 6456 | 1.11408566 | 1.11437756 | **1.11126660** | 0.00311096 | 461.62s | 15953221 |
| 2025 | 91.39ms | 6457 | 1.11352210 | 1.11381798 | **1.11068499** | 0.00313300 | 462.35s | 15842301 |
| 15 | 91.47ms | 6451 | 1.11372333 | 1.11402056 | **1.11089935** | 0.00312121 | 466.09s | 15841741 |
| **Mean** | **91.46ms** | **6452** | **1.11369245** | **1.11398604** | **1.11085863 (std 0.00030217)** | **0.00312742** | **467.78s** | **15953221 worst case** |

Using the bundled four-seed subset and testing against the null hypothesis that the gain over `1.1194` is at most `0.005 nats/byte`, the one-sided t-test gives **t = 8.7892**, **df = 3**, **p = 0.00155**. The mean online best-agree score is **0.76998852 nats/byte** with a 95% CI of **[0.76965525, 0.77032180]**.

## High-Level Takeaways

- The eval-time agreement techniques appear to reduce BPB reliably: all four bundled seeds improve by about `0.0031` BPB versus the matched online LLM baseline, with very little variance in the gain.
- The inference path is still not as optimized as it could be. The current implementation is already fast enough for the budget, but the runtime breakdown suggests there is still headroom in the online state, blending, and probability-extraction path.

## Summary

This submission keeps the `Loader_FullGPTQ_XSA11_BigramHash2816` training stack from PR #1060 as the base point, retunes the training schedule to use `WARMDOWN_ITERS=4000`, and adds a single-pass online n-gram agreement evaluator at the end of `train_gpt.py`.

The online evaluator combines three causal prefix-only experts:

- token n-gram top-token hints
- within-word continuation hints
- word-start first-token hints

At each scored position it chooses at most one hinted token, optionally adds a small agreement boost when multiple experts support the same token, and applies that boost to a single fully normalized distribution derived from the model's own probabilities.

## Why The Eval Is Valid

The justification is the same four conditions used for causal evaluation in this challenge.

1. **Strict causal dependence**
   The expert state at position `t` depends only on the artifact and the strict prefix. The online token and within-word state are updated only from already-scored tokens, and the word-start state is also maintained online from the prefix only.

2. **Full normalized distribution**
   The base model defines a full normalized distribution over the official vocabulary. The online path does not target-condition on the realized token. Instead it picks at most one prefix-derived hinted token and applies a logit-style boost to that token while renormalizing the whole distribution.

3. **Score-before-update**
   The score for position `t` is taken from the pre-update state. Only after the score is fixed does the evaluator update the online expert state with the current token.

4. **Single left-to-right pass**
   Evaluation is one forward pass over the validation stream in the official order. There is no rescoring pass, no retrospective revision, and no selection among multiple executions.

The implementation also keeps the metric calculation honest:

- BPB uses the sentencepiece byte-length lookup tables from `train_gpt.py`
- the full validation set is scored
- validation order is preserved
- GPTQ calibration stays in the training phase via `GPTQ_RESERVE_MS`

## Why Many Earlier N-gram Caches Were Invalid

A number of earlier n-gram-style submissions got very low BPB by exploiting the evaluation harness rather than by defining a valid causal probability model. The main failure modes were:

- **Target-conditioned lookup.** Some implementations asked the cache about the realized next token itself, or used the realized token to decide whether a cache hit existed. That makes the reported `P(x_t | x_{<t})` depend on `x_t`, which breaks causality.
- **Scalar-only blending.** Some implementations blended only the probability of the correct token and never defined a full probability distribution over the vocabulary. That can produce an impressive BPB number without defining a decoder-usable model.
- **Bucket scores treated as token probabilities**, without normalizing over the full vocabulary. Some implementations treated hash-bucket counts or bucket-local matches as if they were already normalized token probabilities. Even when the bucket logic was useful as a hint, the reported score was not a proper full-vocabulary distribution unless it was renormalized over actual tokens.

This evaluator is designed to avoid those failure modes.

At position `t`, the online state uses only the strict prefix to propose at most one hinted token `h_t` together with a prefix-derived confidence. The hinted token is chosen before `x_t` is consulted, and the online state is updated only after the score for `x_t` has already been fixed.

The final score is not taken from a standalone n-gram probability. Instead, it starts from the base model's full softmax distribution `p_t(a)` and applies a one-token logit boost to the hinted token:

`p'_t(a) = exp(beta_t * 1[a = h_t]) p_t(a) / Z_t`

with

`Z_t = 1 - p_t(h_t) + exp(beta_t) p_t(h_t)`.

So the evaluator always corresponds to a single normalized distribution over the real token vocabulary. The realized token `x_t` is used only at the end to read off `p'_t(x_t)` from that already-defined distribution. Hashing in the online state can affect which hint is proposed, but it does not create extra probability mass or bypass renormalization.

In other words, the improvement here, if it is real, comes from a causal prefix-only hint layered on top of a normalized model distribution, not from looking up the gold token or from scoring an unnormalized cache value directly.

## Runtime

The integrated online eval stays under the 10-minute evaluation budget on 8xH100.

- 4-seed mean online eval wallclock: `467.78s` (std `9.06s`)
- benchmark result: `1.11265002 -> 1.10955484 bpb` in `462.67s`

The measured bottlenecks in the benchmark were the online overlay itself rather than the neural forward pass:

- online state maintenance
- chunk blending / agreement logic
- model forward plus targeted probability extraction

## Eval-Time Improvements Tried

Before settling on the final path, I tried and discarded several slower or less defensible variants:

- cache-heavy offline / shared-cache evaluation flows
- exact phrase cache variants that were not the right final legality story for a per-seed online submission
- a Python-only online prototype before moving the hot n-gram state into a native helper
- an earlier multi-GPU design that communicated too much per-token state

The final version uses a local-only distributed design, a native open-addressing online n-gram table in `online_ngram_state.c`, and targeted `logsumexp` / gather extraction rather than a full-vocab `log_softmax` pass for every scored token.

## Run Command

```bash
SEED=1337 \
BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112 XSA_LAST_N=11 \
USE_GPTQ=1 TTT_ENABLED=0 ONLINE_BEST_AGREE_EVAL=1 EVAL_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=600 GPTQ_RESERVE_MS=10000 \
WARMDOWN_ITERS=4000 TIED_EMBED_LR=0.035 ITERATIONS=6700 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py`
- `requirements.txt`
- `online_best_agree_eval.py`
- `online_ngram_state.c`
- `train_seed15.log`
- `train_seed1337.log`
- `train_seed2025.log`
- `train_seed42.log`

## Credits

- **Base training / quantized eval stack**: PR #1060 `Loader_FullGPTQ_XSA11_BigramHash2816`
- **This submission's main addition**: integrated online token / within-word / word-start agreement eval path, packaged so it runs inside the record folder and stays within the official evaluation budget
