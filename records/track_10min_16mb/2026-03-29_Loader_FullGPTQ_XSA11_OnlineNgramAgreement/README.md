# Loader FullGPTQ XSA11 + Online Ngram Agreement

**val_bpb: 1.11084505** (3-seed mean, std 0.00036858) | **15,995,106 bytes worst case** | 8xH100 SXM

Improves the current README leader at `1.1194` by **0.00592984 nats/byte** and **0.00855495 bpb** on the consistent 3-seed reruns.

## Results (8xH100 80GB SXM)

| Seed | step_avg | steps | Standard sliding bpb | Online-pass LLM bpb | **Online best-agree bpb** | Online gain | Eval time | Total bytes |
|------|----------:|------:|---------------------:|--------------------:|--------------------------:|------------:|----------:|------------:|
| 1337 | 91.40ms | 6456 | 1.11408566 | 1.11437756 | **1.11126660** | 0.00311096 | 461.62s | 15995106 |
| 42 | 91.59ms | 6443 | 1.11343872 | 1.11372806 | **1.11058356** | 0.00314451 | 481.04s | 15859698 |
| 2025 | 91.39ms | 6457 | 1.11352210 | 1.11381798 | **1.11068499** | 0.00313300 | 462.35s | 15884186 |
| **Mean** | **91.46ms** | **6452** | **1.11368216** | **1.11397453** | **1.11084505 (std 0.00036858)** | **0.00312949** | **468.34s** | **15995106 worst case** |

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

## Runtime

The integrated online eval stays under the 10-minute evaluation budget on 8xH100.

- 3-seed mean online eval wallclock: `468.34s` (std `11.01s`)
- 8-GPU full-val benchmark log: `online_eval_benchmark.log`
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
- `online_best_agree_eval.py`
- `online_ngram_state.c`
- `train_seed1337.log`
- `train_seed2025.log`
- `train_seed42.log`
- `online_eval_benchmark.log`

## Credits

- **Base training / quantized eval stack**: PR #1060 `Loader_FullGPTQ_XSA11_BigramHash2816`
- **This submission's main addition**: integrated online token / within-word / word-start agreement eval path, packaged so it runs inside the record folder and stays within the official evaluation budget
