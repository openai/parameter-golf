# Non-record: Causal N-gram Logit Blend — Legal, Bug-Free, and Quantitatively Shown Not to Scale

**Author:** @himanshudongre · **Date:** 2026-04-15 · **Track:** Non-record research (sp1024 + sp8192 scaling study)

## TL;DR

This PR is a **rigorous negative result**. It demonstrates that a legal, bug-free causal n-gram additive-logit contribution — the technique that every closed `ngram`-titled record PR in this repo was attempting — does not scale to strong models, and is unlikely to yield a record on top of [#1493](https://github.com/openai/parameter-golf/pull/1493) or any similarly trained SOTA stack.

Why this is useful to the community:

1. **Clean legal reference implementation.** Every previous n-gram PR was closed for a C1/C2/C3/C4 violation per Issue [#1017](https://github.com/openai/parameter-golf/issues/1017). Ours is verified against the specific closures — [#993](https://github.com/openai/parameter-golf/pull/993) (hashed caches), [#1185](https://github.com/openai/parameter-golf/pull/1185) (full-vocab renormalization), and [#959](https://github.com/openai/parameter-golf/pull/959) (two-pass rescoring) — with an automated 8-probe legality harness.
2. **Quantitative scaling curve across 6 model configurations** (2L/4L, 128d/256d, 800/2000/2500/4000 steps, sp1024/sp8192) showing the peak BPB improvement collapses from 0.0515 BPB on a very weak baseline to 0.00018 BPB on the strongest model tested. The extrapolation to real SOTA is clearly sub-threshold.
3. **Localized (bucketed) delta analysis** showing where the marginal gain actually comes from — 100% from "long-range cache hits outside the 2048-token attention window" — and why this architectural floor doesn't save the approach at scale.
4. **Reusable scaffolding.** The legality harness, integration test suite, and localized delta analysis can be applied to any future eval-time adaptation technique (SLOT variants, per-document LoRA, memory-augmented approaches, etc.).

I hope this saves other participants from running the same experiment and submitting the same bugged variants, and gives the reviewer team a clearer picture of what the legal version of this approach actually delivers.

---

## The four legality conditions (from Issue #1017) and how we satisfy each

| Condition | Our implementation | Proof |
|---|---|---|
| **C1 Strict causal** — `p_t(·)` depends only on `A` + `x_1..x_{t-1}` | Frozen count snapshot is taken at chunk start. Lookups read only the frozen snapshot. Updates to the live snapshot happen *after* all windows in a chunk are scored. | `legality_harness.py::test_c1_strict_causal` mutates future tokens and asserts lookups are bit-identical. |
| **C2 Full normalized distribution** — sum to 1 over full vocab Σ, independent of `x_t` | N-gram returns a full V-dim log-prob vector via order-K→2 backoff with add-δ smoothing. Final logits = `neural_logits + α · log_p_ngram` passed through a standard softmax. Blend is a softmax-invariant shift on "no-hit" contexts. | `legality_harness.py::test_c2_full_vocab_normalization` + `::test_c2_xt_independence`. |
| **C3 Score-before-update** — score `p_t(x_t)` first, update state second | The eval loop in `ngram_eval.py::eval_val_ttt_with_ngram` performs all scoring inside `torch.no_grad()` using the frozen snapshot; only after the chunk's scored positions are collected does `ngram.add_token()` run. Re-freeze at chunk boundary. | `legality_harness.py::test_c3_score_before_update` runs a reference cache that never sees chunk tokens and asserts the scoring lookups match. |
| **C4 Single left-to-right pass** | Evaluation is a single traversal of window_starts. No rescore, no second pass, no APIs for retrospective revision. | `legality_harness.py::test_c4_single_pass` asserts no `rescore`/`rebuild`/`two_pass` methods exist on the class. |

**Extra checks against specific reviewer closures:**

- **[#993](https://github.com/openai/parameter-golf/pull/993) "Hashed n-gram models in this way are disallowed"** → `test_no_hashing` asserts count keys are Python `tuple` objects, not integer hash buckets. We use `collections.defaultdict(Counter)` indexed by the exact context tuple.
- **[#1185](https://github.com/openai/parameter-golf/pull/1185) "calculate and renormalize over the whole vocab size"** → we return the full-V log-prob vector on every call, then add to neural logits, then apply one softmax. We never compute a blended probability only for the observed token. See `ngram_eval.py::CausalNGram._lookup_log_probs`.
- **[#959](https://github.com/openai/parameter-golf/pull/959) "two-pass rescoring methods ... leaks eval tokens"** → we do a single left-to-right traversal with frozen-snapshot-per-chunk semantics. No second pass.

**All 8 legality probes pass:**
```
$ python3 legality_harness.py --verbose
  PASS  C1 strict causal
  PASS  C2 full-vocab normalization
  PASS  C2 x_t independence
  PASS  C3 score-before-update
  PASS  C4 single pass
  PASS  no-hashing (ruling #993)
  PASS  blend non-negative + finite
  PASS  backoff fallthrough to unigram
8/8 tests passed
```

**All 4 integration tests pass on both CPU and CUDA** (A40):
```
$ python3 test_integration.py
--- regression (alpha=0) ---           PASS  (bit-identical to baseline)
--- stability (alpha>0 sweep) ---      PASS  (monotonic drop on repeating pattern)
--- legality preserved ---             PASS
--- update-after-score ordering ---    PASS
4/4 tests passed
```

The `regression (alpha=0)` test is the important one: when α=0 the blend short-circuits and BPB must be bit-identical to the unmodified `eval_val_ttt` path. This caught one class of integration bug early.

---

## The scaling curve

All experiments use a GPT-style decoder (`TinyGPT` in `code/tiny_train.py`), the additive-logit blend from `code/ngram_eval.py`, and order-4 causal n-gram with add-δ smoothing (`δ = 0.5`, `min_context_count = 2`). Training data is a prefix of the sp1024/sp8192 val shard (this is not a competition-valid training setup — it's a relative-delta measurement, and the ngram cache itself is strictly eval-time, see `code/ngram_eval.py`). Eval is on a held-out slice at the end of the same shard.

| # | Tokenizer | Model | Steps | Baseline BPT (nats/tok) | **Peak Δ BPT** | **Peak α** | **Peak Δ BPB** | vs 0.0072 record threshold |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| 1 | sp1024 | 2L 128d | 800 | 4.665 | **−0.0860** | 0.5 | **0.0515** | **7.1× above** |
| 2 | sp1024 | 2L 128d | 2500 | 4.145 | **−0.0374** | 0.3 | **0.0224** | 3.1× above |
| 3 | sp1024 | 4L 256d | 2000 | 3.811 | **−0.0132** | 0.2 | **0.0079** | 1.1× above |
| 4 | sp1024 | 4L 256d | 4000 | 3.640 | **−0.00569** | 0.15 | **0.00341** | 0.47× — below |
| 5 | **sp8192** | 4L 256d | 2000 | 5.625 | **−0.00566** | 0.10 | **0.00223** | **0.31× — below** |
| 6 | **sp8192** | 4L 256d | 4000 | 5.114 | **−0.000457** | 0.05 | **0.00018** | **0.025× — below** |

(Δ BPT values converted from `bits/tok` to `nats/tok` via `÷ log₂ e` in the table above; raw `bits/tok` numbers live in `results/results_*.json`.)

**Shrinkage per unit baseline improvement is accelerating toward zero:**

| Transition | ΔBaseline (bpt) | ΔPeak (bpt) | Shrink ratio |
|---|---:|---:|---:|
| Run 1 → 2 (2L 128d, 800 → 2500 steps) | −0.52 | +0.0486 | 9.3% |
| Run 2 → 3 (2L 128d → 4L 256d, 2000 steps) | −0.33 | +0.0242 | 7.3% |
| Run 3 → 4 (4L 256d, 2000 → 4000 steps) | −0.17 | +0.0074 | 4.3% |
| Run 5 → 6 (sp8192 4L 256d, 2000 → 4000 steps) | −0.50 | +0.00512 | 1.0% |

A strict-monotonic linear regression would put the per-unit shrink ratio near 0% in the limit, consistent with a non-zero floor — but that floor is clearly well under the 0.0072 BPB threshold, and the shrink is still happening at every measured step.

### Why the "long-range architectural floor" argument I was optimistic about doesn't save it

My overnight prediction was: "the gain comes from contexts first seen > 2048 tokens ago, which are literally invisible to the neural model's attention window, so it should persist regardless of model strength."

**The localized delta analysis** (`code/localized_delta.py`, `results/results_localized.json`) bucket-decomposes the total delta by (range × doc_position) on the 4L 256d sp1024 model:

| Range bucket × doc position | N | Δ bpt | **Δ × N** (weighted) |
|---|---:|---:|---:|
| `out_of_window × 0-2047` | 38,341 | −0.050 | **−1929** |
| `out_of_window × 2048-4095` | 7,267 | −0.076 | −554 |
| `out_of_window × 4096+` | 13,053 | −0.149 | **−1942** |
| `in_window × all` | 81 | — | −10 |
| `no_hit × 0-2047` | 99,326 | +0.008 | +787 |
| `no_hit × 2048-4095` | 17,350 | +0.004 | +70 |
| `no_hit × 4096+` | 24,581 | −0.007 | −175 |

**100% of the net benefit comes from `out_of_window`.** That's the good news for the "architectural floor" argument.

The bad news (which I didn't anticipate): **the `out_of_window` fraction shrinks with sp8192**. Sp8192 tokens are longer (mean 3.66 bytes/tok vs 2.41 for sp1024), so a 2048-token attention window covers ~52% more bytes of each document. The fraction of positions that are "physically invisible to attention" drops from ~25.4% of tokens (sp1024) to an estimated ~14% (sp8192). At the same time, stronger models also get better at the in-window positions, which doesn't change the out-of-window fraction but does reduce the *baseline uncertainty* at those positions, shrinking the n-gram's lossless-recall advantage. Both effects compound.

This is a **useful insight for other techniques that rely on "outside the attention window" as their source of headroom**: the sp8192 → sp16384 tokenizer migration that's been happening will make that class of techniques less effective, not more.

---

## What about Track B legality of this exact approach?

Issue #1017's Track B section explicitly permits "Causal n-gram caches that accumulate statistics only from already-scored tokens." That's what we built. The concern isn't legality — the concern is that the legal version gives a sub-threshold improvement.

valerio-oai's [#1185 comment](https://github.com/openai/parameter-golf/pull/1185) suggests the legal form "would be more inclined to be treated as legal." Based on the empirical scaling shown here, I believe the canonical response the community should now have is: *it's legal, it's been done cleanly, and it's ~0.0002 BPB on a properly trained model — so don't expect it to produce a record.*

If reviewers would like, I can separately ping the specific closed PRs (#993, #1026, #1185) to point to this PR as the legal reference. Happy to do that after this lands.

---

## Per-rule compliance statement

This is a **non-record** submission. It does not claim a leaderboard position. All code provided is for reproduction of the reported negative result, not for a competitive BPB score.

- Artifact size: **not applicable** (no artifact — this is a research submission)
- Training time: the compressed reproduction recipe runs in ~20 minutes on a single A40 (Phase 1-A) or ~10 minutes on Mac M4 MPS (local runs 1-4)
- Eval time: ~10 minutes for full alpha sweep
- Data: sp1024 + sp8192 shards from `willdepueoai/parameter-golf` and `kevclark/parameter-golf` respectively

No network calls, no external downloads at eval time, no runtime side information. The n-gram state is built entirely from already-scored eval tokens per Track B semantics.

---

## What's in this folder

```
2026-04-15_Causal_NGram_Null_Result/
├── README.md                            ← this file
├── submission.json                      ← metadata
├── code/
│   ├── causal_ngram.py                  ← reference CausalNGram class (module docstring documents legality invariants)
│   ├── ngram_eval.py                    ← production integration: eval_val_ttt_with_ngram
│   ├── legality_harness.py              ← 8 automated legality probes
│   ├── test_integration.py              ← 4 integration tests (α=0 regression, stability, legality preserved, update ordering)
│   ├── kill_switch_analysis.py          ← val-set repetition analysis (doc lengths, long-range hit rates per order)
│   ├── extended_analysis.py             ← bigram-proxy alpha sweep, global vs per-doc cache comparison
│   ├── tiny_train.py                    ← end-to-end train-then-eval pipeline with sweeps
│   └── localized_delta.py               ← per-bucket (range × doc_pos) delta decomposition
├── results/
│   ├── results_tiny_train.json          ← Run 1: 2L 128d sp1024 800 steps
│   ├── results_tiny_long.json           ← Run 2: 2L 128d sp1024 2500 steps
│   ├── results_tiny_bigger.json         ← Run 3: 4L 256d sp1024 2000 steps
│   ├── results_tiny_bigger_long.json    ← Run 4: 4L 256d sp1024 4000 steps
│   └── results_localized.json           ← bucket analysis (4L 256d sp1024 2000 steps)
└── training_logs/
    ├── results_a40_sp8192_phase1a.log   ← Run 5: 4L 256d sp8192 2000 steps (A40)
    ├── results_a40_sp8192_phase1b.log   ← Run 6: 4L 256d sp8192 4000 steps (A40)
    └── results_extended_analysis.log    ← bigram-proxy global vs per-doc alpha sweep (2M tokens)
```

---

## Reproduction

### Legality + integration tests (≤ 10 seconds, any CPU)
```bash
python3 code/legality_harness.py
python3 code/test_integration.py
```
Expected: `8/8 tests passed` and `4/4 tests passed`.

### Val-set repetition analysis (~3 minutes, any CPU, sp1024 val shard needed)
```bash
python3 code/kill_switch_analysis.py --val /path/to/fineweb_val_000000.bin --orders 3,4,5
```

### Tiny training + eval sweep on MPS/CUDA (~5-10 min)
```bash
python3 code/tiny_train.py \
    --val /path/to/fineweb_val_000000.bin \
    --dim 256 --layers 4 --heads 4 \
    --steps 4000 --batch 32 --seq-len 512 \
    --eval-cap 120000 --eval-chunk-tokens 16384 \
    --orders 4 --alphas 0,0.05,0.1,0.15,0.2,0.25,0.3 \
    --out results_my_run.json
```
For sp8192, add `--vocab-size 8192` and point `--val` at the sp8192 shard.

### Localized delta analysis (~5 min on MPS)
```bash
python3 code/localized_delta.py --dim 256 --layers 4 --steps 2000 --order 4 --alpha 0.2
```

---

## Acknowledgements

- valerio-oai for the definitive legality rulings on #993, #1185, #959 — without those closures I would have shipped the same buggy variant.
- @clarkkev and @bigbag for the #1394 and #1493 stacks that define the current SOTA and provided the integration target.
- @NoesisGenesis (@HKati) for Issue #1017 and the formal four-condition framework.
- @SPThole for #1602's autopsy framework — this PR follows its convention of rigorously documenting a negative result so others don't repeat the work.
