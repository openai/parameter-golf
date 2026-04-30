# Engineering Log

This is a technical log for reviewers. It explains how this candidate was selected, which alternatives were rejected, and why the final method is a targeted follow-up rather than a broad validation-tuned sweep.

## Starting Point

The starting point was the submitted PR #1915 record:

- SP8192 CaseOps tokenizer/data path.
- PR #1855-style legal frontier architecture and compression stack.
- Stock top-k LQER.
- Legal per-document score-first LoRA TTT.
- 3-seed mean: 1.06504520 BPB.
- Max package size: 15,922,155 bytes.

That submission stayed frozen. Follow-up experiments were isolated so the clean submitted record could not be contaminated by exploratory changes.

## Methodology

The follow-up work used four rules:

1. Keep legality first: official SP8192 distribution, strict causality, score-before-update, no byte-PPM shortcut, no custom tokenizer, no cross-document validation adaptation.
2. Treat package bytes and runtime as part of the result, not post-processing.
3. Prefer paired or same-execution comparisons whenever the expected effect is small.
4. Close weak mechanisms quickly, but keep evidence for each decision.

The same-execution rule became important after a context-mixer experiment showed that separate eval-time TTT runs can differ at the 1e-5 BPB level even when the mathematical prediction path is intended to be identical. Fresh per-document LoRA state, BF16/fused kernels, and distributed scheduling can create slightly different trajectories. For scoring-only transforms, later comparisons therefore shared the same logits, TTT trajectory, document order, hints, tokens, and bytes.

## Closed Candidate Mechanisms

Several plausible mechanisms were tested before selecting this candidate:

| Mechanism | Result | Decision |
| --- | --- | --- |
| Weighted LQER | best gain about 0.000013 BPB | closed |
| AWQ/no-embedding rescue | package-safe but neutral | closed |
| D-prime bucketing | slightly better score but too slow | closed |
| First-order TTT-aware training | seed42 worsened to 1.07164685 | closed |
| Random-map adapters | sampled BPB worsened by 0.008918 | closed |
| Long-context 4K | full validation worsened by 0.002438 BPB | closed |
| LeakyReLU-slope retrain | seed42 worsened to 1.13888025 | closed |
| Neural Dirichlet context mixer | same-execution identity check passed, but valid slices regressed and full runtime projected about 9180s | closed |

One small mechanism stayed positive: lower eval-time TTT LR. It improved seed42, seed0, and seed1234 by about 0.0005 to 0.0006 BPB and became the base setting for the final follow-up runs.

## Token-Level Causal N-Gram Tilt

The main late signal came from a normalized token-level causal n-gram tilt over the official SP8192 alphabet.

For a strict-prefix hint `h`, the fixed tilt changes the model distribution by

```text
p'(a) = exp(beta * 1[a == h]) * p(a) / Z
Z = 1 + p(h) * (exp(beta) - 1)
```

This is a normalized SP8192 distribution. The hint state is updated only after the current token is scored. This is different from byte-PPM or a custom tokenizer path because the scored alphabet remains the official token alphabet.

The first full seed42 validation of this mechanism produced:

- paired lower-LR control: 1.06373091 BPB
- fixed token n-gram tilt: 1.06182936 BPB
- gain: 0.00190155 BPB
- exact counts: 47,851,520 scored tokens / 151,074,499 scored bytes

That was the first follow-up effect large enough to justify packaging and runtime work.

## Adaptive Hedge

Fixed n-gram tilt has one obvious risk: a single boost strength can overpay the normalizer on some hints and underboost others. Adaptive Hedge was tested as a same-execution mixture over boost temperatures. Mechanically, it acts like a small universal code over n-gram boost strengths.

Across full validation, Adaptive Hedge improved fixed n-gram by almost the same amount in every setting tested:

| Setting | Control | Fixed n-gram | Adaptive Hedge | Hedge vs Fixed |
| --- | ---: | ---: | ---: | ---: |
| seed42 | 1.06372791 | 1.06182636 | 1.06143959 | 0.00038677 |
| seed0 | 1.06452124 | 1.06260772 | 1.06221801 | 0.00038971 |
| seed1234 | 1.06526616 | 1.06331328 | 1.06292962 | 0.00038366 |
| no-Q seed42 | 1.06373272 | 1.06182914 | 1.06144170 | 0.00038744 |

Three-seed means from the same-execution counter batch:

- control: 1.06450510 BPB
- fixed n-gram: 1.06258245 BPB
- Adaptive Hedge: 1.06219574 BPB

The consistency was the important evidence. Adaptive Hedge was not a seed42-only fluctuation; it repeatedly saved about 0.000386 BPB beyond fixed n-gram.

## Transferable Mechanism: Adaptive-Beta Hedge

The most reusable finding from this follow-up is not just the final absolute score. It is that Adaptive-Beta Hedge behaved like a base-independent scoring overlay.

Across four different seed42 base configurations, Hedge added nearly the same gain over fixed n-gram:

| Base configuration | Hedge gain vs fixed n-gram |
| --- | ---: |
| default context, Q/V LoRA active | 0.00038677 BPB |
| default context, Q LoRA disabled | 0.00038744 BPB |
| public-frontier diagnostic base, no prefix adaptation | 0.00038690 BPB |
| 2560 context, Q/V LoRA disabled | 0.00038599 BPB |
| spread | 0.00000145 BPB |

These configurations differ across eval context length, LoRA branch selection, and underlying trained artifact. The near-identical Hedge delta suggests it is correcting a systematic n-gram boost calibration error rather than exploiting one model trajectory.

Mechanistically, fixed n-gram tilt pays a normalizer cost when it boosts a hinted token. A single fixed boost can overpay on weak hints and underboost on strong hints. Adaptive Hedge acts as a small universal code over boost temperatures, selecting a better effective strength online while preserving the same strict-prefix, normalized SP8192 scoring rule.

This is interesting because most improvements in this contest are base-coupled: change the architecture, quantization, context, or TTT target modules and the gain can disappear. Hedge remained stable across several such axes. The honest limit is that the cross-base transfer table is seed42-only for each base; the three-seed evidence exists for the submitted configuration family, not for every base configuration in the table.

## Trajectory Interactions

Scoring transforms and TTT trajectory changes were then separated. The scoring transform is post-hoc over the logits. The trajectory changes alter the logits themselves through eval-time context or LoRA target selection.

Trajectory-changing candidates tested:

| Candidate | BPB | Read |
| --- | ---: | --- |
| tilt-aware TTT objective | 1.06154791 | weak positive |
| 2560 context + no Q/V LoRA, fixed n-gram | 1.06121730 | strong interaction |
| tilted training objective | 1.27728534 | undertrained, not useful under deadline |
| public-frontier diagnostic + Adaptive Hedge | 1.06096543 | useful diagnostic, not best |

The strongest local interaction was 2560 context + no Q/V LoRA combined with Adaptive Hedge:

- paired trajectory control: 1.06306045 BPB
- fixed n-gram: 1.06121689 BPB
- Adaptive Hedge: 1.06083091 BPB
- Adaptive Hedge gain over fixed n-gram: 0.00038599 BPB

This showed that the 2560/no-QV trajectory gain and the Adaptive Hedge scoring gain were mostly complementary.

## Final Runtime And Package Work

The final candidate had to be made self-contained and fast enough.

Package state:

- compressed model: 15,872,234 bytes
- counted `train_gpt.py` wrapper: 57,161 bytes
- total counted bytes: 15,929,395 bytes
- margin under 16,000,000 bytes: 70,605 bytes
- n-gram Python helper and C helper source embedded in counted wrapper
- no uncounted helper files required

Evaluation data safety:

- validation-only data view
- train shards visible during final eval: 0
- validation token shards: 5
- validation byte shards: 5

Runtime tuning was limited to equivalent implementation changes. No scoring constants were retuned. Batch-size checks showed batch 32 was best; batch 48 fit in memory but slowed down from memory pressure/load imbalance.

The final speedup removed a duplicate vocabulary-wide normalization. The previous path computed token CE and then separately computed a full log-softmax to get the n-gram hint probability. The optimized path reuses:

```text
loss = logZ - target_logit
logZ = loss + target_logit
hint_log_prob = hint_logit - logZ
```

This preserves the scoring math while avoiding a second normalization.

Selected proof:

- BPB: 1.06082922
- raw loss sum: 111086708.60261762
- raw bits sum: 160264243.60967380
- inner TTT eval: 544.1s
- total eval wallclock: 566.3s
- wrapper wallclock: 585s
- scored tokens: 47,851,520
- scored bytes: 151,074,499
- doc-order hash: 33236cc6bd19fa6b89e06d441d3fcd8eb37dc8540f6a4f2b627b20af10894a41

## Open Follow-Up Evidence

Seed0 and seed1234 optimized package runs were launched in parallel after the seed42 proof. They are not required for the seed42 package proof, but they should be appended if available before review.

This submission should be read as a focused, legally constrained follow-up: the previously submitted record stack remains frozen, the selected mechanism is measured with exact official accounting, and the final score improvement comes from a targeted eval-time trajectory plus a normalized causal token-level scoring overlay.
