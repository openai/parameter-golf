# Engineering Log

This is a concise technical record of how this candidate was selected. It is included so reviewers can see what was tested, what was rejected, and why this final candidate is isolated from PR #1915.

## Anchor

The starting point was the clean Path A submission in PR #1915:

- SP8192 CaseOps tokenizer/data path.
- PR #1855-style legal frontier architecture and compression stack.
- Stock top-k LQER.
- Legal per-document score-first LoRA TTT.
- 3-seed mean: 1.06504520 BPB.
- Max package size: 15,922,155 bytes.

PR #1915 stayed frozen. All follow-up work happened in separate lanes.

## Follow-Up Results That Were Closed

Several candidate mechanisms were tested and rejected before this path was selected:

| Mechanism | Result | Decision |
| --- | --- | --- |
| Weighted LQER | best gain about 0.000013 BPB | closed |
| AWQ/no-embedding rescue | package-safe but neutral | closed |
| D-prime bucketing | slightly better score, too slow | closed |
| FO-MetaTTT | seed42 worsened to 1.07164685 | closed |
| Random-map adapters | sampled BPB worsened by 0.008918 | closed |
| Long-context 4K | full validation worsened by 0.002438 BPB | closed |
| Slope-0.3 retrain | seed42 worsened to 1.13888025 | closed |
| NDCM/HedgeBiasCache | repaired equivalence, then valid slices regressed and runtime projected about 9180s | closed |

Lower eval-time TTT LR remained positive across seeds, with a projected mean around 1.06449816 BPB. It became the base setting for final follow-up tests.

## Token N-Gram Tilt Signal

A normalized token-level causal n-gram tilt was then tested on top of frozen Path A lower-LR TTT. The key legality constraint was to stay on the official SP8192 alphabet and update only from strict-prefix state.

Full seed42 informational validation produced:

- paired lower-LR control: 1.06373091 BPB
- fixed token n-gram tilt: 1.06182936 BPB
- gain: 0.00190155 BPB
- exact counts: 47,851,520 scored tokens / 151,074,499 scored bytes

That was the first Path A+ follow-up with a large enough signal to justify final-day packaging work.

## Same-Execution Counter Lesson

NDCM showed that separate eval-time TTT runs are not a reliable way to compare small BPB deltas. Even mathematically identical E0/control runs can differ at the 1e-5 BPB level due to fresh LoRA state and distributed numerical trajectory differences.

For the final n-gram slate, scoring-only variants were measured as same-execution counters sharing logits, TTT trajectory, documents, hints, and byte accounting.

## Final Factorized Slate

Two warm 8xH100 SXM pods were used. Data and artifacts were uploaded once and reused.

Queue A tested seed42, seed0, seed1234, and no-Q-only with same-execution n-gram counters. It showed that Adaptive Hedge was stable across seeds:

| Lane | Control | Fixed n-gram | Adaptive Hedge | Hedge vs Fixed |
| --- | ---: | ---: | ---: | ---: |
| seed42 | 1.06372791 | 1.06182636 | 1.06143959 | 0.00038677 |
| seed0 | 1.06452124 | 1.06260772 | 1.06221801 | 0.00038971 |
| seed1234 | 1.06526616 | 1.06331328 | 1.06292962 | 0.00038366 |
| no-Q seed42 | 1.06373272 | 1.06182914 | 1.06144170 | 0.00038744 |

Three-seed means from Queue A:

- control: 1.06450510 BPB
- fixed n-gram: 1.06258245 BPB
- Adaptive Hedge: 1.06219574 BPB

Conclusion: fixed n-gram generalized, and Adaptive Hedge was not a seed42-only artifact. No-Q-only did not add value.

Queue B tested trajectory-changing interactions:

| Lane | BPB | Read |
| --- | ---: | --- |
| tilt-aware TTT | 1.06154791 | weak positive |
| 2560 context + no-Q/V fixed n-gram | 1.06121730 | strong interaction |
| tilted retrain | 1.27728534 | undertrained, not useful |
| public-frontier adaptive Hedge | 1.06096543 | useful diagnostic, not best |

The best interaction was B2 plus Adaptive Hedge:

- paired B2 control: 1.06306045 BPB
- B2 fixed n-gram: 1.06121689 BPB
- B2 Adaptive Hedge: 1.06083091 BPB
- Adaptive Hedge gain over B2 fixed: 0.00038599 BPB

This candidate packages that B2 + Adaptive Hedge path.

## Current Package State

The package is self-contained and under the byte cap:

- model: 15,872,234 bytes
- counted code wrapper: 57,102 bytes
- total: 15,929,336 bytes
- margin: 70,664 bytes

Score reproduction is stable:

- target exploratory score: 1.06083091 BPB
- batch-32 package proof: 1.06083116 BPB
- batch-48 package proof: 1.06083288 BPB

Runtime is the remaining issue:

- batch 32 inner eval: 588.5s
- batch 32 wrapper wallclock: 631s

This draft should become a record-track claim only if the optimized proof clears the official runtime requirement. Otherwise it should be treated as a runtime-caveated follow-up.
