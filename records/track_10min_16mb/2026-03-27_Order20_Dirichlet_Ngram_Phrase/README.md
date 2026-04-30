# Order-20 Dirichlet Posterior + Phrase Cache

**val_bpb: 0.11545** (3-seed mean, std 0.0000010) | **~15.1 MB** | 8xH100 SXM

Extends n-gram backoff from order 15 to order 20, improving over our PR #948 (0.11556 BPB).

## Compliance Note (April 13, 2026)

This submission extends PR #948 with higher-order n-gram backoff. It uses the same eval-time hash-based n-gram cache architecture, which is under active community dispute and has not been ruled on by @0hq or @valerio-oai as of this update. See PR #948 for the full compliance writeup. Summary:

**Dispute threads:** Issue #402, Issue #677, PR #886 (@abaybektursun), PR #900 (@Robert-Sneiderman).

**Key facts about this submission:**

- Does NOT train on val_tokens. Model weights are frozen during eval.
- Does NOT run backward passes on val data.
- Uses a causal backward-looking n-gram cache: statistics are accumulated from tokens that have already been scored.
- Uses Dirichlet-Multinomial posterior predictive smoothing with per-order OBCL concentrations.
- 4M hash buckets, orders 2-20 backoff.

**The open question:** Whether hash collision density in eval-time n-gram caches inflates `P(correct_token)` in a way that makes the reported BPB a measurement artifact rather than real compression (@abaybektursun's argument on PR #886). Or whether the Dirichlet normalization is mathematically valid regardless (@Robert-Sneiderman's argument on PR #900).

I asked about this class of submission on Issue #402 on 2026-04-02 and there has been no maintainer response since. Leaving this PR open pending an official ruling. If ruled invalid, I will retract and close. If ruled valid, the numbers stand.

Distinct from the TTT-on-val class of violations which I have separately retracted in PR #1193, PR #406, and PR #1127.

Thanks to @MatoTeziTanka and the Agora community reviewers for raising the bar on compliance documentation across all PRs.

## Results (8xH100 80GB SXM, Montréal CA, 747 TFLOPS)

| Seed | Val BPB | Eval Time |
|------|---------|-----------|
| 1337 | 0.11544435 | 459s |
| 42 | 0.11546433 | 435s |
| 2025 | 0.11544736 | 438s |
| **Mean** | **0.11545 (std 0.0000010)** | |

## What changed from PR #948

- `NGRAM_ORDER=20` (was 15)
- Added 5 more per-order concentrations: all 1.86 (matching the pattern for high-order matches)
- Everything else identical

## Ablation (1xH100, 200 steps, Kansas City MO)

| Config | BPB | Delta |
|--------|-----|-------|
| Order 15 (baseline) | 0.11906 | — |
| **Order 20** | **0.11873** | **-0.00033** |
| Two-pass rescore | 0.11906 | 0 |
| Int5 quantization | 0.11906 | 0 |
| Comp alpha=0.30 | 0.11906 | 0 |

Order 20 was the only ablation that showed improvement.

## Credits

Same as PR #948. Built on @Robby955 (PR #900), @signalrush (PR #414), @himanshudongre (PR #846), @deanbrr (PR #659), @newjordan (PR #674), @pentxayc (PR #803).

## Run Command

```bash
NGRAM_ORDER=20 \
NGRAM_PER_ORDER_CONC="50.0,50.0,6.95,2.98,2.05,2.05,2.05,1.86,1.86,1.86,1.86,1.86,1.86,1.86,1.86,1.86,1.86,1.86,1.86" \
# ... all other params same as PR #948
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
