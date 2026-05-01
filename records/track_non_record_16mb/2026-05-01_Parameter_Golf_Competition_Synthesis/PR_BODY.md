# Non-record: Parameter Golf Closing Synthesis

This is a non-record methodology PR.  It does **not** claim a leaderboard score.

I am submitting it because my final record attempt failed, but the evidence
around that failure is still useful.  The PR is a map of the 10min/16MB track:
what consistently worked, what failed to transfer, what was legality-sensitive,
and what review harnesses would have saved the community the most compute.

## What this adds

- A competition-wide synthesis through the late PR feed around PR #2103.
- A taxonomy of:
  - clean neural/quantization frontier work,
  - tokenizer/text-representation work,
  - token-level n-gram and byte/PPM eval methods,
  - validation-adaptation / PreQuantTTT methods,
  - byte-denominator failure modes.
- A proposed record-review checklist covering:
  - artifact bytes,
  - train/eval wallclock,
  - C1-C4,
  - byte-denominator accounting,
  - tokenizer provenance,
  - seed policy.
- A source map linking the synthesis to public PRs/issues rather than relying
  on memory.
- A maintainer-facing one-page brief and a longer research-notes layer.
- An OpenAI-research-facing brief on how these insights translate beyond this
  leaderboard to small models, compression, tokenization, and eval-time memory.

## Relationship to my final-day autopsy

This branch also includes the companion folder:

```text
records/track_non_record_16mb/2026-05-01_LastDay_Frontier_Transfer_Autopsy/
```

That folder contains my own final 8xH100 negative results and logs:

- Gate32 + q-aware token-only tilt on PR #2018: failed at pre-quant.
- Gate32 + native PR #2018 n-gram: failed at pre-quant.
- Exact PR #2018 gates + tiny BigramHash: failed at pre-quant.
- CrossWS tokenizer: promising train-proxy token-count result, not deadline-safe.
- Memento/copy memory: corrected no-go after fixing prefix-depth accounting.

This synthesis folder steps back and places those results in the broader
competition context.

## Main thesis

The competition stopped being "just train a better tiny transformer."  The late
frontier was shaped by the interaction of:

- neural training quality,
- quantization damage,
- artifact packing,
- custom tokenization,
- byte accounting,
- legal score-first eval-time adaptation,
- and the exact scored alphabet.

That is why I do not want to package a borderline or disputed result as a
record claim.  The more honest contribution is a synthesis of what evidence
actually survived review pressure.

The broader research claim is that Parameter Golf is a compressed prediction
systems benchmark.  The useful object is:

```text
representation + weights + quantizer + memory + evaluator + legality protocol
```

That matters for small OSS/on-device models and adaptive assistants beyond this
leaderboard.

## Why this should be mergeable

The PR is useful to maintainers because it reduces review entropy:

- it separates clean neural gains from tokenizer/eval/denominator effects,
- it identifies which public claims need explicit C2 or byte-accounting rulings,
- it proposes a compact evidence checklist future record PRs can copy,
- and it includes primary logs for the negative results I personally paid to
  test under the real 8xH100 constraint.

It is useful to competitors because it says which attractive ideas failed, and
why, instead of leaving those failures trapped in private pod logs.

## Compliance posture

This is a non-record document package.  It includes no new scored model and no
leaderboard claim.
