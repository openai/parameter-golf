# Maintainer Brief

This non-record package is meant to be useful even if no leaderboard entry from
me is merged.

## Why this exists

The final week created a review problem: many PRs reported impressive BPB
numbers, but the underlying mechanisms were not comparable.  Some changed the
neural model, some changed the tokenizer, some used score-first TTT, some used
byte-level PPM, and some adapted on validation tokens in ways that are hard to
separate from leakage.

This brief offers a compact way to triage those claims.

## The central observation

Late Parameter Golf results should be read as:

```text
reported BPB =
  neural model quality
+ quantization damage
+ tokenizer / normalization choice
+ byte-denominator accounting
+ legal eval-time causal memory
+ possible validation-adaptation leakage
+ implementation / timing details
```

The leaderboard scalar alone does not identify which term moved.

## The most useful review split

1. **Clean neural stack.**  Standard full-vocab softmax, standard tokenizer,
   score-first TTT.  Low legality risk.  Examples: the #1855 -> #1953 -> #2060
   style lineage and related retunes.

2. **Tokenizer-side stack.**  Custom tokenizer or text transform with byte
   sidecar.  Potentially very valuable, but review should focus on raw-byte
   round-trip, sidecar sums, split provenance, U+2581, byte fallback, and
   whether normalization is lossy.

3. **Token-level eval-time memory.**  Prefix-only n-gram hint, closed-form
   full-vocab renormalization, inside eval timer.  This is the cleanest form of
   eval-time memory.

4. **Byte/PPM eval-time memory.**  Often strictly causal and score-first, but
   C2 depends on whether scoring a byte distribution is accepted for a token
   leaderboard.  This needs a maintainer ruling rather than more BPB tables.

5. **Validation adaptation.**  Must prove score-before-update by token/chunk
   interval.  Post-adaptation scores on the same validation tokens should not
   be treated as record evidence.

## Evidence from my own final runs

My final paid experiments on the PR #2018 frontier all failed at pre-quant:

| Branch | Pre-quant BPB | Why it matters |
|---|---:|---|
| #2018 reference seed 1337 | 1.05124428 | target |
| Gate32 + q-aware token-only tilt | 1.06385301 | not recoverable |
| Gate32 + native n-gram | 1.06434971 | q-aware patch was not the cause |
| exact #2018 gates + BigramHash 512x4 | 1.06471733 | tiny BigramHash did not recover |

This is the clearest operational lesson: on the late frontier, pre-quant BPB is
the first serious kill gate.

## Suggested minimum evidence for future record PRs

```text
artifact bytes per seed
train wallclock per seed
eval wallclock per seed, with preprocessing boundary specified
exact seed set and no replacement after seeing validation
scored alphabet and proof of normalization
score/update interval proof for any adaptive eval state
tokenizer provenance and val/train split rule
byte-sidecar sum equals original UTF-8 bytes on adversarial tests and full val
```

## Why negative results should be mergeable

Good negative results reduce duplicated compute.  In this competition, one bad
8xH100 run could cost more than many people were willing to spend personally.
A non-record PR that cleanly says "this attractive idea fails here, for this
mechanistic reason" is infrastructure for the next researcher.
