# Competition Frontier Audit — 2026-03-20

This note distills a review of the current `openai/parameter-golf` frontier, with emphasis on what looks portable to a valid standard-track `8xH100` run and what does not.

Primary references:
- [Issue #140](https://github.com/openai/parameter-golf/issues/140)
- [PR #198](https://github.com/openai/parameter-golf/pull/198)
- [PR #236](https://github.com/openai/parameter-golf/pull/236)
- [PR #180](https://github.com/openai/parameter-golf/pull/180)
- [PR #179](https://github.com/openai/parameter-golf/pull/179)
- [PR #162](https://github.com/openai/parameter-golf/pull/162)
- [PR #192](https://github.com/openai/parameter-golf/pull/192)
- [PR #135](https://github.com/openai/parameter-golf/pull/135)
- [PR #64](https://github.com/openai/parameter-golf/pull/64)
- [PR #174](https://github.com/openai/parameter-golf/pull/174)
- [PR #168](https://github.com/openai/parameter-golf/pull/168)

## High-confidence themes

1. The durable standard-track frontier has moved to `11L` stacks.
   Depth, enabled by aggressive compression-aware artifact budgeting, appears more important than continuing to polish older `9L` and `10L` families.

2. `WD ~= 0.038-0.04` is part of the real recipe.
   This is not just ordinary regularization. Multiple strong submissions are using weight decay as a quantization/compression aid, shrinking weight magnitudes and improving post-quant behavior.

3. Smaller fixed-time batch can be more important than total tokens.
   The strongest portable training insight from the recent frontier is the `524k` batch result in [PR #236](https://github.com/openai/parameter-golf/pull/236): more updates in `600s` beat fewer, larger steps.

4. Sliding-window eval is materially inflating leaderboard-facing numbers.
   In several top PRs, roughly `0.015-0.023` BPB of the headline score comes from stride-64 sliding eval rather than the model itself. This is useful for competition understanding, but should not be confused with portable training progress.

5. QAT is not yet convincing enough to prioritize.
   It appears in some strong submissions, but the portable evidence is weaker than for `11L + WD + better step count`. It adds complexity and can add runtime/memory cost.

6. Mixed `int5/int6` export is real, but probably not the first move.
   It is a useful serialization trick when size is the dominant bottleneck, but it seems to trade away too much quality/speed if the main bottleneck is training quality per step.

7. SmearGate + BigramHash appear helpful, but secondary.
   They are common in the frontier and likely additive, but the clearest drivers appear to be depth, weight decay, export discipline, and training-step optimization.

## PR-level takeaways

### PR #236

- Strongest portable insight: `TRAIN_BATCH_TOKENS=524288`
- Best throughput of the recent leaders
- Main lesson: in a `600s` budget, more optimizer steps beat more tokens
- Important caveat: the implementation/export story is not as cleanly “int6-all” as the PR text suggests

### PR #198

- Strongest overall public stack
- Important ingredients: `11L`, `WD=0.04`, SWA, SmearGate, BigramHash, FA3
- Main caution: the headline score depends heavily on sliding eval; non-sliding roundtrip is much worse

### PR #179

- Best portable non-sliding `11L` recipe among the earlier 11-layer lines
- Real novelty: use compression headroom for depth and control the quantization gap with strong WD
- Good reference for “what survives if sliding eval is removed”

### PR #180

- Most novel export trick: mixed `int5/int6`
- Useful if the hard problem is bytes
- Probably not the right first branch if the hard problem is model quality in `600s`

### PR #192

- Main novelty: int6 QAT
- Least convincing of the major additions
- Looks more complex than it is valuable for our immediate path

### PR #135 / PR #162

- Important historical lineage: orthogonal init, `MLP 3x`, SmearGate, BigramHash, mixed low-bit export, SWA
- These are best thought of as the build-up to the stronger `11L + WD` frontier rather than the final destination

## What looks durable

- `11L`, `512d`, GQA `8/4`
- `MLP 3x`
- `WD ~= 0.04` on Muon and likely AdamW too
- Late SWA
- Careful artifact/export accounting
- Selective higher precision for the most sensitive tensors
- Possibly SmearGate + BigramHash, but only on top of the above

## What looks brittle or easy to overcredit

- Headline scores that rely on stride-64 sliding eval
- QAT as a default assumption
- Mixed int5 when size is not the primary limiter
- Paid-prefix / answer-storage approaches unless making an explicit rules bet
- Any result whose eval logic has not been audited carefully

## Bottom line

The frontier is no longer “tune the old `10L` leader-core and hope.” The durable line is now:

- `11L`
- compression-aware export
- `WD ~= 0.04`
- late averaging
- and, critically, training for more useful updates inside the `600s` wallclock

If the target is a winning standard-track model to test next, the most important portable lesson is:

> Build around an `11L` WD-heavy stack and adopt the smaller-batch fixed-time training regime before spending more cycles on exotic eval or quantization tricks.
