# Run 5 Summary: Bug-family decomposition — only Bug B affects the SP8192 ratio

## Headline

Of yahya's three LUT bugs, only one (byte_token_wrong_size) produces a measurable ratio change on SP8192 fineweb val. The static classifier flags all three as structural deviations from canonical, but two of them (leading_space_plus_one, missing_is_unused) are empirically no-ops on this specific val state.

This is a meaningful methodological distinction: static deviations are not the same as empirical inflations.

## Results

Eight LUT variants tested (canonical + each bug alone + each pair + all three):

| Configuration | canonical_bytes | buggy_bytes | ratio | Δratio vs canonical |
|---|---|---|---|---|
| canonical (no bugs) | 151,080,878 | 176,332,734 | 1.1671413 | — |
| only_bug_a (leading_space +1) | 151,080,878 | 176,332,734 | 1.1671413 | 0.000000 |
| only_bug_b (byte_token=6) | 152,426,978 | 177,678,834 | 1.1656653 | -0.001476 |
| only_bug_c (missing is_unused) | 151,080,878 | 176,332,734 | 1.1671413 | 0.000000 |
| bugs_a_b | 152,426,978 | 177,678,834 | 1.1656653 | -0.001476 |
| bugs_a_c | 151,080,878 | 176,332,734 | 1.1671413 | 0.000000 |
| bugs_b_c | 152,426,978 | 177,678,834 | 1.1656653 | -0.001476 |
| all_three (yahya's full LUT) | 152,426,978 | 177,678,834 | 1.1656653 | -0.001476 |

The ratio shift between canonical (1.1671413) and yahya's full LUT (1.1656653) is entirely attributable to Bug B. Bugs A and C contribute zero on this val.

## Why Bug A is empirically zero on this val

Bug A bakes a +1 into base_bytes for every leading-space token. The eval-time formula adds +1 only when `(has_leading_space[y] & ~is_boundary[x])` — i.e., for leading-space tokens whose predecessor is non-boundary.

For Bug A's LUT-baked +1 to produce a different byte count than the eval-time formula, there would need to exist leading-space tokens whose predecessor IS boundary. Run 1 found 50,000 boundary predecessors on this val, all `<s>` document separators. By SentencePiece convention, the y-token following a `<s>` is the first token of a new document, which is never a leading-space token. Empirically, `(has_leading_space[y] & ~is_boundary[x]).sum() == has_leading_space[y].sum()` on this val.

So Bug A's LUT-baked +1 produces the same byte count as the canonical eval-time +1: `Σ has_leading_space[y]`. The ratio impact is zero.

This does not generalize. On a val where some leading-space tokens follow boundary tokens (e.g., if a future tokenization run inserted special boundary tokens differently), Bug A would produce a measurable inflation.

## Why Bug C is empirically zero on this val

`n_unused_tokens = 0` in the SP8192 vocabulary. The missing `sp.is_unused` in yahya's boundary predicate has nothing to omit. Bug C is empirically a no-op for any val tokenized with this vocab.

This also does not generalize. A vocab with non-zero unused tokens (rare but possible) would produce measurable Bug C inflation.

## Why Bug B is the dominant effect

256 byte tokens × 5 extra bytes per byte token = 1,280 bytes of vocab-level inflation. Distributed across 269,220 byte-token occurrences in val, this contributes 1,346,100 extra bytes to the canonical denominator. Since the buggy numerator is `canonical_total + eval_add` and eval_add is unchanged across bug configurations, inflating the canonical denominator decreases the buggy/canonical ratio.

This explains why yahya's full LUT (1.1656653) produces a *lower* ratio than canonical (1.1671413), not a higher one. The byte-token bug shrinks the ratio, while the +1 bug — if it had any effect on this val — would inflate it.

## Implication for the gap to yahya's quoted 1.1746

Yahya's quoted ratio is 1.1746 — 0.0089 higher than canonical's 1.1671. On SP8192, no combination of yahya's three LUT bugs can produce a ratio above canonical's. To produce 1.1746 from yahya's LUT structure, the underlying vocab + val state must differ from ours.

This corroborates run 4's finding: the gap lives in tokenizer/val state, not in any property of yahya's code that we can replicate on SP8192.

## Distinction worth naming

There are two senses in which a LUT can have a "bug":

1. **Structural deviation from canonical.** Detectable statically by reading the function. yahya's LUT has three structural deviations.
2. **Empirical inflation on a given val.** Measured by running both LUTs and comparing byte counts. On SP8192 fineweb val, only one of yahya's three structural deviations produces measurable inflation.

The audit's static classifier flags the first sense. The empirical run 5 quantifies the second. A reader of the audit should not conflate them. A structurally-buggy LUT may produce zero inflation on a particular val while still being structurally wrong; correcting it is still appropriate because it would inflate on a different val.

## Files
- run5_bug_decomposition.py / .json / .log
