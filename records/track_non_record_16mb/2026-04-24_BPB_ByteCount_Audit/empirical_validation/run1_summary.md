# Run 1 + 1.5 Summary: Boundary mask is non-trivial but quantitatively inactive

## Headline
The methodology.md section 4 claim that "is_boundary[x_prev] is identically zero on this val stream" is **factually wrong**. Empirical check shows 50,000 boundary-token predecessors in fineweb val. However, the three scoring-mode ratios still converge to 1.1671 to within 0.00000067%, because none of the 50,000 boundary-predecessor positions are followed by a leading-space y-token in this data.

## Run 1: vocab-level + per-position counts

Vocab has 4 special tokens:
- id=0 `<pad>` (control)
- id=1 `<s>` (control)
- id=2 `</s>` (control)
- id=3 `<unk>` (unknown)

Of these, only `<s>` (id=1) appears in val: **50,000 occurrences**, all in predecessor positions. These are document separators inserted by the tokenizer when packing val.

`<pad>`, `</s>`, `<unk>`: 0 occurrences each.

## Run 1.5: high-precision three-mode ratios

| Mode | scored_tokens | canonical_bytes | buggy_bytes | ratio |
|---|---|---|---|---|
| sliding-window-boundary-masked | 40,540,799 | 151,080,878 | 176,332,734 | 1.1671413109 |
| all-tokens-boundary-masked     | 40,540,802 | 151,080,891 | 176,332,748 | 1.1671413031 |
| all-tokens-no-mask             | 40,540,802 | 151,080,891 | 176,332,748 | 1.1671413031 |

Three modes agree to 6+ decimal places.

## Why the modes converge despite a non-trivial mask

The mask `~is_boundary[x_prev]` flags 50,000 positions where the predecessor is `<s>`. Of those 50,000 positions, **the y-token that follows is not a leading-space token**. It is the first token of a new document, which SentencePiece tokenizes without the leading-space prefix.

Therefore `(has_leading_space[y] & ~is_boundary[x]).sum() == has_leading_space[y].sum()` on this data, and the boundary-masked modes produce the same numerator as the no-mask mode.

The only ratio difference comes from the 3-token sliding-window trim (40,540,799 vs 40,540,802 scored), which shifts the ratio by 7.8e-9. This is a numerator/denominator scaling artifact.

## Implication for methodology.md section 4

Replace the claim "is_boundary[x_prev] is identically zero" with:

> The boundary mask flags 50,000 positions in this val stream, all corresponding to `<s>` (id=1) document-separator predecessors. None of those positions are followed by a leading-space y-token (SentencePiece does not prefix the first token of a new document with the leading-space marker). Therefore `(has_leading_space[y] & ~is_boundary[x]).sum() == has_leading_space[y].sum()` on this data, and the three scoring-mode ratios agree to 6+ decimal places.

## Files
- run1_boundary_mask_check.py / .json / .log
- run1_5_scoring_modes.py / .json / .log
