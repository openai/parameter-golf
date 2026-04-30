# Run 3 Summary: Yahya's full LUT reproduction — gap unresolved

## Headline
Yahya's exact `build_sentencepiece_luts` (lines 206-219 of `train_gdn_7k.py`), run on our SP8192 fineweb val with the same canonical/buggy formula the audit tool uses, produces ratio **1.1655** (sliding-window) or **1.1655** (all-tokens). His PR #1734 closure quoted **1.1746**. The 0.77% gap is in the *opposite* direction from what the previous audit writeup claimed (it claimed his code on our val gave 1.1770, within 0.2% of his quoted; this is empirically false).

## Numbers

| LUT × formula | Ratio |
|---|---|
| Canonical PR #1727 LUT, canonical formula, sliding-window | 1.1671413 |
| Canonical PR #1727 LUT, canonical formula, all-tokens | 1.1671413 |
| Yahya's LUT, yahya formula (his +1 baked in), sliding-window | 1.1655024 |
| Yahya's LUT, yahya formula, all-tokens | 1.1655024 |
| Yahya's quoted (PR #1734 closure) | 1.1746 |

Canonical reproduction matches the audit tool to floating-point precision (`canonical_matches_audit_tool: true`).

## Why yahya's actual is below canonical

His byte-token bug (run 2: BUG_PRESENT) inflates his canonical denominator by 1,346,100 bytes. Since the buggy numerator gets the same `eval_add` (25,251,856), and yahya's denominator is larger than canonical's, his computed ratio is smaller than canonical's.

This is opposite to the direction needed to explain his 1.1746 quote, which is *larger* than canonical's 1.1671.

## What we cannot conclude

- We cannot reproduce 1.1746 from yahya's code alone on our val. The 0.77% gap is unexplained.
- Possible causes (not investigated this run):
  - Yahya used a different val shard than ours (we have one shard from fineweb_val_000000.bin; his pipeline may have used a different tokenization run, a different number of shards, or pre-processing variants).
  - Yahya's `eval_val_sliding` pipeline has additional differences from the canonical one that affect the byte sum (e.g., different boundary handling, different stride or seq_len, off-by-one).
  - Yahya's quoted 1.1746 came from a hand calculation or rough estimate, not a direct script output.

## Implications for the audit's prior claims

The methodology.md, writeup.md, and PR #1804 outreach comment on PR #1734 all currently reference "1.1770 within 0.2% of 1.1746." This claim does not survive direct empirical check. Concretely:

- methodology.md section 4 needs the 1.1770 number replaced with 1.1655 and the framing changed from "gap closed to 0.2%" to "gap unexplained without yahya's eval code."
- The PR #1804 outreach comment needs a correction follow-up.

## What is solid and shippable

- Audit tool's reported ratio (1.1671413) is correct, validated by independent reconstruction.
- Yahya's `train_gdn_7k.py` LUT has three deviations from canonical: leading_space_plus_one (his disclosure), byte_token_wrong_size (run 2 BUG_PRESENT), missing_is_unused (visible in line 217). All three are statically detectable.
- The bug family does not appear in plain-text code at the top of the open leaderboard (audit run 1+1.5: 6 CORRECT, 4 OBFUSCATED, 0 BUGGY across top-10).
- Three scoring modes converge to 1.1671413 to within 7.8e-9 (run 1.5).

## Files
- run3_yahya_full_lut.py / .json / .log
