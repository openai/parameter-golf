# Audit Changelog — v1 → v2

**Date**: 2026-04-24
**Tool change**: `scripts/canonical_rescore.py` was extended from a single-bug
detector (the +1 leading-space baking, `leading_space_plus_one`) to a
three-variant classifier that also detects `byte_token_wrong_size` and
`missing_is_unused`. See `audit/methodology.md` §5 for the property
definitions and the regex / window detectors that implement them.

## Headline

**Zero top-10 PRs changed classification under v2.**

Every PR that was `CORRECT` under v1 remains `CORRECT` under the stricter
three-property check; every PR that was `OBFUSCATED` remains `OBFUSCATED`.
The static audit finds no new `BUGGY` PRs in the current top 10.

This strengthens the claim in `audit/writeup.md`: the current top-10 does
not contain the #1698 lineage bug family in plain code. Whether the
obfuscated PRs contain it behind their `lzma.decompress(base64.b85decode(...))`
wrappers remains out of scope for a no-code-execution audit.

## Side-by-side comparison

| PR | Author | v1 status | v2 status | v2 deviations |
|-----|---------|-----------|-----------|---------------|
| #1785 | OE-GOD | OBFUSCATED | OBFUSCATED | [] |
| #1758 | kilojoules | OBFUSCATED | OBFUSCATED | [] |
| #1738 | alertcat | OBFUSCATED | OBFUSCATED | [] |
| #1735 | AjAnubolu | CORRECT | CORRECT | [] |
| #1779 | leon2k2k2k | CORRECT | CORRECT | [] |
| #1769 | dexhunter | CORRECT | CORRECT | [] |
| #1756 | romeerp | CORRECT | CORRECT | [] |
| #1771 | bigbag | OBFUSCATED | OBFUSCATED | [] |
| #1736 | dexhunter | CORRECT | CORRECT | [] |
| #1784 | renqianluo | CORRECT | CORRECT | [] |

Raw v2 JSON is in `audit/per_pr_v2/<pr>.json`. v1 JSON (kept for
comparison) remains in `audit/per_pr/<pr>.json`.

## Known-buggy control: yahya010's PR #1734 train_gdn_7k.py

The v2 classifier was spot-checked on yahya010's self-closed PR #1734
(`records/.../train_gdn_7k.py`) — which yahya's own closure note identified
as having the combined LUT bug:

```
lut_status: BUGGY
lut_bug_detections: ['leading_space_plus_one', 'missing_is_unused']
```

Two of the three canonical-property deviations are detected. The third
(byte-token sizing) is implicit rather than explicit in yahya's code — his
function does not have a `sp.is_byte(...)` branch at all, and byte tokens
fall through to the default `base_bytes[i] = len(piece.encode("utf-8"))`
path. Per the v2 detector's explicit design rule ("absent sp.is_byte
branch ⟹ `INDETERMINATE`, not `DEVIATES`"), the P2 detector correctly
returns INDETERMINATE on yahya's code. The classifier still flags him as
BUGGY via the other two deviations, so no classification is lost — only the
fine-grained deviation list differs from the task-spec description.

## Side notes from the re-audit run

* An earlier v2 run pointed the audit at
  `records/track_10min_16mb/2026-04-19_GatedDeltaNet_MacroPhase_Brotli_LegalTTT/`
  for PR #1735. Investigation showed this directory was NOT on pr-1735 —
  it was staged leftover from a prior pr-1734 session. `git stash` cleaned
  the working tree and the subsequent audit correctly picked up
  `2026-04-18_SP8192_ParallelPreQuantTTT/` (PR #1735's actual record
  directory). No substantive finding; noted here so future re-runs know to
  start from a clean tree.
* The P1 regex was broadened in this pass from `piece.encode("utf-8")`
  verbatim to `<expr>.encode("utf-8")` where `<expr>` is any
  paren-free subexpression. This was required to detect the
  `piece[1:].encode("utf-8")` variant yahya uses. The existing `+1`
  fixture and PR #1727 regression tests still pass, so the broadening
  does not change the top-10 classifications.

## Conclusion

No action needed against any top-10 PR as a result of the v2 audit. The
extended classifier is now available for future audits (obfuscated-PR
de-obfuscation, new submissions) and is documented in
`audit/methodology.md` §5 and `scripts/README_canonical_rescore.md`.
