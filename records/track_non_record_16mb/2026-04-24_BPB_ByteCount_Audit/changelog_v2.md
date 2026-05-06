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


## v2.1 — 2026-04-24 — PR #1795 added, #1785 superseded

In response to @OE-GOD's reply on PR #1804 (the audit's PR), I re-fetched and
audited PR #1795 (the open successor of the closed #1785).

| PR | author | reported BPB | classification | bugs detected |
|---|---|---|---|---|
| #1795 (commit `cb5ad95`) | OE-GOD | 1.01252 | CORRECT | [] |
| #1785 (closed) | OE-GOD | 1.01925 | OBFUSCATED → superseded | n/a |

The static check passes all three properties on PR #1795. Tool output preserved
at `audit/per_pr_v2/1795.json`. Verdict: PR #1795 LUT is canonical. The
inflation-ratio correction does not apply to PR #1795. Frontier of the
LUT-verified correct-LUT entries moves from PR #1735 (1.04290) to PR #1795
(1.01252).

**Scope reminder.** This audit verifies LUT correctness only. PR #1795's
reported 1.01252 includes a byte-level PPM mixture on top of canonical NN
bytes; the mixture's gate legality (an outcome-independent adaptive-λ check)
was verified separately by @nprime06's review on PR #1795 itself (a target-
conditioned gate from an earlier commit was flagged and fixed). The audit
tool does not check gate legality.

## v2.1 addendum — 2026-04-29 — re-check of remaining OBFUSCATED PRs

Re-fetched the three other top-10 OBFUSCATED entries (#1758, #1738, #1771)
to check whether they had similarly converted to readable source after the
audit ran on 2026-04-23. They have not. All three remain OBFUSCATED with
their original wrappers (PRs #1758 and #1738 use the same lzma+exec pattern
as the original snapshot; PR #1771 uses an lzma+runpy variant). No
classification changes; the audit's per_pr_v2 entries for these PRs remain
accurate. Detail in `audit/per_pr_v2/obfuscated_recheck_2026-04-29.md`.


## v2.1 second addendum — 2026-04-29 — gap-bounding via SEQ_LEN/STRIDE invariance test

Run 4 tested whether the 0.77% gap between yahya's quoted 1.1746 and
the audit's 1.1655 reproduction lives in eval pipeline scoring
parameters. Result: the gap is invariant to seq_len ∈ {1024, 2048} and
stride ∈ {64, 1024}; all three tested configurations produce the same
ratio to within 1.6e-6.

The gap therefore does not live in scoring strategy. By elimination
across runs 1-4 (LUT structure, formula, boundary mask, three scoring
modes, eval windowing), the gap must live in tokenizer or val-shard
state. Yahya's `train_gdn_7k.py:58` defaults to SP1024; his audited
submission overrides to SP8192 (per submission.json). His PR #1734
disclosure analysis predates that submission and may have been computed
against the SP1024 default, against a different val shard, or
hand-derived.

The audit cannot replicate yahya's disclosure-time data. The gap is
bounded to data state, not pipeline structure. The audit's headline
numbers are unchanged.

See `audit/empirical_validation/run4_summary.md` and
`audit/empirical_validation/run4_seq_len_1024.py` for detail.


## v2.1 third addendum — 2026-04-29 — bug-family decomposition (run 5)

Decomposed yahya's three LUT bugs by constructing eight LUT variants
(canonical + each bug alone + each pair + all three) and measuring the
ratio for each. On SP8192 fineweb val, only Bug B (byte_token_wrong_size)
produces a measurable ratio change (-0.001476). Bugs A (leading_space_plus_one)
and C (missing_is_unused) are empirically zero on this val — Bug A because
leading-space tokens never follow boundary tokens by SentencePiece convention
(run 1.5), Bug C because the vocab has zero `sp.is_unused` tokens.

Implication: yahya's full LUT produces 1.1655 on SP8192 (Bug B alone shifts
the ratio down from canonical 1.1671 to 1.1655). His quoted 1.1746 is 0.0089
*above* canonical and cannot be produced from any of his three LUT bugs on
SP8192. The 0.77% gap between quoted and reproduced lives in tokenizer/val
state, not in his LUT (run 4 corroboration).

Methodology: introduced the structural-vs-empirical bug distinction in
`audit/methodology.md` to separate "structural deviation from canonical"
(static classifier verdict) from "empirical inflation on this val"
(measurable Δratio). The classifier flags the first; run 5 quantifies
the second.

See `audit/empirical_validation/run5_summary.md` for detail.
