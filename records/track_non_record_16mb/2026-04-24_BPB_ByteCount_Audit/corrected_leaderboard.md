# Corrected Leaderboard — Top-10 Open PRs (April 2026)

**Methodology.** For each of the 10 open PRs with the lowest reported `val_bpb`,
we fetched the PR branch from `openai/parameter-golf` and ran
`scripts/canonical_rescore.py` against the `train_gpt.py` under the PR's
`records/track_10min_16mb/<dated-dir>/`. The tool statically inspects
`build_sentencepiece_luts` for the buggy `+1` pattern from the #1698 lineage
(yahya010, PR #1734 self-closure 2026-04-19) and computes the canonical and
buggy byte totals over the sliding-window scored-token subset
(`seq_len=2048`, `stride=64`) of the SP8192 fineweb val shard. No model is
loaded; the correction factor is `inferred_canonical_bpb = reported_bpb ×
(buggy_bytes / canonical_bytes)` for BUGGY scripts. On SP8192 fineweb val
the `buggy/canonical` ratio is **~1.1671** under the tool's default
scoring mode (`sliding-window-boundary-masked`); yahya010's closure note
quoted **~1.1746** under a different LUT + decoded-stream ground truth.
Both characterize the same bug — see `audit/methodology.md` §4. The
hardware-parity anchor is exp_001 (PR #1727 reproduction, seed 1337,
val_bpb=1.07431, within tolerance of the reported 3-seed mean of
1.07217). Threshold for "Passes" inclusion is `inferred_canonical_bpb ≤
1.0738` (one record-class margin under the merged-SOTA reference).

**Scope caveat.** A `CORRECT` verdict means the LUT is canonical. It does
*not* imply the model artifact achieves the reported BPB, that
`eval_val_sliding` itself is canonical, or that no other measurement
irregularity exists. See `audit/writeup.md` "Scope and limitations".

**Classifier version.** This table reflects both the v1 (single-bug)
and v2 (three-bug) classifier outputs — they agree on every row. See
`audit/changelog_v2.md` for the side-by-side diff.

## Full audited table

Sorted by reported BPB (best first). "Inferred canonical BPB" is the buggy
value × `1.1671` for BUGGY scripts (none in the current top 10); for
CORRECT scripts the reported value already is canonical; for OBFUSCATED
scripts the LUT cannot be verified without executing the encoded blob.

| Rank | PR | Author | Reported BPB | LUT Status | LUT-verified | Inferred Canonical BPB | Passes ≤1.0738? |
|------|----|--------|-------------|-----------|:---:|------------------------|-----------------|
| 1 | #1785 | OE-GOD | 1.01925 | OBFUSCATED | no | unverified — closed/superseded by #1795 | n/a |
| 2 | #1758 | kilojoules | 1.02840 | OBFUSCATED | no | unverified | ? |
| 3 | #1738 | alertcat | 1.03540 | OBFUSCATED | no | unverified | ? |
| 4 | #1735 | AjAnubolu | 1.04290 | CORRECT | yes | 1.04290 | Yes |
| 5 | #1779 | leon2k2k2k | 1.06421 | CORRECT | yes | 1.06421 | Yes |
| 6 | #1769 | dexhunter | 1.06453 | CORRECT | yes | 1.06453 | Yes |
| 7 | #1756 | romeerp | 1.06505 | CORRECT | yes | 1.06505 | Yes |
| 8 | #1771 | bigbag | 1.06513 | OBFUSCATED | no | unverified | ? |
| 9 | #1736 | dexhunter | 1.06549 | CORRECT | yes | 1.06549 | Yes |
| 10 | #1784 | renqianluo | 1.07081 | CORRECT | yes | 1.07081 | Yes |
| anchor | #1727 | yahya010 | 1.07217 | CORRECT | yes | 1.07217 | Yes |

"LUT-verified" is necessary but not sufficient — see the scope caveat above.

## LUT-verified Top 5

After excluding PRs whose `train_gpt.py` is wrapped in
`lzma.decompress(base64.b85decode(...))` and therefore cannot be statically
audited, the LUT-verified frontier is:

| Rank | PR | Author | Canonical BPB |
|------|----|--------|---------------|
| 1 | #1795 | OE-GOD | **1.01252** |
| 2 | #1735 | AjAnubolu | **1.04290** |
| 2 | #1779 | leon2k2k2k | 1.06421 |
| 3 | #1769 | dexhunter | 1.06453 |
| 4 | #1756 | romeerp | 1.06505 |
| 5 | #1736 | dexhunter | 1.06549 |

PR #1795 (OE-GOD, "SP4096 + Byte-Level PPM Adaptive-λ Mixture") leads the LUT-verified frontier as of 2026-04-24, with reported BPB 1.01252 (3-seed mean, full val). PR #1735 (AjAnubolu, "SP8192 + Parallel Pre-Quant TTT") was the previous frontier and remains LUT-verified at 1.04290. The
LUT-verified line by ~0.022 BPB over the next-best PR (#1779). This gap is
large enough that independent reproduction is warranted before treating
#1735 as the authoritative record — the tool verifies the LUT, not the
full training pipeline. PRs #1727 and #1784 (LUT-verified, mid-1.07 range)
are within seed-noise of each other and represent the previous-frontier
QK-Gain stack.

## Caveats

The four OBFUSCATED PRs (#1785, #1758, #1738, #1771) report BPB values
spanning the three-best (#1785, #1758, #1738) and one mid-pack
(#1771). For them we have no way to verify whether the LUT is canonical or
inflated without running the encoded blob in a sandbox; the static tool
returns `OBFUSCATED — cannot verify statically`. We do **not** assert
these PRs are buggy; the OBFUSCATED verdict is neutral and only states
that static inspection does not reach them.

The observation that the three lowest reported BPBs on the current
leaderboard are all OBFUSCATED is a pattern, not a causal claim. The one
self-disclosed data point (yahya010's PR #1734, 1.0108 → ~1.1873) shows
that an obfuscated sub-1.05 submission can turn out to be buggy, but
cannot be generalised — other obfuscated PRs may use canonical LUTs and
simply distribute their code in compressed form.

## Per-PR JSON

Raw tool output for each PR is in `audit/per_pr/<pr>.json`. The driver
script is `audit/run_audit.sh`.
