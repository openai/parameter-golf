# Audit Results — Top-10 Open PRs (snapshot 2026-04-23)

This is the per-PR write-up backing `audit/corrected_leaderboard.md`. For
each PR we record what the static tool found, what the inferred canonical
BPB is (or why we could not compute one), and any inspection notes. The
tool's raw JSON for each PR is in `audit/per_pr/<pr>.json` (v1
single-bug detector) and `audit/per_pr_v2/<pr>.json` (v2 three-variant
detector — both agree on every row; see `audit/changelog_v2.md`).

## Inputs

* **Snapshot date**: 2026-04-23 (leaderboard refreshed via
  `python scripts/pgolf.py leaderboard fetch`).
* **PR set**: top-10 open PRs sorted by reported `val_bpb` ascending.
* **Tool**: `scripts/canonical_rescore.py` (commit visible in
  `git log -- scripts/canonical_rescore.py`).
* **Hardware-parity anchor**: `exp_001/analysis.md` (PR #1727
  reproduction, seed 1337, val_bpb=1.07431, within +0.00214 of the
  reported 3-seed mean of 1.07217).
* **Inflation ratio on SP8192 fineweb val** (`sliding-window-boundary-masked`,
  tool default): 1.1671413 (canonical 151,080,891 bytes vs buggy
  176,332,748 bytes; 25,251,857 leading-space tokens; 633,420 scored
  windows of `seq_len=2048, stride=64`). yahya010's closure quoted ~1.1746
  against a different LUT + decoded-stream reference; both characterise
  the same bug (see `audit/methodology.md` §4).
* **"Pass merged-SOTA" threshold**: inferred canonical BPB ≤ 1.0738.
* **Scope caveat**: "LUT-verified CORRECT" means the LUT is canonical, not
  that the reported BPB is reproducible end-to-end. See
  `audit/writeup.md` "Scope and limitations".
* **Classifier version**: this table reflects both the v1 (single-bug)
  and v2 (three-bug) classifier outputs; they agree on every row. See
  `audit/changelog_v2.md` for the side-by-side diff.

## LUT-verified Top 5

These five PRs (plus the #1727 anchor) are statically confirmed to use
the canonical `len(piece.encode("utf-8"))` LUT. Their reported BPBs
require no LUT correction; full-pipeline correctness still rests on the
cross-entropy numerator being canonically measured.

| Rank | PR | Author | Canonical BPB | Notes |
|------|----|--------|---------------|-------|
| 1 | #1795 | OE-GOD | **1.01252** | "SP4096 + Byte-Level PPM Adaptive-λ Mixture" — LUT-verified frontier (post-2026-04-23 update); supersedes #1785 (closed). Reported NN-only 1.09764 ± 0.00044 (3-seed) matches @clarkkev's 2026-04-01 record (1.09785) within seed noise; -0.07435 BPB delta from byte-level PPM mixture with strict-legal outcome-independent gate (gate legality verified by @nprime06's review on PR #1795 itself, not by this audit). |
| 2 | #1735 | AjAnubolu | **1.04290** | "SP8192 + Parallel Pre-Quant TTT" — previously the LUT-verified frontier; remains LUT-verified |
| 2 | #1779 | leon2k2k2k | 1.06421 | "SP8192 + CaseOps + GatedAttn + QuantGate + Loop4-5 + PhasedTTT + RecurAlpha" |
| 3 | #1769 | dexhunter | 1.06453 | Same family, MLPClip12 variant (5-seed mean) |
| 4 | #1756 | romeerp | 1.06505 | "CaseOps + Recurrence Depth Curriculum" |
| 5 | #1736 | dexhunter | 1.06549 | Same family, earlier variant |

PR #1727 (yahya010, 1.07217) and PR #1784 (renqianluo, 1.07081) are
LUT-verified but rank below the top 5 by reported BPB.

## Per-PR inspection notes

### #1785 — OE-GOD — reported 1.01925 — OBFUSCATED
* Script dir: `records/track_10min_16mb/2026-04-23_SP4096_PPM_AdaptiveMix/`
* Two-line `train_gpt.py`: `import lzma as L,base64 as B` followed by
  `exec(L.decompress(B.b85decode("..."))`.
* LUT cannot be verified statically. `inferred_canonical_bpb` = unverified.
* Conditional arithmetic only (not a claim): if the obfuscated LUT were
  the #1698 buggy variant, the correction would give
  `1.01925 × 1.1671 ≈ 1.1896`. This is numerically close to yahya010's
  self-disclosed 1.1873 for PR #1734, but the similarity is observation,
  not evidence — we have no static or dynamic verification either way.


### #1795 — OE-GOD — reported 1.01252 — CORRECT (added 2026-04-24)

* Open PR (commit `cb5ad95`) submitted 2026-04-24, supersedes the closed #1785.
* `train_gpt.py` ships as readable source (no lzma wrapper). Tool returns `lut_status: CORRECT` with `lut_bug_detections: []` on all three checked properties:
  - `leading_space_noplus`: ✓ (no `+1` baked into LUT)
  - `byte_token_one`: ✓ (`base_bytes_np[token_id] = 1` for `sp.is_byte` tokens)
  - `boundary_predicate_full`: ✓ (predicate includes `sp.is_unused`)
* `build_sentencepiece_luts` is verbatim from @clarkkev's PR #1334.
* Reported breakdown:
  - NN-only sliding BPB mean (3-seed): 1.09764 ± 0.00044, matching @clarkkev's 2026-04-01 record (1.09785) within seed noise.
  - Mixture BPB (NN + byte-level PPM-D order-4 with adaptive-λ outcome-independent gate): 1.01252 (3-seed mean).
  - −0.07435 BPB delta computed on top of canonical NN byte count.
* **Audit scope caveat:** This audit verifies LUT correctness only. It does NOT verify gate legality of the byte-level PPM mixture. Gate legality was independently reviewed by @nprime06 on PR #1795 itself (target-conditioned gate flagged in earlier commit, fixed in `cb5ad95` to a strict-legal outcome-independent form). The 1.01252 number reflects the post-fix submission.
* PR #1804 reply thread on 2026-04-24 invited this re-audit; tool result preserved at `audit/per_pr_v2/1795.json`.

### #1758 — kilojoules — reported 1.02840 — OBFUSCATED
* Script dir: `records/track_10min_16mb/2026-04-20_SP8192_PreQuantTTT_Unfrozen_LR1e3/`
* Same `lzma.decompress(b85decode(...))` pattern as #1785.
* The PR title declares this is "PR #1738 + PreQuant TTT LR=1e-3". PR
  #1738 is itself OBFUSCATED (below).
* `inferred_canonical_bpb` = unverified. Conditional arithmetic (not a
  claim): if buggy, `1.02840 × 1.1671 ≈ 1.2003`.

### #1738 — alertcat — reported 1.03540 — OBFUSCATED
* Script dir: `records/track_10min_16mb/2026-04-19_SP8192_PreQuantTTT_CaseOps_V15/`
* Same obfuscation pattern.
* The PR title declares this is "PR #1735 + CaseOps Tokenizer V15". PR
  #1735 is itself CORRECT (below); the obfuscation here therefore changed
  more than just the tokenizer, and we cannot tell what.
* `inferred_canonical_bpb` = unverified. Conditional arithmetic (not a
  claim): if buggy, `≈1.2084`.

### #1735 — AjAnubolu — reported 1.04290 — CORRECT
* Script dir: `records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/`
* `build_sentencepiece_luts` is the canonical version (no `+1`).
* Was the LUT-verified frontier at the time of the 2026-04-23 snapshot. Superseded as frontier by PR #1795 (OE-GOD, 1.01252) following PR #1795's verified-CORRECT update on 2026-04-24.
* Threshold check: 1.04290 ≤ 1.0738 — would clear the merged-SOTA
  reference by 0.031 if held against the same 1.0738 threshold yahya's
  closure note implied.
* **Caveat.** The 0.021 BPB gap to the next-best LUT-verified entry is
  large enough that independent reproduction is warranted before treating
  this as the authoritative record. LUT correctness is necessary but not
  sufficient.

### #1779 — leon2k2k2k — reported 1.06421 — CORRECT
* Script dir: `records/track_10min_16mb/2026-04-23_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT_RecurAlpha/`
* CaseOps + GatedAttn + Loop4-5 + Phased TTT + Recurrent Alpha stack.
* Canonical BPB 1.06421.

### #1769 — dexhunter — reported 1.06453 — CORRECT
* Script dir: `records/track_10min_16mb/2026-04-22_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT_MLPClip12/`
* 5-seed mean reported. Canonical BPB 1.06453.

### #1756 — romeerp — reported 1.06505 — CORRECT
* Script dir: `records/track_10min_16mb/2026-04-20_SP8192_CaseOps_GatedAttn_QuantGate_Loop134_Curriculum_PhasedTTT/`
* CaseOps Tokenizer + Recurrence Depth Curriculum. Canonical BPB 1.06505.

### #1771 — bigbag — reported 1.06513 — OBFUSCATED
* Script dir: `records/track_10min_16mb/2026-04-22_SP8192_CaseOps_V13_L2_LoRA_TTT/`
* Wrapper variant: `_c=lzma.decompress(base64.b85decode("..."))` followed by
  `tempfile`/`runpy` execution. Detector handles both inline-`exec` and
  this `runpy` form.
* `inferred_canonical_bpb` = unverified. Conditional arithmetic (not a
  claim): if buggy, `≈1.2434`. The reported BPB sits at the top of the
  1.064-1.066 cluster of LUT-verified PRs, which is consistent with (but
  not evidence of) a canonical LUT.

### #1736 — dexhunter — reported 1.06549 — CORRECT
* Script dir: `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/`
* Same family as #1769 / #1779. Canonical BPB 1.06549.

### #1784 — renqianluo — reported 1.07081 — CORRECT
* Script dir: `records/track_10min_16mb/2026-04-23_GatedAttn_AlphaLoRA144_WarmStart_1.07081/`
* "GatedAttn + Alpha-Scaled LoRA + Warm-start A + WD 1.0" — 3-seed mean.
* Canonical BPB 1.07081.

## Summary table (LUT-verified only)

Reproduced from `audit/corrected_leaderboard.md` for convenience. Sorted
by canonical BPB ascending. Only includes PRs whose `train_gpt.py` was
statically classified as CORRECT.

| Rank | PR | Author | Canonical BPB | Δ to next |
|------|----|--------|---------------|-----------|
| 1 | #1735 | AjAnubolu | 1.04290 | +0.02131 |
| 2 | #1779 | leon2k2k2k | 1.06421 | +0.00032 |
| 3 | #1769 | dexhunter | 1.06453 | +0.00052 |
| 4 | #1756 | romeerp | 1.06505 | +0.00008 |
| 5 | #1736 | dexhunter | 1.06549 | +0.00532 |
| (anchor) | #1727 | yahya010 | 1.07217 | +0.00136 (above #1784) |
| 6 | #1784 | renqianluo | 1.07081 | — |

PR #1735's canonical BPB lead of 0.02131 over the next-best LUT-verified
result is a substantial gap. The static audit verifies the LUT only, not
the full training pipeline or the reported BPB. Whether the gap reflects
a genuine capability step-change or a path the tool does not inspect is
outside the scope of this audit; we flag it as "LUT-verified,
reproduction-pending" rather than "verified record".

## What the obfuscated entries imply

The four OBFUSCATED entries (#1785, #1758, #1738, #1771) include the
three lowest reported BPBs in the leaderboard. We state the observation
neutrally and explicitly note this is not a causal claim: the tool cannot
tell whether any of these PRs are buggy or canonical. yahya010's
self-closure of PR #1734 (also obfuscated, reported 1.0108,
self-confirmed canonical ~1.1873) is the only data point we have on
what's behind a sub-1.05
obfuscated submission. We do not extrapolate from one case. We record
the pattern as an observation (not a causal claim): every sub-1.05 entry
on the current leaderboard is in obfuscated code, and the only sub-1.05
entry with a self-disclosed LUT classification was buggy. This is
information a reviewer may weigh; it is not evidence that any specific
obfuscated PR is buggy.

The LUT-verified frontier sits at 1.04290 (PR #1735) — below the 1.0738
threshold but above the three sub-1.05 reported OBFUSCATED entries
(which are unverified in either direction).
