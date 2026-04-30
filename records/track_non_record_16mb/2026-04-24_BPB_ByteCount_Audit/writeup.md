# Measurement Integrity Note: BPB Byte-Count Audit of the #1698 Lineage

**Type**: Non-record PR — tooling + methodology contribution.
**Track**: `track_non_record_16mb`
**Authors of this PR**: (filer)
**Acknowledgement**: This work systematizes the byte-count discrepancy that
**yahya010** discovered and self-reported in PR #1734 closure on 2026-04-19.

---

## TL;DR

* yahya010 self-reported in PR #1734 closure that
  `build_sentencepiece_luts` in the #1698 lineage bakes a `+1` into the byte
  LUT for leading-space tokens, while `eval_val_sliding` then adds the same
  `+1` again, double-counting.
* That double-count inflates the byte denominator of BPB by **~16.71%** on
  the sliding-window scored subset that PR #1727's `eval_val_sliding`
  actually uses (151,080,891 canonical vs 176,332,748 buggy bytes on SP8192
  fineweb val, 633,420 windows of `seq_len=2048, stride=64`). yahya010's
  closure quoted **~17.46%** against a different reference — his own
  #1734 LUT applied to the decoded-stream ground truth. Both ratios
  characterize the same underlying bug; the small numerical difference is
  a scoring-strategy + LUT-construction artefact, documented in
  `audit/methodology.md` §4. Reported buggy BPBs translate to canonical
  BPBs via `canonical = reported × inflation_ratio` where the ratio is
  whichever one matches the PR's own scoring.
* We publish `scripts/canonical_rescore.py`: a static LUT inspection +
  byte-count tool that requires no GPU, no checkpoint, and no reproduction
  run. Drop in any `train_gpt.py` and it returns the LUT classification,
  the exact inflation ratio over the actual scored-token subset, and the
  inferred canonical BPB. The tool supports three `--scoring-mode`
  variants so reviewers can reproduce both the 1.1671 and 1.1746 numbers.
* The classifier is a **three-variant** detector: beyond the +1
  leading-space bake (`leading_space_plus_one`) it also checks
  `byte_token_wrong_size` (sp.is_byte branch sizing byte tokens by UTF-8
  length of the literal `"<0xXX>"` string) and `missing_is_unused`
  (boundary predicate omits `sp.is_unused`). yahya010's PR #1734
  `train_gdn_7k.py` is the case where multiple variants co-occur. The
  extended classifier applied to the current top-10 PRs produces the
  same classification as the single-bug detector (6 CORRECT, 4
  OBFUSCATED) — see `audit/changelog_v2.md`.
* Applying the tool to the **top 10 open PRs by reported BPB** as of
  2026-04-23: 6 are CORRECT (canonical LUT verified), 4 are OBFUSCATED
  (`lzma.decompress(base64.b85decode(...))` — LUT cannot be verified
  statically). The LUT-verified correct-LUT frontier as of 2026-04-24 is
  **PR #1795** (OE-GOD, 1.01252), which supersedes the closed #1785; PR #1735
  (AjAnubolu, 1.04290) is also LUT-verified, followed by the cluster of 1.064-1.071 PRs
  anchored by the reproducible PR #1727 stack.

This is a **tooling and methodology contribution**, not a disqualification
petition. The intent is to give future submitters a one-command self-check
("did I inherit the #1698 LUT bug?") and to help reviewers separate
LUT-verified results from unverified ones.

---

## The bug, in one paragraph

Canonical SentencePiece BPB attributes one byte to the leading space of a
piece beginning with the `▁` marker, but only when the previous token is
*not* a boundary token (UNK / control / unused). The #1700-line
implementation (PR #1727 line 196) writes `base_bytes_np[token_id] =
len(piece.encode("utf-8"))` after stripping the `▁`, then in
`eval_val_sliding` adds `(has_leading_space[y] & ~is_boundary[x_prev])`. The
#1698 line writes `base_bytes_np[token_id] = len(piece.encode("utf-8")) + 1`
inside the leading-space branch — so the `+1` is *already* baked into the
LUT — and then *also* adds the boundary-gated `+1` at eval time. Each
leading-space scored token is therefore credited with one extra byte beyond
canonical. On SP8192 fineweb val, leading-space tokens account for 62.3% of
all val tokens, so the byte denominator is inflated by ~16.71% and the
reported BPB is correspondingly deflated.

Why we can correct without re-running the model: the cross-entropy
numerator is independent of the LUT. `bpb = (loss × N_tokens) / (ln(2) ×
byte_count)`. Multiply both sides by the `buggy_bytes / canonical_bytes`
ratio and you recover the canonical BPB from the buggy reported value.

---

## Methodology (full version: `audit/methodology.md`)

For each PR:

1. `git fetch upstream pull/<N>/head:pr-<N>` and check it out.
2. Find the `train_gpt.py` under `records/track_10min_16mb/<latest-dated-dir>/`.
3. Run `scripts/canonical_rescore.py` against that script + the SP8192
   tokenizer + the fineweb_val shard.
4. Tool returns:
   * `lut_status`: `CORRECT` / `BUGGY` / `OBFUSCATED` / `UNKNOWN`
   * `inflation_ratio`: `1.0` for CORRECT, computed buggy/canonical for
     BUGGY (~`1.1671` on SP8192), `null` otherwise.
   * `inferred_canonical_bpb`: `reported_bpb × inflation_ratio` if both are
     known; `null` otherwise.
   * `passes_merged_sota_threshold`: boolean, threshold default 1.0738 (one
     record-class margin under the merged-SOTA reference).

Hardware parity is anchored by exp_001: a verbatim PR #1727 reproduction on
8×H100 SXM, seed 1337, val_bpb = 1.07431, within 0.00214 of the reported
3-seed mean of 1.07217 — confirming our toolchain (torch 2.8.0+cu128) sees
the same numbers as upstream and that the audit's analytic correction can
be trusted. See `experiments/exp_001/analysis.md`.

---

## Scope and limitations

What "LUT-verified CORRECT" does and does not mean:

* **Does mean** the `build_sentencepiece_luts` function in the PR's
  `train_gpt.py` uses the canonical `len(piece.encode("utf-8"))` pattern
  (no `+1` for leading-space tokens) and is not wrapped in
  `lzma.decompress(base64.b85decode(...))`.
* **Does not imply** the model artifact the PR ships achieves its reported
  BPB. The tool verifies the LUT only; the cross-entropy numerator of BPB
  is taken as given.
* **Does not imply** that `eval_val_sliding` itself is canonical. A PR
  that modified the eval loop would not be caught by this tool. We assume
  upstream-faithful eval logic.
* **Does not rule out** other measurement irregularities — modified val
  shards, different tokenizers, custom BPB definitions. Independent
  reproduction remains the gold standard for a contested record.

What the OBFUSCATED verdict does and does not mean:

* **Does mean** the tool's static regex found a `*.decompress(*.b85decode(...))`
  chain and could not locate a readable `build_sentencepiece_luts`
  implementation.
* **Does not mean** the PR is buggy. The OBFUSCATED verdict is neutral;
  verifying the LUT inside the wrapper requires sandbox execution, which
  is out of scope for this audit.

PR #1795's reported 1.01252 leads the LUT-verified frontier with a -0.030 BPB margin over PR #1735, but the gain comes from a byte-level PPM mixture on top of a canonical NN base, not from a different LUT or eval shape. PR #1795's NN-only mean (1.09764) tracks @clarkkev's 2026-04-01 record (1.09785) within seed noise, so the audit verifies the byte-count denominator only; the mixture's gate legality was verified separately by @nprime06's review on PR #1795 itself. PR #1735's **0.021 BPB lead** over the next-best CORRECT result (#1779 at
1.06421) is sufficiently large that independent reproduction is warranted
before treating it as authoritative for record-class comparisons. The tool
verifies only the LUT, not the full training pipeline; a wide gap like
this could be real or could reflect some other path that the tool does not
inspect. The frontier PR #1735 reading is "LUT-verified, reproduction
pending", not "verified as the true top".

---

## Tool usage

```bash
python scripts/canonical_rescore.py \
    --train-script <path-to-PR-train_gpt.py> \
    --tokenizer    data/tokenizers/fineweb_8192_bpe.model \
    --val-data     'data/datasets/fineweb10B_sp8192/fineweb_val_*.bin' \
    --reported-bpb 1.02840 \
    --pr-number    1758
```

Output (JSON to stdout / `--output`):

```json
{
  "pr_number": 1758,
  "script_path": "...",
  "lut_status": "OBFUSCATED",
  "inflation_ratio": null,
  "inferred_canonical_bpb": null,
  "passes_merged_sota_threshold": null,
  "notes": "Code is lzma/b85-obfuscated; LUT cannot be verified statically."
}
```

For a CORRECT script the output looks like:

```json
{
  "pr_number": 1735,
  "lut_status": "CORRECT",
  "inflation_ratio": 1.0,
  "inferred_canonical_bpb": 1.0429,
  "passes_merged_sota_threshold": true
}
```

For a BUGGY script the output reports the exact byte counts, the inflation
ratio, and the corrected BPB.

Tests covering CORRECT (PR #1727), BUGGY (four synthetic fixtures — one
per bug variant plus the triple-bug case), OBFUSCATED (both inline-`exec`
and `runpy`-style wrappers), UNKNOWN, the three scoring-mode variants,
and the full end-to-end rescore are in `tests/test_canonical_rescore.py`
(20 tests, all green).

---

## Results (full version: `audit/results.md` and `audit/corrected_leaderboard.md`)

| Rank | PR | Author | Reported | LUT status | LUT-verified† | Canonical BPB |
|------|----|--------|---------|-----|:---:|-----------|
| 1 | #1795 | OE-GOD | 1.01252 | CORRECT | yes | **1.01252** | (added 2026-04-24, supersedes #1785) |
| — | #1785 | OE-GOD | 1.01925 | OBFUSCATED | no | superseded by #1795 |
| 2 | #1758 | kilojoules | 1.02840 | OBFUSCATED | no | unverified |
| 3 | #1738 | alertcat | 1.03540 | OBFUSCATED | no | unverified |
| 4 | #1735 | AjAnubolu | 1.04290 | CORRECT | yes | **1.04290** |
| 5 | #1779 | leon2k2k2k | 1.06421 | CORRECT | yes | 1.06421 |
| 6 | #1769 | dexhunter | 1.06453 | CORRECT | yes | 1.06453 |
| 7 | #1756 | romeerp | 1.06505 | CORRECT | yes | 1.06505 |
| 8 | #1771 | bigbag | 1.06513 | OBFUSCATED | no | unverified |
| 9 | #1736 | dexhunter | 1.06549 | CORRECT | yes | 1.06549 |
| 10 | #1784 | renqianluo | 1.07081 | CORRECT | yes | 1.07081 |

† "LUT-verified" means the tool statically confirmed a canonical
`build_sentencepiece_luts`. Under the v2 (three-variant) classifier
this requires all three canonical properties — `leading_space_noplus`,
`byte_token_one`, and `boundary_predicate_full` — to match. This is
necessary but not sufficient for a trustworthy BPB — see "Scope and
limitations" above. The v2 classifier reproduces the same
classification as v1 on every row of this table; see
`audit/changelog_v2.md` for the side-by-side.

**LUT-verified frontier: PR #1795 (OE-GOD) at reported BPB 1.01252** as of 2026-04-24 (audit run on commit `cb5ad95`); previous frontier PR #1735 (AjAnubolu) at 1.04290 remains LUT-verified,
with PR #1779 the next-best LUT-verified entry at 1.06421. The 0.021 BPB
gap is large enough that independent reproduction is warranted before
treating #1735 as the authoritative record.

Four PRs in the top 10 (#1785, #1758, #1738, #1771) returned OBFUSCATED
and could not be statically audited. We do not claim these are buggy; we
state the observation neutrally: the three lowest reported BPBs on the
current top-10 snapshot are all in obfuscated code, and the only sub-1.05
submission with a self-disclosed LUT classification (yahya010's PR #1734,
1.0108 → ~1.1873) was buggy. This is a pattern, not a causal claim. A
naive application of the 1.1671 ratio *if* the bug were present would
yield #1785 → ~1.190, #1758 → ~1.200, #1738 → ~1.208, and #1771 → ~1.243,
but this arithmetic is only meaningful if the obfuscated LUTs actually
match the #1698 lineage, which we have not verified and cannot verify
without sandbox execution of the wrapped code.

---

## Attribution

Verbatim from the PR #1734 closure comment by **yahya010**, 2026-04-19:

> "build_sentencepiece_luts bakes +1 into LUT for leading-space tokens,
> then eval_val_sliding adds +1 again at eval. Buggy code overcounts bytes
> by 17.46% vs canonical sp.decode_ids().encode('utf-8'). Reported
> val_bpb=1.0108 corresponds to canonical val_bpb≈1.1873..."

yahya010's quoted ratio (1.1746) was computed against his own #1734 LUT,
which has two additional byte-counting differences from the #1727-style
LUT: byte tokens are sized by `len("<0xXX>".encode("utf-8"))` (6 bytes)
rather than 1, and `sp.is_unused` tokens are not treated as boundary.
Our tool's three `--scoring-mode` variants converge to 1.1671 on SP8192
fineweb val when applied to the #1727-style LUT shape. Running yahya's
exact LUT (lines 206-219 of `train_gdn_7k.py`) against the same val
stream and applying the same canonical/buggy formulation as the audit
tool gives **1.1655**, not the quoted 1.1746. The 0.77% gap is in the
opposite direction from what canonical-vs-buggy alone would predict.
Empirical run 4 has since shown the gap is invariant to eval pipeline
windowing parameters (seq_len ∈ {1024, 2048}, stride ∈ {64, 1024}),
ruling out the eval pipeline as the gap's source. The gap lives in
tokenizer or val-shard state we do not have access to. Both reported numbers describe
the same underlying defect (leading-space bytes baked into the LUT and
re-added at eval); the residual numerical disagreement remains
unresolved. Full analysis and the empirical reproduction in
`audit/empirical_validation/run3_summary.md`. Methodology in
`audit/methodology.md` §4, per-property detection design in §5.

This audit extends yahya010's finding by:

1. Publishing a tool anyone can run without reproducing on GPU.
2. Applying it to the full set of currently-open top-10 PRs.
3. Documenting the scoring-strategy sensitivity explicitly so the two
   quoted ratios are no longer a source of confusion.
4. Detecting the two *additional* LUT-construction bugs in yahya's
   own train_gdn_7k.py (byte-token sizing, missing `is_unused` in the
   boundary predicate) as explicitly-named deviations in the tool's
   JSON output, so future submissions can be checked for each variant
   individually.

---

## Framing

We do not request any PR be re-classified or closed. The competition
maintainers and authors are best positioned to decide whether obfuscated
submissions are eligible for record consideration. Our contribution is:

1. **A reusable tool** (`scripts/canonical_rescore.py`) that any submitter
   can run before filing — including a regex check that catches the buggy
   `+1` pattern in seconds.
2. **A clean methodology document** (`audit/methodology.md`) defining
   canonical BPB rigorously enough that disagreements about "what is
   canonical" can be resolved by code rather than discussion.
3. **A snapshot leaderboard** (`audit/corrected_leaderboard.md`,
   `audit/results.md`) that distinguishes *verified* canonical BPB from
   *reported* BPB, so reviewers do not have to re-derive that distinction
   per-PR.

The LUT-verified frontier (PR #1795 at canonical 1.01252 as of 2026-04-24; previously PR #1735 at 1.04290, both leading the
cluster around 1.064-1.071) is the cleanest statement we can make from
static inspection alone. Whether the 0.021 BPB gap between #1735 and the
next-best LUT-verified entry reflects a genuine capability step-change
or a reporting artefact is outside the scope of this audit; we flag it as
"reproduction-pending" rather than "verified record".
