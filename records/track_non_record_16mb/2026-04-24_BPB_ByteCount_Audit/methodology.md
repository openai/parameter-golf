# Methodology — Canonical BPB Byte-Count Audit

This document is the standalone reference for what `canonical BPB` means in
this audit, how the inflation ratio is derived, and what the sliding-window
scored-token subset is. It is the source you cite in disputes; the
implementation in `scripts/canonical_rescore.py` is its operational
realization.

---

## 1. Canonical BPB definition

```
canonical_bpb = (mean_cross_entropy_loss_in_nats / ln(2)) / canonical_bytes_per_token
```

where `canonical_bytes_per_token` is computed over the same scored-token
subset that the eval loop uses (see §3), and the per-token byte count
follows the rule:

```
bytes_for_token(y, prev_x) =
    base_bytes(y)
    + (has_leading_space(y) AND NOT is_boundary_token(prev_x))
```

with:

* `base_bytes(t) = len(sp.id_to_piece(t).strip("▁").encode("utf-8"))` for
  non-boundary, non-byte tokens.
* `base_bytes(t) = 1` for SentencePiece byte tokens
  (`sp.is_byte(t)` true).
* `base_bytes(t) = 0` for boundary tokens (`sp.is_control(t)`,
  `sp.is_unknown(t)`, `sp.is_unused(t)`).
* `has_leading_space(t) = sp.id_to_piece(t).startswith("▁")`.
* `is_boundary_token(t) = sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t)`.

This rule is what the **upstream** `eval_val_sliding` in PR #1727
(`train_gpt.py` lines 2117-2150) actually computes. The audit anchors
"canonical" to the upstream eval logic — not to a separate reference
implementation — so the corrected number is what *anyone running the
upstream eval with the corrected LUT would measure*.

---

## 2. The bug, in code

**Correct LUT** (PR #1727, `build_sentencepiece_luts`, line ~196):

```python
for token_id in range(sp_vocab_size):
    if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
        continue
    is_boundary_token_np[token_id] = False
    if sp.is_byte(token_id):
        base_bytes_np[token_id] = 1
        continue
    piece = sp.id_to_piece(token_id)
    if piece.startswith("▁"):
        has_leading_space_np[token_id] = True
        piece = piece[1:]
    base_bytes_np[token_id] = len(piece.encode("utf-8"))   # <-- no +1
```

**Buggy LUT** (#1698 lineage; reproduced in the audit fixture
`tests/fixtures/buggy_train_gpt.py` and self-confirmed by yahya010 in PR
#1734 closure):

```python
    base_bytes_np[token_id] = len(piece.encode("utf-8")) + 1   # <-- +1 baked in
```

Both versions then run an *identical* `eval_val_sliding`, which adds
`(has_leading_space[y] & ~is_boundary_token[x_prev])`. Hence each
leading-space scored token receives one extra byte of credit beyond the
canonical eval-bytes count.

---

## 3. Sliding-window scored-token subset

`eval_val_sliding` slides a window of `seq_len=2048` tokens with a stride
of `64` over the validation tokens. Each window's "scored" range is the
last `seq_len - context_size = stride = 64` tokens, except the first window
(`ws=0`) which scores all `seq_len` tokens. The window at position `ws` is
included iff `ws + context_size < total_tokens`.

Across all included windows, the scored y-positions form a contiguous
tile of `val_tokens[1 : total_tokens + 1]`, with the corresponding x-prev
positions forming `val_tokens[0 : total_tokens]`. This means the byte sum
collapses to two array reductions:

```python
y = val_tokens[1 : total_tokens + 1]
x = val_tokens[0 : total_tokens]
canonical_bytes = base_bytes[y].sum() + (has_leading_space[y] & ~is_boundary[x]).sum()
buggy_bytes     = canonical_bytes + has_leading_space[y].sum()
inflation_ratio = buggy_bytes / canonical_bytes
```

The `+ has_leading_space[y].sum()` is exact: the buggy LUT adds `+1` for
every leading-space token regardless of whether the prev token is a
boundary. The eval still adds the gated `+1`, so the difference per
leading-space token is exactly one — accumulated across the scored y subset
gives the byte-total delta.

On SP8192 fineweb val (40,540,803 raw val tokens, 633,420 windows of
`seq_len=2048, stride=64`):

* `canonical_byte_count` = 151,080,891
* `buggy_byte_count`     = 176,332,748
* `leading_space_token_count` = 25,251,857
* `inflation_ratio` = 1.16713

These numbers are exact and reproducible by running
`scripts/canonical_rescore.py` against any `train_gpt.py` plus the SP8192
tokenizer + val data.

---

## 4. Inflation ratio is sensitive to scoring strategy

The inflation ratio `buggy / canonical` depends on which y-tokens are
scored and whether the eval-time boundary mask is applied. The tool
supports three modes via `--scoring-mode`:

| Mode | y-tokens scored | `boundary_mask` | Models what |
|------|-----------------|-----------------|-------------|
| `sliding-window-boundary-masked` (default) | Sliding-window tile (`seq_len=2048`, `stride=64`, last window trimmed) | `~is_boundary[x_prev]` | What PR #1727's `eval_val_sliding` actually computes — the number the buggy eval pipeline reports |
| `all-tokens-boundary-masked` | Flat `val_tokens[1:N]` slice | `~is_boundary[x_prev]` | Generic "score every token, gate by prev" computation |
| `all-tokens-no-mask` | Flat `val_tokens[1:N]` slice | `1` (all ones) | Naive "every leading-space adds one byte" no-gate computation |

In all three modes `buggy − canonical = sum(has_leading_space[y])` (the
LUT adds +1 per leading-space token regardless of gate), so the three
ratios differ only through the canonical denominator.

**Empirical values on SP8192 fineweb val (40,540,803 tokens):**

| Mode | canonical bytes | buggy bytes | ratio |
|------|-----------------|-------------|-------|
| `sliding-window-boundary-masked` | 151,080,891 | 176,332,748 | **1.1671** |
| `all-tokens-boundary-masked` | 151,080,891 | 176,332,748 | **1.1671** |
| `all-tokens-no-mask` | 151,080,891 | 176,332,748 | **1.1671** |

The three numbers coincide on this validation stream for two reasons:

1. The sliding windows with the last-window-trimmed logic tile the full
   `val_tokens[1:N]` span (`last_end = total_tokens`), so the sliding-window
   and all-tokens subsets are identical.
2. `is_boundary[x_prev]` is identically zero over this stream — the
   fineweb val tokens never contain a control/unknown/unused SentencePiece
   token as a predecessor. The boundary mask is therefore a no-op on this
   data.

### Why yahya010's 1.1746 differs by 0.75%

yahya010's PR #1734 closure quoted a ratio of 1.1746. The three variants
above converge to 1.1671 on the same val data. The residual 0.75% gap is
**not** a scoring-strategy artifact; it comes from the LUT used in PR
#1734 itself, which has two additional differences from the canonical
`build_sentencepiece_luts` in PR #1727:

* **Byte tokens (`sp.is_byte`).** Canonical sets `base_bytes = 1`.
  PR #1734's `train_gdn_7k.py:213` instead uses
  `base_bytes[i] = len(piece.encode("utf-8"))`, which for a byte piece
  `"<0x00>"` evaluates to 6 rather than 1. There are ~269,000 byte tokens
  in val, contributing ~1.35M extra bytes to the buggy numerator.
* **`is_unused` gating.** Canonical treats `sp.is_unused` tokens as
  boundary (zero byte contribution). PR #1734's boundary predicate uses
  only `sp.is_control | sp.is_unknown`, so any `is_unused` tokens in val
  (or as predecessors) are scored normally.

Running yahya's exact LUT (lines 206-219 of his `train_gdn_7k.py`)
against the same val stream, with the same canonical/buggy formulation
the audit tool applies, gives:
`canonical = 152,576,975`, `buggy = 177,828,831`, ratio = **1.1655**
(sliding-window-boundary-masked; the all-tokens variant gives the same
ratio to 8 decimal places). This is 0.77% *below* yahya's quoted 1.1746,
in the opposite direction one would expect if his quote were a clean
buggy-vs-canonical computation on the same data. The byte-token bug in
his LUT inflates his canonical denominator by 1,346,100 bytes relative
to the #1727 LUT, which decreases the ratio rather than increasing it.

The discrepancy between yahya's quoted 1.1746 and our reproduction's
1.1655 has been bounded by empirical run 4. We replicated yahya's exact
`eval_val_sliding` scoring formula and tested three windowing
configurations: seq_len=2048/stride=64 (audit default), seq_len=1024/
stride=64 (yahya's code default per `train_gdn_7k.py:69`), and
seq_len=1024/stride=1024 (no overlap). All three produce yahya's ratio
to within 1.6e-6 of each other. The 0.77% gap is invariant to eval
pipeline windowing parameters. By elimination across runs 1-4, the gap
must live in tokenizer or val-shard state that we cannot replicate
without yahya's exact disclosure-time data — most likely the SP1024
tokenizer his code defaults to (line 58), a different val shard, or a
hand-derived estimate in PR #1734 itself. This narrows the unknown
from "the gap is unexplained" to "the gap is bounded to data state,
not pipeline structure."

Empirical reproductions at `audit/empirical_validation/run3_yahya_full_lut.py`
and `audit/empirical_validation/run4_seq_len_1024.py`.

## Structural deviations vs empirical inflations

The classifier flags structural deviations from canonical. Empirical run 5
distinguishes these from observable inflation on the audited val:

| Bug family | Structural deviation? | Empirical Δratio on SP8192 fineweb val |
|---|---|---|
| Bug A — leading_space_plus_one | Yes | 0.000000 |
| Bug B — byte_token_wrong_size | Yes | -0.001476 |
| Bug C — missing_is_unused | Yes | 0.000000 |

Bug A is empirically a no-op on this val because leading-space y-tokens never
follow boundary x-tokens (run 1.5 finding); the LUT-baked +1 produces the
same byte count as the eval-time +1. Bug C is empirically a no-op because
the SP8192 vocabulary contains zero `sp.is_unused` tokens. Bug B is the only
deviation that produces measurable inflation, and it shifts the ratio
*downward* (canonical 1.1671 → yahya 1.1655) because it inflates the
canonical denominator by 1,346,100 bytes.

This distinction matters because:

* A structurally-buggy LUT may produce zero inflation on a particular val while still being structurally wrong. Correcting it is appropriate because it would inflate on a different val (e.g. one where leading-space tokens follow boundary tokens, or a vocab with `sp.is_unused` populated).
* Yahya's quoted 1.1746 is 0.0089 *above* canonical 1.1671. No combination of his three LUT bugs can produce a ratio above canonical's on SP8192 — Bug B shrinks the ratio, Bugs A and C are no-ops. The 0.77% gap between his quoted 1.1746 and our reproduction's 1.1655 cannot live in his LUT structure on this val. By corollary with run 4, it lives in tokenizer/val state we cannot replicate.

Empirical decomposition at `audit/empirical_validation/run5_bug_decomposition.py`.

### Which variant should a reviewer cite?

* To characterize "what does PR #1727's eval pipeline overcount?" — use
  `sliding-window-boundary-masked`. The tool defaults to this because it
  is the ratio that corresponds to the reported BPB of the buggy
  submissions we are correcting.
* To characterize "how much does a naive count-every-leading-space
  method differ from a sp.decode-based ground truth?" — use
  `all-tokens-no-mask`. On SP8192 fineweb val this is numerically the
  same as the default.
* yahya's 1.1746 is a *different* ratio: it is his own buggy-LUT output
  divided by his own sp.decode ground truth. It characterizes the same
  bug (baked +1 for leading-space tokens, double-counted at eval), but
  with additional LUT-construction differences folded in. Both ratios
  point to the same underlying defect in the #1698 lineage; the numerical
  correction we apply to any specific PR's BPB depends on *that PR's*
  LUT and eval scoring.

Bottom line: both numbers are valid characterizations of the same bug.
The one the static tool reports (1.1671) is the one that applies to the
current buggy-but-not-obfuscated PRs because they inherited the #1727
LUT shape; yahya's 1.1746 applies to his own #1734 where the LUT was
additionally idiosyncratic.

---

## 5. Three-variant LUT classifier (v2)

The classifier in `scripts/canonical_rescore.py` was extended from a
single-bug detector (the +1 bake, §2 above) to a three-variant detector
that also checks the byte-token branch and the boundary predicate. All
three properties must match canonical for a PR to classify as `CORRECT`.

### Canonical properties

| # | Name in tool | Canonical form | Deviation |
|---|--------------|----------------|-----------|
| P1 | `leading_space_plus_one` | `base_bytes[t] = len(piece.encode("utf-8"))` with no +1 after stripping the `▁` | `... + 1` baked into the LUT (the #1698 bug, §2) |
| P2 | `byte_token_wrong_size` | `if sp.is_byte(t): base_bytes[t] = 1` (literal 1) | `sp.is_byte` branch assigns something other than 1, e.g. `len(piece.encode("utf-8"))` (= 6 for `"<0xXX>"`) |
| P3 | `missing_is_unused` | Boundary predicate is `sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t)` | Predicate has `is_control` + `is_unknown` but not `is_unused` |

### Detector approach (regex / window)

* **P1** uses two regexes over the full source: `_LEADING_PLUS1_RE`
  matches `base_bytes[...] = len(<expr>.encode("utf-8")) + 1` (captures
  both `piece.encode(...)` and `piece[1:].encode(...)` forms);
  `_LEADING_NOPLUS_RE` matches the same assignment without the trailing
  `+ 1`. One matches ⟹ status; neither matches ⟹ `INDETERMINATE`.
* **P2** locates `if sp.is_byte(<id>):` lines then scans the next 1-6
  indented lines for a `base_bytes[...] = <rhs>` assignment. `rhs == "1"`
  ⟹ `MATCHES_CANONICAL`; any other RHS ⟹ `DEVIATES`; no branch located ⟹
  `INDETERMINATE`.
* **P3** scans every `is_control(` call site, grabs a ±120-char window,
  and checks whether `is_unknown(` and `is_unused(` (both required to
  include the opening paren so comment text does not trigger false
  positives) appear within the window. Both present ⟹
  `MATCHES_CANONICAL`; only `is_unknown(` present ⟹ `DEVIATES`; no
  `is_control(` call at all ⟹ `INDETERMINATE`.

### Classification rules

* Any property DEVIATES ⟹ `BUGGY`. The JSON field `lut_bug_detections`
  lists the deviating property names.
* All three properties MATCH ⟹ `CORRECT`.
* No deviations, not all three matching, obfuscation regex matches ⟹
  `OBFUSCATED`.
* Otherwise ⟹ `UNKNOWN`.

### Design note: DEVIATES vs INDETERMINATE

The P2 and P3 detectors return `DEVIATES` only when the relevant
construct is *present but wrong*. Absence of the construct — e.g. a
function that handles byte tokens via the default path without an
explicit `sp.is_byte` check — yields `INDETERMINATE`, not `DEVIATES`. This
is deliberate: a no-`sp.is_byte`-branch function IS functionally buggy
for byte tokens (scoring them as UTF-8 length of `"<0xXX>"` rather than
1), but a static detector that inferred "buggy" from "absent" would
false-positive on any pedagogical LUT fragment that happens to elide
rare cases. The conservative rule produces a classifier that can miss a
bug variant it does not explicitly see, but will not false-accuse a
simpler script.

**Consequence for yahya010's PR #1734 `train_gdn_7k.py`.** His function
has no `sp.is_byte` branch (byte tokens go through the default path and
are sized at 6 rather than 1), and his boundary predicate lacks
`sp.is_unused`. The v2 classifier reports:

```
lut_status: BUGGY
lut_bug_detections: ['leading_space_plus_one', 'missing_is_unused']
```

The byte-token bug is implicit (falls through the default path) rather
than explicit, so the P2 detector returns `INDETERMINATE` rather than
`DEVIATES` — matching the design rule above. The classification is still
correct (BUGGY), with two of the three deviations explicitly named.
yahya010's own 1.1746 ratio combines all three bug effects against his
own decoded-stream ground truth; the tool's default 1.1671 ratio
characterizes only the +1 component. See §4 for the detailed numerical
comparison.

### Conservative arithmetic

The `inflation_ratio` field in the tool's JSON output is computed by the
val-data math in §3 — that math accounts only for the
`leading_space_plus_one` effect. For BUGGY PRs with additional
deviations the `inflation_ratio_includes` JSON field explicitly lists
which bugs the arithmetic covers (currently only
`["leading_space_plus_one"]`). An arithmetic correction for
`byte_token_wrong_size` or `missing_is_unused` would require rebuilding
the PR's specific LUT against the val stream — still a no-GPU static
operation, but not a simple ratio multiplication.

---

## 6. Scope and what this audit does **not** claim

* **Cross-entropy is treated as given.** We do not re-run any model. The
  arithmetic correction `canonical_bpb = reported_bpb × inflation_ratio`
  applies only when (a) the buggy LUT is the source of byte mismatch and
  (b) the model's loss-in-nats was correctly measured by the submitter. If
  a PR has a separate cross-entropy bug, this audit does not catch it.
* **OBFUSCATED scripts are not classified.** Single-line
  `lzma.decompress(base64.b85decode(...))` wrappers — whether executed
  inline via `exec` or via `runpy` after assigning to a local — are flagged
  as `OBFUSCATED`. The static tool cannot determine the LUT status without
  decoding and executing the wrapped code, which is out of scope for a
  no-code-execution audit.
* **No claim is made that any specific obfuscated PR is buggy.** The
  closest precedent is yahya010's own PR #1734 (obfuscated, reported
  1.0108, self-disclosed as canonical ~1.1873). Other obfuscated PRs may
  use the correct LUT internally; we simply cannot verify until the
  authors publish the de-obfuscated source.
* **Per-PR variance is one seed.** Hardware parity is anchored by exp_001
  (one seed within tolerance of the upstream 3-seed mean). For a sharper
  check we would need at least two more seeds; the current evidence is
  sufficient for the analytic correction but not for a record-class
  comparative claim.

---

## 7. Why the static-only design is correct here

The byte-count denominator of BPB depends only on the tokenizer and the
val-token sequence. It does *not* depend on the model checkpoint, the
training data, the optimizer, or the random seed. So the canonical /
buggy ratio is the **same number** for every submission that uses the
SP8192 tokenizer + the standard fineweb val shard, regardless of model
architecture. We compute it once (`1.1671`) and apply it as a multiplier
to any reported BPB whose source `train_gpt.py` is statically classified
as BUGGY. This is a faster, cheaper, and more reliable audit than
reproducing each PR on a GPU — and it eliminates any "your hardware is
different" objection because no hardware is involved beyond the static
inspection.

---

## 8. Tool reference

```bash
python scripts/canonical_rescore.py \
    --train-script <path> \
    --tokenizer    <sp.model> \
    --val-data     '<glob-of-val-shards>' \
    [--seq-len 2048] [--stride 64] \
    [--reported-bpb FLOAT] \
    [--pr-number INT] \
    [--threshold 1.0738] \
    [--output JSON_PATH]
```

JSON output schema:

| Field | Meaning |
|---|---|
| `lut_status` | `CORRECT` / `BUGGY` / `OBFUSCATED` / `UNKNOWN` |
| `lut_bug_detections` | list of deviation names — subset of `leading_space_plus_one`, `byte_token_wrong_size`, `missing_is_unused` (empty for CORRECT) |
| `detected_bugs_description` | human-readable summary of the named deviations |
| `inflation_ratio_includes` | which bugs the arithmetic ratio accounts for (currently just `["leading_space_plus_one"]` when applicable) |
| `inflation_ratio` | `1.0` for CORRECT, computed for BUGGY with the +1 bug, `null` otherwise |
| `computed_inflation_ratio` | always the raw `buggy/canonical` from the val data (for the +1 effect) |
| `inferred_canonical_bpb` | `reported_bpb × inflation_ratio` if both known; null if the +1 arithmetic doesn't apply (e.g. a non-P1 BUGGY PR) |
| `passes_merged_sota_threshold` | inferred_canonical_bpb ≤ threshold |
| `canonical_byte_count`, `buggy_byte_count` | totals on the scored y-subset |
| `leading_space_token_count`, `scored_token_count`, `num_windows` | sanity counters |
| `notes` | human-readable caveats (e.g. "OBFUSCATED — cannot verify statically" or multi-bug conservative-ratio warning) |

Tests in `tests/test_canonical_rescore.py` (20 tests) exercise CORRECT,
BUGGY, OBFUSCATED (both `exec(...)` and runpy patterns), UNKNOWN, the
synthetic byte-counting math, the three scoring-mode variants, the
three-variant deviation detectors (single-bug and triple-bug fixtures
under `tests/fixtures/buggy_*.py`), and the end-to-end rescore against
PR #1727 and the buggy fixture.
