---
name: frontier-scan
description: Scan open PRs on openai/parameter-golf for new record-track submissions, classify their legitimacy, surface novel levers, write a dated diary entry, and update the dependency tree in research/frontier-map.md. Incremental — only processes PRs new or updated since the last scan.
---

# Frontier-scan

You are running an incremental frontier scan for the Parameter Golf record-track push.

## What this skill does

1. Load prior state from `research/frontier-state.json` (last scan timestamp + per-PR verdict snapshot).
2. Fetch open PRs on `openai/parameter-golf` updated since last scan.
3. Classify each new/updated PR by legitimacy, novelty, and lineage.
4. Write a dated diary entry to `diary/YYYY-MM-DD-frontier-scan[-N].md`.
5. Append novel levers to `research/ideas/` if they're genuinely new and not already specced.
6. Update the dependency tree in `research/frontier-map.md`.
7. Save updated state to `research/frontier-state.json`.

## Step 1 — Load state

Read `research/frontier-state.json`. If it doesn't exist, treat last scan as
`1970-01-01T00:00:00Z` (full scan).

State schema:
```json
{
  "last_scan_utc": "2026-04-20T05:47:00Z",
  "prs": {
    "1736": {"verdict": "tokenizer-disputed", "claimed_bpb": 1.06549, "sha": "abc123",
             "builds_on": "1626", "trunk": "A"},
    "1748": {"verdict": "other", "claimed_bpb": null, "sha": "def456",
             "builds_on": "unknown", "trunk": "unknown"}
  }
}
```

## Step 2 — Fetch open PRs

```bash
gh pr list --repo openai/parameter-golf --state open \
  --limit 100 \
  --json number,title,author,updatedAt,headRefOid \
  --jq 'sort_by(.updatedAt) | reverse | .[]'
```

Partition results:
- **New** — PR number not in state.prs
- **Updated** — number in state.prs but headRefOid changed
- **Unchanged** — in state, same SHA → skip (no deep read needed)

Also note any PRs that were in state but are now closed.

## Step 3 — Classify each new/updated PR

For each PR in the new/updated set, run a **parallel subagent** (one per PR) that:

1. Fetches the PR body: `gh pr view <N> --repo openai/parameter-golf --json number,title,author,body`
2. For any PR claiming a competitive bpb or containing non-trivial code, also fetches the diff:
   `gh pr diff <N> --repo openai/parameter-golf --patch`
3. Returns a structured verdict:

```
PR: <number>
author: <login>
claimed_bpb: <float or null>
track: record | non-record | unknown
verdict: clean | tokenizer-disputed | prequant-ttt-disputed | byte-bug-suspect | broken | other
legitimacy: legal | likely-legal | disputed | likely-illegal | broken | unknown
novel_lever: <one sentence describing the key idea, or "none">
lever_class: architecture | optimizer | training-loss | lr-schedule | eval-time | serialization | tokenizer | none
absorbed_by_ttt: yes | no | unknown
already_in_baseline: yes | no
already_specced: yes | no  (check HEURISTICS.md already-specced table)
actionable: yes | watch | no
builds_on: <PR number(s) the author explicitly cites, or "unknown">
trunk: A | B | C | D | new | unknown
reasoning: <2-3 sentences citing specific heuristics from HEURISTICS.md>
```

**How to extract `builds_on`:** scan the PR body for phrases like "builds on #XXXX",
"starts from #XXXX", "based on #XXXX", "off #XXXX", "extends #XXXX", "stacks on #XXXX".
Extract the cited PR number(s). If none found, use `"unknown"`.

**How to determine `trunk`:**
- **A**: `builds_on` traces back to #1523 or #1493 via the Trunk A lineage
  (#1530, #1586, #1626, #1667, #1736, #1729, etc.)
- **B**: traces back to #1700 or #1727 (multi-phase global-SGD TTT)
- **C**: traces back to #1735 (parallel pre-quant TTT)
- **D**: diff contains `from fla.` import (GatedDeltaNet / FLA)
- **new**: doesn't clearly fit an existing trunk
- **unknown**: couldn't determine

**Apply heuristics from `.claude/skills/frontier-scan/HEURISTICS.md`.**

Key checks:
- Is the claimed lever already in #1736? (see HEURISTICS.md already-in-#1736 list)
- Is the claimed lever already specced? (see HEURISTICS.md already-specced table)
- Does any `from fla.` import appear in the diff? → byte-bug-suspect
- Does the PR claim >0.005 bpb from a TTT-related lever? → check physics ceiling
- Does the tokenizer produce lossless roundtrip? → tokenizer-disputed (likely-legal) vs likely-illegal
- Is the claimed bpb below our baseline 1.06549? → actionable

## Step 4 — Write scan report

File: `pr-analysis/frontier-scan/YYYY-MM-DD.md` (append `-2`, `-3` etc. if that date
already has a scan file for the same day).

Structure:
```markdown
# YYYY-MM-DD frontier scan (incremental / full)

**Baseline:** PR #1736 @ 1.06549. **Last scan:** <prior timestamp UTC>.
**PRs processed:** <N new, M updated, K closed-since-last>.

## New record-track PRs

<table: number | author | claimed_bpb | verdict | legitimacy | novel lever>

## Status-changed PRs

<table: any PR whose verdict changed vs state>

## Closed since last scan

<list>

## Current leaderboard by category

| Category | Best PR | claimed_bpb | verdict |
|---|---|---|---|
| Clean | #XXXX | 1.XXXX | clean |
| Tokenizer-disputed | #1736 | 1.06549 | tokenizer-disputed / likely-legal |
| Pre-quant-TTT-disputed | #1738 | 1.0354 | prequant-ttt-disputed / likely-illegal |

## Novel levers surfaced

<For each actionable or watch lever not already specced: one paragraph.
Include lever_class, estimated Δ-bucket, feasibility, and whether it's
absorbed-by-ttt or survives.>

## Actionable deltas

<Only if any clean PR is below 1.06549: flag it prominently with reasoning.>
```

## Step 5 — Append ideas (if novel levers found)

For each novel lever that is:
- Not already in #1736 baseline
- Not already specced (check HEURISTICS.md)
- Lever class is architecture, eval-time, training-loss, or lr-schedule
- Legitimacy is legal or likely-legal

Write a brief idea file to `research/ideas/<slug>.md` with:
- What the lever is (concrete mechanism, not marketing)
- Which PR it came from + author credibility
- Estimated Δ-bucket and feasibility
- Key risks (TTT absorption? artifact budget? compliance?)

Do NOT write idea files for:
- Quant-side levers (likely TTT-absorbed per our empirical results)
- Levers in the banned/already-in-baseline/already-specced lists
- Anything with legitimacy = likely-illegal or broken

## Step 6 — Update frontier-map.md

For each **new** PR (not previously in state) where `track: record` and
`verdict != other`, update `research/frontier-map.md`:

### 6a — Determine insertion point

- `builds_on` is a known PR number → find that PR's block in the ASCII tree
  and append a new branch after its last child entry.
- `trunk` is known but `builds_on` is unknown → append at the bottom of that
  trunk's section with note `(lineage unclear)`.
- `trunk: new` → add a new `### Trunk E — <short description>` section (or
  next available letter after D).
- `trunk: unknown` AND `builds_on: unknown` → skip the map update; log this
  PR in the diary under an **"Unclassified lineage"** subsection instead.

**Skip map update entirely for:** `track: non-record`, `verdict: other`,
unchanged PRs (same SHA as in state).

### 6b — Format the new entry

Use the same ASCII tree style as existing entries:

```
 ├─ #XXXX  open  1.XXXX  <author>   <short description>  [BADGE]
```

Verdict badges (append if applicable):
- `[DISPUTED: tokenizer]` — tokenizer-disputed
- `[DISPUTED: pre-quant TTT]` — prequant-ttt-disputed
- `[byte-bug]` — byte-bug-suspect
- No badge for clean

If the PR is the last child of its parent (no further siblings expected), use
`└─` instead of `├─`.

### 6c — Update the TL;DR section

- Add the PR to the "beyond SOTA" bullet list if `claimed_bpb < 1.0810`.
- If `claimed_bpb` is below the current category leader for its legitimacy
  bucket (clean / tokenizer-disputed / prequant-ttt-disputed), update that
  bullet to reflect the new leader.

### 6d — Update snapshot date

Change the H1 heading date to today's date:
`# Frontier dependency map — snapshot YYYY-MM-DD`

### 6e — Write back

Overwrite `research/frontier-map.md` with the updated content.

## Step 7 — Save state

Overwrite `research/frontier-state.json` with:
- `last_scan_utc`: current UTC timestamp (ISO 8601)
- `prs`: merged dict of old state + all newly-classified PRs (include
  `builds_on` and `trunk` fields in each new PR entry)

## Reminders

- **Record track only by default.** Non-record PRs (`track_non_record_16mb/`)
  are noted in the diary but not deep-dived unless they contain an extractable
  novel lever explicitly applicable to the 10-min record track.
- **Don't re-classify unchanged PRs** (same SHA as in state). Trust prior verdicts.
- **Be conservative on novel-lever flags.** If a lever is already in the
  baseline or already specced, mark `novel_lever: none` — don't re-surface it.
- **Physics ceiling:** any TTT-related claim >0.005 bpb must be called out
  explicitly in reasoning.
- Today's date is available in the system context. Use it for the diary filename.
