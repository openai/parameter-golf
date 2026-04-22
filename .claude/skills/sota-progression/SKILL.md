---
name: sota-progression
description: Deep analysis of the official SOTA submission tree on openai/parameter-golf. Starts with a scoped interview before fetching anything. Produces a structured breakdown of each step — architectural delta, required re-tuning, and mechanistic explanation of why bpb dropped. Use for historical understanding or mining ideas not yet tried on our stack.
---

# SOTA progression analysis

You are about to analyze the official merged submission history on `openai/parameter-golf`.

**Before fetching any PRs or writing any analysis, conduct the interview below.**

---

## Phase 1 — Interview

Ask the user ALL of these questions before proceeding. Do not start the analysis until you have answers.

### Q1 — Scope
What portion of the official tree do you want to analyze?
- **Full tree** — every merged submission from the beginning
- **Range** — e.g. "from #1000 to #1493" or "last 10 merges"
- **Lineage** — trace a specific path (e.g. "how did we get to #1493?")
- **Single step** — just explain one merge in depth

### Q2 — Goal
What are you trying to get out of this?
- **Historical understanding** — how did the competition evolve? What paradigm shifts happened?
- **Idea mining** — what levers in the official tree have we NOT tried on our current #1736 stack?
- **Both**

### Q3 — Depth
How deep per submission?
- **Skim** — architectural delta only (1–2 sentences per step)
- **Standard** — arch delta + required re-tuning + one-sentence mechanistic explanation
- **Deep** — arch delta + tuning delta + mechanistic why + speculation on what would have happened without each change + ideas that spawn from it

### Q4 — Output
Where should the analysis live?
- **In-conversation only** — just talk through it, no files written
- **Diary entry** — written to `diary/YYYY-MM-DD-sota-progression.md`
- **Research doc** — written to `research/sota-progression-<slug>.md` (more permanent reference)

### Q5 — Focus filter (optional)
Is there a specific angle you care about?
- e.g. "only architecture changes, skip pure tuning steps"
- e.g. "focus on TTT-related innovations"
- e.g. "only steps that crossed a round-number bpb threshold"
- Or: no filter, analyze everything

---

## Phase 2 — Fetch the official tree

Once the interview is complete, fetch the merged submissions:

```bash
gh pr list --repo openai/parameter-golf --state merged \
  --limit 100 \
  --json number,title,author,mergedAt \
  --jq 'sort_by(.mergedAt) | .[]'
```

Filter to record-track submissions only (title contains "Record:" or files touch
`records/track_10min_16mb/`). Apply the scope from Q1.

For each submission in scope, fetch the body:
```bash
gh pr view <N> --repo openai/parameter-golf --json number,title,author,body,mergedAt
```

Run fetches **in parallel** (one subagent per PR, or batch if the list is short).

---

## Phase 3 — Analysis loop

For each submission (in merge order), produce a structured entry based on the
depth level from Q3:

### Skim format
```
#XXXX (<author>, <mergedAt>, <bpb>) — <one sentence: what changed>
```

### Standard format
```
## #XXXX — <title> (<author>, bpb: X.XXXX, Δ: −0.XXXX vs prior)

**Architectural delta:** <what changed in the model/training/eval>
**Re-tuning required:** <hyperparameters that had to change alongside it>
**Why it worked:** <one sentence mechanistic explanation>
```

### Deep format
```
## #XXXX — <title> (<author>, bpb: X.XXXX, Δ: −0.XXXX vs prior)

**Architectural delta:** <specific code-level change>
**Re-tuning required:** <what hyperparam adjustments were needed and how they found them>
**Why it worked:** <mechanistic explanation — what does this change enable that wasn't possible before?>
**Without this change:** <speculation — what would have happened if this was skipped?>
**Ideas it spawns:** <what does this suggest we could try on our current #1736 stack?>
```

---

## Phase 4 — Synthesis

After the per-submission entries, always write a synthesis section regardless of depth:

### Patterns
- What is the typical Δ per step? (calibrate expectations)
- What classes of change drove the most bpb? (arch / tokenizer / TTT / quant / kernel)
- Were there any paradigm shifts (sudden large Δ) vs incremental steps?
- What required re-tuning most often?

### Load-bearing ideas
Which innovations were truly load-bearing (the Δ collapsed without them) vs
"nice-to-have" (small incremental)?

### Gaps on our stack (if goal includes idea mining)
Cross-reference the official tree against our current #1736 baseline
(see `CLAUDE.md` already-in-baseline list). Flag any lever from the official
tree that is NOT in our baseline and NOT already specced.

---

## Phase 5 — Output

Write to the location specified in Q4.
- **Default (written doc):** `pr-analysis/sota-progression/<slug>.md` where slug
  reflects the scope (e.g. `full-tree-to-1493.md`, `range-1400-1493.md`)
- **Diary:** `diary/YYYY-MM-DD-sota-progression.md`
- **In-conversation only:** no file written

The `pr-analysis/` directory is the canonical home for all PR relationship
analysis in this repo. `pr-analysis/sota-progression/` is specifically for
official merged SOTA progression analyses.

---

## Reminders

- **Official submissions only.** Do not mix in unmerged open PRs — they're a
  different analysis (use `frontier-scan` for those).
- **bpb is the north star.** Always anchor each step to the Δ bpb, not just
  "what changed."
- **Re-tuning is as important as the idea.** A lever that required a 6-point LR
  sweep to work tells you something different from one that worked drop-in.
- **Typical step size is −0.002 to −0.004 bpb.** Larger claims in open PRs that
  don't appear in the official tree are a signal that something didn't validate.
- **The official tree is ground truth.** What made it through organizer review and
  3-seed verification is a much stronger signal than claimed numbers in open PRs.
