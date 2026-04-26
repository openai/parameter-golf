---
name: wrap-session
description: Invoke at session end — when the human signals stop, when wall-clock runs short, or at a natural endpoint (major win, exhausted thread). Writes the session summary to summaries/, runs the rotate script that moves per-session entries from journal.md into journals/YYYY-MM-DD_<slug>.md, leaves only durable cross-session sections in active journal, commits. Without this, journal.md grows unbounded and the next session starts with bloated context.
---

# Wrap Session

Sessions end. Without an explicit wrap, journal.md keeps growing and "Current threads" blurs with episodic notes. This skill lands the session cleanly so the next agent gets a clean handoff.

## 1. Time check + scope

```bash
date
```

Note: session start time (top of `journal.md`), wall-clock duration, experiments completed, promotes landed. These numbers go into the summary's top-of-file section in step 2.

Pick a **slug** for this session: short, kebab-case, names the dominant theme (e.g. `recurrence-swiglu`, `ortho-init-sweep`, `quantization-pivot`). The same slug is used for both the summary and the rotated journal so they're paired.

## 2. Write the session summary

`summaries/YYYY-MM-DD_<slug>.md` — narrative handoff for the next agent. Aim for ~3–5k words. **Use the slug from step 1 verbatim — the rotated journal in step 3 will use the same `YYYY-MM-DD_<slug>.md` filename so summary and journal are paired by name.**

The previous summary at `summaries/2026-04-25_overnight_session.md` is the model — match its structure:

- **Top of file**: best `val_bpb`, cumulative gain vs canonical, count of experiments, session duration, one-line theme.
- **Stack of confirmed wins**: ordered list of promoted wins this session with Δ and journal heading pointers (so future agents can `mdq` directly to the entry).
- **Cross-experiment lessons**: numbered, each tied to a journal heading. These are what the next agent needs to know.
- **Dead axes**: what was tried, what hurt, with experiment paths so they don't get re-tried by accident.
- **Set in stone vs still hypothesis**: separate verified facts (multi-seed, derived) from one-seed claims.
- **Predictions vs actuals**: if `scratch/` has a predictions table from the start of the session, score it. The point isn't accuracy — it's calibration. Honest scoring helps the next agent weight your confidence claims.
- **Walk reflections**: skim `walks/*.md` from this session and pull any `[WORTH_TESTING]` / `[WORTH_DERIVING]` items that didn't get done — they belong in followups.
- **Follow-ups for next session ranked by EV**: the next agent's starting menu.
- **Reflections**: what went well, what didn't, anti-patterns. The next agent learns from honesty about failures more than from the wins.

## 3. Rotate the journal

```bash
bash .claude/skills/wrap-session/scripts/rotate.sh <slug>
```

**Pass the same slug used for the summary in step 2** — the script creates `journals/YYYY-MM-DD_<slug>.md` so the archived journal and its summary share a name and can be paired by anyone scanning either directory.

The script:
- Cuts everything from `## Entries (newest first)` to EOF in `journal.md`.
- Writes it (with a header) into `journals/YYYY-MM-DD_<slug>.md`.
- Leaves a fresh empty `## Entries (newest first)` marker at the end of `journal.md` for the next session.
- Errors loudly if the target archive file already exists or if the marker isn't found.

What stays in `journal.md`: `## Current threads`, `## Confirmed-paying axes`, `## Dead axes`, `## Open questions` — anything above the entries marker. Those are the durable cross-session layer.

## 4. Curate Current threads

The script preserves these sections verbatim. Now read them top to bottom and prune:

- **Current threads**: keep the *current* best, the verified frameworks, the load-bearing formulas (lr_mul, cross-seed variance baseline, quant_tax range). Drop "Prior winner: X" lines — they're history, searchable in `journals/`.
- **Confirmed-paying axes**: keep — this is the durable wins ledger. Add this session's new wins.
- **Dead axes**: keep — prevents re-trying. Add anything new that hurt this session.
- **Open questions**: keep unresolved items, including anything new the session surfaced. Drop any question this session answered.

Goal: `journal.md` stays under ~500 words after curation. Anything longer means episodic notes are leaking in — push them to the rotated file.

## 5. Commit

```bash
git status   # spot-check what's about to land
git add summaries/ journals/ journal.md results.tsv
git commit -m "Wrap session YYYY-MM-DD <slug>: <one-line session result>"
```

`scratch/` and `walks/` are gitignored / left alone — leave brainstorm and walk artifacts in place for future reference; they're useful even after the session.

## 6. Hand off

The very last thing: append a single line under `## Open questions` (or `## Current threads` if it shapes the next session's whole direction) describing the immediate next move. Examples:

- "Next session: SEED=42 confirm of exp 0060 first."
- "Next session: pivot from recurrence-axis tuning to coverage (sliding-window, smear-gate, EMA)."

That line is the next agent's first concrete action. Done.
