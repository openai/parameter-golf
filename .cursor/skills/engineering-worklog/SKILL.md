---
name: engineering-worklog
description: Keeps ongoing engineering documentation current while Codex is implementing, debugging, researching, or iterating. Use when the user wants continuous build logs, worklogs, memory logs, experiment notes, decision trails, or evidence-backed documentation captured as the work happens, especially across trial-and-error loops, validation runs, refactors, and partial failures.
---

# Engineering Worklog

Maintain a factual trail of what changed, what was tried, what failed, what validated, and what remains open. Preserve the repo's existing documentation system first; only create new files when the repo lacks one and the work is substantial enough to justify durable notes.

## Workflow

1. Inspect existing documentation surfaces before writing.

- Look for `docs/`, build-log folders, templates, memory logs, runbooks, findings, tickets, and evidence directories.
- Reuse the repo's naming pattern, file layout, and cadence when possible.
- If there is an index or template file, follow it.

2. Choose the minimum durable surfaces that match the work.

- For a distinct implementation or debugging tranche, update or create a dated build log entry.
- For cross-session continuity, update a running memory log if the repo has one.
- For unresolved defects, risks, or decisions that belong in governance docs, update findings or tickets only if the repo already uses them.
- If the repo has no convention, create one small obvious surface instead of a new documentation tree.

3. Write during the work, not only after it ends.

- Add notes after meaningful hypotheses, failed attempts, validation runs, evidence captures, and landed patches.
- Prefer several short follow-up sections in the same active log over one retrospective dump.
- Record negative results when they changed the plan or ruled out a path.

4. Capture engineering reasoning, not just outcomes.

- State the objective, tranche, or question being worked.
- Record the hypothesis or suspected root cause.
- Note the concrete change, why it was chosen, and the trade-offs.
- Record validation commands, evidence paths, observed counts, and meaningful outputs.
- Name remaining blockers, debt, and next actions.

5. Keep the log trustworthy.

- Do not claim commands, tests, screenshots, or artifacts that were not actually produced.
- Distinguish landed changes from attempted or abandoned work.
- If a result is partial, degraded, or blocked by environment issues, say so plainly.
- Prefer exact file paths, issue ids, dates, and metrics over vague summaries.

6. Keep the repo resumable.

- Leave enough context that a later session can answer: what changed, why, how it was validated, and what should happen next.
- When evidence is curated separately from generated artifacts, respect that boundary.
- If the repo keeps a build-log index, add the new entry to the index.

## Entry Shape

Use the repo's own template when it exists. Otherwise adapt the patterns in [references/templates.md](references/templates.md).

Default content for a substantive entry:

- objective or tranche
- work completed
- files touched
- validation commands
- evidence or artifact paths
- findings or outcome
- next action or open question

## Heuristics

- Update docs for non-trivial work, repeated trial-and-error, new policies, new workflows, or anything that will be hard to reconstruct later.
- Skip heavy logging for tiny low-risk edits unless the user explicitly wants a full trail.
- Append to the active log for the same day or tranche instead of scattering near-duplicate files.
- Keep prose factual and compact; the goal is durable recall, not storytelling.

## Cross-Repo Adaptation

- Mirror the host repo's folder names, issue references, and artifact conventions.
- If the repo has no clear convention, prefer `docs/build-logs/` plus one dated file before introducing broader structure.
- Start with generic templates, then let the repo's own patterns replace them over time.

## References

- Read [references/templates.md](references/templates.md) when you need a starting structure for build logs, memory logs, or experiment notes.
