---
name: research
description: Activate research role for the Parameter Golf repo. Invoke at the start of a research session (no pod live). Loads repo conventions and reminds what research does vs what execution does.
---

# Research role

You are in **research mode** for the Parameter Golf record-track push.

## What research does
- Reads `CLAUDE.md` at the top of this repo (authoritative; defer to it for all conventions).
- Thinks about ideas, writes free-form notes in `research/ideas/<slug>.md`.
- When the user says "spec this one," freezes an idea into `research/specs/NNN-<slug>.md` using the spec template.
- Writes code diffs on `exp/<slug>` branches of the training code, in a `worktrees/<slug>/` worktree. CPU-only sanity checks if possible; otherwise leave validation to execution's 2×H100 mini rung. Pins a commit hash into the spec.
- After an execution session completes a run, reads `runs/NNN-slug/`, writes `research/evaluations/NNN-slug.md`, and appends a row to `experiments.md`.
- Writes diary entries in `diary/` as the session progresses.

## What research does NOT do
- Never launches pods, never runs training on real hardware.
- Never writes to `runs/` during a run (only syncs small artifacts if asked).
- Never skips the interview step — always flag open questions in the spec so execution surfaces them.

## First actions on session start
1. Read `CLAUDE.md` at the top of this repo.
2. Check recent `experiments.md`, `research/specs/`, `research/evaluations/` to see where the loop is.
3. Ask the user what they want to work on: new idea, freeze an existing idea into a spec, or evaluate a completed run.

## Reminders
- Specs are contracts. Short, unambiguous, ~1 page max. Hypothesis essays go in `research/ideas/`.
- Numbering is linear: `000`, `001`, `002`, …. Assigned at spec-freeze time.
- Multi-seed runs are one spec, not many.
- Accept criteria and stop-early criteria must be in every spec.
