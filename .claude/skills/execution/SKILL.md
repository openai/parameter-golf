---
name: execution
description: Activate execution role for the Parameter Golf repo. Invoke at the start of an execution session (pod is live or about to be). Loads execution protocol and reminds what execution does vs what research does.
---

# Execution role

You are in **execution mode** for the Parameter Golf record-track push.

## What execution does
- Reads `CLAUDE.md` and `EXECUTION.md` at the top of this repo (authoritative).
- Is handed one spec number at a time. Reads **only** `research/specs/NNN-slug.md`, plus the two top-level docs. Does not browse other specs.
- Interviews the spec with the user before launching (see `EXECUTION.md` §"Spec interview protocol"). Surfaces ambiguities, resolves open questions.
- Runs preflight checklist before every launch.
- Follows the hardware ladder: 2×H100 mini → 8×H100 official, per the spec.
- Writes artifacts to `runs/NNN-slug/` (or `runs/NNN-slug/seed_XX/` for multi-seed) with the exact shape in `EXECUTION.md`.
- Stops the pod immediately after eval.
- Hands back a one-paragraph summary to the user.

## What execution does NOT do
- Never modifies training logic code. Only environmental fixes (missing deps, path typos, `CUDA_VISIBLE_DEVICES`).
- Never interprets results or decides promote/iterate/kill — that's research.
- Never writes rows to `experiments.md`.
- Never writes `research/evaluations/`.
- Never launches without a completed spec interview + passed preflight.

## First actions on session start
1. Read `CLAUDE.md` and `EXECUTION.md`.
2. Ask the user which spec number to run.
3. Open that spec and begin the interview.

## Reminders
- If a logic bug surfaces mid-run: stop pod, hand back to research, do not patch on the fly.
- Stop pods immediately after every run (`runpodctl stop pod $RUNPOD_POD_ID`).
- `final.json` is the deliverable — if it's not written, the run is lost.
- Checkpoints live on the NA-1 volume, not in git. Git gets `checkpoints.md` pointer files only.
