---
name: agent-lab
description: Parameter Golf agent-lab research loop — experiments registry, commit format, metrics, and files to update after each run. Use when working in agent_lab/, autonomous training experiments, or docs/build-logs for agent lab sessions.
---

# Agent lab (Parameter Golf)

## Before changing code

1. Read **`agent_lab/program.md`** (hard constraints).
2. Read **`agent_lab/experiments.tsv`** — what was tried, verdicts, best commit so far.
3. Read **`.cursor/rules/parameter-golf.mdc`** (challenge guardrails).

## After each full training run

1. Append **`agent_lab/results.tsv`** (gitignored loop log) if you use it — columns per `program.md`.
2. Append **`agent_lab/experiments.tsv`** (tracked) with stable **`AL-YYYYMMDD-NNN`** id, parent commit, hypothesis, **verdict** (`correct` / `wrong` / `partial` / `n_a`), metric, `val_bpb`, notes.
3. Commit with **`feat(agent-lab):`** or **`docs(agent-lab):`** and **rich body** (see **`agent_lab/COMMIT_CONVENTIONS.md`**).
4. Update **`docs/build-logs/<date>-agent-lab.md`** — short journal entry: what you believed, what happened, what you learned.

## Stable experiment IDs

- Pattern: **`AL-YYYYMMDD-NNN`** (NNN = 001, 002, … per day).
- Same ID in: `experiments.tsv`, commit body `Exp:`, build log headings.

## Primary metric

Default: **`final_int8_ttt_lora`** line — lower **`val_bpb`** is better. Do not mix with zlib roundtrip across comparisons unless documented.

## Official challenge time limits (leaderboard)

Training **and** evaluation each have a **~10 minute** budget on **8×H100** for record submissions — see **`agent_lab/CHALLENGE_TIMELIMITS.md`**. Local dev on other GPUs can exceed that; optimize the **script** so official eval still fits when you submit.

## Adapt this skill

When you discover friction (slow TTT, unclear metric, bad defaults), **edit this SKILL.md** or **`agent_lab/program.md`** so the next session inherits the lesson.
