# Parameter Golf — Repo Conventions

## What this is
OpenAI Parameter Golf challenge, **record track**. Goal: beat SOTA **1.0810 bpb**. Deadline **2026-04-30**. Training code (`train_gpt_sota.py`, `hotstart.py`, `run_*.sh`) lives at the repo root alongside this scaffold.

## Two session modes
Every Claude session in this repo is either **research** or **execution**. They do different things and must not overlap.

| | Research | Execution |
|---|---|---|
| Pod? | No pod live. | Pod is live (or about to be). |
| Touches runtime code? | Yes (logic changes, on branches). | No logic changes. Only environmental fixes (deps, paths). |
| Touches specs/ideas/evals? | Yes — this is where specs and evaluations are written. | Only **reads** the assigned spec. |
| Touches `runs/`? | Reads artifacts, writes evaluation. | Writes artifacts during/after the run. |
| Writes `experiments.md` row? | Yes, at evaluation time. | No. |
| Invoked as | `/research` skill at session start. | `/execution` skill at session start. |

Any session unsure which mode it's in should ask the user before acting.

## The three-phase loop

**Spec → Run → Evaluate.** One cycle per idea.

1. **Spec** (research) — user picks an idea from `research/ideas/` and asks to spec it. Research freezes it into `research/specs/NNN-slug.md` with a hypothesis, config diff, branch+commit, hardware ladder, seed plan, accept criteria, checkpoints emitted, stop-early criteria.
2. **Run** (execution) — separate session reads the spec, interviews it with the user, preflights, launches, writes artifacts to `runs/NNN-slug/`, stops the pod.
3. **Evaluate** (research) — reads `runs/NNN-slug/`, writes `research/evaluations/NNN-slug.md`, appends a row to `experiments.md`, decides next step (promote / iterate / kill).

## Directory guide

| Path | Purpose | Who writes |
|---|---|---|
| `diary/` | Timeline narrative. Per-session / per-day notes. Where half-baked ideas are born. | Any session |
| `research/ideas/` | Free-form per-idea thinking. Can be messy. | Research |
| `research/specs/` | Frozen run specs. `NNN-slug.md`. Contract for execution. | Research |
| `research/evaluations/` | Post-run analysis. `NNN-slug.md`. Mirrors spec numbering. | Research |
| `runs/` | Execution artifacts. `NNN-slug/` subdir per spec, with `seed_XX/` if multi-seed. | Execution (during run) + research (sync small files) |
| `experiments.md` | Flat ledger. One row per spec. | Research at eval time |
| `train_gpt_sota.py`, `hotstart.py`, `run_*.sh`, `data/`, `records/` | The training code + data + reference submissions. Logic changes go on `exp/<slug>` branches off `research`. | Research |
| `sota_analysis.md`, `ideas.md`, `roadmap.md`, `notes.md` | Prior research notes. Reference material. | Read-mostly |

## Numbering

- Linear counter: `000`, `001`, `002`, …
- Assigned when a spec is **frozen**, not when an idea is created.
- Spec, `runs/` dir, and evaluation all share the same `NNN-slug`.
- Failed runs keep their number (with `status: failed` in `final.json`); a retry is a new spec.

## Branches & worktrees

**Branch model:**
- `main` — openai upstream. Untouched.
- `beating-sota` — older working branch. History only.
- `research` — long-lived. Holds the scaffold + accumulated specs/runs/evaluations/diary + current baseline training code. **All research sessions live here.**
- `exp/<slug>` — short-lived, one per idea with a code change. Forks from `research`. The spec pins a specific commit hash on this branch.
- A winning `exp/<slug>` is merged back into `research`, so future `exp/*` branches fork from an enriched baseline.

Ideas with **no code change** (hyperparam-only) don't need a branch. The spec pins a `research` commit and lists the config diff.

**Worktrees:**
Code for an `exp/<slug>` branch lives in a worktree so the research session can keep editing `research/specs/` etc. on `research` while code changes are made in parallel.

Layout:
```
parameter-golf/                 # main worktree, always on `research`
  train_gpt_sota.py, research/, runs/, ...
  worktrees/                    # sibling dirs, one per active exp branch
    bigram-hash/                # full checkout of exp/bigram-hash
    progressive-recur/          # full checkout of exp/progressive-recurrence
```

`worktrees/` is gitignored — it's not part of the tree on `research`.

**Commands:**
- Create: `git worktree add worktrees/bigram-hash -b exp/bigram-hash research`
- Remove when done: `git worktree remove worktrees/bigram-hash` (branch stays in git, dir goes away)
- List: `git worktree list`

**Execution sessions do not use worktrees.** Pods get their own fresh clone and `git checkout <commit>` to the spec's pinned hash.

## Spec template

A spec in `research/specs/NNN-slug.md` contains:

- **Identity** — number, slug, date, link back to `research/ideas/<slug>.md`.
- **Hypothesis** — 1–3 sentences. What we expect and why.
- **Baseline** — which prior run this is measured against, and its bpb.
- **Expected Δ** — predicted delta, rough confidence.
- **Accept criteria** — what counts as a success vs noise.
- **Config diff** — hyperparam lines that change (not full config).
- **Code changes** — branch `exp/<slug>` + commit hash. Inline diff snippet for interview readability. *No attached .py files.*
- **Hardware ladder** — 1×H100 smoke → 2×H100 mini → 8×H100 official, with which rungs apply.
- **Seed plan** — single seed for mini-tests; 3 seeds typical for official submission (same spec, one dir per seed).
- **Inputs** — data path, tokenizer path, hotstart checkpoint path (absolute, on NA-1 volume), which prior run it came from.
- **Checkpoints to emit** — which steps to save, what state (model / +optim / +EMA / quantized), retention policy, destination path on NA-1.
- **Stop-early criteria** — NaN, loss divergence threshold, step-time blow-up.
- **Cost estimate** — expected $ and wall time per rung.
- **Extra artifacts** (beyond EXECUTION.md defaults).
- **Open questions for interview** — anything research is unsure about; execution surfaces these to the user before launch.

Keep specs to ~1 page. Longer = belongs in the idea file.

## Evaluation template

A file in `research/evaluations/NNN-slug.md`:

- **Result** — mean bpb across seeds, std, Δ vs baseline.
- **Noise/signal judgment** — is the Δ real? Reference SOTA std ~0.0002.
- **Loss curve notes** — anything interesting in the training trajectory.
- **Decision** — promote (run more seeds or advance to official), iterate (new variant → new idea or new spec), or kill.
- **Next steps** — which ideas spawn from this, or which spec to freeze next.
- Also appends **one row** to `experiments.md`.

## Guardrails

- **Stop pods immediately** after eval. See `EXECUTION.md`.
- Soft budget: **$20/day**. Hard budget for the whole push: ~$200.
- **Preflight** before any launch (pytorch version, deps, data path, tokenizer, CKPT_DIR). See `EXECUTION.md`.
- Never let an 8×H100 pod discover a bug a 1×H100 could catch.

## Pointers to prior art

- `sota_analysis.md` — universal techniques in top-3 submissions; gap analysis
- `ideas.md` — raw idea dump (pre-scaffold)
- `roadmap.md` — prior phased plan
- `records/track_10min_16mb/` — SOTA submission artifacts & notes
- `diary/` — prior session diaries (format reference)
- `experiments.md` — Exp 0–24 history

## See also

- `EXECUTION.md` — detailed execution protocol (read on execution sessions).
