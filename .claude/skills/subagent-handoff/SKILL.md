---
name: subagent-handoff
description: Invoke whenever the next experiment requires a code change beyond an env-var tweak — adding a class, modifying a forward path, applying a known diff. The big architectural changes that "feel like too much work" are exactly what subagents handle well; dispatch liberally. Carries the pre-flight plan check, the spawn prompt template, and the review checklist. Round-trip is ~20-30 minutes including verification — well below what your gut estimates.
---

# Subagent Handoff

You've written a complete plan.md. The subagent's job is to read it and edit `train_gpt.py`. Your job is to verify the plan is ready, spawn cleanly, and review what comes back.

**Threshold for dispatching: anything with >20 lines of code change, any new module/class, any non-trivial refactor.** Subagents one-shot this kind of task cleanly when the plan is precise. The biggest experiments of a session (Hymba-style parallel heads, depth recurrence, BigramHash bolt-ons, new SSM families) are *exactly* the work this skill is built for. The friction is in your head, not in the spawn.

## 1. Plan must be complete first

The harness has a PreToolUse hook on the Agent tool (`.claude/hooks/validate-subagent-plan.sh`) that blocks the spawn if any of the four `<!-- ... -->` template sections in the referenced plan.md is still unfilled. You'll see a deny message if so — fill the placeholders and try again.

The Change section in particular needs to specify exact functions, the expected diff in pseudocode, and any env-var gate — the subagent cannot ask you questions mid-task, so vague Change = drift.

## 2. Spawn

```
Read `experiments/NNNN_<slug>/plan.md`. Implement the change in
`experiments/NNNN_<slug>/train_gpt.py`. Single-file is preferred;
But if it became difficult/unclear to navigate the growing
train_gpt.py, or if plan.md calls for additional files, put them in
`experiments/NNNN_<slug>/modules/` (that subdir is the only path
`new_experiment.sh` forks; anything outside of train_gpt.py,
env.sh, and modules/ is silently dropped on the next fork).
Update plan.md's "Notes from execution" section with what was
done and any deviations. Do not run any experiments. Return a
one-paragraph summary of what you changed and any concerns.
```

Use the `general-purpose` subagent unless a specialized one (code-architect, code-reviewer) clearly fits the task.

## 3. Review

When the subagent returns, verify the diff matches the plan — don't trust the summary alone:

- `git diff experiments/NNNN_<slug>/train_gpt.py` — does the diff match what plan.md described?
- Read plan.md's Notes section — any deviations the subagent flagged?
- Spot-check the env-var gate: default value should produce the canonical code path; the new value should activate the change.

**Decide**:
- Diff matches plan, no concerning deviations → run with `launch-and-await`.
- Small tweak (one-line fix, type adjustment) → do it yourself.
- Significant rework needed (subagent misunderstood, plan was incomplete) → roll back, write a *new* plan, spawn a *new* subagent. **One-shot per plan** — never iterate inside a subagent thread.

## 4. Subagent never runs experiments

The subagent only edits code and updates plan.md. **You** run the experiment via `launch-and-await`. This separation keeps the subagent's scope tight and lets you catch problems before burning a 5-minute run.
