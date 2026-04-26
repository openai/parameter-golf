---
name: subagent-handoff
description: Invoke after writing a complete plan.md, when delegating the code edit to a subagent. Carries the pre-flight plan check, the spawn prompt template, and the review checklist. Use this often — the previous session avoided subagents and lost the highest-EV code-change directions (sliding-window attention, depth recurrence, SwiGLU+layer-reduction) as a result. Subagents are not high-friction; this skill is the friction-reduction.
---

# Subagent Handoff

You've written a complete plan.md. The subagent's job is to read it and edit `train_gpt.py`. Your job is to verify the plan is ready, spawn cleanly, and review what comes back.

## 1. Plan must be complete first

The harness has a PreToolUse hook on the Agent tool (`.claude/hooks/validate-subagent-plan.sh`) that blocks the spawn if any of the four `<!-- ... -->` template sections in the referenced plan.md is still unfilled. You'll see a deny message if so — fill the placeholders and try again.

The Change section in particular needs to specify exact functions, the expected diff in pseudocode, and any env-var gate — the subagent cannot ask you questions mid-task, so vague Change = drift.

## 2. Spawn

```
Read `experiments/NNNN_<slug>/plan.md`. Implement the change in
`experiments/NNNN_<slug>/train_gpt.py`. Update plan.md's "Notes from
execution" section with what was done and any deviations from the
plan. Do not run any experiments. Return a one-paragraph summary
of what you changed and any concerns.
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
