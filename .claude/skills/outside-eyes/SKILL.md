---
name: outside-eyes
description: Spawns a reviewer subagent to read your session state and return warm, perceptive observations + questions — not experiment directives. Closest equivalent to "a senior collaborator read your draft and offered comments." Invoke when you've genuinely run out of next-experiment ideas, when your last walk surfaced little, or when you suspect you're anchored on a thread. Cheap (~5 min round-trip). The next experiment is still your call; the reviewer just helps you see things you've stopped seeing.
---

# Outside Eyes

Dispatch immediately, no thinking required:

1. Use the Agent tool with `subagent_type: general-purpose`.
2. The prompt is the full content of `.claude/skills/outside-eyes/reviewer_prompt.md` — load it with Read and pass it verbatim as the spawn prompt. Don't paraphrase, don't pre-summarize the session.
3. Block on the result (~3-5 min).

The reviewer reads the session state on its own (journal/summary/results/walks/parking_lot/papers) and returns observations + questions in a reviewer voice — strengths it notices, patterns it sees, weak claims it questions, unexplored implications it surfaces. It will NOT return a "next experiments" list. That's intentional.

## When to invoke

- Last walk produced little, or the last few walks have been circling.
- You've drained your parking_lot.md of ideas and the next experiment doesn't feel obvious.
- You suspect you're anchored on a single axis but can't see what to pivot to.
- You've completed a promote and want a fresh read on what the result actually opens up.
- You feel "the session is done" but the human hasn't said stop — that feeling is the trigger; check it against an outside read.

## What to do with the output

The reviewer's observations are inputs, not orders. Translate them into your own next experiment, on your own terms. A pattern observation might prompt a clean experiment to test the pattern; a weak-claim flag might prompt a quick seed-sweep tightening; an unexplored-implication question might prompt a new direction you actually feel ownership of. Sometimes you'll disagree with an observation — that's also useful, it sharpens your reasoning.

If the reviewer's observations all hit ground you've already considered and consciously parked: re-examine your parking reasoning. The bias to over-prune big code changes is the most common failure mode; if the reviewer keeps surfacing the same parked items, that's a flag.

## Distinct from take-a-walk

| Skill | Source of input | Cost |
|---|---|---|
| `take-a-walk` | Your own back-of-mind | ~5 min, no subagent |
| `outside-eyes` (this) | A fresh subagent reading the session | ~5 min, one subagent dispatch |

Walks are for when reflection alone might unstick you. Outside-eyes is for when you'd benefit from a perspective that doesn't share your anchoring. Use either, both, in any order.
