---
name: outside-eyes
description: Spawns a reviewer subagent — via a Bash script that runs `claude -p` in a fresh session — to read your work and return warm, perceptive observations + questions. The reviewer prompt is hidden from you on purpose, so the review is uncontaminated by your own anchoring. Closest equivalent to "a senior collaborator read your draft and offered comments." Invoke when you suspect you're anchored, your last walk surfaced little, you've run out of next-experiment ideas — OR when you have a clean story but hours of compute left and the obvious-feeling next experiment would be more rigor-polish than a new bet. Earlier than you think. Cheap (~5 min round-trip). The next experiment is still your call.
---

# Outside Eyes

Dispatch with one Bash call, no thinking or reading required:

```
bash .claude/skills/outside-eyes/outside_eyes.sh
```

That's it. The script spawns a fresh reviewer subagent in a separate Claude session (via `claude -p`). The reviewer reads the session state on its own (journal/summary/results/walks/parking_lot/papers) and returns observations + questions in a reviewer voice — strengths it notices, patterns it sees, weak claims it questions, unexplored implications it surfaces.

The reviewer will NOT return a "next experiments" list. That's by design — translating observations into experiments is your job, on your own terms.


## When to invoke yourself

- Last walk produced little, or the last few walks have been circling.
- You've drained your parking_lot.md of ideas and the next experiment doesn't feel obvious.
- You suspect you're anchored on a single axis but can't see what to pivot to.
- You've completed a promote and want a fresh read on what the result actually opens up.
- You feel "the session is done" but the human hasn't said stop — that feeling is the trigger; check it against an outside read instead of stopping.
- **You have a clean story and compute left.** This is the sneakiest trigger. The story feels packaged, the next experiment looks like another seed-confirm or σ-tightening on something already-promoted, and you can't easily name a *new bet* you'd rather make. Outside-eyes catches drift-to-rigor-polish; the previous overnight session had four consecutive same-family experiments before the reviewer flagged it.
- **You can't crisply state your next-experiment hypothesis.** If asked "what's the disconfirming prediction for the next run?" and the answer is fuzzy, that's the trigger. Vague hypotheses are the agent's body language for "I don't actually know what I'm asking" — outside-eyes can sharpen it.

## What to do with the output

The reviewer's observations are inputs, not orders. Translate them into your own next experiment, on your own terms. A pattern observation might prompt a clean experiment to test the pattern; a weak-claim flag might prompt a quick seed-sweep tightening; an unexplored-implication question might prompt a new direction you actually feel ownership of. Sometimes you'll disagree with an observation — that's also useful, it sharpens your reasoning.

If the reviewer's observations all hit ground you've already considered and consciously parked: re-examine your parking reasoning. The bias to over-prune big code changes is the most common failure mode; if the reviewer keeps surfacing the same parked items, that's a flag.

## Distinct from take-a-walk

| Skill | Source of input | Cost |
|---|---|---|
| `take-a-walk` | Your own back-of-mind | ~5 min, no subagent |
| `outside-eyes` (this) | A fresh subagent reading the session | ~5 min, one `claude -p` subprocess |

Walks are for when reflection alone might unstick you. Outside-eyes is for when you'd benefit from a perspective that doesn't share your anchoring. Use either, both, in any order.
