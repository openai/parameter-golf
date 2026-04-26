---
name: outside-eyes
description: Spawns a subagent to read your session state and propose 3-5 concrete next experiments. Closest equivalent to "the human walked over and suggested something." Invoke when you've genuinely run out of next-experiment ideas, when your last walk surfaced little, or when you suspect you're anchored on a thread. Cheap (~5 min round-trip), and the suggestions don't have to be acted on — they're another perspective, not orders.
---

# Outside Eyes

You're a researcher. The human who'd normally walk over and ask "have you tried X?" isn't here. This skill is the closest equivalent: spawn a fresh subagent that reads your session and offers concrete next directions.

## When to invoke

- Last walk produced little, or the last few walks have been circling.
- You've drained your parking_lot.md of ideas and the next experiment doesn't feel obvious.
- You suspect you're anchored on a single axis but can't see what to pivot to.
- You've completed a promote and want a list of follow-ups before picking one yourself.
- You feel "the session is done" but the human hasn't said stop — that feeling is the trigger; check it against an outside view.

You don't need to invoke this regularly. It's specifically the "I'd ask the human if they were here" tool.

## How

Spawn a `general-purpose` subagent with read-only access (no Bash, no Edit, no Write). Prompt:

```
You are an outside reviewer for an autonomous SSM-research session in
parameter-golf-ssm. The agent feels stuck on what to try next. Your
job: read the session state and propose 3-5 concrete next experiments.

Read in this order:
1. journal.md (especially Current threads, Stack of confirmed wins, Dead axes, Open questions)
2. summaries/ — the most recent session summary at the top
3. results.tsv — sweep down for axes the agent has and hasn't explored
4. scratch/parking_lot.md if it exists
5. walks/*.md (most recent 2-3 only) — note any [WORTH_TESTING] tags that didn't get done
6. PAPERS.md and TECHNIQUES_INDEX.md for techniques the agent may have missed

Then propose 3-5 next experiments. For each:
- One-line description (what's the experiment).
- Specific axis or code change required (env var? new module? subagent task?).
- Why it's high-EV at this point in the session — what specific gap or
  axis it would close.
- A rough size estimate (env-var: 5 min; subagent + run: 30 min).

Bias toward bigger experiments (architectural changes, new modules) over
env-var sweeps — the agent tends to under-dispatch big changes. Also flag
any "dead axis" the agent thinks is dead but isn't, or any [CONJECTURE] in
the journal that hasn't actually been tested.

If you genuinely can't find 3-5 high-EV directions, say so honestly —
that's also useful information. But search hard before concluding that.

Return: a numbered list of 3-5 experiments, ranked by EV, with the
specifics above for each. Aim for actionable specificity ("dispatch a
subagent to add Hymba-style parallel attn+SSM heads to Block.forward")
not generic advice ("explore more architectures").
```

Use the `general-purpose` subagent. The subagent should not run experiments or edit files; it's purely advisory.

## What to do with the output

The suggestions are another perspective, not orders. Read them, weigh against your own thinking:

- If 1-2 suggestions feel right and clearly higher-EV than what you had → pick one, write its plan.md, dispatch.
- If suggestions overlap with what you already had → confirms direction, but also tells you the outside view didn't add much novelty here.
- If suggestions are all things you already considered and rejected → reread your rejection reasoning. The bias to over-prune big code changes is the most common failure mode; the subagent's repetition of those ideas is a flag.
- If the subagent reports genuinely few high-EV options → take that seriously. Combined with several quiet walks, this is a reasonable signal that the session has explored its accessible space. Note it in journal Current threads as a possible "ready for human input" state, and continue working on whatever has *some* EV (e.g., extending seed sentinels, writeup polish, follow-up sweeps that you'd otherwise dismiss as low-EV).

Don't let the subagent's output become the new ground truth either. Your judgment is still the final call. The skill exists to break anchoring, not to replace decision-making.

## Distinct from take-a-walk

| Skill | Source of ideas | Cost |
|---|---|---|
| `take-a-walk` | Your own back-of-mind | ~5 min, no subagent |
| `outside-eyes` (this) | A fresh subagent reading the session | ~5 min, one subagent dispatch |

Walks are for when reflection alone might unstick you. Outside-eyes is for when reflection isn't unsticking you and you'd benefit from a perspective that doesn't share your anchoring. Use either, both, in any order.
