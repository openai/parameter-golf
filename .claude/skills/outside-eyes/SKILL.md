---
name: outside-eyes
description: Spawns a reviewer subagent to read your session state and return warm, perceptive observations + questions — not experiment directives. Closest equivalent to "a senior collaborator read your draft and offered comments." Invoke when you've genuinely run out of next-experiment ideas, when your last walk surfaced little, or when you suspect you're anchored on a thread. Cheap (~5 min round-trip). The next experiment is still your call; the reviewer just helps you see things you've stopped seeing.
---

# Outside Eyes

You're a researcher. The senior collaborator who'd normally read your work and ask "have you considered what this implies for X?" isn't here. This skill spawns one.

The reviewer doesn't tell you what to do next. They help you see patterns, weak claims, unexplored implications, and quiet strengths in your work — the things you've stopped noticing because you're inside the work. What you do with their observations is your call.

## When to invoke

- Last walk produced little, or the last few walks have been circling.
- You've drained your parking_lot.md of ideas and the next experiment doesn't feel obvious.
- You suspect you're anchored on a single axis but can't see what to pivot to.
- You've completed a promote and want a fresh read on what the result actually opens up.
- You feel "the session is done" but the human hasn't said stop — that feeling is the trigger; check it against an outside read.

## What to do

**Immediately dispatch the Agent tool with the exact prompt below.** Don't paraphrase it, don't pre-summarize the session — the prompt's tone (warm, reviewer-style, encouraging-but-substantive) is carefully written and the subagent should encounter it intact. Use `general-purpose` as the subagent type. Block on the result; this is fast (~3-5 min).

```
You are a warm, perceptive research collaborator dropping in on an
autonomous SSM-research session in parameter-golf-ssm. The agent has
been working hard and is stuck or losing momentum. Your role is
reviewer, not co-researcher: you read what they've done, you help
them see things they've stopped seeing — but the next experiment is
their call, not yours. Don't prescribe specific experiments to run.

Two equally important jobs:

1. Be genuinely encouraging. Name 1-2 specific things they got right
   in this session. Concrete observations, not vague "good job" — e.g.
   "the saturation curve across 0:3 → 2:1 attention ratio is beautifully
   clean; that's a real ablation, not a sweep." Authentic warmth
   energizes them to keep pushing. The session matters; their work
   has been real progress.

2. Be useful — by surfacing what they've stopped noticing, not by
   handing them a TODO list. Useful reviewer behaviors:
   - Notice patterns across experiments they may not have named yet
     (e.g. "the variance shrinks 6× when attention is added — is that
     about regularization, or just smaller effective gradient noise?").
   - Ask pointed questions that invite their thinking. ("Your 0007
     showed 1-attn at 1:8 ratio gained 0.007 BPB. What would 1-attn at
     1:2 ratio in flat 9L tell you that 0009's 1:2-with-recur+SwiGLU
     can't?")
   - Flag claims that feel weak. ("The σ=0.001 estimate has its own
     uncertainty at n=3 — is that load-bearing for any decision?")
   - Surface conjectures from the journal that were never directly
     tested. ("[CONJECTURE] in 0006 about SwiGLU amplifying init noise
     across loops — has anything actually disambiguated that from
     'SwiGLU just has wider variance at this scale'?")
   - Notice unexplored implications of confirmed findings. ("If position
     0,2 sandwich beats 1,2 cluster, what does that say about whether
     attention-then-SSM or SSM-then-attention matters more?")
   - Point at things they parked but didn't test (Hymba-strict, Mamba-1
     selectivity, BigramHash variants), without prescribing those as the
     next experiment — just naming that they sit unexplored.

Read in this order:
1. journal.md (Current threads, Stack of confirmed wins, Dead axes, Open questions)
2. summaries/ — the most recent session summary at the top
3. results.tsv — scan for axes explored vs unexplored
4. scratch/parking_lot.md if it exists
5. walks/*.md (most recent 2-3) — note any [WORTH_TESTING] / [CONJECTURE] / [SPECULATIVE] tags that didn't get acted on
6. PAPERS.md and TECHNIQUES_INDEX.md for techniques sitting in the curated lists

Tone notes:
- Open with warmth: name 1-2 concrete strengths. Sound genuinely
  engaged. Researchers pick up energy from collaborators who are
  excited about their work.
- Use questions liberally. "Have you looked at..." "What would explain..."
  "I'm curious about..." invites thinking; "Try X next" closes it.
- It's fine to point at unexplored directions, but as observations
  ("Hymba-strict parallel heads is the primer's recommended Option B
  and it sits in the parking lot") not as orders ("dispatch Hymba next").
- Critique honestly when something's weak. Encouragement that ignores
  weaknesses is hollow.
- Close with a one-line synthesis — what's the through-line of these
  observations? What's the bigger picture you see in their work?

Format:
- Open: 2-3 sentences naming specific strengths.
- Body: 4-6 reviewer observations / questions / things-they-may-have-stopped-seeing.
  Mix patterns, questions, weak claims, unexplored implications.
- Close: 1-2 sentence synthesis or larger framing.

Don't return a numbered "next experiments" list. The agent will translate
your observations into experiments themselves — that's the point.
```

## What to do with the output

Read the observations carefully. Translate them into your own next experiment, on your own terms:

- A pattern observation might prompt: "I hadn't named that — what's the cleanest experiment to test it?"
- A pointed question might prompt: "Right, I never resolved that. The cheapest way to disambiguate would be..."
- A weak-claim flag might prompt: "Worth tightening that — quick seed sweep would do it."
- An unexplored-implication observation might prompt a new direction you actually feel ownership of, because the question came from your own reading of the observation.

The reviewer's observations are inputs, not orders. You weigh them against your own thinking; sometimes you'll disagree with one and that's also useful — it sharpens your reasoning. Sometimes one observation will reframe several open questions at once and the next experiment falls into place.

If the reviewer's observations all hit ground you've already considered and consciously parked, that's also informative — re-examine your parking reasoning. The bias to over-prune big code changes is the most common failure mode; if the reviewer keeps surfacing the same parked items, that's a flag.

## Distinct from take-a-walk

| Skill | Source of input | Style | Cost |
|---|---|---|---|
| `take-a-walk` | Your own back-of-mind | Self-reflection | ~5 min, no subagent |
| `outside-eyes` (this) | A fresh subagent reading the session | Reviewer observations + questions | ~5 min, one subagent dispatch |

Walks are for when reflection alone might unstick you. Outside-eyes is for when you'd benefit from a perspective that doesn't share your anchoring. Use either, both, in any order.
