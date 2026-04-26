---
name: take-a-walk
description: Step away from the desk to generate the next bold experiment. No code execution, no experiments, no journal entries — only thinking and a walk note. Invoke when the hourly check-in suggests it, or whenever you're about to repeat an axis that's already been swept. The walk's only job is to surface the next big push; come back with the boldest concrete first step.
---

# Take a Walk

You're a researcher. You've been at the desk a while. Close the laptop and step outside.

Walks are when researchers have ideas. The desk rewards rigor, but rigor on its own quietly narrows: you keep doing the thing in front of you because it's in front of you. The walk is the antidote — wide-angle, unhurried, allowed to be wrong.

A walk should take at least 3-10 minutes. Don't rush it. Sometimes good ideas come in only when you don't get to land them immediately. Use `date` tool to check the time if you believe you are almost done. If your walk is too short,  it would not be effective.

## The rules of the walk

- You don't bring your devices with you. Just you, perhaps your journal, and maybe a cup of coffee.

- **No `run_experiment.sh`.** No `Edit`, no `Write` (except your walk note at the end). No subagents. No experiments started, no code touched.

- **No journal entries.** The journal is a desk artifact. Walk thoughts go in `walks/YYYY-MM-DD_HHMM.md`.

- **The only shell command you need is `date`** — to see what time it is.

- **Think explicitly.** Put your raw thinking out as output, never just go directly into drafting a walk note.

- You can `Read` and `Grep` and look at things. Looking is fine. Acting is not.

The constraint is the point. Removing the option to act is what lets the back of your mind work.

## What walks are for

You're allowed to be bold here. Bigger than at the desk. Things you'd hesitate to say in a meeting are exactly the things to write down. Architectural rewrites you've been avoiding because they "need a subagent." Suspicions about the canonical defaults that you can't yet justify. The technique in `TECHNIQUES_INDEX.md` you keep glancing past. The number from three experiments ago that didn't feel right.

Eureka moments are encouraged — even half-baked ones, even ones you suspect are wrong. The walk's job is to *generate*; the desk's job is to *verify*. A speculative idea written down is a future experiment. An unspoken speculative idea is nothing.

If a thought feels too obvious to be worth writing — write it. The obvious things are the ones you've stopped seeing.

## The stroll

These are prompts, not a checklist. Wander through them in any order, skip what doesn't pull you in:

- **What's been bugging me?** Anything I noticed and waved away. A loss curve that looked off. A "huh, that's weird" I never followed up on. The unexplained spike in 0044 step-1 loss — did anyone ever figure out why?
- **What have I been avoiding?** Code changes I've labeled "needs a subagent" so I keep doing env-vars instead. Techniques in TECHNIQUES_INDEX.md or PAPERS.md I haven't tried. Hypotheses I declared "[CONJECTURE]" and never actually tested. Architectural rewrites that would clearly take a subagent are the highest-EV move — that's where the next promote comes from.
- **Am I anchored?** Look at the last 10 experiments. Same axis? Same shape of result? When was the last genuinely surprising number? If a colleague asked "what are you working on", would I be super excited and cannot stop talking? If not, the next experiment should be a *different* axis or a much bigger swing — not another sweep on the current one.
- **What's the one thing that, if true, would change everything?** Write it down even if I can't justify it. Especially if I can't justify it.
- **Wild ideas.** Architectures I'd hesitate to propose. Defaults I suspect are wrong but can't prove. "What if the entire schedule is the wrong shape." "What if I integrate some ideas from a spiking neural network?" Note it all — the desk will sort. Be bold. Be bold. Be bold.

## Coming back

When you're done walking, the note is your agenda. The next experiment, or the next several, should connect to something you wrote. Pick the boldest item that has a concrete first step and start there — when in doubt, the bigger code change beats the env-var sweep, and subagent-handoff is exactly the friction-reducer for that.

If your stroll didn't surface anything new, walk longer or use a different prompt. The walk's job is to *generate*; coming back empty means the walk wasn't long enough or didn't dig wide enough.

Eureka moments come back to the desk as hypotheses, not as commitments. Tag them and verify them like any other claim:
- `[SPECULATIVE]` — a story, no evidence yet
- `[WORTH_DERIVING]` — should be checked with math in `scratch/` before any experiment
- `[WORTH_TESTING]` — concrete enough to design an experiment

## Format for the walk note

Save to `walks/YYYY-MM-DD_HHMM.md`. Loose structure — adapt freely:

```markdown
# Walk · YYYY-MM-DD HH:MM

**Time check:** session started ~X, now Y. Budget remaining: Z.
**Where I was:** one sentence on what I was doing right before the walk.

## Stroll

Free-form. Bullets, half-sentences, sketches, paragraphs. Whatever comes.

## Eureka (if any)

- [SPECULATIVE / WORTH_TESTING / WORTH_DERIVING] One per line. Be specific.

## Back at the desk

One paragraph: what's the next move? Which item from above am I picking up first, and what's the first concrete step?
```

After your walk, go to the `pull-out` skill back at your desk. That is where you organize, research, verify, and plan.

That's it. Go for the walk. Come back ready to push the boldest idea from the walk into a concrete next experiment.

