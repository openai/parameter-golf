---
name: pull-out
description: Mental shift from focused execution back to higher-level reassessment. Invoke after finishing an experiment, hitting a blocker, before writing a plan, or every several actions to check whether the current direction is still the right one. Stay in this state as long as you need.
---

# Pull Out

You're at the desk but lifting your head from the immediate task. Pull-out is reasoning time — still rigorous, still bounded, but oriented at the thread and the next few moves rather than the current cell. Most of the work cycle is `pull-out → zoom-in → pull-out → zoom-in`; this is where the thinking happens between executions. Plans, derivations in `scratch/`, journal updates, and retelling the chain all live here.

The four phases below are roughly sequential. Keep it light when the moment is light — a routine LR sweep doesn't need every section. Go deep when the thread is wrapping up, the reasoning feels shaky, or you're about to commit to a new direction.

## 1. Assess — where am I?

- **What did I just learn?** One sentence on the result. If "nothing surprising happened," say that — surprise is the signal.
- **What thread am I on?** Where does this sit in the bigger arc? Is the thread still worth pursuing, or has it played out?
- **Time and budget.** Glance at wall-clock. Recent rate (experiments per hour, gains per experiment) vs what's left.
- **Last 5–10 experiments — pattern check.** All same axis? Mostly noise? When was the last clear signal? "More than 5 ago" is a flag.

## 2. Reason — does the chain hold up?

The desk's job is to verify what the experiment loop produces. This is where you do it.

- **Retell the chain.** As if to a smart, curious teenager who isn't in the field — what have I been doing recently and *why*? If you stumble, hand-wave, or notice a gap, that's the thing to fix. Note it in the journal.
- **Verify each step.** Where did each conclusion come from — derived, observed, or assumed? Any `[CONJECTURE]`-tagged claims in recent journal entries — have they actually been tested, or am I building on them?
- **Use `scratch/`.** High-level thinking often needs paper. Sketch the math, plot the schedule, compute the parameter count, write the small Python that confirms the variance claim. If you've been at the desk for hours with no `scratch/` file, that's a tell.
- **Have we already investigated this?** Don't re-derive what's already on disk. Use the `search_journal` skill if you forget the search syntax.

## 3. Plan — what's next?

- **Highest-EV next move.** Not "the next obvious thing" — they're often different. Compare against parked ideas, untouched techniques in `TECHNIQUES_INDEX.md`, and untested hypotheses in the journal. Should you keep what you are working on? Or park it and move on? Or combine it with some other ideas?
- **Sketch the next ~1–3 experiments.** What's the order? What's the disconfirming prediction for each? If the next experiment doesn't have a clear disconfirmation yet, it's not ready for `zoom-in`.

## 4. Reflect — how have I been working?

A quick meta-check on process, not content.

- Have I been jumping to experiments without deriving anything first?
- Am I anchored on a single axis when the data says move on?
- Am I rationalizing skipping careful planning for "just one more env-var"?
- Am I direct-promoting single-seed wins past their error bars?
- **Have I labeled any claim as load-bearing for the writeup without the seed-count to back it?** Mechanism findings I'd put in the paper deserve the same honesty as a promote: name the seed count *in the claim itself* (`mechanism finding [n=1]`, `mechanism confirmed [n=2 cross-seed]`). Single-seed mechanism work is fine — surfacing it is what pulls the program forward — but the *label* should match the evidence so future-you and the reviewer don't mistake exploration for confirmation.

Note any malpractice and correct it forward. If reflection alone isn't unblocking the feeling that you're in the wrong place entirely, invoke `take-a-walk` — that's what the walk is for.



## 5. Hand off — ready to zoom back in

Land everything before switching modes. Once you `zoom-in`, the thinking is over and the doing begins.

- Record reasoning and reflections in `journal.md`.
- Write the next experiment's `plan.md` carefully (Question / Hypothesis / Change / Disconfirming) — fully enough that you or a subagent can go straight to executing. The harness will refuse to run if it isn't filled.
- Capture parked ideas in `scratch/parking_lot.md` so they're not lost when focus narrows.
- Then call the `zoom-in` skill.
