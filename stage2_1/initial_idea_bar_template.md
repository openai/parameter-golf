# Initial Idea Bar Template

This is the meta template for future `parameter-golf` idea generation.

It exists because the current bottleneck is no longer execution discipline. The bottleneck is the quality of the initial ideas.

The frontier has already shown what weak idea generation looks like:

- too local
- too narrow
- too easy to implement
- too small to beat a strong default

This template is meant to prevent that.

## Core Principle

Initial ideas must be:

- wide enough to challenge the current default
- consequential enough to matter at the `600s` deployed horizon
- code-mandatory, not discussion-only
- evaluated as full-pipeline mechanisms, not isolated local tweaks

If an idea does not clear that bar, it should not enter the run slate.

## Enigma Principles To Carry Over

These are the search principles that should stay invariant:

- mechanism-first, not coefficient-first
- broad frontier coverage before pruning
- dual-baseline thinking: current promoted winner and true production/default
- explicit negative knowledge
- explicit gap analysis
- swing hard enough that a real win is possible
- faithful implementation matters as much as hypothesis quality

The point is not to retune one known winner.
The point is to sample from the full space of plausible winners.

## The Initial Idea Standard

A valid initial idea must satisfy all of:

1. It breaks a false invariant
- one regime for the whole run
- one objective for the whole run
- one checkpoint
- one data order
- one context budget
- one update law for all parameter families
- one export target

2. It has a first-order causal story
- not "might help"
- not "this won in another PR"
- a direct explanation of why deployed `val_bpb` should move

3. It is large enough to matter
- expected gain should be bigger than the control noise floor
- if the likely effect is only a few `1e-3`, it is not a lead hypothesis unless it sits in a known first-order lane

4. It is code-mandatory
- if the idea cannot be expressed as an actual runtime patch or code path, it is not ready
- "we should try X" without a patch surface is not an idea, it is a topic

5. It considers the full picture
- training quality
- deployment damage
- artifact budget
- evaluation policy
- runtime/throughput budget

If it only talks about one of those in isolation, it is not mature enough.

## Code Mandatory Rule

Every initial idea must name:

- exact code surface to patch
- what phase it changes
- what object it changes
- what logs or summaries will show the effect

Minimum patch specification:

- target file(s)
- target function or loop
- new env or patch gate
- expected metrics affected

If this cannot be written in a few lines, the idea is still vague.

## Swing Hard Rule

Do not spend initial slots on ideas that are only plausible as small helpers.

Initial ideas should be willing to:

- split the training process into phases
- change the export object
- reallocate bytes to a different subsystem
- introduce a new deployment-alignment objective
- stage data order or context budget by phase
- separate parameter families under different late laws

The point is not recklessness.
The point is to ensure the slate contains mechanisms that could actually beat the current SOTA-aligned default.

## Full-Picture Rule

Every idea must be written against the full score decomposition:

- representation quality
- deployment / quantization damage
- eval-policy lift
- throughput / step budget
- artifact-size constraint

For each idea, explicitly answer:

- what part of the score it attacks first
- what part it might worsen
- why the net effect could still be positive

This prevents "good train-loss idea, bad submission" mistakes.

## Dry-Idea Elimination

Before any run, reject ideas that fit any of these patterns:

### 1. Pure Local Retune
- same mechanism, different coefficient
- same family, same phase, same causal story

### 2. No Distinct Bottleneck
- cannot say what failure mode it fixes
- only says "may help optimization"

### 3. No Plausible Route To Deployed Gain
- improves raw training only
- ignores export, artifact, or eval consequences

### 4. No Code Surface
- cannot name the patch region
- depends on hand-wavy future work

### 5. Too Small For The Stage
- likely gain below noise floor
- only plausible as a child of a stronger parent

### 6. Confounded By Construction
- mixes multiple causal stories in one initial slot
- cannot be interpreted if it wins

If an idea fails any of these, kill it before scheduling.

## Pre-Run Expected Impact Analysis

Every candidate must include an explicit expected-impact estimate before it is allowed into the slate.

Use this format:

### Expected Impact

- Primary lane:
  - `train`
  - `deploy`
  - `eval`
  - `throughput`
  - `artifact`

- Expected effect size:
  - `large`: `>= 0.01 BPB`
  - `medium`: `0.003-0.01 BPB`
  - `small`: `< 0.003 BPB`

- Why that scale is plausible:
  - concrete mechanism, not analogy-only

- Earliest horizon where signal should appear:
  - `90s`
  - `180s`
  - `600s`
  - `full champion`

- Failure signature:
  - what bad result would mean the idea is structurally wrong

- Stackability:
  - what kind of winner it could combine with later

If the expected impact is `small`, the idea should not be a lead slot.

## The Required Initial Idea Fields

Every idea should be written in this template:

### Name

Short mechanism name.

### Broken Invariant

What assumption this idea explicitly breaks.

### Mechanism

What actually changes in the system.

### Why It Could Beat The Current Default

Why this is big enough to matter against a strong SOTA-aligned base.

### Expected Impact

- lane
- scale
- horizon

### Full-Picture Tradeoff

- what it improves
- what it may damage
- why net gain is still plausible

### Patch Surface

- file(s)
- function(s)
- env / patch gate

### Kill Rule

What result kills it immediately.

### Child Path

If it wins, what more specific descendant should be tested next.

## Coverage Requirement For Any Initial Slate

Before running, the slate must include at least one idea from each:

1. Phase split
2. Deploy-alignment
3. Budget reallocation
4. Checkpoint or export selection
5. Parameter-family split
6. Data or context split
7. Wildcard anti-dominant mechanism

If one of these buckets is empty, the slate is not broad enough.

## Distinctness Requirement

At least half of the initial ideas must differ on two or more of:

- bottleneck attacked
- phase targeted
- score lane attacked
- object changed
- expected failure signature

If too many ideas differ only by coefficient or exact schedule shape, the slate is invalid.

## What Strong Initial Ideas Tend To Look Like

Strong initial ideas usually do at least one of:

- internalize deployment damage into training
- stage behavior by warmup / bulk / late phase
- reallocate bytes to buy a different architecture or export rule
- choose the exported object explicitly instead of implicitly
- separate tensor families or objectives that were previously coupled

Weak initial ideas usually do:

- one more helper patch
- one more late schedule tweak
- one more local variant of a current winner

## Final Rule

The generator should not ask:

- what tweak might help?

The generator should ask:

- what assumption about the process is false, and what patch would exploit that?

That is the bar for initial ideas now.





The issue is that our hypothesis generator keeps sampling from too small a neighborhood of the space.

We are good at generating:
- helper patches
- refinements
- local schedule changes
- mild architecture nudges
- mechanisms that help one metric lane

We are bad at generating:
- process splits
- objective splits
- budget reallocations
- mechanisms that change the dominant failure mode
- ideas that are large enough to survive a strong default

That is why the search keeps finding “interesting but too small.”

**What We Learned**
The winning PRs imply a very different hypothesis standard:

- the strongest wins are not micro-tweaks
- they usually combine a stronger model family with a better deploy path
- they often change when or where optimization pressure is applied
- they treat post-quant deployed loss as first-order
- they break a false invariant in the pipeline

Our generator mostly did not do that. It kept asking:
- what patch might help this stack?

It should ask:
- what assumption in this whole process is false?

**The Actual Failure In Hypothesis Generation**
We are under-sampling 5 operator classes:

- `stage`
  Different rules by warmup, bulk, late phase.
- `reallocate`
  Move budget from one subsystem to another.
- `externalize/internalize`
  Pull deploy loss into training, or push selection/export into a later step.
- `factorize`
  Separate parameter families, objectives, checkpoints, or contexts.
- `invert`
  Challenge the dominant story instead of polishing it.

Instead, we over-sample:
- `tighten`
- `specialize`
- minor `borrow`

That creates narrow search.

**What A Better Generator Must Do**
For every new stage, force generation across these axes:

1. Process split
- one regime for warmup
- one for bulk
- one for late phase
- ask what should change at each boundary

2. Objective split
- raw fit objective
- deploy/compression objective
- eval objective
- ask whether they should be optimized at different times

3. Budget split
- time
- bytes
- context
- parameter count
- ask where budget should be moved, not just tuned

4. State split
- raw weights vs EMA vs SWA vs best checkpoint
- ask whether “last checkpoint” is even the right object

5. Parameter-family split
- embeddings
- trunk matrices
- control scalars
- attention/value paths
- ask whether they should obey different late laws

6. Data split
- not just one shard order
- phase-specific ordering
- maybe different data regimes by phase

If a hypothesis does not break at least one of those in a meaningful way, it is probably too local.

**A Better Acceptance Rule**
Do not admit a hypothesis just because it is plausible.

Admit it only if:
- it changes a first-order causal story
- it would still matter against the current SOTA-aligned default
- it has a believable route to lower deployed score
- it is not just a coefficient version of an already-tested idea

**How To Generate Wider Hypotheses**
Use a forced template.

For each stage, require 1 candidate from each bucket:
- phase-split mechanism
- deploy-alignment mechanism
- architecture/capacity reallocation mechanism
- checkpoint/export selection mechanism
- parameter-family split mechanism
- data-order split mechanism
- wildcard anti-dominant mechanism

Then kill redundant ones.

That would have prevented `stage2_1` from being mostly “good helpers around one trunk.”

**The Core Shift**
The search should stop mutating “the model.”
It should start mutating:
- the training process
- the timing of objectives
- the object that gets exported
- the way budget is allocated across time and tensor families

That is where evolutionary search can still find a real winner.

If you want, I can turn this into a concrete `hypothesis_generation_failures.md` and a replacement generator rubric for future stages.