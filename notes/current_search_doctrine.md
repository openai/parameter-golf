# Current Search Doctrine

This note summarizes the current thinking after pushing on `pg_enigma`, breadth/depth exploration, and the failure modes of prompt-shaped search.

## The main mistake

We kept asking the model to do too many things at once:

1. infer the latent bottleneck structure of the problem from raw code
2. invent the search space
3. generate good interventions
4. stay inside executable reality

That collapses into human-shaped prompting. The result is not evolutionary search. It is guided brainstorming.

The fix is:

**do not ask the model to search the space**

**build a space it can search**

## First-principles reframing

The problem is not "what edit should we try?"

The problem is:

**what can we vary, cheaply and independently, that changes the objective in a predictable direction?**

If we cannot answer that, we do not have a search space yet. We only have text.

So the real first question is not:

- which path should I explore?
- what 10 ideas should I brainstorm?
- what family should I propose?

It is:

**what are the actual corridors of motion in this maze?**

## Step 1 is a mutation map

For a file, the first useful artifact is not a path slate or a hypothesis set. It is a **mutation map**.

The mutation map should answer:

1. what role this file plays
2. what local contracts it must preserve
3. what state it reads and writes
4. what intervention surfaces are actually live
5. what cheap runtime signals reveal whether a mutation here helps
6. what mutation families are valid for this file
7. what danger zones create fake wins or broken invariants
8. what 3 first safe experiments are worth trying

This is better than a leverage map because it turns the file into a **runtime search contract**, not just a diagnosis.

## The search object

The search object should not be "all possible edits."

It should be:

**bounded intervention units over live surfaces**

Examples:

- env diff
- code diff
- controller diff
- small generated subroutine

Each candidate should carry:

- target surface
- mutation family
- mutation scale
- coupled knobs / touched region
- expected cheap signal
- evaluation result

This is the genotype. The patch or realized change is the phenotype.

## Proposal policy is not survival policy

This is the biggest conceptual correction.

There should be two different machines inside one loop:

1. a **wild proposer**
2. a **ruthless empiricist**

### Wild proposer

The proposer should be allowed to emit interventions that are:

- ugly
- asymmetric
- conditional
- controller-shaped
- non-obvious
- not naturally human-legible

The proposer should not be filtered by:

- elegance
- plausibility
- readability
- minimality
- human taste

If those constraints are pushed into proposal, the search collapses around the baseline.

### Ruthless empiricist

Survival should be determined only by:

- hard invariants
- executable surface bounds
- cheap metric ladder
- robustness checks
- continuation value
- compute budget

So the rule is:

**remove taste from proposal**

**put discipline into survival**

## Mutation scales

The search grammar needs multiple scales.

### A. Scalar / threshold edits

Examples:

- LR
- EMA
- clip norms
- activation thresholds

### B. Policy / controller edits

Examples:

- when looping turns on
- when a subsystem becomes active
- time-based vs signal-based schedules
- layer coverage over time

### C. Structural behavior edits

Examples:

- move logic across phases
- split optimizer regimes
- insert delayed gates
- couple calibration to training signals

### D. Search-generated subprograms

Examples:

- scheduler function
- controller function
- partitioning rule
- calibration policy
- compression heuristic

If only A exists, search becomes tuning.
Real empirical discovery needs B/C/D to exist too, even if they get smaller budget.

## Semantic mutation, not syntactic mutation

Blind token mutation is too wasteful.
Human-plausible editing is too narrow.

The right unit is **semantic mutation operators** such as:

- replace a fixed schedule with a conditional controller
- partition parameters into different update regimes
- gate a subsystem by phase or signal
- move a computation from every step to every k-th step
- search over operator composition instead of only constants

These are still bounded, but they are large enough to find non-human wins.

## Novelty must be explicit

If selection rewards only immediate metric gain, the search collapses into local tuning.

The loop needs some explicit novelty pressure on:

- operator family
- code path
- touched module / region
- behavior profile
- metric profile

This does not mean rewarding randomness forever.
It means preserving enough diversity for weird but fertile lineages to survive long enough to get descendants.

## Archive lineages, not only winners

The archive should keep multiple fronts, not just a champion:

- best score
- best quantized-gap improvement
- best throughput-quality tradeoff
- most novel controller behavior
- weird but stable

This is how large interventions get a path to mature.

**Do not optimize one champion. Cultivate families.**

## Radical edits need their own budget and continuation rule

Large interventions should not be forced to compete directly against scalar tweaks under the same tiny proxy.

Use a portfolio like:

- 60-70% local exploitation
- 20-30% medium interventions
- 5-10% radical interventions

The radical bucket is allowed to:

- touch more regions
- alter controller logic
- synthesize small subprograms
- look ugly

It still must pass invariants.
It just gets evaluated under a different early continuation rule:

- is it behaviorally distinct?
- is it stable enough to continue?
- does it open a new basin?
- does it show promising secondary signals?

Not only:

- did the early scalar proxy improve immediately?

## What the controller should infer

The controller prompt should not ask for ideas.
It should infer:

- the current unit of heredity
- the coupling structure of the search space
- which operator families deserve probability mass now
- what novelty means in this regime
- when local search is deceptive
- when the representation itself is exhausted

This is a posterior over search operators, not a brainstorm over hypotheses.

## What this means for `pg_enigma`

The old breadth/depth design was still too human-shaped.
It was asking for:

- paths
- families
- narratives

That is useful as analysis, but it is not the real evolutionary core.

The better outer loop is:

1. mutation map for the active file / surface
2. choose one intervention surface
3. choose one mutation family
4. let a wild proposer emit one bounded candidate
5. run a cheap evaluation ladder
6. archive the result by surface / family / behavior
7. reweight the next proposal distribution

In other words:

**instrument -> map surfaces -> propose bounded mutations -> evaluate -> archive -> update search policy**

## What this means for the pr1394 training file

For `frontier_rebase/pr1394/train_gpt_human.py`, the mutation map is currently the best step-1 artifact.

It identifies:

- local contracts
- danger zones
- high-leverage surfaces
- cheap signals
- safe first experiments

The immediate searchable surfaces are things like:

- loop activation / recurrence timing
- EMA horizon
- optimizer schedule / parameter-group behavior
- GPTQ calibration / clip / bit settings
- attention / XSA / skip-path control surfaces

That map should become the runtime representation of the maze for this file.

## Short version

The current doctrine is:

- stop using human-shaped exploration as the search core
- build mutation maps first
- separate proposal from survival
- let the proposer be weird but bounded
- let empirical survival, not plausibility, decide what gets more compute
- preserve lineages and novelty
- search over controllers and subprograms, not only scalar knobs

The main question is no longer:

**what idea should we try?**

It is:

**what odd but bounded behavior deserves one more unit of compute?**
