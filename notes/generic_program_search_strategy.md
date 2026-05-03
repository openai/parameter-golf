# Generic Program Search Strategy

This note captures the current generic strategy for a harness that optimizes **arbitrary programs**, not only training code.

The key lesson is:

**do not ask the model to invent the search space from raw source**

**build a searchable space, then run evolution inside it**

## 1. The search problem

For a general program, the search space is not "all possible edits."

It is:

**the set of bounded interventions on live bottlenecks, under hard contracts, scored by a cheap evaluation ladder**

The most basic question is:

**what can we vary, cheaply and independently, that changes the objective in a predictable direction?**

If we cannot answer that, we do not have a search space yet.

## 2. Step 0: define the measurement contract

Before any mutation, define what survival means.

At minimum:

- primary objective
- secondary cost metrics
- hard invariants
- cheap proxy signals
- full evaluation signal

Examples:

- latency under correctness tests
- quality under size/throughput constraints
- post-export metric rather than pre-export metric
- robustness under held-out or adversarial cases

If the measurement contract is wrong, the search will optimize the wrong thing very efficiently.

## 3. Step 1: build mutation maps, not idea lists

The first artifact for any target file or module should be a **mutation map**.

A mutation map should answer:

1. primary role
2. blast radius
3. local contracts
4. state read / state written
5. live intervention surfaces
6. cheap runtime signals
7. valid mutation families
8. danger zones
9. first safe experiments

This turns code into a **runtime search contract**.

### Generic mutation-map template

```md
primary_role: "..."
blast_radius: low|medium|high

local_contracts:
  - "..."

state_touched:
  reads:
    - "..."
  writes:
    - "..."

intervention_surfaces:
  - location: "function_or_block"
    kind: "threshold|schedule|cache|state_update|branch|partition|controller|..."
    leverage: low|medium|high
    risk: low|medium|high

cheap_signals:
  - metric: "..."
    why_it_matters: "..."

valid_mutation_families:
  - "..."

danger_zones:
  - "..."

first_safe_experiments:
  - "..."
```

## 4. Search object

The search object should be a **bounded intervention unit**, not a free-form idea.

Examples:

- env diff
- code diff
- controller diff
- small generated subroutine
- dataflow / scheduling rule change
- parameter-group partition rule

Each candidate should carry:

- target region
- operator family
- mutation scale
- parameters / diff
- expected cheap signal
- preserved contracts
- observed metrics
- novelty summary

The patch is the phenotype. The bounded intervention description is the genotype.

## 5. Proposal policy is separate from survival policy

There should be two different machines inside one loop:

1. **wild proposer**
2. **ruthless empiricist**

### Wild proposer

Allowed to emit:

- ugly logic
- asymmetric rules
- conditional controllers
- non-obvious operator compositions
- weird but bounded subprograms

Not filtered by:

- elegance
- readability
- human plausibility
- minimality

### Ruthless empiricist

Determines survival only by:

- invariant preservation
- executable surface bounds
- metric movement
- cost
- robustness
- novelty
- lineage value

The rule is:

**remove taste from proposal**

**put discipline into survival**

## 6. Mutation grammar

Mutate **operators**, not tokens.

### A. Knob operators

- `x -> x + delta`
- `x -> x * factor`
- enum switch
- boundary shift
- horizon widen / narrow

### B. Controller operators

- fixed -> signal-triggered
- always-on -> phase-gated
- one policy -> early/late split
- global threshold -> per-group threshold
- static allocation -> adaptive allocation

### C. Structural operators

- split one state/update group into several groups
- move one step before/after another
- activate a subsystem only in some region or phase
- change interaction between compute and export / cache / batching
- alter ordering between stages

### D. Synthesized-rule operators

- generate a small function `f(state, stats, step) -> scalar`
- generate a gating rule `g(context) -> bool`
- generate a partitioning rule
- generate a compression / caching / scheduling policy

## 7. Region map

Do not allow every queue to mutate every region.

Every target should be partitioned into regions like:

- low-risk control
- medium-risk behavior
- high-risk deep behavior
- forbidden / metric-defining zones

Region permissions should be derived from the mutation map, not hardcoded globally.

This is what keeps radical search bounded.

## 8. Portfolio queues

Do not run one unified queue.

Use multiple queues with different rules.

### Queue A — local exploit

Goal: steady gain  
Typical edits: scalars, thresholds, schedules, calibration knobs, cache sizes, per-group ratios

### Queue B — controller search

Goal: better policies  
Typical edits: delayed activation, adaptive schedules, phase switches, per-group control logic

### Queue C — structural search

Goal: open new basins  
Typical edits: multi-block changes, new grouping logic, compute movement across phases, alternate update ordering

### Queue D — radical synthesis

Goal: non-human discoveries  
Typical edits: small generated controller functions, odd gating rules, synthesized update or compression subroutines

### Budget rule

Budget should be expressed as a fraction of the **available** compute, not a fixed total like "100 GPU units."

Let total available compute be `B`.
Allocate fractions of `B`.
If `B` is small, Queue D can collapse to zero.
If archive quality is weak, Queue C/D can stay near zero.

The right optimization target is:

**expected improvement + expected information gain + expected lineage value per next unit of compute**

## 9. Evaluation ladder

Every candidate should pass through a multi-fidelity ladder.

### Stage 0 — static gate

Must pass:

- patch applies
- build / import / syntax succeeds
- signatures / contracts intact
- no forbidden region touched

### Stage 1 — smoke dynamics

Check:

- finite outputs
- no crashes
- no NaNs
- throughput not catastrophic
- memory within bound

### Stage 2 — cheap comparative screen

Compare with baseline or control on:

- early proxy
- runtime / memory
- objective-adjacent metric
- wallclock-adjusted utility

### Stage 3 — deeper continuation

Only candidates that earn more budget proceed.

### Stage 4 — expensive evaluation

Reserved for elites, promising structural mutants, and unusual lineages that justify more compute.

## 10. Queue-specific continuation

Different queues should have different survival rules.

### Queue A

Continue only if cheap score improves.

### Queue B

Continue if:

- cheap score improves, or
- trajectory shape looks promising at acceptable cost

### Queue C

Continue if:

- stable
- not clearly dominated
- behaviorally novel on a relevant axis

### Queue D

Continue if:

- invariants preserved
- behaviorally distinct
- at least one promising signal
- compute cost not explosive

This is how weird edits survive long enough to matter.

## 11. Novelty scoring

Novelty should be explicit.

Useful axes:

- behavioral novelty
- structural novelty
- controller-form novelty
- region / module coverage novelty
- metric-profile novelty

Behavior space matters more than diff space.

Examples:

- later but stronger improvement
- different cost/quality curve
- different robustness profile
- reduced export gap
- different branch / controller activation profile

## 12. Archive fronts

Do not keep one leaderboard.

Keep multiple fronts, such as:

- best objective
- best efficiency
- best novelty-survival tradeoff
- best lineage seeds
- best deployment / export / post-processing behavior

The archive should store:

- queue type
- operator family
- regions touched
- metrics by stage
- novelty score
- continuation decision
- descendant yield

Many candidates are valuable as parents, not as endpoints.

## 13. Parent selection

Parents should be selected differently by queue.

### Queue A

Use top current performers.

### Queue B

Use good performers plus unusual stable trajectories.

### Queue C

Use novelty front, lineage seeds, and partially good but underexplored candidates.

### Queue D

Use only robust unusual survivors, not junk novelty.

## 14. Instrumentation

To support non-obvious interventions, log not only win/loss but also behavioral changes.

Examples:

- branch / controller activation counts
- per-group statistics
- stage timings
- memory by phase
- objective gap between proxy and deployment metric
- subsystem activity over time
- export / post-processing divergence

Without this, weird candidates are impossible to diagnose or breed effectively.

## 15. Prompt roles

The model should not do everything in one prompt.

Use separate roles.

### Cartographer

Builds the mutation map and region map.

### Wild proposer

Generates bounded interventions inside the allowed grammar.

### Empirical judge

Scores survival using rules, metrics, and compute.

### Historian

Updates archive fronts and lineage value.

This keeps proposal free and judgment strict.

## 16. The shortest generic loop

For each round:

1. choose queue by available budget and current archive state
2. select parent from the queue-appropriate archive front
3. sample an operator family allowed in the chosen region
4. generate one bounded candidate
5. run static gate
6. run smoke and cheap comparative stages
7. score objective, cost, novelty, and continuation value
8. kill, archive, or continue
9. periodically breed from fertile lineages, not just top scorers

## 17. The main anti-patterns

Avoid:

- asking the model to infer the whole search space from raw code alone
- letting the proposer optimize for elegance
- forcing every queue to use the same continuation rule
- using one proxy as the only selection signal
- storing only winners
- collapsing radical and local edits into one queue
- letting novelty mean "random diff"

## 18. Short version

The generic strategy is:

- define the measurement contract
- build mutation maps
- search over bounded intervention units
- separate wild proposal from empirical survival
- use multiple queues with different continuation rules
- preserve novelty and lineage value
- allocate compute as fractions of the next available budget, not fixed totals

The harness should not ask:

**what edit seems smart?**

It should ask:

**what bounded intervention deserves the next unit of compute?**
