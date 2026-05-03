# Search Reset

## Goal

Start fresh with a search process that avoids local patch-chasing and instead searches over **bounded programs** that can be mutated, ranked, retired, and composed.

The key shift is:

> do not search patches  
> search **Phase -> Trigger -> Action -> Portfolio -> Selector** programs

---

## 1. Core shift

### Old pattern

- strong hypotheses in prose
- weaker code proxies
- mixed lanes in one tournament
- local patch tuning before the family premise is proven

### New pattern

- freeze one executable base
- search only a few first-order mechanism families
- compile each family into the **smallest decisive probe**
- mutate specs, not source files
- compose only surviving families

---

## 2. The unit of search

Each candidate should be a **bounded execution program**, not a patch pack.

### Candidate grammar

```text
candidate:
  base_variant
  family
  broken_invariant
  signals[]            # cheap observable state
  phase_boundaries[]   # early/mid/late or equivalent
  triggers[]           # threshold/event logic
  actions[]            # what changes when a trigger fires
  portfolio[]          # alternate late finishers/export modes
  selector             # how the winning artifact/state is chosen
  matched_control
  early_signal
  expected_horizon
  kill_rule
  composition_role
  code_burden
```

### Why this is the right abstraction

The repo already keeps rediscovering this same object:

- `hm2` -> bootstrap / handoff / receiver
- `stage3_2` -> controller spec
- `stage3_4` -> late finisher portfolio
- `stage3_5` -> event-triggered branch tournament

These are not separate species. They are the same search object in different forms.

---

## 3. Freeze the base

Do **not** search these at the same time as program search:

- tokenizer
- architecture depth/width
- training base stack
- frontier rebase

Pick one executable base and keep it fixed for a full search cycle.

If the base changes, start a **new cycle**.

This separates:

1. **base search**
2. **family search**
3. **within-family mutation**
4. **cross-family composition**

Right now those are getting mixed, which is why the search keeps collapsing into local noise.

---

## 4. Search families before children

The first question is:

> which broken invariant is real?

Not:

> what is the best tuned version of this idea?

### First fresh-start family set

Use only **3-4 first-order families**:

| Family | Broken invariant | Minimal probe |
| --- | --- | --- |
| **Handoff** | one mechanism should own all 600s | fixed bootstrap -> late receiver |
| **Selection** | last checkpoint/export state is best | late snapshot selector |
| **Portfolio** | one late finisher must be chosen in advance | 2-finisher late branch portfolio |
| **Eval/export isolation** | train/export/eval can be judged together | same-checkpoint export/eval bakeoff |

Do not start with compounds.

Do not start with more than four families.

---

## 5. Hypothesis compiler

No idea should become code until it is compiled into this contract:

1. **Broken invariant**
2. **Smallest decisive probe**
3. **Matched control**
4. **Early signal**
5. **Kill rule**
6. **Code burden**

If one of those is missing, the idea stays in notes.

### Example

#### Bad

- "late deployment intelligence"

#### Good

- broken invariant: the last checkpoint is the best export target
- smallest decisive probe: choose best of last `K` deployed late checkpoints
- matched control: same run, export final checkpoint only
- early signal: chosen checkpoint differs from final step
- kill rule: chosen checkpoint always collapses to final or EMA
- code burden: snapshot capture + selector only

---

## 6. Only implement primitives

Do not implement full stories first.

Implement a small primitive library:

1. fixed handoff
2. event trigger
3. checkpoint selector
4. late receiver
5. 2-finisher portfolio
6. export/eval-only mode

Everything else should be composed out of these.

If a hypothesis cannot be expressed with these primitives, either:

- reject it for this cycle, or
- add **one** new primitive and test that primitive directly

---

## 7. Tournament structure

## A. Family admission pack

Maximum 8 slots:

- 2 controls
- 4 family probes
- 1 novelty/challenger
- 1 repeat/diagnostic

Goal:

- decide which families are real
- retire fake families early

### Promotion rule

Promote only if:

- it beats matched control beyond control spread, **or**
- it clearly shows the exact early signal it was supposed to show

If controls are unstable, **abort ranking**.

## B. In-family mutation pack

Only for surviving families.

Mutate:

- boundaries
- triggers
- action magnitudes
- selector mode
- portfolio width/depth

Do not mutate unrelated architecture or base settings here.

## C. Cross-family composition pack

Only compose proven survivors.

Every composite must state:

- which families it combines
- why they should interact
- how they may interfere

No "try all good things together" packs.

## D. Long-run final

Only finalists.

Rank only on final deployed outcome.

---

## 8. Rules to avoid local minima

1. **No child before the family premise survives**
2. **No compound before two solo families survive**
3. **If controls drift, abort the pack**
4. **If the early signal does not appear, retire the family**
5. **If the lane is wrong, do not rank the result**
6. **Retire aggressively**

Use this heuristic:

- unproven family -> search breadth
- proven family -> search depth
- two proven families -> search composition
- anything else -> retire

---

## 9. What to log every run

Every run should emit machine-readable evidence:

- `manifest.json` - candidate spec as executed
- `metrics.json` - lane-specific metrics
- `events.json` - triggers, handoffs, branch choices
- `selection.json` - why the final artifact/state won
- `failure.json` - whether the declared failure mode happened

Without this, the harness is producing ideas, not learning.

---

## 10. How to know a pack is invalid

A pack is invalid if any of these happen:

- control repeat is wildly different from base control
- an eval/export hypothesis is still run as a full training job
- the candidate dies because the runner budget is wrong for its lane
- the stage claims a base rebase that the materialized run configs do not use

If the pack is invalid, do not rank it. Fix the harness first.

---

## 11. Practical next step for this repo

If starting over here, do this:

1. freeze one executable base
2. build one unified candidate schema around:
   - signals
   - phase boundaries
   - triggers
   - actions
   - portfolio
   - selector
3. implement only the primitive library
4. run one 4-family admission pack
5. retire half the families
6. mutate only the survivors
7. compose only surviving families

Do **not** create another stage folder first.

Create one reusable harness.

---

## 12. Short version

The core mistake has been searching for better patches when the real reusable object is a **bounded state-conditioned program**.

The fresh-start strategy is:

1. freeze the base
2. search families, not children
3. compile every hypothesis into a minimal executable probe
4. implement primitives, not stories
5. mutate specs, not source
6. compose only proven survivors
7. require machine-readable evidence from every run

That is how to narrow the search space without collapsing into local patch tuning.
