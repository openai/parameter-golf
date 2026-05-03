# Search Verifier Codex

## Purpose

This file is the **verifier** for the new search process.

Its job is not to invent hypotheses.

Its job is to answer:

> is this search track structurally valid?

For the current `search_harness.py` flow, many hard invariants are already enforced in code before the verifier runs:

- schema shape
- catalog membership
- single frozen base
- first-order candidate slots
- unique family groups
- frozen env alignment

So this verifier should act as a **cheap audit**, not a second generator.
It should spend effort on causal usefulness, distinctness, instruction match, and remaining compound risk.

That means checking whether a proposed search track:

- is searching the right unit
- is isolating the right causal lane
- is using the right controls
- is budgeting the tournament correctly
- is mutating at the right level
- is collecting enough evidence to learn

This verifier should be run:

1. **before** implementing a new family pack
2. **before** promoting survivors into mutation
3. **before** composing winners
4. **after** a pack runs, to decide whether the evidence is trustworthy

---

## What is a "track"?

A **track** is the whole executable search proposal for one cycle:

```text
track:
  base_contract
  candidate_grammar
  families[]
  family_admission_pack
  promotion_rules
  retirement_rules
  mutation_policy
  composition_policy
  primitive_backlog
  evidence_contract
```

The verifier is not judging whether the ideas are genius.

It is judging whether the track is **coherent enough to teach you something**.

---

## Core principle

The verifier should reject any track that collapses back into:

- patch-chasing
- lane mixing
- uncontrolled family proliferation
- unstable anchors
- non-falsifiable compounds
- no-evidence execution

---

## Verification gates

Every track must pass all of these gates.

## Gate 1: Base Freeze Gate

### Question

Does the track keep the executable base fixed for this cycle?

### Pass if

- tokenizer is fixed
- architecture base is fixed
- training base stack is fixed
- the same frozen base is used for all family admission comparisons

### Fail if

- some families silently include tokenizer or architecture shifts
- some candidates are evaluated on a different base than others
- frontier rebase is bundled into the same family search cycle

### Why this matters

If the base moves during family search, you are no longer comparing families.
You are comparing different problems.

---

## Gate 2: Unit-of-Search Gate

### Question

Is the track searching **bounded programs**, not loose patch packs?

### Pass if

Each family can be expressed in terms of:

- signals
- phase boundaries
- triggers
- actions
- portfolio
- selector

### Fail if

- the family is just a code diff category
- the family can only be described as "better export" or "better training"
- the proposal hides a full patch bundle behind one family name

### Why this matters

If the searchable object is not explicit, mutation and attribution will both fail.

---

## Gate 3: Broken-Invariant Distinctness Gate

### Question

Are the proposed families actually different first-order stories?

### Pass if

Each family breaks a different false invariant.

### Fail if

- two families are really threshold variants of the same idea
- a "portfolio" family is just a child of a "handoff" family
- a family differs only by aggressiveness, not by mechanism

### Why this matters

The first pack is for ontology search, not local tuning.

---

## Gate 4: Smallest-Decisive-Probe Gate

### Question

Has each family been reduced to the smallest executable probe that could falsify it?

### Pass if

Each family has:

- one decisive probe
- one matched control
- one expected early signal
- one kill rule

### Fail if

- the first executable version is already a full stack
- the family needs multiple unproven helpers to exist
- there is no cheap falsifier

### Why this matters

If the first executable form is too large, failure becomes uninterpretable.

---

## Gate 5: Lane Integrity Gate

### Question

Is each family being judged on the lane it claims to attack?

### Pass if

- eval/export-only ideas can be judged on the same checkpoint
- training-side ideas are not pretending to be export-only evidence
- lane-specific metrics exist

### Fail if

- an eval-policy hypothesis is run as a full training tournament when a same-checkpoint bakeoff was possible
- an export-heavy hypothesis is judged mostly on whether it fits in the training screen budget
- all families are ranked on one metric despite attacking different lanes

### Why this matters

Lane mixing is one of the main reasons this repo keeps producing ambiguous results.

---

## Gate 6: Anchor Integrity Gate

### Question

Are the controls strong enough and stable enough to anchor ranking?

### Pass if

- there is a matched control for the lane
- there is a repeat control or equivalent noise estimate
- ranking is blocked if controls diverge too much

### Fail if

- control repeat differs massively from base control
- candidates are ranked against a broken anchor
- no abort rule exists for invalid packs

### Why this matters

If the anchor is unstable, the ranking is fiction.

---

## Gate 7: Budget Alignment Gate

### Question

Does the tournament give each lane the right budget?

### Pass if

- training-heavy families get training budget
- export-heavy families get enough export/calibration budget
- eval-only families use same-checkpoint evaluation budget

### Fail if

- all families are forced through the same wallclock contract regardless of lane
- export-heavy families predictably die because export cost was ignored
- the screen measures scheduling debt instead of mechanism value

### Why this matters

Bad budget partitioning turns good ideas into false negatives.

---

## Gate 8: Mutation-Level Gate

### Question

Is the track mutating the right thing at the right time?

### Pass if

- family search happens before child mutation
- child mutation happens only after family survival
- composition happens only after solo survival

### Fail if

- the first pack already contains compounds
- within-family tuning starts before the family premise is proven
- family structure and child tuning are mixed in one step

### Why this matters

This is the main anti-local-minimum discipline.

---

## Gate 9: Composition Discipline Gate

### Question

Are compositions justified rather than opportunistic?

### Pass if

Every composite states:

- parent families
- expected interaction
- likely interference mode
- matched comparison against best solo parent

### Fail if

- the composite is just "all good things together"
- one or more parents are not yet validated
- there is no explicit reason the interaction should help

### Why this matters

Compositions are where search usually becomes muddy again.

---

## Gate 10: Evidence Contract Gate

### Question

Will the run emit enough evidence to teach the next cycle something?

### Pass if

The track requires:

- `manifest.json`
- `metrics.json`
- `events.json`
- `selection.json`
- `failure.json`

### Fail if

- only copied code and config are preserved
- trigger/handoff/branch events are not logged
- failure modes cannot be checked after the run

### Why this matters

Without a machine-readable evidence contract, the harness does not learn.

---

## Verdict classes

The verifier should return exactly one verdict.

### `PASS`

The track is structurally sound and can be run.

### `PASS_WITH_WARNINGS`

The track is mostly sound but has minor weaknesses that should be watched.

### `REDESIGN`

The track contains meaningful structural mistakes, but the core idea can be salvaged.

### `FAIL`

The track is too invalid to run usefully.

---

## Scoring rubric

Use this simple scoring system:

- each gate = `0`, `1`, or `2`
  - `0` = fail
  - `1` = weak / partial
  - `2` = clear pass

### Hard fail gates

If any of these are `0`, the track cannot get `PASS`:

- Gate 1: Base Freeze
- Gate 4: Smallest-Decisive-Probe
- Gate 5: Lane Integrity
- Gate 6: Anchor Integrity
- Gate 10: Evidence Contract

### Suggested interpretation

- `17-20` -> `PASS`
- `13-16` -> `PASS_WITH_WARNINGS`
- `9-12` -> `REDESIGN`
- `<9` -> `FAIL`

Hard fail gates override the total.

---

## Questions the verifier must answer

For every track, the verifier must explicitly answer:

1. Is the base truly frozen?
2. Are these families genuinely distinct?
3. Is each family reduced to the smallest falsifiable probe?
4. Are lane claims honest?
5. Are controls trustworthy?
6. Is the budget aligned to the lane?
7. Is mutation happening at the right level?
8. Are compositions disciplined?
9. Will the run leave enough evidence?
10. If this track fails, will we know **why**?

If the answer to the last question is "no", the track should not run.

---

## Recommended verifier output

Use this schema:

```json
{
  "verdict": "PASS|PASS_WITH_WARNINGS|REDESIGN|FAIL",
  "score": {
    "total": 0,
    "by_gate": {
      "base_freeze": 0,
      "unit_of_search": 0,
      "family_distinctness": 0,
      "smallest_probe": 0,
      "lane_integrity": 0,
      "anchor_integrity": 0,
      "budget_alignment": 0,
      "mutation_level": 0,
      "composition_discipline": 0,
      "evidence_contract": 0
    }
  },
  "hard_fail_gates": ["string"],
  "strengths": ["string"],
  "warnings": ["string"],
  "redesign_requirements": ["string"],
  "questions_to_resolve": ["string"],
  "run_recommendation": {
    "should_run_now": true,
    "must_fix_before_run": ["string"],
    "safe_scope_if_unfixed": ["string"]
  }
}
```

---

## Copy-paste agent prompt

Use this with an LLM or coding agent **after** it proposes a track.

```text
You are a verifier for a bounded-program search harness.

Your job is NOT to improve the track.
Your job is to decide whether the track is structurally valid.

You must verify the track against these gates:

1. Base Freeze
2. Unit of Search
3. Broken-Invariant Distinctness
4. Smallest-Decisive-Probe
5. Lane Integrity
6. Anchor Integrity
7. Budget Alignment
8. Mutation Level
9. Composition Discipline
10. Evidence Contract

Scoring:
- each gate = 0, 1, or 2
- hard fail gates:
  - Base Freeze
  - Smallest-Decisive-Probe
  - Lane Integrity
  - Anchor Integrity
  - Evidence Contract

Verdicts:
- PASS
- PASS_WITH_WARNINGS
- REDESIGN
- FAIL

Rules:
- do not reward creativity if the structure is invalid
- do not assume controls are valid unless the track explicitly says how they are validated
- do not allow compounds before solo family survival
- do not allow lane claims that the tournament cannot actually isolate
- do not allow a track that cannot explain failure

Return JSON only using this schema:

{
  "verdict": "PASS|PASS_WITH_WARNINGS|REDESIGN|FAIL",
  "score": {
    "total": 0,
    "by_gate": {
      "base_freeze": 0,
      "unit_of_search": 0,
      "family_distinctness": 0,
      "smallest_probe": 0,
      "lane_integrity": 0,
      "anchor_integrity": 0,
      "budget_alignment": 0,
      "mutation_level": 0,
      "composition_discipline": 0,
      "evidence_contract": 0
    }
  },
  "hard_fail_gates": ["string"],
  "strengths": ["string"],
  "warnings": ["string"],
  "redesign_requirements": ["string"],
  "questions_to_resolve": ["string"],
  "run_recommendation": {
    "should_run_now": true,
    "must_fix_before_run": ["string"],
    "safe_scope_if_unfixed": ["string"]
  }
}
```

---

## How to use this in practice

### Before coding

Use the verifier on the proposed track.

If verdict is:

- `PASS` -> implement
- `PASS_WITH_WARNINGS` -> implement carefully
- `REDESIGN` -> revise track first
- `FAIL` -> discard the track

### After a run

Run the verifier again, but now include actual evidence:

- did the controls hold?
- did the early signal appear?
- did the lane isolation actually happen?
- did the track emit the required files?

This prevents "interesting but invalid" runs from poisoning the next cycle.

---

## Short version

This codex exists to stop you from wasting time on tracks that are:

- conceptually appealing
- badly isolated
- weakly falsifiable
- under-instrumented
- or anchored to broken controls

It should be treated as a **hard gate** between:

1. hypothesis generation
2. implementation
3. tournament execution

If a track does not pass this verifier, do not run it.
