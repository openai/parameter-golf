# Stage 3.1 Hypothesis Stage Bar

Date: 2026-03-25

## Purpose

Before `stage3_1` gets repaired in code, its ideas need to clear a higher bar.

The problem is not only orchestration. The problem starts earlier:

- some ideas are interesting but not consequential enough
- some ideas do not state a clear causal `why`
- some ideas do not state expected impact in a way that can be screened before running
- some ideas do not say what failure would mean

So no new `stage3_1` hypothesis should become a slot until it passes this bar.

## Minimum Standard

Each hypothesis must specify all of the following.

### 1. Broken Invariant

What false invariant does the idea break?

Examples:

- one export policy for all layers
- one loss for the whole run
- one checkpoint is always the right export target
- one capacity allocation across all depths
- one late-phase behavior for all tensors

If the idea does not break a false invariant, it is probably too local.

### 2. Mechanism

What actually changes in the system?

This must be implementation-level, not slogan-level.

Good:

- per-layer bit allocation by Fisher rank
- staged objective with explicit phase boundaries
- companding inside quantization, with inverse at dequantization

Bad:

- “better quantization”
- “more aligned training”
- “information-theoretic export”

### 3. Why

Every idea must state the causal reason it should help the final score.

Required form:

- current bottleneck
- what the patch changes
- why that should move `final deployed val_bpb`

Example shape:

- current bottleneck: quantization damage is still large relative to raw loss
- change: non-uniform bit allocation sends bits to high-sensitivity layers
- why: the marginal loss increase per quantization error is not uniform across layers, so uniform int6 wastes bytes on low-value layers

If the `why` is not specific enough to falsify, the idea is not ready.

### 4. Expected Impact

Every idea must give a pre-run expectation.

Not just “could help.”

Required fields:

- dominant metric lane:
  - `train quality`
  - `export / quant damage`
  - `eval policy`
  - `throughput / more steps`
  - `artifact size`
- expected sign:
  - positive
  - neutral-risky
  - negative but enabling
- expected magnitude:
  - tiny: `< 0.001`
  - small: `0.001 - 0.003`
  - medium: `0.003 - 0.008`
  - large: `> 0.008`
- expected horizon:
  - sanity
  - screen
  - only long run

If the idea cannot state expected magnitude and horizon, it is not ready.

### 5. Observable Early Signal

What cheap signal should move before the expensive run?

Examples:

- `quant_damage` should shrink
- `post_quant_bpb` should improve at equal raw val
- `step_avg_ms` should improve materially
- `submission_size_bytes` should drop enough to justify reinvestment
- `pre_quant_bpb` should improve without worsening quant damage

If there is no cheap observable signal, the idea may still be valid, but it must be marked explicitly as `late-stage only`.

### 6. Matched Control

Each idea must state what it should be compared against.

Examples:

- export-only idea vs same-checkpoint export control
- architecture idea vs same training runner control
- child composition vs best solo parent, not raw baseline

If the idea does not declare a matched control, it is not ready.

### 7. Likely Failure Mode

Every idea must say how it could fail.

Examples:

- too much extra compute in the hot loop
- better raw loss but worse compressed score
- size improvement too small to fund a real reinvestment
- one lane dominates and the composition does not add
- effect appears only in short runs and reverses later

This matters because we want informative failures, not vague failures.

### 8. Kill Rule

Before running, define what would kill the idea.

Examples:

- no improvement beyond control noise at `180s`
- `step_avg_ms` worse by more than `10%` with no compensating gain
- artifact shrinks but no plausible reinvestment path exists
- solo mechanism loses badly, so the compound child is cancelled

No hypothesis should enter a pack without a kill rule.

### 9. Composition Rule

Every idea must say whether it is:

- a solo lead hypothesis
- a support/helper hypothesis
- a child-only composition hypothesis

This prevents the stage from promoting helper ideas as if they were lead mechanisms.

### 10. Code Burden

Every idea must say:

- patchable in current root script
- requires runner support
- requires export-only runner
- requires checkpoint/state plumbing
- requires architecture rewrite

If the code burden is not explicit, the stage becomes dishonest about what is runnable.

## Bar For Entry

An idea may enter `run_configs.json` only if:

1. it breaks a real false invariant
2. it has a concrete mechanism
3. it has a causal `why`
4. it has expected impact with magnitude and horizon
5. it has an observable early signal or is marked late-stage only
6. it has a matched control
7. it has a likely failure mode
8. it has a kill rule
9. it has a declared composition role
10. its code burden is explicit

If any one of these is missing, the idea should stay in notes, not in the runnable stage.

## What Does Not Clear The Bar

These do not clear the bar by default:

- retunes with no new causal story
- analogies with no implementation consequence
- ideas that only say “expected gain: 0.003” without a mechanism
- ideas that cannot say what should change in sanity/screen/final
- compounds introduced before the solo parent is understood

## What Clears The Bar

Examples of ideas that can clear the bar:

- a true export-only checkpoint-selection lane with deployed-score selection
- a staged curriculum that changes data order by phase
- a staged objective that is explicit about extra compute and expected gain channel
- per-layer bit allocation with a clear byte-neutral argument and same-checkpoint control

## Required Template

Every future `stage3_1` idea should be written in this template before implementation:

- `Name`
- `Lane`
- `Broken invariant`
- `Mechanism`
- `Why`
- `Dominant metric lane`
- `Expected impact`
- `Expected horizon`
- `Observable early signal`
- `Matched control`
- `Likely failure mode`
- `Kill rule`
- `Composition role`
- `Code burden`

## Short Version

The initial idea stage must exceed the bar before the code stage starts.

That means every idea must come with:

- a real `why`
- an expected impact
- an expected horizon
- an observable signal
- a failure mode
- a kill rule

If not, the stage is still under-specified even before any code is written.
