# Stage 3.2 Bar Assessment

Date: 2026-03-25

## Verdict

`stage3_2` clears the idea-stage bar more convincingly than `stage3_1`, but not uniformly across all controller families.

The lead families that clearly clear the bar are:

- `H201 Late Deploy Gate`
- `H202 Best-State Controller`
- `H205 Alternating Objective Controller`
- `H204 Family-Split Warmdown`

These clear the bar because they all:

- break a real false invariant
- change a first-order process story
- have a believable route to lower deployed `val_bpb`
- are large enough to matter against a strong static base

The families that are useful but weaker as lead ideas are:

- `H203 Curriculum-by-State`
- `H207 Context Budget Controller`
- `H206 Systems-Aware Controller`

And the one that is too composite for initial admission is:

- `H208 Composite Late Policy`

## Family-by-Family Assessment

### H201 Late Deploy Gate

- Broken invariant:
  - one deploy-alignment law for the whole run
- Why it clears the bar:
  - many static deploy helpers are plausible late and harmful early
  - a gated late policy changes exactly that first-order process mismatch
- Expected lift:
  - `0.004 - 0.012 BPB`
- Why that scale is plausible:
  - late-QAT-like mechanisms and late geometry already show directional promise in prior stages
  - the likely win is not “more deploy pressure,” it is “deploy pressure at the right time”
- Earliest horizon:
  - `180s`, stronger at `600s`
- Composition role:
  - lead hypothesis

### H202 Best-State Controller

- Broken invariant:
  - the last checkpoint is the correct export target
- Why it clears the bar:
  - this attacks the train-to-deploy mismatch directly
  - checkpoint selection is a full-pipeline mechanism, not a local helper
- Expected lift:
  - `0.003 - 0.010 BPB`
- Why that scale is plausible:
  - EMA already suggests exported-state choice matters
  - best-of-late-state selection can win even when raw final loss does not
- Earliest horizon:
  - mostly `600s`
- Composition role:
  - lead hypothesis

### H203 Curriculum-by-State

- Broken invariant:
  - one data order for the whole run
- Why it partially clears the bar:
  - it is a real process split
  - but the likely gain is smaller and more interaction-sensitive than the top lead families
- Expected lift:
  - `0.002 - 0.006 BPB`
- Why that scale is plausible:
  - shard ordering has already looked close to relevant, but not dominant by itself
- Earliest horizon:
  - `180s`
- Composition role:
  - secondary lead or strong support

### H204 Family-Split Warmdown

- Broken invariant:
  - one late adaptation law for all parameter families
- Why it clears the bar:
  - parameter-family specialization is exactly the kind of false invariant we have not attacked enough
  - it changes late optimization and deploy robustness at once
- Expected lift:
  - `0.002 - 0.008 BPB`
- Why that scale is plausible:
  - embeddings/head/control tensors are unlikely to want the same late behavior as trunk matrices
- Earliest horizon:
  - mostly `600s`
- Composition role:
  - lead hypothesis

### H205 Alternating Objective Controller

- Broken invariant:
  - one blended objective every step
- Why it clears the bar:
  - this is a strong process mutation, not a coefficient retune
  - it has a believable path to internalize deploy loss without poisoning the whole trajectory
- Expected lift:
  - `0.004 - 0.015 BPB`
- Why that scale is plausible:
  - staged objective ideas are already among the strongest process-level concepts
  - sparse objective pulses may recover much of that value with less compute damage
- Earliest horizon:
  - `180s` to `600s`
- Composition role:
  - lead hypothesis

### H206 Systems-Aware Controller

- Broken invariant:
  - heavy mechanisms should stay enabled regardless of step cost
- Why it does not fully clear as a lead:
  - it is useful, but mostly as a support controller
  - by itself it is more likely to preserve wins than create them
- Expected lift:
  - `0.000 - 0.004 BPB`
- Why that scale is plausible:
  - it mainly prevents self-inflicted wallclock losses
- Earliest horizon:
  - `90s`
- Composition role:
  - support-only

### H207 Context Budget Controller

- Broken invariant:
  - one context policy for the whole run
- Why it partially clears the bar:
  - it is a real process split
  - but it is likely more interaction-sensitive and less directly causal than the top deploy families
- Expected lift:
  - `0.002 - 0.007 BPB`
- Why that scale is plausible:
  - context richness probably does not pay uniformly over all phases
- Earliest horizon:
  - `180s`
- Composition role:
  - secondary lead or support

### H208 Composite Late Policy

- Broken invariant:
  - late behavior should be controlled by one mechanism at a time
- Why it does not clear for initial admission:
  - it is too composite before the primitives are validated
  - if it wins or loses first, attribution is weak
- Expected lift:
  - potential `0.008 - 0.020 BPB`
- Why that scale is plausible:
  - it combines several first-order late-process levers
- Earliest horizon:
  - only long run
- Composition role:
  - child-only, not initial lead

## What This Means For Code

The first controller implementation should focus on:

1. `H201 Late Deploy Gate`
2. `H202 Best-State Controller`
3. `H205 Alternating Objective Controller`
4. `H204 Family-Split Warmdown`

And only then support:

5. `H203 Curriculum-by-State`
6. `H206 Systems-Aware Controller`

Do not implement `H208` first.

## Short Version

`stage3_2` passes the bar at the idea level if it is centered on:

- late deploy gating
- best-state selection
- alternating objectives
- family-specific late rules

Those are the controller families most likely to create a real lift instead of another small static-helper gain.
