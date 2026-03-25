# Hailmary Curation

This is the post-subagent curation pass for the moonshot set.

The raw [`hypotheses.md`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hailmary/hypotheses.md) was intentionally broad. The steelman and fault-finding pass says the set is useful, but too duplicated to be a clean patch pack.

## 2026-03-24 Update

The latest upstream PR scan changes two parts of the curation:

- curriculum is no longer too speculative to track; shard ordering has now shown measurable leaderboard movement
- K-LoRA plus Min-NLL should be treated as a distinct eval-time adaptation protocol, not bundled into generic eval maximization or generic LoRA TTT

## 2026-03-25 Principle Check

After the `stage2_1` postmortem and the new initial-idea bar, the main `hailmary` issue is no longer breadth on paper. It is breadth in the runnable slate.

Current `hailmary` still overweights what was easy to patch:

- export alignment
- eval maximization
- explicit priors
- geometry
- throughput

The new bar says that is not enough.

The next lead moonshot set must be anchored more heavily in:

- phase-split training
- checkpoint/export selection
- parameter-family late rules
- two-stage curriculum
- two-stage context budget
- alternating objective microcycles

So the right reading now is:

- current packs are useful runnable probes
- the real rebuild target is the principle-aligned pack in [`run_configs.json`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hailmary/run_configs.json) and the design note in [rebuild.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hailmary/rebuild.md)

## What The Subagents Agreed On

The genuinely distinct high-upside lanes are:

1. export / quantization
2. eval policy
3. explicit priors
4. architecture/context specialization

They also agreed on what is not a clean first-wave hypothesis:

- capacity reinvestment by itself
- full-stack "simpler trunk" narratives
- generic throughput stories without a measured bottleneck

## Distinct First-Wave Patch Families

### P1. Export Alignment

- core story:
  - reduce deployed loss directly
- representative patches:
  - Full GPTQ
  - active late QAT aligned to export
  - checkpoint selection by deployed score
- expected outcome:
  - `0.005` to `0.020` BPB if export is the main remaining bottleneck
- why distinct:
  - same-checkpoint bakeoff cleanly isolates `L_export`

### P2. Eval Maximization

- core story:
  - change the scoring context, not the trained model
- representative patches:
  - exact overlap-aware eval
  - doc-aware eval reset
  - legal TTT children
  - K-LoRA plus Min-NLL selection as a separate child protocol
- expected outcome:
  - `0.005` to `0.025` BPB
- why distinct:
  - same-checkpoint eval bakeoff cleanly isolates `L_eval`
  - note: the stronger K-LoRA plus Min-NLL branch is a fresh-port target on the current merged root, not an already-exposed toggle

### P3. Explicit Low-Order Prior

- core story:
  - inject transition structure the trunk cannot learn fast enough
- representative patches:
  - CountInitBigram
  - trigram sidecar
- expected outcome:
  - `0.003` to `0.015` BPB
- why distinct:
  - attacks `L_repr` through explicit short-range statistics, not better export or eval

### P4. Phase-Aligned Training

- core story:
  - late training should optimize the deployed artifact, not the raw checkpoint
- representative patches:
  - EMA export
  - tight SWA
  - active late QAT
- expected outcome:
  - `0.003` to `0.012` BPB
- why distinct:
  - attacks train/export mismatch, not model class directly

### P5. Context / Value Specialization

- core story:
  - context and value transport are under-modeled
- representative patches:
  - XSA-all or stronger XSA placement
  - VRL
  - VE128
- expected outcome:
  - `0.002` to `0.012` BPB per mechanism, potentially larger in a funded stack
- why distinct:
  - attacks `L_repr` through the content/context path rather than explicit priors

## Reframed / Deferred

- capacity reinvestment:
  - child of export success, not a lead mechanism
- throughput reinvestment:
  - only after profiling proves a real step bottleneck
  - not a lead moonshot family anymore
- curriculum:
  - promoted from vague moonshot to real second-wave lane because open `#650` gives it measured effect
- simpler-core branch:
  - composition result, not a patch primitive
- geometry:
  - useful support lane, but no longer a lead moonshot family after the `stage2_1` finalist failure

## Expected Outcomes That Survived Review

The subagents were broadly aligned on these ranges:

- export alignment:
  - best upside of the set
- eval maximization:
  - highest-confidence non-training score mover
- explicit priors:
  - real moonshot if low-order structure is the underfit region
- phase alignment:
  - medium upside, high plausibility
- context/value specialization:
  - real but more interaction-sensitive

## Patch-Worthy First Wave

The first patch wave should be:

1. exact overlap-aware eval
2. CountInitBigram
3. active late QAT
4. EMA export

The next wave should be:

1. Full GPTQ
2. tight SWA
3. XSA-all / stronger XSA placement
4. curriculum / shard ordering
5. VRL / VE128
6. K-LoRA plus Min-NLL as an explicitly separate TTT-protocol branch

Reason:

- first-wave items are distinct enough and map cleanly to the current root script
- second-wave items are either larger ports or more interaction-sensitive

## Rebuild Priority

Once the current easy patches are exhausted, the next true `hailmary` build-out should prioritize:

1. deployed checkpoint selection
2. late deploy alignment
3. two-stage curriculum
4. parameter-family late freeze
5. two-stage context budget
6. alternating objective microcycles
