# Scientific Plan: From Working Prototype to Optimized Beast

## Operating hypothesis

**Primary hypothesis:** the crawler’s value is not raw recurrence by itself. The durable gain likely comes from a narrow combination of:

- width unlocked by sharing
- a small regularization / representational benefit from the shared bottleneck
- oracle coupling that exploits the crawler’s long-range signal
- post-processing choices that either preserve or destroy that signal

**Secondary hypothesis:** Triton/CUDA kernel work may matter, but mainly as an **enabler** for broader search and for preserving numerics in the true bottlenecks, not as magic by itself.

## What success looks like

We do not optimize a single number in isolation.

For every arm we should record:
- model-only validation BPB
- n-gram / final BPB
- compressed artifact size
- pre-quant vs post-quant BPB gap
- throughput (tok/s or step time)
- stability notes (NaNs, compile breaks, drift, runaway quant gap)

## Stage 0 — Lock the specimen

Status: done.

- frozen reference copy: `locked/Bandit_locked/`
- writable baseline: `working/Bandit_stable/`
- provenance and hashes recorded in `manifests/`

## Stage 1 — Reproduce baseline in the lab

Goal: establish a trustworthy Bandit baseline before any ablation.

### Hypothesis 1
Bandit must be reproducible in the Octavian lab before we trust any future deltas.

### Actions
- run the stable copy unmodified
- capture:
  - seed
  - wallclock
  - model-only BPB
  - final n-gram BPB
  - export size
  - quantization gap
- repeat on at least 2-3 seeds if budget allows

### Deliverable
A baseline table inside the lab so every future ablation is compared against the same specimen.

## Stage 2 — Separate the crawler into true causal components

Goal: identify what actually contributes signal.

### Family A: width vs sharing vs loops

**Hypothesis A1:** most of the gain is width from reduced unique depth.

Arms:
- flat width-matched control
- shared crawler baseline
- fewer loops
- more loops
- same effective depth, different unique/shared mix

Need to compare under approximately matched parameter count and matched post-processing.

### Family B: instruction mechanism

**Hypothesis B1:** FLOW instructions are doing more useful work than raw looping.

Arms:
- FLOW on (current)
- instructions off
- static orthogonal offsets fallback
- reduced `INST_DIM`
- increased `INST_DIM`
- tied vs untied loop up-projections

Questions:
- Does FLOW improve model-only BPB?
- Does it improve post-quant robustness?
- Is the bottleneck too small / too large?

### Family C: DeltaNet

**Hypothesis C1:** DeltaNet may help only if implemented with the fast/causal kernel path and tuned carefully; otherwise it may be dead weight or instability.

Arms:
- DeltaNet off (baseline)
- DeltaNet on with small head count
- DeltaNet on with canonical chunk delta rule path
- compare Python fallback vs FLA kernel if both available

Questions:
- Does it lower model-only BPB?
- Does it survive quantization?
- Does it actually help the oracle downstream?

## Stage 3 — Attack the actual weak point: post-processing destruction

This is the highest-priority scientific target.

### Hypothesis 3
The crawler’s small real advantage is being damaged during SWA / quantization / export, not during raw training.

### Family D: SWA / EMA fragility

Arms:
- SWA on vs off
- lower / higher `SWA_EVERY`
- earlier vs later EMA start
- reduced EMA decay smoothing
- disable only for crawler-sensitive runs

Questions:
- Does pre-quant BPB improve or worsen?
- Does the post-quant gap shrink?
- Is there a setting where the crawler advantage survives export?

### Family E: quantization policy

**Hypothesis E1:** the shared crawler block needs a different quantization treatment than the flat path.

Arms:
- crawler int8 on vs off
- loop-aware GPTQ on vs naive quantization
- quantize flat first / crawler second calibration order
- preserve instruction path at higher precision
- preserve delta path at higher precision when enabled

Questions:
- Which submodules are causing the quant gap?
- Is the crawler block itself the issue, or the instruction/control tensors around it?

### Family F: quantization sensitivity map

This should be a surgical ablation, not a guessing contest.

Submodule groups:
- flat blocks
- crawler block(s)
- loop instruction projection
- loop up-projections
- final norm / output head
- DeltaNet projections if enabled

Measure BPB hit from forcing each group to safer precision.

## Stage 4 — Oracle coupling and final-BPB improvement

Goal: not just compress, but improve final BPB.

### Hypothesis 4
The crawler may produce a better uncertainty structure than a flat model even if raw BPB gains are small. The oracle/mixer may not yet be exploiting that structure efficiently.

### Family G: entropy-adaptive alpha / Cubric

Arms:
- fixed alpha
- adaptive alpha current default
- narrower alpha range
- wider alpha range
- Cubric cadence sweep
- no Cubric warm-start vs warm-start

Questions:
- Does Bandit prefer a different alpha regime than X-WING?
- Are we overtrusting or undertrusting the oracle when crawler entropy changes?

### Family H: learned mixer

Arms:
- no learned mixer
- mixer on with current neural floor
- lower neural floor
- higher neural floor
- train mixer on crawler-specific features only if needed later

Questions:
- Can the mixer rescue cases where the crawler is better on hard tokens but slightly worse globally?
- Does learned blending convert crawler signal into lower final BPB?

## Stage 5 — Triton / kernel optimization track

This is a parallel systems track, not the first scientific ablation.

### Hypothesis 5
The most valuable kernel work is where it either:
1. preserves numerics in crawler-specific paths, or
2. dramatically expands search velocity.

### Best kernel candidates

#### K1. DeltaNet fast path
- If DeltaNet becomes promising, ensure the fast kernel path is always used.
- Investigate whether a Triton specialization beats the current path or improves stability.

#### K2. Shared n-gram scoring / table update path
- The eval stack does a lot of CPU/Numpy work.
- A Triton/CUDA path for chunk scoring / hashing / table lookup would massively accelerate search.
- This is likely one of the highest leverage engineering targets if eval is the bottleneck.

#### K3. Loop-aware quant calibration
- Hessian collection / calibration may be a hidden throughput sink.
- If calibration blocks experimentation, accelerating it is worth real effort.

#### K4. Crawler loop instruction path
- likely lower priority for speedup, but worth checking if repeated projection/expansion becomes hot at scale.

## Recommended execution order

### Priority 1 — immediate
1. reproduce Bandit baseline in Octavian lab
2. quantify pre-quant vs post-quant gap cleanly
3. run SWA/EMA/quant fragility ablations
4. run width-vs-sharing controls at matched budget

### Priority 2 — near-term
5. instruction path ablations (`INST_DIM`, FLOW on/off, static vs dynamic)
6. oracle alpha / Cubric retune specifically for crawler outputs
7. learned mixer evaluation

### Priority 3 — contingent
8. DeltaNet reintroduction only if it shows a clean gain
9. Triton work on the actual bottleneck identified by profiling

## The practical thesis

If I had to bet right now:

- the route to a beast is **not** "more loops"
- it is **preserve the crawler’s tiny real signal through post-processing**, then
- **retune the oracle/mixer to cash that signal into final BPB**, while
- using systems work to speed the search and protect numerics

## First experiment slate I would run

1. **Baseline reproduction**
2. **SWA on/off × crawler_int8 on/off**
3. **loop-aware GPTQ on/off**
4. **FLOW on/off with same width**
5. **`CRAWLER_LOOPS`: 2 vs 4 vs 6** at matched budget
6. **`INST_DIM`: 0 / 16 / 32 / 64**
7. **adaptive alpha range sweep for Bandit**
8. **mixer on/off with neural floor sweep**

That slate should tell us whether the main unlock is:
- architecture,
- post-processing,
- oracle coupling,
- or systems/runtime.
