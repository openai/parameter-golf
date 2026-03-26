# Cubric — Temporal Weight Sharing via Skiptrace

## Concept

A shared block (crawler bank) at the U-Net bottleneck fires periodically,
computes a refinement delta, and that delta is injected with learned
exponential decay on subsequent steps. The model gets the quality benefit
of weight-shared depth at near-zero compute cost.

**Training behavior:**
- Step N (crawler fires): `delta = bank(x) - x`, cache delta, inject at full strength
- Steps N+1..N+k: inject cached delta with `sigmoid(scale) * sigmoid(decay)^age`
- Eval: always fire bank (no caching)

**Learned parameters:**
- `crawler_decay_logit`: controls how fast the cached delta goes stale
- `crawler_inject_scale`: controls overall injection strength (starts at 0)

## Origin

Discovered during Frugendorff cadence ablation campaign (2026-03-24):
1. H1/H2: recursion (C-step double-firing) is overhead at all scales
2. H4-A/B: crawler bank learns better per step (+1.26% at step 1500)
   but loses on sliding due to 15% compute overhead
3. Insight: periodic firing + decaying injection gets the quality
   at ~1.5% overhead

## Research Axes

Each axis should be tested on a single GPU with small fast models
(8L/384d or smaller) to maximize iteration speed.

### Axis 1: Cadence Sweep
How often should the bank fire? Test cadences 4, 10, 20, 50.
- Hypothesis: there's a sweet spot where quality saturates but
  compute stays low. Too rarely = delta too stale. Too often =
  might as well run every step.

### Axis 2: Decay Behavior
Does the learned decay converge to a meaningful value?
- Monitor `sigmoid(crawler_decay_logit)` over training
- If it goes to 1.0: model wants the delta to persist forever (cache is stable)
- If it goes to 0.0: model kills the injection immediately (cache is useless)
- If it's 0.5-0.9: genuine temporal sharing is happening

### Axis 3: Injection Scale
Does the model learn to use the skiptrace?
- Monitor `sigmoid(crawler_inject_scale)` over training
- If it stays near 0: the concept doesn't help, model disables it
- If it grows: the model is actively using the cached delta

### Axis 4: Bank Depth
How many loops per firing? Test 1, 2, 3 loops.
- More loops = better delta but more compute per firing
- At cadence 10 with 3 loops, overhead is ~4.5% vs ~1.5% for 1 loop

### Axis 5: Model Scale
Does skiptrace help more on small or large models?
- Test on 6L/256d (tiny), 8L/384d (small), 11L/512d (GS v7)
- Hypothesis: helps more on small models (capacity-starved)

### Axis 6: Bank Position
Does the bank need to be at the bottleneck?
- Test: after encoder (bottleneck), middle of decoder, before final norm
- The bottleneck is the information pinch point — should be best

## Running Locally

All scripts in this folder default to NPROC=1 for single-GPU testing.
Override with NPROC=8 for multi-GPU pods.

```bash
# Single axis test (~2 min each on H100)
bash concepts/cubric/sweep_cadence.sh

# Full evaluation
bash concepts/cubric/eval_all.sh
```

## Key Files

- `train_cubric.py`: training script with skiptrace (copied from GS_v7_crawler_bank_cadence.py)
- `sweep_cadence.sh`: cadence 4/10/20/50 sweep
- `eval_all.sh`: all axes sequential
- `results/`: per-run output
