# Agent Lab Ideas

This is the high-level hypothesis bank. Not every idea should become an experiment immediately. Use this file to track what is active, what is new, and what has already been weakened by evidence.

## Active

### I-20260329-001 - Depth Needs Step Support

- Category: architecture + optimization
- Hypothesis: extra depth helps on this stack only if the branch also recovers enough optimizer steps inside the same `600s` budget.
- Why it might work:
- [`AL-20260329-002`](./experiments.tsv) showed depth alone was step-starved.
- [`AL-20260329-003`](./experiments.tsv) and [`AL-20260329-004`](./experiments.tsv) showed depth becomes competitive or winning when step count rises.
- Status: active
- Related tranche: [`T-20260329-A`](./tranches.md#t-20260329-a---capacity-vs-step-frontier)

### I-20260329-002 - Speed Recovery With Less Batch Noise

- Category: systems + architecture
- Hypothesis: there is a cleaner way to recover speed or artifact headroom than pushing batch ever smaller.
- Why it might work:
- `131072` batch only improved on `196608` by `0.0003`, which is close to noise.
- `NUM_KV_HEADS=1` was not the answer, but the question remains valid.
- Status: active
- Related tranche: [`T-20260329-A`](./tranches.md#t-20260329-a---capacity-vs-step-frontier)

## New

### I-20260329-003 - Compression-Aware Capacity

- Category: compression
- Hypothesis: some forms of capacity growth compress materially better than others, so raw parameter count is not the whole story.
- Why it might work:
- the current best branch is already close to the 16 MB cap
- Status: active
- Related tranche: [`T-20260329-C`](./tranches.md#t-20260329-c-width-winner-size-recovery)
- Evidence so far:
- [`AL-20260329-010`](./experiments.tsv) proved the raw width winner is strong, but too large to submit

### I-20260329-011 - Width Winner Can Be Saved By Mild Byte Cuts

- Category: architecture + compression
- Hypothesis: the raw winner likely does not need a dramatic redesign; a mild trim such as lower `MODEL_DIM`, one less MLP notch, or one less layer may recover validity while keeping most of the gain.
- Why it might work:
- the raw winner is only strong after the width-plus-step interaction clicks, so the safest next move is to shave bytes around that shape instead of abandoning it
- Status: active
- Related tranche: [`T-20260329-C`](./tranches.md#t-20260329-c-width-winner-size-recovery)
- Evidence so far:
- [`AL-20260329-011`](./experiments.tsv) says a mild global dim trim fixes size but gives back too much score
- [`AL-20260329-012`](./experiments.tsv) says a one-notch MLP trim is a much cleaner byte cut; it produced the new best valid frontier at `1.3838`
- [`AL-20260329-013`](./experiments.tsv) says global dim trimming is not hopeless, but even the stronger `DIM448` version still loses to the one-notch MLP trim
- [`AL-20260329-014`](./experiments.tsv) says cutting one whole layer is also a weaker byte-saving mechanism than cutting one MLP notch
- [`AL-20260329-015`](./experiments.tsv) says two lighter cuts together are a respectable backup, but still clearly weaker than the one-notch MLP trim

### I-20260329-012 - Smaller Valid Width Models Need Different Training Dynamics

- Category: optimizer
- Hypothesis: once width-oriented candidates are made smaller, some of the remaining loss versus the raw winner may be recoverable with more steps or a retuned matrix LR.
- Why it might work:
- B1 already showed a strong interaction between architecture and step count
- Status: active
- Related tranche: [`T-20260329-D`](./tranches.md#t-20260329-d-slim-winner-optimization-recovery)
- Evidence so far:
- [`AL-20260329-012`](./experiments.tsv) shows one size-recovered width-oriented survivor is already strong enough to deserve direct optimization follow-ups instead of more blind structural cuts
- [`AL-20260329-015`](./experiments.tsv) keeps a second valid survivor alive, so tranche D can compare “optimize the winner” versus “rescue the backup”
- [`AL-20260329-016`](./experiments.tsv) shows the main winner was still meaningfully step-limited; extra steps, not architecture changes, were the immediate source of the next large gain
- [`AL-20260329-017`](./experiments.tsv) shows a simple LR bump at the old batch is not the answer; any remaining optimizer gain likely has to be evaluated on top of the `98304` line, not instead of it
- [`AL-20260329-018`](./experiments.tsv) shows the `98304` winner also does not want this simple LR bump, so the default LR is currently the best setting among the tested options
- [`AL-20260329-019`](./experiments.tsv) shows the fallback line was also under-trained, but even after step recovery it still remains behind the main frontier
- [`AL-20260329-020`](./experiments.tsv) closes the loop: the fallback line also rejects the LR bump, so the optimizer lesson from tranche D is consistent across both survivors

### I-20260329-004 - Schedule or Optimizer Retune For 10L

- Category: optimizer
- Hypothesis: once `10` layers is no longer step-starved, the next gain may come from retuning learning dynamics rather than from further batch reduction.
- Why it might work:
- the current frontier suggests raw step-count gains are flattening
- Status: new
- Related tranche: none yet

### I-20260329-005 - Structural Capacity Instead of Pure Depth

- Category: architecture
- Hypothesis: some other capacity increase, such as a different projection or MLP structure, may beat the current `10`-layer frontier without exhausting artifact headroom.
- Why it might work:
- depth won, but only after step support, and size is now nearly capped
- Status: new
- Related tranche: none yet

### I-20260329-007 - MLP Width Versus Depth

- Category: architecture
- Hypothesis: moving some capacity budget from depth into MLP width could improve quality or compression efficiency more cleanly than another depth push.
- Why it might work:
- the current frontier suggests depth helps, but we have not yet asked whether `MLP_MULT` is the better place to spend parameters
- Status: active
- Related tranche: [`T-20260329-B`](./tranches.md#t-20260329-b---architecture-necessity-audit)
- Evidence so far:
- [`AL-20260329-007`](./experiments.tsv) says pure width at fixed depth is not promising in this naive form; it was slower, worse, and oversize
- [`AL-20260329-009`](./experiments.tsv) says width becomes more plausible when paired with one fewer layer, but it still trails the anchor and misses the size cap slightly
- [`AL-20260329-010`](./experiments.tsv) says width also needs step recovery; with more steps it became the best raw scorer, but the artifact failure got worse

### I-20260329-010 - The Current MLP May Already Be Too Wide

- Category: architecture
- Hypothesis: a thinner MLP plus one more layer may beat the current `10L / MLP2` balance because the model is over-spending capacity inside each block.
- Why it might work:
- in small models, MLPs can dominate parameter count quickly; more transformations may be a better use of budget than fatter hidden layers
- Status: active
- Related tranche: [`T-20260329-B`](./tranches.md#t-20260329-b---architecture-necessity-audit)
- Evidence so far:
- [`AL-20260329-006`](./experiments.tsv) says `10L / MLP1` alone is not enough; thinner blocks gained steps and headroom but still lost on `val_bpb`
- [`AL-20260329-008`](./experiments.tsv) says even `11L / MLP1` is not enough; moving width into one extra layer did not beat the `10L / MLP2` balance

### I-20260329-008 - Residual Controls And Skip Paths Are Overbuilt

- Category: architecture
- Hypothesis: some of `resid_mix`, `attn_scale`, `mlp_scale`, or `skip_weights` may be unnecessary or overly expensive in complexity relative to the quality they add.
- Why it might work:
- these controls are distinctive to this script and may be carrying historical baggage rather than current necessity
- Status: new
- Related tranche: none yet

### I-20260329-009 - Output Path Is Mismatched To The 10-Layer Regime

- Category: architecture
- Hypothesis: the current tying, initialization, or logit softcap choices may be leaving quality on the table now that the model is deeper and better trained.
- Why it might work:
- output-path choices affect both optimization behavior and compression cost, but have not been tested in the current frontier
- Status: new
- Related tranche: none yet

### I-20260329-013 - The Frontier Is Now Attention-Limited

- Category: attention
- Hypothesis: after tranche C fixed width allocation and tranche D fixed step count, the next meaningful gain may come from changing attention geometry rather than from more optimizer nudging.
- Why it might work:
- the current best line is stable on shape and optimization, but we have barely explored attention geometry beyond `kv1` versus `kv2`
- `NUM_HEADS`, `NUM_KV_HEADS`, and `QK_GAIN_INIT` are already exposed and can test this family cheaply and honestly
- Status: active
- Related tranche: [`T-20260329-E`](./tranches.md#t-20260329-e-attention-geometry-audit)
- Evidence so far:
- [`AL-20260329-021`](./experiments.tsv) says the frontier does respond to attention geometry; `q4/kv2` is better than the previous `q8/kv2` winner
- [`AL-20260329-022`](./experiments.tsv) says the direction is specifically toward fewer wider heads, not toward more narrower ones
- [`AL-20260329-023`](./experiments.tsv) says less KV sharing helps some, but not enough to beat the wider-head direction
- [`AL-20260329-024`](./experiments.tsv) says softer QK init is competitive but still secondary to the head-geometry win
- [`AL-20260329-025`](./experiments.tsv) says sharper QK init is the better side of the bracket, but the head-geometry win is still the dominant signal

### I-20260329-014 - The Frontier Is Now Output-Path-Limited

- Category: output path
- Hypothesis: after width allocation, step count, and first-pass attention geometry are improved, the next meaningful gain may come from the output path: tying, logit calibration, or output-specific learning dynamics.
- Why it might work:
- the output path is still mostly untouched in this repo's search history
- tying and logit softcaps directly affect calibration and expressivity in a tiny-model regime
- all the main knobs are already env-exposed, so this family is cheap to test honestly
- Status: active
- Related tranche: [`T-20260329-F`](./tranches.md#t-20260329-f-output-path-audit)
- Evidence so far:
- [`AL-20260329-026`](./experiments.tsv) says output expressivity is a first-class lever; untied outputs produced a large frontier jump without breaking the size cap
- [`AL-20260329-027`](./experiments.tsv) says output calibration matters on top of that; tighter logit clipping improved the untied frontier again
- [`AL-20260329-028`](./experiments.tsv) says the calibration result is directional; looser clipping lost to the tighter softcap

## Parked

### I-20260329-006 - KV1 As The Main Frontier Lever

- Category: attention
- Hypothesis: reducing `NUM_KV_HEADS` to `1` is a strong frontier move for the current `10`-layer setup.
- Why parked:
- [`AL-20260329-005`](./experiments.tsv) suggests `kv1` is not competitive with the best `kv2` depth branches on this stack
- Status: parked
