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
- Status: new
- Related tranche: none yet

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

## Parked

### I-20260329-006 - KV1 As The Main Frontier Lever

- Category: attention
- Hypothesis: reducing `NUM_KV_HEADS` to `1` is a strong frontier move for the current `10`-layer setup.
- Why parked:
- [`AL-20260329-005`](./experiments.tsv) suggests `kv1` is not competitive with the best `kv2` depth branches on this stack
- Status: parked
