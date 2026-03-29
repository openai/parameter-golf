# Agent Lab Tranches

This file is the high-level research-program map. Each tranche should have a real question, explicit controls, and a stopping or pivot rule. Link back to exact experiments and detailed logs instead of repeating every metric inline.

## T-20260328-A - Local Baseline Calibration

- Status: completed
- Goal: establish a usable local baseline and identify the first major levers on the 3090 stack.
- Fixed controls:
- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- Main findings:
- `NUM_KV_HEADS 4 -> 2` helped
- `TRAIN_BATCH_TOKENS 524288 -> 262144` helped a lot
- `MATRIX_LR 0.04 -> 0.06` helped modestly
- Key experiments:
- [`AL-20260328-001`](./experiments.tsv)
- [`AL-20260328-002`](./experiments.tsv)
- [`AL-20260328-003`](./experiments.tsv)
- [`AL-20260328-004`](./experiments.tsv)
- Deeper notes:
- [`docs/build-logs/2026-03-28-agent-lab.md`](../docs/build-logs/2026-03-28-agent-lab.md)

## T-20260329-A - Capacity vs Step Frontier

- Status: active
- Goal: determine how much extra model capacity the current local runtime can support inside the fixed `600s` budget, and whether extra depth wins only when the branch also gets enough optimizer steps.
- Main question:
- Is the best local frontier on this stack “more depth plus more steps”, and if so, where does that frontier flatten?
- Fixed controls:
- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged
- What we tested:
- refreshed baseline on the current runtime
- `10` layers alone
- `10` layers with smaller batch
- `10` layers with much smaller batch
- `10` layers with cheaper attention via `NUM_KV_HEADS=1`
- Current findings:
- `10` layers alone was wrong because it lost too many steps.
- `10` layers plus `196608` batch was a clear win.
- `10` layers plus `131072` batch was only a marginal further improvement, suggesting the frontier is flattening.
- `10` layers plus `NUM_KV_HEADS=1` was worse than the best `kv2` depth branches.
- Current best:
- [`AL-20260329-004`](./experiments.tsv) at `1.3913`
- Key experiments:
- [`AL-20260329-001`](./experiments.tsv)
- [`AL-20260329-002`](./experiments.tsv)
- [`AL-20260329-003`](./experiments.tsv)
- [`AL-20260329-004`](./experiments.tsv)
- [`AL-20260329-005`](./experiments.tsv)
- Stop or pivot rule:
- Stop this tranche when repeated nearby runs suggest the `196608` vs `131072` difference is noise-level, or when new frontier pushes clearly trade too much artifact headroom for too little score.
- Likely next pivot:
- move from “buy more steps” to “free useful capacity or improve optimization without adding much more batch noise.”
- Deeper notes:
- [`docs/build-logs/2026-03-29-agent-lab.md`](../docs/build-logs/2026-03-29-agent-lab.md)

## T-20260329-B - Architecture Necessity Audit

- Status: active
- Goal: break the model into major components and ask, one family at a time, whether each piece is actually earning its bytes, compute, and optimization complexity.
- Main question:
- after the first capacity frontier is partly mapped, is the next gain more likely to come from a better distribution of capacity or from simplifying/removing overbuilt structure?
- Fixed controls:
- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged
- Planned investigation families:
- MLP width versus depth
- residual controls and skip topology
- output path choices such as tying and logit softcap
- compression-aware architectural tradeoffs
- Working surface:
- [`architecture_review.md`](./architecture_review.md)
- Planned pivot rule:
- if a family shows only noise-level differences after 2-3 runs, move to the next component instead of overfitting one local knob

### Tranche B1 - MLP Width vs Depth

- Research question:
- with the current `10`-layer line, are we getting more value from extra transformations, or would some of that budget work better as fatter MLPs?
- Why this tranche now:
- depth already proved it can help when step-starvation is fixed
- we have not yet asked whether the current `MLP_MULT=2` is too small, too large, or simply the wrong place to spend the next byte of capacity
- Controls for this 5-run set:
- use env vars rather than code edits
- keep `NUM_KV_HEADS=2`
- keep `MODEL_DIM=512`, `NUM_HEADS=8`, tied embeddings, tokenizer, and validation unchanged
- use the full `600s` training cap
- use `final_int8_ttt_lora` as the primary metric
- Main anchor for comparison:
- [`AL-20260329-003`](./experiments.tsv) is the cleanest width-vs-depth anchor because it is strong (`1.3916`) and leaves more artifact headroom than [`AL-20260329-004`](./experiments.tsv)
- Planned experiments:

#### B1-E1 - Anchor Replay

- Shape:
- `NUM_LAYERS=10`
- `MLP_MULT=2`
- `TRAIN_BATCH_TOKENS=196608`
- Goal:
- re-establish the clean comparison point for this tranche on the current stack before we judge width moves
- Hypothesis:
- the `10L x MLP2` line is still the best balanced starting point for width-vs-depth comparisons
- What it teaches:
- whether later differences are real architecture effects or just runtime noise

#### B1-E2 - Deeper But Thinner

- Shape:
- `NUM_LAYERS=11`
- `MLP_MULT=1`
- `TRAIN_BATCH_TOKENS=196608`
- Goal:
- test the opposite extreme: spend more budget on depth while making each layer cheaper
- Hypothesis:
- if the current model is over-spending on MLP width, more layers with a thinner MLP may train better inside the same wall-clock budget
- What it teaches:
- whether the current win is really about depth, or about total block capacity

#### B1-E3 - Mild Width Reallocation

- Shape:
- `NUM_LAYERS=9`
- `MLP_MULT=3`
- `TRAIN_BATCH_TOKENS=196608`
- Goal:
- test whether one less layer plus a moderately fatter MLP beats the current depth-biased anchor
- Hypothesis:
- some capacity is better spent inside each block than on one extra transformation step
- What it teaches:
- whether width can replace a layer cleanly at roughly similar training conditions

#### B1-E4 - Stronger Width Shift

- Shape:
- `NUM_LAYERS=8`
- `MLP_MULT=3`
- `TRAIN_BATCH_TOKENS=196608`
- Goal:
- push farther toward width and see whether the score keeps improving or falls apart
- Hypothesis:
- if width is the real missing ingredient, a shallower-but-wider model should remain competitive or improve while also changing the compression profile
- What it teaches:
- whether the frontier bends toward width, not just toward depth

#### B1-E5 - Width With Step Recovery

- Shape:
- `NUM_LAYERS=9`
- `MLP_MULT=3`
- `TRAIN_BATCH_TOKENS=131072`
- Goal:
- test whether width, like depth, only works after we recover more optimizer steps
- Hypothesis:
- a wider MLP may look mediocre at `196608` only because it is compute-starved; smaller batch may unlock it the same way it unlocked extra depth
- What it teaches:
- whether any width loss is fundamental, or just another fixed-budget step problem

#### Stop Rule For B1

- If `B1-E3` and `B1-E4` are both clearly worse than the anchor, width is probably not the next best place to spend capacity.
- If `B1-E2` wins or stays close, the model may still be under-layered relative to its MLP size.
- If `B1-E3` or `B1-E5` wins, the next tranche should move from pure depth to width-aware architecture design.
