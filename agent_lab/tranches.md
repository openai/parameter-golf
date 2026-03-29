# Agent Lab Tranches

This file is the high-level research-program map.

Use it to answer:

- what tranche is active
- what question that tranche is trying to answer
- what is fixed
- what we already learned
- what comes next

For exact run results, use [`experiments.tsv`](./experiments.tsv). For longer reasoning, use the dated build log under [`docs/build-logs/`](../docs/build-logs/).

## T-20260328-A: Local Baseline Calibration

**Status:** completed

**Goal**  
Establish a usable local baseline on the 3090 stack and identify the first levers worth keeping.

**Fixed controls**

- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`

**Main findings**

- `NUM_KV_HEADS 4 -> 2` helped
- `TRAIN_BATCH_TOKENS 524288 -> 262144` helped a lot
- `MATRIX_LR 0.04 -> 0.06` helped modestly

**Key experiments**

- [`AL-20260328-001`](./experiments.tsv)
- [`AL-20260328-002`](./experiments.tsv)
- [`AL-20260328-003`](./experiments.tsv)
- [`AL-20260328-004`](./experiments.tsv)

**Deeper notes**

- [`docs/build-logs/2026-03-28-agent-lab.md`](../docs/build-logs/2026-03-28-agent-lab.md)

## T-20260329-A: Capacity vs Step Frontier

**Status:** active but mostly mapped

**Goal**  
Determine how much extra capacity the current local runtime can support inside a fixed `600s` budget, and whether extra depth only wins when the branch also gets more optimizer steps.

**Main question**  
Is the best local frontier on this stack “more depth plus more steps”, and if so, where does that frontier flatten?

**Fixed controls**

- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged

**What we tested**

- refreshed baseline on the current runtime
- `10` layers alone
- `10` layers with smaller batch
- `10` layers with much smaller batch
- `10` layers with cheaper attention via `NUM_KV_HEADS=1`

**What we learned**

- `10` layers alone was wrong because it lost too many steps
- `10` layers plus `196608` batch was a clear win
- `10` layers plus `131072` batch was only a marginal further improvement
- `10` layers plus `NUM_KV_HEADS=1` was worse than the best `kv2` branches

**Current best inside this tranche**

- [`AL-20260329-004`](./experiments.tsv) at `1.3913`

**Key experiments**

- [`AL-20260329-001`](./experiments.tsv)
- [`AL-20260329-002`](./experiments.tsv)
- [`AL-20260329-003`](./experiments.tsv)
- [`AL-20260329-004`](./experiments.tsv)
- [`AL-20260329-005`](./experiments.tsv)

**Stop or pivot rule**  
Stop this tranche when nearby reruns suggest the `196608` vs `131072` difference is mostly noise, or when more frontier pushes cost too much artifact headroom for too little quality.

**Likely next pivot**  
Move from “buy more steps” to “reallocate capacity more intelligently.”

**Deeper notes**

- [`docs/build-logs/2026-03-29-agent-lab.md`](../docs/build-logs/2026-03-29-agent-lab.md)

## T-20260329-B: Architecture Necessity Audit

**Status:** active

**Goal**  
Break the model into major components and ask, one family at a time, whether each piece is actually earning its bytes, compute, and optimization complexity.

**Main question**  
After the first capacity frontier is partly mapped, is the next gain more likely to come from a better distribution of capacity, or from simplifying or removing overbuilt structure?

**Fixed controls**

- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged

**Investigation families**

- MLP width versus depth
- residual controls and skip topology
- output path choices such as tying and logit softcap
- compression-aware architectural tradeoffs

**Working surface**

- [`architecture_review.md`](./architecture_review.md)

**Pivot rule**  
If a family shows only noise-level differences after a few well-chosen runs, move to the next component instead of overfitting one local knob.

### B1: MLP Width vs Depth

**Status:** completed

**Research question**  
With the current `10`-layer line, are we getting more value from extra transformations, or would some of that budget work better as fatter MLPs?

**Why this sub-tranche now**

- depth already proved it can help when step-starvation is fixed
- we still have not asked whether `MLP_MULT=2` is too small, too large, or simply the wrong place to spend capacity

**Controls for this 5-run set**

- use env vars rather than code edits
- keep `NUM_KV_HEADS=2`
- keep `MODEL_DIM=512`
- keep `NUM_HEADS=8`
- keep tied embeddings, tokenizer, and validation unchanged
- use the full `600s` training cap
- use `final_int8_ttt_lora` as the primary metric

**Anchor**

- [`AL-20260329-003`](./experiments.tsv) is the cleanest comparison point because it is strong at `1.3916` and leaves more artifact headroom than [`AL-20260329-004`](./experiments.tsv)

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `B1-E1` | `10L / MLP1 / batch 196608 / kv2` | Test a thinner MLP at fixed depth | The current `MLP_MULT=2` may already be wider than this regime needs | Whether reducing MLP width helps by freeing compute without losing too much quality |
| `B1-E2` | `10L / MLP3 / batch 196608 / kv2` | Test a wider MLP at fixed depth | The current model may be under-spending capacity inside each block | Whether pure width helps before we change layer count |
| `B1-E3` | `11L / MLP1 / batch 196608 / kv2` | Reallocate width into more layers | The best use of budget may be deeper but thinner blocks | Whether more transformations beat block-internal width |
| `B1-E4` | `9L / MLP3 / batch 196608 / kv2` | Reallocate depth into more width | Some capacity may be better spent inside each block than on one extra layer | Whether width can replace a layer cleanly |
| `B1-E5` | `9L / MLP3 / batch 131072 / kv2` | Test width with step recovery | Width may also need more optimizer steps, just like depth did | Whether a width loss is fundamental or just another fixed-budget step problem |

**Decision rule for B1**

- if `B1-E1` beats the anchor, the current MLP is likely too wide
- if `B1-E2` beats the anchor, pure width deserves a larger follow-up tranche
- if `B1-E3` wins, the model is probably under-layered relative to its MLP size
- if `B1-E4` or `B1-E5` wins, the next tranche should become width-aware rather than purely depth-aware

**Outcome**

- `B1-E1` (`10L / MLP1`) lost: thinning the MLP at fixed depth bought steps and headroom, but not enough quality
- `B1-E2` (`10L / MLP3`) lost badly and broke the artifact cap
- `B1-E3` (`11L / MLP1`) also lost, so deeper-but-thinner does not beat the current `10L / MLP2` balance
- `B1-E4` (`9L / MLP3 / 196608`) was the first promising width branch, but still slightly over the cap and still behind the valid anchor
- `B1-E5` (`9L / MLP3 / 131072`) produced the best raw score so far at `1.3899`, which strongly suggests width needed more steps, but it is invalid at `17.68 MB`

**Reading**

- width is not dead, but it only became competitive after both reducing depth and recovering more steps
- the best valid frontier is still [`AL-20260329-004`](./experiments.tsv) at `1.3913`
- the most interesting follow-up is no longer “is width good?” but “can the `9L / MLP3` winner be made challenge-valid without losing its score?”

## T-20260329-C: Width Winner Size Recovery

**Status:** completed

**Goal**  
Take the raw width-biased near-miss, [`AL-20260329-010`](./experiments.tsv), and learn which byte cuts preserve the score best while recovering challenge-valid size.

**Main question**  
Can the `9L / MLP3 / 131072 / kv2` branch be pulled under `16 MB`, and which structural cut loses the least performance per byte saved?

**Fixed controls**

- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged
- keep `TRAIN_BATCH_TOKENS=131072` unless the experiment explicitly says otherwise
- keep `NUM_KV_HEADS=2`
- keep tied embeddings

**Anchors**

- raw winner: [`AL-20260329-010`](./experiments.tsv) at `1.3899`, but invalid at `17,680,105` bytes
- current best valid comparator: [`AL-20260329-012`](./experiments.tsv) at `1.3838`

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `C1-E1` | `9L / MLP3 / DIM480 / batch 131072 / kv2` | Small dimension trim | A mild `MODEL_DIM` cut may recover enough bytes while preserving most of the width gain | Whether width can survive a modest global shrink |
| `C1-E2` | `9L / MLP3 / DIM448 / batch 131072 / kv2` | Stronger dimension trim | A larger `MODEL_DIM` cut may cross under the cap with an acceptable quality hit | How steep the score-vs-dim tradeoff is around the raw winner |
| `C1-E3` | `9L / MLP2 / DIM512 / batch 131072 / kv2` | One-notch MLP shrink | Most of the width gain may survive with `MLP_MULT=2` once steps stay high | Whether the last MLP notch is the main byte offender |
| `C1-E4` | `8L / MLP3 / DIM512 / batch 131072 / kv2` | One-layer trim instead of width trim | The 9th layer may be less valuable than the third MLP notch in this regime | Whether depth or width is the cheaper place to save bytes |
| `C1-E5` | `8L / MLP3 / DIM480 / batch 131072 / kv2` | Two mild trims together | Two small cuts may preserve score better than one aggressive cut | Whether combined light cuts dominate single hard cuts |

**Decision rule for C**

- if `C1-E1` or `C1-E3` is valid and close to `1.3899`, width has a clear path to a challenge-valid frontier
- if only the more aggressive trims become valid, the next question becomes whether optimization can claw back the lost score
- if none of the five get close to the raw winner, the width branch may be too byte-hungry in its current form

**Results so far**

- [`AL-20260329-011`](./experiments.tsv) (`C1-E1`, `9L / MLP3 / DIM480 / 131072 / kv2`) proved a mild global dim trim is enough to recover size validity, but not enough score. It landed at `1.3970` and 15.64 MB.
- [`AL-20260329-012`](./experiments.tsv) (`C1-E3`, `9L / MLP2 / 512 / 131072 / kv2`) produced a much stronger answer: `1.3838`, 14.73 MB, valid, and the new best frontier. This suggests the third MLP notch was the wrong place to spend bytes once the high-step regime was already in place.
- [`AL-20260329-013`](./experiments.tsv) (`C1-E2`, `9L / MLP3 / DIM448 / 131072 / kv2`) showed that stronger global shrinking can recover much more of the width branch than `DIM480` did. It finished at `1.3915` and 14.19 MB. Useful, but still clearly behind `9L / MLP2`.
- [`AL-20260329-014`](./experiments.tsv) (`C1-E4`, `8L / MLP3 / 512 / 131072 / kv2`) answered the “drop a layer instead” question. It finished at `1.3921`, but the artifact was still 16.29 MB, so the one-layer cut was not enough and was less effective than dropping one MLP notch.
- [`AL-20260329-015`](./experiments.tsv) (`C1-E5`, `8L / MLP3 / DIM480 / 131072 / kv2`) showed that two lighter cuts together are better than either fallback cut alone. It landed at `1.3906` and 14.42 MB. Valid, solid, but still not close enough to threaten the `9L / MLP2` winner.

**Current reading**

- structural trims matter more than uniform dim trims
- among global dim trims, `DIM448` is the first one that looks respectable
- one MLP-notch cut is currently dominating the dim-trim approach on both score and size
- dropping one layer alone is not the clean byte-saving move
- the combined-light-cuts backup is worth remembering, but the clear tranche result is that `9L / MLP2 / 131072` is the right survivor to optimize next

## T-20260329-D: Slim Winner Optimization Recovery

**Status:** active

**Goal**  
Take the two real tranche-C survivors and ask whether optimization or step-recovery can improve them further, with most of the attention on the new `9L / MLP2` winner.

**Main question**  
Is the new valid winner still step-limited or slightly under-tuned, and can the best fallback line be made genuinely competitive?

**Fixed controls**

- one training shard
- `600s` training cap
- primary metric `final_int8_ttt_lora`
- tokenizer and validation semantics unchanged
- focus on the most plausible size-recovered shapes from tranche C

**Why this tranche exists**

- tranche C already identified the structural winner
- B1 and tranche C both suggest strong interactions between shape and step count
- the next informative question is now optimization, not another broad structural sweep

**Anchors**

- primary winner: [`AL-20260329-012`](./experiments.tsv) at `1.3838`, 14.73 MB
- fallback survivor: [`AL-20260329-015`](./experiments.tsv) at `1.3906`, 14.42 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `D1-E1` | `9L / MLP2 / batch 98304 / kv2` | More steps on the winner | The new winner may still be slightly step-limited inside 600s | Whether the frontier improves by cashing in more updates |
| `D1-E2` | `9L / MLP2 / batch 131072 / kv2 / MATRIX_LR=0.065` | Slightly higher matrix LR on the winner | The winner may want more aggressive matrix motion without changing its step count | Whether the remaining loss is optimizer mismatch rather than capacity |
| `D1-E3` | `9L / MLP2 / batch 98304 / kv2 / MATRIX_LR=0.065` | Interaction test on the winner | Extra steps and slightly higher LR may only work together | Whether the winner still has a two-knob optimization gain available |
| `D1-E4` | `8L / MLP3 / DIM480 / batch 98304 / kv2` | More steps on the best fallback | The smaller backup may look weak only because it has not fully cashed in its saved compute | Whether the fallback line deserves to stay alive |
| `D1-E5` | `8L / MLP3 / DIM480 / batch 98304 / kv2 / MATRIX_LR=0.065` | Interaction test on the fallback | The backup line may need both more steps and stronger updates to become interesting | Whether the backup is only one combo away from relevance |

**Decision rule for D**

- if one of the optimized slim candidates beats [`AL-20260329-004`](./experiments.tsv) while staying under the cap, it becomes the new valid frontier
- if optimization does not recover the slimmer candidates, the next tranche should pivot from width rescue toward compression-aware structural changes outside the width family
