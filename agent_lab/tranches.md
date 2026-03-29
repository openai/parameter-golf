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

**Status:** completed

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

- primary structural winner: [`AL-20260329-012`](./experiments.tsv) at `1.3838`, 14.73 MB
- fallback survivor: [`AL-20260329-015`](./experiments.tsv) at `1.3906`, 14.42 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `D1-E1` | `9L / MLP2 / batch 98304 / kv2` | More steps on the winner | The new winner may still be slightly step-limited inside 600s | Whether the frontier improves by cashing in more updates |
| `D1-E2` | `9L / MLP2 / batch 131072 / kv2 / MATRIX_LR=0.065` | Slightly higher matrix LR on the winner | The winner may want more aggressive matrix motion without changing its step count | Whether the remaining loss is optimizer mismatch rather than capacity |
| `D1-E3` | `9L / MLP2 / batch 98304 / kv2 / MATRIX_LR=0.065` | Interaction test on the winner | Extra steps and slightly higher LR may only work together | Whether the winner still has a two-knob optimization gain available |
| `D1-E4` | `8L / MLP3 / DIM480 / batch 98304 / kv2` | More steps on the best fallback | The smaller backup may look weak only because it has not fully cashed in its saved compute | Whether the fallback line deserves to stay alive |
| `D1-E5` | `8L / MLP3 / DIM480 / batch 98304 / kv2 / MATRIX_LR=0.065` | Interaction test on the fallback | The backup line may need both more steps and stronger updates to become interesting | Whether the backup is only one combo away from relevance |

**Results so far**

- [`AL-20260329-016`](./experiments.tsv) (`D1-E1`, `9L / MLP2 / 98304 / kv2`) landed at `1.3721` and 15.48 MB. This is a major frontier jump and strongly confirms that the tranche-C winner was still step-limited.
- [`AL-20260329-017`](./experiments.tsv) (`D1-E2`, `9L / MLP2 / 131072 / kv2 / MATRIX_LR=0.065`) landed at `1.3909` and 14.92 MB. This is a clear regression and says the big gain did not come from a simple LR mismatch at the old batch size.
- [`AL-20260329-018`](./experiments.tsv) (`D1-E3`, `9L / MLP2 / 98304 / kv2 / MATRIX_LR=0.065`) landed at `1.3786` and 15.55 MB. Still strong, but worse than `98304` alone, so the step win does not want this LR bump on top.
- [`AL-20260329-019`](./experiments.tsv) (`D1-E4`, `8L / MLP3 / DIM480 / 98304 / kv2`) landed at `1.3808` and 15.19 MB. This is a real rescue of the fallback line, but it still does not overtake the main `9L / MLP2 / 98304` frontier.
- [`AL-20260329-020`](./experiments.tsv) (`D1-E5`, `8L / MLP3 / DIM480 / 98304 / kv2 / MATRIX_LR=0.065`) landed at `1.3853` and 15.28 MB. The fallback line also rejected the LR bump.

**Current reading**

- extra steps matter more than we thought on the `9L / MLP2` line
- LR alone does not rescue the old-batch line
- the `98304` winner also does not improve with this LR bump
- the fallback line can be rescued somewhat with more steps, but it is still the backup, not the main frontier
- the LR bump helps neither survivor; the current best-tested story is “more steps yes, simple LR bump no”

**Outcome**

- best result from this tranche: [`AL-20260329-016`](./experiments.tsv) at `1.3721`
- main conclusion: both promising survivors were under-trained at `131072`, but neither wanted `MATRIX_LR=0.065`
- next pivot: return to architecture, compression-aware capacity, or output/residual mechanics, now using `9L / MLP2 / 98304 / kv2` as the operating frontier

**Decision rule for D**

- if one of the optimized slim candidates beats [`AL-20260329-004`](./experiments.tsv) while staying under the cap, it becomes the new valid frontier
- if optimization does not recover the slimmer candidates, the next tranche should pivot from width rescue toward compression-aware structural changes outside the width family

## T-20260329-E: Attention Geometry Audit

**Status:** completed

**Goal**  
Use the new frontier, [`AL-20260329-016`](./experiments.tsv), as the base model and ask whether the next gain comes from attention geometry rather than more optimizer fiddling.

**Main question**  
Is the current `9L / MLP2 / 98304 / q8-kv2 / QK_GAIN_INIT=1.5` setup the right attention shape, or is the frontier now limited by head geometry and attention sharpness?

**Why this tranche exists**

- tranche C already solved the main width-vs-size allocation question
- tranche D already solved the immediate optimization question
- the next compute-worthy pivot should target a different component family
- attention is the cleanest next family because several relevant knobs are already env-exposed and can be tested without speculative code edits

**Base controls**

- anchor shape: `9L / MLP2 / MODEL_DIM=512 / TRAIN_BATCH_TOKENS=98304`
- keep `MAX_WALLCLOCK_SECONDS=600`
- keep primary metric `final_int8_ttt_lora`
- keep tied embeddings and tokenizer/validation semantics unchanged
- keep default optimizer settings from the current winner unless the experiment explicitly changes them

**Anchor**

- [`AL-20260329-016`](./experiments.tsv) at `1.3721`, 15.48 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `E1` | `NUM_HEADS=4, NUM_KV_HEADS=2` | Test fewer, wider query heads | The current frontier may be over-fragmenting attention; wider heads could improve a small model's attention quality | Whether the model wants fewer, wider attention heads |
| `E2` | `NUM_HEADS=16, NUM_KV_HEADS=2` | Test more, narrower query heads | The current frontier may be under-headed; more heads could improve routing diversity | Whether the model wants more attention subspaces even at smaller head_dim |
| `E3` | `NUM_HEADS=8, NUM_KV_HEADS=4` | Reduce KV sharing | `kv2` may now be too aggressive on the stronger frontier; giving queries more distinct keys/values may help quality enough to justify the cost | Whether the main line is limited by over-shared KV projections |
| `E4` | `QK_GAIN_INIT=1.0` | Test flatter attention sharpness at init | The current `1.5` gain may make attention too sharp early in training on the step-rich frontier | Whether softer initial attention improves learning dynamics |
| `E5` | `QK_GAIN_INIT=2.0` | Test sharper attention at init | The current `1.5` gain may be too conservative and a stronger signal could help the model focus faster | Whether more aggressive initial attention helps the same frontier |

**Why these five are worth the compute**

- `E1` and `E2` bracket query-head geometry without changing model size dramatically
- `E3` directly tests whether the current `kv2` choice is now the bottleneck rather than the solution
- `E4` and `E5` bracket attention sharpness around the existing setting, so we learn whether the current init is too flat, too sharp, or already near the right point

**Decision rule for E**

- if `E1` or `E2` wins, the next tranche should keep the new head geometry fixed and probe neighboring attention settings
- if `E3` wins, the new frontier may have outgrown `kv2`, and future capacity planning should treat KV sharing as a first-class tradeoff again
- if `E4` or `E5` wins, attention sharpness was mis-set and we should tune around the winning side rather than touching architecture broadly
- if none win clearly, attention geometry is probably not the next bottleneck and the next tranche should pivot to output-path or residual-control simplification

**Results so far**

- [`AL-20260329-021`](./experiments.tsv) (`E1`, `q4/kv2`) landed at `1.3709` and 15.33 MB. This is a real frontier improvement and strongly suggests the current model wants fewer, wider query heads rather than the previous `q8` default.
- [`AL-20260329-022`](./experiments.tsv) (`E2`, `q16/kv2`) landed at `1.3968` and 14.40 MB. This is a clear regression and says the E1 win was directional evidence for wider heads, not a generic reward for changing head count.
- [`AL-20260329-023`](./experiments.tsv) (`E3`, `q8/kv4`) landed at `1.3766` and 15.31 MB. This is respectable and better than the old `q8/kv2` anchor, but it still does not overtake `q4/kv2`.
- [`AL-20260329-024`](./experiments.tsv) (`E4`, `QK_GAIN_INIT=1.0`) landed at `1.3777` and 15.45 MB. Competitive, but still clearly behind `q4/kv2`.
- [`AL-20260329-025`](./experiments.tsv) (`E5`, `QK_GAIN_INIT=2.0`) landed at `1.3743` and 15.48 MB. Better than the softer bracket, but still not enough to beat `q4/kv2`.

**Current reading**

- the frontier appears to be attention-geometry-sensitive after all
- the model appears to prefer fewer, wider query heads rather than more, narrower ones
- less KV sharing helps some, but not enough to beat the wider-head direction
- softer QK init is not enough to beat the wider-head direction
- sharper QK init is better than softer QK init, but still secondary to the head-geometry win

**Outcome**

- best result from this tranche: [`AL-20260329-021`](./experiments.tsv) at `1.3709`
- main conclusion: attention geometry is a real frontier lever, and the strongest gain in this tranche came from fewer, wider query heads (`q4/kv2`)
- secondary conclusion: less KV sharing and QK-gain tuning can help somewhat, but neither beat the `q4/kv2` change
- next pivot: use `9L / MLP2 / 98304 / q4-kv2` as the new anchor and test output-path or residual-control simplification next

## T-20260329-F: Output Path Audit

**Status:** planned next

**Goal**  
Use the new frontier, [`AL-20260329-021`](./experiments.tsv), as the base model and ask whether the next gain comes from output-path expressivity or calibration rather than from more attention work.

**Main question**  
Is the current `9L / MLP2 / 98304 / q4-kv2 / tie_embeddings / logit_softcap=30` setup leaving quality on the table because the output path is too constrained, too weakly regularized, or learning at the wrong rate?

**Why this tranche exists**

- tranche C solved the main width-allocation question
- tranche D solved the immediate optimization question
- tranche E solved the first-pass attention question
- the output path is the next worthwhile component family because it is both underexplored and already env-exposed in several meaningful ways

**Base controls**

- anchor shape: `9L / MLP2 / MODEL_DIM=512 / TRAIN_BATCH_TOKENS=98304 / NUM_HEADS=4 / NUM_KV_HEADS=2`
- keep `MAX_WALLCLOCK_SECONDS=600`
- keep primary metric `final_int8_ttt_lora`
- keep tokenizer/validation semantics unchanged
- keep the current best optimizer defaults except when the experiment explicitly changes the output-path learning rate

**Anchor**

- [`AL-20260329-021`](./experiments.tsv) at `1.3709`, 15.33 MB

**Planned experiments**

| ID | Shape | Goal | Hypothesis | What it teaches |
|---|---|---|---|---|
| `F1` | `TIE_EMBEDDINGS=0` | Untie embeddings and output head | The frontier may need a more expressive output head than tying allows, and the extra bytes may still fit the cap | Whether output expressivity is a real bottleneck |
| `F2` | `LOGIT_SOFTCAP=20` | Tighten logit clipping | The current softcap may be too loose, letting logits become poorly calibrated | Whether stronger output regularization helps the frontier |
| `F3` | `LOGIT_SOFTCAP=40` | Relax logit clipping | The current softcap may be too restrictive and suppressing useful confidence | Whether the output path wants less saturation |
| `F4` | `TIED_EMBED_LR=0.03` | Slow down tied output updates | The tied embedding/output matrix may be learning too aggressively for the current frontier | Whether the output path wants a gentler learning rate |
| `F5` | `TIED_EMBED_LR=0.07` | Speed up tied output updates | The tied embedding/output matrix may be under-updated relative to the rest of the model | Whether the output path wants stronger updates |

**Why these five are worth the compute**

- `F1` tests a qualitatively different hypothesis: expressivity versus byte cost
- `F2` and `F3` bracket output calibration around the current softcap so we can tell if the current setting is too strict, too loose, or already near the right point
- `F4` and `F5` bracket the learning dynamic of the tied output path around the current `0.05`, which is more informative than another ad hoc optimizer poke

**Decision rule for F**

- if `F1` wins, the next tranche should treat untied outputs as a serious frontier direction and optimize around their size budget
- if `F2` or `F3` wins, the next tranche should tune around the winning softcap side before touching architecture again
- if `F4` or `F5` wins, output-path learning dynamics are mis-set and deserve a small local optimization tranche
- if none win clearly, the output path is probably not the next bottleneck and the next pivot should move to residual-control simplification

**Results so far**

- [`AL-20260329-026`](./experiments.tsv) (`F1`, untied outputs) landed at `1.3614` and 15.78 MB. This is a major frontier jump and strongly suggests the current `q4/kv2` line was output-path-limited.
- [`AL-20260329-027`](./experiments.tsv) (`F2`, untied + `LOGIT_SOFTCAP=20`) landed at `1.3582` and 15.77 MB. This is another real improvement and says the untied frontier also wants tighter output clipping.
- [`AL-20260329-028`](./experiments.tsv) (`F3`, untied + `LOGIT_SOFTCAP=40`) landed at `1.3628` and 15.77 MB. Still strong, but clearly behind the tighter softcap.

**Current reading**

- output expressivity matters a lot more than the repo had previously explored
- untied outputs should now be treated as the new working assumption for this tranche
- tighter softcap already helps on top of untied outputs
- the softcap bracket is now directional: `20` beat `40`
- the remaining tranche-F runs should now ask whether output-head learning dynamics can improve the untied frontier further

**Adaptive follow-up plan**

- after `F1` won clearly, the remaining four runs were upgraded to focus on the untied frontier itself:
- `F2`: `TIE_EMBEDDINGS=0, LOGIT_SOFTCAP=20`
- `F3`: `TIE_EMBEDDINGS=0, LOGIT_SOFTCAP=40`
- `F4`: `TIE_EMBEDDINGS=0, HEAD_LR=0.004`
- `F5`: `TIE_EMBEDDINGS=0, HEAD_LR=0.012`
- this is a better use of compute than continuing to bracket tied-output knobs after untied outputs already showed a large win
