# Stage 2_1 Postmortem

## Conclusion

The `stage2_1` pack was not wrong because it was noisy. It was wrong because it mostly searched along the wrong causal scale. The run found several mechanisms that helped early training dynamics, throughput, or compression geometry, but none of them changed the underlying 600s finalist outcome enough to beat the current SOTA-aligned default control. The clean control was not a fluke: the two control repeats in the finalist screen finished at `1.49460870` and `1.49629451`, which is the practical noise floor for this stage, and every candidate except `T4` landed worse.

## What Was Tested

The search covered four distinct mechanism classes:

- Training-side refinements: `MuonWD`, `LeakyReLU^2`, `EMA`, `XSA4`, `Partial RoPE + LN Scale`.
- Context/data-ordering changes: `XSA-all`, shard-order curriculum.
- Optimization geometry changes: `weight_perturbation`, `gradient_centralization`.
- Systems throughput changes: `FA3` on top of candidate stacks.

This was a good surface map, but most of the mechanisms lived in the same narrow part of the search space: they altered how the current 11L stack trained, not what class of solution it could reach.

## What Failed And Why

The failures split into four buckets.

`Weak ideas`

- `train_arch` did not survive the 600s horizon. The best screen result there was `B7 frontier_base_plus_partial_rope_lnscale` at `2.8245823`, but the finalist screen put the corresponding winner at `1.56392781`, worse than control by `0.06931911`.
- `context_all` also did not survive. `D4 xsa_all + ln_scale` looked fine in the short screen, but the finalist screen ended at `1.56414210`, again worse than control by `0.06953340`.
- The lesson is that these were small local reshufflings of the same training trajectory, not a new solution class.

`Phase-misaligned`

- `throughput` and `FA3` were real systems wins, but they were not score wins. In the short screen, `U7 throughput_geometry_plus_fa3` was the best candidate at `2.41711783` and ran at `719.61 ms/step` versus about `848 ms/step` for the controls. In the 600s finalist screen, the same family still ran faster and took more steps (`835` vs `708` for control), but it lost on score: `1.51766269` versus `1.49460870`.
- That is the signature of a throughput multiplier, not a new optimization mechanism. More steps helped, but not enough to overcome the control's better final basin.

`Compressibility-tilted`

- `weight_perturbation` looked strong in the geometry screen because `F2` reached `2.50648798`, but the effect did not scale. The finalist geometry slot ended at `1.57552369`, worse than control by `0.08091499`.
- The artifact got larger too: `T5` produced a `16.4 MB` int8+zlib payload versus `13.0 MB` for control. That is a red flag for the compressed submission path, even when the early screen looks healthy.
- The mechanism likely found a flatter or more noise-tolerant basin, but the gain was too small to offset the drop in raw fit at 600s.

`Noisy-short-horizon false positives`

- `curriculum` is the clearest example. It was nearly tied in the finalist screen, but still lost: `T4` finished at `1.49888131` versus control `1.49460870`. The raw pre-quant loss was only slightly better (`1.4241` vs `1.4247`), which is below the practical value of a stage like this. The post-quant gap then moved the wrong way.
- The short screen recommended the curriculum slot, but the 600s horizon showed that the gain was too small to matter.

## Why The Control Won

The control won because the current SOTA-aligned default was the only thing that was consistently good across the whole pipeline: stable optimization, reasonable compressibility, and no extra mechanism overhead. It did not need to recover from a bad perturbation or pay for extra noise.

The finalist screen makes that clear:

- `T0` control A: `1.49460870`
- `T1` control B: `1.49629451`
- `T4` curriculum: `1.49888131`
- `T6` throughput + FA3: `1.51766269`
- `T2` train_arch winner: `1.56392781`
- `T3` context_all winner: `1.56414210`
- `T5` geometry winner: `1.57552369`
- `T7` mix: `1.57585658`

The control repeat gap is only `0.00168581`, so anything smaller than a few thousandths is basically noise. `T4` was the only near-tie, and it still lost. Every other candidate was meaningfully worse.

## What The Results Say About The Real Bottleneck

The bottleneck is not “we need a faster run” and it is not “we need one more local training trick.” `FA3` already proved that extra throughput can buy more steps, and `weight_perturbation` already proved that you can move the compressed artifact in a favorable direction, but neither changed the final score enough.

So the current bottleneck is the interaction of base model, training trajectory, and compressed final artifact. The search space we explored mostly changed the path to the same model family. It did not change the model family enough.

## Next-Hypothesis Bar

Future hypotheses need to satisfy all of this before they are worth another stage:

- They must show a real 600s advantage, not just a 90s or 180s screen win.
- They must improve the final compressed score, not only pre-quant loss or step rate.
- They must survive rebasing onto the control and still work when the same 8-GPU budget is spent on the long finalist horizon.
- They must be tied to a distinct causal story, not another retune of the current training stack.

If a proposal cannot clear that bar, it is not a next-stage hypothesis. It is a local variant of something we already learned is too small.
