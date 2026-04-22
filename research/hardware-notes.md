# Hardware notes — empirical observations from runs

## JP pod throughput variance

**Pod-to-pod variance on AP-JP-1 8×H100 SXM: ±1–2% throughput at post-loop phase.**

Measured across 6 runs (008, 015, 016, 017, 019, 019b), all AP-JP-1, all same model size.
Pre-loop (steps 500–2000): all pods land within 1% of each other (~8.0–8.1M tok/s). Post-loop
(steps 4000–4500): spread widens to ±2%, driven by pod thermal state / node contention.

Implication: a throughput difference of <2% between two runs on different pods is pod noise,
not a code signal. Step-count differences of <~100 steps (out of ~4700–4800) are similarly
within noise.

## Loop45 activation throughput drop

Loop45 (loop_start=4, loop_end=5) activates at ~35% of training (~step 2100–2200). This
causes an immediate ~6–7% throughput drop followed by a continued gradual decline:

| Phase | tok/s range | notes |
|---|---|---|
| Pre-loop (steps 0–2100) | 8.0–8.1M | stable, pod-independent |
| Just after activation (~step 2500) | 7.4–7.6M | −7 to −8% vs pre-loop |
| Mid-training (~step 3000) | 6.9–7.1M | −12 to −14% |
| Late (~step 4000) | 6.4–6.6M | −18 to −21% |
| Final (~step 4500+) | 6.2–6.4M | −20 to −23% |

The gradual post-activation decline (not a clean step function) is likely GPU thermal throttling
from sustained higher compute load per step once the loop block runs multiple times.

## α blend overhead

Measured post-loop drift from step 2000 → step 4500 across α variants:

| α type | drift | example runs |
|---|---|---|
| No α (plain residual) | −19.5% | 008 |
| Tensor α (learnable, starts at 1.0) | −20.8 to −21.9% | 015, 016, 017 |
| Constant α (hardcoded at endpoint values from step 1) | −22.5 to −22.9% | 019, 019b |

Constant α drifts ~3pp more than no-α. Tensor α sits in between but overlaps pod variance.
Root cause of constant-α extra drift: endpoint α values (1.078–1.430) are non-identity from
step 1 of loop activation, forcing real HBM reads of x_before every looped layer every pass.
Tensor α starts at 1.0 (identity = zero overhead) and only drifts away in the final phase.

The lerp → algebraic rewrite (019 → 019b) had zero measurable throughput effect.

## tok/s metric is cumulative, not instantaneous

`train_gpt.py` logs `tok_per_sec = step * batch_tokens / elapsed_time` — a **running average
from step 1**, not a per-step rate. The "drift" in logged numbers is the weighted average
converging toward the true post-loop rate as fast pre-loop steps get diluted. All logged
tok/s numbers before the final steps are higher than actual step speed.

To get true instantaneous throughput between two log points t1→t2:

```
instant_tps = (t2 - t1) / (t2/tps2 - t1/tps1)
```

Instantaneous throughput at steps 4000→4500 (true per-step speed at end of training):

| run | cumulative at step 4500 | instantaneous 4000→4500 |
|---|---|---|
| 008 (no α) | 6.446M | **5.49M** |
| 015 (tensor α) | 6.334M | **5.49M** |
| 016 (tensor α) | 6.257M | **5.35M** |
| 017 (tensor α) | 6.387M | **5.41M** |
| 019 (constant α) | 6.230M | **5.15M** |
| 019b (constant α) | 6.263M | **5.20M** |

True constant-α overhead at end of training: **~5–6% instantaneous** vs no-α baseline.
Pod-to-pod variance (016 vs 015 despite same code) is still ±2–3% even on instantaneous.

## Reference runs

| run | pod | final step | pre-loop tok/s | final tok/s |
|---|---|---|---|---|
| #1736 (dexhunter, original) | unknown | 4854 | — | — (no snapshots) |
| 008 | xy1bfwkcfds0ax | 4828 | 8.009M | 6.446M |
| 015 | k9wwhapqaufb0u | 4761 | 8.077M | 6.293M |
| 016 | w20na0rp2710e8 | 4708 | 8.006M | 6.214M |
| 017 | 9crwq3fldt5tfj | 4784 | 8.069M | 6.300M |
| 019 | jzsfonth5x0fe1 | 4697 | 8.078M | 6.230M |
| 019b | 6pyy9q7aatvgpb | 4716 | 8.083M | 6.226M |
