# Idea — Throughput analysis of 008 / 015 / 016 / 017 / 019 / 019b at full scale

**Created:** 2026-04-21
**Status:** open — starting point for throughput analysis today

## Runs under analysis

All runs are 8×H100 full-pipeline (#1736 base) at 11L/512d, seed 42, same CaseOps / GatedAttn / QuantGate / PhasedTTT config. Differences are only in the recur-α treatment:

| run | config | α storage | blend op | notes |
|---|---|---|---|---|
| **008** | baseline, no α | — | — | #1736 reproduction, no recur-alpha |
| **015** | recur-alpha α=0 init | `nn.Parameter`, learned | `α*x_new + (1-α)*x_before` (manual) | first α run |
| **016** | recur-alpha α=1 init | `nn.Parameter`, learned | manual | init-sensitivity test |
| **017** | recur-alpha full pipeline | `nn.Parameter`, learned | manual | 015 but also TTT wiring added |
| **019** | constant α, lerp | Python literal (017 endpoint values) | `torch.lerp(x_before, x_new, α)` | 018c's "92% recovery" recipe at full |
| **019b** | constant α, manual-algebraic | Python literal | `x_before + α*(x_new - x_before)` | manual form to avoid lerp primitive |

Loop activation (looped layers 3-4-5 run 2× extra = 3× total) is at step ~2142. α blend sites fire only after activation. `ENABLE_LOOPING_AT=2142` is the default for all runs above.

## Computing interval tok/s

The `tok/s` value logged in `train.log` is cumulative:
```
tok_per_sec = step * train_batch_tokens / (approx_training_time_ms / 1e3)
```
This is a dragging average over all training time so far, not the instantaneous rate. For in-window rate between consecutive log entries at steps s₁ and s₂ with reported values r₁ and r₂:
```
interval_rate = (s₂ - s₁) / (s₂/r₂ - s₁/r₁)
```
(`train_batch_tokens` cancels.)

015/016/017/019/019b all log every 100 steps. 008 logs every 500 steps.

## Per-100-step interval tok/s (M)

008 is shown only at 500-step marks (sparse log). 015/016/017/019/019b are dense.

| end | 008 | 015 | 016 | 017 | 019 | 019b |
|---|---|---|---|---|---|---|
| 200 | · | 8.03 | 7.91 | 8.01 | 7.96 | 8.07 |
| 300 | · | 8.01 | 7.95 | 8.03 | 8.02 | 8.02 |
| 400 | · | 8.03 | 7.93 | 7.99 | 8.01 | 8.03 |
| **500** | **8.05** | **8.30**★ | **8.23**★ | **8.30**★ | **8.28**★ | **8.24**★ |
| 600 | · | 8.01 | 7.90 | 7.98 | 8.02 | 7.97 |
| 700 | · | 8.02 | 7.93 | 8.00 | 7.99 | 8.02 |
| 800 | · | 8.01 | 7.93 | 7.98 | 7.95 | 8.02 |
| 900 | · | 7.91 | 7.95 | 8.04 | 8.01 | 8.04 |
| **1000** | **8.01** | **8.30**★ | **8.24**★ | **8.27**★ | **8.28**★ | **8.30**★ |
| 1100 | · | 8.00 | 7.93 | 7.97 | 7.99 | 8.00 |
| 1200 | · | 8.03 | 7.94 | 7.98 | 8.04 | 8.00 |
| 1300 | · | 8.00 | 7.92 | 7.99 | 8.03 | 8.02 |
| 1400 | · | 7.98 | 7.94 | 7.99 | 8.00 | 8.00 |
| **1500** | **7.97** | **8.29**★ | **8.25**★ | **8.29**★ | **8.31**★ | **8.29**★ |
| 1600 | · | 7.99 | 7.92 | 8.01 | 8.01 | 7.99 |
| 1700 | · | 7.97 | 7.94 | 7.99 | 8.02 | 8.01 |
| 1800 | · | 8.01 | 7.93 | 7.99 | 8.03 | 7.94 |
| 1900 | · | 8.31★ | 8.26★ | 8.30★ | 8.31★ | 8.29★ |
| **2000** | **8.00** | 8.01 | 7.93 | 7.99 | 8.02 | 8.02 |
| 2100 | · | 7.94 | 7.93 | 7.99 | 8.02 | 8.01 |
| 2200 | · | **6.31** | **5.77** | **6.12** | **6.34** | **6.34** |
| 2300 | · | 5.47 | 5.35 | 5.33 | **4.21** | 5.49 |
| 2400 | · | 5.58 | 5.48 | 5.46 | 5.60 | 5.60 |
| **2500** | **5.95** | 5.46 | 5.35 | 5.33 | 5.46 | 5.46 |
| 2600 | · | 5.48 | 5.35 | 5.33 | **4.80** | 5.50 |
| 2700 | · | 5.48 | 5.36 | 5.33 | **4.70** | 5.48 |
| 2800 | · | 5.47 | 5.37 | 5.33 | 5.48 | 5.47 |
| 2900 | · | 5.59 | 5.49 | 5.44 | 5.60 | **3.76** |
| **3000** | **5.48** | **4.52** | 5.36 | 5.35 | 5.47 | **4.76** |
| 3100 | · | 5.47 | 5.37 | 5.36 | **4.78** | **4.80** |
| 3200 | · | 5.47 | 5.36 | 5.36 | 5.49 | 5.49 |
| 3300 | · | 5.58 | 5.49 | 5.47 | 5.60 | 5.61 |
| 3400 | · | 5.41 | 5.31 | 5.36 | **4.83** | 5.48 |
| **3500** | **5.49** | **4.48** | **4.40** | 5.35 | **4.60** | **4.81** |
| 3600 | · | 5.48 | **4.44** | 5.33 | 5.48 | **4.77** |
| 3700 | · | 5.43 | **4.41** | 5.34 | 5.48 | 5.49 |
| 3800 | · | 5.61 | 5.50 | 5.44 | 5.60 | 5.61 |
| 3900 | · | **4.54** | 5.37 | 5.33 | 5.50 | 5.48 |
| **4000** | **5.49** | **4.51** | 5.38 | 5.37 | **4.79** | 5.50 |
| 4100 | · | 5.46 | 5.32 | 5.40 | **4.66** | **4.71** |
| 4200 | · | 5.48 | 5.36 | 5.41 | **4.69** | **4.74** |
| 4300 | · | 5.60 | 5.45 | 5.51 | 5.61 | 5.61 |
| 4400 | · | 5.45 | 5.35 | 5.38 | 5.49 | 5.48 |
| **4500** | **5.49** | 5.47 | 5.33 | 5.35 | 5.49 | 5.49 |
| 4600 | · | 5.47 | 5.36 | 5.30 | 5.45 | 5.49 |
| 4700 | · | 5.49 | 5.39 | 5.29 | — | 5.51 |

★ = 500-step-boundary logging artifact (see Observation 1).

## Starting-point observations

1. **500-step-boundary logging inflates rates by ~4%.** Intervals ending at 500/1000/1500/2000 show ~8.3M (vs neighbors at ~8.00). Post-activation, some 500-boundary rows and every-5th-interval peaks (2900, 3300, 3800, 4300) show elevated ~5.6M vs neighbors ~5.35. Likely a `train_time_ms` counter quirk when validation fires. Treat ★ rows as artifacts.

2. **Pre-activation (steps 0-2100): flat ~8.0M across all runs.** No drift. The "drift from 8.1M → 6.4M" seen in cumulative tok/s was an averaging artifact — the actual in-window rate is constant pre-activation.

3. **Loop activation is a step function at step ~2200-2300, not a gradual drift.** Costs ~33% of throughput: 8.0 → 5.35 on 008. Ramp is two intervals (2200→2300) because loop activates at step 2142 so the 2100-2200 interval is partially looped.

4. **Post-activation (steps 2500+): flat plateau ~5.3-5.5M across all runs.** No continuing decay. The model runs at steady-state throughput once the loop is on.

5. **017 is unusually steady post-activation.** Holds 5.29-5.47 almost every interval. Other runs have sporadic low-rate intervals (4.4-4.8) that drag their averages down.

6. **019 has idiosyncratic dip intervals.** Steps 2300, 2600-2700, 3100, 3400, 4000-4200 all show 4.2-4.8M. These dips, not a steady tax, are what make 019's average look bad.

7. **α tax on a clean run is ~1.5%.** 008 steady at 5.48-5.49; 017 steady at 5.30-5.40 → ~1.5% gap. Larger per-bucket gaps in 015/016/019/019b may be mostly node contention / scheduling noise rather than α-specific overhead.

## Dip-interval analysis (the real throughput story)

Post-activation, most runs' steady state is ~5.48M tok/s (matching 008 exactly). The α "tax" we see in averages is almost entirely driven by **sporadic dip intervals** where tok/s collapses to 3.7-4.8M for 1-3 consecutive intervals. Listing intervals below 5.0M (post step 2400):

| run | dip intervals | count |
|---|---|---|
| 017 | *(none)* | **0** |
| 016 | 3500, 3600, 3700 | 3 (one contiguous cluster) |
| 015 | 3000, 3500, 3900, 4000 | 4 |
| 019b | **2900**, 3000, 3100, 3500, 3600, 4100, 4200 | 7 |
| 019 | 2300, 2600, 2700, 3100, 3400, 3500, 4000, 4100, 4200 | 9 |

**Three-tier pattern:**
- **017 is uniquely clean** — 0 dips post-activation, steady at 5.29-5.51 every interval.
- **016 has one contiguous dip cluster** (3500-3700) — looks like a single event, not a cycle.
- **015 is scattered spiky** — 4 non-sequential dips, no obvious rhythm.
- **019 and 019b partially share a dip cycle** — both dip at 3100, 3500, 4100, 4200. Not identical step-for-step but same rough cadence (~every 500-700 steps).

**Key observation:** constant-α runs (019, 019b) have more dips than any tensor-α run. Tensor-α max is 4 dips; constant-α min is 7. So constant-α at full scale doesn't just add steady overhead — it appears to make the run *more dip-prone*. The constant-folded kernel schedule may interact badly with whatever causes the dips.

**Same-config variance warning.** 015 and 017 have nominally identical configs (tensor α, manual blend, seed 42) but 015 has 4 dips while 017 has 0. That's same-config JP-pod variance — meaningful enough to add 4 dip intervals to a single run's throughput profile. Any single-run throughput number must be read through this noise.

**Working hypotheses for what the dips are** (not yet distinguished):
1. **Periodic internal events** — scheduled computation every ~500-700 steps (LR schedule transition, optimizer state compaction, Muon backend re-spinup, something in CaseOps/QuantGate). Would be reproducible at the same step indices on a rerun.
2. **Pod-level contention** — noisy neighbor or shared-NIC NCCL hiccups; constant-α runs may be more vulnerable due to less latency-tolerant launch patterns. Would be random, not reproducible.

**Cheapest diagnostic:** rerun 019 (or 019b) with the same seed/commit on a different JP pod. If the dip steps are reproducible → deterministic internal event. If random → pod contention.

## Overhead decomposition (validated 2026-04-22 via spec 020/020b)

Per-step data from 4×H100 diagnostic runs:

| source | regime | median step_time_ms | post/pre ratio | interpretation |
|---|---|---|---|---|
| 008 (8×H100, no α) | pre-loop | — | — | baseline |
| 008 | post-loop | — | **+45.7%** | pure loop FLOPs |
| 020 (4×H100, literal α) | pre-loop | 190.6 | — | baseline on this pod |
| 020 | post-loop | 277.6 | **+45.6%** | loop FLOPs + literal α |
| 020b (4×H100, buffer α) | pre-loop | 197.9 | — | baseline on this pod |
| 020b | post-loop | 292.8 | **+47.9%** | loop FLOPs + buffer α |

**Decomposition (using 008's +45.7% as pure-loop reference):**
- Literal-α blend cost: ~0 pp steady-state (+45.6% ≈ +45.7%) — but incurs 12 compile stalls (3-13s each) per 2036 steps in 020
- Buffer-α blend cost: +2.3 pp (+47.9% − +45.7%) — zero stalls, uniform cost

**Of the ~48% post-activation slowdown on a buffer-α run:**
- ~95% is loop FLOPs (3 layers × 2 extra passes = 6 layer-equivalents on 11)
- ~5% is the α blend op itself (per-token mul/add at 6 sites)

**Conclusion:** the α "tax" we've been worrying about is genuinely tiny (~5% of the slowdown, ~2 pp of step time). The dominant cost is the loop itself. If we ever want to reclaim throughput, the loop is the 20× bigger lever.

## Log-correlation pass (done; partial explanation)

Scanning 019b's train.log for events near each dip step reveals:

- **Single mid-training validation at step 4000** (per `val_loss_every: 4000`). This correlates with the 4100-4200 dip cluster: 100 steps 4000→4100 took 12s, 4100→4200 took 18s (50% slower). Validation disrupts the first couple training intervals after it.
- **But val-disruption is constant-α-specific.** 015 also ran val at step 4000 and did *not* dip at 4100-4200 (intervals were 5.46 / 5.48 — clean). Only the constant-α runs (019, 019b) dip after val. Likely mechanism: the train→eval→train mode switch triggers CUDA graph re-capture or allocator state reset, and constant-folded kernels rebuild worse than tensor-α equivalents.
- **Step 3500 dips in 4 of 5 α runs** (015, 016, 019, 019b — not 017). No logged event corresponds. Candidates: an unlogged internal scheduler transition, or a thermal-envelope crossover point after which contention becomes more likely. 017's same-config-as-015 immunity suggests node matters.
- **Other dip clusters (2900-3100 in 019b, 2300/2600/2700 in 019) have no log-correlated event.** These look like noise on top of the constant-α baseline vulnerability.
- **No checkpoint writes, LR-schedule transitions, EMA events, or other scheduled events** show up in the log at dip steps.

Remaining mysteries: what causes step 3500 to dip in 4 runs; what causes the 2300/2600/2700 and 2900-3100 clusters in the constant-α runs specifically.

## Open questions for today

- What causes the idiosyncratic dip intervals (019 at 2300/2600/2700, 015 at 3000/3500/3900/4000, 016 at 3500-3700)? Are they at repeatable step indices, or random?
- Is the ★ 500-step-boundary inflation really just a validation-logging artifact, or is it masking something real?
- Is the ~1.5% baseline α tax (017 vs 008 on steady intervals) real, or is 008 itself on a faster node?
- Loop-activation costs 33% of throughput. Is that the bigger lever to pursue?

## Data source scripts

One-liner to reproduce the per-interval table — regenerates from train.logs and 019b's final.json (see session transcript for the exact Python).
