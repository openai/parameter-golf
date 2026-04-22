# Spec 029 screen — 4×H100 NE-1, seed 314

**Pod:** `tbk3t7a9o5tifp` @ US-NE-1, 4×H100 SXM, $11.96/hr, `runpod/parameter-golf:latest`
**Commit:** `c3a99b3` (exp/029-full-stack) — 025b frozen + LoRA warm-start-A + depth curriculum + alpha_info fix + pre-warm
**Seed:** 314
**Launched:** 2026-04-23 16:36 UTC
**Training stopped:** 17:09 UTC — `stopping_early: wallclock_cap` at step 4742, train_time 1196.1s
**TTT killed by user:** ~17:24 UTC (stalled in recompile loop, rb value already converged)
**Pod left RUNNING** for follow-up work.

## Result: SCREEN GATE MISSED ❌

Primary gate: pre-quant EMA val_bpb **< 1.068**. Observed: **1.07007** → **miss by +0.00207**.

| metric | 025b seed 42 | 026 screen 314 | **029 screen 314** | 029 target | verdict |
|---|---|---|---|---|---|
| val@4000 | 1.1079 | 1.1159 | **1.1128** | ≤ 1.108 | ❌ (+0.0048 miss, "marginal" band) |
| final-step val_bpb (step 4742) | — | 1.0682 | **1.0704** | — | |
| **pre-quant EMA val_bpb** | **1.06917** | 1.06770 ✓ | **1.07007** | **< 1.068** | ❌ |
| quantized (post-GPTQ) | — | 1.07725 | **1.07926** | — | |
| TTT aggregate rb (interrupted) | — | — | **~1.0811** | — | partial — not a final number |
| submission size (brotli) | — | 15.97 MB | 15.98 MB | < 16 MB | ✓ |

**Read:** 029 regresses vs 025b baseline (−0.0009 on pre-quant EMA) and vs 026 screen (−0.00237). Depth curriculum + NUM_LOOPS=3 with 024b-converged alpha values does not help at this scale.

## Training trajectory markers (all fired correctly)

```
step   108                (pre-warm: loop_warmup:depth_upgraded looping_depth:4)   ← startup validation
step 2243   frac 0.350    layer_loop:enabled depth:3                                ← loop activation
step 3614   frac 0.670    loop_depth:upgraded depth:4                               ← depth upgrade
step 4742                 stopping_early (wallclock cap)
```

Recompile count stabilized at 56 after pre-warm — **zero mid-run recompiles** during training (the c3a99b3 pre-warm fix worked as designed; 68 total after TTT triggered more).

## Matched-step train_loss Δ vs 025b seed 42

```
step  | 025b_42 | 029_314 | Δ        | phase
------+---------+---------+----------+-------
 500  | 2.6759  | 2.6636  | -0.0123  | both depth=1 (seed noise)
1000  | 2.7667  | 2.7724  | +0.0057  | both depth=1
1500  | 2.5932  | 2.5979  | +0.0047  | both depth=1
2000  | 2.6538  | 2.6575  | +0.0037  | both depth=1
2500  | 2.5000  | 2.5105  | +0.0105  | 025b depth=2 (from step 2122), 029 depth=1 (until 2243)
3000  | 2.5687  | 2.5820  | +0.0133  | 025b d2, 029 d3
3500  | 2.3875  | 2.4021  | +0.0146  | both in loop (029 +1 depth)
3600  | 2.4488  | 2.4685  | +0.0197  | 029 about to upgrade to d4
3700  | 2.5159  | 2.5353  | +0.0194  | 029 just upgraded to d4
3800  | 2.4982  | 2.5193  | +0.0211
3900  | 2.5125  | 2.5290  | +0.0165  ← narrowing after d4 kick
4000  | 2.3758  | 2.3866  | +0.0108
4500  | 2.3442  | 2.3491  | +0.0049
4600  | 2.3690  | 2.3695  | +0.0005
4700  | 2.3345  | 2.3338  | -0.0007  ← briefly beats 025b
```

Δ compressed late (depth=4 adaptation working), but the earlier +0.015-0.02 window dominated the EMA average, giving the 1.07007 endpoint. Pre-step-2100 Δ (both at depth=1) is pure seed/pod noise.

## Throughput (cumulative tok/s)

```
step  | 029_314  | 025b_42  | Δ%      | phase
------+----------+----------+---------+-------
 500  | 4.22M    | 3.98M    | +5.8%   | both d=1
2100  | 4.21M    | 3.99M    | +5.7%   | 029 just past d=1→d=3
3600  | 3.55M    | 3.21M    | +10.6%  | 029 d=3, 025b d=2
4000  | 3.33M    | 3.22M    | +3.4%   | 029 d=4 (from 3614), 025b d=2
```

NE-1 pod ran ~6% faster hardware than 025b's pod — part of the apparent throughput advantage is pod variance, not an architecture win.

## Why 029 regressed (hypotheses)

1. **Alpha mistuning dominates.** Alpha/beta buffers were converged by 024b under NUM_LOOPS=2. Spec 029 uses NUM_LOOPS=3 (extra pass) + depth curriculum (1→3→4), using the same shared alpha across all passes. Pass 2 (in NUM_LOOPS=3) and the depth=4 regime never existed during 024b's training. Convergence cost likely +0.002-0.005 bpb.

2. **Depth=4 kicks in too late + warmdown.** `LOOP_DEPTH_UPGRADE_AT=0.67` means depth=4 only runs for 33% of the budget (step 3614→4742), during LR warmdown. EMA smoothing averages ~67% depth=3 vs 33% depth=4, so the eval is effectively more depth=3 than depth=4.

3. **Seed 314 isn't magic here.** 026 screen also used seed 314 and passed — so the seed alone isn't to blame. But in 026's config (025c per-pass + LoRA, no depth curriculum), seed 314's post-EMA was 1.06770. Here with 029's stack it's 1.07007.

## TTT partial observation

The screen launched with `PHASED_TTT_ENABLED=0` expecting "no TTT". But that flag only disables the *phased* variant — it collapses TTT to `num_phases:1`, which is a single continuous adaptation over all 48k suffix docs. That turned out to be the **slowest** TTT mode:

- Per-batch ~12s on 4×H (vs 8×H num_phases=3 is much faster)
- Repeated Dynamo recompiles on TTT's rotary `_seq_len_cached` guard (new sequence-length buckets force recompile)
- After 34/782 batches (~6 min of real progress + stall), rb converged at **1.0811** — used as best-available TTT estimate
- Projected full completion: 2+ hours, with progress still stalling in recompile

User directed kill of the TTT process; pod kept alive for handoff.

**Lesson:** always use `PHASED_TTT_ENABLED=3 PHASED_TTT_NUM_PHASES=3` — =0 is not "disabled" but "slow single-phase." Now in memory (`feedback_phased_ttt_enabled.md`).

## Incidents / lessons

1. **`/runpod` symlink layout on NE-1** — data is at `/workspace/parameter-golf/data/...`, so per-subdir symlinks (`/runpod/{data, parameter-golf, runs}`) needed. Single-level `/runpod → /workspace` does not work on NE-1. Set up once at preflight.

2. **Pre-warm slow but works.** The c3a99b3 pre-warm compiles 3 depth states × 4 cu_seqlens buckets × 2 ranks = many graphs. Took ~10-12 min vs baseline ~5 min. Once past warmup, 0 mid-run recompiles during training (good).

3. **TTT recompile cascade.** Running TTT at num_phases=1 with guard-sensitive rotary cache triggers per-doc-length recompiles. This is why our screen's TTT eval stalled. Architectural, not a config bug.

4. **Late-training Δ compression** is real: Δ went from +0.021 to near zero by step 4700. If LOOP_DEPTH_UPGRADE_AT were earlier (0.5 or 0.33), depth=4 might have had time to drive Δ negative. Candidate follow-up spec.

## Cost

- ~$11.96/hr × 50 min = **~$10** for screen training + GPTQ + sliding + partial TTT
- Plus ~$6.17 from prior spec 026 screen (same pod class)

## Decision / next steps

Per spec 029's accept table (> 1.068 pre-quant EMA → "Regression; Kill; debug curriculum interaction"):

- **Do not proceed to 8×H full pipeline for 029** as currently speced.
- **Best candidate rework:** un-freeze alpha OR re-converge alpha for NUM_LOOPS=3 + depth=4 regime. Research decision.
- **Alternative:** earlier LOOP_DEPTH_UPGRADE_AT (0.33 or 0.5) to give depth=4 more training time.
- **Fallback:** return to spec 026's 8×H full pipeline — its screen passed (pre-quant EMA 1.06770 < 1.068). Blocked on 8×H stock.

## Artifacts

- `train.log` — full stdout (143 KB)
- `b196d323-…txt` — rank 0 log
- `launch.out` — small (cu_seqlens banner)
- `notes.md` — this report
- `final.json` — structured results
