# 2026-04-20 pt.3 — SpinQuant exhausted: spec 010b results and what we learned

**Session kind:** research, post-execution-runs for spec 010b. Continuation of pt.2 (SpinQuant results + regime finding) and pt.1 (baseline migration design). **Days to deadline:** 10.

## TL;DR

- Spec 010b ran `attn_only` and `mlp_only` on 8×H100, plus confirmed `port_1695` and baseline numbers. **All five SpinQuant variants land within 0.00009 bpb of each other at final val_bpb.** Full null across the family.
- My working hypothesis from pt.2 ("attention rotation carries long-doc help, MLP rotation carries short-doc hurt") is **refuted**. Neither subset produces port_1695's early-batch trajectory lead; the two rotations together create a nonlinear, emergent effect that neither delivers alone.
- The "port_1695 dropping to rb=1.0524" mid-eval was rank-0's running average over 98 batches of its local doc shard, **not** a preview of final val_bpb. There's an 80× compression between rank-0 rb spread and final val_bpb spread across variants.
- **SpinQuant is fully exhausted on this stack.** Spending more on rotation variants is not going to move the metric.
- Strategic pivot: spec 011 (tapered WD retrain). Anything upstream of phased TTT is what's left.

## What ran today (morning → evening summary)

| run | mode | final val_bpb | Δ vs baseline |
|---|---|---|---|
| spec 008 (morning) | #1736 reproduction | — (projected) | ref |
| spec 009 | baseline | 1.067283 | ref |
| spec 009 | internal_only (R_a) | 1.067309 | +0.000026 |
| spec 010 | port_1695 (all 4 sites, online) | 1.067232 | −0.000050 |
| spec 010b | attn_only | 1.067225 | **−0.000059** (best) |
| spec 010b | mlp_only | 1.067288 | +0.000005 |

**Spread across 5 variants: 0.00009 bpb.** #1736's reported 3-seed std is 0.00070 bpb. We are ~10× below that. All variants are statistically indistinguishable from baseline. Total spend on the SpinQuant arc today: ~$36.

## The session's mistake I want to call out

In pt.2, I wrote an idea file (`research/ideas/rotation-regime-dependence.md` and `port-1695-long-vs-short.md`) based on the per-batch trajectory data from spec 010. That data showed rotation helps long docs (−0.007 bpb), hurts short docs (+0.015 bpb). I interpreted this as a *real, exploitable* regime-dependence and designed spec 010b to isolate which rotation sites carried which half.

The mistake: I was reading rank-0's running-average bpb (`rb` column in `ttp:` log lines) and treating it as a preview of the final val_bpb. It is not. Rank 0 sees only 1/8 of the eval docs; the log's `rb:1.0524` was a real number over rank 0's local doc mix but had no predictive power for the aggregated metric across all 8 ranks.

When spec 010b's data came in, I compounded the mistake by reading an `rb:1.0657` from mlp_only at batch 780 and telling the user "mlp_only might actually net positive!" The user then asked me to check `final.json` — and the actual aggregate came out at +0.000005, above baseline.

**Lessons from the false-signal episode:**

1. **Always check `final.json` before interpreting a trend.** Rank-0 rb is a useful progress indicator, not a metric.
2. **80× compression from rank-0 rb spread → final val_bpb spread is not small noise.** It's an artifact of (a) token-weighted aggregation across 8 ranks vs batch-weighted rb on rank 0, and (b) phased TTT's LoRA adapting uniformly and absorbing variant-specific pre-TTT differences.
3. **"Regime-dependent" is real at rank-0 level but unexploitable at the aggregated metric level** on this eval distribution, because the distribution across ranks smooths the regime effect.

I'm leaving the two idea files in place but will update them to reflect what we actually learned — specifically, that the regime-dependence is a real property of the forward pass but does NOT translate to a leaderboard lever.

## What's genuinely real from the investigation

1. **SpinQuant rotation changes the quantized forward pass.** Rank-0 `rb` values span 0.0075 bpb across variants. The rotation is doing something mechanistically.
2. **TTT LoRA absorbs that difference at the global-aggregation level.** `diagnostic_quantized` (pre-TTT) variants span 0.00012 bpb; `quantized_ttt_phased` (post-TTT) variants span 0.00009 bpb. TTT is the dominant effect (~−0.013 bpb) and it's basically variant-independent.
3. **Attention rotation alone has near-zero effect** on rank-0's trajectory. The `attn_only` run's `rb` column was byte-identical to baseline at 4 decimals for the first ~200 batches. Physical explanation: `softmax(QK.T) V` is rotation-equivariant in V's head_dim, and the attention weights in #1736's stack don't have outlier structure for rotation to smooth.
4. **MLP rotation alone has the inverse rank-0 regime from port_1695.** `mlp_only` starts with a large rank-0 rb lead (1.0900 at batch 5 vs baseline's 1.1142), hurts in the middle (1.0876 at batch 25 vs baseline's 1.0771), then converges.
5. **port_1695's rank-0 rb lead is emergent** — neither `attn_only` nor `mlp_only` comes close (port_1695 at batch 5 is rb=1.0595 vs attn_only's 1.1142 and mlp_only's 1.0900).

The first four points are genuinely interesting about how rotations interact with quantized #1736. The fifth rules out a simple additive decomposition across rotation sites.

## The "rank" conversation, summarized

User asked: "what is rank?" Thread (captured here because it's a useful methodology note):

- `torchrun --nproc_per_node=8` launches 8 GPU processes; each is a "rank" (0–7).
- Each rank processes ~98 of the 782 eval batches (disjoint subsets).
- Only rank 0 writes to the log file; ranks 1–7 compute silently.
- The final `val_bpb` in `final.json` comes from `all_reduce(loss, tokens, bytes)` across all 8 ranks, so it's aggregated over the full eval.
- **The `rb:` column in `ttp:` log lines = rank 0 only.** Different rank-0 doc shards get different `rb` trajectories on different runs/seeds.

We discussed adding per-rank logging (~30 LOC patch + rerun) to validate the regime-dependence across rank boundaries. Decided: not worth the time for the leaderboard push; worth doing later if we want to write this up properly as a research artifact.

## Decisions for research

1. **SpinQuant fully exhausted.** No further rotation variants worth testing for leaderboard purposes. Seed sweeps, layer-selective, `full` (static R₀ + fold), `attn_in_only` — all projected to land within 0.0001 bpb of baseline based on the 010b data.
2. **Update idea files.** The regime-dependence observation is real *mechanistically* but **does not translate to leaderboard lever** on this eval distribution with this TTT stack. The idea files need a correction section.
3. **Move to spec 011** (tapered Muon WD retrain). Full training run, modifies the trained weights themselves, upstream of TTT, not something TTT can absorb. Code patch still unwritten (~30–50 LOC).
4. **Add a methodology note to EXECUTION.md:** *"Rank-0 `rb` column in `ttp:` log lines reflects 1/8 of the eval. Only `final.json` val_bpb is the submission number. Do not extrapolate trends from rb."*
5. **Keep the per-rank logging idea as a low-priority research followup** — useful if we ever want to publish the SpinQuant negative result with proper statistics. Not on the 04-30 critical path.

## Cost and budget

| Spec / arc | Cost |
|---|---|
| Spec 008 (morning, execution) | ~$16 |
| SpinQuant investigation (009 + 010 + 010b) | ~$36 |
| **Today's total** | **~$52** |
| **Project spend to date** | **~$52** of $200 budget |

$148 remaining, 10 days left. Plenty of runway for spec 011 (~$20), a plausible spec 012 (whatever lever 011 points at), and a final 3-seed confirmation on the winning stack (~$30–40).

## Open questions for the next session

1. Spec 011's WD-taper patch: what's the cleanest insertion point in `train_gpt.py`'s training loop? Goes into the Muon optimizer's per-step update logic.
2. Does tapered WD on #1736 reproduce #1729's claimed small positive? If yes, we have our first leaderboard lever of the push. If no, SwiGLU or layerwise LR decay are the next candidates.
3. Does any training-time change on #1736's stack unlock novel levers that TTT *can't* absorb? This is the meta-question — if TTT absorbs quant-side changes, does it also absorb training-time changes? Spec 011 is the first test.
4. Per-rank analysis of spec 010's data (deferred): would confirm the rotation regime-dependence replicates across rank boundaries. Worth doing if we end up writing this up as a research artifact post-deadline.

## What I'd do differently if we ran SpinQuant from scratch

- **Demand final.json numbers before any interpretation.** Don't read `rb` trajectories as forecasts.
- **Run `baseline` + `port_1695` first** before drilling into site ablations. The "SpinQuant is exhausted" finding would have been clear 6 hours earlier, saving the spec 010b design cycle.
- **Look at `diagnostic_quantized` (pre-TTT) spread first.** If it's under ~0.001 across variants, TTT will smash everything to near-identical post-TTT numbers. Spec 009's 0.00003 gap was already a strong hint.
- **The regime-dependence framing was too fast.** "Rotation helps long, hurts short" was a real observation but I leapt to "this is exploitable" before checking whether the aggregate was exploitable. The aggregate is what matters.

## State going into next session

- `fork/research` at commit `8815c4d` (rotation-regime-dependence idea file pushed).
- All 5 SpinQuant variants measured; runs in `runs/{009,010,010b}-*/`.
- Spec 011 doc ready; code not written.
- Spec 010b's summary.md has the full comparative analysis; worth reading before touching SpinQuant again.

Done for the day on SpinQuant. Tomorrow: spec 011 patch + run.
