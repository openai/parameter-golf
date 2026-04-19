# 2026-04-19 → 20 — Spec 004 null result + two methodology bugs found

Spec 004 was a three-phase saga. **QK_GAIN_INIT extensions above 5.25 don't help.** Clean kill. More interesting than the null result: we discovered two methodology bugs that have been silently contaminating our screening workflow for multiple specs.

## The null

Final numbers (Phase 3, clean verification, matched RNG trace):

| Run | QK | pre-quant post-EMA | Δ vs spec 000 |
|---|---|---|---|
| spec 000 | 5.25 | 1.09289 | (base) |
| 004c | 6.0 | 1.09193 | −0.00096 |

That −0.001 is **effectively zero** — within 5× SOTA seed std, smaller than the natural +0.002 benefit 004c got from running 103 extra training steps (3952 vs 3849). QK=6.0 is tied with QK=5.25. QK=5.5 similarly (from Phase 1 Run B data, not cleanly verified but no reason to expect it wins where 6.0 doesn't).

Four consecutive screen specs (001, 002, 003, 004) all killed now. All four had "should be a small win" priors from near-SOTA submissions or literature. None transferred to the April-09 SOTA stack.

## Methodology bug #1: bf16 non-determinism across physical pods is HUGE

**Phase 1's Run A showed Δ −0.109 at step 1000 vs spec 000.** That would have been the biggest single-experiment win I've ever pitched. It looked real — monotonic divergence, consistent across steps 200/400/600/800/1000.

It was 100% noise. Same seed, same config, same code, same commit. Different physical H100 node. bf16 operations are non-deterministic across hardware at the bit level, and those tiny rounding differences **compound over ~1000 steps to ~0.1 bpb of train_loss divergence**.

Phase 3 re-verified on a similar pod with matched RNG trace: Δ −0.0009, not −0.109. The "signal" was entirely hardware.

**Direct implication for me:** everything I said about "same seed + spec-000 log as control = clean screen" was wrong. That methodology silently has ~0.01-0.1 bpb of pod-variance noise baked in, dwarfing the 0.0005-0.002 bpb signals we're trying to detect.

Fix going forward: for any training-time screen where pod matters, we need either:
- An in-run control (paired variants on the same pod, like spec 003 did correctly)
- A final-stage eval number rather than mid-training train_loss comparison
- Multiple seeds to average out the hardware noise (expensive)

Spec 003 accidentally did this correctly (we stayed on the same pod as Exp 24 reference... actually no, we used a different pod. The spec 003 result holds up because BigramHash showed a consistent +0.001 bpb bias at the final stage, which is harder to explain by bf16 drift than mid-training divergence. But we should re-examine spec 003's numbers with fresh skepticism.)

## Methodology bug #2: VAL_LOSS_EVERY affects the RNG trace

Phase 2 (004b) set `VAL_LOSS_EVERY=200` expecting it to give us denser val data. Instead, dense val sampling consumed RNG calls between training steps, which **shifted training data ordering** compared to the baseline (which used VAL=4000). So "same seed" didn't mean "same data order" anymore.

This explains some weird discrepancies in spec 001 and spec 002 that I'd attributed to calibration differences. It was actually RNG-trace drift from val-cadence differences.

Fix: for any same-seed comparison, match the baseline's VAL_LOSS_EVERY exactly. Documented in memory.

## The one consistent theme across all four kills

Something keeps coming up: **artifact size grows with the intervention.** 

- Hessian-SDClip at λ≥0.40: >16MB artifact
- SWA+EMA C1 (pure SWA): artifact slightly grows
- BigramHash: ~16.2MB artifact
- QK_GAIN=6.0: 16.046MB artifact — **over the 16MB cap**

Every single "promising candidate" we've shelved has produced weights that compress worse. The quant pipeline is a tight coupling: any intervention that sharpens some signal (attention peaks, quant thresholds, embedding variance) tends to raise entropy elsewhere and hurt Brotli compression.

SOTA seems to be at a multi-axis optimum where every knob we push disturbs the budget balance. Not just bpb-vs-nothing, but bpb-vs-size-vs-entropy.

## Cost reality check

Spec 004 was spec'd at $7 and cost **$19.77** (2.8× over). Causes:
- Phase 2 re-run needed after RNG-trace confusion
- Phase 3 verification needed after Phase 1 signal turned out to be pod noise
- NCCL crash during Phase 3 post-training — recoverable because pre-quant landed before crash, but lost some wall clock

Running total after 4 specs:
- Spec 000: $13.10
- Spec 001: $1.90
- Spec 002: $2.60
- Spec 003: ~$4
- Spec 004: $19.77
- **Total: ~$41-43 / $200 hard budget**
- **Remaining: ~$157, 10 days to 2026-04-30**

## Where this leaves us strategically

The "screen cheap candidates" playbook has produced zero wins in four attempts. At $43 for zero bpb improvement, the ROI is terrible. I can't credibly keep pitching "cheap screen for small Δ" as the path forward.

Options I see:

**1. Pivot to bigger bets.** Accept that the remaining candidates worth trying are full-retrain architectural changes. Things like SwiGLU, expanded depth recurrence (num_loops=3 or loop_layers=2-6), or untied MLPs in loop passes. Each is ~$10-15 per run with the methodology fixes baked in. Budget for 3-5 such runs = $50-60.

**2. One last cheap shot: AR self-gen GPTQ calibration.** Still in the idea queue. Quant-time, hotstart-able, ~$2. Different mechanism than the four shelved candidates (it's calibration-data-not-clip-formula). Low expectations but not-yet-tested.

**3. Accept the 1.08622 baseline.** Stop spending. Write up what we learned, submit what we have. We'd be ~0.005 bpb over the leaderboard, which is worse than SOTA but we'd have a complete research record of what doesn't work on this stack. Honest outcome but not a record.

**4. Re-examine whether we missed something structural.** Maybe the whole "improve SOTA's pipeline" framing is wrong and the record path requires a different architecture (e.g., very different layer count, different attention pattern, different recurrence structure). Bigger rethink required.

My read: I'd do (2) as a quick $2 finisher on the cheap-screen track — if AR self-gen works, great; if not, we have clean data showing all cheap candidates are exhausted. Then (1) for the remainder of the budget: commit to 2-3 structural bets and accept that each is a $10-15 gamble.

Option (3) stings but is honest if (1)+(2) fail.

## Lessons to bank

- **Pod-level bf16 non-determinism compounds fast.** Don't compare train_loss across pods without explicit variance characterization.
- **`VAL_LOSS_EVERY` affects RNG trace.** Match it to the baseline for any seeded comparison.
- **Kill on `stopping_early`, skip post-training eval** on screens. Saves $3-4/run.
- **Artifact size is coupled to bpb interventions.** Track both on every screen, not just bpb.
- **"Near-SOTA had technique X, it gave +N bpb"** does NOT transfer to the current stack. The architecture has moved too far. Four consecutive near-SOTA-inspired candidates have now failed with the same null pattern. Stop treating these as high-EV.

## Takeaway on my own framing

I've been setting unreasonable expectations on cheap screens. A candidate that gives +0.0005 bpb is historically a valid record contribution. But I kept pitching them as "likely to give +0.002-0.005" based on source-submission claims that didn't transfer. Going forward I should:

- Frame cheap screens as "rule out" tests, not "unearth +0.003" tests.
- Budget expectations: 30% probability of ANY positive signal per cheap screen, ~0.0005 bpb expected if positive.
- For record-level Δ, expect to need 2-3 stacked small wins + at least one $10-15 full-retrain architectural change.

This is not good news for our 10-day timeline, but it's accurate.
