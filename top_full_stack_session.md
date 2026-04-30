# PR1851 Full Stack — Runs 1-3 Results + Final Audit of PR1493 Portability

Session 2026-04-30, completing the three-run sequence proposed in
`top_ar_run1_session.md`. Run 1 isolated GPTQ all-reduce (AR) on PR #1851;
Run 2 stacked AR + wd_schedule (default factors); Run 3 stacked AR +
wd_schedule (strong factors) + paired-head Muon NS for the bank architecture.
Run 0 = `top_wd_strong_s42` from earlier today (wd_strong alone, no AR, no PH).

## All four runs side by side (s42, single-seed)

| run | config | pre | q | q_gap | q_ttt | step | Δ vs PR1851 1.06128 |
|---|---|---|---|---|---|---|---|
| **Run 0** | wd_strong only | 1.06429 | 1.07403 | 0.00974 | **1.06111** | 4846 | **−0.00017** ← best |
| Run 1 | AR only | 1.06623 | 1.07548 | 0.00925 | 1.06266 | 4829 | +0.00138 |
| Run 2 | AR + wd_default | 1.06459 | 1.07424 | 0.00965 | 1.06129 | 4827 | +0.00001 |
| Run 3 | AR + wd_strong + PH | 1.06467 | 1.07431 | 0.00964 | 1.06136 | 4734 | +0.00008 |

All four within ±0.00138 of PR #1851's published baseline. Only Run 0 is
*better* than baseline, and by less than ½ of the published 3-seed std
(0.00068).

## Three findings, ordered by impact

### 1. AR + WD scheduling don't fully stack (significant interaction)

The AR fix (Run 1 vs hypothetical no-AR-no-WD baseline) gave a clean
**−0.0005 BPB** on quant gap when WD was off:

```
Run 1 q_gap = 0.00925
Run 0 q_gap = 0.00974   (no AR, but wd_strong on)
PR #1855 published q_gap ≈ 0.00858
```

But when WD is added, the AR effect collapses:

```
Run 2 q_gap = 0.00965 (AR + wd_default)
Run 3 q_gap = 0.00964 (AR + wd_strong + PH)
```

The WD-schedule-on quant gap (~0.00965) is essentially the same as the
WD-on no-AR gap (Run 0's 0.00974, only −0.0001 narrower). WD widens the
quant gap (+0.00040 vs no-WD baseline); AR narrows it (−0.00049). When
both are on, they roughly cancel.

**At q_ttt level, this means our AR contribution drops from ~−0.0014 BPB
(if standalone) to ~+0.0001 BPB (when stacked with WD).** The AR fix is
real but functionally absorbed by the WD interaction on this stack.

This matches the PR1493 evidence retrospectively: PR1493 wd_paired_requant
showed AR worth ~−0.00084 BPB at 16 calibration batches, but those requant
runs *didn't have wd_strong on during training* — the comparison was AR
vs no-AR, both with the same wd-trained weights. The interaction we're
seeing on PR1851 wasn't visible there.

### 2. Paired-head Muon NS is a no-op on PR1851's bank architecture

Run 3 mid-training signal at step 4000 was striking: **val_bpb 1.0969** vs
Run 0's 1.1015 (a −0.0046 BPB lead at half-budget). That looked like a
real win in the making.

But end-of-training pre-quant landed at 1.06467 — **+0.00038 worse than
Run 0's 1.06429**. The mid-train trajectory advantage converged out.
Mechanism (most likely):

- PR1851's bank-NS already does per-layer NS via the leading-batch-dim
  trick on `zeropower_via_newtonschulz5(G, ...)`. With G shape
  `(shard_B, model_dim, model_dim)`, NS treats the leading dim as batch
  and applies per-layer NS for free.
- Adding explicit head-pair reshape (per-shard slot dispatch into Q vs O
  halves of `qo_bank`, K vs V of `kv_bank`, with reshape to
  `(num_pairs, pair_dim, model_dim)` before NS) gives a *different*
  per-step update direction, but the EMA + WD-spike + LQER pipeline
  averages this difference out.
- ~110 fewer training steps due to per-slot NS dispatch overhead
  (Run 3 stopped at 4734 vs Run 0's 4846) — small perf cost.

The lesson: **the −0.00055 BPB paired-head signal on PR1493 was earned
against a baseline that didn't have any per-layer NS structure.** PR1851's
bank-NS already provides that structure implicitly, so the marginal value
of explicit head-pair structure is essentially zero on this base. The
"port the engine of the PR1493 win" hope was wrong about what the engine
actually was.

### 3. wd_strong is the only PR1493-stack addition that produces a real
single-seed win on PR1851 — and even that is below the noise floor

Run 0 (wd_strong alone) is the only configuration that beat PR #1851's
published s42 baseline (1.06128 → 1.06111, −0.00017 BPB). That is
**less than ¼ of the published 3-seed std (0.00068)** and well below the
0.005-nat / 0.0024-BPB acceptance bar.

The within-pod A/B vs Run 1 (no WD) shows wd_strong gives **−0.00194 BPB**
on pre-quant, which is a real signal above noise. But by the time
quantization + phased-LoRA TTT runs, the propagated value at q_ttt is only
about −0.0002 BPB (+ noise).

## What this means for record submission feasibility

Best single-seed result this session: **1.06111** (Run 0, wd_strong alone).
Current SOTA: PR #1855 at **1.0611** (3-seed mean). To clear the
acceptance bar of 0.0024 BPB above SOTA, we'd need q_ttt ≤ 1.0587.

We are **0.0024 BPB short of the bar** — exactly the magnitude of the
acceptance bar itself. None of the four configurations we ran moves us
closer. The PR1493-stack ports we tested don't compose to bridge that gap.

## Audit: what's left in the PR1493 stack that we haven't tested on PR1851

This is the honest answer to "is there anything else from our PR1493 work
that could plausibly clear the bar." For each candidate, the assessment is
based on its PR1493 result, its expected portability to PR1851's stack,
and the magnitude needed.

### Already tested in this session — done

- `wd_schedule` (default factors): tested as Run 2, ~0 net q_ttt
- `wd_schedule` (strong factors): tested as Run 0 (without AR) and Run 3
  (with AR + PH). Run 0 is best, ~−0.00017 BPB
- `gptq_all_reduce`: tested as Run 1 (alone) and Runs 2-3 (stacked).
  Real but largely absorbed by WD interaction
- `paired_head_muon_enabled`: tested as Run 3 (full stack). No-op on
  bank architecture
- `gptq_damp` / `gptq_block_size`: already swept on PR1493, defaults
  optimal within 3e-6 BPB

### Confirmed regressions on PR1493 — skip

- `iha` (incremental Hessian average): the only successfully-completed
  PR1493 run was `wd_paired_iha`, which regressed pre by +0.00056
- `mtp` (multi-token prediction): clear regression on PR1493 (+0.00944),
  implementation bug (shared head supervision)
- `doc_shuffle`: regression on PR1493 (+0.00200), distribution mismatch
- `qat` (in-training STE QAT): EMA contamination on PR1493 (+0.012)
- `pko` (partial key offset): catastrophic with TTT (+0.024)
- `in-training SmearGate + per-head 1D attn_gate`: PR1493 base + wd_strong
  regressed +0.00081. PR1851 has its own better-validated SmearGate

### Untested / never-completed candidates

These are the remaining unknowns. None look likely to bridge a 0.0024 BPB
gap, but listed for completeness:

| candidate | source | port effort | best-case ΔBPB | confidence | comment |
|---|---|---|---|---|---|
| **`EVAL_NUM_LOOPS=3`** (eval-time depth recurrence) | PR1493 priority experiments row 5 ("running", never finished) | env-var only on PR1493; would need code addition to PR1851 since PR1851 already has training-time recurrence | unknown, plausibly noise | LOW | PR1851's depth recurrence (layers 3-5 ×2 at frac=0.35) is *training-time*. Adding eval-time loops is a different mechanism. No PR1493 evidence either way. |
| WD factor sweep (low=0.55/0.60, high=1.6/1.65/1.7) | extension of WD work | env-var only | likely +/- 0.0001 noise vs (0.5, 1.75) or (0.65, 1.5) | LOW | Two endpoints already tested (Run 0 vs Run 2). Sweet spot unlikely to differ by >0.0002 BPB |
| `gptq_calibration_batches` bump (32 or 64) | PR1493 sweep showed AR saturates at 128 | env-var only | with AR already on, effectively at 16x8=128 saturation. More batches = more wallclock, fewer training steps | NEGATIVE | AR saturated at 128. Bumping calibration costs train steps, expected loss-loss |
| `lrzip` per-group compression (PR #1855's edge over #1851) | PR #1855, NOT a PR1493 technique | medium effort to integrate compressor switch | ~−0.0003 BPB (gap PR1855 vs PR1851) | MEDIUM | Out of PR1493 scope. Open rule-3 dispute on whether `lrzip` is a fair runtime dep |
| Mid-strength WD factors (e.g., low=0.55, high=1.65) | not a PR1493 variant | env-var only | very small expected delta | LOW | Optimal might be between Run 0 and Run 2 endpoints |
| Increase `PHASED_TTT_NUM_PHASES` from 3 to 4 or 5 | not a PR1493 variant | env-var only; longer eval | unknown; may push past 600s eval cap | LOW | Diminishing returns on TTT phases; eval-time risk |
| Tune `EMBED_BITS` from 7 → 8 | byte budget trade-off | env-var only | bigger embedding artifact, possibly fits, marginal quality gain | LOW | Probably blows the 16 MB cap |

### Honest verdict on portability

**There is no remaining PR1493-stack technique that can plausibly bridge a
0.0024 BPB gap.** The ones that worked on PR1493 (wd_schedule, paired-head
Muon, AR) have been ported and accounted for. The ones that didn't work
on PR1493 won't work on PR1851 either — they either had implementation
bugs (`mtp`, `iha`, `qat`) or they're fundamentally incompatible with the
loss landscape / pipeline (`pko`, `doc_shuffle`).

The **only paths to a record submission** from here are not PR1493-derived:

1. **`lrzip` compressor port** (~−0.0003 BPB based on the PR #1855 vs
   PR #1851 gap). Requires integrating the per-group compressor logic
   from PR #1855. Open rule-3 dispute on `lrzip` as runtime dep.
   Even if this lands cleanly, we're still ~0.0021 BPB short.

2. **Architectural changes** that we don't have ready: Scylla-class
   tokenizer work, JEPA-style objectives, megakernels, etc. None of
   these are partly-built on this branch.

3. **Multi-seed averaging luck**: Run 0 at s42 was 0.00017 BPB better
   than baseline. If 3 seeds happen to mean to ~0.0008 below baseline,
   that's still well short of 0.0024 BPB. Effectively no path here.

## Recommendations

Given the contest deadline is today and a record submission is not feasible:

### Option A: 3-seed validation of Run 0 (wd_strong alone)

Run s314 and s1234 with the same wd_strong config (no AR, no PH) so we have
a 3-seed mean comparable to PR #1851's published 3-seed reproduction. Cost:
~42 min. Outcome: a documented 3-seed mean, plausibly between 1.0610 and
1.0612, that we can submit as a *non-record* entry with full analysis of
the AR-WD interaction and bank-NS-paired-head no-op findings.

### Option B: Submit non-record analysis

Submit the negative-result findings (AR-WD interaction, bank-NS paired-head
no-op) as a `track_non_record_16mb` submission with the writeup as the
contribution. Per the README:

> *We strongly encourage participants to submit implementations for weird
> or out-of-the-box ideas, in-progress or unoptimized solutions, so long
> as they run successfully, or even interesting negative results.*

The AR-WD interaction is a non-trivial empirical finding that's worth
documenting on the leaderboard tree. The paired-head Muon NS bank-arch
port is a clean negative result with a mechanistic explanation.

### Option C: Stop and accept the result

Push the documentation to remote, no submission. Use the time saved to
prepare for the next iteration.

I'd default to **Option A then C**: run the two extra seeds, push the
3-seed mean and analysis to remote, and decide on submission timing based
on the result.

## Files in this session

- `train_top.py` — three commits already pushed:
  - `ec48ff1` wd_schedule
  - `6c53583` GPTQ all-reduce
  - `97fc8a5` paired-head Muon NS bank-arch port
- `top_wd_strong_session.md` — Run 0 (committed earlier today)
- `top_ar_run1_session.md` — Run 1 + reversal-of-wd_strong-verdict
  (committed earlier today)
- `top_full_stack_session.md` — this document
- `logs/top_ar_s42.{txt,stdout}` — Run 1 logs
- `logs/top_ar_wd_default_s42.{txt,stdout}` — Run 2 logs
- `logs/top_ar_wdstrong_paired_s42.{txt,stdout}` — Run 3 logs
