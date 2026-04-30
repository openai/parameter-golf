# Run 4: PR1851 + 9-Hparam Stack + wd_strong + AR — Session, 2026-04-30

This run tests the colleague-provided "9 PR1855 hparam overrides" as a hybrid
on top of our existing PR #1851 base + wd_strong + AR. Outcome: **best
quality of the session** (q_ttt = 1.05950, beats PR #1855 published s42 by
0.00039 BPB), but **artifact busts the 16 MB cap by 140 KB** because we don't
have PR #1855's per-group compressor. This run cemented the pivot to PR #1855
base for Run 5.

## Setup

```bash
RUN_ID=top_pr1855_hparams_s42 SEED=42 \
CASEOPS_ENABLED=1 EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
GPTQ_ALL_REDUCE=1 \
WD_SCHEDULE_ENABLED=1 WD_SCHED_LOW_FACTOR=0.5 WD_SCHED_HIGH_FACTOR=1.75 \
EMBED_CLIP_SIGMAS=14.0 \
MLP_CLIP_SIGMAS=11.5 \
WARMDOWN_FRAC=0.85 \
BETA2=0.99 \
TTT_BETA2=0.99 \
TTT_WEIGHT_DECAY=0.5 \
TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
PHASED_TTT_PREFIX_DOCS=2500 \
torchrun --standalone --nproc_per_node=8 train_top.py
```

Base: `train_top.py` (PR #1851 + 3 patches: wd_schedule, AR, paired-head Muon
NS — paired-head OFF here).

The 9 highlighted hparams are colleague-suggested values, claimed to be
PR #1855's greedy-tuned stack. Verified later against PR #1855's upstream
README — all 9 match exactly. That confirmation came after Run 4 launched.

## Results

```text
pre   = 1.06330575   (post-EMA, pre-quant)
q     = 1.07238835   (post-LQER asymmetric quantization)
q_ttt = 1.05950377   (post-phased-LoRA TTT — primary metric)
size  = 16,140,607 B (BUSTS 16 MB cap by 140,607 B with brotli)
stop  = 4788/20000   (wallclock cap = 592169 ms)
total_eval_time = 520.0s
```

### Comparison to all prior runs of the session and PR #1855 published

| run | config | pre | q | q_gap | q_ttt | artifact | Δ q_ttt vs Run 4 |
|---|---|---|---|---|---|---|---|
| Run 0 | wd_strong only (no AR, no PH) | 1.06429 | 1.07403 | 0.00974 | 1.06111 | 15,948,542 | +0.00161 |
| Run 1 | AR only (no WD) | 1.06623 | 1.07548 | 0.00925 | 1.06266 | 15,956,401 | +0.00316 |
| Run 2 | AR + wd_default | 1.06459 | 1.07424 | 0.00965 | 1.06129 | 15,950,537 | +0.00179 |
| Run 3 | AR + wd_strong + PH | 1.06467 | 1.07431 | 0.00964 | 1.06136 | 15,948,493 | +0.00186 |
| **Run 4** | **9hp + wd_strong + AR** | **1.06331** | **1.07239** | **0.00908** | **1.05950** | **16,140,607** | — |
| PR #1855 published s42 | (their stack, 16 cal AR-off) | 1.06396 | 1.07254 | 0.00858 | 1.05989 | 15,897,259 | +0.00039 |
| PR #1855 3-seed mean | | — | — | — | 1.06108 | 15,901,919 | +0.00158 |

**Run 4 is the best q_ttt of the session and 0.00039 BPB better than
PR #1855's *published s42*.** That margin is single-seed and below their
3-seed std (0.00090), but the within-pod evidence (Runs 0-4 all on this pod)
showing wd_strong's pre-quant gain is real makes this not pure pod-noise.

## Three findings

### 1. The 9 hparams transfer cleanly through to final model quality

In contrast to paired-head Muon NS (Run 3), which gave a striking mid-train
signal (−0.0046 BPB at step 4000) that converged out by pre-quant time
(+0.00038 vs Run 0), the 9 hparams **carry their gain through**:

```text
Mid-train val at step 4000:  Run 4: 1.0956 (best)
                             Run 3: 1.0969 (also good, but lost it)
                             Run 0: 1.1015

Pre-quant val (post-EMA):     Run 4: 1.06331 (best, kept it)
                              Run 3: 1.06467 (lost the gain)
                              Run 0: 1.06429
```

Mechanism: paired-head NS changes the optimizer's update direction (a
*trajectory* change). The EMA + WD-spike + LQER pipeline averages it out.
The 9 hparams change *what's actually being trained*: tighter clipping
(MLP=11.5, EMBED=14.0) preserves outliers better; longer warmdown
(WARMDOWN_FRAC=0.85) gives more time to converge under low LR; smaller
TTT-LoRA rank (80) and tuned TTT-Adam β2 (0.99) reshape the TTT recovery.
None of this is averaged out by the EMA.

### 2. Tightest quant gap of the session (0.00908)

Run 4's q_gap (q − pre = 0.00908) is the smallest gap of any run today,
beating Run 1's AR-alone gap (0.00925) by 0.00017 BPB. The MLP/EMBED
clipping changes preserve more of the weight distribution's outliers,
which LQER's asymmetric int4 rank-4 correction can then exploit. AR
contributes another piece of the gap-narrowing.

Still 0.00050 wider than PR #1855's published 0.00858. The remaining gap
likely lives in the per-group compressor's correlated reordering, which we
don't have, AND in pod/seed environment noise.

### 3. The hparam stack busts the 16 MB cap with brotli alone

Artifact: **16,140,607 B = 140,607 B over cap**.

Why: tighter MLP/EMBED clipping preserves more outliers, which means the
quantized weight distribution has slightly larger dynamic range. brotli
compresses that ~140 KB worse than the noisier baseline weights. PR #1855's
README explicitly claims their per-group lrzip+brotli compressor saves
**~280 KB** vs straight brotli on this stack — meaning their model would
land ~16,180,000 B with brotli, busting cap by an even bigger margin if
they didn't have pergroup.

**Consequence**: this run's q_ttt is real but unsubmittable. We cannot
ship a 16,140,607-byte artifact. We need pergroup compression.

## Decision: pivot to PR #1855 base for Run 5

Earlier in the session I had re-confirmed (with the user) the previous
session's choice to clone PR #1851 instead of PR #1855. The argued
justification was: "fewer disputes, no lrzip dep". The colleague's analysis
and Run 4's evidence overturn that:

| | PR #1851 (where we are) | PR #1855 (#1) |
|---|---|---|
| 3-seed mean | 1.06145 | 1.06108 (−0.00037) |
| Per-group lrzip+brotli compressor | NO | YES (~280 KB smaller artifact) |
| 9 greedy-tuned hparams | NO | YES (matched manually in Run 4) |
| Open disputes that affect us | val_docs (inherited) | val_docs (inherited) + lrzip (resolved) |

Run 4 demonstrates the 9 hparams alone deliver −0.0014 BPB at q_ttt
*without* the compressor. With the compressor, we'd recover ~280 KB of
artifact budget, which we could spend on quality elsewhere if needed,
and at minimum we'd be under-cap.

`lrzip 0.651` installed via `add-apt-repository -y universe; apt-get
install -y lrzip` mid-session.

PR #1855 source files (`train_gpt.py` 3,753 lines, `lossless_caps.py`
identical to PR #1851's, `prepare_caseops_data.py` ~10 lines of comment
differences, README + submission.json + requirements.txt) downloaded to
`_top_ref_1855/`.

`train_top_1855.py` = PR #1855's `train_gpt.py` + same surgical patches we
applied to PR #1851 (wd_schedule + AR). 41 lines added, 3 modified, syntax
OK. Paired-head Muon NS port skipped — confirmed no-op on bank arch (which
PR #1855 also uses).

## Run 5 (queued, will auto-launch when Run 4 GPUs free)

```bash
RUN_ID=top1855_wd_strong_ar_s42 SEED=42 \
CASEOPS_ENABLED=1 EMBED_BITS=7 \
ITERATIONS=20000 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup \
WD_SCHEDULE_ENABLED=1 WD_SCHED_LOW_FACTOR=0.5 WD_SCHED_HIGH_FACTOR=1.75 \
GPTQ_ALL_REDUCE=1 \
torchrun --standalone --nproc_per_node=8 /parameter-golf/train_top_1855.py
```

Difference from PR #1855's published run command:
- We add `WD_SCHEDULE_ENABLED=1, WD_SCHED_LOW_FACTOR=0.5, WD_SCHED_HIGH_FACTOR=1.75`
- We add `GPTQ_ALL_REDUCE=1`
- `GPTQ_RESERVE_SECONDS=4.0` instead of their `0.5` (give AR all-reduces
  enough headroom; costs ~33 train steps)

If Run 5's q_ttt lands at ~1.0590, we have a single-seed result very close
to the **1.0588** acceptance bar (clear PR #1855 by 0.005 nats). If similar
across 3 seeds, that'd be a real submission candidate.

## Honest acceptance-bar math

```text
Current SOTA (PR #1855 3-seed mean) = 1.06108 BPB
Acceptance bar (SOTA - 0.005 nats / ~0.0023 BPB) ≈ 1.0588 BPB

Run 4 single-seed = 1.05950
  - vs SOTA mean: -0.00158 BPB (~1.75σ given 0.00090 std)
  - vs acceptance bar: +0.00070 BPB short

Run 5 (predicted, if PR #1855 base + our patches stacks similarly)
  ~ 1.0590 to 1.0595 single-seed
  3-seed mean ~ 1.0593 ± 0.001
  - still 0.0005-0.0010 short of bar
```

So even this best-case stack likely *just misses* the record bar by ~half a
sigma. A real submission attempt would need either (a) one of the 3 seeds
to be lucky and hit the bar, AND/OR (b) the wd_strong gain to compose
better with PR #1855's stack than we're forecasting.

**But:** even at q_ttt 1.0593, this is a clear non-record submission with
documented findings (AR + WD interaction, hybrid hparam transfer, bank-NS
paired-head no-op).

## Files committed in this commit window

- `top_run4_session.md` — this document
- `logs/top_pr1855_hparams_s42.{txt,stdout}` — Run 4 logs
- (`train_top_1855.py` + `_top_ref_1855/*` were committed already in
  `611b598`)

## Run 5 status

Auto-launched ~17:23 UTC, hparam dump confirmed all 19+ critical settings
(LQER, fused CE, pergroup compressor, AR, wd_strong, all 9 PR1855
overrides). Mid-train val + post-train diagnostics imminent.
