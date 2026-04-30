# Run 5: PR1855 + wd_strong + GPTQ AR

Session 2026-04-30. This document records the result of running the PR #1855
script base with our two surviving PR1493-derived additions stacked on top:

- `wd_schedule` with strong factors: `low=0.5`, `high=1.75`
- GPTQ Hessian all-reduce (`GPTQ_ALL_REDUCE=1`)

The run was launched from `train_top_1855.py`, which is PR #1855's
`train_gpt.py` plus the surgical `wd_schedule` and GPTQ all-reduce patches.
Paired-head Muon NS was intentionally not included because Run 3 already
showed it is a no-op or slight regression on the bank-parameter architecture.

The numbers below are copied from the remote run report. The corresponding
Run 5 logs are not present in this checkout at the time this document is
created.

## Setup

Base: PR #1855 stack:

- CaseOps tokenizer and byte sidecar
- BOS-fixed SmearGate
- Sparse attention gate
- Polar-Express Muon
- fused softcapped CE
- LQER asymmetric rank-4 correction
- per-group `lrzip + brotli` compression
- PR #1855's 9 greedy hparam overrides
- phased LoRA TTT

Additional patches from our stack:

```bash
WD_SCHEDULE_ENABLED=1
WD_SCHED_LOW_FACTOR=0.5
WD_SCHED_HIGH_FACTOR=1.75
GPTQ_ALL_REDUCE=1
```

Expected run command shape:

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
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 \
LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup \
WD_SCHEDULE_ENABLED=1 WD_SCHED_LOW_FACTOR=0.5 WD_SCHED_HIGH_FACTOR=1.75 \
GPTQ_ALL_REDUCE=1 \
torchrun --standalone --nproc_per_node=8 train_top_1855.py
```

## Result

Reported final Run 5 primary metric:

```text
q_ttt = 1.06009
```

The full reported pre/q/size breakdown is not available in this checkout.
The analysis below therefore treats q_ttt as authoritative and avoids
over-interpreting stage-level mechanics beyond what can be compared to
Run 4 and PR #1855.

## Comparison

| run | base | additions | q_ttt | artifact status | note |
|---|---|---|---:|---|---|
| Run 4 | PR #1851 + manually-applied PR #1855 9 hparams | wd_strong + AR | 1.05950 | invalid, 16,140,607 B | best score, but brotli-only artifact busts cap by 140,607 B |
| **Run 5** | **PR #1855 full script** | **wd_strong + AR** | **1.06009** | expected valid via pergroup | 0.00059 worse than Run 4 |
| PR #1855 published s42 | PR #1855 | none | 1.05989 | valid, 15,897,259 B | Run 5 is 0.00020 worse |
| PR #1855 3-seed mean | PR #1855 | none | 1.06108 | valid, 15,901,919 B mean | current SOTA mean |

### Run 5 vs Run 4

Run 5 is **+0.00059 BPB worse** than Run 4 within the same broad experiment
window, despite using the full PR #1855 script and per-group compressor.

Two plausible mechanisms:

1. **PR #1855 was greedy-tuned without `wd_strong`.** Adding the strong
   late WD schedule changes the training distribution that the 9 hparams,
   LQER, and phased TTT were selected around. The resulting interaction is
   mildly negative.
2. **The PR #1855 code delta is not just a compressor.** The extra source
   relative to the PR #1851-hybrid path may include small training-side
   differences, not only serialization. Those differences do not appear to
   compose favorably with `wd_strong` on this pod.

We cannot cleanly disentangle these without an unmodified PR #1855
reproduction on the same pod. That run was intentionally skipped to save
time.

### Run 5 vs PR #1855 published s42

Run 5 is **+0.00020 BPB worse** than PR #1855's published seed-42 result:

```text
Run 5 q_ttt              = 1.06009
PR #1855 published s42   = 1.05989
delta                    = +0.00020 BPB
```

Possible explanations:

- pod-to-pod environmental noise
- a real regression from `wd_strong + AR` on this base
- both

The earlier session already observed a large pod/reference mismatch in this
family, so a 0.00020 difference is not enough to attribute confidently.
Still, the important practical read is clear: **Run 5 does not improve the
published PR #1855 seed-42 result.**

## Acceptance-bar reality check

Current SOTA from PR #1855:

```text
PR #1855 3-seed mean      = 1.06108
PR #1855 3-seed std       = 0.00090
Run 5 single-seed q_ttt   = 1.06009
Run 5 vs SOTA mean        = -0.00099 BPB
```

The record acceptance rule is framed as **0.005 nats** better than current
SOTA, which is roughly **0.0023 BPB** for this stack. That puts the practical
bar around:

```text
acceptance target ~= 1.0588 BPB
Run 5 q_ttt       = 1.06009 BPB
gap to bar        ~= +0.00129 BPB short
```

Even if Run 5's 3-seed mean held near the seed-42 value, it would still miss
the acceptance bar by about 0.0013 BPB. This is not a record candidate.

## What this says about our PR1493-derived additions

At this point, the useful PR1493-derived ideas have been tested on the modern
CaseOps/SmearGate/LQER/phased-TTT base:

| addition | evidence | verdict |
|---|---|---|
| `wd_strong` | Run 0 was best on PR #1851 base, but Run 5 is worse than PR #1855 s42 | weak single-seed signal; not record-sized |
| GPTQ all-reduce | helped low-calibration PR1493 requant; did not stack cleanly with WD on PR #1851/1855 | keep only if it does not cost steps or quality |
| paired-head Muon NS | Run 3 mid-train lead vanished; final q_ttt regressed vs Run 0 | drop on bank architecture |
| IHA | regressed on PR1493 and costs steps | skip |
| MTP | clear PR1493 regression | skip |
| QAT | multiple PR1493 variants net negative | skip |
| PKO | breaks TTT / net negative | skip |
| doc shuffle | PR1493 regression | skip |
| Hessian damp/block sweeps | defaults already optimal within noise | skip |

There is no remaining PR1493-stack technique with enough evidence or expected
magnitude to close the roughly 0.0013 BPB gap from Run 5 to the acceptance
bar.

## Remaining experiment options

### Option 1: Run two more seeds of Run 5

Run seeds 0 and 1234 with the exact Run 5 config.

Cost: about 42 minutes.

Expected outcome: a 3-seed mean around 1.0601 with roughly PR #1855-like
variance. This would likely be a strong **non-record** result, not a record.

This is useful only if the goal is a clean non-record submission documenting
that `wd_strong + AR` does not beat PR #1855 after 3-seed validation.

### Option 2: Run 5b, drop AR

Config:

```bash
PR #1855 + wd_strong + pergroup, with GPTQ_ALL_REDUCE=0
```

Hypothesis: AR's gain is absorbed by the PR #1855 9-hparam/LQER pipeline and
may slightly hurt once `wd_strong` is present. Dropping AR may recover some
training steps and remove a small interaction.

Cost: about 21 minutes.

This is the best remaining cheap test because it directly probes whether Run 5
was hurt by AR. Expected gain is small, probably below 0.0003 BPB, but it is
more plausible than resurrecting paired-head or IHA.

### Option 3: Milder WD high spike

Config:

```bash
WD_SCHED_LOW_FACTOR=0.5
WD_SCHED_HIGH_FACTOR=1.5
```

Hypothesis: the high 1.75 late-WD spike helps pre-quant but worsens the
quant/LQER/TTT handoff. A softer high spike might preserve more of the pre
gain while avoiding quant-gap damage.

Cost: about 21 minutes.

This is plausible but lower EV than dropping AR because Run 5 already shows
the fully tuned PR #1855 stack does not like the current wd_strong interaction.

### Option 4: Stop and submit/write up as non-record

This is the pragmatic path if compute/time is tight. The result is not a
record, but the negative result is still useful:

- `wd_strong` does not compose cleanly with the full PR #1855 stack.
- GPTQ AR is not a free win once LQER/pergroup/PR1855 hparams are active.
- Paired-head Muon NS is structurally redundant with bank-wise batched NS.

## Recommendation

Run exactly one more single-seed variation before deciding:

```text
Run 5b = PR #1855 + wd_strong + pergroup, no AR
```

If Run 5b is not clearly better than `1.06009`, stop. If it improves by at
least 0.0004 BPB, then consider 3-seeding that config for a non-record
submission. If it somehow lands near or below 1.0588, switch back into record
candidate mode.

The probability of a record from remaining PR1493-derived stack pieces is low.
The best use of remaining time is a single targeted interaction test, not more
broad stacking.

## Files

- `train_top_1855.py` - PR #1855 script plus `wd_schedule` and GPTQ AR patches
- `_top_ref_1855/` - upstream PR #1855 record files
- `top_run4_session.md` - prior PR1851+9hp+wd+AR run, invalid due to size
- `top_run5_pr1855_wdstrong_ar_session.md` - this document
