# Arch+Sched Sweep — Hypothesis

**Date:** 2026-03-31
**Parent:** Rascal II (1.10986874 BPB, seed 444)
**Pod:** 4×H100
**Seed:** 444

---

## What this sweep is

Six 1-variable probes against the Rascal II baseline. All run at
`MAX_WALLCLOCK_SECONDS=600`, `NPROC=4`. On 4×GPU the LR warmdown is active
from step 1 (warmdown_ms ≈ 637s > 600s wallclock), so QAT fires at ~step 2800
and SWA at ~step 2650 — both inside the window.

---

## Cases

### baseline
Exact `sota_now.sh` env. Control. Expected: 1.10986874 BPB (may vary slightly
from wallclock jitter on 4×GPU vs 8×GPU).

### rope_32
`ROPE_DIMS`: 16 → 32
**Hypothesis:** More rotary dimensions give the model richer positional
encoding. Locked at 16 for conservatism; 32 may help without hitting the size
gate (purely algorithmic, zero size impact).

### bigram_3072
`BIGRAM_VOCAB_SIZE`: 2048 → 3072
**Hypothesis:** Competition leaders (PR #1019, #1179) use 3072 buckets. More
buckets = less hash collision in the 2-gram space. Est. +~50KB artifact
increase — well within 445KB headroom. This is the exact competition target.

### bigram_4096
`BIGRAM_VOCAB_SIZE`: 2048 → 4096
**Hypothesis:** Upper-bound test — if 3072 is good, does 4096 give more?
**Risk:** size gate. If this fails on size, 3072 is the answer.

### qat_early
`LATE_QAT_THRESHOLD`: 0.15 → 0.25
**Hypothesis:** Starting QAT earlier (~step 2420) gives more quantization-aware
fine-tuning steps before the run ends. Could tighten quant_gap.

### qat_late
`LATE_QAT_THRESHOLD`: 0.15 → 0.05
**Hypothesis:** Starting QAT later (~step 3120) lets the float model converge
further before QAT noise is introduced. Could improve post_ema_bpb at the cost
of fewer QAT steps.

### swa_dense
`SWA_EVERY`: 50 → 10
**Hypothesis:** More frequent weight averaging produces a smoother ensemble.
SWA fires at the same step, but accumulates 5× more snapshots before the run
ends. May help sliding_bpb without touching the training dynamics.

### gptq
`SKIP_GPTQ`: 1 → 0
**Hypothesis:** Full Hessian GPTQ is the biggest single gap vs competition.
The code is already written (vault lines 552–643). GPTQ_RESERVE_MS=30000 takes
30s off the training window → ~170 fewer steps on 4×GPU (~5% fewer steps).
Competition sees -0.003 to -0.009 BPB gain. Hessian error compensation should
more than offset the lost steps at our model size.

### warmdown_4k
`WARMDOWN_ITERS`: 3500 → 4000
**Hypothesis:** Longer warmdown gives the LR schedule more room to decay
smoothly. Competition leaders use 4000. Smallest expected gain (~-0.0005 BPB)
but zero risk — schedule change only.

---

## What to look for

| Metric | Why |
|--------|-----|
| `sliding_bpb` | Race metric — this is the score |
| `post_ema_bpb` | Float model quality; isolates training signal from quant |
| `quant_gap` | `int6_bpb - post_ema_bpb`; lower = QAT working |
| `size_bytes` | Must stay ≤ 16,000,000 bytes |
| `qat_step` | Confirms threshold fired at expected step |

A case is interesting if `sliding_bpb` drops vs baseline. `post_ema_bpb`
dropping but `sliding_bpb` flat = quant degradation eating the gain.
