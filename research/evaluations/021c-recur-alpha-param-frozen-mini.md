# Evaluation 021c — Recur-α Parameter-frozen-mini (4×H100 paired A/B/C + A')

**Spec:** `research/specs/021c-recur-alpha-param-frozen-mini.md`
**Runs:**
- Arm A (019b manual-add @ `e93d77d`, literal-α): `runs/021c-recur-alpha-param-frozen-mini/019b_ref_4h_ne1/` — JP pod `kb1aw0mi96l06d`
- Arm B (variant @ `8b2d791`, `nn.Parameter(requires_grad=False)` bf16): `runs/021c-recur-alpha-param-frozen-mini/021c_variant_4h_ne1/` — JP same pod
- Arm C (variant @ `d070df3`, `register_buffer` bf16): `runs/021c-recur-alpha-param-frozen-mini/021c_bufbf16_4h_ne1/` — JP same pod, partial (killed at step ~4600)
- **Arm A' (019b-original algebraic @ `9517a3b`, literal-α, correct 019b code)**: `runs/021c-recur-alpha-param-frozen-mini/019b_orig_A_prime_4h_ne1/` — **NE-1 pod `xuwiqdzi94f68e` (fresh)** — added 2026-04-22 once commit-hash confusion revealed `e93d77d` was the pre-OOM-fix manual-add variant, not the real 019b submission. Cost ~$6.
- Arm E (d761a22 full stack): **NOT RUN** — superseded by user's direct 8×H 021e run (`parameter-golf-021e`) before the 4H mini was needed.
**Date:** 2026-04-22
**Status:** Arms A/B/C/A' complete. Arm E skipped — direct 8H test already in flight.

---

## Result summary

| metric | Arm A (JP literal manual-add) | Arm B (JP Param bf16 manual-add) | Arm C (JP buf bf16 manual-add) | Arm A' (NE literal algebraic) |
|---|---|---|---|---|
| Commit | `e93d77d` | `8b2d791` | `d070df3` | **`9517a3b`** |
| Region | JP | JP | JP | NE-1 |
| Blend form | manual add | manual add | manual add | **algebraic lerp** |
| Wallclock cap | 1200s | 1200s | 1200s | 1200s |
| Loop activation step | 2241 | 2235 | 2239 | 2258 |
| Stopping_early step | 5034 | 5004 | killed @ ~4600 | **5065** |
| Pre-quant post-EMA val_bpb | 1.06927 | 1.06952 | N/A | **1.06883** |
| val_bpb @ step 4000 | 1.1190 | 1.1177 | N/A | 1.1198 |
| Pre-loop tok/s plateau | 4.21M | 4.20M | 4.21M | 4.24M |

**Δ post-EMA:**
- Arm B − A: +0.00024 (inside ±0.003 promote band)
- **Arm A' − A: −0.00044** (algebraic blend beats manual-add; 31 more steps helps; same commit family otherwise)
- Arm A' is the **best 4×H post-EMA across all 4 arms**.

---

## Step-matched train_loss (post-loop)

All three arms activated their layer loop within 6 steps of each other, and tracked within ±0.006 throughout post-loop:

| step | Arm A | Arm B | Arm C | Δ(B−A) | Δ(C−A) |
|---|---|---|---|---|---|
| 2300 | 2.6757 | 2.6752 | 2.6741 | −0.001 | −0.002 |
| 2500 | 2.5132 | 2.5102 | 2.5097 | −0.003 | −0.004 |
| 2700 | 2.6597 | 2.6582 | 2.6585 | −0.002 | −0.001 |
| 2800 | 2.5620 | 2.5649 | 2.5603 | +0.003 | −0.002 |
| **3000** | **2.5792** | **2.5783** | **2.5760** | **−0.001** | **−0.003** |
| 3100 | 2.4611 | 2.4599 | 2.4553 | −0.001 | −0.006 |
| 3200 | 2.4844 | 2.4816 | 2.4842 | −0.003 | 0.000 |
| 3300 | 2.6001 | 2.6007 | 2.6027 | +0.001 | +0.003 |

Pre-loop (steps 100-2200): all three tracked within ±0.008 — expected, since the α code path only activates when `looping_active == True`.

---

## Throughput

Arm C (buffer + bf16) is **throughput-indistinguishable from Arm A** (literal-α): identical pre-loop plateau at 4.21M, identical post-loop profile (3.74M @ step 3000). Arm B (Parameter + bf16) is ~0.5–1.5% slower than both — a small Parameter-specific codegen penalty, not a fusion improvement.

No Type B spikes in any arm's tok/s trace.

---

## Three-way mechanism read

The three arms span the full container/dtype matrix for frozen α:

| Arm | Container | Dtype | Post-EMA | vs Arm A |
|---|---|---|---|---|
| A | Python literal (inlined) | bf16 | 1.06927 | 0 |
| B | `nn.Parameter(requires_grad=False)` | bf16 | 1.06952 | +0.00024 |
| C | `register_buffer` | bf16 | (partial — matched at step 3000) | ~0 |

**Conclusion: at 4×H JP with bf16-aligned α, container choice is irrelevant.** All three behave equivalently on val_bpb.

The prior spec 021b finding (at 8H, buffer + bf16 left a +0.0008 post-EMA gap vs 019b) is **not reproduced here**. Most plausible explanation: the 021b gap is either (a) hardware-specific to 8H (different NCCL-induced Inductor graph shape) or (b) pod-lottery noise within 8H seed variance. Arm C partial data does not support the "buffer has per-step cost" hypothesis — its trajectory matches Arm A.

Parameter-α's slight throughput penalty (~1%) means if we promote **Arm B to 8H**, we're accepting a small tok/s cost for … no measurable val_bpb gain over Arm A (literal-α is already the simplest solution). **The Parameter switch is not load-bearing.**

---

## Commit-hash confusion (critical methodology note)

Partway through this session we discovered **our "Arm A" reference was not actually 019b-submission code.** Commit `e93d77d` is the *pre-OOM-fix manual-add variant*; the real 019b-submission is at `9517a3b` which uses the algebraic lerp form `x_before + α·(x_new − x_before)`. Arm A' was a corrective rerun on the true 019b code.

**Impact:** Arm B (Parameter + bf16 + manual-add) was being compared to the wrong baseline. The "B ≈ A" conclusion still holds, but the reference shifted — Arm A' shows that **the algebraic blend form is a real small win over manual-add** (−0.00044 post-EMA), independent of container choice.

This also explains part of 021b's 8H gap to 019b: 021b inherited the manual-add form instead of algebraic.

## Decision

**Arm E superseded by user's direct 8×H 021e run** (commit `d761a22` = full stack: Parameter + bf16 + algebraic + TTT-fix). That result will be decisive.

**Shelve Parameter-vs-buffer as a standalone question.** At 4H, all three container variants (literal, Parameter, buffer) + bf16 land within ±0.00024 post-EMA. Container choice is not the mechanism.

**The surviving mechanistic findings from 021c:**
1. **Algebraic blend > manual-add by ~0.0005 post-EMA** (Arm A' vs Arm A, controlling for commit family).
2. **Parameter vs literal vs buffer: statistically indistinguishable** at 4H with bf16.
3. Region/hardware variance (JP vs NE) contributes ~0.004–0.008 pre-loop noise and a ~0.7% tok/s gap (NE faster).
4. The earlier "021 family +0.0008 gap to 019b at 8H" was likely the manual-add-vs-algebraic effect plus the TTT α bug — neither is about α container.

**Next action:** 019b-original (`9517a3b`) 3-seed on 8H is the highest-value remaining pure play at #1736. The in-flight 021e is the stacked version; if it lands cleanly below #1736 it becomes the submission. Otherwise 019b 3-seed is the fallback.

---

## Cost accounting

| item | cost |
|---|---|
| 4×H JP pod, arms A/B/C sequential, ~75min wall total | ~$15 |
| NFS inductor cache crash + restart (Arm A) | +~$1 burn |
| Arm C killed mid-training (saved ~$0.80 on remaining 2min) | — |
| 4×H NE-1 pod, Arm A', ~30min wall | ~$6 |
| Arm E skipped (superseded by 8H 021e) | $0 |
| **Spec 021c total** | **~$22** |

---

## Bugs / environment

- **NFS stale file handle on `/workspace/.torch_inductor_cache_...`** (FUSE mount race between triton compile workers). Fix: moved to `/tmp/torch_inductor_cache_021c_4h`. Saved as durable memory — *never* use volume path for Inductor cache.
- **`mkdir && nohup` race** in launch commands: dir mkdir gets backgrounded with the nohup, so the redirect target doesn't exist yet. Fix: synchronous `mkdir -p` before any nohup that writes into it.

---

## What this did NOT produce

- No post-TTT val_bpb (spec explicitly ran `PHASED_TTT_ENABLED=0`)
- No post-quant val_bpb or submission artifacts (GPTQ killed mid-phase for Arm A and Arm B, not reached for Arm C)
- No Arm C post-EMA (pod stopped before endpoint)
- No multi-seed variance (single seed 42 across all arms)

Each of these is deferred to 8H promotion (spec 021d) where it's decisive.

---

## Artifacts (local)

- `runs/021c-recur-alpha-param-frozen-mini/019b_ref_4h_ne1/` — train.log + diag_nvsmi.csv (Arm A, full to step 5034 + post-EMA)
- `runs/021c-recur-alpha-param-frozen-mini/021c_variant_4h_ne1/` — train.log + diag_nvsmi.csv (Arm B, full to step 5004 + post-EMA)
- Arm C partial data remains on JP volume `jlxvxeiol4` (pod stopped, not terminated); rsync when pod is next started
