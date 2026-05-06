# [Non-Record Submission] SP8192 Stack Ablation: A/B/C/D/E Base Construction, R1–R9 Eval-Time Sweep, OWC Compression-Entropy Analysis, and 12+ Negative Results

## Summary

This is a **non-record submission** documenting a systematic ablation campaign on the SP8192 architecture family, built on top of `#1394` and `#1413`.

**What this PR contains:**

- A 5-factor base-construction ablation (A/B/C/D/E) identifying `D` (parallel residual + loop adjustment) as the canonical base, validated across 5 seeds with mean score-first TTT BPB of **1.08129** (σ = 0.00059)
- A 9-run eval-time and export-time sweep (R1–R9) testing TTT optimizer variants, training modifications, and post-training quantization strategies
- A measured stack-specific finding: **OWC and CDQuant improve raw BPB but create a compression-entropy penalty under Brotli that makes them incompatible with the 16 MB artifact cap** on this stack
- 12+ cleanly measured negative results (RMSDecay TTT, Cautious Muon, freeze-4, attention-only OWC salvage, CDQuant stacking)
- Legality-conscious evaluation design aligned with Issue `#1017` (causal, score-before-update, single-pass; GPTQ calibration uses training data only; n-gram tilt operates on the validation stream using strict-prefix statistics)

**Why this is non-record:**

- The canonical `D` 5-seed mean (1.08129 BPB) is tied with the current merged SOTA (`#1493`: 1.0810) but ~0.007 BPB behind the clean open legal frontier (~1.074 BPB as of 2026-04-13)
- The best single-seed follow-up (`R1_e_baseline`: 1.08079 BPB) ran in 605 seconds, exceeding the 600-second eval time limit — not submission-valid as a lead number
- Multi-seed validation on Pegasus `8×H100` failed across 3 independent submission cycles (container dependencies → checkpoint config mismatch → OOM), producing no submission-valid evidence beyond the RunPod measurements; these failed reruns are operational context only, not a missing primary evidence requirement for this package
- The OWC/CDQuant path that improves raw BPB by ~0.003 exceeds the artifact cap by >1 MB and cannot be salvaged by scope narrowing

**What was learned:**

The primary contribution is the measured finding that post-training weight quantization (OWC/CDQuant) creates a compression-entropy penalty under Brotli that dominates any BPB gain when the artifact must fit under a hard byte cap. This has direct implications for any stack that uses GPTQ + Brotli as its export path.

---

## Rule Compliance Snapshot

- [x] **Non-record framing is explicit.** This package does not claim a new SOTA or a submission-valid lead result.
- [x] **Canonical evidence base is stable.** The primary evidence is the RunPod `D` 5-seed bundle (mean **1.08129**, `σ = 0.00059`), not Pegasus partials and not `R1`.
- [x] **Canonical `D` fits the core hard limits.** All five canonical `D` artifacts are under `16,000,000` bytes, canonical training runs are under `600s`, and canonical TTT eval runs are under `600s`.
- [x] **`R1_e_baseline` is correctly caveated.** It is presented only as the best measured single-seed follow-up and remains explicitly marked as `605s`, over the eval limit.
- [x] **Legality claims stay inside Issue `#1017`.** No pre-quant validation TTT, no custom tokenizer, no multi-pass rescoring, no grouped OWC claim, and no Pegasus result is treated as validation-grade evidence.
- [x] **Submission-style package contents now exist.** The records folder contains `README.md`, `REPORT.md`, `submission.json`, packaged canonical `train_gpt.py`, five canonical `D` train logs, `d_submission_summary.tsv`, `r1_e_baseline.log`, and `r_series_combined_summary.tsv`.
- [x] **Provenance is anchored to the immutable archive.** The package-local `train_gpt.py` is a checksum-verified single-file consolidation derived from archived seed-0 `train_gpt.py` plus the archived helper chain.

This report is grounded in the packaged logs and summaries plus preserved run archives; it is not a fresh end-to-end `8×H100` rerun.

## Audit Checks

- [x] Cross-checked canonical `D` seed metrics against `d_submission_summary.tsv`
- [x] Cross-checked the R-series table against `combined_summary.tsv`
- [x] Verified the package-local `train_gpt.py` matches the archived seed-0 script by SHA256
- [x] Verified package metadata is consistent across this report, `README.md`, `ARTIFACT_MAP.md`, and `submission.json`
- [x] Verified the mutable `ParallelResid7_TiltPrep/train_gpt.py` is no longer used as the canonical measured artifact

## What This PR Claims

- `D` is the canonical, best-supported base from this branch because it is backed by a clean 5-seed RunPod bundle.
- `R1_e_baseline` is a real measured single-seed follow-up signal on top of `D`, but only as a follow-up signal.
- OWC/CDQuant create a fixed-Brotli compression-entropy penalty on this stack that overwhelms their raw BPB gain under the 16 MB cap.
- The negative-results inventory is substantial enough to be useful to other Track A efforts working on nearby stacks.
- The failed Pegasus reruns are documented as operational context, not as a missing validation prerequisite for the package's main claim.

## What This PR Does Not Claim

- It does not claim a record, an "almost-record," or a submission-valid lead result.
- It does not claim multi-seed confirmation for `R1_e_baseline`.
- It does not claim Pegasus produced completion-valid validation evidence.
- It does not claim grouped OWC, CDQuant salvage, APM, or any other follow-up idea is already demonstrated on this stack.
- It does not treat the missing Pegasus reruns as a gap in the main evidence line, because the package's primary evidence is the RunPod `D` 5-seed bundle.

---

## 1. Base Construction: A/B/C/D/E Ablation

### Design

Five configurations were tested on RunPod `8×H100 SXM`, all sharing the SP8192 architecture from `#1413` with `QK_GAIN_INIT=5.0` and legal score-first TTT:

| Run | Configuration | Key Env Overrides |
|-----|--------------|-------------------|
| **A** | Faithful `#1413` mirror | none |
| **B** | Parallel residual from layer 7 | `PARALLEL_RESIDUAL_START=7` |
| **C** | Loop adjustment (layers 3–5) | `LOOP_START=3 LOOP_END=5` |
| **D** | B + C combined | `PARALLEL_RESIDUAL_START=7 LOOP_START=3 LOOP_END=5` |
| **E** | D + n-gram tilt (eval-only) | D env + `SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1` |

### Why `D` became the canonical base

`D` combines the two strongest independent modifications (parallel residual start and loop adjustment) and produced the best training-side quality across seeds. The combination is additive — neither B nor C alone matches D's quality.

Run `E` is eval-only: it reuses D's trained checkpoint and applies an n-gram tilt layer at evaluation time. It represents the best measured single-seed eval-time enhancement on top of the D base, subject to the 605s wallclock caveat documented below.

### Canonical D results (5 seeds)

| Seed | Sliding s64 BPB | Score-First TTT BPB | Artifact Bytes |
|------|-----------------|---------------------|----------------|
| 0 | 1.08261 | 1.08093 | 15,992,638 |
| 42 | 1.08401 | 1.08114 | 15,990,501 |
| 1234 | 1.08248 | 1.08092 | 15,990,023 |
| 1337 | 1.08259 | 1.08112 | 15,989,185 |
| 2025 | 1.08379 | 1.08233 | 15,989,883 |

**5-seed mean TTT BPB: 1.08129** (σ = 0.00059)
**Max artifact: 15,992,638 bytes** (under 16,000,000 cap)

Additional sixth seed (7): TTT BPB 1.08168, 15,994,511 bytes. All-6 mean: 1.08135.

### R1 as best single-seed follow-up

`R1_e_baseline` applied the `E`-style eval (n-gram tilt, `SKIP_TRAINING=1`) on the D seed-0 checkpoint:

- **BPB: 1.08079** (best measured single-seed BPB from the entire campaign; wallclock caveat below)
- **Wall time: 605 seconds** (exceeds 600s eval limit — not submission-valid as a clean lead number)
- Artifact bytes: reuses D seed-0 artifact (15,992,638)

This establishes the eval-time tilt as a real signal (+0.00014 over D seed-0 TTT), but the wallclock overshoot means it cannot be presented as a clean submission number without freeze-4 or similar latency reduction.

---

## 2. What Changed During This Campaign

This campaign improved because it updated its own assumptions, not because it stacked more tricks. Three corrections reshaped the experimental plan midstream:

**Corrected parameter count.** Early analysis assumed the SP8192 model was ~5M parameters. The actual count is ~47.5M. This was not a rounding error — it changed how the literature on mixed-precision quantization applied to this architecture. At 47.5M parameters, the model sits near the crossover where int6 quantization stops being conservative and starts being constrained. The literature discount for quantization-aware BPB recovery dropped from ~3× to ~1.5–2×, which ruled out several proposed techniques.

**Corrected eval-time budget.** The R1 n-gram tilt run hit 605 seconds — over the 600-second eval time limit. The campaign had assumed eval compute was underutilized. In fact, the legal TTT pipeline (score-before-update, single-pass, chunk-wise cosine LR decay) already saturates the time budget. This killed all proposals that would add eval-time compute (APM cascade, multi-pass refinement) without first freeing budget through freeze-4 or equivalent latency reduction.

**Corrected legality description.** The n-gram tilt was initially described as using "training-data-only calibration." Audit revealed the hints and betas are precomputed from the validation stream (causally, from the strict prefix at each position). This is consistent with Issue `#1017` but requires precise language. The correction forced all legality claims in this PR to be grounded in the actual code path rather than inherited assumptions.

These corrections explain why the branch stopped where it did. The experimental plan was designed to be updated by evidence, and the evidence said to stop.

---

## 3. R-Series Eval-Time and Export-Time Sweep

Nine experiments (R1–R9) were run on the D seed-0 checkpoint to test eval-time TTT modifications, training variants, and post-training quantization strategies. All runs used RunPod `8×H100 SXM`.

### Results summary

| Run | Category | BPB | Bytes | Wall (s) | Status | Interpretation |
|-----|----------|-----|-------|----------|--------|----------------|
| **R1** `e_baseline` | eval-time tilt | **1.08079** | (D ckpt) | 605 | over 600s limit | best measured BPB; not submission-valid due to 605s wallclock |
| **R2** `e_rmsdecay_low` | TTT optimizer | 1.59781 | (D ckpt) | 493 | legal | catastrophic; RMSDecay decay=0.001 |
| **R3** `e_rmsdecay_high` | TTT optimizer | 1.47583 | (D ckpt) | 493 | legal | catastrophic; RMSDecay decay=0.005 |
| **R4** `e_freeze4` | eval-time TTT | 1.08093 | (D ckpt) | 478 | legal | neutral vs R1 (Δ = +0.00014) |
| **R5** `e_combo` | TTT optimizer | 1.47494 | (D ckpt) | 495 | legal | catastrophic; RMSDecay + freeze combo |
| **R6** `d_cautious_muon` | training variant | 1.08170 | 15,993,951 | 484¹ | legal | negative vs D baseline (Δ = +0.00077) |
| **R7** `d_owc` | export quant | **1.07832** | **17,166,438** | 480¹ | **over cap** | best raw BPB; +1.17 MB over cap |
| **R8** `d_cdquant_owc` | export quant | 1.07840 | 17,156,305 | 474¹ | **over cap** | no gain over R7 alone |
| **R9** `d_full_stack` | combined | 1.07916 | 17,202,081 | 476¹ | **over cap** | worse than R7; stacking is negative |

¹ Wall time shown is for the eval pass only; training variants (R6, R7, R8, R9) also ran ~1200s training passes.

Additionally, a salvage attempt was run:

| Run | Category | BPB | Bytes | Status | Interpretation |
|-----|----------|-----|-------|--------|----------------|
| `requant_export_s0` | attention-only OWC | 1.08158 | 16,275,138 | **over cap** | scope narrowing fails: still over cap, worse BPB than R1 |

### Key findings from the R-series

1. **RMSDecay TTT is catastrophically bad** on this stack. All three variants (R2, R3, R5) degraded BPB by 0.39–0.52. This optimizer is incompatible with the SP8192 TTT configuration.

2. **Freeze-4 is neutral.** Freezing 4 of 8 TTT blocks (R4) produced BPB within 0.00014 of R1 while reducing wall time from 605s to 478s. This is a viable latency lever but does not improve quality.

3. **Cautious Muon training is negative.** R6 degraded BPB by 0.00077 relative to D baseline — a small but consistent penalty.

4. **OWC produces the best raw BPB but is illegal under the byte cap.** R7 achieves 1.07832, a 0.00247 improvement over R1 — but the artifact is 17.17 MB, exceeding the 16 MB cap by 1.17 MB.

5. **CDQuant adds nothing on top of OWC.** R8 (CDQuant + OWC) is within 0.00008 of R7 alone, suggesting the two techniques target the same quantization error surface on this architecture.

6. **Full stacking is negative.** R9 (Cautious Muon + OWC + CDQuant) is worse than R7 alone by 0.00084 BPB and produces an even larger artifact.

7. **Scope narrowing does not salvage OWC.** The attention-only requant attempt (`requant_export_s0`) still exceeds the cap by 275 KB and produces worse BPB than R1.

---

## 4. Rate-Distortion Under Fixed Brotli

### The finding

The competition fixes the compressor (Brotli quality 11) and the artifact cap (16,000,000 bytes). This means the real optimization objective is not raw quantization quality — it is **BPB per compressed byte**. Better quantization can be strictly worse once Brotli is fixed and the artifact is already sitting at ~15.99 MB with only ~7 KB of headroom.

OWC (Optimal Weight Clipping) and CDQuant (Cross-Domain Quantization) demonstrate this directly: they improve raw quantization quality but **increase the entropy of the quantized weight representation**, making the compressed artifact too large to submit.

### Measured evidence

| Configuration | TTT BPB | Compressed Bytes | Δ BPB vs D | Δ Bytes vs D |
|--------------|---------|-----------------|------------|--------------|
| D baseline (no OWC) | 1.08093 | 15,992,638 | — | — |
| R7 (OWC) | 1.07832 | 17,166,438 | −0.00261 | **+1,173,800** |
| R8 (CDQuant + OWC) | 1.07840 | 17,156,305 | −0.00253 | **+1,163,667** |
| requant attn-only | 1.08158 | 16,275,138 | +0.00065 | **+282,500** |

### Mechanism

Standard GPTQ with fixed `clip_sigmas` produces weight distributions that are highly compressible — the clipping creates strong statistical regularity that Brotli exploits. OWC optimizes the clipping range per-tensor to minimize quantization error, which is the right objective for BPB but scatters the weight values more uniformly across the quantized range, destroying the compressibility structure.

The tradeoff is:
- **OWC wins on raw quantization quality** (lower reconstruction error → lower BPB)
- **OWC loses on compressed artifact size** (higher entropy → more bytes under Brotli)

Under a hard byte cap like the challenge's 16,000,000 bytes, the compression penalty dominates. The D baseline uses ~15.99 MB, leaving only ~7 KB of headroom. OWC adds ~1.17 MB. No realistic scope narrowing can bridge this gap.

### Implication

The competition cannot use custom entropy coders (ANS, arithmetic coding) — the compressor is fixed. This means classical rate-distortion theory provides guidance but not drop-in solutions. The practical constraint is: **any quantization strategy that scatters weight values more uniformly across the quantized range will hurt Brotli even if it helps reconstruction error.**

For stacks near the byte cap, the correct strategy is to use fixed, tuned `clip_sigmas` that preserve the statistical regularity Brotli exploits — or to find enough byte headroom elsewhere (e.g., through depth-sharing or architecture changes) to afford the entropy cost. On this stack, with ~7 KB headroom vs. +1.17 MB OWC penalty, there is no realistic scope narrowing that bridges the gap.

---

## 5. Negative Results Inventory

The following modifications were measured and found to be negative or neutral on this stack. All measurements are on RunPod `8×H100 SXM` using the D seed-0 checkpoint unless otherwise noted.

### Training-side negatives

| Modification | Result | Notes |
|-------------|--------|-------|
| **Cautious Muon** (`CAUTIOUS_MUON=1`) | +0.00077 BPB vs D | R6; consistent penalty across train+eval |
| **RMSDecay TTT** (decay=0.001) | +0.518 BPB vs D | R2; catastrophic degradation |
| **RMSDecay TTT** (decay=0.005) | +0.395 BPB vs D | R3; still catastrophic |
| **RMSDecay + freeze combo** | +0.394 BPB vs D | R5; freeze-4 does not rescue RMSDecay |

### Eval-time negatives/neutrals

| Modification | Result | Notes |
|-------------|--------|-------|
| **Freeze-4 TTT** (`TTT_FREEZE_BLOCKS=4`) | +0.00014 BPB vs R1 | R4; neutral quality, 21% faster (478s vs 605s) |

### Export-side negatives

| Modification | Result | Notes |
|-------------|--------|-------|
| **OWC full-scope** | −0.00261 BPB but +1.17 MB | R7; quality wins, size loses; illegal |
| **CDQuant + OWC** | −0.00253 BPB but +1.16 MB | R8; no gain over OWC alone |
| **Full stack** (Cautious Muon + OWC + CDQuant) | −0.00177 BPB but +1.21 MB | R9; worse than OWC alone |
| **Attention-only OWC requant** | +0.00065 BPB and +0.28 MB | Scope narrowing fails on both axes |

### Earlier campaign negatives (pre-R-series)

| Modification | Stack | Result | Notes |
|-------------|-------|--------|-------|
| **QK_GAIN=5.0** | 07c1 | negative | Falsified on Pegasus `8×H100` |
| **MLP_MULT=3.08** | 07c1 | negative | Falsified on Pegasus `8×H100` |
| **MLP_MULT=3.5** | 07c1 | quality-positive but over cap | 17.18 MB artifact |
| **GPTQ percentile clip search** | 05c-plus | negative | Destroyed zstd compressibility; artifact exceeded 16 MB |
| **Full Hessian GPTQ** | 05c-plus | negative | 7 ablations, all worse than legacy row-max |
| **LeakyReLU² + GPTQ** | 05c-plus | neutral | Activation change not root cause of GPTQ failure |
| **FA3 on NGC 25.02** | 07c1 | negative | 11.44× kernel speedup negated by pip torch downgrade |
| **SWA (Stochastic Weight Averaging)** | 05c-plus | dead code | Collected but never applied in #1019 / #634; use EMA only |

---

## 6. Why This Branch Stops Here

This branch became an evidence branch rather than a frontier branch because three independent constraints converged:

1. **Eval budget is saturated.** R1 ran in 605 seconds, 5 seconds over the 600-second limit. The legal TTT pipeline — score-before-update, chunk-wise cosine LR decay, 8-GPU distributed scoring — leaves essentially no headroom for additional eval-time computation. Freeze-4 (R4) saved 127 seconds by eliminating backward passes through 4 of 8 TTT blocks, but at near-zero quality gain (Δ = +0.00014 BPB). This is a clean systems result: the eval time budget is the binding constraint on eval-time improvements, not the quality of the underlying TTT strategy.

2. **Export budget is saturated.** The D baseline produces artifacts at ~15.99 MB, leaving ~7 KB of headroom under the 16 MB cap. OWC wins 0.003 BPB in raw quality but adds 1.17 MB. Attention-only scope narrowing still exceeds the cap by 275 KB. There is no realistic export-side improvement that fits in the remaining byte budget without architectural changes that free headroom first.

3. **Training quality is competitive but not frontier.** The canonical D 5-seed mean (1.08129 BPB) is tied with the merged SOTA (`#1493`: 1.0810) but 0.007 BPB behind the clean open frontier (~1.074). Closing that gap requires architectural or training-side changes, not more export tuning on the same checkpoint.

The combination means further work on this specific checkpoint and export path has diminishing returns. The measured evidence is complete enough to publish.

---

## 7. Directions Explicitly Ruled Out

The campaign explicitly evaluated and rejected several directions. These are documented kills, not unexplored options:

| Direction | Kill Reason | Evidence |
|-----------|------------|---------|
| **Pre-quant validation TTT** | Legality-sensitive; flagged in `#1517`/`#1550`; no maintainer ruling | Issue `#1017` |
| **Casefold tokenizer** | Legality-sensitive; actively disputed in `#1578`/`#1585`; README flags custom tokenizers for scrutiny | Challenge README |
| **Pruning / Lottery Ticket** | Wrong match for Brotli-capped export; zeroed weights don't compress well under Brotli's LZ77 | Literature review |
| **RFN / attribution graph** | All five kill criteria met: no clean transformer bridge, closest work is by other researchers, time-to-result exceeds 1 week, expected BPB improvement < 0.001, cannot be honestly framed as thesis-derived | EV analysis |
| **Architecture switch** | Not a near-term, challenge-shaped path; would require full retraining and new baseline | Campaign scope |
| **OWC salvage (any scope)** | Irreconcilable size penalty under Brotli; 4 variants tested, all over cap | R7, R8, R9, requant |
| **RMSDecay TTT optimizer** | Catastrophic degradation: +0.39 to +0.52 BPB across 3 variants | R2, R3, R5 |

These decisions were made before the evidence was complete and updated as measurements came in. The campaign treated each direction as a hypothesis with a pre-defined kill criterion, not as an open-ended exploration.

---

## 8. Legality and Evaluation Design

All evaluation in this campaign follows the causal, score-before-update TTT protocol aligned with Issue `#1017`:

- **Causal scoring:** Each token is scored using only preceding context. The TTT update for position `t` is applied *after* scoring position `t`.
- **Single-pass evaluation:** No multi-pass or iterative refinement on the validation data.
- **Training-data-only GPTQ calibration:** GPTQ Hessian collection uses training-split data only. No validation or eval tokens are consumed before quantization.
- **No pre-quant validation TTT:** The model is quantized directly from training checkpoints. TTT runs on the quantized model at eval time.
- **No custom tokenizers:** Standard SentencePiece tokenizer from the challenge repository. No casefold, normalization, or vocabulary modifications.

The n-gram tilt layer (used in R1 and freeze-4) operates on the **validation stream** using strict-prefix statistics only: for each position `p`, the n-gram hash tables contain counts accumulated from `val_tokens[0..p-1]` only, the hint is looked up, scoring occurs, and only then is `val_tokens[p]` added to the tables. This is causal and single-pass, but the data source is the validation stream itself (not training data). The tilt adjusts per-token NLL using an exponential-family mixture of n-gram expert predictions, applied after the base model scores each position.

**Caveat on "token-only" framing:** The default configuration sets `NGRAM_WITHIN_BETA=0.0` and `NGRAM_WORD_BETA=0.0`, which zeroes the beta weight for within-word and word-boundary experts. However, those experts still emit candidates, and any matching hint receives `agree_bonus` even if the originating expert had beta zero. This means the tilt is not purely single-expert even at zero beta — the agree-bonus pathway can still be influenced by experts whose direct contribution is zeroed. We do not call this configuration "token-only" without this qualification.

This PR does not rely on any technique whose legality is currently disputed (pre-quant validation TTT per `#1517`/`#1550`, casefold tokenizer per `#1578`/`#1585`).

---

## 9. Pegasus Multi-Seed Validation Attempts

Three independent attempts to validate `R1_e_baseline` on Pegasus `8×H100` Slurm cluster all failed, each with a distinct root cause.

### Cycle 1: Container dependency failure
- **Jobs:** `2771724`, `2771725`
- **Failure:** Stock NGC PyTorch 26.03 container missing `flash_attn_interface`, `sentencepiece`, and `brotli`
- **Resolution:** Cancelled; switched to saved FA3 container at `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`

### Cycle 2: Checkpoint config mismatch
- **Jobs:** `2771763`, `2771764`
- **Failure:** `skip_weights` shape `[8, 512]` vs model `[7, 512]`; `skip_gates` same mismatch
- **Root cause:** Eval wrapper reused `pr1413_combo` checkpoints without replaying the archived D env (`QK_GAIN_INIT=5.0`, `LOOP_START=3`, `LOOP_END=5`, `PARALLEL_RESIDUAL_START=7`)
- **Resolution:** Added `run_meta.env` staging and sbatch-time env import

### Cycle 3: Out of memory
- **Jobs:** `2773187`, `2773188`
- **Failure:** Slurm `OUT_OF_MEMORY` after ~3h45m runtime at `--mem=64G`
- **Partial results before death:** seed 42 sliding_window `1.08402`, seed 1337 sliding_window `1.08259`
- **Root cause:** N-gram tilt state allocates ~120 MB/rank (`hints_cpu` + `betas_cpu`); combined with model load exceeded 64G Slurm allocation
- **Resolution:** Would require `--mem=128G` or higher; not resubmitted

### Status

No submission-valid Pegasus evidence was produced. The partial cycle-3 results (seed 42: 1.08402, seed 1337: 1.08259 sliding window) are consistent with the RunPod 5-seed D bundle but are not completion-valid because neither job finished the legal TTT pass.

The RunPod 5-seed D measurement (mean 1.08129, σ = 0.00059) remains the primary evidence base.

These Pegasus failures are reported as operational context only. They do not create a missing primary-evidence gap for this package because the package's claim is anchored to the completed RunPod `D` 5-seed bundle, while `R1_e_baseline` is already presented only as an over-limit single-seed follow-up signal.

---

## 10. Contribution Summary

This PR contributes:

1. **Systematic ablation study:** 5 base configurations × 5+ seeds, plus 9 eval/export variants — all on the same SP8192 architecture family with controlled single-variable changes.

2. **OWC compression-entropy finding:** Measured evidence that optimal per-tensor weight clipping trades Brotli compressibility for quantization quality, creating an irreconcilable size penalty under hard byte caps. This is a measured GPTQ+Brotli rate-distortion finding on this stack, with likely relevance to nearby export pipelines; it is not a new quantization method.

3. **Negative results inventory:** 12+ cleanly measured negatives covering TTT optimizers (RMSDecay), training modifications (Cautious Muon), eval-time strategies (freeze-4), and export techniques (OWC, CDQuant, attention-only requant, GPTQ clip search, Full Hessian GPTQ).

4. **Legality-aligned evaluation:** All measurements follow the causal score-before-update protocol from Issue `#1017`, with no reliance on disputed techniques (pre-quant validation TTT, casefold tokenizers).

5. **Operational lessons:** Three distinct Pegasus failure modes documented (container deps, checkpoint config replay, memory allocation for n-gram tilt), useful for anyone running similar eval-time augmentation on Slurm clusters.

---

## 11. Position Relative To The Frontier

The public frontier as of 2026-04-13 is increasingly split between two fundamentally different approaches:

- **Track A (Neural Optimization):** Optimized transformers with systems-level tuning — fused kernels, improved parallel residuals, attention variants, better TTT. Merged SOTA is `#1493` at 1.0810 BPB. Clean open frontier reaches ~1.074 BPB (`#1518`, `#1560`). This track improves the neural model itself.

- **Track B (Bayesian Compression):** Posterior mixing, Dirichlet priors, n-gram backoff hierarchies. These treat the neural LM as one component of a larger statistical system. Unverified claims reach below 1.02 BPB. The gap between Track A and Track B is not explained by architecture differences — it is explained by the scoring paradigm itself.

This PR is a **Track A contribution**: a clean, auditable transformer evidence package built on the SP8192 architecture family. It does not attempt to compete with Track B approaches. Its value lies in the measured ablation sweep, the rate-distortion finding under fixed Brotli, and the negative-results inventory — all of which are directly useful to other Track A efforts.

The canonical D 5-seed mean (1.08129 BPB) is tied with the merged SOTA. It is 0.007 BPB behind the clean open frontier. This gap is real but not large enough to dismiss the evidence as irrelevant — the ablation structure and export-path findings apply to any stack within ~0.01 BPB of this baseline.

---

## 12. Next Clean Hypothesis

The only remaining high-upside eval-time extension on this architecture family is **APM (Adaptive Posterior Mixing)** — logistic-domain mixing of the base model's predictions with lightweight online estimators. APM is structurally different from the n-gram tilt tested in this campaign: it operates in logistic space rather than NLL space and uses a proper posterior update rather than a fixed-beta exponential mixture.

The prerequisite is restored eval budget: R1 already hits 605 seconds, so any eval-time addition requires freeze-4 (saves 127 seconds) or equivalent latency reduction first. This is a separate branch hypothesis, not a continuation of the D/R1 evidence line.

---

## 13. Reproducibility

### Evidence bundles

This records folder packages the review-critical evidence directly:

| Artifact | Location |
|----------|----------|
| D 5-seed canonical summary | `d_submission_summary.tsv` |
| D canonical train logs | `train_seed0.log`, `train_seed42.log`, `train_seed1234.log`, `train_seed1337.log`, `train_seed2025.log` |
| R-series summary | `r_series_combined_summary.tsv` |
| Best measured single-seed follow-up log | `r1_e_baseline.log` |
| Packaged canonical script | `train_gpt.py` |
| Minimal runtime manifest | `requirements.txt` |

### Scripts

| Script | Purpose |
|--------|---------|
| `train_gpt.py` | Package-local single-file D training + eval script (24,711 code bytes; consolidates archived seed-0 `train_gpt.py`, `ngram_tilt.py`, and `fused_expert_kernel.cpp` into one counted code file; packaged SHA256 `4f2ab2ca43105e94ea1b09924a7580a5446c72be47c2ff1d580c9c604fba69dd`) |

### External provenance references

These archive paths were used to build and verify the packaged evidence:

- canonical `D` archive root:
  `artifacts/runpod_pull/pr1413_archive_20260407_213205/`
- R-series archive root:
  `artifacts/runpod_pull/runpod_r_experiments_20260409_182045/`
- package-local single-file script sources:
  - `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/train_gpt.py`
  - `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/ngram_tilt.py`
  - `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/fused_expert_kernel.cpp`

### Hardware

All foreground measurements: RunPod `8×H100 SXM`, 80 GB per GPU.
Pegasus attempts: DFKI Pegasus cluster, `8×H100 SXM` via Slurm.

### Canonical result for this PR

**D 5-seed mean score-first TTT BPB: 1.08129** (σ = 0.00059, max artifact 15,992,638 bytes, all under 16 MB cap).
