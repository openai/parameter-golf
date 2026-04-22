# Spec 021c — Recur-α as nn.Parameter(requires_grad=False), 4×H100 NE-1 mini

**Slug:** `recur-alpha-param-frozen-mini`
**Created:** 2026-04-22
**Status:** READY
**Links to:** `research/specs/021-recur-alpha-buffer.md`, `research/specs/021b-recur-alpha-buffer-bf16.md`, `research/evaluations/021-recur-alpha-buffer.md`

## Context

Buffer-α variants on 8×H100 have been tested:
- `cb5cd78` buggy α: post-loop train_loss **+0.010-0.012** above 019b
- `dc0b5f8` α-fix: post-loop train_loss **+0.006-0.011** above 019b
- `d070df3` bf16 buffer: post-loop train_loss **+0.002-0.007** above 019b (closed ~half the gap)

Remaining gap candidate: **nn.Parameter vs register_buffer container semantics.** Inductor may treat Parameter tensors as compile-time-constant-like (recompile on value change) while buffers are treated as mutable runtime inputs. If so, frozen Parameter could enable the same const-folding that 019b's Python-literal α gets.

This spec tests that hypothesis on **4×H100 NE-1** (mini rung) to avoid burning another 8×H trial before we have directional signal. Per the `feedback_small_h_before_full_trial` policy.

## Hypothesis

With α as `nn.Parameter(requires_grad=False)` (bf16 dtype retained), Inductor const-folds the α scalars into the blend kernel at compile time. Post-loop train-loss trajectory tracks 019b within pod-variance tolerance. If confirmed → promote to 8×H official with 3-seed plan.

## Baseline

**Paired matched-H reference:** 019b @ commit `e93d77d` on same 4×H100 NE-1 hardware, same day. Required because:
- 019b's 8×H trajectory does NOT directly compare to 4×H (prior 021-4H showed wildly different per-step loss due to half batch + doubled wallclock + schedule interaction).
- A paired 4×H run of 019b gives us the per-step loss trajectory we're trying to match, on the same hardware.

## Expected Δ

Projection relative to matched-H 019b reference:
- Per-step train_loss: match 019b-4H within ±0.003 post-loop.
- Final step: ~4736 (same as 021-4H buggy — throughput is not expected to change from buffer→Parameter).
- If loss tracks 019b-4H → 8×H official promotion expected to land post-TTT in **1.058-1.064 range** (per spec 021's original projection).

Confidence: moderate. The Inductor-const-fold-Parameter hypothesis is principled but unverified.

## Accept criteria

**Primary (step-matched post-loop train_loss vs 019b-4H reference):**

| 021c-4H vs 019b-4H @ step 3000 | Bucket | Next |
|---|---|---|
| within ±0.003 | Gap closed | **Promote to 8×H official** (new spec 021d) |
| +0.004 to +0.006 | Gap partially closed | Hold — analyze with step 3500/4000 samples, decide |
| > +0.006 | Gap unchanged from 8×H bf16 | Shelve buffer-/Parameter-α arc. Pivot to 019b 3-seed replication |

**Secondary (throughput sanity):**
- Final step ≥ 4730 (match 021-4H buggy)
- Zero Type B spikes in nvsmi / tok/s log

## Config diff

Identical to spec 021 — same env block, same `RECUR_ALPHA_ENABLED=1`, same `MATRIX_LR=0.026`, same `ENABLE_LOOPING_AT` default (0.35), **`MAX_WALLCLOCK_SECONDS=1200`** for 4×H.

Only change is the code.

## Code changes

**Variant arm:**
- **Branch:** `exp/recur-alpha-buffer`
- **Commit:** **`8b2d791`** — `nn.Parameter(requires_grad=False)` replacing `register_buffer`, optimizer guard updated to check `requires_grad`. Builds on `d070df3` (bf16).

**Reference arm:**
- **Branch:** `exp/recur-alpha-manual-constant-full`
- **Commit:** `e93d77d` — 019b's pinned commit, no changes.

## Hardware ladder

**4×H100 NE-1 paired.** Both arms on same-region 4×H100 NE-1 hardware. Run sequentially on same pod if possible to share inductor cache and avoid re-provisioning.

**Do NOT run on 8×H100** — that's the next step if this mini promotes.

## Seed plan

Seed 42 for both arms (matched). Single-seed is sufficient for directional signal on step-matched loss; multi-seed reserved for 8×H official promotion.

## Inputs

- Data: CaseOps dataset, NE-1 `/workspace/data/...`
- Tokenizer: `fineweb_8192_bpe.model` bundled
- Hotstart: none

## Run protocol

**Install brotli on the pod once (non-blocking for this eval-agnostic test, but still good hygiene):**
```bash
pip install --break-system-packages brotli
```

### Arm A — 019b reference on 4×H NE-1

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout e93d77d

mkdir -p /workspace/runs/021c-recur-alpha-param-frozen-mini/019b_ref_4h_ne1
mkdir -p /workspace/.torch_inductor_cache_021c_4h_ne1

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/021c-recur-alpha-param-frozen-mini/019b_ref_4h_ne1 \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache_021c_4h_ne1 \
MAX_WALLCLOCK_SECONDS=1200 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/021c-recur-alpha-param-frozen-mini/019b_ref_4h_ne1/train.log 2>&1
```

Note: `PHASED_TTT_ENABLED=0` — we don't need TTT for a training-loss-curve comparison, and skipping TTT saves ~10 min per arm.

### Arm B — 021c variant on 4×H NE-1 (same pod)

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 8b2d791

mkdir -p /workspace/runs/021c-recur-alpha-param-frozen-mini/021c_variant_4h_ne1

# Verify the change landed
grep "nn.Parameter" train_gpt.py | grep recur_alpha
grep "requires_grad=False" train_gpt.py | head -2

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/021c-recur-alpha-param-frozen-mini/021c_variant_4h_ne1 \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache_021c_4h_ne1 \
MAX_WALLCLOCK_SECONDS=1200 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/021c-recur-alpha-param-frozen-mini/021c_variant_4h_ne1/train.log 2>&1
```

### Arm C — 021 bf16-buffer (register_buffer, not Parameter) on 4×H NE-1 (same pod, runs AFTER Arm B)

**Purpose:** isolate whether the Parameter-vs-buffer switch (Arm B's change) is load-bearing, or whether `register_buffer` + bf16 is already enough to match Arm A. Three-way comparison (A: literal / B: frozen Parameter bf16 / C: buffer bf16) closes the book on which part of the commit stack actually matters.

**Prerequisite:** only launch Arm C after Arm B completes and endpoint metrics are captured. If Arm B had any failure, redo Arm B first before Arm C.

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout d070df3

mkdir -p /workspace/runs/021c-recur-alpha-param-frozen-mini/021c_bufbf16_4h_ne1

# Verify the buffer form (not Parameter) landed
grep "register_buffer" train_gpt.py | grep recur_alpha       # must match
grep "nn.Parameter" train_gpt.py | grep recur_alpha          # must be EMPTY (this commit uses buffer)
grep "dtype=torch.bfloat16" train_gpt.py | head -1           # must match

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/021c-recur-alpha-param-frozen-mini/021c_bufbf16_4h_ne1 \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache_021c_4h_ne1 \
MAX_WALLCLOCK_SECONDS=1200 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/021c-recur-alpha-param-frozen-mini/021c_bufbf16_4h_ne1/train.log 2>&1
```

**Arm C accept criteria (vs Arm A @ step 3000 train_loss):**

| Arm C vs Arm A | Interpretation |
|---|---|
| Within ±0.003 | Container choice doesn't matter at 4H; bf16 alone was enough |
| +0.003 to +0.007 | Parameter specifically closed the gap — strong confirmation for Arm B |
| > +0.007 | Buffer has per-step cost even at 4H; Parameter definitively the right pick |

**Cost:** ~$3 additional on same pod (~22 min training, no TTT/GPTQ). Informational only — does NOT change the 8H Arm B promotion decision, which proceeds regardless.

### Arm A' — 019b ORIGINAL on correct commit (9517a3b) — pairs with Arm E

**Why this exists:** Arm A used commit `e93d77d`, which is the PRE-OOM-fix variant of 019b (uses manual-add blend). The actual 019b-submission that landed post-TTT 1.06628 was on commit `9517a3b` (child of e93d77d), which uses the algebraic-lerp blend form:
```python
x = x_before + alpha * (x_new - x_before)  # 9517a3b
vs
x = alpha * x_new + (1.0 - alpha) * x_before  # e93d77d
```
These are mathematically equivalent but numerically different in bf16. Arm A' gives the correct baseline for comparing Arm E against.

**Commit:** `9517a3b` on `exp/recur-alpha-manual-constant-full` (019b-submission's actual commit)

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 9517a3b

mkdir -p /workspace/runs/021c-recur-alpha-param-frozen-mini/019b_original_algebraic_4h_ne1

# Sanity-verify the algebraic form
grep -c "x_before + alpha \* (x_new - x_before)" train_gpt.py   # should be 4 (all sites algebraic)
grep "_ALPHA_CONSTANTS_017" train_gpt.py | head -1               # should match (literal α)

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/021c-recur-alpha-param-frozen-mini/019b_original_algebraic_4h_ne1 \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache_021c_4h_ne1 \
MAX_WALLCLOCK_SECONDS=1200 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/021c-recur-alpha-param-frozen-mini/019b_original_algebraic_4h_ne1/train.log 2>&1
```

**Purpose:** correct 4×H reference for 019b. Compared to the existing Arm A (`e93d77d`, manual-add), this tells us if the algebraic form itself has any effect at 4×H.

**Cost:** ~$3.

### Arm E — Parameter+bf16+algebraic+TTT-fix on 4×H NE-1 — the candidate for spec 021e

**Commit:** `d761a22` on `exp/recur-alpha-buffer` (already pushed).

**Stack:**
- α as `nn.Parameter(requires_grad=False)` (021c/8b2d791)
- bf16 dtype (021-bf16/d070df3)
- Correct α values (dc0b5f8)
- TTT α fix (931bd7c) — doesn't matter at 4×H since TTT disabled, but part of the commit
- Algebraic blend form (d761a22) ← the key form change

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout d761a22

mkdir -p /workspace/runs/021c-recur-alpha-param-frozen-mini/021e_variant_4h_ne1

# Sanity-verify
grep "nn.Parameter" train_gpt.py | grep recur_alpha            # must match
grep "dtype=torch.bfloat16" train_gpt.py | head -1             # must match
grep -c "x_before + alpha \* (x_new - x_before)" train_gpt.py  # must be 4 (algebraic at all sites)

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/021c-recur-alpha-param-frozen-mini/021e_variant_4h_ne1 \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache_021c_4h_ne1 \
MAX_WALLCLOCK_SECONDS=1200 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/021c-recur-alpha-param-frozen-mini/021e_variant_4h_ne1/train.log 2>&1
```

**Arm E vs Arm A' accept criteria (step 3000 train_loss):**

| Arm E vs Arm A' | Interpretation |
|---|---|
| Within ±0.003 | Parameter + algebraic form matches literal + algebraic — container change is noise. 8H 021e safe to promote. |
| +0.003 to +0.007 | Slight penalty from Parameter container, but 8H may still win via TTT fix. Promote 021e with caution. |
| > +0.007 | Parameter specifically hurts even with algebraic form. Mechanism may not win at 8H. |

**Arm A' vs Arm A (existing) comparison — free scientific bonus:**

If Arm A' tracks Arm A within ±0.003 → the algebraic form doesn't matter at 4×H, and the TTT fix is carrying all of 021e's expected 8H gain.
If Arm A' clearly beats Arm A by +0.003-0.010 → algebraic form is real, and it's part of 021e's story.

**Cost:** ~$3.

### Full mini cost if all arms run

| Arm | Commit | Purpose | Cost |
|---|---|---|---|
| Arm A (existing) | e93d77d | manual-add literal reference | done |
| Arm B (existing) | 8b2d791 | Parameter+bf16+manual-add | done |
| Arm C (queued) | d070df3 | buffer+bf16+manual-add | ~$3 |
| **Arm A'** | **9517a3b** | **algebraic literal (CORRECT 019b reference)** | **~$3** |
| **Arm E** | **d761a22** | **Parameter+bf16+algebraic (021e candidate)** | **~$3** |

Full 5-arm comparison: ~$15 total. All sequential on same NE-1 pod.

## Checkpoints / artifacts

Per arm:
- `train.log` — full training trace, every-100-step log entries
- `final.json` — `val_bpb_pre_gptq_post_ema`, `stopping_early_at_step`, `layer_loop_enabled_at_step`, tok/s snapshots, `recur_alpha_is_parameter` (for 021c arm)

No quantized artifact needed — this is a training-curve-comparison test, not a submission.

## Stop-early criteria

- NaN/inf in train_loss → halt
- **Step 3000 train_loss gap vs reference > +0.006** → halt variant arm early; 8×H promotion is off the table.
- `layer_loop_enabled_at_step` outside [2000, 2300] → halt

## Cost estimate

| item | cost |
|---|---|
| 4×H100 NE-1 Arm A (019b-ref) × ~22 min | ~$3 |
| 4×H100 NE-1 Arm B (021c-variant, Parameter+bf16) × ~22 min | ~$3 |
| 4×H100 NE-1 Arm C (buffer+bf16 isolation) × ~22 min | ~$3 |
| **Three-arm mini total** | **~$9** |
| (Conditional) 8×H official promotion (spec 021d) | ~$10 additional |

## What this does NOT test

- Post-TTT val_bpb — we run with `PHASED_TTT_ENABLED=0` to save ~20 min. TTT behavior only matters if the mini promotes to 8×H.
- Post-quant val_bpb — same rationale.
- Submission-band behavior — 4×H trajectory is not directly submission-comparable (spec 021 4×H note), but the variant-vs-reference A/B is valid on matched hardware.

## Open questions for interview

1. **Is a 4×H100 NE-1 pod currently available?** If not, this blocks. Do NOT substitute 8×H (per user-saved memory `feedback_dont_substitute_expensive_hardware`).
2. **Sequential-same-pod or two separate pods?** Same-pod sequential is cheaper and keeps cache warm. If pod variance is a concern, two pods in parallel adds ~$0 (same total runtime) but consumes two slots. Recommend: same-pod sequential.
3. **If Arm A (019b) shows a surprising 4×H-specific trajectory** (e.g. wildly different from its 8×H run, as 021-4H did), that's a confound — proceed with Arm B but interpret the comparison vs Arm A only, not vs 019b-8H.
4. **Promotion criterion strictness**: 019b-4H is a single-seed reference; some seed variance is expected. Propose: "within ±0.003" on step-3000 loss is the pass bar, with tolerance widened to ±0.004 at steps 4000+.
5. **Halt policy on Arm B if already promoted/rejected by step 3000**: halt and move on; no need to wait for endpoint.

## Decision tree

- **Arm B tracks Arm A** (loss gap within ±0.003): promote to 8×H official. Spec `021d` forthcoming. Projection: post-TTT ~1.058-1.064, likely beats #1736.
- **Arm B partially closes gap** (+0.004-0.006): reference: pivot. Likely remaining mechanism is autograd-graph-level, not readily fixable.
- **Arm B does NOT close gap** (>+0.006): shelve buffer-/Parameter-α arc entirely. Pivot to **019b 3-seed replication** on 8×H (its 1.06628 misses #1736 by only 0.00018, within seed-std).
