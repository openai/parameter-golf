# Spec 014 — BPB-weighted cross-entropy loss (port from #1519)

**Slug:** `bpb-weighted-loss`
**Created:** 2026-04-20
**Links to idea:** `research/ideas/bpb-weighted-loss.md`.

## Hypothesis

Training CE currently weights all tokens equally; the eval metric (bits-per-byte) weights tokens by source UTF-8 byte count. Multi-byte tokens like " the" (4 bytes) contribute 4× to eval but 1× to gradient signal. Reweighting training CE by `byte_weights[targets]` aligns the training objective with the eval metric, reducing the gradient-signal/metric mismatch.

## Baseline

Spec 008's seed-42 val_bpb (`runs/008-1736-reproduction/seed_42/final.json`) = 1.0697 (endpoint bare). Comparison is Δ vs that number.

## Expected Δ

**−0.002 to −0.005 bpb** on top of #1736, if the lever transfers. Author's 1×RTX-5090 result was −0.019 vs a weak baseline; heavy discount on top of #1736 for:

- Smaller base gap (they had more room to improve)
- TTT absorption (our spec 010 finding: post-TTT Δ < pre-TTT Δ)
- CaseOps approximation (we use surface-bytes, author had plain SP1024)

Author's warning: the lever destabilizes training on large vocabs (GPT-2 50K). **SP8192 is our risk zone** — plausibly safe but not guaranteed.

## Accept criteria

- Training completes without NaN / divergence / train_loss blow-up.
- Post-quant post-TTT val_bpb measured (screening mode OK for first pass).
- Artifact < 16 MB (no artifact change; no new params).
- **Decision criterion:**
  - Δ ≤ −0.002 → promote, run 3-seed confirmation.
  - Δ ∈ (−0.002, −0.0005] → promote cautiously; stack candidates.
  - Δ ∈ (−0.0005, +0.001) → null; shelve.
  - Δ > +0.001 → regression; confirm it's not destabilization; either way shelve.

## Config diff vs spec 008

```
BPB_WEIGHTED_LOSS_ENABLED=1
```

No other changes.

## Code changes

- **Branch:** `exp/bpb-weighted` (worktree at `worktrees/bpb-weighted/`).
- **Commit:** `ab6a131`.
- **Patch target:** `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py`.
- **Patch scope:** 44 LOC net (+39/−5). Adds:
  - 1 new `Hyperparameters` field (default off).
  - `GPT.forward` rewritten to check for `_bpb_byte_weights` buffer and weight per-token CE accordingly. Falls through to original `reduction="mean"` path when buffer absent.
  - `train_model` registers the buffer from `val_data.base_bytes_lut` (clamp_min=1) BEFORE `torch.compile` so fullgraph traces it.
  - Startup log echoes enabled/mean/min/max.
- **Default-off invariant:** with `BPB_WEIGHTED_LOSS_ENABLED=0`, no buffer registered → `getattr(self, "_bpb_byte_weights", None) is None` → falls through to the original `F.cross_entropy(..., reduction="mean")` path. Verified byte-identical to spec 008.

## Hardware ladder

- [x] **2×H100 smoke (~5 min, ~$1).** Purpose: catch (a) NaN/divergence from destabilization, (b) compile-time issues with the buffer-guarded path. `ITERATIONS=500 BPB_WEIGHTED_LOSS_ENABLED=1 torchrun --nproc_per_node=2 train_gpt.py`. Pass criterion: 500 steps complete, train_loss decreasing smoothly. **Do NOT skip this smoke** — the destabilization risk is real per #1519's explicit warning.
- [x] **8×H100 full training run, seed 42 (~$20).** Read endpoint bare val_bpb from `screen_endpoint.txt`.

### Early-stop guidance

Same protocol as spec 011/013: executor + user monitor `train.log` via `tail -f`. Compare train_loss vs spec 008 at matched step.

**Kill criteria (any of these = stop pod):**
- NaN or inf in train_loss at any step → automatic kill.
- train_loss *not decreasing* or oscillating wildly in steps 1–500 → destabilization; kill and investigate.
- train_loss > spec 008's + 0.05 for 5+ consecutive late-training log entries → lever is actively hurting, not worth finishing.

**Default if ambiguous:** let it finish. Unlike BigramHash, there's no RNG drift — any endpoint Δ is attributable to the reweighting.

### Pre-registered expectations

Unlike BigramHash (zero-init → late divergence), BPB-weighted CE **modifies the gradient from step 1**. Expected trajectory:

| Step range | Expected behavior |
|---|---|
| **0–300** | train_loss may diverge from spec 008 visibly; direction depends on gradient direction. Early divergence up to 0.01–0.02 nats is OK. |
| **300–1500** | curves settle into parallel trajectories. If BPB-weighted is helping, gap is negative (better). If hurting, gap is positive (worse). |
| **1500–4500** | gap widens slowly; endpoint Δ is extrapolation of mid-trajectory slope. |
| **Post-TTT val_bpb** | Δ typically smaller than train_loss Δ due to TTT absorption. Target: −0.002 to −0.005. |

**Surprising would indicate a bug:**
- train_loss variance >> spec 008 (byte weights are smooth; variance should be similar).
- train_loss decreases much faster than spec 008 in first 100 steps (byte weights mean up-weighting common multi-byte tokens; should be similar learning rate to baseline).
- Step time >> spec 008 (the reweighting is 3 extra ops; no visible compute cost).

## Seed plan

Single seed (42) for screen. 3-seed only if Δ ≤ −0.002.

## Inputs

- **Data:** same CaseOps dataset as spec 008.
- **Tokenizer:** bundled with #1736.
- **Hotstart:** none, full from-scratch training.

## Execution protocol

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

mkdir -p /workspace/runs/014-bpb-weighted-loss/seed_42

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/014-bpb-weighted-loss/seed_42 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
BPB_WEIGHTED_LOSS_ENABLED=1 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/014-bpb-weighted-loss/seed_42/train.log 2>&1
```

Expected startup log:
- `bpb_weighted_loss: enabled=True mean_weight=X.XXX min=1 max=N`
  (mean should be ~3–5 for SP8192; max depends on longest surface piece; min=1 from clamp).

## Checkpoints to emit

- `final_model.pt` (pre-GPTQ FP) — reusable if lever lands.
- Submission artifact + `final.json` (optional in screening mode).

## Stop-early criteria

- NaN in train_loss → halt.
- train_loss oscillating or diverging in first 500 steps → halt.
- Step time > 2× spec 008 → halt.

## Cost estimate

| Item | Cost |
|---|---|
| 2×H100 smoke | ~$1 |
| 8×H100 full screening run | ~$5 (endpoint-only) or ~$20 (with full TTT+eval) |
| **Total** | **~$6–$21** |

## Open questions for interview

1. **Screening mode vs full eval?** Plan: screening mode for first pass (endpoint bare val_bpb, no TTT). Saves ~$15. If the bare Δ is promising, rerun with full eval.
2. **Does the mean byte weight affect effective LR?** The loss scale changes from `mean` to `weighted_mean` which has the same order of magnitude (~3–5× higher absolute value, divided by mean weight). Net LR effect should be ≈ identity, but worth sanity-checking the train_loss value at step 1 is close to spec 008's 9.0180.
3. **Should TTT's LoRA loss also be BPB-weighted?** Plan: NO for this pass. TTT path is untouched. Can revisit if the training-only weighting lands.

## What this spec does NOT do

- Does not touch TTT's loss computation (`forward_ttt` still uses uniform per-token loss).
- Does not use the context-aware `val_bytes` sidecar — uses surface-piece `base_bytes_lut` instead. Approximation is known; acceptable for a screen.
- Does not run 3-seed — single-seed screen only.
- Does not combine with any other spec's lever. Pure 008 + BPB-weighted only.
