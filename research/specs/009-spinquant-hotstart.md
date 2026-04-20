# Spec 009 — SpinQuant V1 hotstart on top of #1736

**Slug:** `spinquant-hotstart`
**Created:** 2026-04-20
**Links to idea:** `research/ideas/1736-improvement.md` (spec-009 section)
**Depends on:** spec 008 complete, `runs/008-1736-reproduction/seed_42/pre_gptq.pt` present.

## Hypothesis

Hadamard rotation of weight matrices before GPTQ quantization (SpinQuant V1) spreads weight-distribution outliers uniformly across input dimensions. This reduces quantization error at fixed bit-width, improving post-quant val_bpb without touching the float-precision forward pass. Witnessed on PR #1695 (X-Abhishek-X) at claimed −0.005 bpb on top of a #1529-adjacent base; expected to compose cleanly with #1736 since the quant stage is orthogonal to CaseOps / attention gates / phased TTT.

## Baseline

Spec 008's reproduced seed-42 val_bpb (target ~1.06610 ± 0.003). Exact number is whatever spec 008 actually lands; this spec compares Δ against that.

## Expected Δ

- **Strong:** −0.005 to −0.007 bpb vs spec 008 → SpinQuant confirmed as a free quant lever on #1736, stacks with CaseOps.
- **Weak:** −0.001 to −0.004 bpb → partial benefit; investigate whether rotation classes were fully applied or whether #1736's int6 GPTQ is already less outlier-sensitive than #1695's stack.
- **Null or negative:** |Δ| ≤ 0.001 or positive → implementation bug suspected (likely a missed consistent-pair rotation); halt and debug before proceeding.

## Accept criteria

### Phase 1 — FP invariance sanity (before GPTQ)
- Load `pre_gptq.pt`, run one forward pass on a small batch, record logits.
- Apply all rotation classes (see "Rotation structure" below).
- Re-run the same forward pass post-rotation, compare logits.
- **Must match within float tolerance** (max abs diff ≤ 1e-3 bf16, ≤ 1e-5 fp32). If not, rotation pairs are inconsistent → halt, debug.

### Phase 2 — GPTQ + TTT + eval
- GPTQ quantization completes under same config as #1736 (`EMBED_BITS=7`, `MLP_CLIP_SIGMAS=12.0`, `ATTN_CLIP_SIGMAS=13.0`, etc.).
- Artifact < 16,000,000 bytes.
- Phased TTT eval completes within 600 s.
- val_bpb is reported.

### Primary success
- **val_bpb < spec 008 baseline by ≥ 0.003** (moves in the expected direction with plausible magnitude).
- Ideally ≥ 0.0072 for a standalone-record claim (0.005 nat threshold), but not required — this is a screen.

## Config diff

Same env block as spec 008 (identical GPTQ / TTT / gate settings). Two additions:

```
SPINQUANT_ENABLED=1
SPINQUANT_SEED=42                  # seed for the random orthogonal / signed-Hadamard generator
HOTSTART_FP_CKPT=/workspace/runs/008-1736-reproduction/seed_42/pre_gptq.pt
```

No training run. No `MATRIX_LR` / `MUON` / dataset settings matter — this spec skips training entirely.

## Rotation structure

Three classes, all fixed (not learned), stored as non-parameter buffers:

| Class | Shape | Applied to | # distinct Rs | Constraint |
|---|---|---|---|---|
| **Residual-stream R₀** | d_model × d_model | Embedding weight, attn input projections (Q/K/V slices of `qo_bank`/`kv_bank`), attn output projection input side, MLP W1 input side, MLP W2 output side, lm_head input side | 1 (global, shared across all 11 layers) | must be orthogonal; applied as `W ← R₀·W` or `W·R₀ᵀ` per in/out side |
| **Per-layer attn R_a^ℓ** | d_head × d_head | Internal Q·Kᵀ rotation per layer | 11 (or fewer if banked) | applied to Q-out and K-out consistently |
| **Per-layer MLP R_m^ℓ** | d_ff × d_ff | Internal W1→W2 rotation per layer | 11 | applied to W1-out and W2-in consistently |

**R construction:** preferred `R = diag(±1) · Hadamard(d)` (signed Hadamard — structured, outlier-spreading). Fallback: random orthogonal via `torch.linalg.qr(torch.randn(d,d))`.

**Critical:** for `R₀`, every residual-stream read/write side must use the same R₀ across all layers (including across Loop45 recurrence passes — key R₀ by "residual stream" not by "invocation"). Any miss breaks float invariance.

## Code changes

- **Branch:** `research` (this is a commitment-class change — quant lever becomes part of our baseline if it lands).
- **New file:** `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/spinquant_hotstart.py` — a standalone script that:
  1. Loads the FP checkpoint specified by `HOTSTART_FP_CKPT`.
  2. Generates R₀, R_a^ℓ, R_m^ℓ using `SPINQUANT_SEED`.
  3. Applies rotations to banked weight tensors (slicing `qo_bank` / `kv_bank` as appropriate).
  4. Sanity-checks FP forward invariance on one batch (Phase 1 accept).
  5. Invokes #1736's existing GPTQ + TTT + eval pipeline on the rotated weights (reusing functions from `train_gpt.py`).
  6. Writes the artifact + bpb to `runs/009-spinquant-hotstart/`.
- **No modifications** to `train_gpt.py` other than exposing a couple of its GPTQ/eval functions as importable (if they're currently inlined under `if __name__ == "__main__"`).
- **Reference:** read #1695's diff when available; port rotation bookkeeping onto #1736's banked layout.

## Hardware ladder

- [x] **1×H100** — sufficient. No training, just rotate + GPTQ (~2 min) + TTT eval (~6–10 min). Could also use 2×H100 if DDP is needed for TTT parallelization, but TTT eval in #1736 runs on 8 ranks per its phased-TTT setup — may need to check whether single-rank eval is supported.
- **Fallback:** 8×H100 if phased TTT requires multi-rank.

## Seed plan

Single seed (42), matching spec 008. Compares directly against spec 008's seed-42 number.

## Inputs

- **FP checkpoint:** `runs/008-1736-reproduction/seed_42/pre_gptq.pt` (spec 008 output).
- **Data:** same CaseOps dataset as spec 008 (on persistent volume, already prepared).
- **Tokenizer:** bundled with #1736 submission dir, unchanged.

## Execution protocol

Single pod, single seed, single pass:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

mkdir -p /workspace/runs/009-spinquant-hotstart

NCCL_NET=Socket DATA_DIR=./data \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
SPINQUANT_ENABLED=1 SPINQUANT_SEED=42 \
HOTSTART_FP_CKPT=/workspace/runs/008-1736-reproduction/seed_42/pre_gptq.pt \
SEED=42 \
torchrun --standalone --nproc_per_node=8 spinquant_hotstart.py \
  > /workspace/runs/009-spinquant-hotstart/run.log 2>&1
```

## Kill protocol

- FP invariance check fails (Phase 1) → halt, save rotation seeds + diff stats, flag research.
- GPTQ calibration fails or hangs > 5 min → halt.
- Eval hangs > 15 min → halt, stop pod.
- After successful completion: stop pod per memory default.

## Stop-early criteria

- Phase 1 FP invariance: max abs logit diff > 1e-3 (bf16) → halt before wasting GPTQ time.
- Artifact size > 16 MB → halt, flag.
- val_bpb > spec 008 baseline + 0.003 (got *worse*) → likely rotation error, halt.

## Checkpoints to emit

**None.** Spec 009 is pure post-training, no new FP state worth saving. The only artifact is the rotated-and-quantized `.ptz` submission + the log.

## Cost estimate

| Item | Cost |
|---|---|
| Pod spin-up | $1 |
| Phase 1 invariance check (1×H100, ~1 min) | $0.10 |
| Phase 2 GPTQ + TTT eval (8×H100, ~10–15 min) | $3 |
| Buffer for debug | $2 |
| **Total** | **~$6** |

If FP invariance fails first try and we debug once, total rises to ~$10.

## Extra artifacts

- `runs/009-spinquant-hotstart/run.log` — full log
- `runs/009-spinquant-hotstart/artifact.ptz` — SpinQuant + GPTQ quantized submission
- `runs/009-spinquant-hotstart/rotation_seeds.json` — reproducibility: records the `SPINQUANT_SEED` and any per-layer seed offsets used
- `runs/009-spinquant-hotstart/invariance_report.json` — Phase 1 diff stats (max/mean logit diff pre vs post rotation)
- `runs/009-spinquant-hotstart/final.json` — val_bpb, Δ vs spec 008, artifact size, wall times

## Open questions for interview

1. **Can #1736's `train_gpt.py` export its GPTQ + TTT entry points as importable functions?** If they're all inlined under `main()`, the hotstart script has to duplicate orchestration code. Preferable: minimal refactor in `train_gpt.py` to split `main()` into named helpers that `spinquant_hotstart.py` can call.
2. **Banked layout accounting** — `qo_bank` and `kv_bank` in #1736 concatenate Q+O (or Q+O projections) and K+V respectively. Need to verify which slices correspond to which residual-stream reads/writes before rotating (a diagram from the code will confirm).
3. **Loop45 consistency** — confirm that `R₀` applied to residual-stream weights of layers 4 and 5 is invariant across the multiple invocations of those layers in the recurrence.
4. **Phased TTT compatibility** — phased TTT adapts weights during eval. SpinQuant rotation must be applied *before* phased TTT sees the weights, so TTT adapts the rotated weights. Verify the ordering in #1736's eval pipeline.
5. **Reference code availability** — is #1695's diff visible (is the PR source readable)? If yes, port directly. If not, we re-derive from the SpinQuant paper and cross-check against the banked layout.

## What this spec does NOT do

- Does not retrain any weights.
- Does not tune `SPINQUANT_SEED` — first run uses 42; if rotation-seed sensitivity matters (unlikely for Hadamard) we can sweep later.
- Does not change CaseOps, gates, TTT, or any other non-quant lever.
- Does not emit a pre-quant checkpoint (spec 008's serves for all quant-family hotstarts).
- Does not run multi-seed — matches spec 008's single-seed convention.
