# Spec 001 — Hessian-SDClip λ screen

**Slug:** `hessian-sdclip`
**Created:** 2026-04-19
**Links to idea:** `research/ideas/hessian-sdclip.md`

## Hypothesis
Modulating SDClip by Hessian-derived per-row importance (`c_i = k · σ_i · [1 + λ(r_i − 1)]`) reduces post-quant bpb on the spec-000 weights. Ported from near-SOTA submission (2026-04-06_HessianSDClip_ProgressiveRecurrence). This is a **hotstart-screen** — no training, just re-quantize spec-000's `ckpt_final_pre_ema` with different λ values and read the resulting quantized bpb.

## Baseline
Our spec 000 run: **val_bpb_quantized = 1.10430** (raw, no sliding, no TTT). λ=0 variant in this spec must reproduce this within ~0.0001.

## Expected Δ
+0.0002 to +0.0010 bpb at the sweet spot (likely around λ=0.175 per near-SOTA author). High-confidence that the *sweep reveals a sweet spot*, lower confidence on absolute magnitude transferring to our architecture.

## Accept criteria
- **Validity gate:** λ=0.0 run must produce `val_bpb_quantized ∈ [1.10420, 1.10440]` (within 0.0001 of spec 000's 1.10430). If not, the code change has a bug — fail the spec before sweeping.
- **Signal gate:** at least one non-zero λ with `Δ_quant ≤ −0.0005` vs λ=0 control. If nothing passes this, mark spec as "no signal, promoted to kill" in evaluation.

## Config diff
Hyperparam-only (no training). Compared to spec 000:
- `HESSIAN_CLIP_LAMBDA` swept over **{0.00, 0.05, 0.10}** (3 values — "does it change anything at all?" probe at the conservative low end).
  - 0.00 = control (validity check, must reproduce 1.10430).
  - 0.05 = gentle modulation.
  - 0.10 = still-gentle but a visible step up.
  - **Follow-ups live during the execution session, not as a separate spec.** If signal appears, user drives the executioner to fill in finer values (possibly going smaller like 0.02, or extending up to 0.15+). If nothing moves, stop.
- All other env vars identical to spec 000: `BIGRAM_VOCAB_SIZE=0 QK_GAIN_INIT=5.25 SEED=42`.
- `TTT_ENABLED=0` for the screen (spec 000's env had it 1 — explicitly disable here; screen is quant-only, TTT applies only to promoted λ in future follow-up).

## Code changes
- Branch: `exp/hessian-sdclip`
- Commit: `74c8385`
- Diff (inline for interview):
  ```python
  # new hyperparam
  hessian_clip_lambda = float(os.environ.get("HESSIAN_CLIP_LAMBDA", 0.0))

  # in gptq_quantize_weight, when lambda > 0:
  col_importance = diagH_orig / diagH_orig.mean()
  row_importance = (W_orig.abs() * col_importance.unsqueeze(0)).mean(dim=1)
  row_importance = row_importance / row_importance.mean().clamp_min(1e-10)
  adj = 1.0 + hessian_clip_lambda * (row_importance - 1.0)
  s = (clip_sigmas * row_std * adj / clip_range).clamp_min(1e-10).to(torch.float16)
  ```
  Lambda=0 path is a no-op (uses original SDClip formula). Change total: +12 / −3 lines.

## Hardware ladder
- [ ] 1×H100 — **primary choice** (nothing here parallelizes across GPUs in a useful way — GPTQ is sequential per-matrix).
- [ ] 2×H100 mini — **fallback only**, if 1×H100 has memory pressure on the calibration pass or if CLAUDE.md's "mini rung" is required for any code-change spec.
- [ ] 8×H100 — not used.

## Seed plan
Single seed (42) — matches the hotstart checkpoint. This is a deterministic screen (given fixed calibration data + fixed weights), so seed variance is not a factor within a λ.

## Inputs
- **Hotstart checkpoint:** `/workspace/runs/000-sota-replication/checkpoints/ckpt_final_pre_ema_step3849.pt` (NA-1 volume `hvpdph5i3g`, from spec 000).
- Data: `/workspace/data/datasets/fineweb10B_sp8192/` (for GPTQ calibration + val eval only).
- Tokenizer: `/workspace/data/tokenizers/fineweb_8192_bpe.model`.
- Base repo commit: `74c8385` on `exp/hessian-sdclip`.

## Checkpoints to emit
No training checkpoints (no training happens). But **save the quantized submission artifact per λ** for cheap follow-up evaluation:

- Filename: `runs/001-hessian-sdclip/quantized_lambda_{VAL}.ptz` (one per λ value run).
- Content: the post-GPTQ + Brotli-compressed model (same format as `final_model.int6.ptz` in normal runs).
- Size: ~16 MB each × 8 λ = ~130 MB total on NA-1 volume. Negligible.
- Retention: keep through the record-track push (through 2026-04-30). If a λ promotes, we run sliding + TTT on its saved `.ptz` without re-quantizing (saves ~3-4 min per follow-up eval).
- Destination: `/workspace/runs/001-hessian-sdclip/` directory.

## Execution protocol
**REQUIRED: reuse the Hessian across all λ values.** The Hessian depends only on the (fixed) weights, not on λ. Recomputing it per-λ doubles cost for zero benefit.

Ideal flow (single script invocation, loops over λ, accepts user-driven additions):

```python
# Pseudocode — execution should implement as a single script
ckpt = load_checkpoint(ckpt_path)
apply_ema(ckpt)
hessians = collect_hessians(...)  # ONCE, reused below

initial_lambdas = [0.00, 0.05, 0.10]
for lam in initial_lambdas:
    h.hessian_clip_lambda = lam
    quantized = gptq_mixed_quantize(ckpt.state_dict, hessians, h)
    brotli_save(quantized, f"/workspace/runs/001-hessian-sdclip/quantized_lambda_{lam}.ptz")
    bpb = eval_val_quantized(h, quantized)
    write_json({"lambda": lam, "val_bpb_quantized": bpb, ...}, f"lambda_{lam}.json")

# Pause here for user decision. If more lambdas, loop again with reused `hessians`.
```

Env at launch: `BIGRAM_VOCAB_SIZE=0 QK_GAIN_INIT=5.25 TTT_ENABLED=0 SEED=42`. The λ values loop inside the script — NOT set via env.

**Keep the pod alive after the initial 3 λ** so the user can decide whether to add more values (e.g. 0.02, 0.15) based on results, without re-paying the Hessian + setup cost. Stop the pod only once the user says the sweep is done.

Naïve flow (one torchrun per λ, each re-collecting Hessian) is **not acceptable** here — flag and fix if `hotstart.py` works that way. If `hotstart.py` can't cleanly loop over λ with shared Hessian state, execution writes a small wrapper that calls `collect_hessians` once and `gptq_mixed_quantize` + `eval_val` per λ. JSON schema per λ: `{lambda: X, val_bpb_quantized: Y, val_bpb_pre_quant: Z, artifact_size_bytes: N}`.

## Stop-early criteria
**Early termination is a first-class goal:**

- **Validity fail:** if λ=0.00 run does NOT reproduce 1.10430 ± 0.0001, **stop immediately**. Code bug. Don't waste the other 2 runs.
- **Signal fail:** if λ=0.05 AND λ=0.10 both show `|Δ_quant| < 0.0002` vs control (within noise), user decides live with executioner whether to try larger λ (0.15, 0.175) or kill. Don't auto-continue.
- **Signal success:** if either λ=0.05 or 0.10 shows `Δ_quant ≤ −0.0003`, user drives executioner to fill in neighboring values (finer grid around the winning point).
- **Standard:** NaN / obvious failure → kill and mark failed.

## Cost estimate
- 1×H100 NA-1 at ~$2.50/hr.
- Flow: one-time setup (load + EMA + Hessian) ~3-5 min, then per-λ (GPTQ + eval + save) ~2 min × 3 = ~6 min.
- **Base wall: ~9-11 min. Base cost: ~$0.45.**
- User-driven follow-ups during executioner session may add 1-4 more λ values at ~$0.08 each. Running cap: ~$1.00 unless something surprising keeps expanding.
- If 1×H100 has memory pressure, fall back to 2×H100: ~$1.00 base.

## Extra artifacts
- One JSON per λ at `runs/001-hessian-sdclip/lambda_{VAL}.json` with at least: `lambda`, `val_bpb_quantized`, `val_bpb_pre_quant`, `artifact_size_bytes` (post-Brotli, for sanity).
- One quantized model artifact per λ at `runs/001-hessian-sdclip/quantized_lambda_{VAL}.ptz` (see Checkpoints section above).
- Aggregate table `runs/001-hessian-sdclip/summary.md` — all 8 (or fewer if early-killed) λ values with bpb and Δ vs control. **This is the artifact research needs for evaluation.**
- `notes.md` — execution's run narrative (if any issues).

No train.log (no training), no checkpoints.

## Open questions for interview
- Confirm `ckpt_final_pre_ema_step3849.pt` is still on the NA-1 volume at the expected path. (Should be; spec-000 kept checkpoints.)
- Confirm `hotstart.py` supports the `--mode requant_eval` workflow — if not, execution may need to write a small wrapper that calls the existing `collect_hessians` + `gptq_mixed_quantize` + `eval_val` functions directly. **Execution: if the CLI doesn't match, surface this immediately before burning pod time.**
- Confirm `TTT_ENABLED=0` is passed for the screen (default is 0 but spec 000's env was 1 — don't inherit).
- Confirm 2×H100 pod has the SOTA volume mounted at `/workspace/`.
