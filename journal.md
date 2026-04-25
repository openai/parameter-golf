# Journal

Session start: 2026-04-25 13:32 EDT (continuation of overnight session). Goal this session: cover BIG code-level directions (sliding-window attention, depth recurrence, parallel residuals, SwiGLU+depth-trim, GPTQ-style int6 quant), not micro-tune env-vars. Subagent-driven.

## Current threads

- **Submittable best**: **2.10275 (exp 0057, `winners/2026-04-25_recur_3x3_swiglu_mlp3`)**. SEED=1337. SEED=42 confirm (0058) gave 2.10579 — cross-seed Δ 0.003 (typical), mean **2.10427**, **mean Δ vs 0051 = +0.0055** (just over noise floor). Cumulative gain vs canonical 2.5212 → **+0.418 (~16.6%)**. Code-level: K=3 unique blocks looped L=3 times (effective depth 9, no U-Net skips) + SwiGLU(mlp=3). Artifact 5.998 MB → **10 MB cap headroom remaining for further compounds**.
- Prior winner: 2.10971 (exp 0051, MUON_BACKEND_STEPS=15 stack on canonical-architecture).
- **Best non-submittable** (size violation): exp 0044 SwiGLU(mlp=3) at val_bpb 2.11489, artifact 16.46 MB — superseded by 0057 which now fits.
- **Cross-seed variance baseline**: 0.0024–0.0027 for stable configs. Larger Δ between seeds (0.008–0.015) is a marker of an outlier run, not the true effect.
- **Quant_tax sanity range**: 0.002–0.005 healthy. <0.001 is a red flag (mode collapse, freak run).
- **lr_mul formula**: after warmup, `(iterations - step) / warmdown_iters`. With ITERATIONS=200, WARMDOWN_ITERS=300, peak lr_mul=0.667 at step 0 of warmdown (after 30-step warmup).
- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. Bit-reproduces Apr-18 reference.

## Confirmed-paying axes (in order of contribution, post-quant Δ)

| Axis | Δ | Tag | Exp |
|---|---|---|---|
| LR schedule rewrite (warmup10+warmdown600) | +0.116 | low | 0005 |
| TRAIN_BATCH_TOKENS 8192 → 16384 | +0.082 | high | 0013 |
| Schedule push (warmup20+warmdown400) | +0.055 | low | 0015 |
| Batch 16k→24k + LR 0.06→0.045 | +0.045 (single-seed; mean +0.038) | high | 0036 |
| TIED_EMBED_INIT_STD 0.005 → 0.05 | +0.038 (cumulative across 0023+0024) | high | 0024 |
| Schedule push (warmdown 300) | +0.029 | low | 0020 |
| MUON_BACKEND_STEPS 5 → 15 (Newton-Schulz iters) | +0.023 (cumulative 0049→0051) | med | 0051 |
| MATRIX_LR 0.04 → 0.06 (at batch=16k) | +0.016 | med | 0021 |
| MLP_MULT 2 → 4 | +0.014 | high | 0008 |

## Dead axes (no signal at this regime)
QK_GAIN, LOGIT_SOFTCAP, MUON_MOMENTUM, BETA1, BETA2, ROPE_BASE, GRAD_CLIP_NORM, TIED_EMBED_LR scale-up, SCALAR_LR scale-up, ORTHO_INIT, TRAIN_SEQ_LEN=2048, TRAIN_BATCH_TOKENS=32768, NUM_LAYERS=11, MLP_MULT=5+, LeakyReLU².

## Open questions (carried forward)
- Why does batch=32k mode-collapse even at MATRIX_LR=0.03? Conjecture: 4-sequences-per-microstep loses critical stochasticity. Untested directly (would need code change to grad_accum_steps).
- Does SwiGLU(mlp=3) + NUM_LAYERS=8 fit cap and retain the +0.011 gain? Untested.
- Does sliding-window attention (record-heavy) improve anything at d=512/9L/200-step? Untested.
- Does parallel-residual structure help at this scale/budget? Untested.
- Does GPTQ-style int6 with calibration free enough cap-room to use bigger MLP? Untested.

## Entries (newest first)

## 2026-04-25 · exp 0058 · SEED=42 confirms 0057 (cross-seed Δ 0.003, mean Δ vs 0051 +0.0055) — promoted

SEED=42 of 0057. val_bpb_post = **2.10579** (vs SEED=1337's 2.10275). Cross-seed Δ = 0.003 — at the typical baseline (0.0024-0.0027) for stable configs. Both seeds positive vs 0051 (+0.007 SEED=1337, +0.004 SEED=42), mean Δ = **+0.0055** (just at the +0.005 protocol boundary).

Per protocol "if Δ holds across both seeds, advance" → **promoted as `winners/2026-04-25_recur_3x3_swiglu_mlp3`**.

The SEED=1337 single-seed Δ was at the upper edge of the cross-seed range — consistent with the previous session's lesson that single-seed wins overstate by ~10-20%. The mean +0.0055 is the more honest claim.

Cumulative gain vs canonical baseline (2.5212): single-seed +0.418, mean-seeded +0.417 → 2.10427.

## 2026-04-25 · exp 0057 · SwiGLU(mlp=3) + K=3 L=3 recurrence — NEW BEST 2.10275 (Δ+0.007), pending SEED=42

**Question**: Does the SwiGLU gain (+0.011 at 9L from 0044, cap-violating) survive when combined with K=3 L=3 depth recurrence (-0.013 standalone from 0056)?

**Setup**: Forked 0056. Added `MLP_TYPE=swiglu` env-gated SwiGLU MLP (w_gate, w_up silu-gated → w_down). MLP_MULT=3 (the 0044 winning width). All other env: 0051 winner + recurrence vars from 0056.

**Prediction** [CONJECTURE]: Δ in [-0.020, +0.010] vs 0051. Optimistic case argued the gating would compound under repeated invocation; pessimistic case argued recurrence would interfere with gating efficiency.

**Disconfirming**: Δ ≤ -0.020 → SwiGLU+recur compound fails.

**Result**: val_bpb_post = **2.10275**, **Δ = +0.00696 vs 0051** (in [+0.005, +0.010] judgment-call zone). Pre-quant Δ = +0.008 — pre/post Δ match. Quant_tax 0.0024 (normal). **Artifact 5.998 MB** (vs 0051's 15.18 MB) — **~10 MB cap headroom remaining** for further compounds.

Trajectory comparison vs 0051 / 0056:
- Step 10: 0057 5.97 vs 0056 5.80 vs 0051 6.22 (0057 between recurrence-only and 9L)
- Step 200 train_loss: 0057 3.53 vs 0056 3.57 vs 0051 3.55 (0057 best)
- Step_avg: 0057 3721 ms ≈ 0056 3638 ms (recurrence dominates step time, SwiGLU adds ~3%)

**Conclusion** [LIKELY, pending SEED=42]: SwiGLU's gating advantage **compounds with recurrence**. The SwiGLU's ~+0.011 base gain (from 0044) plus the recurrence's -0.013 base cost (from 0056) sum to nearly tied; the additional ~+0.007 over the linear sum means **the gating benefits more from repeated invocation than from distinct invocation**. Each recurrent loop applies the gate to a different residual-stream state — multiple gating decisions per token at the same param cost.

This refutes the "gain shrinks under recurrence" pessimistic prior. Direction is GO.

**[transfer:med]** — gating + recurrence is a record-validated stack (e.g. 2026-04-09 SP8192_3LayerRecur_ParResid_QK525_LegalTTT scores 1.0810). Our 200-step result suggests the compounding kicks in even at short training.

**Followups**:
1. **SEED=42 confirm of 0057** (mandatory by protocol for +0.005-0.010 Δ).
2. **Push SwiGLU width**: K=3 L=3 + SwiGLU(mlp=4 or 8) — cap allows.
3. **Push loops**: K=3 L=4 or K=3 L=5 (eff depth 12 or 15 at same param count).
4. **Push K**: K=4 L=3 (more block diversity).

## 2026-04-25 · exp 0056 · depth recurrence (K=3 L=3) costs only -0.013 and frees 10.6 MB cap

**Question**: Does depth recurrence (K=3 distinct blocks looped L=3 times, effective depth 9) train at our 200-step regime, and how much does it cost val_bpb? The motivating prize is the cap savings: 3 unique blocks ≈ 9M params (vs 26M canonical) → ~10 MB freed, opening compounds (SwiGLU, wider MLP, more loops).

**Setup**: Forked 0051 winner. Subagent-implemented (~50 lines) `NUM_UNIQUE_LAYERS=3 NUM_LOOPS=3` env-var-gated path: when active, builds K Block instances and loops them L times in `GPT.forward`, dropping U-Net skip-weights entirely. When `NUM_UNIQUE_LAYERS=0`, code path is unchanged (canonical U-Net). Required one main() fix: `base_model.skip_weights.numel()` access guarded with `hasattr`.

**Prediction** [CONJECTURE]: Δ in [-0.040, +0.005]. 200 steps may not give the loop structure time to specialize, and dropping U-Net skip-weights costs separately.

**Disconfirming**: Δ ≤ -0.030 → too costly, park direction.

**Result**: val_bpb_post=**2.12273**, **Δ=-0.013 vs 0051** (just below noise floor +0.010 but a genuine loss in pre-quant too: -0.013 — pre/post Δ match cleanly). Artifact **5.37 MB** (vs 0051's 15.18 MB) → **10.6 MB cap headroom freed**. Quant_tax 0.001026 (low — usually a freak-run flag, but pre/post Δ match so the gain is real, not a quant artifact). Step_avg 3638 ms (slightly faster than 0051's 3836 ms; same compute count but no skip-weight gather/scatter).

Trajectory comparison vs 0051:
- Step 10: 0056 5.80 vs 0051 6.22 (0056 ahead — recurrent params get more updates per step)
- Step 100: 0056 3.82 vs 0051 ~3.85 (close)
- Step 185: 0056 3.49 vs 0051 3.46 (0051 pulls ahead late)
- Step 200: 0056 3.57 vs 0051 3.55 (0056 -0.02 train_loss)

The val_bpb cost (-0.013) is small enough that the freed cap room should be exploitable. **Don't promote** as a winner — the absolute val_bpb is worse — but **park as scaffolding** for compounds.

**Conclusion** [LIKELY]: depth recurrence at K=3 L=3 trains acceptably at 200 steps; the recurrent params get effectively 3× the update count per training step, partially compensating for fewer distinct blocks. The major cost is loss of U-Net skip structure, not the recurrence itself. Cap freed: ~10 MB. Direction is GO for compounds.

**[transfer:med]** — at H100 20k-step, the loop structure has more time to specialize → recurrence cost likely shrinks or reverses. Records that use depth-recur (e.g. 2026-04-04, 2026-04-09) score in the 1.08-1.09 range (top tier).

**Followups queued**:
1. **0057 K=3 L=3 + SwiGLU(mlp=3)**: combine cap-freed recurrence with the SwiGLU win 0044 found cap-violating. Predicted: pull -0.013 toward 0 or beat 0051.
2. **0058 isolation: K=9 L=1** (recurrent codepath, no actual recurrence): isolates the U-Net-skip-removal cost from recurrence cost. If 0058 ≈ 0051, U-Net skip is the cost; if 0058 ≈ 0056, recurrence is.
3. **0059 K=3 L=3 + MLP_MULT=8** (very wide MLP, ~9 MB int8): tests width-via-recurrence frontier.

## 2026-04-25 · exp 0055 · parallel residuals hurt -0.038 standalone

**Question**: Does PaLM-style parallel residual (`attn` and `mlp` see the SAME post-norm input, residuals added in parallel) help on top of 0051? Records use it heavily as a stack component.

**Setup**: Forked 0051 winner. Code-only change in `Block.forward`: instead of `x = x + attn(...); x = x + mlp(post-attn x)`, do `attn_out = attn(...); mlp_out = mlp(SAME x); x = x + attn_scale*attn_out + mlp_scale*mlp_out`. Same params, same scales — just the residual order differs.

**Prediction** [CONJECTURE]: Δ in [-0.005, +0.010]. Records use parallel-resid in stacks; standalone effect at small scale typically neutral or slightly positive.

**Disconfirming**: Δ ≤ -0.010 → parallel residuals actively hurt at d=512/200 steps; the inter-block sequential dependency does real work.

**Result**: val_bpb_post=2.14780, **Δ=-0.038 vs 0051 (clear loss, 4× the noise floor)**. Pre-quant Δ also -0.036 — not a quant artifact. Quant_tax 0.0031 (normal). Trajectory bit-identical to 0051 through step 10 (init dynamics are unchanged), but train_loss diverges by step 80 (3.96 vs 0051's 3.88) and stays behind to the end (3.60 vs 3.55).

**Conclusion** [VERIFIED standalone, single-seed]: parallel residuals as a *standalone change* on this stack actively hurt. The residual ordering matters at d=512 / 200 steps — feeding MLP the post-attention output (sequential) gives ~0.04 of val_bpb that the parallel form throws away.

Why might records still use it: in the records' stacks (depth-recur + qk-gain + parallel-muon + ...) the loss compounds with other gains and the FLOPs savings (1 norm forward) buys ~5-10% wallclock for more steps. At our small scale the dependency loss outweighs.

**[transfer:low]** for the standalone null result. Don't carry forward solo. Keep on the table only as part of a depth-recurrence stack — records that combine the two may compensate via the loop structure.

## 2026-04-25 · session start

Session start at 13:32 EDT picking up after the overnight session. Plan: cover BIG code-level directions (parallel residuals, depth recurrence, SwiGLU+layer-cut, sliding-window attention, GPTQ-style quant) rather than env-var micro-tuning. Subagent-driven for code changes >15 lines. Expectation: 5-8 experiments this session, each giving a coverage-style signal, not a tuned optimum.


