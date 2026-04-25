# Journal entries — 2026-04-25 (afternoon, code-level directions session)

Rotated from `journal.md`. See `summaries/2026-04-25_code_directions_session.md` for the narrative handoff.

## 2026-04-25 · exp 0061 · smear-gate on 0057 hurts -0.027 (real)

**Question**: Does smear-gate (per-channel sigmoid mix of current and previous-token x at start of Block.forward) help on top of the 0057 recurrent+SwiGLU winner? Record-validated technique (`smear-gate` tag in records like 2026-03-19_smeargate_orthoinit_muonwd).

**Setup**: Forked 0057. Added `smear_gate_logit` parameter (shape `(dim,)`, init -2.0 → sigmoid≈0.119 light-smear prior) to `Block.__init__`, env-gated by `USE_SMEAR_GATE=1`. In `Block.forward`, mix `x = (1-g)*x + g*x_prev` before x0-mix and attention. ~10 lines code, self-implemented.

**Prediction** [LIKELY]: -0.005 to +0.015 vs 0057.

**Disconfirming**: Δ ≤ -0.010 vs 0057 → smear-gate prior conflicts with our stack.

**Result**: val_bpb_post = **2.13022**, **Δ = -0.0275 vs 0057** (clear loss, ~4× noise floor). Pre-quant Δ matches (-0.028) — real training cost, not quant artifact. Quant_tax 0.0018 (low-normal). Trajectory was healthy throughout (step 1 = 16.09 — note: smearing reduces the typical 20.7 init-anomaly because adjacent-token averaging dampens the lm_head log-prob spread).

**Conclusion** [VERIFIED standalone, single-seed]: smear-gate's prior interferes with our stack at d=512/200 steps. Possible reasons:
1. Attention at our 9-effective-layer recurrence is already getting all the local-pattern signal it needs; the explicit channel-wise pre-attention smearing adds correlation that gets washed out.
2. Init at -2.0 logit means sigmoid≈12% mix from step 0, which is a strong-ish prior that costs the model time to undo.
3. In records, smear-gate is paired with bigram-hash + int6 + muon-wd; the SUM of these may combine well even when the individual gain is small/negative.

**[transfer:low]** standalone — record evidence requires the broader stack.

## 2026-04-25 · exp 0060 · per-loop scalars (Universal Transformer-style) neutral at 200 steps

**Question**: 0057 K=3 L=3 has the same `attn_scale`, `mlp_scale`, `resid_mix` shared across all 3 invocations of each block. Universal Transformers / ALBERT / Relaxed Recursive Transformers identify this as the bottleneck for recurrent transformers — the model can't tell "first call" from "third call". Does adding per-loop indexed scalars (each block has L=3 separate copies) help?

**Setup**: Forked 0057. Subagent-implemented (~30 lines): added `PER_LOOP_SCALES=1` env-gate. When active and recurrent, `Block.{attn_scale, mlp_scale, resid_mix}` become `(num_loops, ...)`-shaped, indexed by `loop_idx` passed from `GPT.forward`. Default-off path is byte-identical to 0057 (verified via diff and trajectory match through step 10).

**Prediction** [LIKELY]: Δ +0.005 to +0.020 vs 0057. Theoretically grounded.

**Disconfirming**: Δ ≤ -0.005 → per-loop scalars hurt; Δ in [-0.005, +0.005] → neutral, pivot.

**Result**: val_bpb_post = **2.10412**.
- vs 0057 SEED=1337 (2.10275): **Δ = -0.0014** (essentially tied, within typical cross-seed noise 0.003).
- vs 0057 mean (2.10427): Δ = +0.0002 (neutral).
- vs 0051 (2.10971): Δ = +0.0056 (matches the 0057 mean improvement; per-loop didn't add value over 0057).

Pre-quant Δ vs 0057 = +0.0013 (similar — pre/post match, no quant artifact). Quant_tax 0.0024 (normal). Artifact 6.04 MB.

**Conclusion** [LIKELY]: at 200 steps, the model **cannot specialize the per-loop scalars meaningfully**. They start at unity (identity), and 200 steps + 12K extra params don't develop any per-loop differentiation big enough to register. The 0057 baseline (shared scalars across loops) is already as good as the model can do at this training budget.

This is consistent with the broader observation: every "specialization-takes-training-steps" technique (mlp_mult=5+, num_layers=11+, depth-recur K=large, per-loop) is flat or hurts at 200 steps. The records that use these techniques benefit from 20k+ steps of training.

**[transfer:med]** — at H100 20k-step the per-loop differentiation should kick in. For the smoke regime, this direction is parked.

**Followups**: pivot to a different axis. Smear-gate (record-validated, ~10 lines) or wider-MLP (mlp=8, env-var-only push). Both are simpler than per-loop scaling and may pay better at 200 steps.

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
