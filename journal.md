# Journal

## Current threads

- **Submittable best**: **2.08687 (exp 0062, `winners/2026-04-25_recur_3x3_swiglu_mlp8`)**. SEED=1337 single-seed (SEED=42 confirm pending). Δ=+0.017 vs 0057 mean. **Cumulative gain vs canonical 2.5212 → +0.434 (~17.2%)**. Code-level: K=3 unique blocks looped L=3 times + SwiGLU(mlp=8, hidden=4096 per block). Artifact 12.24 MB → 3.76 MB cap headroom remaining.
- Prior winner: 2.10275 (exp 0057, K=3 L=3 + SwiGLU mlp=3, mean 2.10427).
- **MLP-width curve inside K=3 L=3 recurrence (single-seed)**: mlp=3 → 2.10275, mlp=4 → 2.09706, mlp=8 → 2.08687. Monotonic, hasn't saturated. mlp=10 (~14.7 MB est) and mlp=11 (~16 MB) still fit cap; mlp=12+ over.
- **Cross-seed variance baseline**: 0.0024–0.003 for stable configs. Larger Δ between seeds (0.008–0.015) is a marker of an outlier run, not the true effect. Single-seed wins at the +0.005-+0.010 boundary tend to overstate by ~10-20%; protocol-mandated SEED=42 confirm rolls them back to the mean.
- **Quant_tax sanity range**: 0.002–0.005 healthy. <0.001 used to be a red flag for mode-collapse, but 0056 had quant_tax=0.001 and was a clean run (pre/post Δ matched) — **rule refined**: low quant_tax + matching pre/post Δ is fine; low quant_tax + diverging Δs is the actual freak-run signature.
- **lr_mul formula**: after warmup, `(iterations - step) / warmdown_iters`. With ITERATIONS=200, WARMDOWN_ITERS=300, peak lr_mul ≈ 0.567 at step 30, decaying to ~0 by step 200.
- **Init=0.05 step-1 anomaly**: train_loss=20.7 at step 1 (vs expected ~6.93 from ln(1024)) is a structural feature of init=0.05+warmup=30, not a bug. Confirmed across 0051, 0044, and all derivatives. The model recovers smoothly to ~6.0 by step 10. Mechanism not fully derived; likely an LR-warmup × tied-embedding interaction in the first few optimizer steps.
- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. Bit-reproduces Apr-18 reference.

## Stack of confirmed wins (cumulative path canonical → current best)

| Axis | Δ | Tag | Exp | Tier |
|---|---|---|---|---|
| LR schedule rewrite (warmup10+warmdown600) | +0.116 | low | 0005 | env |
| TRAIN_BATCH_TOKENS 8192 → 16384 | +0.082 | high | 0013 | env |
| Schedule push (warmup20+warmdown400) | +0.055 | low | 0015 | env |
| Batch 16k→24k + LR 0.06→0.045 | +0.045 (single) / +0.038 (mean) | high | 0036 | env |
| TIED_EMBED_INIT_STD 0.005 → 0.05 | +0.038 (cum.) | high | 0024 | env |
| Schedule push (warmdown 300) | +0.029 | low | 0020 | env |
| MUON_BACKEND_STEPS 5 → 15 | +0.023 (cum.) | med | 0051 | env |
| MATRIX_LR 0.04 → 0.06 (at batch=16k) | +0.016 | med | 0021 | env |
| MLP_MULT 2 → 4 (then 4 → 3 inside SwiGLU) | +0.014 (relu² at 9L) | high | 0008 | env |
| **K=3 L=3 depth recurrence + SwiGLU(mlp=3)** | **+0.0055 (mean) vs 0051** | **med** | **0057** | **code** |

## Dead axes (verified at this regime — don't re-test without changing other levers)
QK_GAIN, LOGIT_SOFTCAP, MUON_MOMENTUM, BETA1, BETA2, ROPE_BASE, GRAD_CLIP_NORM, TIED_EMBED_LR scale-up, SCALAR_LR scale-up, ORTHO_INIT, TRAIN_SEQ_LEN=2048, TRAIN_BATCH_TOKENS=32768, NUM_LAYERS=11, MLP_MULT=5+, LeakyReLU², parallel-residuals (standalone, -0.038), per-loop scalars (Universal Transformer-style, neutral), smear-gate (init -2.0, -0.027 in our recurrent stack).

## Open questions
- **Most important next move**: K=3 L=3 + SwiGLU(mlp=4 or 8) — wider per-block capacity inside recurrence. `experiments/0059_swiglu_recur_3x3_mlp4` is **prepped and ready to run** (env-set to mlp=4; bump to 8 for the bigger test).
- **K=9 L=1 isolation**: tells us whether the 0056 -0.013 recurrence cost is from depth-recur itself or from U-Net-skip removal. Just env-var changes (`NUM_UNIQUE_LAYERS=9 NUM_LOOPS=1`).
- **More-loops compound**: K=3 L=4 or L=5 — env-var only, no extra params, eff depth 12+. Tests if recurrence-iteration-count is the lever.
- **Bigram-hash embedding** (record-validated, ~30 lines, subagent): replace `tok_emb` with bigram-hash + projection.
- **EMA over weights** (record-validated, ~30 lines, subagent): track exponential moving average of weights for eval — should help at 200 steps where late-step train_loss is bouncy.
- **GPTQ-style pre-quant calibration** (top-record technique): bake calibration-aware scaling into weights so harness's int8 quant is cleaner. Multi-hundred lines. Highest payoff if cap is the real bottleneck.
- **Sliding-window attention in TRAINING** (banded mask, `is_causal=False`, custom attn_mask): records' "SWA" tags are mostly sliding-window EVAL not training-time; this is a genuinely untested-by-records direction.
- **Re-test smear-gate with init -6 logit** (sigmoid≈0.0025 ≈ off): -2.0 may have been too active. ~5-line change.
- **Why does init=0.05 produce step-1 train_loss=20.7?** [WORTH_DERIVING] — math doesn't quite predict 20.7 from naive analysis; understanding might reveal a better init.
- **Why does batch=32k mode-collapse even at MATRIX_LR=0.03?** Conjecture: 4-sequences-per-microstep loses critical stochasticity. Untested directly (would need code change to grad_accum_steps).

## Entries (newest first)

## 2026-04-25 · exp 0062 · K=3 L=3 + SwiGLU(mlp=8) — NEW BEST 2.08687, Δ+0.0102 vs 0059 mlp=4

**Question**: With 0059 (mlp=4) showing the wider-MLP curve still paying inside recurrence, does pushing further to mlp=8 (hidden=4096 per block) keep paying or hit a capacity ceiling?

**Setup**: Forked 0057. Single env-var change: `MLP_MULT=8`. SwiGLU MLP: w_gate(d, 8d) + w_up(d, 8d) + w_down(8d, d). 21.3M params raw, expected artifact ~13 MB.

**Prediction** [CONJECTURE]: Δ +0.005 to +0.020 vs 0057. Wider keeps paying based on 0059 result.

**Disconfirming**: Δ ≤ -0.005 vs 0057 → very wide MLP under-trains at 200 steps.

**Result**: val_bpb_post = **2.08687** — direct-promote-zone win.
- **Δ vs 0059 mlp=4 (current preliminary winner): +0.01019** (above +0.010 noise floor → direct-promote)
- Δ vs 0057 SEED=1337: +0.01588
- Δ vs 0057 mean: +0.01740 (huge against the mean-anchor)
- Δ vs 0051 (original winner): +0.02284
- Pre-quant Δ vs 0059: +0.0114 (matches post-quant; clean training gain)
- Quant_tax 0.00247 (normal). Artifact **12.24 MB** (vs cap 16 MB; 3.76 MB headroom).
- Step_avg 5396 ms (vs 0059's 4206 ms — 28% slower per step due to 2× wider MLP).

Trajectory comparison vs 0059 / 0057:
- Step 100: 0062 3.76 vs 0059 ~3.81 vs 0057 ~3.82
- Step 200 train_loss: 0062 3.50 vs 0059 3.52 vs 0057 3.55

**Conclusion** [LIKELY, pending SEED=42]: Width-inside-recurrence is the dominant lever. Each unit of mlp_mult adds ~+0.002 of val_bpb improvement. The curve is monotonic from mlp=3 (2.103) → mlp=4 (2.097) → mlp=8 (2.087). The compound (recurrence × wide gating) is much more powerful than either alone.

**Cumulative gain vs canonical (2.5212)**: **+0.434 (~17.2%)**. Best result of the day.

**[transfer:high]** — wider MLP is the most robust H100-transfer lever; this should hold cleanly at 20K-step training.

**Followups**:
1. SEED=42 confirm of 0062 (mandatory for direct-promote within 5 experiments).
2. Push curve: mlp=10 (~14.7 MB, fits) or mlp=11 (~16 MB, borderline). The marginal gain may diminish but still likely positive.
3. Combine: K=3 L=4 (more loops) + SwiGLU(mlp=8) → eff depth 12, but artifact stays at 12.24 MB.

## 2026-04-25 · exp 0059 · K=3 L=3 + SwiGLU(mlp=4) — NEW BEST 2.09706 (Δ+0.007 vs 0057 mean)

**Question**: Does SwiGLU MLP_MULT=4 (vs 0057's mlp=3) inside K=3 L=3 recurrence pay? 0057's cap is at 6 MB; 10 MB headroom is unused. mlp=4 → 7.3 MB artifact, still well under cap.

**Setup**: Forked 0057. Single env-var change: `MLP_MULT=4`. No code change. SwiGLU MLP architecture same (w_gate + w_up + w_down).

**Prediction** [LIKELY]: Δ +0.005 to +0.020 vs 0057 — wider per-block MLP inside recurrence should help, records cap at mlp=3-4 in their stacks.

**Disconfirming**: Δ ≤ -0.000 vs 0057 — wider hurts at recurrence.

**Result**: val_bpb_post = **2.09706** (vs 0057 SEED=1337 2.10275, mean 2.10427). 
- Δ vs 0057 SEED=1337: **+0.00569** (judgment-call zone)
- **Δ vs 0057 mean: +0.00721** (above noise floor)
- Δ vs 0051 mean (orig benchmark): +0.0127 (clear win)
- Pre-quant Δ vs 0057 SEED=1337: +0.0046 (matches post-quant; clean training gain)
- Quant_tax 0.00126 (low-normal, similar to 0057's 0.0024). Artifact 7.28 MB.
- Step_avg 4206 ms (vs 0057's 3721 ms — 13% slower per step due to 33% wider MLP).

Trajectory comparison: at step 165, 0059 train_loss = 3.5397 vs 0057 3.5533 (0059 +0.014 ahead). Final step 200 train_loss 0059 = 3.52 vs 0057 = 3.53.

**Conclusion** [LIKELY, pending SEED=42]: Wider per-block SwiGLU pays inside recurrence. The mlp=3 → mlp=4 increase is a 33% expansion of hidden dim per block (1536 → 2048), giving the gating more interaction capacity. Records that cap at mlp=3-4 standalone don't have the recurrent invocation pattern; with K=3 L=3, each MLP runs 3× per token, amplifying the value of more capacity.

The headroom story matters: cumulative gain since canonical (2.5212) is now **+0.424 (~16.8%)**. Cap is at 7.28 MB — still 8.7 MB headroom for more compounds. The strategic position keeps improving.

**[transfer:high]** — wider MLP is well-known robust at H100 scale; this is exactly the kind of "mlp_mult=4 wins" finding that transfers cleanly.

**Followups**: 0062 (mlp=8, launching) tests if even wider helps; SEED=42 confirm of 0059 mandatory before final promote.

## 2026-04-25 · session resumed at ~15:55 EDT

User pointed out I prematurely closed the session at 15:17 after misreading "find time to wrap up what you have and go take-a-walk" as a session-end signal. Per program.md, that was a walk request, not a stop request. NEVER STOP until manually told. Resuming on the highest-EV next move from the walk note: **K=3 L=3 + SwiGLU(mlp=8)** as the cap-utilizing push.


