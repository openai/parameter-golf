# Journal

## Current threads

- **Submittable best (committed)**: **2.08687 (exp 0062, `winners/2026-04-25_recur_3x3_swiglu_mlp8`)**. SEED=1337 single-seed. Δ vs 0057 mean = +0.0174. **Cumulative gain vs canonical 2.5212 → +0.434 (~17.2%) single-seed**. Code-level: K=3 unique blocks looped L=3 times + SwiGLU(mlp=8, hidden=4096 per block). Artifact 12.24 MB / 16 MB cap.
- **Best unpromoted**: **2.07994 (exp 0063)** at mlp=11, single-seed, Δ+0.007 vs 0062. Held back per methodology — waiting for SEED=42 confirms.
- **MLP-width curve inside K=3 L=3 recurrence (single-seed)**: mlp=3 → 2.10275 (0057 mean 2.10427), mlp=4 → 2.09706, mlp=8 → 2.08687, mlp=11 → 2.07994. Monotonic, ~+0.0023 per mlp_mult unit, hasn't saturated. Cap is the ceiling: mlp=12+ exceeds 16 MB.
- **Methodology debt (HIGH PRIORITY for next session)**: 0059, 0062, 0063 are all single-seed direct-promote-zone wins without SEED=42 confirms. Per protocol "within 5 experiments" rule, SEED=42 of 0062 is overdue. The +0.434 cumulative claim is single-seed; honest cross-seeded estimate is likely +0.40 to +0.43.
- **Cross-seed variance baseline**: 0.0024–0.003 for stable configs. Single-seed wins at the +0.005-+0.010 boundary tend to overstate by ~10-20% (lesson from 0057→0058 confirm). Apply this shrinkage as a back-of-envelope for any unconfirmed magnitude claim.
- **Quant_tax sanity**: 0.002–0.005 healthy. <0.001 + matching pre/post Δ is fine; <0.001 + diverging Δs is the freak-run signature.
- **lr_mul formula**: after warmup, `(iterations - step) / warmdown_iters`. With ITERATIONS=200, WARMDOWN_ITERS=300, peak lr_mul ≈ 0.567 at step 30, decaying to ~0 by step 200.
- **Init=0.05 step-1 anomaly**: train_loss=20.7 at step 1 is a structural feature of init=0.05+warmup=30, not a bug. Recovers smoothly to ~6.0 by step 10. Mechanism not derived; smear-gate evidence (0061) shows token-mixing reduces it to 16.1.
- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB.

## Stack of confirmed wins (cumulative path canonical → current best)

| Axis | Δ | Tag | Exp |
|---|---|---|---|
| LR schedule rewrite (warmup10+warmdown600) | +0.116 | low | 0005 |
| TRAIN_BATCH_TOKENS 8192 → 16384 | +0.082 | high | 0013 |
| Schedule push (warmup20+warmdown400) | +0.055 | low | 0015 |
| Batch 16k→24k + LR 0.06→0.045 | +0.045 / +0.038 mean | high | 0036 |
| TIED_EMBED_INIT_STD 0.005 → 0.05 | +0.038 | high | 0024 |
| Schedule push (warmdown 300) | +0.029 | low | 0020 |
| MUON_BACKEND_STEPS 5 → 15 | +0.023 | med | 0051 |
| MATRIX_LR 0.04 → 0.06 (at batch=16k) | +0.016 | med | 0021 |
| MLP_MULT 2 → 4 (relu² at 9L) | +0.014 | high | 0008 |
| K=3 L=3 depth recurrence + SwiGLU(mlp=3) | +0.0055 mean vs 0051 | med | 0057 |
| **SwiGLU mlp=3 → mlp=8 inside recurrence** | **+0.0174 single vs 0057 mean** | **high** | **0062** |
| SwiGLU mlp=8 → mlp=11 (unpromoted) | +0.007 single | high | 0063 |

## Dead axes (verified — don't re-test without changing other levers)
QK_GAIN, LOGIT_SOFTCAP, MUON_MOMENTUM, BETA1, BETA2, ROPE_BASE, GRAD_CLIP_NORM, TIED_EMBED_LR scale-up, SCALAR_LR scale-up, ORTHO_INIT, TRAIN_SEQ_LEN=2048, TRAIN_BATCH_TOKENS=32768, NUM_LAYERS=11, MLP_MULT=5+ (relu², not SwiGLU), LeakyReLU², parallel-residuals (standalone, -0.038), per-loop scalars (Universal Transformer-style, neutral), smear-gate (init -2.0, -0.027 in our recurrent stack).

## Open questions (next session priorities — ranked by EV)

**MUST-DO FIRST (methodology debt):**
1. **SEED=42 of 0062** — the committed winner is unconfirmed. Per protocol, this is overdue. ~14 min wallclock.
2. **SEED=42 of 0063** — confirms whether mlp=11 actually beats mlp=8. ~22 min. If confirmed, promote 0063 (becomes new winner).

**Then (independent-mechanism axes — highest-EV unexplored):**
3. **EMA over weights** (record-validated, ~30 lines, subagent): track exponential moving average of weights for eval. Mechanism is INDEPENDENT of gating/recurrence/width — should compound, not compete. Likely +0.005-0.015.
4. **K=9 L=1 isolation** — disambiguates depth-recur cost vs U-Net-skip-removal cost in 0056. One env-var change, ~6 min experiment.
5. **Bigram-hash embedding** (record-validated, ~30 lines, subagent): adds effective vocabulary at near-zero cap cost.
6. **Mini-GPTQ pre-quant calibration** (~50-100 lines, subagent): bake per-row scale optimization into weights to lower quant_tax. Highest payoff if the artifact-cap/quant-tax is the real bottleneck.

**Lower-EV (defer until above are done):**
7. K=3 L=4 (more loops at fixed params) — env-var only.
8. Re-test smear-gate with init -6 (off-by-default) — ~5-line change.
9. Sliding-window attention in TRAINING (banded mask) — untested by records.
10. **Why does init=0.05 produce step-1 train_loss=20.7?** [WORTH_DERIVING]
11. **Why does batch=32k mode-collapse even at MATRIX_LR=0.03?** [WORTH_DERIVING]

**Forbidden until items 1-2 done**: pushing the mlp-width axis further. mlp=11 is the cap edge; going beyond requires a code-level cap-saving change (GPTQ or similar) that should be done as a separate axis, not as more width-axis tuning.

## Entries (newest first)
