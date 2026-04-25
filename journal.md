# Journal

## Current threads

- **Submittable best**: **2.10275 (exp 0057, `winners/2026-04-25_recur_3x3_swiglu_mlp3`)**. SEED=1337. SEED=42 confirm (0058) gave 2.10579 — cross-seed Δ 0.003 (typical), mean **2.10427**, **mean Δ vs 0051 = +0.0055** (just over +0.005 protocol boundary, both seeds positive). Cumulative gain vs canonical 2.5212 → **+0.418 (~16.6%)**. Code-level: K=3 unique blocks looped L=3 times (effective depth 9, no U-Net skips) + SwiGLU(mlp=3). Artifact 5.998 MB → **~10 MB cap headroom remaining for further compounds**.
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
