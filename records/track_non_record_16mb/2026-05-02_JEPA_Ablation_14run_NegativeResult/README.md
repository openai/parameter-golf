# JEPA-on-LM 14-run ablation — non-record submission (2026-05-02)

This is a **non-record submission documenting a comprehensive negative result**:
JEPA auxiliary objectives do **not** improve `val_bpb` on parameter-golf at
the 17.06M-param / sp1024 / FineWeb scale. The cleanest recipe ties
baseline exactly. We submit this to formalize the negative finding so
future JEPA submitters don't re-run the same grid.

## TL;DR

- **Best JEPA variant** (`jepa-var-zero`, α=0.001, `VAR_WEIGHT=0`):
  `val_bpb = 1.2311` at step 50K — **exact tie with same-seed baseline**.
- Same-seed JEPA-vs-baseline gap: **+0.0007 to +0.0009** across two seeds
  (1337, 42).
- Cross-seed baseline variance: **0.0022**, larger than the JEPA gap →
  statistically indistinguishable.
- λ matters by orders of magnitude. λ=0.001 = parity. λ=0.005 ≥ +0.005 BPB
  cost. λ=0.2 (the obvious "JEPA paper" default) costs +0.018 BPB.

## Track

`non-record-unlimited-compute-16mb` — but **the model artifact was not
quantized for this submission**. We're submitting an ablation finding,
not a leaderboard candidate. The val_bpb reported is the pre-quant
running val_bpb at step 50K.

## Setup

All variants share one architectural backbone:

- **Backbone**: `BaselineGPT`, 17,059,912 params
- **Layers**: `NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- **Activation**: `relu_sq`
- **Tied embeddings**: `TIE_EMBEDDINGS=1`
- **Tokenizer/data**: `sp1024` BPE on FineWeb 10B
- **Batch**: `TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024`
- **Optimizer**: Muon (matrices) + Adam (scalars) — parameter-golf default
- **Schedule**: linear warmdown, 1200 step warmdown, 10-step warmup
- **Validation**: `VAL_LOSS_EVERY=10000`

JEPA variants add a **single** small predictor MLP (model_dim → 64 →
model_dim, zero-init on output) totaling **65,536 params (+0.4%)**:

- **JEPA total**: 17,125,448 params
- All 14 runs use the **same** model dim/layers/heads — only loss weights
  and JEPA env vars differ. **Param-count clean**.

## What we tested (14-run grid)

Final `val_bpb` at step 50K, sorted ascending. Star (*) = wallclock cap
hit on slower hardware before step 50K; `step` column shows actual.

| run | seed | config | step | **val_bpb** | Δ vs same-seed baseline |
|---|---|---|---|---|---|
| `baseline-seed42`     | 42   | control                                | 50K | **1.2289** | 0 (own baseline) |
| `tiny-lambda-seed42`  | 42   | α=0.001                                 | 50K | 1.2298     | +0.0009 |
| **`var-zero`**        | 1337 | **α=0.001, VAR_WEIGHT=0**               | 50K | **1.2311** | **0.0000 ✅ TIE** |
| `baseline-promo`      | 1337 | control                                | 50K | 1.2311     | 0 (own baseline) |
| `tiny-lambda-v3`      | 1337 | α=0.001                                 | 50K | 1.2318     | +0.0007 |
| `half-lambda`         | 1337 | α=0.0005                                | 50K | 1.2318     | +0.0007 |
| `chunk16`             | 1337 | α=0.001, CHUNK=16                       | 50K | 1.2318     | +0.0007 |
| `aux+token-tiny`      | 1337 | α=β=0.001                               | 50K | 1.2361     | +0.0050 |
| `tenth-lambda`*       | 1337 | α=0.0001                                | 40K | 1.2362     | tied @ 40K |
| `covar-v3`            | 1337 | α=0.005, COVAR_WEIGHT=0.05              | 50K | 1.2374     | +0.0063 |
| `token-only-tiny`*    | 1337 | β=0.001                                 | 40K | 1.2408     | +0.0046 (40K) |
| `injection-v2`*       | 1337 | α=0.005, INJECTION=1                    | 40K | 1.2456     | +0.0094 (40K) |
| `aux-v1`              | 1337 | α=0.2 (the "JEPA paper" default)        | 50K | 1.2492     | +0.0181 |
| `aux-low-v2`*         | 1337 | α=0.005                                 | 30K | 1.2553     | +0.0060 (30K) |

(Cross-seed baseline gap = 1.2311 − 1.2289 = **0.0022**, our noise floor.)

## Component-by-component verdict at the whisper regime (λ=0.001)

| component active | effect on val_bpb @ 50K |
|---|---|
| Path A MSE alone (VAR_WEIGHT=0)                    | **0.000** ← exact baseline |
| Path A + VICReg variance reg (VAR_WEIGHT=0.1)      | +0.0007 (within seed noise) |
| Path A + V-JEPA off-diag covariance (COVAR=0.05)   | +0.0063 |
| Path B (token decoder via tied LM head) alone      | +0.0046 |
| Path A + Path B both at whisper                    | +0.0050 |
| Path A + injection (zero-init latent into hidden)  | +0.0094 |
| Higher λ: 0.005                                    | +0.005 to +0.010 |
| Higher λ: 0.2                                      | +0.018 (catastrophic, v1 default) |

## Three findings

1. **λ matters most, by orders of magnitude.** PR #832 (winner pattern)
   used λ=0.001. We confirm parity at that magnitude. Going to λ=0.005
   already costs ≥0.005 BPB. λ=0.2 (a common JEPA paper default) costs
   0.018 BPB. This is the single most consequential knob.

2. **VICReg variance reg adds small harm at this λ.** With λ already at
   the noise floor, the variance hinge `relu(1 - z_std)` injects a tiny
   asymmetric force that nudges JEPA away from baseline. Setting
   `VAR_WEIGHT=0` recovers exact parity (`var-zero` row above).

3. **Path B (token-decoder JEPA) hurts even at β=0.001.** The JEPA
   token-CE competes with main CE for the tied LM head, so even whisper
   magnitudes pull the head in two directions. Path A (hidden-state aux
   MSE) is benign at small λ because it doesn't touch the LM head.

## Reproducibility

- **Architecture**: `jepa_lm.py` (this directory) — also published in the
  `crucible-community-tap` at
  [`architectures/jepa_lm/`](https://github.com/eren23/crucible-community-tap/tree/main/architectures/jepa_lm).
  Tap commit `bc93273`.
- **Training script**: `train_gpt.py` (this directory) is a thin
  compatibility wrapper that delegates to
  `src/crucible/training/torch_backend.py` from the
  [Crucible](https://github.com/eren23/crucible) ML platform (commit `969cac5`).
- **Compute**: 4× RunPod RTX 4090 (3 dedicated + 1 shared overnight). All
  variants ran the `promotion` preset (~2h wallclock,
  `MAX_WALLCLOCK_SECONDS=7200`, target `ITERATIONS=100000`, 65,536
  `TRAIN_BATCH_TOKENS`).
- **Total cost**: ~$15 over ~16 GPU-hours.
- **W&B**: project `parameter-golf`, entity `eren23`. Run names match the
  table above (e.g. https://wandb.ai/eren23/parameter-golf/runs/n22iw31q
  for `var-zero`).
- **Full ablation finding** (per-step val_bpb curves CSV, structured
  finding doc): `crucible-community-tap` at
  [`findings/parameter-golf-jepa-ablation/`](https://github.com/eren23/crucible-community-tap/tree/main/findings/parameter-golf-jepa-ablation).

### Repro command (var-zero, the baseline-tying recipe)

```bash
# Install the JEPA tap plugin
crucible tap add https://github.com/eren23/crucible-community-tap
crucible tap install jepa_lm --type architectures

# Run var-zero
MODEL_FAMILY=jepa_lm \
JEPA_ALPHA=0.001 \
JEPA_BETA=0 \
JEPA_VAR_WEIGHT=0 \
JEPA_COVAR_WEIGHT=0 \
JEPA_CHUNK=8 \
JEPA_PREDICTOR_DIM=64 \
JEPA_INJECTION=0 \
SEED=1337 \
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
ACTIVATION=relu_sq TIE_EMBEDDINGS=1 \
TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 \
ITERATIONS=100000 WARMUP_STEPS=10 WARMDOWN_ITERS=1200 \
VAL_LOSS_EVERY=10000 \
MAX_WALLCLOCK_SECONDS=7200 \
PYTHONPATH=src python -m crucible.cli.main run experiment --preset promotion
```

## Why this is publishable as a non-record submission

- 14 runs at the same N (17.06M / 17.13M with predictor), promotion-tier
  budget each (~2h wallclock, 50K steps).
- Two-seed paired baselines (1337, 42) establish a 0.0022 noise floor —
  roughly **2.5× larger than any JEPA-vs-baseline gap we measured at
  the cleanest configs**.
- λ sweep across 4 orders of magnitude (0.0001, 0.0005, 0.001, 0.005, 0.2).
- Path ablation (A only / B only / both / injection / covar).
- Three previously untested knobs added: `chunk16`, `var-zero`, `tenth-lambda`.

The cleanest negative-result JEPA submission on parameter-golf to date.
PR #896 was a single-config failure; this is a saturated grid that
identifies *exactly* which JEPA components hurt and which is benign.

## Files

- `README.md` — this file
- `submission.json` — leaderboard metadata (track, val_bpb, ablation JSON)
- `train.log` — full training stdout for the best-variant `jepa-var-zero` run
- `jepa_lm.py` — the architecture plugin (also in `crucible-community-tap`)
- `train_gpt.py` — entry-point shim for the Crucible torch backend

## Next directions (not yet tested)

1. **Span-masking** (PR #1581 approach): replace target tokens with a
   learned mask in the context-encoder pass. Forces non-trivial
   prediction. Requires double forward pass — implementation cost is real.
2. **Phased α ramp**: pure AR (30%) → AR+JEPA ramp (50%) → pure AR
   cooldown (20%). PR #832 schedule.
3. **EMA target encoder** (BYOL-style). PR #896 already showed no-gain
   at this scale, deprioritized.
4. **Different backbone scale**: PR #832 won at 24M / byte-level. Maybe
   JEPA helps below 17M but hurts above. Untested here.
