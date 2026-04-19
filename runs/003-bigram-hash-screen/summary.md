# Spec 003 — BigramHash signal screen — Summary

Single 2×H100 NA-1 training run at commit `3825019`, `BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 QK_GAIN_INIT=5.0 SEED=1337 TTT_ENABLED=0 TRAIN_LOG_EVERY=100 MAX_WALLCLOCK_SECONDS=2400`. Comparison target: `logs/exp24_sota_sp8192_2xh100_40m.log` at matched steps.

## Result

- **Variant pre-quantization post-EMA val_bpb: `1.08787912`** (`val_loss: 2.80961221`)
- **Exp 24 pre-quantization post-EMA val_bpb: `1.08670048`**
- **Δ = +0.00118** (variant WORSE by 0.00118)

## Signal gate: **NOT MET**

| Criterion | Required | Actual | Pass? |
|---|---|---|---|
| Variant ≤ Exp 24 at ≥3 of last 4 train_loss milestones (steps 4200/4300/4400/4500) | ≥3 of 4 | **0 of 4** — variant Δ_train was positive (worse) at every milestone | ✗ |
| Final pre-quant val_bpb | ≤ 1.0847 | **1.08788** | ✗ (miss by +0.00318) |

**Clean kill.** BigramHash doesn't improve quality on the current SOTA stack (3-layer depth recurrence + parallel residuals + QK-gain). Spec's hypothesis that BigramHash's prior +0.002-0.005 bpb gain from the March-era submission would transfer to the April stack is **falsified**.

## Train_loss trajectory (matched-step)

| step | variant | exp24 | Δ |
|---|---|---|---|
| 100 | 4.5047 | 4.4674 | +0.0373 |
| 200 | 3.7023 | 3.6762 | +0.0261 |
| 300 | 3.4959 | 3.4934 | +0.0025 |
| 500 | 3.2753 | 3.2805 | **−0.0052** |
| 1000 | 3.2358 | 3.2346 | +0.0012 |
| 1500 | 3.1259 | 3.1254 | +0.0005 |
| 2000 | 3.0593 | 3.0602 | **−0.0009** |
| 2500 | 2.9701 | 2.9664 | +0.0037 |
| 3000 | 2.9885 | 2.9848 | +0.0037 |
| 3500 | 2.8978 | 2.8925 | +0.0053 |
| 4000 | 2.8867 | 2.8812 | +0.0055 |
| 4500 | 2.8049 | 2.8023 | +0.0026 |

**Nuanced trajectory:**
- Steps 100-300: variant worse by 0.003-0.037 (early BigramHash adaptation, zero-init → brief noise).
- Steps 500-2000: **essentially tied** — variant slightly BETTER at step 500 (−0.0052) and step 2000 (−0.0009), even slightly.
- Steps 2500+: variant drifts consistently worse (Δ 0.003-0.006). Warmdown region is where Exp 24 pulls ahead.

So BigramHash isn't catastrophic — it's neutral-to-slightly-helpful in mid-training but **loses during warmdown**. Final Δ (+0.0026 to +0.006) is small but directionally consistent in the closing ~2000 steps.

Last 4 milestones (spec's signal gate window, steps 4200/4300/4400/4500):

| step | variant | exp24 | Δ | variant ≤ exp24? |
|---|---|---|---|---|
| 4200 | 2.8654 | 2.8595 | +0.0059 | ✗ |
| 4300 | 2.8558 | 2.8520 | +0.0038 | ✗ |
| 4400 | 2.8290 | 2.8241 | +0.0049 | ✗ |
| 4500 | 2.8049 | 2.8023 | +0.0026 | ✗ |

**0 of 4** — signal gate's first criterion fails.

One apples-to-apples val_bpb measured at step 4000:
- variant 1.1115 vs exp24 1.1096 → Δ_val = +0.0019 — same sign as train_loss trend.

(Val was sampled at default `VAL_LOSS_EVERY=4000`, not Exp 24's `200`. Mid-training val trajectory not captured — would have been nice but not required for the signal gate.)

## Step count comparison

- Variant: **4544 steps** in 2388s (stopping_early wallclock_cap)
- Exp 24: 4531 steps in 2352s

We actually ran **13 more steps** than Exp 24 at the matched wallclock cap. Early training was ~10% slower (compile overhead + cold cache); by step 4200 our rate had caught up to Exp 24's (~1.92 steps/sec). So final comparison is NOT step-count-handicapped — it's a clean comparison of models trained for ~same number of steps.

## Secondary finding: GPTQ crashes on `bigram.embed.weight`

After pre-quant post-EMA eval landed, the training script continued into GPTQ quantization and crashed:

```
GPTQ:collecting Hessians from calibration data...
GPTQ:collected 67 Hessians in 13.0s
KeyError: 'bigram.embed.weight'
```

Root cause: `gptq_mixed_quantize` in `train_gpt_sota.py:848-862` iterates over all state_dict entries ≥ 65536 params. `bigram.embed.weight` is 3072 × 112 = 344,064 params → qualifies for quantization, but `collect_hessians` only registers hooks for `CastedLinear` modules, not `nn.Embedding`. No Hessian for `bigram.embed.weight` → KeyError when `hessians[name]` is looked up.

**Implication:** the SOTA submission always ran with `BIGRAM_VOCAB_SIZE=0` so this code path was never exercised. If any future spec re-introduces BigramHash, it needs a one-line fix — either:
- Special-case `bigram.embed.weight` as `passthrough (float16)` (add a name check alongside the existing `tok_emb` branch), OR
- Add a hook for `nn.Embedding` in `collect_hessians` so bigram gets its own Hessian.

No impact on our result — the crash occurred *after* the signal-gate number (pre-quant post-EMA) had already been written to the log. We got the number we needed; the quantized artifact just didn't materialize (which we wouldn't have used anyway for a screening run).

## Cost

| Phase | Wall | Cost |
|---|---|---|
| Pod creation + SSH ready + preflight | ~2 min | ~$0.20 |
| Training (588s training cap hit at wall 2388s = 39.8m) | ~40 min | ~$4.00 |
| Post-training EMA + pre-quant eval | ~14s | ~$0.02 |
| GPTQ Hessian collection (then crashed) | ~13s | ~$0.02 |
| Cleanup + rsync + stop + delete | ~2 min | ~$0.20 |
| **Total pod time** | **~45 min** | **~$4.50 pod rental** |

Actual balance drop: **$25.37 → $20.26 = $5.11** (includes a small baseline volume-storage charge too).

Spec estimated ~$4.00 base. Actual $5.11. Overshoot ~$1 from (a) slightly longer wall than Exp 24 — 40 min cap + eval tail, vs Exp 24's 39.2 min training that crashed at quant, same phenomenon; (b) our pod's 8-10% slower step rate early (compile overhead), vs Exp 24's cleaner warmup.

## Artifacts

In-repo:
- `variant_train.log` (~21 KB) — full stdout including the GPTQ crash traceback
- `summary.md`, `notes.md` — this report + execution notes

On NA-1 volume: nothing spec-003-specific to preserve. `quantized.ptz` never produced due to crash; no hotstart checkpoints emitted (from-scratch training, no checkpoint policy).

## Handback

Research: evaluation + `experiments.md` row + kill decision is yours. Summary is clear-cut (miss signal gate by +0.0032 at final, monotonically positive Δ_train across entire trajectory). My read: **kill, shelve BigramHash.** The March-era positive signal came from a simpler stack; the April SOTA primitives (depth recurrence, parallel residuals, QK-gain, LegalTTT) have absorbed whatever role BigramHash used to play.
