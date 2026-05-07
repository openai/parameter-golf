# Depth Recurrence in Parameter Golf — Research Summary

Ivan Verbovoy (@iverbovoy) · 20.03.2026 → 20.04.2026

## TL;DR

Single-person submission exploring **depth recurrence** (3 shared transformer blocks × 4 repeats = 12 effective layers) as an alternative to the flat 10-11 layer architectures used by the leaderboard. Best result: **val_bpb 1.1324 (3-seed mean)** on the 10-min track (PR [#1453](https://github.com/openai/parameter-golf/pull/1453)). Additional **4-hour non-record 1.0889** (PR [#895](https://github.com/openai/parameter-golf/pull/895)). OpenAI-acknowledged the approach as novel and published a dedicated non-record PR [#363](https://github.com/openai/parameter-golf/pull/363) inspired by similar exploration.

## Architecture

```
tok_emb (+ optional BigramHash) + value_embeds × 2
  │
  for repeat in {0..3}:
    for block in {A, B, C}:            # 3 shared blocks
      x += loop_embed[layer_idx]        # per effective layer
      x += Σ value_scales[l,e] * ve_e   # per effective layer
      x += cross_repeat_scale * block_out_prev_repeat  # stateful recurrence
      x = block(x, x0, use_xsa=(layer_idx ≥ xsa_start))
  final_norm + tied LM head + softcap
```

Key weight-sharing components:
- **loop_embed** `(effective_depth, model_dim)` — positional signal per effective layer
- **cross_repeat_scales** `(num_blocks, num_repeats-1, dim)` — stateful residual from prev repeat
- **resid_mix** — learned per-dim mix between current and block-0 residual
- **XSA** — last 4 effective layers subtract self-value projection
- **Hedge Mixer** — eval-time online mixture of Neural + Unigram + Bigram + Trigram(hash 65K) + Entropy experts

## Progression

| Date | PR | Track | Key idea | val_bpb |
|:----:|:--:|:-----:|:---------|--------:|
| 20.03 | [#148](https://github.com/openai/parameter-golf/pull/148) | 10min | Depth Recurrence + Cross-Repeat Skip | 1.2196 |
| 25.03 | [#784](https://github.com/openai/parameter-golf/pull/784) | 10min | + XSA(4) + LeakyReLU²(0.5) | 1.2065 |
| 26.03 | [#835](https://github.com/openai/parameter-golf/pull/835) | 10min | + Progressive Depth (2→3→4 repeats) | 1.1980 |
| 26.03 | [#856](https://github.com/openai/parameter-golf/pull/856) | 10min | + Hedge Mixer | 1.1454 |
| 26.03 | **[#895](https://github.com/openai/parameter-golf/pull/895)** | 4h | 4-hour Progressive Depth | **1.0889** |
| 05.04 | [#1384](https://github.com/openai/parameter-golf/pull/1384) | 10min | + tuned schedule + WD + SWA (3-seed) | 1.1441 |
| 07.04 | **[#1453](https://github.com/openai/parameter-golf/pull/1453)** | 10min | + **Int7 attn + Int5 MLP mixed quant** (3-seed) | **1.1324** |

## Experiments catalog

### What worked (baseline 1.1324)

| Technique | Effect | Notes |
|-----------|-------:|:------|
| Depth Recurrence 3×4 | — | Core architecture, enables 23.7M params in 16MB |
| Cross-Repeat Skip | −0.03 | Prev-repeat residual makes recurrence stateful |
| Value embeds (2 tables) | −0.07 | Critical. Adds per-layer token lookup |
| XSA last 4 | −0.01 | Self-value bias removal at top layers |
| Progressive Depth (0.30:2, 0.50:3, 1.0:4) | −0.005 | Ramp repeats during training |
| SWA (start 0.6, every 30) | −0.01 | ~44 checkpoints averaged |
| Hedge Mixer (5 experts) | −0.05 | Eval-time mixture, but stochastic (std 0.013) |
| **Int7 attn + Int5 MLP mixed quant** | −0.012 | Frees 2MB for d=880 mlp×3 vs d=832 mlp×2 |
| Muon optimizer + WD=0.04 | — | Standard for challenge |

### What did NOT improve mean 1.1324

Tested on 1–3 seeds and verified neither sliding nor hedge-mean improves:

| Technique | Result | Why |
|-----------|:------:|:----|
| BigramHash 2048×112 | −0.005 ❌ | Too few buckets, hash collisions dominate |
| BigramHash 3072×112 | +0.005 ❌ | Single-seed −0.003 but 3-seed mean worse: stabilizes hedge but cuts peaks (seed 7 went 1.1193→1.1444) |
| BigramHash 4096×112 | +0.004 ❌ | Past sweet spot, sparse buckets degrade |
| Noisy QAT (default) | +0.011 ❌ | Noise on int5 MLP too large (~amax/15), SWA collects pre-QAT checkpoints |
| LoRA rank-2 per-repeat (attn.proj, mlp.proj) | +0.013 ❌ | Per-repeat signal already saturated by loop_embed + cross_repeat_scales |
| XSA-all (12 layers) | worse | Optimum is last 4, early XSA hurts |
| Inter-repeat RMSNorm | worse | Breaks scaling balance |
| EMA (τ=0.997) | +22ms/step | CPU overhead > benefit at our scale |
| Partial RoPE + VRL + LN Scale (combined) | worse | Too many interacting changes |
| MuonEq-R optimizer | diverged | Incompatible with our Muon setup |
| Auxiliary losses (edge-of-chaos regularization) | neutral | χ stabilized but bpb unchanged at 5 repeats |
| 3×6 d=960 | worse | Fewer steps dominates |
| 6×2 d=640/736 | worse | Too narrow |
| 4L × 3rep | worse | Fewer unique blocks in limited compute |
| TTT (LoRA-based) | −0.002 | Positive but 410s eval; dropped for budget |
| SD-clip k=3.5, k=10 | worse | Percentile-search already near optimum for int8 |

### GPTQ with Hessian error compensation (3-seed validated)

Implemented column-wise GPTQ with training-data calibration (no access to val). Collects `X^T X` per `nn.Linear` over 5 training batches, then column-by-column quantization with Cholesky(H_inv) error compensation. ~100 lines added to 1496-line submission.

| Seed | roundtrip Δ | sliding Δ | hedge Δ |
|------|------------:|----------:|--------:|
| 1337 | −0.0034 | −0.0033 | +0.008 |
| 42 | −0.0007 | −0.0008 | −0.0006 |
| 7 | −0.0013 | −0.0013 | +0.023 |
| **3-seed mean** | **−0.0018** | **−0.0018** | **+0.010** |

**Deterministic improvement** on sliding/roundtrip (both −0.002). **Hedge mean worse by +0.010** — submission #1453's seed 7 hedge was unusually low (1.1193) and we couldn't reproduce that luck in our session.

Implication: GPTQ makes the model genuinely better (sliding/roundtrip = deterministic metric of model quality), but `val_bpb` is scored on hedge which has ±0.013 seed variance + ±0.008 session variance. The model-level gain gets dominated by hedge stochasticity.

Not submitting GPTQ as replacement — #1453 remains the best hedge-mean result. GPTQ-enhanced code kept as reference.

## Key insights

### 1. Depth recurrence is viable but not SOTA for this challenge

Our 1.1324 (3-seed) vs SOTA 1.1147 (abaybektursun's flat 11×512 + AR Self-Gen GPTQ + BigramHash 3072×112). Gap ~0.018. Evangelinehelsinki's separate exploration found flat 11L beats 3×3 recurrence by ~0.025 at same trick stack. **Recurrence trades unique parameters for effective depth**, which helps fit 23.7M params in 16MB but underperforms flat architecture per-layer.

### 2. Hedge Mixer dominates and destabilizes

Hedge gives ~−0.05 bpb lift over sliding but has huge variance:
- **±0.013 bpb between seeds** (same config)
- **±0.008 bpb between sessions** at identical model weights (sanity-run confirmed roundtrip/sliding match to 0.0002, hedge diverged 0.008)

Most architectural gains get absorbed by hedge noise. Deterministic metrics (sliding, roundtrip) are the reliable signal.

### 3. Weight-sharing saturates quickly

On 3×4 recurrence:
- loop_embed + cross_repeat_scales + value_scales already provide per-repeat variance
- LoRA per-repeat on top **hurt** (+0.006 sliding) — the model was already using available capacity
- Inter-repeat RMSNorm also hurt

Additional per-repeat degrees of freedom have diminishing/negative returns.

### 4. Progressive Depth schedule matters

Shifting schedule from (0.40:2, 0.65:3, 1.0:4) to **(0.30:2, 0.50:3, 1.0:4)** gave −0.004 bpb — 55% more full-depth training steps. Combined with longer warmdown (3000 vs 2000) and denser SWA (every 30 vs 50) at higher start frac (0.6 vs 0.4) for ~44 averaged checkpoints.

### 5. Mixed quantization > uniform

Separating attn (int7, 63 levels) from MLP (int5, 16 levels):
- Attention quality drop dominates total loss at low precision → keep attn higher
- MLP tolerates aggressive quantization → allows 2MB saving
- 2MB saved → model width up from d=832 mlp×2 → d=880 mlp×3

Gain: −0.012 bpb.

### 6. Calibration data makes GPTQ work

Original percentile-search GPTQ ("GPTQ-lite" in our code) only optimizes per-row clip point via MSE. Full GPTQ with column-wise Hessian error compensation gave deterministic −0.002..−0.003 on sliding. Training-data calibration worked; AR self-gen calibration would likely stabilize further.

## Files

- Main submission: `records/track_non_record_16mb/2026-04-08_DepthRecurrence_Int7MixedQuant_HedgeMixer/` (PR #1453 backing)
- 4-hour submission: PR #895
- Experimental code variants in repo root: `train_gpt_refactored.py`, `train_gpt_exp1.py`, etc.

## Reproduction

Config used for PR #1453 (submitted):
```
MODEL_DIM=880 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
NUM_LAYERS=3 NUM_REPEATS=4
QUANT_LEVELS=63 MLP_QUANT_LEVELS=16
PROG_DEPTH="0.30:2,0.50:3,1.0:4"
WARMDOWN_ITERS=3000
SWA_START_FRAC=0.6 SWA_EVERY=30
MATRIX_LR=0.018 MUON_WD=0.04
XSA_LAST_N=4 QK_GAIN_INIT=1.5
USE_HEDGE=1 HEDGE_ETA=0.1
MAX_WALLCLOCK_SECONDS=600
```

3 seeds tested (1337, 42, 7) on 8× H100 SXM 80GB, PyTorch 2.5.1.

## Resource footprint

- RunPod compute grant: ~$950 of $1000 used
- ~25 full training runs + calibration experiments
- 1 person, 32 days

## Acknowledgments

Thanks to OpenAI for running this challenge and sponsoring the compute grant. Thanks to **abaybektursun**, **thwu1**, **Raahil Shah**, **Evangelinehelsinki** for publishing detailed submissions that informed several of my experiments (particularly GPTQ calibration, BigramHash sizing, and the noisy-QAT analysis for recurrent architectures).
