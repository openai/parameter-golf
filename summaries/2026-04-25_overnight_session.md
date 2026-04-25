# Overnight Session Summary — 2026-04-25

54 experiments. **Cumulative gain vs canonical baseline (val_bpb 2.5212 → 2.10971): +0.412 (≈16.4%)**.

The submittable best is exp 0051 — kept under the 16 MB artifact cap. SwiGLU at MLP_MULT=3 (exp 0044) hit val_bpb 2.11489 but exceeded the cap; recorded as informative non-submittable.

Results.tsv has every row with status / description. `journal.md` has narrative entries; this file links experiment paths and journal headings together for future-agent navigation.

---

## Final winner — submittable

**Path**: `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_24k_matrix_lr_045_init_05_muon_backend_15/`
(corresponds to exp `experiments/0051_muon_backend_steps_15/`)

val_bpb_post_quant = **2.10971** (single-seed, SEED=1337). Artifact 15.18 MB.

Env-var deltas vs canonical (no code changes):
```
LR_WARMUP_STEPS=30          # canonical 0
WARMDOWN_ITERS=300          # canonical 1200
MLP_MULT=4                  # canonical 2
TRAIN_BATCH_TOKENS=24576    # canonical 8192 → also bumped VAL_BATCH not needed at seq=1024
MATRIX_LR=0.045             # canonical 0.04
TIED_EMBED_INIT_STD=0.05    # canonical 0.005
MUON_BACKEND_STEPS=15       # canonical 5
```

Everything else is canonical default. See the env.sh inside the winner folder for the exact set.

---

## Stack of confirmed wins (in order they were promoted)

Each row links the experiment folder, the SEED=42 confirm where present, and the matching journal section heading.

| # | Lever | Δ post-quant | Tag | Promoted as | SEED=42 confirm | Journal heading |
|---|---|---|---|---|---|---|
| 1 | Schedule rewrite (WARMDOWN_ITERS=600 + LR_WARMUP_STEPS=10) | +0.116 | low | `winners/2026-04-25_warmdown_600_warmup_10/` (exp 0005) | exp 0006 | `## 2026-04-25 · exp 0005 + 0006 · schedule rewrite — first big win (Δ +0.116)` |
| 2 | MLP_MULT 2 → 4 | +0.014 | high | `winners/2026-04-25_warmdown_600_warmup_10_mlp_mult_4/` (exp 0008) | none — direct promote | `## 2026-04-25 · exp 0007 + 0008 · capacity scaling on the new schedule (mlp2 → mlp4)` |
| 3 | TRAIN_BATCH_TOKENS 8192 → 16384 (seq stays 1024) | +0.082 | high | `winners/2026-04-25_warmdown_600_warmup_10_mlp_mult_4_batch_16k/` (exp 0013) | exp 0014 | `## 2026-04-25 · exp 0013 + 0014 · batch=16384 is the real lever; seq=2048 hurts` |
| 4 | Schedule push (WARMDOWN_ITERS=400 + LR_WARMUP_STEPS=20) | +0.055 | low | `winners/2026-04-25_warmdown_400_warmup_20_mlp_mult_4_batch_16k/` (exp 0015) | exp 0016 | `## 2026-04-25 · exp 0015 + 0016 · schedule push (WARMDOWN_400 + WARMUP_20) lands +0.055` |
| 5 | Schedule push (WARMDOWN_ITERS=300 + LR_WARMUP_STEPS=30) | +0.029 | low | `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_16k/` (exp 0020) | none — direct promote | (covered in `## 2026-04-25 · exp 0017 + 0018 + 0019 · sentinel clean; batch=32k mode-collapses; depth ceiling` neighborhood; 0020 promote in commit `bda13c6`) |
| 6 | MATRIX_LR 0.04 → 0.06 | +0.016 | med | `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_16k_matrix_lr_06/` (exp 0021) | none — direct promote | (commit `7f68d70`; supplemented by 0022 result) |
| 7 | TIED_EMBED_INIT_STD 0.005 → 0.02 (4×) | +0.011 | high | `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_16k_matrix_lr_06_init_02/` (exp 0023) | none — just over noise floor | (commit `e5731e3`) |
| 8 | TIED_EMBED_INIT_STD 0.02 → 0.05 (10× canonical) | +0.027 | high | `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_16k_matrix_lr_06_init_05/` (exp 0024) | none — direct promote | (commit `191aaa7`) |
| 9 | TRAIN_BATCH_TOKENS 16k → 24k + MATRIX_LR 0.06 → 0.045 | +0.045* | high | `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_24k_matrix_lr_045_init_05/` (exp 0036) | exp 0047 (mean Δ +0.038, cross-seed Δ 0.014 — high variance) | `## 2026-04-25 · exp 0036 · batch=24k + LR=0.045 wins +0.045 — refutes 0018 batch ceiling` and `## 2026-04-25 · exp 0047 · SEED=42 of 0036 — winner real but cross-seed variance higher than typical` |
| 10 | MUON_BACKEND_STEPS 5 → 10 | +0.0066 | med | `winners/.../muon_backend_10/` (exp 0049) | exp 0050 | `## 2026-04-25 · exp 0049 + 0050 · MUON_BACKEND_STEPS 5→10 lands marginal +0.0066` |
| 11 | MUON_BACKEND_STEPS 10 → 15 | +0.013 | med | `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_24k_matrix_lr_045_init_05_muon_backend_15/` (exp 0051, **final winner**) | none — direct promote, but 0050 already showed cross-seed reproducibility for the +5 step | `## 2026-04-25 · exp 0051 + 0052 · MUON_BACKEND_STEPS=15 wins clean +0.013, then saturates` |

\*Single-seed Δ. SEED=42 confirmation (0047) gave mean Δ +0.038. The +0.045 figure should be read as upper bound.

---

## Cross-experiment lessons (each tied to a journal entry)

These are the narrative findings worth carrying into the next session. The full reasoning is in `journal.md` under the listed headings.

1. **The canonical schedule was the dominant under-training factor at 200 steps.** WARMDOWN_ITERS=1200 produced effective avg lr_mul=0.083 — the model trained at ~8% of the design LR throughout. Three confirmed schedule pushes added up to +0.200 cumulative.
   - Journal: `## 2026-04-25 · exp 0005 + 0006 · schedule rewrite — first big win (Δ +0.116)`, `## 2026-04-25 · exp 0015 + 0016 · schedule push (WARMDOWN_400 + WARMUP_20) lands +0.055`
   - Crucial trajectory observation: at LR_WARMUP_STEPS=10 the lr_mul=1.0 peak hits while cumulative-LR is still ramping → step-9 train_loss spike to ~7.06 (above step 1's 6.94). At LR_WARMUP_STEPS=20 the peak lands on a more stable basin, no spike.

2. **Schedule-masking can hide real positives AND negatives.** QK_GAIN_INIT=5 looked like noise (+0.002) under the canonical schedule (exp 0003) but actively HURT (-0.028) on the proper schedule (exp 0011).
   - Journal: `## 2026-04-25 · exp 0009 + 0010 + 0011 + 0012 · capacity ceiling, qk_gain reversal, seq+batch win`
   - Implication: pre-promote architectural ablations should always re-test on the current winner schedule. The 0002 (MLP_MULT) and 0003 (QK_GAIN) discards under canonical schedule were not free-lunch information — they had to be re-evaluated.

3. **Capacity at 200 steps caps quickly.** MLP_MULT=4 wins (+0.014 in 0008), MLP_MULT=5 plateaus (refuted by SEED=42 in 0010), NUM_LAYERS=11 plateaus (0019). Records using these at H100 20k-step don't transfer to the 200-step smoke.
   - Journal: `## 2026-04-25 · exp 0007 + 0008 · capacity scaling on the new schedule (mlp2 → mlp4)`, `## 2026-04-25 · exp 0017 + 0018 + 0019 · sentinel clean; batch=32k mode-collapses; depth ceiling`
   - Failure pattern: low quant_tax + zero pre-quant Δ. New params end up small/structureless with insufficient training budget.

4. **Batch and LR are coupled, not separate axes.** batch=32k+MATRIX_LR=0.06 mode-collapses (exp 0018, train_loss 0.55 with val_loss 4.43); batch=24k+MATRIX_LR=0.045 wins +0.045 (exp 0036). The "32k ceiling" was an LR-coupling failure, not a batch ceiling. But batch=32k still failed at MATRIX_LR=0.03 (exp 0037) — cause unclear; possibly the 4-sequences-per-microstep configuration loses critical stochasticity.
   - Journal: `## 2026-04-25 · exp 0017 + 0018 + 0019 · sentinel clean; batch=32k mode-collapses; depth ceiling` and `## 2026-04-25 · exp 0036 · batch=24k + LR=0.045 wins +0.045 — refutes 0018 batch ceiling`

5. **Init scale 0.005 → 0.05 was the most surprising hidden bug.** Canonical TIED_EMBED_INIT_STD severely under-initialized embeddings; bigger init gave +0.038 across two experiments. Optimum is precise: 0.05 wins (0024), 0.07 hurts -0.058 (0039), 0.04 ties (0040, noise), 0.1 catastrophic -0.096 (0025).
   - Journal: see promote commits `e5731e3` (0023), `191aaa7` (0024); 0025-0030 ceiling sweep.

6. **MUON's Newton-Schulz iterations were under-converged at canonical 5.** Sweep MUON_BACKEND_STEPS ∈ {5, 10, 12, 15, 20}: optimum at 15 (val 2.10971), then saturates at 20 (val 2.118). Found *after* declaring env-var space exhausted (around exp 0040) — reminder that "saturation" is sometimes premature.
   - Journal: `## 2026-04-25 · exp 0049 + 0050 · MUON_BACKEND_STEPS 5→10 lands marginal +0.0066`, `## 2026-04-25 · exp 0051 + 0052 · MUON_BACKEND_STEPS=15 wins clean +0.013, then saturates`

7. **Most env-var axes are genuinely flat at this regime.** Of ~25 unique knobs explored, only 7 produced robust wins. Dead axes (with paths for future-agent reference):
   - QK_GAIN: `experiments/0003_qk_gain_5/`, `experiments/0011_qk_gain_5_on_winner/` (actively hurts on proper schedule)
   - LOGIT_SOFTCAP: `experiments/0026_logit_softcap_15_on_winner/`
   - MUON_MOMENTUM: `experiments/0029_muon_momentum_09/`, `experiments/0030_muon_momentum_085/`
   - BETA1: `experiments/0034_beta1_095_on_winner/`
   - BETA2: `experiments/0032_beta2_099_on_winner/`, `experiments/0033_beta2_099_on_winner_seed42/`
   - ROPE_BASE: `experiments/0035_rope_base_20k_on_winner/`
   - GRAD_CLIP_NORM: `experiments/0038_grad_clip_1_on_winner/`
   - TIED_EMBED_LR scale-up (HURTS): `experiments/0022_tied_embed_lr_075_on_winner/`
   - SCALAR_LR scale-up (HURTS): `experiments/0031_scalar_lr_06_on_winner/`
   - TRAIN_SEQ_LEN=2048 (HURTS, decomposed by 0013): `experiments/0012_seq_len_2048_on_winner/`
   - TRAIN_BATCH_TOKENS=32768 (mode-collapse): `experiments/0018_batch_32k_on_winner/`, `experiments/0037_batch_32k_lr_03/`
   - NUM_LAYERS=11 (ceiling): `experiments/0019_num_layers_11_on_winner/`
   - ORTHO_INIT (code change, HURTS): `experiments/0041_ortho_init_on_winner/`
   - LeakyReLU² (code change, noise): `experiments/0048_leaky_relu2_mlp4/`

   Journal: `## 2026-04-25 · exp 0025-0030 · ceiling sweep — init/softcap/schedule/momentum`, `## 2026-04-25 · exp 0037-0040 · batch=32k still hard-fails; init/clip retest on 0036 confirms saturation`, `## 2026-04-25 · session summary · 43 experiments, +0.395 from canonical baseline, env-var search exhausted`

8. **Quant_tax is a useful sanity signal.** Anomalously low quant_tax (≤0.001) repeatedly correlated with mode-collapse-like degeneracy or freak-run gains that didn't reproduce: 0009 (mlp5 false win), 0018 (batch=32k mass-collapse), 0019 (11L tiny gain that's actually quant-tax artifact), 0032 (BETA2 freak), 0044 (SwiGLU info-only). Healthy runs have quant_tax 0.002–0.005.

---

## Important caveats / methodology

- **Single-seed promotions may overstate Δ by ~10–20%.** The 0036 promotion claimed +0.045 from a single seed; SEED=42 confirm in 0047 gave mean Δ +0.038 (cross-seed Δ 0.014, about 5× the typical 0.0024 we saw on other configs). Cross-seed-confirmed wins (0005/0006, 0013/0014, 0015/0016, 0049/0050) are the most reliable. Single-seed direct-promotes (0008, 0021, 0023, 0024, 0036, 0051) likely have ±0.005 error bars.
- **Cross-seed variance baseline**: 0.0024–0.0027 for typical configs. Larger Δ between seeds (0.008–0.015) is a marker of an outlier run, not the true effect.
- **Regression sentinel** (exp 0017, `experiments/0017_regression_check_001/`): bit-exact reproduction of the 0001 baseline (val_bpb 2.52115777). Harness drift is zero.
- **Best non-submittable**: exp `experiments/0044_swiglu_mlp_3/` hit val_bpb 2.11489 at artifact 16.46 MB → SIZE_VIOLATION. SwiGLU(mlp_mult=3) genuinely improves training (+0.011 vs 0036) but doesn't fit cap. SwiGLU(mlp_mult=2) at `experiments/0045_swiglu_mlp_2/` fits cap (12.7 MB) but loses the gating advantage (Δ noise). Lower-LR variant `experiments/0046_swiglu_mlp_3_lr_035/` fits cap at 15.7 MB but the LR drop kills the gain. SwiGLU's benefit not extractable here without other compensating architecture changes (e.g. fewer layers).

---

## Recommendations for H100 20k-step transfer

The **[transfer:high]** changes should hold:

1. `TIED_EMBED_INIT_STD=0.05` — likely the biggest single-line transfer. Canonical 0.005 is a real bug regardless of step count.
2. `MUON_BACKEND_STEPS=15` — Newton-Schulz convergence is a universal numerical property; 5 was under-converged at any scale.
3. `MLP_MULT=4` — capacity wins are robust; the records that train longer with mlp4 are at H100 scale.
4. **Don't extend `TRAIN_SEQ_LEN` beyond 1024 at this model size** — exp 0012 vs 0013 showed seq=2048 actively hurts at d=512.
5. **Bigger batch + proportionally lower LR** scales monotonically except at batch=32k (a regime change we didn't fully diagnose).

The schedule constants (WARMDOWN_ITERS=300, LR_WARMUP_STEPS=30, MATRIX_LR=0.045) are tuned for the 200-step compute-starved regime and **must be re-optimized at 20k steps** — they're [transfer:low].

QK_GAIN_INIT=5.0 helps in records using sp4096/sp8192 but actively HURTS at sp1024 (exp 0011). Don't carry that forward unless changing tokenizer.

---

## Forward-looking notes for next session

- **Env-var space is largely exhausted.** Remaining untested env-vars (ADAM_EPS, MUON_MOMENTUM_WARMUP_*) are unlikely to give big gains.
- **Code-level changes attempted**: ORTHO_INIT (0041, hurts), SwiGLU (0044/0045/0046, doesn't fit cap), LeakyReLU² (0048, noise). Untested but record-grounded: sliding-window attention, parallel residuals, smear-gate, depth-recurrence. These all need substantial code changes (>20 lines, multiple functions) — subagent territory per the protocol.
- **Highest-EV next experiment classes**: any code change that reduces parameter count to free up cap room for SwiGLU, OR sliding-window attention which records use heavily.
- **Could explore**: TRAIN_BATCH_TOKENS=20480 (20k) is invalid because 20480 / (8 × 1024) = 2.5 — non-integer sequences per micro-step. To get a smooth batch curve between 16k and 24k, would need to change `grad_accum_steps` (currently hardcoded `8 // world_size`). Also subagent territory.

---

## Experiment categorization

The 54 runs break down into a few different shapes:

### Multi-point sweeps (parameter scans, mapped a curve)

| Axis | Points covered | Outcome | Depth |
|---|---|---|---|
| **MLP_MULT** (capacity) | 2 (canonical), 3, 4, 5 | Optimum at 4. Mlp=5 SEED=42 disconfirmed (0010). | Medium — 4 points + SEED=42 confirm |
| **TIED_EMBED_INIT_STD** (init scale) | 0.005, 0.02, 0.04, 0.05, 0.07, 0.1 | Optimum precise at 0.05. Catastrophic at 0.1 (-0.096). | Deep — 6 points, peak well-bracketed |
| **WARMDOWN/WARMUP schedule** | (1200,0), (600,10), (400,20), (300,30), (250,35) | Diminishing returns: +0.116, +0.055, +0.029, +0.005 (plateau confirmed by SEED=42 in 0028) | Deep — 4 push levels, both seeds at boundary |
| **TRAIN_BATCH_TOKENS** | 8192 (canonical), 16384, 24576, 32768 (twice with different LR) | Optimum at 24k. 32k mode-collapses at every LR tested. | Medium — 4 points; 32k tested twice with different LR but didn't isolate the failure mode |
| **MATRIX_LR** | 0.04 (canonical), 0.045, 0.05, 0.06 | Optimum varies with batch — 0.06 at batch=16k, 0.045 at batch=24k. SEED=42 disconfirmed 0.05 at batch=24k. | Medium — multiple batch-LR pairs |
| **MUON_BACKEND_STEPS** | 5 (canonical), 10, 12, 15, 20 | Optimum at 15 (the most "buried" finding of the session). Saturates 15→20. | Deep — 5-point sweep, peak well-bracketed |
| **MUON_MOMENTUM** | 0.95 (canonical), 0.9, 0.85 | Both lower values gave noise-band Δ ~+0.003. Not a meaningful axis. | Shallow — 3 points, all flat |

### Single-axis ablations (one-shot tests of specific knobs)

| Axis | Result | Path |
|---|---|---|
| QK_GAIN_INIT=5.0 | Tested twice (canonical schedule + winner schedule). Noise on canonical, **hurts -0.028 on winner**. Schedule-masking lesson. | `experiments/0003_qk_gain_5/`, `experiments/0011_qk_gain_5_on_winner/` |
| LOGIT_SOFTCAP=15 (vs 30) | Hurts -0.044. Tighter cap clips signal. | `experiments/0026_logit_softcap_15_on_winner/` |
| BETA1=0.95 (Adam, vs 0.9) | Hurts -0.024. | `experiments/0034_beta1_095_on_winner/` |
| BETA2=0.99 (Adam, vs 0.95) | SEED=1337 freak (+0.010), refuted by SEED=42 (-0.006). | `experiments/0032_beta2_099_on_winner/`, `experiments/0033_beta2_099_on_winner_seed42/` |
| ROPE_BASE=20000 (vs 10000) | Noise (+0.0025). | `experiments/0035_rope_base_20k_on_winner/` |
| GRAD_CLIP_NORM=1.0 (vs disabled) | Noise (+0.003). | `experiments/0038_grad_clip_1_on_winner/` |
| SCALAR_LR=0.06 (vs 0.04) | Hurts -0.012. Small-param-count LRs are sensitive. | `experiments/0031_scalar_lr_06_on_winner/` |
| TIED_EMBED_LR=0.075 (vs 0.05) | Hurts -0.012. | `experiments/0022_tied_embed_lr_075_on_winner/` |
| NUM_LAYERS=11 (vs 9) | Noise (+0.003), tight against artifact cap. | `experiments/0019_num_layers_11_on_winner/` |
| TRAIN_SEQ_LEN=2048 (vs 1024) | Decomposed via 0013: seq=2048 alone HURTS by 0.059. | `experiments/0012_seq_len_2048_on_winner/` |

### Creative attempts / code-level changes (~30 lines or less)

These are the runs where I edited `train_gpt.py` inside the experiment folder rather than just env-vars. Per the protocol, all stayed under the 20-line "do it yourself" bar; subagent territory (sliding-window, parallel-resid, depth-recurrence) was not entered.

| Attempt | Code change | Outcome | Path | Verdict |
|---|---|---|---|---|
| **ORTHO_INIT** | Added `nn.init.orthogonal_` branch to `_init_weights` (~10 lines) gated by env-var | Hurts -0.022 | `experiments/0041_ortho_init_on_winner/` | Records use it but kaiming + Muon + ReLU² is the right combo here; one-shot tested, not pursued |
| **SwiGLU activation** | Replaced ReLU² MLP with `silu(gate) * up → down` (~15 lines), env-var-gated `MLP_TYPE=swiglu` | **Real +0.011 gain at MLP_MULT=3 BUT artifact 16.46 MB → size_violation, non-submittable**. MLP_MULT=2 fits but loses gain. Lower-LR variant fits but kills gain. | `experiments/0044_swiglu_mlp_3/` (size violation), `experiments/0045_swiglu_mlp_2/`, `experiments/0046_swiglu_mlp_3_lr_035/` | **Most interesting non-promoted finding.** SwiGLU does help here but doesn't fit cap at d=512/9L. Worth retrying with reduced layer count or other compensating changes. |
| **LeakyReLU²** | Trivial: `F.leaky_relu(x, 0.1)**2` instead of `relu(x).square()` | Noise (+0.001) | `experiments/0048_leaky_relu2_mlp4/` | Activation choice not a meaningful axis at this regime |

### Methodology / sanity-check experiments

| Purpose | Experiment | Result |
|---|---|---|
| Regression sentinel (canonical baseline rerun) | `experiments/0017_regression_check_001/` | Bit-exact match to 0001 baseline (val_bpb=2.52115777). Harness drift = 0. |
| SEED=42 confirms (separate from sweeps) | 0006, 0014, 0016, 0028, 0033, 0043, 0047, 0050 | Cross-seed variance baseline 0.0024–0.0027 for stable configs. Larger variance (0.008–0.015) is the signature of marginal/freak results. |

### Killed runs

| Experiment | Why | What was learned |
|---|---|---|
| `experiments/0004_warmdown_600/` | TaskStop'd at step 10 — step 2 train_loss spiked from 6.94 → 8.40 (LR overshoot from cold tok_emb at lr_mul=0.333 with no warmup) | Motivated 0005's WARMUP_STEPS=10. The fix was the journal's first-recorded "warmup is non-negotiable when raising effective LR." |

---

## How deeply I went into each axis (subjective)

- **Schedule (warmdown/warmup)**: Deep. 4 push levels with both seeds at the plateau boundary. Confident the optimum at 200 steps is around (warmdown=300, warmup=30). [transfer:low]
- **Capacity (MLP_MULT)**: Deep enough. Ceiling at 4 confirmed by SEED=42 disconfirm of mlp=5. Could test mlp=4.5 if non-integer multipliers were supported, but that's untestable as-is.
- **Init scale**: Deep. 6 points scanned, optimum well-bracketed at 0.05 (catastrophic at 0.1, hurts at 0.07, ties at 0.04, hurts at 0.02). High confidence.
- **Batch size**: Medium-deep. 4 points. **Open question**: why batch=32k specifically fails even at MATRIX_LR=0.03. Did NOT test batch=20480 (invalid — non-integer sequences per micro-step under hardcoded grad_accum_steps=8). Didn't try batch=28672 (28k) which would also be valid (28672/8=3584=3.5 × 1024 — also non-integer, skip).
- **Per-optimizer LRs**: Shallow on each. MATRIX_LR scanned at 4 points — confident peak around 0.045 at batch=24k. TIED_EMBED_LR and SCALAR_LR scaled UP only (and both hurt); did NOT test scaling them down (e.g. TIED_EMBED_LR=0.04, SCALAR_LR=0.03). Possible small wins there.
- **Optimizer betas (BETA1, BETA2)**: Shallow. Each tested in one direction only.
- **MUON optimizer internals**: BACKEND_STEPS deep (5-point sweep). MOMENTUM shallow (3 points, all flat). Did NOT touch MUON_MOMENTUM_WARMUP_START / MUON_MOMENTUM_WARMUP_STEPS.
- **Architecture (depth, width, head config)**: Shallow. NUM_LAYERS tested at 11 only (one-shot, ceiling). Never touched MODEL_DIM, NUM_HEADS, NUM_KV_HEADS — those are tied to the canonical 512/8/4 setting.

---

## What's "set in stone" vs still hypothesis

### Set in stone (verified, multi-evidence)

- **Cross-seed Δ baseline ≈ 0.0024 for stable configs.** Confirmed by 0005/0006, 0013/0014, 0015/0016, 0049/0050 all giving cross-seed Δ in [0.0024, 0.0027].
- **lr_mul formula at our settings**: `(iterations - step) / warmdown_iters` after warmup, exactly as I corrected the journal in commit `cdbd108`-ish. The 0001 trajectory bit-reproduces Apr-18, the 0017 sentinel bit-reproduces 0001.
- **batch=32k regime change is real, not just LR.** Replicated at MATRIX_LR=0.06 (0018) and MATRIX_LR=0.03 (0037). Both mode-collapsed.
- **MLP_MULT capacity ceiling at 4.** Verified by SEED=42 disconfirm of MLP=5 (0009 vs 0010).
- **TIED_EMBED_INIT_STD optimum precise at 0.05.** 6-point sweep brackets cleanly.
- **MUON_BACKEND_STEPS optimum at 15.** 5-point sweep brackets cleanly (5 < 10 < 12 < 15 > 20).
- **Schedule-masking effect on QK_GAIN.** Verified by direct comparison of 0003 (canonical schedule, noise) vs 0011 (winner schedule, hurts -0.028). Same change, same comparison axis, opposite signs once schedule is fixed.

### Still hypothesis (one-seed evidence or not directly tested)

- **0036 mean Δ vs 0024 ≈ +0.038** — single SEED=42 confirm (0047) had unusually high cross-seed variance (0.014). The win is real but exact magnitude is uncertain. A second SEED would tighten this.
- **MATRIX_LR at batch=24k is precisely 0.045.** 0.05 was tested with SEED=42 disconfirm; 0.04 with MUON=15 (0054) hurt. But fine-grain sweep (e.g. 0.042, 0.048) was not done. The 0.045 setting was inherited from 0036's "scale LR proportionally with batch from the 0008 baseline" reasoning.
- **MUON_BACKEND_STEPS=15 wins by exactly +0.013.** Direct-promoted from a single seed. Cross-seed confirm exists for the smaller MUON=10 step (0049/0050) but not for the 15 jump itself.
- **The "batch=32k specifically loses critical stochasticity at grad_accum=8 / 4-sequences-per-microstep"** explanation is [LIKELY] but not directly verified. Would need to either change grad_accum_steps (code change) or test batch=24576+grad_accum=4 to isolate the per-microstep-sequence-count effect.
- **SwiGLU genuinely helps but doesn't fit cap at this architecture.** One non-submittable run (0044). Direction is clear, magnitude uncertain. Different architecture (NUM_LAYERS=8 + SwiGLU?) might fit and would be a new winner candidate.

### Cool ideas that worked but I didn't extend

- **The schedule-masking framework** (lesson 2 above) was a [CONJECTURE] in the early-session journal. The 0011 result (qk_gain hurts on proper schedule) verified it as a one-off. I never went back and re-tested 0002 (MLP_MULT=3) on the proper schedule to see if the canonical-schedule "noise" also hid a real effect (could be positive given mlp=4 worked). That would be a cheap follow-up.
- **The LR-batch coupling theory** (lesson 4 above) was [CONJECTURE] from the 0018 mode-collapse analysis. Confirmed by 0036's success. But the Adam-variance-saturation explanation is [LIKELY] — the actual mechanism wasn't directly tested. Would need to read out gradient magnitudes / variances during a 32k vs 24k run.

### Cool ideas that didn't fully verify

- **SwiGLU non-submittable win.** Path: `experiments/0044_swiglu_mlp_3/`. val_bpb 2.11489 < current winner 2.10971's official, but artifact 16.46 MB > cap. The +0.011 gain over the relu² mlp4 baseline (0036) is real (pre/post Δ both positive, normal quant_tax). Not pursued because cap-fit variants (mlp=2 lost gating, lower-LR killed gain) didn't recover the win.
- **The 0019 NUM_LAYERS=11 noise result might have been schedule-masked too.** I never re-tested with a different schedule once the new winner was established. Per the schedule-masking lesson, this is a [CONJECTURE]-worthy revisit. But artifact at 11L was already tight (15.7 MB), and now at MUON_BACKEND_STEPS=15 our base artifact is 15.18 MB — adding 11L would exceed cap.

---

## Follow-ups for next session (ranked by EV)

1. **Verify 0051 with SEED=42** — the final winner promotion was direct (Δ +0.013 above noise floor with single seed). The session has shown that high-cross-seed-variance can occur at "winner" configs (cf. 0036/0047). One more SEED=42 run on the 0051 config would solidify the +0.412 cumulative claim. Cost: 1 experiment (~12 min).
2. **Re-test 0002 (MLP_MULT=3) on the 0051 winner schedule.** Schedule-masking framework predicts the canonical-schedule "noise" of +0.002 might have hidden a real effect. Cost: 1 experiment.
3. **SwiGLU + reduced architecture to fit cap.** SwiGLU(mlp=3) + NUM_LAYERS=8 (one fewer layer) would shed ~3M params, putting artifact at ~13–14 MB; the gating advantage might survive. Cost: 1 code-change experiment with subagent.
4. **Sliding-window attention.** Records use it heavily; never tested. Larger code change → subagent. Could free attention compute for more depth.
5. **Tighten MATRIX_LR at batch=24k.** Try 0.042, 0.048 to bracket the optimum more precisely. Expected gain ≤ +0.005, but with cross-seed noise of 0.0024 this is at the edge of detectability.
6. **Lower-LR direction for embedding/scalar params.** TIED_EMBED_LR=0.04 and SCALAR_LR=0.03 are untested. The 1.5× scale-ups both hurt; the optimum might be DOWN, not at-canonical.
7. **Investigate batch=32k failure mode directly.** Read out gradient norms/variances during training to confirm the "Adam variance saturation" hypothesis vs the alternate "grad_accum=8 micro-batch loses stochasticity" hypothesis.

## File pointers for the next agent

- `journal.md` — narrative of every promote, with **Question/Setup/Prediction/Disconfirming/Result/Conclusion** structure.
- `results.tsv` — every experiment with status (`keep`/`discard`/`parked`/`crash`/`sentinel`) and one-line description.
- `winners/` — snapshots of every promotion in chronological order. Each has its own `env.sh`, `plan.md`, `train_gpt.py`, `result.json`, `run.log`.
- `experiments/` — gitignored scratch state for every run, including discards. The plan.md inside each carries the hypothesis and notes-from-execution.
- `git log winners/` — chronological list of promotes; commit messages summarize each Δ.
- This file (`summaries/2026-04-25_overnight_session.md`) — high-level navigation.
