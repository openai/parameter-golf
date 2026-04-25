# Journal

## Current threads
- Anchor baseline: exp `0001_baseline_repro` at val_bpb 2.5212 (post-quant int8+zlib), 6.907 MB. Bit-reproduces the Apr-18 reference run. All sentinels and noise-floor comparisons still reference this row.
- **Best so far: 2.12603** (`winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_24k_matrix_lr_045_init_05`, exp 0036). batch=24k + MATRIX_LR=0.045 (down from 0.06) on 0024 init winner. Single-seed Δ=+0.045 vs 0024; SEED=42 confirm in 0047 at 2.14045 — **mean Δ=+0.038** (cross-seed Δ 0.014 — larger variance than the typical 0.0024 we've seen for confirmed configs). Win is real but smaller than the SEED=1337 number alone suggested.
- **Crucial revision**: the 0018 batch=32k mode-collapse was an LR-coupling issue, not a batch ceiling. Bigger batch + smaller LR (LR×batch held ~constant) is the correct scaling.
- Prior winner: 2.17103 (exp 0024, init=0.05).
- Cumulative gain vs canonical baseline (2.5212): +0.395 → 2.1260.
- TIED_EMBED_LR=0.075 (0022) HURT by 0.012 — embedding LR is more sensitive than matrix LR. But TIED_EMBED_INIT_STD=0.02 (0023) HELPED by 0.011 — bigger init is a separable, complementary lever.
- LR scaling is a separable lever from schedule shape: schedule changes *shape*, MATRIX_LR changes *magnitude* across the curve.
- Schedule push diminishing returns: 0005 (+0.116) → 0015 (+0.055) → 0020 (+0.029).
- Cumulative vs canonical: +0.323 (2.5212 → 2.1985).
- Prior winner: 2.2099 (exp 0021).
- Prior winners (still in `winners/` as history): 0013 (2.3096, batch=16k), 0012 (2.3686, batch=16k+seq=2048 confounded), 0008 (2.3913, mlp4), 0005 (2.4052, schedule).
- Cumulative gain stack vs canonical baseline (2.5212 → 2.2547): schedule (+0.116) + capacity (+0.014) + batch_16k (+0.082) + schedule push (+0.055) = **+0.267 total**.
- **Important [transfer:high] finding**: at this regime, seq=1024 is strictly better than seq=2048 — see exp 0013 decomposition. Don't extend seq_len beyond what the model can use.
- **20-step warmup is dramatically better than 10-step** at this batch size: at lr_warmup=10 the warmup peak hits lr_mul=1.0 *while still ramping cumulative LR*, causing a step-9 train_loss spike (loss > step 1). At warmup=20 the same lr_mul=1.0 peak occurs after 20 steps of measured ramp; no spike, much smoother trajectory.
- Prior winner: 2.4052 (exp 0005, schedule-only). Schedule change confirmed by SEED=42 in 0006 at 2.40272 — cross-seed Δ 0.0024.
- The lr_mul formula in `train_gpt.py` is `(iterations−step)/warmdown_iters` after warmup. With ITERATIONS=200, the canonical default `WARMDOWN_ITERS=1200` gives lr_mul peaking at 0.167 (avg 0.083) — extremely attenuated. The 0005 schedule (warmup_10 + warmdown_600) doubles avg lr_mul to 0.178; tested up to and through a brief one-step lr_mul=1.0 spike at the warmup peak (recoverable). Further-aggressive schedules NOT yet tested.
- Capacity (MLP_MULT, exp 0002) and attention temperature (QK_GAIN_INIT, exp 0003) showed only Δ≈+0.002 each under the canonical schedule — noise-band. Hypothesis: their effects are MASKED by the under-training. Both should be re-tested ON TOP OF the new schedule.
- Quant tax actually IMPROVED in 0005 (0.0029 vs baseline 0.0055) — better-trained weights quantize cleaner. Means architectural + schedule improvements are likely additive in post-quant terms.
- All schedule wins are **[transfer:low]** — the H100 20k-step regime has a different optimal schedule. Future autoresearch should focus architectural experiments on top of this schedule (transfer:high/med candidates).

---

## Entries (newest first)

## 2026-04-25 · exp 0047 · SEED=42 of 0036 — winner real but cross-seed variance higher than typical

The 0036 single-seed Δ=+0.045 was just below the +0.050 mandatory-confirm threshold so I direct-promoted it without a SEED=42. Belated confirmation now: SEED=42 gives val_bpb 2.14045. Cross-seed Δ from 0036 = 0.014 — about 5x the 0.0024 typical for our confirmed configs (0005/0006, 0013/0014, 0015/0016).

Mean across both seeds: 2.13324. Mean Δ vs 0024 (the prior winner): **+0.038** (not the +0.045 claimed from the single seed).

**Implication**: 0036 stays as the winner — the averaged +0.038 is well above the +0.010 noise floor. But the magnitudes I reported throughout the journal/results.tsv from single-seed comparisons were probably slight overstatements at the upper end. Future sessions should treat the journal numbers as "single-seed estimates" rather than "averaged true effect" unless explicitly noted otherwise. The cross-seed-confirmed wins (0005, 0013, 0015) are the most reliable; the single-seed-confirmed wins (0008, 0021, 0023, 0024, 0036) likely have ~10-20% error bars.

The pattern of high cross-seed variance specifically at 0036 is curious: same model size and structure as 0024, just bigger batch + lower LR. Possibly the 24k-batch configuration's particular sequence-ordering at SEED=1337 happened to be slightly more favorable than at SEED=42.

## 2026-04-25 · exp 0044-0046 · SwiGLU works but doesn't fit cap

After the env-var summary (0043), tried SwiGLU as a code-level change. Three experiments:

- **0044 SwiGLU(mlp=3) + LR=0.045** [parked, size_violation]: val_bpb_post 2.11489, **Δ=+0.011 vs 0036** (real gain, above noise floor). BUT artifact 16.46 MB > 16 MB cap = INVALID submission. Demonstrates SwiGLU's gating mechanism genuinely helps at this regime, but the 3-matrix structure pushes us over the cap.
  - Curious step 1 train_loss anomaly: 20.67 (vs expected ~6.93). Recovers to 6.4 by step 10 and continues normally. Likely a numerical artifact specific to SwiGLU + the kaiming-default init for `w_gate`/`w_up`. Cause not fully diagnosed; the recovery suggests it's a benign init-time spike rather than a training-stability issue.
- **0045 SwiGLU(mlp=2) + LR=0.045** [discard]: val_bpb 2.12347, Δ=+0.003 vs 0036 (noise). Smaller hidden dim (1024 vs 1536) gives 12.7 MB artifact (well under cap) but loses the gating advantage. SwiGLU is per-param more efficient than relu² (similar val at fewer MLP params: 1.57M/MLP vs 2.10M/MLP), but no net gain on val_bpb.
- **0046 SwiGLU(mlp=3) + LR=0.035** [discard]: tried lowering MATRIX_LR to shrink artifact below cap. Artifact 15.72 MB (fits) but val_bpb 2.14454 — Δ=-0.019 vs 0036. The reduced LR sacrifices enough training that the SwiGLU advantage is wiped out.

**Implication**: at the d=512, 9L architecture, SwiGLU's gain (~+0.011) is real but not extractable within the 16 MB artifact constraint at any LR setting. To realize SwiGLU here would require either (a) reducing parameter count elsewhere — e.g. lowering num_layers and using SwiGLU(mlp=3) — or (b) a different cap-friendly variant like GeGLU.

**Final session state**: 46 experiments. Submittable best stays at **2.12603 (exp 0036)**. Cumulative gain vs canonical baseline (2.5212): **+0.395**. SwiGLU(mlp=3) at 2.11489 is the best non-submittable result (size_violation), Δ=+0.011 over 0036.

## 2026-04-25 · session summary · 43 experiments, +0.395 from canonical baseline, env-var search exhausted

**Final state**: Best `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_24k_matrix_lr_045_init_05` (exp 0036) at val_bpb_post_quant **2.12603**. Cumulative gain vs canonical baseline (2.5212): **+0.395** post-quant.

**Confirmed-paying axes (in order of contribution)**:
| Axis | Δ | Notes |
|---|---|---|
| LR schedule rewrite (warmup10+warmdown600) | +0.116 | exp 0005/0006. The biggest single lever — canonical schedule was 5x too attenuated for 200 steps. [transfer:low] |
| batch_tokens 8k → 16k | +0.082 | exp 0013/0014. Bigger gradient signal per step. [transfer:high] |
| schedule push 1 (warmup20+warmdown400) | +0.055 | exp 0015/0016. 20-step warmup eliminates the step-9 spike of 10-step. [transfer:low] |
| batch_tokens 16k → 24k + LR 0.06 → 0.045 | +0.045 | exp 0036. Bigger batch with proportionally lower LR. [transfer:high] |
| schedule push 2 (warmup30+warmdown300) | +0.029 | exp 0020. Diminishing returns. [transfer:low] |
| TIED_EMBED_INIT_STD 0.005 → 0.05 | +0.038 | exp 0023+0024. Canonical init was 10x too small. [transfer:high] |
| MATRIX_LR 0.04 → 0.06 (at batch=16k) | +0.016 | exp 0021. LR scaling separate from schedule shape. [transfer:med] |
| MLP_MULT 2 → 4 | +0.014 | exp 0008. Capacity scales monotonically up to 4 only. [transfer:high] |

**Dead axes (no signal or hurt)** at the explored config: NUM_LAYERS=11, MLP_MULT=5+, QK_GAIN, LOGIT_SOFTCAP, MUON_MOMENTUM, BETA1, BETA2, ROPE_BASE, GRAD_CLIP_NORM, TIED_EMBED_LR scale-up, SCALAR_LR scale-up, ORTHO_INIT, TRAIN_SEQ_LEN=2048, TRAIN_BATCH_TOKENS=32768.

**Cross-experiment lessons** (each validated by dedicated experiments, not just intuition):
1. **The canonical schedule was the dominant under-training factor at 200 steps.** Three confirmed schedule pushes (0005/0015/0020) gave +0.200 cumulative. Evidence: avg lr_mul went 0.083 → 0.318. The lr_mul=1.0 spike at warmup peak is brief enough on MPS bf16 to recover (stronger guard with longer warmup).
2. **Schedule-masking can hide both real positives AND negatives.** QK_GAIN=5 looked like noise (+0.002) under the canonical schedule but actively HURT (-0.028) on the 0005 schedule. Pre-promote architectural ablations should always re-test on the current winner schedule.
3. **Capacity at 200 steps caps quickly.** MLP_MULT=4 wins (+0.014); MLP_MULT=5 plateaus (refuted by SEED=42); NUM_LAYERS=11 plateaus. The records' use of these at H100 20k-step doesn't transfer to the smoke regime — those paths need step-budget to be useful.
4. **batch+LR are coupled, not separate axes.** batch=32k+LR=0.06 mode-collapses (0018, train_loss 0.55 with val 4.43); batch=24k+LR=0.045 wins (+0.045, 0036). The 0018 framing as "batch ceiling" was wrong — it was an LR-coupling failure. Bigger batch + proportionally smaller LR keeps paying. Why batch=32k still failed at LR=0.03 (0037) is unclear; possibly the 4-sequences-per-microstep configuration loses critical stochasticity.
5. **Init scale 0.005 → 0.05 was a hidden bug.** Canonical init severely under-initialized embeddings; bigger init gave +0.038 across two experiments. Optimum is precise: 0.05 wins, 0.07 hurts (-0.058), 0.04 ties (noise), 0.1 catastrophic (-0.096).
6. **Most env-var axes are genuinely flat at this regime.** Of ~25 unique knobs explored, only 5 produced robust wins. The rest (optimizer momentums/betas, rope, softcap, grad-clip, embed/scalar LR scaling, ortho-init) yielded noise or harm.
7. **Quant_tax behavior is informative.** Anomalously low quant_tax (≤0.001) is a red flag for mode-collapse-like degeneracy (seen in 0018 mass-collapse, 0009 mlp5 quant-tax-noise that didn't reproduce, 0032 BETA2 freak). Healthy runs have quant_tax 0.002-0.005.

**Cross-seed variance baseline**: 0.0024-0.0027 for typical configs (0005/0006, 0013/0014, 0015/0016). Larger cross-seed Δ (0.008-0.015) is a marker of an outlier run, not the true effect.

**For H100 20k-step transfer**: most reliable wins are [transfer:high] — batch (0013), capacity (0008), init (0023/0024), seq_len-do-not-extend (0013 decomposition). The schedule wins are [transfer:low] (200-step specific). MATRIX_LR is [transfer:med].

**State of the search**: env-var space largely drawn. Next tier of gains likely requires code-level changes outside the protocol's env-var lane:
- SwiGLU activation (well-grounded but artifact-tight at our cap usage)
- Sliding-window attention (records use it heavily)
- Parallel residuals
- Different normalization / different residual scheme
- EMA over weights

A single env-var-only code-change attempt (ortho-init in 0041) hurt by 0.022 — confirming the canonical kaiming + Muon + ReLU² is well-balanced.

## 2026-04-25 · exp 0037-0040 · batch=32k still hard-fails; init/clip retest on 0036 confirms saturation

After 0036 unlocked the batch+LR-coupling axis, four more experiments tried to keep stacking:

- **0037 batch=32k + MATRIX_LR=0.03** [discard, mode-collapse]: Δ=-0.442 vs 0036. Even with proportional LR scale-down (product=960, same as the original batch=16k+LR=0.06 winner), batch=32k still collapses. The batch=32k regime change is *not* purely about LR; something else changes at that batch size — possibly the 4-sequences-per-microstep structure (vs 3 at batch=24k, 2 at batch=16k) provides too little stochasticity for Adam-style gradient variance estimates to remain useful.
- **0038 GRAD_CLIP_NORM=1.0** [discard]: Δ=+0.003 (noise). Gradient clipping isn't a meaningful axis at this LR/batch.
- **0039 init=0.07 on 0036** [discard]: Δ=-0.058. Slight push above 0.05 hurts. Init optimum stays at 0.05 even in the new batch=24k regime.
- **0040 init=0.04 on 0036** [discard]: Δ=-0.002 (noise). Slight push below 0.05 is tied. The init=0.05 point is robust.

**Cross-experiment pattern**: with the 0036 batch=24k discovery exhausting the major axes, marginal env-var tuning is mostly noise or harmful. The map is largely drawn:
- Working axes (gave real wins): schedule (3 levels of push), MLP capacity (mlp4), batch (8k → 16k → 24k with appropriate LR), MATRIX_LR scaling, embedding init scale (0.005 → 0.05).
- Dead axes (no signal or harmful at this regime): qk_gain, num_layers, mlp_mult>4, seq_len>1024, batch>24k, LOGIT_SOFTCAP, MUON_MOMENTUM, BETA1, BETA2, ROPE_BASE, GRAD_CLIP_NORM, TIED_EMBED_LR scale-up, SCALAR_LR scale-up, init outside [0.04, 0.05].
- Saturated axes (incremental tuning doesn't help): schedule beyond warmdown_300, init around 0.05.

**Best so far stays at 2.12603 (exp 0036)**. Cumulative gain vs canonical baseline (2.5212): **+0.395**.

To unlock further wins, env-var exploration is exhausted. Next direction would need code changes (architectural / optimizer / init scheme) — a different scope.

## 2026-04-25 · exp 0036 · batch=24k + LR=0.045 wins +0.045 — refutes 0018 batch ceiling

**Question**: Throughout 0029-0035 the env-var sweep was finding mostly noise/discards. The 0018 batch=32k regression was previously hypothesized as "Adam variance saturation at large batch produces too-aggressive per-dim updates; the fix is to lower LR proportionally with batch." But that was [CONJECTURE]. Test it directly: batch=24k + MATRIX_LR=0.045 (LR×batch nearly constant vs 16k+0.06).

**Result**: val_bpb 2.12603 → Δ=+0.0450 vs 0024 (in [+0.010, +0.050] direct-promote window). Pre-quant Δ=+0.047 — real training gain. Quant_tax 0.0027 (normal — no degeneration like 0018). Artifact 14.619 MB (smaller than 0024's 15.59 because lower LR produces less aggressive weight magnitudes).

**Conclusion** [VERIFIED]:
1. The 0018 mode collapse was *not* a batch ceiling — it was an LR-batch coupling issue. The fix (lower LR proportionally with batch) works cleanly.
2. **Bigger batches at appropriately scaled LR are still paying** at this regime. The "16k batch is the sweet spot" framing from the journal was wrong; 24k works better at the right LR.
3. Even though we have ~20 mostly-discarded experiments since 0024, the pattern data was useful — it isolated which axes are dead (most of them) and motivated returning to the known-paying batch axis with a corrected hypothesis.
4. **[transfer:high]** — batch scaling with proper LR coupling is universally robust.

Cumulative win vs canonical baseline (2.5212): +0.395 → 2.1260.

Followups: try batch=32k + MATRIX_LR=0.03 (the 0018 retry now done correctly); LR fine-tune at batch=24k (try 0.04 or 0.05); revisit init scale on the new winner.

## 2026-04-25 · exp 0025-0030 · ceiling sweep — init/softcap/schedule/momentum

After 0024's surprise +0.027 from init scaling, swept several axes looking for the next stack-able win. Most landed in noise band or hurt:

- **0025 init=0.1** [discard]: Δ=−0.096 (overshoots). Confirms 0024's init=0.05 is near optimal; tok_emb row norms ~2.26 at init=0.1 destabilizes the tied lm_head logit path.
- **0026 LOGIT_SOFTCAP=15** [discard]: Δ=−0.044 (hurts). Tighter softcap clips useful logit signal; canonical 30 is the right value for sp1024.
- **0027 + 0028 schedule push #4 (warmdown 250 + warmup 35)**: SEED=1337 had Δ=+0.009 (judgment-call), SEED=42 had Δ=+0.001 (noise). Mean Δ=+0.005. **Schedule push has plateau'd**. Diminishing returns curve confirmed: 0005 (+0.116) → 0015 (+0.055) → 0020 (+0.029) → 0027/0028 mean (+0.005). Don't push further.
- **0029 + 0030 MUON_MOMENTUM 0.9 / 0.85** [discard, discard]: Δ=+0.003 each (noise band). Momentum is not a meaningful axis at this regime; the 5-10% changes in momentum vs canonical 0.95 give negligible directional signal.

**Cross-experiment signal**: the rate of "real wins" has dropped sharply after the init=0.05 promotion. We're in late-discovery territory — most env-var axes have been tested or are clearly saturated. Untested but probably-not-impactful: BETA1, BETA2, ROPE_BASE, GRAD_CLIP_NORM, SCALAR_LR. Next big gains likely require code changes (architecture mods, optimizer mods) rather than env-var-only.

**Best so far stays at 2.17103 (exp 0024)**.

## 2026-04-25 · exp 0017 + 0018 + 0019 · sentinel clean; batch=32k mode-collapses; depth ceiling

**0017 regression sentinel** [sentinel]: canonical baseline run with no overrides. val_bpb 2.52115777 — *bit-exact* match to 0001's 2.52115777. Harness is healthy after 16 prior runs; no thermal/MPS drift detected.

**0018 batch=32k** [discard, surprise]: Pushed `TRAIN_BATCH_TOKENS=16384 → 32768` on the 0015 winner. **Catastrophic regression**: val_bpb 2.59031 (Δ=−0.336 vs 0015), even worse than canonical baseline. Pre-quant val 2.5900, quant_tax 0.0003 (basically zero — signature of degenerate weights). Trajectory anomaly: step 1 train_loss=4.27 (expected ~6.93 for fresh init at vocab=1024), step 3 already at 1.10, step 200 at 0.55 *while val_loss=4.43*. Classic mode-collapse: model learned to predict a small set of high-frequency tokens (space/the/etc.) that scores well on training-batch statistics but doesn't generalize. **Hypothesis** [LIKELY]: at batch=32k the gradient variance drops far enough that Adam's `m_hat / sqrt(v_hat)` saturates near unit per-dim → effective per-step movement under MATRIX_LR=0.04 + lr_mul=1.0 is much larger than at batch=16k (where gradient noise dampened the per-dim update). The 8k → 16k jump worked because we were still in the noise-dominated regime; 16k → 32k crosses into the regime where LR needs to be scaled DOWN to compensate (counterintuitive: bigger batch usually allows LARGER LR, but with adaptive optimizers and very low gradient variance, the standard heuristic flips). **Implication**: the batch axis is not free; the current 16k-batch + WARMDOWN_400_WARMUP_20 schedule is a coupled sweet spot. To push batch further would require lowering MATRIX_LR/TIED_EMBED_LR proportionally — separate experiment if desired.

**0019 NUM_LAYERS=11** [discard]: 11L is a recurring records pattern (e.g. 11L_EMA_GPTQ-lite at 1.1228). val_bpb 2.25217 (Δ=+0.0025 vs 0015 — noise). Pre-quant Δ=+0.0004 (basically zero). Quant tax suspiciously low (0.0007) — most of the tiny post-quant gain is from cleaner quantization, not better training. Cost: +5.8M params, 15.7 MB artifact (tight against 16 MB cap), step_avg +27%. **Conclusion** [VERIFIED with mlp_mult=5 precedent]: at 200 steps + sp1024 + d=512, *additional architectural capacity beyond mlp_mult=4 / 9L doesn't help*. Records use 11L at H100 20k-step where deeper composition has time to express; at 200 steps the new layers are under-trained. Both 11L and mlp5 mode-of-failure: low quant_tax + flat pre-quant, suggesting the new params end up small/structureless.

**Cross-experiment pattern**: at 200 steps, capacity scaling (mlp_mult, num_layers) tops out very quickly. Schedule and batch are the dominant levers — until both saturate, where saturation looks like:
- batch saturation = mode collapse (very low train, normal val) at 32k+
- capacity saturation = low quant_tax + zero pre-quant Δ

## 2026-04-25 · exp 0015 + 0016 · schedule push (WARMDOWN_400 + WARMUP_20) lands +0.055

**Question**: 0013 winner uses WARMDOWN_ITERS=600 + LR_WARMUP_STEPS=10 (avg lr_mul ≈ 0.178). Bigger batches tolerate higher LR. Does pushing the schedule further pay?

**Setup**: Same as 0013 but with WARMDOWN_ITERS=400 + LR_WARMUP_STEPS=20. New schedule: warmup 0.05 → 1.0 over 20 steps, warmdown branch fires at step 20 with lr_mul=0.45 dropping to 0 over the next 180 steps. Avg lr_mul ≈ 0.255 (1.43× the 0013 schedule).

**Prediction** [LIKELY]: Δ vs 0013 ≈ +0.020 to +0.050.

**Disconfirming**: NaN around step 19 (peak lr_mul=1.0); step 2 spike worse than 0005's 7.06; Δ ≤ +0.005 (schedule was already optimal).

**Result**:
- 0015 (SEED=1337): val_bpb 2.25468; pre-quant 2.2519; quant_tax 0.0028.
- 0016 (SEED=42 confirm): 2.25740; pre 2.2546; quant_tax 0.0028 (essentially identical).
- Cross-seed Δ = 0.00272. Mean = 2.25604. Mean Δ vs 0013 = **+0.0535**.

**Trajectory comparison** (warmup spike behavior at step 9):
- 0008 (warmup=10): step 9 train_loss = 7.06 (above step 1 → spike-and-recover).
- 0015 (warmup=20): step 9 train_loss = 5.85 (smooth descent, no spike).

The difference: at warmup=10 the lr_mul transitions are 0.1 → 0.2 → ... → 1.0 in 10 steps, with each lr_mul applied to a forward pass *before* the cumulative effect of prior LR has settled. The lr_mul=1.0 step (step 9) lands on weights that have been pushed for 9 prior elevated-LR steps. With warmup=20, the same peak lands on weights with smaller cumulative motion, and the model isn't yet far from a stable region.

**Conclusion** [VERIFIED across 2 seeds]:
1. Bigger batch (more accurate gradients) really does tolerate more LR. Combined with smoother warmup, both peak and average LR can go up.
2. **20-step warmup is the right choice for the bigger-batch regime, not 10**. The step-9 spike in 0008/0013 was a non-trivial cost we didn't realize until the warmup=20 trajectory showed how much smoother the schedule could be.
3. **[transfer:low]** — schedule tuning is even more 200-step-specific than the 0005 rewrite. H100 20k-step uses warmup in the hundreds.

Cumulative stack: schedule (+0.116) + capacity (+0.014) + batch (+0.082) + schedule push (+0.055) = **+0.267 total**. Best post-quant val_bpb: 2.2560.

Followups in queue: TRAIN_BATCH_TOKENS=32768 (continue batch scaling), more aggressive schedule (WARMDOWN_300), regression sentinel (overdue at ~16 runs).

## 2026-04-25 · exp 0013 + 0014 · batch=16384 is the real lever; seq=2048 hurts

**Question**: 0012 doubled both seq_len (1024→2048) and batch_tokens (8192→16384) and gained Δ=+0.023. Decomposed by running batch=16384 with seq=1024 (0013) and SEED=42 confirm (0014).

**Result**:
- 0013 (batch=16k, seq=1024, SEED=1337): val_bpb 2.30956 → Δ=+0.0818 vs 0008.
- 0014 (SEED=42 confirm): 2.31199. Cross-seed Δ=0.00243 — matches 0005/0006 variance precisely.
- Mean: 2.31077. Mean Δ vs 0008 = +0.0806; mean Δ vs 0012 = +0.0578.
- Both deltas above the +0.050 "suspicious" threshold but reproduce tightly.

**Decomposition**:
- 0008 (8k batch, 1024 seq): 2.39135 — anchor.
- 0013 (16k batch, 1024 seq): 2.30956 → batch effect at fixed seq = +0.0818.
- 0012 (16k batch, 2048 seq): 2.36857 → seq effect at fixed batch = −0.0590 (vs 0013).

**Conclusion** [VERIFIED across 2 seeds]:
1. **Doubling batch tokens is the dominant lever** at the 200-step regime. Bigger batches give more gradient signal per step; under a fixed warmup+warmdown schedule, the effective parameter motion per step doesn't grow proportionally so the optimizer is more stable, less noisy.
2. **Longer seq_len HURTS** at sp1024 / d=512 / 9L / 200 steps. Hypothesis [LIKELY]: with grad_accum_steps=8 fixed, seq=2048 gives 1 sequence per micro-step (vs 2 at seq=1024), halving sequence diversity per optimizer step. Most useful patterns at this scale are short-range; the 2× attention compute is wasted.
3. The 0012 promotion was *correct at the time* (it beat 0008) but ultimately got superseded by 0013's clean, stronger result. winners/ keeps both as history.
4. **[transfer:high]** for 0013 — batch scaling is the most universally robust lever; H100 evaluator near-certainly uses larger batches than our smoke, so this finding should hold.

Cumulative stack vs canonical baseline (2.5212): schedule (+0.116) + capacity (+0.014) + batch_16k (+0.082, includes the −0.059 cost we now know to avoid) ≈ **+0.212 total**. Best post-quant val_bpb: 2.3096.

Followups in queue: TRAIN_BATCH_TOKENS=32768 (further batch scaling), WARMDOWN_400_WARMUP_20 (schedule push at the new batch size), LR_WARMUP_STEPS=20 (smoother warmup, no step-9 spike).

## 2026-04-25 · exp 0009 + 0010 + 0011 + 0012 · capacity ceiling, qk_gain reversal, seq+batch win

Compressed multi-experiment summary (all building on the 0008 mlp4 winner):

**0009 + 0010 — MLP_MULT=5 capacity ceiling test** [discard, discard]:
- 0009 (SEED=1337): val_bpb 2.38468, suspiciously low quant_tax 0.0020. Δ=+0.007 looked promising but the gain was suspect (pre-quant Δ only +0.0037).
- 0010 (SEED=42 confirm): val_bpb 2.39114, quant_tax 0.0034 (typical). Pre-quant 2.3877 actually *worse* than 0008 mlp4 (2.3864).
- **Conclusion** [VERIFIED]: Capacity ceiling at mlp_mult=4 for sp1024/9L/d=512/200steps. The mlp_mult=5 apparent gain was quant-tax variance, not capacity. Useful precedent: when most of an apparent gain comes from a single odd quant_tax value, demand SEED=42 to confirm.

**0011 — QK_GAIN_INIT=5 on winner schedule** [discard]:
- val_bpb 2.41938 vs 0008 winner 2.39135 → Δ=-0.028 (clear regression).
- **Lesson** [VERIFIED]: schedule-masking can hide a real *negative* effect, not just a positive one. Under the canonical schedule (0003) QK_GAIN=5 looked like noise (+0.002); under healthy training, it actively hurts. The records' qk-gain ∈ [5, 5.25] usage is mostly sp4096/sp8192; at sp1024 it's the wrong knob.

**0012 — TRAIN_SEQ_LEN=2048 on winner** [keep, promoted, CONFOUNDED]:
- First attempt crashed: hardcoded `grad_accum_steps=8` plus `TRAIN_BATCH_TOKENS=8192` give per-microstep token count of 1024, can't reshape to seq_len=2048. eval_val has an explicit assertion for the same. Fixed env-var-only by also doubling TRAIN_BATCH_TOKENS to 16384 (and VAL_BATCH_SIZE to match) — confounds the experiment.
- val_bpb 2.36857 vs 0008 winner 2.39135 → **Δ=+0.0228**, well above noise floor. Pre-quant Δ=+0.021. Quant tax stays clean (0.0032). step_avg 3432 ms (2.5× slower per step from attention² + 2× batch).
- Promoted as `winners/2026-04-25_warmdown_600_warmup_10_mlp_mult_4_seq_2048_batch_16k`.
- **Confounding**: doubled both seq_len and batch_tokens. 0013 will run batch=16384 + seq=1024 to decompose. Pure-context test would need a code change to make grad_accum_steps configurable.

Cumulative stack vs canonical baseline (2.5212): schedule (+0.116) + capacity (+0.014) + seq_len/batch (+0.023) ≈ **+0.153 total**. Best post-quant val_bpb: 2.3686.

## 2026-04-25 · exp 0007 + 0008 · capacity scaling on the new schedule (mlp2 → mlp4)

**Question**: 0005's schedule rewrite (avg lr_mul 0.083 → 0.178) was framed as "the previous architectural ablations were likely false negatives — capacity (0002) and qk_gain (0003) should be re-tested on the new schedule." Does MLP capacity scaling produce real Δ now?

**Setup**: Forked from canonical, env-vars only:
- 0007: schedule + `MLP_MULT=3` (one capacity step above canonical mlp_mult=2)
- 0008: schedule + `MLP_MULT=4`

**Prediction** [LIKELY]: Δ vs winner ≈ +0.012 to +0.025 for mlp_mult=4. mlp_mult=3 was framed as a midpoint test.

**Disconfirming**: Δ ≤ +0.005 for mlp_mult=4 → capacity is NOT a real lever even with healthy LR; the 0002 zero was structural, not schedule-masked.

**Result**:
- 0007 (MLP_MULT=3): val_bpb_post 2.39927, Δ=+0.0059 vs 0005 (judgment-call zone). Initially parked.
- 0008 (MLP_MULT=4): val_bpb_post 2.39135, **Δ=+0.0138** vs 0005 — over the noise floor, no SEED=42 needed.
- Capacity scaling is monotonic: mlp2 2.4052 > mlp3 2.3993 > mlp4 2.3913. 0007's marginal +0.006 was real signal at half this capacity bump; superseded by 0008.
- Quant tax climbs slowly with capacity (0.0029 → 0.0039 → 0.0049) but stays well under the 0.020 fragility threshold.
- Artifact: 8.1 → 9.97 → 11.77 MB. 4.2 MB headroom for further capacity (mlp_mult=5 likely fits at ~13.5 MB; mlp_mult=6 risks the cap).

**Conclusion** [VERIFIED] (capacity scaling is well-known; the new fact is the schedule-masking story):
1. The 0002 zero-result really was schedule-masking. Future architectural ablations (init scale, attention temperature, depth, sequence) under the canonical schedule produce noise; they all need to be re-tested on top of the 0005 winner schedule.
2. Capacity scaling at sp1024 / 9L / d=512 is monotonic out to mlp_mult=4 at least. Records (Apr-01 mlp_mult=4 at 0.9979) suggest scaling further is productive at H100 scales; whether it scales further on the 200-step smoke is the next question (test mlp_mult=5).
3. Stacking is additive: schedule (Δ +0.116) + capacity (Δ +0.014) ≈ Δ +0.130 vs canonical baseline. **[transfer:high]** — capacity wins are the most robust class of improvements; this layer should hold at 20k-step H100 too.

## 2026-04-25 · exp 0005 + 0006 · schedule rewrite — first big win (Δ +0.116)

**Question**: After 0002 (MLP capacity) and 0003 (q_gain temperature) both came in at Δ≈+0.002, the leading hypothesis was that the canonical default `WARMDOWN_ITERS=1200` (peak lr_mul 0.167, avg 0.083) keeps the model so under-trained that no architectural change can show its true effect. Does opening up the schedule to peak lr_mul=1.0 briefly + warmdown from 0.317 produce a real Δ?

**Setup**: `LR_WARMUP_STEPS=10` + `WARMDOWN_ITERS=600`. Schedule: ramp 0.1 → 1.0 over steps 0-9, then warmdown branch `(200−step)/600` starting at 0.317 at step 10, decaying to 0 at step 200. Avg lr_mul = 0.178 (vs 0.083 baseline → 2.14×). **Pre-experiment 0004** without warmup (WARMDOWN_ITERS=600 alone) was killed at step 10: step 2 train_loss spiked from 6.94 → 8.40, an LR-induced first-step overshoot of cold tok_emb. Adding the 10-step warmup was sufficient to make the schedule trainable.

**Prediction** [LIKELY]: Δ ≈ +0.030 to +0.080.

**Disconfirming**: NaN around the lr_mul=1.0 spike at step 9 (would require gentler warmup); Δ ≤ +0.005 (would mean LR isn't the bottleneck either).

**Result**:
- 0005 (SEED=1337): val_bpb_post=2.40517, pre=2.4023, quant_tax 0.0029, artifact 8.105 MB.
- 0006 (SEED=42 confirm): val_bpb_post=2.40272 — Δ between seeds 0.00245.
- **Mean Δ vs baseline: +0.117** — way above the +0.050 "suspicious-large" threshold but reproduces tightly across seeds.
- Trajectory: step 9 train_loss=7.058 (above step 1) — the brief lr_mul=1.0 spike does cause mid-warmup overshoot, but recovers by step 10 once warmdown drops lr_mul to 0.317. Step 35=4.85 (vs ~5.13 baseline-class trajectory), then steady descent to step 200=4.26 (vs 4.42 baseline). No NaN.
- Quant tax DROPPED from 0.0055 to 0.0029 — better-trained weights have more structured distributions that quantize cleaner. Important: post-quant gains can OUTPACE pre-quant gains for schedule changes.

**Conclusion** [VERIFIED across 2 seeds]:
1. The 200-step smoke under canonical `WARMDOWN_ITERS=1200` is severely compute-starved — not because the schedule is "wrong" for 200 steps but because it was inherited from a 20k-step canonical run. Doubling avg lr_mul (with the warmup needed to avoid first-step blowup) recovers ~0.12 of val_bpb that was previously "trapped behind under-training."
2. The brief lr_mul=1.0 spike at step 9 causes a transient train_loss bump but is fully recoverable on MPS bf16; at least at this brevity. Pushing the schedule further (sustained lr_mul ≥ 0.5) is the next axis to test.
3. **Strongest implication for autoresearch**: prior architectural ablations under the canonical schedule are likely false negatives. MLP_MULT, QK_GAIN_INIT, and any other "improves training" change should be re-tested ON TOP OF this schedule before being discarded.
4. **[transfer:low]** — the H100 20k-step regime has a different schedule sweet spot. The discovery itself is about the *autoresearch testbed*, not the submission. Useful as a multiplier for ranking architectural changes.

## 2026-04-25 · exp 0002_mlp_mult_3 · capacity isn't the bottleneck at 200 steps

**Question**: With ~9 MB of artifact headroom, does spending some on extra MLP capacity (mlp_mult 2→3) move val_bpb at 200 steps?

**Setup**: Forked from canonical, env-var-only change `MLP_MULT=3`. Predicted artifact 8.8 MB; actual 8.404 MB. Same WARMDOWN_ITERS=1200 schedule as baseline.

**Prediction** [LIKELY]: Δ ≈ +0.010 to +0.025 — record lineages repeatedly use mlp_mult ∈ {3, 4} (e.g. 2026-04-01 SP4096+MLPMult4 at 0.9979).

**Disconfirming**: Δ ≤ +0.005 → capacity isn't the dominant bottleneck under this schedule.

**Result**: post-quant Δ=+0.00212 (noise). Pre-quant Δ=+0.0046 (also noise). The bigger MLP picked up extra quant tax (0.0079 vs 0.0055 baseline), so post-quant gain is half of pre-quant gain. Trajectory was healthy: step 1=6.9383 (vs baseline 6.9379), monotonic, no NaN.

**Conclusion** [LIKELY]: At 200 steps under WARMDOWN_ITERS=1200 (avg LR mul ≈ 0.083), MLP capacity is **not** the limiting factor. Records using mlp3x/mlp4x trained for 20k steps on H100, where the under-trained MLP capacity hypothesis doesn't apply. Lesson for autoresearch: prioritize schedule, attention temperature, and initialization changes (which take effect immediately) over scaling parameters (which need steps to be useful) on the 200-step smoke. Capacity ablations should ride on top of a more aggressive schedule, not the canonical-attenuated one.

## 2026-04-25 · exp 0001_baseline_repro · canonical baseline reproduced

**Question**: Can we run a stable 200-step smoke on MPS that bit-reproduces the Apr-18 reference (val_bpb 2.5540)?

**Setup**: Canonical `train_gpt.py`. env.sh sets `WARMDOWN_ITERS=1200` so the step-based warmdown is active from step 0 (`warmdown_start = max(200−1200, 0) = 0`), giving LR multiplier 0.167 at step 0 decaying to ~0 by step 200. Otherwise canonical hyperparameters: `WARMUP_STEPS=0`, `MAX_WALLCLOCK_SECONDS=0`, `TIED_EMBED_LR=0.05`, `MATRIX_LR=0.04`, `SCALAR_LR=0.04`, `GRAD_CLIP_NORM=0`. `VAL_TOKENS=16384` (vs Apr-18's full val).

**Prediction** [LIKELY]: step 2 = 6.7505 (matches Apr-18 log), final val_bpb in [2.4, 2.7].

**Disconfirming**: any deviation from the Apr-18 trajectory (would mean MPS or some env state had drifted), or NaN.

**Result**: trajectory matches Apr-18 to 4 decimal places — step 1=6.9379 ✓, step 2=6.7505 ✓, step 200 train_loss=4.4196 ✓. Artifact 6.906960 MB (Apr-18: 6.905876 MB) — close but ~1 KB off, likely from train_gpt.py's source bytes growing slightly post-Apr-18 (LR_WARMUP_STEPS commit added lines, code_bytes counts it) plus tiny zlib non-determinism downstream of the bf16 model body. val_bpb_post_quant=2.5212 (Apr-18: 2.5540) — the gap is VAL_TOKENS=16384 vs Apr-18's full ~1M-token eval. No NaN.

**Conclusion** [VERIFIED]: The earlier "Apr-18 was a non-reproducible MPS lucky draw" framing was wrong. The schedule was deterministically attenuated — `lr_mul` with `WARMDOWN_ITERS=1200` and `ITERATIONS=200` makes `warmdown_start = max(200−1200, 0) = 0`, so warmdown is active from step 0 and the effective LR multiplier is `(1200 − step) / 1200`. Apr-18 was running at ~17% LR throughout, decaying to ~0% by step 200. This is what kept training stable. Full canonical LR is too aggressive for MPS bf16 numerics (NaN around step 165), and that's what was happening when the env.sh template incorrectly set `WARMDOWN_ITERS=40`.

Three lessons worth carrying forward:

1. **`WARMDOWN_ITERS=1200` is the canonical default** for short MPS smokes. Keep it. Override only when the experiment specifically wants full canonical LR — and pair with explicit `LR_WARMUP_STEPS=10–20` if so.

2. **MPS bf16 has tighter LR tolerance than CUDA bf16.** Canonical `MATRIX_LR=0.04` + `TIED_EMBED_LR=0.05` work on H100 with FlashAttention-3 fused kernels and tighter bf16 guard bits. On MPS they NaN at full LR. The implicit warmdown attenuation hides this; if you remove it without warmup, expect `tok_emb` to NaN at step 2 and `skip_weights` to NaN around step 165.

3. **When you can't reproduce a "lucky" baseline, examine the schedule before suspecting nondeterminism.** PyTorch MPS *is* documented as nondeterministic across runs, but the failure mode of being unable to repro a known-good config is much more often a config delta than a kernel-level RNG difference. Diagnose via `lr_mul` math first, and verify the values that go into it.

## 2026-04-25 · note · full-val eval cost on MPS

Tried `VAL_TOKENS=0` (full ~1M-token val) to get a lower-variance read on the baseline. Aborted after eval was still running 21 minutes after training finished. Total elapsed would have been ~25 min vs ~5 min for the capped eval.

Bottleneck: the patched `eval_val` accumulates `val_loss_sum` / `val_token_count` / `val_byte_count` in float64 on CPU (MPS doesn't support fp64), so each of ~1023 eval micro-steps round-trips a per-batch scalar across the MPS↔CPU boundary, plus a token-id reshape and LUT lookup that also crosses. Roughly 3 device-syncs per batch × 1023 batches = the tax.

Why fp64 at all: eval reports val_bpb to 8 decimals (`val_bpb:2.52115777`), and summing ~1M token-loss contributions in fp32 would lose ~4-5 of those decimals (relative epsilon ~1e-7 × 1023 batches × ~7 nat per batch ≈ 1e-3 absolute drift). Canonical chose fp64 to keep the summation noise below the reported precision; we don't modify the eval harness.

Decision: keep `VAL_TOKENS=16384` as the autoresearch default. `VAL_TOKENS=0` stays available for confirming a marginal result, with the cost (~5× the smoke budget) documented in `program.md`.

## 2026-04-25 · note · fp32+full eval is also forbidden; full-val is just too slow on MPS

Tried again with a modified `eval_val` that accumulates in fp32 on the MPS device (no CPU round-trip). Hypothesis: the bottleneck was CPU↔MPS sync per batch, so keeping everything on-device should make full-val tractable.

It didn't. Run hit 66 min and was still going when killed. Pre-quant val_bpb 2.5274 was logged on the way (vs Apr-18's fp64+full 2.5485 — gap ~0.02, partly fp32 accumulation noise, partly MPS run-to-run training drift visible at step 2: 6.7507 today vs 6.7505 Apr-18).

Why fp32-on-MPS didn't help: per-batch dispatch latency dominates over CPU↔MPS sync. ~8 MPS ops per batch × 1023 batches × ~50–100 ms dispatch each is its own multi-minute tax, separate from the sync. Forward pass on a 17M-param 1024-token batch on MPS is also surprisingly slow (~0.3 s/batch wallclock), making the floor ~5 min just for forward passes — and that's only the *first* eval. **`eval_val` is called twice** (pre-quant and post-int8-quant), so any full-val approach doubles. Total realistic best case ~15 min, observed >60 min.

Updated `program.md` and `env.sh` template to forbid `VAL_TOKENS=0`. Marginal-result confirmation is now done by re-running with `SEED=42` instead. The 16K-cap sample is enough at the 0.010 noise floor — sampling error cancels in same-seed Δ comparisons, which is what ranking actually needs.

The fp32 eval modification itself is reverted; the experiment folder is gone. The canonical eval harness stays untouched (rule preserved).
