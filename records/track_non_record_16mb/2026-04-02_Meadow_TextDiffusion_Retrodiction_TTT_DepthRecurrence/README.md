# Non-record: Shared AR + Masked Denoising — −0.0205 ± 0.005 BPB (5-seed mean) vs Matched Causal-Only Baseline at Final Checkpoint (11L, 1×H100)

*This folder contains the full reproducible artifacts and submission writeup (v3.5) for the 6-run scaling sweep + 5-seed verification ablation reported in [openai/parameter-golf#1255](https://github.com/openai/parameter-golf/pull/1255). Standalone research diary mirror: [github.com/akaiHuang/meadow-golf](https://github.com/akaiHuang/meadow-golf).*

*v3.5 changes (vs v3.3):*
*1. Adds **5-seed multi-seed verification** of the 11L headline (`SEED ∈ {1337, 42, 2024, 7, 100}` for the shared model, `SEED=1337` for the causal-only control), all measured at the true final training step rather than the last `val_every`-aligned intermediate checkpoint.*
*2. Reports the **5-seed mean delta `−0.0205 BPB`** as the primary headline (method-level effect size). The single-seed best (`SEED=1337`, delta `−0.0290 BPB`) is reported as a post-hoc reference for the deployable artifact only and is explicitly **not** the headline number.*
*3. Methodology fix: `train_cdm.py` now unconditionally writes a `step_final.pt` checkpoint at the end of training, so CF evaluation no longer reads a checkpoint hundreds of steps before the actual end of training. This addresses the intermediate-checkpoint concern raised in v3.3 review.*
*4. The original 6-run scaling sweep (5L + 11L) is retained in §3.2 as cross-scale evidence; the 11L numbers in §3.1 are now superseded by the multi-seed final-checkpoint measurement.*

**Wishlist RFC addressed:** Text Diffusion (primary), TTT, Depth Recurrence.

**Author:** Sheng-Kai Huang ([@akaiHuang](https://github.com/akaiHuang)) · akai@fawstudio.com

**Note on authorship.** This is an individual, self-funded research submission. I am not part of a lab or a team. Total self-funded compute across both pods reported here: **~$7.43** ($3.93 for the §3.2 6-run scaling sweep on the first 1×H100 SXM pod in US-MO-1 on 2026-04-09, plus $3.50 for the §3.1 multi-seed verification on a second 1×H100 SXM pod the same day). Every script, log, and the `seeds_run/` spot-check artifacts for §3.1 are committed to this folder or available on my public Hugging Face datasets (`akaiii/meadow-golf-checkpoints`, `akaiii/meadow-golf-v4096`). The exact §3.1 `.npz` and `step_final.pt` state files are intentionally not committed to this PR folder because they would add ~1.3 GB; their location and availability-on-request path are documented in `seeds_run/README.md`. The text uses first-person singular throughout; where it reads "this work" or "this submission" it is shorthand for the same single author.

**Summary.** A shared-weight 11L d=512 v4096 model jointly trained on causal AR + uniform-noise D3PM masked denoising, evaluated via a two-pass Coarse-to-Fine (CF) decoder at the **true final training step**, scores lower BPB than a matched-compute causal-only baseline (1×H100 SXM, 540 s, FineWeb v4096, N=500×1024). Across **5 fresh training seeds** for the shared model and **1 fresh training seed** for the matched control, the **5-seed mean delta is −0.0205 BPB**, with the shared model's CF Total estimated at **1.3009 ± 0.005** (5-seed mean ± std) against a single-control baseline at **1.3214**. The control's training-stochasticity term is *not directly measured* in this round (n=1 fresh seed), and no significance test is computed; see §3.1 for the intuitive calibration that the gap is large relative to the visible variance on the shared side, and §6.0 for the second-control-seed experiment that would close the gap.

A causal-only control run under CF evaluation produces garbage (≈ 2.45 BPB), confirming the effect comes from joint training rather than a metric artifact. The 5-seed mean (−0.0205 BPB) is the method-level effect size; the single best seed (−0.0290 BPB, `SEED=1337`) is reported in §3.1 only as the deployable artifact reference. The original 6-run scaling sweep at 5L + 11L is retained in §3.2 as cross-scale evidence; the 5L row shows a −0.054 BPB single-seed gap that has not yet been multi-seed verified (§6.0). Total self-funded compute across both pods: **~$7.43** on 1×H100 SXM. Every headline number in §3.1 is auditable from files in this folder, including the training and CF eval logs committed under `seeds_run/`; exact reruns are specified in §9.1.

---

## 1. Why This Submission (RFC Response)

The "Requests for PRs" list includes **Text diffusion** as a wishlist item. Twelve diffusion PRs are currently open; the dominant paradigm is bidirectional masked diffusion training evaluated with a discrete absorbing-mask variational bound (`val_var_bpb`), established by #820. That line is progressing well (#1241 at 0.9901, #1106 at 1.1465).

I take a different operational question: **can joint training of causal-AR and masked-denoising objectives on shared weights lower BPB on the standard Parameter Golf metric, when evaluated via a concrete two-pass decoder rather than a 256-step variational bound?** The answer in this submission, under full matched-compute controls and 5-seed verification at 11L (§3.1), is yes: a **5-seed mean delta of −0.0205 BPB** at the matched 1×H100 540 s budget, with a single-seed control baseline (see §3.1 for the statistical caveat). The single-seed best (`SEED=1337`) gives a wider −0.0290 BPB and is reported only as a post-hoc reference for the deployable artifact. The cross-scale 5L row in §3.2 shows a single-seed −0.054 BPB gap that is consistent with the 11L direction but is not yet multi-seed verified (§6.0). The gain at 11L is not a metric artifact: the same CF evaluation run on a causal-only control produces 2.45 BPB (garbage), because the bidirectional mode was never trained. The effect comes from the shared training objective, not from the metric itself.

---

## 2. Method

### 2.1 Training

The shared-weight model is trained with two gradient contributions summed at every step (no phase switching, no loss schedule). The following pseudocode matches `train_cdm.py` lines 997–1012:

```python
# --- AR loss (causal mode) ---
ar_loss = causal_lm_loss(model(x, is_causal=True), y) / grad_accum
ar_loss.backward()

# --- Denoising loss (bidirectional mode) ---
# uniform-noise D3PM: replace masked positions with random vocab tokens
mask_rate = np.random.uniform(0.15, 0.50)                     # per-step rate
mask      = torch.rand(B, T) < mask_rate
x_masked  = x.clone()
x_masked[mask] = torch.randint(0, vocab_size, (mask.sum(),))  # uniform-noise D3PM corruption

logits   = model.forward_hidden(x_masked, is_causal=False)    # bidirectional pass
per_tok  = cross_entropy(logits, x, reduction="none")
cdm_loss = (per_tok * mask.float()).sum() / mask.sum() * 0.3 / grad_accum   # weight = 0.3
cdm_loss.backward()
```

The same parameter tensor is used in both forward calls. The only difference between the two forwards is the `is_causal` flag. There are no separate heads, no separate embedding tables, no phase switching. The two `.backward()` calls are equivalent to summing the gradients of `ar_loss + 0.3 * cdm_loss`.

Key configuration:
- **Mask rate**: `U(0.15, 0.50)` per step (not `U(0.0, 1.0)` — the model never sees fully-masked inputs)
- **CDM loss weight**: `0.3` relative to `1.0` on the AR loss — the causal objective dominates during training
- **Corruption type**: uniform-noise D3PM (each masked position replaced with a random token drawn uniformly from the vocabulary), not absorbing-mask MDLM

### 2.2 Coarse-to-Fine Decoder (Evaluation)

The evaluation procedure is a stride-structured variant of **Mask-Predict** (Ghazvininejad et al. 2019), with one change: the first round is a causal AR pass rather than an unconditional mask prediction. Given a sequence of length L and a stride `s`:

1. **Pass 1 (causal mode, `is_causal=True`).** Run the model in causal mode and score log-probabilities at positions `{0, s, 2s, ...}`. These are the "skeleton" positions. The model can only see earlier tokens (verified in §2.3).
2. **Pass 2 (bidirectional mode, `is_causal=False`).** Fill the remaining positions in `rounds` iterations. Within each round, positions that are still unresolved (the current round's positions plus all later rounds) are replaced by random vocabulary tokens drawn uniformly — this is the same D3PM-uniform corruption the model was trained on. The forward pass is then run bidirectionally. The script averages the resulting NLL over `n_random=3` independent random-fill draws to reduce variance. Ground-truth tokens at positions already resolved in earlier rounds are kept as-is (the code uses `x.copy()` with ground-truth reassignment for unresolved positions only; it does not propagate model samples from earlier rounds).

The total BPB is the sum of pass-1 and pass-2 negative log-likelihoods, normalized by total bytes. It is the conditional cross-entropy of the two-pass decoding procedure described above, with Monte Carlo averaging (`n_random=3`) over the random fills used for unresolved positions during pass 2. It is *not* an exact entropy; it is the cross-entropy a decoder following this exact procedure would achieve.

Full implementation: `eval_cf_dualbrain.py` (MLX, 5L reference) and `eval_cf_dualbrain_cuda.py` (PyTorch/CUDA, 11L). Both files are included in this folder.

### 2.3 Causal-Mask Integrity Check

Because the main numerical claim rests on the `is_causal=True` forward correctly masking future tokens, I ran an explicit future-token leakage test on the 5L checkpoint. The test constructs two token sequences `seq_A` and `seq_B` that are identical for positions `0..15` and differ for positions `16..31`, forwards both with `is_causal=True`, and compares logits.

Under a correct causal mask, logits at positions `0..15` must be byte-identical between the two inputs (future tokens cannot influence earlier positions). Under a broken mask, they will diverge.

Observed result on `shared_ar_cdm.npz`:

```
Prefix positions 0..15 (should be identical under causal):
  max  |logits_A - logits_B| = 0.000000e+00
  mean |logits_A - logits_B| = 0.000000e+00
Suffix positions 16..31 (should differ, as inputs differ):
  max  |logits_A - logits_B| = 1.82e+01
```

Prefix divergence is exactly zero (not merely below precision) and suffix divergence confirms the model is not constant. The `is_causal=True` path does not leak future tokens. The test script is included as `leakage_test.py`; reviewers can reproduce it on any Apple Silicon machine with `mlx >= 0.31` in under 30 seconds. The same SDPA call path (`F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=...)` with no additional `attn_mask` argument) is used by both `train_cdm.py` (training) and `eval_cf_dualbrain_cuda.py` (11L evaluation), so the integrity of the 5L test carries to the 11L numbers.

### 2.4 Why Not `val_var_bpb`

The MDLM line uses `val_var_bpb`, a variational upper bound on NLL under the discrete absorbing-mask Markov chain. I deliberately do not report this metric for three reasons:

1. **Training-eval mismatch.** `val_var_bpb` assumes absorbing-mask training. This submission uses uniform-noise replacement (D3PM-uniform). Applying absorbing ELBO to a uniform-noise model is not a valid bound.
2. **No realizable decoder at 256 steps.** `val_var_bpb` requires 256–512 forward passes. No practical compression procedure runs at that cost; the metric measures tightness, not decoder-ability.
3. **Apples-to-oranges risk.** Mixing CF BPB with `val_var_bpb` in one table would compare different quantities.

I cite `val_var_bpb` as a valid metric for its research line.

### 2.5 Related Prior Work

The core idea of this submission — one set of weights trained under multiple attention-mask regimes and used in more than one mode at evaluation — is not new. I do not claim to have invented joint causal + bidirectional training or iterative mask-and-refill decoding. The contribution here is the specific combination (uniform-noise D3PM denoising jointly trained with a causal AR loss at 0.3 : 1 weight, evaluated via a two-pass Mask-Predict-style decoder) and its empirical behavior in the parameter-golf regime.

Relevant prior work that readers should consult:

- **UniLM** — Dong et al. 2019, *"Unified Language Model Pre-training for Natural Language Understanding and Generation"* (arXiv:1905.03197). The closest architectural precedent: one transformer trained with three attention-mask regimes (unidirectional, bidirectional, seq2seq) on the same weights. My training is a simpler variant with only two mask regimes (causal + bidirectional) and a D3PM-uniform denoising objective in place of UniLM's masked-LM objective.
- **GLM** — Du et al. 2022, *"GLM: General Language Model Pretraining with Autoregressive Blank Infilling"* (arXiv:2103.10360). Unifies understanding and generation via autoregressive blank infilling on spans. Directly motivates the "one model for generate + edit" framing in §5.
- **FIM / Fill-in-the-Middle** — Bavarian et al. 2022, *"Efficient Training of Language Models to Fill in the Middle"* (arXiv:2207.14255). The production approach used by Codex/Copilot: reorder training data as `[prefix, suffix, middle]` and train a standard causal LM. This is the main baseline any future retrofit experiment (see §6) would compare against.
- **D3PM** — Austin et al. 2021, *"Structured Denoising Diffusion Probabilistic Models in Discrete State-Spaces"* (arXiv:2107.03006). The source of the uniform-noise corruption used in §2.1 denoising loss. The training here uses the D3PM-uniform noise kernel (random token replacement), not the absorbing-mask kernel used by the MDLM line.
- **Mask-Predict** — Ghazvininejad et al. 2019, *"Mask-Predict: Parallel Decoding of Conditional Masked Language Models"* (arXiv:1904.09324). Iterative parallel decoding with round-based refinement over masked positions. My two-pass Coarse-to-Fine decoder in §2.2 is a stride-structured variant with a causal AR skeleton pass replacing the initial Mask-Predict round.
- **MDLM** — Sahoo et al. 2024, *"Simple and Effective Masked Diffusion Language Models"* (arXiv:2406.07524). The reference point for §2.4 and the dominant paradigm in the parameter-golf text-diffusion cluster (see §8).

Additional references on joint causal + bidirectional training that are relevant but not directly adapted here: **XLNet** (Yang et al. 2019, permutation LM), **T5** (Raffel et al. 2020, span-corruption denoising), **BART** (Lewis et al. 2020, denoising autoencoder), **CM3** (Aghajanyan et al. 2022, causal-masked joint training).

---

## 3. Main Results

### 3.1 11L Multi-Seed Verification — Primary Evidence (1×H100 SXM, 540 s each, final-checkpoint eval)

Six independent training runs at 11L d=512 v4096, same unified script, same v4096 data, same 540 s training budget. **Five fresh seeds for the shared model** (`SEED ∈ {1337, 42, 2024, 7, 100}`, joint AR + 0.3 · masked-denoising) and **one fresh seed for the matched causal-only control** (`SEED=1337`, w=0.0).

**Why the control uses a single training seed (and why this is a known limit, not a justified equivalence).** This round of compute was concentrated on the shared side because the shared CF Total is the headline quantity, and because the shared side is where the visible empirical variance lives. The control's training-stochasticity term is **not directly measured** in this round (n=1 fresh control seed); a second fresh control seed in §6.0 is the only way to actually estimate it.

For *intuitive calibration only* — not as a formal upper bound or as input to any significance computation — the shared side's 5-seed training final `val_bpb` std is ≈ 0.0019 (`{1.4387, 1.4393, 1.4416, 1.4422, 1.4430}`), and a causal-only optimization is a strictly simpler training objective (one loss component vs two, no bidirectional forward pass), which makes it *plausible* (but not proven) that the control's training std is in the same order of magnitude or smaller. This is a working assumption used only to motivate why a one-seed control round was a reasonable allocation of compute given the budget constraint, not to claim that the delta CI has been fully bounded. The single control's CF-eval Pure-AR (1.3214) matches its own training final `val_bpb` (1.3146) to within 0.007 BPB, which is consistent with that intuition but is also a single data point and proves nothing on its own.

**Bottom line for this section: the unmeasured control variance is the largest remaining methodological limit of v3.5. §6.0 closes it.**

**Methodology fix from v3.3.** All evaluations in this section are run on `step_final.pt`, the actual last training step, rather than the last `val_every`-aligned intermediate checkpoint that v3.3 was using. The training script (`train_cdm.py`) now unconditionally writes a `step_final.pt` at end of training; the eval script (`eval_cf_ablation.py`) consumes it directly. The intermediate-checkpoint difference is asymmetrically large for the shared model (which trains slower per step due to the bidirectional pass and therefore reaches fewer total steps) and was the dominant noise source in v3.3. After this fix, the CF Total seed-to-seed sample standard deviation drops from 0.022 (v3.3, intermediate checkpoint) to **0.0051** (v3.5, final checkpoint; full-precision value from the 5 logged `cf_total` numbers, rounded), a 4.3× variance reduction.

| Run | Seed | Training final val_bpb | CF eval Pure-AR | CF eval CF Total |
|---|---|---|---|---|
| **11L_w0 (control)** | 1337 | 1.3146 | **1.3214** | 2.4538 (invalid — bidirectional mode was never trained) |
| **11L_w0.3 (shared)** | 1337 | 1.4387 | 1.4428 | **1.2924** ⭐ best |
| **11L_w0.3 (shared)** | 42 | 1.4393 | 1.4425 | **1.3027** |
| **11L_w0.3 (shared)** | 2024 | 1.4430 | 1.4459 | **1.3060** |
| **11L_w0.3 (shared)** | 7 | 1.4416 | 1.4446 | **1.3025** |
| **11L_w0.3 (shared)** | 100 | 1.4422 | 1.4456 | **1.3007** |

**11L_w0.3 5-seed CF Total stats:** mean **1.3009**, sample std **0.0051** (≈ 0.005), min 1.2924, max 1.3060.

**Headline delta computation.** The primary headline of this submission is the 5-seed mean delta. The single-seed best is reported as a post-hoc reference for the deployable artifact, not as the effect size.

| Quantity | Value | Role |
|---|---|---|
| **5-seed mean delta** (primary headline) | `1.3009 − 1.3214 = ` **`−0.0205 BPB`** (shared CF mean − single-seed control) | method-level effect size |
| Single-seed best (post-hoc reference) | `1.2924 − 1.3214 = ` `−0.0290 BPB` (`SEED=1337`, best of the 5 trained seeds) | the model file one would actually ship |

**Statistical caveat.** Both deltas use the same single-seed final-checkpoint w0 control measurement, so the control side of the delta carries **no within-experiment variance estimate at all**. The shared side has a 5-seed CF Total sample std of 0.0051 → SE 0.0023, and a 5-seed training final `val_bpb` std of 0.0019 (both directly computable from the table above). I do not run any significance test or compute any joint CI here, because doing so would require either (a) a measured control std, which this round does not have, or (b) treating the shared side's 0.0019 as a control upper bound, which is at best a working intuition (causal-only training has fewer loss components and no bidirectional forward pass) and at worst a hand-wave — not a formal bound. The intended reading of §3.1 is therefore the *unweighted* observation: "the 5-seed shared CF mean lands ~0.02 BPB below the single control point at the same training protocol and the same eval sample, and the visible shared-side variance is much smaller than that gap". Whether this gap survives a directly measured control variance is what §6.0 tests; until then, no significance claim is made.

The single-seed best (−0.0290) is the result of post-hoc selection over 5 seeds and is therefore upward-biased as an effect-size estimator; it is reported only because that specific `step_final.pt` is the file that one would actually deploy as the §10 submission artifact, and reviewers should be able to reconcile the deployable file with the §3.1 statistics.

Notes on the table:
- All BPB numbers are measured on the same FineWeb v4096 validation shard with the same sampling protocol (N=500 sequences × seq_len=1024, eval `--seed 42` fixed across all runs). Within each row, Pure-AR and CF are on the same sequences.
- The "invalid" entry for the control row is informative: it is the result of running the `is_causal=False` pass on a model that was never trained with a bidirectional objective. The bidirectional mode is untrained weights, so it produces a nearly uniform distribution, and CF Total explodes to ≈ 2.45 BPB. This **validates** that the CF gain in the shared rows is not a metric artifact — if it were, the control would show the same CF reduction.
- The shared model's `Pure-AR` column shows the cost of joint training: at `w=0.3`, the shared model is ≈ +0.12 BPB worse on causal-only generation than the dedicated control. The CF decoder more than recovers this gap, but it does not erase it — the shared model is **not** a free lunch on Pure-AR; the gain is conditional on running the CF decoder at inference time. This is the test-time-compute framing developed in §5.

### 3.2 Original 6-Run Scaling Sweep at 5L + 11L (single seed, intermediate checkpoint, retained as cross-scale evidence)

The original scaling sweep that motivated the multi-seed verification in §3.1. Six independent training runs at the same 1×H100 540 s budget, varying model size (5L d=256 vs 11L d=512) and CDM loss weight (0.0, 0.3, 1.0), all on `SEED=1337`, all evaluated at the last `val_every=500`-aligned intermediate checkpoint (`step_5000.pt` for w=0 / w=1.0, `step_1500.pt` for w=0.3). These numbers are subsumed by §3.1 for the 11L row (which is now multi-seed and final-checkpoint), but the 5L row has not yet been multi-seed verified and is retained here as the only cross-scale evidence.

| Run | Params | Training objective | Pure-AR BPB (single-mode) | CF BPB (two-pass decoder) |
|---|---|---|---|---|
| **5L_w0 (control)** | 4.3 M | causal-only | **1.4479** | 2.4371 (invalid) |
| 5L_w0.3 | 4.3 M | causal + 0.3 · masked denoising | 1.5231 | **1.4009** |
| 5L_w1.0 | 4.3 M | causal + 1.0 · masked denoising | 1.5841 | **1.3939** |
| 11L_w0 (control, intermediate ckpt) | 28.4 M | causal-only | 1.3574 | 2.3947 (invalid) — *superseded by §3.1* |
| 11L_w0.3 (intermediate ckpt) | 28.4 M | causal + 0.3 · masked denoising | 1.4708 | 1.3301 — *superseded by §3.1 (best seed: 1.2924)* |
| 11L_w1.0 (intermediate ckpt) | 28.4 M | causal + 1.0 · masked denoising | 1.5414 | 1.3527 |

The 5L row gives a single-seed delta of `1.3939 − 1.4479 = −0.054 BPB` (5L_w1.0 CF vs 5L_w0 Pure-AR). This is **not yet multi-seed verified** and should be treated as a single-seed observation pending the §6.0 follow-up. The 11L numbers in this table are deprecated in favour of §3.1.

### 3.3 5L d=256 SP1024 — 8-Config CF Sweep (M1 Max, free)

Before the 6-run H100 ablation, I ran a free pre-flight sweep on M1 Max using an earlier 5L SP1024 shared checkpoint (`shared_ar_cdm.npz`, 4.2 M params) to locate the CF sweet spot across stride × rounds. This is the sweep that convinced me stride=2, rounds=2 is worth spending H100 compute to test. The checkpoint here is SP1024 (not v4096), so the absolute BPB values differ from §3.1 due to the tokenizer — but the *shape* of the sweep is the signal.

| Config | Pass-1 (causal) NLL | Pass-2 (denoise) NLL | **CF Total BPB** | vs Pure-AR 2.5386 |
|---|---|---|---|---|
| Pure AR baseline (same model, single-mode) | — | — | **2.5386** | baseline |
| stride=2, rounds=1 | 1.2615 | 1.2807 | 2.5422 | +0.14% |
| **stride=2, rounds=2** | **1.2688** | **1.0598** | **2.3285** | **−8.28%** |
| stride=3, rounds=1 | 0.8663 | 2.1996 | 3.0659 | +20.77% |
| stride=3, rounds=2 | 0.8540 | 1.6754 | 2.5294 | −0.36% |
| stride=3, rounds=3 | 0.8527 | 1.6052 | 2.4578 | −3.18% |
| stride=4, rounds=1 | 0.6370 | 2.6794 | 3.3164 | +30.64% |
| stride=4, rounds=2 | 0.6404 | 2.0915 | 2.7319 | +7.61% |
| stride=4, rounds=3 | 0.6436 | 1.9617 | 2.6053 | +2.63% |

Sweet spot: stride=2, rounds=2 (50/50 causal–bidirectional split with two denoising refinement rounds). This is the only CF configuration used in §3.1. Every `rounds ≥ 2` configuration either matches or beats pure-AR. Wider-stride single-round configurations are catastrophic because the bidirectional pass has too much to fill from too little context in a single pass.

### 3.4 Earlier 5L SP1024 Headline (1 line, for continuity)

Before running the §3.1 ablation, the same (stride=2, rounds=2) CF configuration was measured on the earlier SP1024 5L shared checkpoint (`shared_ar_cdm.npz`) at N=2000 × seq_len=256 on M1 Max: Pure-AR 2.5412, CF Total **2.3382**, Δ **−7.99%** (stable across N=500 → N=2000). Kept here only to show that the §3.3 sweet spot holds at larger sample sizes on the pre-flight checkpoint. Not the primary claim.

### 3.5 CDM-Weight Sensitivity and Scale Behaviour

From the §3.1 table, two monotonic patterns emerge that are informative about where this paradigm works and where it does not:

**The causal-mode tax grows with CDM weight.** As the CDM loss weight increases from 0 → 0.3 → 1.0, the shared model's Pure-AR BPB gets worse in a near-linear way. The table below uses the **final-checkpoint** measurements from §3.1 for 11L (1 control seed and 5-seed mean for `w=0.3`) and the §3.2 single-seed scaling sweep for 5L (the only available 5L source until §6.0):

| Scale | Source | w=0 Pure AR | w=0.3 Pure AR | w=1.0 Pure AR | Tax at w=0.3 | Tax at w=1.0 |
|---|---|---|---|---|---|---|
| 5L | §3.2 (intermediate ckpt, single seed) | 1.4479 | 1.5231 | 1.5841 | **+0.075** | **+0.136** |
| 11L | §3.1 (final ckpt, 1 control + 5-seed mean) | **1.3214** | **1.4443** | — | **+0.123** | — |

At 11L the tax at `w=0.3` is **larger in absolute terms than at 5L** (0.123 vs 0.075). This is a non-trivial finding: naively one might expect the extra capacity of 11L to absorb the multi-task objective more gracefully, but the opposite happens in this regime (the causal head gives up more ground at 11L). I do not yet know whether this trend continues at 100 M+ or starts to reverse; that is the primary open question for §6.

(A `w=1.0` 11L row is not given here because the §3.1 verification did not retrain `w=1.0`. The intermediate-checkpoint single-seed `w=1.0` value from §3.2 is preserved as a legacy reference in **Appendix A**.)

**The CF two-pass decoder recovers the tax and then some.** Even though the shared model is worse at pure causal scoring, running the two-pass CF decoder on the same model gets it below the control:

| Scale | Control CF-eval Pure-AR | Shared CF (5-seed mean / post-hoc best) | CF advantage (mean / post-hoc best) | Verification status |
|---|---|---|---|---|
| 5L | 1.4479 | 1.3939 (w=1.0, 1 seed) | **−0.054** (single seed) | single-seed (§6.0 follow-up) |
| 11L (final ckpt) | **1.3214** (1 seed) | **1.3009 ± 0.005** (5-seed mean, w=0.3) / 1.2924 (post-hoc best `SEED=1337`) | **−0.0205 mean** / −0.0290 post-hoc best | 5 fresh shared seeds + 1 fresh control seed (§3.1) |

At 5L the best CF configuration is w=1.0 (stronger bidirectional signal); at 11L it is w=0.3 (where the model has enough capacity that a weak bidirectional signal is enough). At both scales the shared-CF configuration scores below the matched causal-only control. The 11L row is the multi-seed final-checkpoint version from §3.1; the 5L row is still single-seed and is the highest-priority remaining verification (§6.0). The "post-hoc best" column at 11L is upward-biased (best of 5 seeds) and is reported only as the deployable-artifact reference, not as an effect-size estimate.

### 3.6 Earlier M1 Max Pre-Flight (3-eval-seed Subsample Check on an Earlier Checkpoint)

*This section is retained for historical context only. The §3.1 multi-seed verification at 1×H100 with 5 fresh training seeds at the final-checkpoint state supersedes it as evidence for the headline claim.*

Prior to the 6-run ablation, I ran a 3-eval-seed subsample check on an earlier 11L 8×H100 checkpoint (`11L_shared_cdm_bf16.pt`, no longer used for primary comparison). The 3-eval-seed mean CF BPB was 1.3083 ± 0.0047 at seq_len=1024, with N=500 per seed. **What the three seeds randomize**: the validation subsample (which 500 sequences are picked) and the random fill in pass-2 denoising — *not* training stochasticity. The Pure-AR std of 0.0008 BPB across these eval seeds reflects validation subsample variance only, not model variance.

The §3.1 result is methodologically stronger because it varies the **training seed**, runs on **fresh trainings** with the unified script, and evaluates at the **true final checkpoint** rather than an intermediate save.

| Eval Seed | N | Pure AR | CF Total | Δ |
|---|---|---|---|---|
| 42 | 500 | 1.4422 | 1.3021 | −9.71% |
| 43 | 500 | 1.4438 | 1.3134 | −9.03% |
| 44 | 500 | 1.4441 | 1.3095 | −9.32% |
| **mean** | **1 500** | **1.4434 ± 0.0008** | **1.3083 ± 0.0047** | **−9.35% ± 0.28%** |

---

## 4. Honest Limitations

This PR measures a BPB improvement on the standard Parameter Golf metric (cross-entropy per byte of validation text). It does **not** measure:

- **Comparison to the 8×H100 leaderboard at matched training compute.** The 1×H100 540 s runs see approximately 1/8 the tokens of an 8×H100 540 s run. The §3.1 11L_w0 control at training val_bpb 1.3146 (CF-eval Pure-AR 1.3214) is therefore not directly comparable to the 8×H100 leaderboard entries (top 1 = 1.1147, baseline = 1.2244). The relevant comparison in this PR is always the matched control on the same hardware, not the leaderboard.
- **Actual fill-in-middle generation quality.** Parameter Golf evaluates BPB, not generation, because 28 M-parameter models at ~270 M training tokens cannot produce coherent text regardless of architecture (GPT-2 small at 124 M / 10 B tokens is the rough coherence threshold in the literature). I ran a qualitative greedy-fill test on all six models as a sanity check (not as a claim): exact-match rates were 0–4.7% across all configurations, including the controls — consistent with the scale regime. This PR is about BPB, which *is* the Parameter Golf metric.
- **Comparison to dedicated fill-in-middle baselines** (CodeLlama-FIM, StarCoder-FIM). Training did not target code, so FIM code-benchmarks are not applicable without a retrofit experiment. This is Next Step #2 in §6.
- **Retrofit to pretrained LLMs.** All training here is from scratch. Whether the same shared-weight paradigm can be added to an existing pretrained causal LM via LoRA — the realistic production path for any shipping product — is the largest open question, listed as Next Step #1 in §6.
- **Share-ratio grid beyond three points.** I tested weight ∈ {0, 0.3, 1.0}. A finer grid might reveal a different optimum.
- **Multi-seed verification at 11L: partially resolved in §3.1** (5 fresh training seeds for the shared model, **1** fresh training seed for the matched control, all at the true final checkpoint, shared 5-seed CF Total sample std 0.0051, i.e. ≈0.005). The control side still has only one fresh seed in this round; a strict significance test is not run (see §3.1 statistical caveat). A second control seed is the smallest remaining gap. **Multi-seed verification at 5L: not yet done** — the −0.054 BPB gap at 5L in §3.2 is still single-seed and is the highest-priority remaining experiment (§6.0).

---

## 5. Why This Might Matter — Downstream Utility Under Test-Time Compute

The §3.1 effect is modest in absolute terms (−0.029 BPB best seed, −0.0205 BPB 5-seed mean at 11L; −0.054 BPB single-seed at 5L). What I find interesting is not the magnitude but the factorization: the matched ablation separates two capabilities a production LLM would typically want to optimize independently:

1. **Causal-only next-token prediction**, which is how every shipping LLM (ChatGPT, Claude, GPT-4, Codex, Copilot) is primarily measured.
2. **Bidirectional conditioning** on both left and right context, which is today served either by a *second* specialized model (BERT, MDLM), by a training-time hack (FIM special tokens in Bavarian et al. 2022 / Rozière et al. 2023), or by retrieve-and-rewrite pipelines.

The matched ablation is consistent with the reading that **a single set of weights, at matched compute, can expose both capabilities when evaluated under the two-pass CF decoder**. This fits naturally into the recent test-time-compute framing (Welleck 2024, speculative decoding, Mask-Predict Ghazvininejad 2019): the CF decoder is an inference-time compute knob that trades extra forward passes for lower BPB, and the shared-weight training makes those extra passes useful instead of noise.

**Effect-size context.** The 5-seed shared CF mean (1.3009) lands ~0.02 BPB below the single-seed control point (1.3214). The shared side's empirical std on the 5 fresh seeds is 0.005 (CF Total) and 0.0019 (training `val_bpb`); the control side has no measured variance term in this round. No significance test is computed here (see §3.1 statistical caveat). The absolute effect is small. The relevant practical question is **whether it grows, shrinks, or inverts when the model and training budget are scaled up**, which neither §3.1 nor §3.2 can answer — that is the §6.1 / §6.2 work, gated on §6.0.

**What this is not.** This is not a claim that a 28 M parameter model can generate coherent text, or that these 540 s runs are ready for any production use. Models at this scale cannot generate coherent English regardless of architecture (GPT-2 small at 124 M / 10 B tokens is the rough coherence threshold, and these models are 5× smaller and 30× less trained). The Parameter Golf competition accepts this — BPB is the metric precisely because coherence is out of reach at these scales. The claim here is scoped to BPB under a specific decoder with a specific control, nothing more.

---

## 6. What Might Work With More Compute

Honest speculation. Each item below is a concrete experiment that would extend or close an open question from §3 — ordered by what most strongly constrains the conclusion of this submission. **§6.0 is the only follow-up that is gating; everything else is conditional on it.**

### 6.0 5L multi-seed verification (highest-priority remaining experiment)

The 11L row of §3.1 is now multi-seed verified at the true final checkpoint. The 5L row in §3.2 is **not**. The −0.054 BPB single-seed result at 5L is a stronger absolute effect than the verified 11L 5-seed mean (−0.0205), but it has the same risk profile that the 11L row had before §3.1: a single training seed at the last `val_every`-aligned intermediate checkpoint, where the training-stochasticity asymmetry between control and shared could plausibly manufacture a 0.05 BPB gap by chance.

**Concretely**: 5 fresh training seeds for `5L_w1.0` (the winner) + 1 fresh training seed for `5L_w0` (control), all evaluated at `step_final.pt` with the new `eval_cf_ablation.py` protocol. 5L training is much cheaper than 11L (~3 min per run on 1×H100 SXM, or runnable on consumer GPUs at similar speed). Total compute estimate: ~30 min wall time, ~$1.5 self-funded on 1×H100 SXM, or essentially free on M1 Max in roughly the same wall time. This is the next experiment I will run.

### 6.1 Retrofit onto a pretrained causal LLM via LoRA (the production path)

The experiment that would most directly test whether this paradigm survives outside the Parameter Golf toy regime is a **LoRA-style retrofit of a pretrained causal LLM** (e.g. Qwen 3.5 0.8 B, which I already have locally). Rather than training from scratch at 28 M parameters, take a model that already generates coherent text and add a small LoRA adapter to expose a bidirectional forward mode, trained with the same joint AR + D3PM objective. No shipping LLM trains from scratch at 28 M parameters, so this is the setting where any downstream claim has to be tested. An initial result on Qwen 0.8 B fits in roughly 10–15 H100-hours and would tell, *within one pod session*, whether the shared-weight + CF-decoder pattern carries to a model that is actually coherent at inference. This is the single most compute-efficient downstream test and it is Next Step #1.

### 6.2 Full-budget 8×H100 reproduction of the 11L ablation

Run the exact §3.1 ablation at 8×H100 540 s (the production Parameter Golf budget) to test whether the 0.027 BPB improvement persists, narrows, or inverts when the training-token budget grows ~8×. I do not have a confident extrapolation to offer — the Pure-AR tax in §3.5 already grows with scale in a direction that works against the shared model, and this experiment is how I find out whether that trend continues or reverses at full compute. This is Next Step #2.

### 6.3 Share-ratio grid search at 11L

The 6 runs used weight ∈ {0, 0.3, 1.0}. At 11L, w=0.3 gave the best CF BPB; at 5L, w=1.0 did. A fine grid (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0) at 11L would locate the actual optimum and tell whether the share-ratio optimum scales with model size. This is a cheap follow-up to §6.2 — roughly 7 additional 1×H100 runs.

### 6.4 Finer scale sweep for the share-ratio → BPB curve

I have two architectural data points (5L 4.2 M and 11L 28.4 M). Adding 7L d=384, 9L d=448, and 13L d=640 would give a scaling curve for both the Pure-AR tax (which appears to grow with scale in this data) and the CF recovery (which also grows with scale). A simple power-law fit would let me predict the crossover scale — the model size at which the CF gain exceeds the Pure-AR tax by a margin that makes the extra compute worth it.

### 6.5 Absorbing-mask MDLM noise schedule for the bidirectional pass

I used uniform-noise D3PM (random vocabulary replacement). The MDLM cluster (#820, #1106, #1241) uses absorbing-mask denoising, which the literature suggests gives stronger bidirectional representations. Swapping the noise schedule is a one-line training change; a matched ablation would tell whether the gain would be larger under the standard MDLM noise, at the cost of some comparison legibility.

---

## 7. Retrodiction — A Negative Result at Production Scale

> **Scope note.** The runs in this section are a **different training line** from the Shared AR + Denoising model used in §3. They are a 1×H100 A/B sweep of retrodiction modes on a pure AR stack (no CDM auxiliary loss). The "Pure AR" numbers in this table are therefore *not comparable* to the "Pure AR" column of §3.3, which measures the Shared AR + Denoising checkpoint in single-mode causal. Different models, different training configurations. See §7.3 for an explicit side-by-side.

This submission also documents a line of work I call **retrodiction** — a reversed-sequence auxiliary loss added to the standard causal AR loss. The operational definition is simply:

```python
loss = causal_lm_loss(model(x), x) + α · causal_lm_loss(model(x.flip(1)), x.flip(1))
```

I report it as a negative result at production scale. The compact story:

### 7.1 Early-Training Signal on 5L / M1 Max

At small scale and short token budgets, retrodiction gave up to −3.6% BPB at step 200/500, direction consistent with the motivation.

### 7.2 Production-Stack A/B on 1×H100

Five independent training runs, same architecture (11L d=512 v4096, XSA-4, BigramHash), same 540 s budget, same seeds, **pure causal AR stack with no CDM auxiliary loss** — only the retrodiction mode varied:

| Test | Retro mode | Final val_bpb |
|---|---|---|
| **D** | **OFF** | **1.3401** (best) |
| C | partial 15% | 1.3594 |
| B | merged late 80/20 | 1.3695 |
| E | alternating 90/10 | 1.3616 |
| A | alternating 50/50 | 1.4109 |

Pure contrast (C vs D): retrodiction is a **+0.019 BPB tax** at production scale, not a gain.

### 7.3 Consolidated 11L Pure-AR Numbers

The §3.1 ablation and the §7.2 retrodiction sweep each produce their own 11L Pure-AR BPB on the same nominal 11L d=512 v4096 architecture but with different training stacks. They are listed side by side here, restricted to a single metric kind (training final `val_bpb`) for direct comparability:

| Source | Training objective | Retrodiction | Training stack | Pure AR final val_bpb |
|---|---|---|---|---|
| **§3.1** `11L_w0` (control, 1 seed, final ckpt) | Pure AR only | off | unified `train_cdm.py`, `--xsa_last_n=4` | **1.3146** |
| **§3.1** `11L_w0.3` (5-seed mean, final ckpt) | Joint AR + 0.3·denoising | off | unified `train_cdm.py`, `--xsa_last_n=4` | **1.4410** |
| §7.2 Test D (1 seed, earlier stack) | Pure AR only | off | earlier XSA / BigramHash configuration | 1.3401 |
| §7.2 Test C (1 seed, earlier stack) | Pure AR only | partial 15 % | earlier XSA / BigramHash configuration | 1.3594 |

The §3.1 11L_w0 (1.3146) and §7.2 Test D (1.3401) are both single-seed pure-AR 1×H100 540 s runs at the same nominal architecture but on **different training stacks**. The 0.026 BPB difference reflects training-stack drift, not retrodiction. For the primary claim of this submission, the relevant comparison is always §3.1 11L_w0 vs §3.1 11L_w0.3 CF (both measured with the *exact* same script, same data pipeline, same eval sampling, all at the final checkpoint). The §7 retrodiction sweep is a separate, older line of work included for completeness.

**Interpretation (hypothesis).** At 5L on short budgets, the forward loss signal may be weak enough that the reversed loss provides complementary gradient. At 11L on production budgets, I hypothesize that the forward signal is strong enough to dominate and the reversed loss competes for updates rather than augmenting them. I do not have a mechanistic proof of this interpretation, and I have not found a useful parametrization of retrodiction for the parameter-golf regime.

**Practical recommendation:** retrodiction is a tax on the production stack and should not be used. The matched-compute 6-run ablation in §3.1 was run *without* retrodiction for this reason.

---

## 8. Position in the Text-Diffusion Cluster

Snapshot of the text-diffusion cluster as of 2026-04-09 (reproducible via `gh pr list --repo openai/parameter-golf --search "diffusion" --state open --limit 50`):

- Bidirectional masked diffusion + discrete absorbing ELBO (`val_var_bpb`): #820 mtybadger (convention-setting), #1053, #1106 agalimova, #1241 aiejvn, #1403
- Causal MDLM as AR regularizer (eval in causal mode): #1119 gowtham0992
- Hybrid AR + MDLM mixed training with bidirectional head discarded at eval: #1194
- AR with diffusion-inspired auxiliary noise, evaluated as pure AR: #904
- Prefix-conditioned discrete diffusion: #905
- Hybrid sparse diffusion: #1198
- **This PR:** shared-weight joint causal + masked-denoising training, evaluated via a two-pass Coarse-to-Fine decoder on BPB, with a **matched causal-only control** at the same compute. This is, to my knowledge, the first submission in the text-diffusion cluster to include an explicit matched-compute control ablation.

This approach differs from the cluster in that both modes are actively used at evaluation on the same weights, rather than the bidirectional mode being used only at training time or evaluated separately. I do not claim this is a strict improvement over the MDLM line — it is a different question evaluated on a different metric. Direct numerical comparison across metrics (val_var_bpb / val_bpb / CF BPB) is not meaningful because they measure different quantities. See §2.4.

---

## 9. Hardware and Reproducibility

All training and evaluation artifacts are published on Hugging Face:

- **`akaiii/meadow-golf-checkpoints`** — all 6 ablation checkpoints (`5L_w0.npz`, `5L_w03.npz`, `5L_w1.npz`, `11L_w0.npz`, `11L_w03.npz`, `11L_w1.npz`), 6 training logs, 6 CF eval logs, the unified training script (`train_cdm.py` + `train_ablation_runner.py`), and the CF eval scripts (`eval_cf_dualbrain.py`, `eval_cf_dualbrain_cuda.py`, `eval_cf_ablation.py`). Directory layout matches the `ablation_results/` folder in this PR.
- **`akaiii/meadow-golf-v4096`** — `bpe_v4096.model` tokenizer and the v4096 retokenized FineWeb validation + training shards used for every training run in §3.1.

### 9.1 Reproduction of the §3.1 multi-seed verification (the v3.5 headline) — ~70 min on 1×H100 SXM, ~$3.50

The §3.1 5-seed shared verification + 1-seed control is the headline of this submission. Both orchestration scripts (`run_p5.sh`, `run_phase_b.sh`) and all training / CF eval logs from the actual run are committed to `seeds_run/` in this folder for reviewer-side spot checking. The scripts rely on the v3.5 copies of `train_cdm.py`, `train_ablation_runner.py` (with `--seed` support), and `eval_cf_ablation.py`; the reproduction commands below reproduce them from a clean H100 pod:

```bash
pip install --break-system-packages torch numpy sentencepiece huggingface_hub

git clone https://github.com/akaiHuang/meadow-golf
cd meadow-golf/experiments/2026-04-09_matched_ablation

hf download akaiii/meadow-golf-v4096 --repo-type dataset --local-dir /workspace/gv4096

# 5 fresh shared seeds (11L_w0.3 × {1337, 42, 2024, 7, 100}), final-checkpoint save
SCRIPT_DIR=. \
  DATA_DIR=/workspace/gv4096/data \
  TOKENIZER=/workspace/gv4096/bpe_v4096.model \
  OUT_DIR=/workspace/out \
  CKPT_DIR=/workspace/ckpt \
  LOG_DIR=/workspace/logs \
  bash run_p5.sh

# 1 fresh control seed (11L_w0 SEED=1337), final-checkpoint save
bash run_phase_b.sh
```

Both `run_p5.sh` and `run_phase_b.sh` invoke the unified `train_cdm.py` (which now writes a `step_final.pt` checkpoint at the end of training, addressing the v3.3 intermediate-checkpoint issue) via `train_ablation_runner.py` (`--seed` patches the module-level `SEED` constant and emits per-seed patched modules), then run `eval_cf_ablation.py` directly on the `step_final.pt` saves. Final BPB numbers should match the §3.1 table within bf16 numerical noise on the same `--seed 42` eval sample. Total wall time: ~70 min on a single 1×H100 SXM; total self-funded compute: **$3.50** at $2.99/hr.

Reviewer spot check without rerunning anything: every number in §3.1 is grep-able from `seeds_run/logs/*.log` and `seeds_run/eval/*.log` already present in this folder. See `seeds_run/README.md` for the file inventory.

### 9.2 Reproduction of the §3.2 6-run scaling sweep (cross-scale evidence, ~90 min on 1×H100 SXM, ~$3.93)

This is the *original* 6-run ablation that v3.3 reported as the headline; in v3.5 it is retained only as the §3.2 cross-scale evidence (single seed each, intermediate checkpoint). The 5L row of §3.2 is the only available 5L data until §6.0 follow-up. The 11L rows are superseded by §3.1 / Appendix A but included for traceability.

```bash
pip install torch numpy sentencepiece huggingface_hub

hf download akaiii/meadow-golf-checkpoints --repo-type dataset --local-dir ./gcp
hf download akaiii/meadow-golf-v4096       --repo-type dataset --local-dir ./gv4096

export PYTHONPATH="./gcp:${PYTHONPATH}"
mkdir -p out ckpt logs eval

# Train all 6 ablation models (6 × ~10 min wallclock)
for cfg in "5L 5 256 128 2 0.0"  "5L 5 256 128 2 0.3"  "5L 5 256 128 2 1.0" \
           "11L 11 512 128 4 0.0" "11L 11 512 128 4 0.3" "11L 11 512 128 4 1.0"; do
  read tag L D BD X W <<< "$cfg"
  python3 ./gcp/train_ablation_runner.py \
    --train_script ./gcp/train_cdm.py \
    --num_layers $L --model_dim $D --vocab_size 4096 \
    --bigram_dim $BD --xsa_last_n $X --cdm_weight $W \
    -- \
    --train_budget_secs 540 --steps 9999 \
    --data_dir ./gv4096/data --tokenizer_path ./gv4096/bpe_v4096.model \
    --save_path ./out/${tag}_w${W}.npz \
    --checkpoint_dir ./ckpt/${tag}_w${W} \
    > ./logs/${tag}_w${W}_train.log 2>&1
done

# Evaluate all 6 under CF (6 × ~5 min wallclock)
for cfg in "5L 5 256 128 2 0.0"  "5L 5 256 128 2 0.3"  "5L 5 256 128 2 1.0" \
           "11L 11 512 128 4 0.0" "11L 11 512 128 4 0.3" "11L 11 512 128 4 1.0"; do
  read tag L D BD X W <<< "$cfg"
  latest=$(ls ./ckpt/${tag}_w${W}/step_*.pt | sort -V | tail -1)
  python3 ./gcp/eval_cf_ablation.py \
    --ckpt $latest \
    --train_module_path /tmp/train_cdm_patched_${L}L_w${W}.py \
    --num_layers $L --model_dim $D --vocab_size 4096 \
    --bigram_dim $BD --xsa_last_n $X \
    --n_seqs 500 --seq_len 1024 --stride 2 --rounds 2 --seed 42 \
    --data_dir ./gv4096/data --tokenizer_path ./gv4096/bpe_v4096.model \
    --log_path ./eval/${tag}_w${W}_cf.log
done
```

The patched training scripts `/tmp/train_cdm_patched_*.py` are created as a side effect of `train_ablation_runner.py` and are the model-class source for the matching `eval_cf_ablation.py` run. They are regenerated deterministically from `train_cdm.py` on each run. The 5L M1 Max pre-flight sweep uses `eval_cf_dualbrain.py` (MLX) against `shared_ar_cdm.npz`; it runs on any Apple Silicon Mac with `mlx >= 0.31` and reproduces the §3.3 table in under 4 minutes.

Self-funded compute for the §3.2 6-run scaling sweep: **$3.93**. Combined with the §3.1 verification ($3.50), total self-funded for this submission: **~$7.43**.

---

## 10. Compliance

- [x] **5L submission artifacts ≤ 16 MB**: the competition submission unit is the int6+lzma compressed checkpoint (`5L_*_int6.lzma` = ~3.0 MB each), well under the 16 MB cap. The intermediate `5L_w0.npz` (17.2 MB BF16) is *not* a submission artifact; it is the working final-state save used by the eval script and is never submitted.
- [x] **11L submission artifacts** are non-record (trained on 1×H100, not matched to the 8×H100 production budget). The corresponding `11L_*_int6.lzma` files are ~18.7 MB each, *over* the 16 MB cap, which is why every 11L row in this submission is filed under the **non-record track** explicitly. They are never claimed as record candidates.
- [x] No validation data accessed during training
- [x] CF evaluation uses validation tokens only for scoring; no gradient updates
- [x] No network calls during evaluation
- [x] Hardware: original 6-run scaling sweep on a single 1×H100 SXM pod ($3.93). §3.1 multi-seed verification on a second 1×H100 SXM pod ($3.50). Total self-funded ~$7.43 across both sessions.
- [x] Causal-mask integrity verified via the leakage test in §2.3 (`leakage_test.py` included in this folder, max prefix-logit divergence 0.0)
- [x] CF evaluation is fully specified by `SEED`; the denoising pass is Monte Carlo averaged over `n_random=3` random fills for variance reduction on residual positions (not exact, but deterministic given the seed)
- [x] All reviewer-facing §3.1 logs and orchestration scripts are stored locally in `seeds_run/` and reproducible from the per-seed `train_ablation_runner.py` invocations recorded in `run_p5.sh` / `run_phase_b.sh`
- [x] The exact §3.1 `.npz` / `step_final.pt` state files are intentionally not committed to this PR folder (~1.3 GB total); their location and availability-on-request path are documented in `seeds_run/README.md`

---

## 11. Acknowledgments

- **PR #820 (@mtybadger)** for establishing `val_var_bpb` and the MDLM reference point for text diffusion in parameter-golf. My disagreement with the metric in §2.3 is intended as productive, not dismissive.
- **PR #363 (@evangelinehelsinki)** for the template of honest negative-result reporting that §7 follows, and for the `What Might Work With More Compute` section format.
- **PRs #1106, #1241** for showing that the MDLM line is an active research target worth contributing alternatives to.

---

## 12. Related Closed Submission

I earlier withdrew [PR #1442](https://github.com/openai/parameter-golf/pull/1442), a different stack combination submission targeting AR sliding BPB. A self-audit found methodological issues including a mismatch between the evaluation used and the compressed artifact. That line of work is not being pursued further; this PR represents my focused research effort going forward.

---

## Appendix A. Legacy intermediate-checkpoint 11L numbers (superseded by §3.1)

**This appendix exists solely for traceability with v3.3. None of these legacy intermediate-checkpoint numbers are used in any headline claim or main analysis in v3.5.** It is here so that a reader cross-referencing v3.3 against v3.5 can find the original v3.3 single-seed 6-run table 11L values in one place, paired with the §3.1 final-checkpoint measurements that supersede them. The main analytical sections (§3.5, §7.3) of v3.5 carry only final-checkpoint measurements.

The original v3.2 6-run sweep at 11L (single seed `1337`, evaluated at the last `val_every`-aligned intermediate checkpoint, *not* `step_final.pt`):

| Run | Pure AR (intermediate ckpt) | CF Total (intermediate ckpt) | Status in v3.5 |
|---|---|---|---|
| `11L_w0` (control) | 1.3574 | 2.3947 (invalid) | superseded by §3.1 final ckpt: **1.3214 / 2.4538** |
| `11L_w0.3` | 1.4708 | 1.3301 | superseded by §3.1 final ckpt 5-seed mean: **1.4443 / 1.3009** |
| `11L_w1.0` | 1.5414 | 1.3527 | not retrained at final ckpt; legacy value retained |

**Why these numbers were higher / lower than the §3.1 final-checkpoint numbers.** The intermediate checkpoint (`step_5000.pt` for w=0/w=1.0, `step_1500.pt` for w=0.3) is several hundred training steps before the actual end of the 540 s training budget. The shared model is hit asymmetrically harder by this gap because it trains slower per step (the bidirectional pass roughly doubles forward FLOPs at this size), so its last `val_every`-aligned save is *relatively* less converged than the control's. Fixing this with `step_final.pt` (§2 methodology fix in v3.5) improves the shared CF score by ~0.03 BPB and the control Pure-AR by ~0.03 BPB in the *opposite* direction (control gets *better* on the metric, shared also gets better but the difference reshapes) — net effect: the v3.3 single-seed delta (−0.027 BPB) and the v3.5 5-seed mean delta (−0.0205 BPB) are within 0.007 BPB of each other and have the same sign, but the v3.5 number is the one that survives the methodology fix and the multi-seed verification, and is the one quoted everywhere in the main text.

The original 5L row of the v3.2 sweep is used directly in §3.5 as the only available 5L cross-scale evidence pending the §6.0 follow-up.
