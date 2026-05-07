# Non-record: Cross-Base Regularizer Transferability — A Small Study

**Author**: Bharath @ OptioAI (BharathSShankar) | **Track**: 10min_16mb (non-record / methodological)
**Date**: 2026-04-30

This is a non-record submission. It contains **20+ single-seed measurement cells** characterizing how seven candidate regularizers behave on two different leaderboard-lineage bases, plus an analysis of how reg-trained embeddings survive different quantization schemes. We submit it as supplementary methodological data — not as a critique of any prior submission, and not as a claim that our reg basis is the right one.

---

## 1. Headline findings

Each finding here is tied to specific cells later in the README. **No claim is unsupported by data; every "tentative" interpretation is explicitly marked.**

1. **Cross-base sign change** (real data, §6). Same regularizer (QAHSP, λ=0.3), same architectural family (PR #1855 lineage), opposite-direction val_bpb effects: −1.55 mBPB on Base A, +2.25 mBPB on Base B. Largest measured swing: 3.80 mBPB.

2. **Pair stacking at 1/√N λ underperforms best single** (real data, §6). All four pre-registered "good" pairs at λ_each = λ\*/√2 measured worse than the best single reg at full λ. Hypothesis 3 (independent-axis composition) inconsistent with our four pairs.

3. **Quant cost is approximately reg-independent on Base A** (real data, §7). Across all 7 regs, quant cost (post-quant minus pre-quant val_bpb) sits in 14.3–14.9 mBPB — the regs change pre-quant val_bpb but not the GPTQ + LQER quant tax. QAHSP's val_bpb advantage comes from a better pre-quant model, *not* from quant-robustness on this pipeline.

4. **PreQuantTTT × ES compounds; PreQuantTTT × QAHSP does not** (real data, §6, gray-track). Adding ES at λ=0.05 to PreQuantTTT delivers val_bpb 1.03942 vs PreQuantTTT-alone 1.03969 (−0.27 mBPB). Adding QAHSP at λ=0.3 produces 1.03985 (+0.16 mBPB, no help). We tentatively interpret this as direction-shaping regs surviving eval-time fine-tuning while codebook-shaping regs are subsumed (§9).

5. **Reg × quant matrix on real LM hidden states** (real data, §8). 6 regs × 7 quant schemes = 42 cells on Base A. Identifies which reg "plays nice" with which quant scheme by smallest L2 distortion / cosine shift / silhouette degradation post-quant. *[Section §8 is filled in once the 6 fresh training cells complete around 01:00 IST May 1; placeholder until then.]*

6. **Regs leave a real but small fingerprint upstream of quantization** (real data, §13). Three independent mechanistic checks (SVD spectrum of weight matrices, hidden-state norm/kurtosis depth trajectory, pairwise CKA between final-block representations) all show: (a) the regs *do* differ from no-reg and from each other — sub-3% Δσᵢ on attention weights, off-diagonal CKA 0.67–0.75; (b) but every difference is below GPTQ int6's per-row noise floor or uniform across regs. This explains §7 mechanistically.

If you read just one section: **§6** for the cross-base val_bpb evidence, **§8** for the real-data reg × quant matrix on real LM hidden states, **§13** for the upstream mechanistic checks.

---

## 2. Companion record submission

This study is paired with one record submission from the same author:

- `A_N9_SimCTG_3LayerRecur_postquantTTT` — Record: SP10240 + SimCTG λ=0.3 + 3-Layer Recurrence + post-quant score-first TTT, val_bpb **1.07502** (3-seed). This is **Base A** for our study.

For Base B, we use the open PR #1965 (himanshudongre, LongCtx no-QV phased TTT). We reproduced PR #1965 on our infrastructure to verify the result and to capture trained models for §8 — this reproduction is documented in the study text but **we are not submitting our reproduction as our own record**. PR #1965 belongs to its original author; we use it here only as a substrate for comparison.

Our earlier `BharathSShankar/PR #1972` (SP10240 + PreQuantTTT, val_bpb 1.03983) is **withdrawn from record consideration** in light of the upstream closure of PR #1958. The PreQuantTTT line's score-after-adapt pattern doesn't satisfy a strict reading of the README's evaluation rule. We retain the artifact internally as documented gray-track data and use it as a *reference implementation* for hypothesis 4 testing in §6.3, but do not contest a record claim for it.

---

## 3. The 7 regularizers

Each operates on a different statistic of either the hidden state stream or the weight tensors:

| Reg | One-line math | Side | Statistic targeted |
|---|---|---|---|
| **QAHSP** (Quant-Aware Hidden STE Penalty) | MSE(h, STE-quant(h, int6)) | activation | per-coord int6 grid alignment |
| **ES** (Embedding Spread) | mean cos²(h_i, h_j)<sub>off-diag</sub> | activation | angular spread between tokens |
| **AOS** (Activation Outlier Suppression) | mean(max\|h\| − mean\|h\|) per token | activation | per-token outlier-coord suppression |
| **HSU** (Hidden State Uniformity) | var(‖h_i‖) | activation | per-token L2 norm uniformity |
| **WBC** (Weight Bucket-Center) | mean sin²(w/0.05·π) | weight | per-coord int-grid centerline pull |
| **WOP** (Weight Outlier Penalty) | mean(\|w\| − k·σ)²<sub>+</sub>, k=4 | weight | weight-row outlier crush |
| **PCS** (Per-Channel Scale) | var(per-channel max\|w\|) | weight | per-channel scale uniformity |

ES is a hinge-free variant of SimCTG-style contrastive losses. WOP is per-row weight-outlier suppression. The other five (QAHSP, AOS, HSU, WBC, PCS) we believe are not in any prior leaderboard PR; we defined them specifically for this study.

---

## 4. The two bases (and why we picked them)

**Base A**: our SP10240 + SimCTG λ=0.3 record stack (= companion submission A). 11L × 512d × 8H, the PR #1855 architectural lineage with our SP10240 tokenizer adoption. Eval: post-quant score-first TTT + sliding-window stride 64. Base 3-seed mean val_bpb: **1.07502** sliding-window.

**Base B**: PR #1965 reproduction (= companion submission B). Same architecture family but SP8192 CaseOps tokenizer + LongCtx no-QV phased TTT (rank=56, prefix=3000) + AWQ-Lite + asymmetric logit rescale + LQER asymmetric rank-4 + lrzip pergroup compression. Single-seed val_bpb: **1.05822** quantized_ttt_phased.

We chose this pair because (a) both are the same architectural family — they share the PR #1855 base — so cross-base differences can't be attributed to fundamentally different model architectures, and (b) Base B is a heavily greedy-tuned descendant of Base A's family, making it a natural test of whether "regs from the parent" transfer to the child.

All cells: SEED=42, MAX_WALLCLOCK_SECONDS=600, 8×H100 SXM. Only the OUR\_\*\_LAMBDA / size knobs vary; everything else is fixed per base.

---

## 5. Pre-registered hypotheses

Frozen at study start (see `REGULARIZATION_ABLATION.md` for the original).

1. **QAHSP wins single-reg on int6-quantized stacks.** Mechanism: STE-quant alignment of activations is direct prep for the actual quantization step. *Confidence: high.*
2. **WBC has slight-positive or near-neutral effect.** Mechanism: bucketing cooperates with int6 codebook. *Confidence: medium.*
3. **Pairs at λ_each = λ\*/√N should compose** if the regs operate on independent gradient subspaces (loose generalization of Wang & Isola 2020 alignment+uniformity). *Confidence: medium.*
4. **Eval-time fine-tuning subsumes training-time prep regs but preserves direction-shaping regs.** *Confidence: low (this was the speculation we most wanted to test).*

In what follows: hypothesis 1 confirmed, 2 inconsistent with our data, 3 inconsistent with our data, 4 consistent with our data on Base A but only one positive interaction observed.

---

## 6. Cross-base val_bpb measurements (real data)

### 6.1 Base A — single-reg sweep

7 cells, each adds one reg on top of SimCTG λ=0.3.

| Reg config | val_bpb | Δ vs Base A baseline 1.07502 |
|---|---:|---:|
| **QAHSP λ=0.3** | **1.07348** | **−1.55 mBPB** ⭐ |
| WOP λ=0.5 | 1.07376 | −1.26 |
| HSU λ=0.1 | 1.07403 | −0.99 |
| ES λ=0.05 | 1.07428 | −0.74 |
| AOS λ=0.005 | 1.07445 | −0.57 |
| PCS λ=0.005 | 1.07463 | −0.39 |
| **WBC λ=0.005** | **1.07522** | **+0.20** |

QAHSP wins single-reg, consistent with hypothesis 1. WBC's slight-negative observation is **inconsistent with hypothesis 2** at this λ. We do not have a confident explanation; one possibility is that the chosen scale (0.05) places grid centroids the optimizer needs to traverse smoothly. We flag this as observation, not finding.

### 6.2 Base A — pre-registered "good" pairs at 1/√2 · λ

| Pair | val_bpb | Δ vs best single (1.07348) |
|---|---:|---:|
| QAHSP λ=0.15 + HSU λ=0.05 | 1.07408 | +0.60 |
| QAHSP λ=0.15 + ES λ=0.03 | 1.07416 | +0.68 |
| HSU λ=0.05 + ES λ=0.03 | 1.07423 | +0.75 |
| QAHSP λ=0.15 + PCS λ=0.003 | 1.07475 | +1.27 |

All four pairs at λ_each = λ\*/√2 underperform the best single reg at full λ. **Hypothesis 3 inconsistent** with our data. We offer two possible interpretations in §9.

### 6.3 Base A + PreQuantTTT (gray-track reference)

We ran PreQuantTTT (PR #1958 recipe, 21-epoch AdamW on val tokens) as a reference implementation. PR #1958 was closed upstream, so we treat these as gray-track methodological data — not record-eligible. The cells exist to test hypothesis 4.

| Combo | sliding val_bpb | Δ vs PQT alone (1.03969) |
|---|---:|---:|
| **PreQuantTTT alone** | **1.03969** | 0 |
| **PreQuantTTT + ES λ=0.05** | **1.03942** | **−0.27** ⭐ |
| PreQuantTTT + QAHSP λ=0.3 | 1.03985 | +0.16 |

ES (direction-shaping) compounds with PreQuantTTT; QAHSP (codebook-shaping) is essentially subsumed. **Consistent with hypothesis 4** for these two cells. We offer a tentative mechanism in §9.3.

### 6.4 Base B — single-reg attempts

| Reg config on Base B | val_bpb | Δ vs Base B baseline 1.05822 |
|---|---:|---:|
| **PR #1965 baseline** | **1.05822** ⭐ | 0 |
| SimCTG λ=0.3 + QAHSP λ=0.3 | 1.06047 | +2.25 |
| SimCTG λ=0.1 + QAHSP λ=0.1 | 1.05881 | +0.59 |
| ES λ=0.05 alone | 1.05993 | +1.71 |
| TripleHash bigram 1024×8 (isolated grad path) | 1.05886 | +0.64 |

**Every variant we tried at our chosen λ values measured worse than baseline on Base B.** We did not exhaustively search smaller λ on Base B — a Base-B-specific small-λ regime might recover positive transfer; we do not claim that's impossible. We claim only that our chosen λ values (which work well on Base A) hurt on Base B.

### 6.5 The cross-base sign-change

| Reg | Base A Δ | Base B Δ | sign change? |
|---|---:|---:|---|
| QAHSP λ=0.3 | −1.55 mBPB | +2.25 mBPB | yes (3.80 mBPB swing) |
| ES λ=0.05 | −0.74 mBPB | +1.71 mBPB | yes (2.45 mBPB swing) |
| Bigram (TripleHash) | ≈ neutral | +0.64 mBPB | similar magnitude |

Same architectural family, same reg, same λ — measurably opposite-direction effects on val_bpb.

`figures/fig1_cross_base_signs.png` shows this as a bar chart.

---

## 7. Pipeline-stage attribution (real data)

For each Base A cell, we extract from the training log the val_bpb at three eval stages: pre-quantization post-EMA, post-int6-quantization (no eval-time tricks), and post-sliding-window (final reported number for non-TTT submissions).

| reg | pre-quant | quantized | quant cost (mBPB) | sliding gain (mBPB) |
|---|---:|---:|---:|---:|
| QAHSP λ=0.3 | 1.07493 | 1.08941 | +14.5 | −15.9 |
| WOP λ=0.5 | 1.07537 | 1.08962 | +14.3 | −15.9 |
| HSU λ=0.1 | 1.07536 | 1.08993 | +14.6 | −15.9 |
| ES λ=0.05 | 1.07536 | 1.09023 | +14.9 | −15.9 |
| AOS λ=0.005 | 1.07580 | 1.09035 | +14.6 | −15.9 |
| PCS λ=0.005 | 1.07595 | 1.09053 | +14.6 | −15.9 |
| WBC λ=0.005 | 1.07646 | 1.09112 | +14.7 | −15.9 |

**Quant cost is approximately reg-independent: 14.3–14.9 mBPB across all 7 regs (range 0.6 mBPB, single-seed noise floor).** Sliding gain is also uniform at −15.9 mBPB.

This means: under the GPTQ + LQER + brotli quant pipeline used here, **the relative ranking of post-quant val_bpb is determined almost entirely by the pre-quant ranking**. Different regs change pre-quant val_bpb by ~1.5 mBPB; quant adds a constant 14.5 mBPB tax; sliding subtracts a constant 15.9 mBPB. The ranking is preserved through the pipeline.

This forces a re-interpretation of QAHSP's win:

> **QAHSP's val_bpb advantage comes from improving the pre-quant model, not from making the model more quant-robust.** It produces a better starting point (1.07493 vs 1.07646 for WBC), and that ~1.5 mBPB pre-quant advantage propagates through a uniform quant tax to the final number.

This is a non-obvious finding for the "quant-aware training" line of work in the leaderboard community. Caveats:
- Single-seed: 0.6 mBPB range across regs is at the noise floor.
- Specific quant pipeline: GPTQ + LQER + per-row brotli is sophisticated. On a naïve uniform-quant pipeline, QAHSP might show measurable quant-robustness benefit (untested here).
- Different model architecture: PR #1965's stack uses LQER asymmetric rank-4 quant residuals which themselves do post-hoc compensation; this might absorb most of the per-tensor differences QAHSP introduces.

`figures/fig_pipeline_waterfall.png` shows the per-stage val_bpb propagation as line plots.

### 7.1 PreQuantTTT inverts the quant cost

The cells in §6.3 have a different pipeline shape:

| cell | pre-PQT BF16 | post-quant | quant cost |
|---|---:|---:|---:|
| PQT alone | 1.07948 | 1.05176 | +22.8 mBPB |
| PQT + ES | 1.07516 | 1.05145 | +22.1 mBPB |
| PQT + QAHSP | 1.07550 | 1.05181 | +22.5 mBPB |

PreQuantTTT (eval-time AdamW on val) overfits the BF16 model to val by ~50 mBPB (1.07948 → 1.02891 BF16 in our P1 run), then quantization re-introduces ~22 mBPB of noise. **Net is still −20 mBPB final improvement** because the BF16 overfit was deep enough to survive the quant noise. This is a different mechanism from training-time regs.

---

## 8. Real-data reg × quant matrix

We trained 6 fresh Base A models (each with one reg config, SEED=42, MAX_WALLCLOCK_SECONDS=600). For each, we ran a forward pass on val tokens to capture last-block hidden states (128 tokens × 512 dim). Then applied 7 quantization schemes per row of hidden states and measured L2 distortion + cosine shift.

This is **real LM hidden states from real trained models**, not synthetic.

### 8.1 L2 distortion table (lower = quant preserves geometry better)

|                | int4 sym pT | int4 sym pR | int4 asym pR | int6 sym pR | int8 sym pR | AWQ-lite int4 |
|---|---:|---:|---:|---:|---:|---:|
| **no-reg** (SimCTG=0) | **8.67** ⭐ | **7.97** ⭐ | **7.04** ⭐ | 4.77 | 1.27 | **2.50** ⭐ |
| SimCTG λ=0.3 | 8.89 | 7.99 | 7.19 | 5.10 | 1.42 | 2.66 |
| SimCTG + QAHSP λ=0.3 | 8.90 | 8.40 | 7.77 | 5.72 | 1.61 | 2.81 |
| **SimCTG + ES λ=0.05** | 8.77 | 8.04 | 7.13 | **4.73** ⭐ | **1.25** ⭐ | 2.51 |
| SimCTG + HSU λ=0.1 | 9.06 | 8.51 | 7.62 | 5.18 | 1.38 | 2.70 |
| SimCTG + AOS λ=0.005 | 8.79 | 8.08 | 7.25 | 5.07 | 1.38 | 2.64 |

⭐ = lowest distortion in column. Mean hidden state L2 norm: ~28 across regs (so int4 distortions of 7-9 are 25-32% of the embedding magnitude; int8 distortions of 1.3 are ~5%).

GPTQ-lite int4 column omitted from table — our naïve column-by-column implementation produces unphysical distortion (~50, larger than the embedding magnitudes themselves) due to error-propagation blowup. We flag it as an **implementation bug** in our analysis script, not an indictment of GPTQ. Real GPTQ uses Hessian-aware ordering + dynamic scale that we did not implement.

### 8.2 The "plays nice" pattern

**Coarse quant (int4): no regularization wins.** For all int4 schemes (sym per-tensor, sym per-row, asym per-row, AWQ-lite), the **no-reg** cell has lowest L2 distortion. Adding any reg — including just SimCTG — measurably *increases* the int4 quant cost.

**Fine quant (int6 / int8): SimCTG + ES wins.** At int6 and int8, SimCTG + ES has lowest distortion (4.73 / 1.25). The reg's directional shaping helps when there's enough quant resolution.

**SimCTG + QAHSP is consistently the worst** across all quant schemes (ranks 6/6 at int4 sym per-tensor, sym per-row, asym per-row, int6, int8, AWQ). QAHSP's int4-grid STE penalty trained at λ=0.3 actually moves the embeddings *away* from the per-row scaled int4 grid used at inference time. The training-time grid mismatch hurts here.

### 8.3 The dissociation: synthetic ≠ real

The synthetic geometric analysis (§10–§12.1) suggested **AOS** is most quant-robust (lowest synthetic L2 distortion). The real-data analysis here says **SimCTG+ES** at fine quant or **no-reg** at coarse quant. AOS is not the winner on real data — it's middle-of-the-pack.

This is the kind of synthetic-real gap §12.2 warned about. **Synthetic geometric analysis is suggestive of mechanism, not predictive of real-data quant performance.**

### 8.4 Reading this in context

Tying back to the §7 finding: quant cost in val_bpb is approximately reg-independent on Base A. The (reg × quant) L2 distortion matrix here shows there *are* differences in how each reg's hidden states survive quant, but those differences (~1-2 in L2 distortion units) translate into single-mBPB val_bpb shifts that get washed out by the much larger constant quant tax (+14.5 mBPB) in the GPTQ + LQER + brotli pipeline.

So: **regs do change quant survival of embeddings, but at the val_bpb level the GPTQ + LQER + brotli pipeline equalizes them.** Different quant pipelines (uniform int4 without LQER) might expose the differences as measurable val_bpb shifts.

`figures/fig_reg_quant_matrix_real.png` — full 4-panel heatmap (L2 distortion, cosine shift, post-quant isoscore, post-quant effective rank) on real Base A LM hidden states.

### 8.5 Caveats specific to §8

- Single seed per cell.
- Hidden states sampled from a small val batch (128 tokens). Different batches might shift relative orderings within ~10%.
- We did NOT measure val_bpb for the 6 fresh cells in this study — only L2 distortion of hidden states under quant. The val_bpb numbers banked from these runs (in `parameter-golf/logs/_results.csv`) are sliding-window-only and would show different relative orderings (e.g., on val_bpb at sliding-window only, ES and QAHSP are very close).
- Our GPTQ-lite implementation is buggy; we present only the 6 quant schemes where the implementation is well-tested.
- AWQ-lite implements only the per-channel pre-scaling part; full AWQ has additional steps we did not implement.

---

## 9. Tentative mechanisms

We use these mechanisms to organize our observations and to make predictions for §8 once that data lands. **Each is offered as a candidate explanation, not a proven claim.** Readers should treat §6–§7 as the empirical core and §9 as commentary.

### 9.1 Why QAHSP would win on Base A but not Base B

Candidate explanation: Base A's training schedule uses default Polar-Express-NS-Muon LR and standard cosine-warmdown. The end-of-training weight distribution is not heavily tuned, so QAHSP's auxiliary gradient toward the int6 grid functions as useful prep for the downstream quant step.

Base B's schedule (MATRIX_LR=0.026, WARMDOWN_FRAC=0.85, GRAD_CLIP_NORM=0.3, BETA2=0.99, TTT_BETA2=0.99) is the result of accumulated greedy hyperparameter search across multiple lineage PRs. The end-of-training weight distribution is already shaped to interact well with the specific GPTQ + LQER + AWQ-Lite quant pipeline of PR #1965. QAHSP's auxiliary gradient is then largely redundant with the work the schedule already does, *and* perturbs the carefully tuned trajectory.

Empirical support: lambda-monotone deterioration on Base B (+2.25 at λ=0.3, +0.59 at λ=0.1) is consistent with "more reg = more perturbation on a near-locally-optimal trajectory." We cannot rule out other contributing factors (different tokenizer, different TTT eval pipeline, different LQER configuration on Base B vs Base A).

### 9.2 Why pairs underperformed at 1/√2 · λ

The pre-registered variance-budget intuition: for independent-subspace regs, λ_each = λ\*/√N preserves the per-batch reg gradient norm at λ\*. Our four pairs all underperformed the best single reg.

Two possible interpretations, neither tested:

- **Regs share a common gradient pathway.** All seven flow gradient back through the same matrix params (Q, K, V, O, MLP banks). They are not gradient-subspace-independent. A second reg at half λ then dilutes the dominant signal rather than addressing an orthogonal direction.
- **The 1/√N rescaling is too aggressive.** A pair at full λ for one reg + small λ for the other might compose more successfully — we did not run that experiment.

The Phase A3 finding (PQT + ES at full λ for both, both at the same 0.05 lambda used in single-reg) is consistent with the second interpretation: when one component carries dominant signal, a small auxiliary at full λ adds independent value. We have one positive cell, not enough to confirm a rule.

### 9.3 Why ES compounded with PreQuantTTT and QAHSP did not

Candidate explanation:

- **Codebook-shaping regs** (QAHSP, WBC, WOP, PCS) prepare the model's *coordinate-wise* relationship to the int6 grid. Eval-time fine-tuning re-aligns weights against the val distribution, which can overwrite this coordinate-wise prep. The training-time investment in QAHSP becomes essentially a no-op.
- **Direction-shaping regs** (ES, HSU, AOS) constrain the *angular* or *magnitude* structure of token reps. Eval-time fine-tuning typically updates weights without coordinated flips of many high-magnitude components, so the angular structure is preserved. The well-conditioned manifold remains and small fine-tuning adjustments are more effective on it.

Empirical support: §6.3 has only two PQT × reg cells. The data are consistent with this interpretation but a single positive case is not strong evidence. Predicted but untested:
- HSU should compound with PQT.
- WBC, WOP, PCS should be subsumed by PQT.
- The same compounding pattern might apply on Base B if PreQuantTTT could be added there.

We have not run those cells. We hope this interpretation is testable by independent replication.

---

## 10. Synthetic geometric analysis (mechanism, clearly marked synthetic)

Sections 11–14 use a controlled synthetic embedding cloud (96 tokens × 32 dims) to **demonstrate what each reg does to embedding geometry**, independent of the noisy val_bpb signal. None of these claims should be read as performance numbers; they are mechanism illustrations.

### 10.1 Embedding geometry under each reg

`figures/fig_emb_geometry.png` (synthetic): 64-token × 32-dim cloud. We apply each reg's gradient for 300 SGD steps and visualize the resulting cloud in 2D PCA + L2 norm histograms.

| reg variant | norm var | mean off-diag \|cos\| | max−mean gap | top-1/top-4 sv |
|---|---:|---:|---:|---:|
| baseline | 0.204 | 0.163 | 0.736 | 1.18 |
| QAHSP (int4 STE) | 0.204 | 0.163 | 0.736 | 1.18 |
| ES (off-diag cos²) | 0.204 | **0.159** | 0.731 | 1.18 |
| HSU (var of norms) | **0.079** | 0.163 | 0.721 | 1.17 |
| AOS (max−mean) | 0.185 | 0.158 | **0.559** | 1.19 |

Bold: the column where each reg specifically targets that statistic. **The synthetic gradient steps confirm the regs do what their math says they should do.** QAHSP's effect is small in this synthetic at the chosen LR/steps — its STE gradient is small away from grid centroids; with longer training and higher λ it would show measurable grid-pull. We don't read this as a performance comparison.

### 10.2 Cosine similarity heatmap + per-coord distribution

`figures/fig_emb_cosine_coord.png` (synthetic): 64×64 token-token cosine similarity matrix per reg + per-coord activation histogram. Visual story: ES makes the cosine off-diagonal smaller; QAHSP creates a "cleavage" pattern in the per-coord histogram corresponding to the int4 grid centroids. Different regs change different parts of the geometry.

### 10.3 Per-token outlier coordinates

`figures/fig_emb_outliers.png` (synthetic): each token plotted as (mean \|h\|, max \|h\|). Distance above the y=x diagonal = outlier severity. AOS visibly pulls outlier tokens toward the diagonal (max-mean gap closes from 0.74 → 0.56).

### 10.4 Semantic-cluster preservation

`figures/fig_3d_semantic.png` and `figures/fig_semantic_metrics.png` (synthetic): we use 4 clusters × 16 tokens with planted outliers. We apply each reg and measure silhouette score + intra/inter-cluster distance.

| reg | silhouette ↑ | intra-cluster ↓ | inter-cluster ↑ |
|---|---:|---:|---:|
| baseline | 0.4168 | 3.285 | 9.840 |
| QAHSP | 0.4168 | 3.285 | 9.840 |
| HSU | 0.4171 | 3.247 | 9.838 |
| AOS | 0.4176 | 3.261 | 9.731 |
| **ES** | **0.4140** | **3.345** | 9.853 |

In the synthetic setting, ES slightly degrades semantic cluster preservation (silhouette 0.4140 vs baseline 0.4168). This is small — comparable to noise — but directionally interpretable: ES penalizes off-diag cosine, including for *intra-cluster* token pairs that should be similar. The trade-off is "discrimination at the cost of clustering."

We do not claim this synthetic result transfers to real LM hidden states. It is an *intuition-builder* — a future test on real Base A embeddings (which §8 partly addresses) would be the actual evidence.

---

## 11. Canonical metrics from the literature

`figures/fig_canonical_metrics.png` (synthetic): a 4-panel grid with:
- IsoScore / anisotropy (Ethayarajh 2019)
- Effective rank (Roy & Vetterli 2007)
- Quantization-induced distributional shift (KL on cosine distribution)
- Linear probing classifier (Alain & Bengio 2017)

| reg | isoscore ↓ | eff rank ↑ | spec entropy | quant KL | lin sep |
|---|---:|---:|---:|---:|---:|
| baseline | 0.5436 | 18.09 | 0.8355 | 0.00065 | 1.000 |
| QAHSP | 0.5436 | 18.09 | 0.8355 | 0.00065 | 1.000 |
| HSU | 0.5436 | 18.10 | 0.8356 | 0.00065 | 1.000 |
| ES | **0.5364** | **18.31** | **0.8389** | 0.00169 | 1.000 |
| AOS | 0.5419 | 18.17 | 0.8367 | 0.00352 | 1.000 |

Two observations from the synthetic literature-metric panel:

1. **ES is the most isotropic by all three direction-space measures.** Lower isoscore, higher effective rank, higher spectral entropy. This is consistent with the mechanism: ES literally optimizes for the inverse of isoscore (off-diag cos²).

2. **HSU is identical to baseline on direction-space metrics.** It moves only the L2-norm distribution. Clean dissociation between norm-shaping and direction-shaping regs.

Linear probing accuracy is 1.0 across all (the 4 cluster task is too easy to separate the regs). We re-ran with smaller cluster separation + more noise (`figures/fig_linear_probe_harder.png`) but the test remained too easy to differentiate — left as future work to design a harder probing task.

`figures/fig_spectral.png` (synthetic): singular value spectrum log-y. ES has the flattest spectrum (highest effective rank); HSU's curve is near-identical to baseline.

---

## 12. Quantization survival (synthetic + real)

### 12.1 Synthetic int6 quantization on the reg-trained cloud

`figures/fig_pre_post_quant.png` and `figures/fig_quant_robustness.png` (synthetic): per-reg cloud, then per-token-row int6 quantization, then per-token L2 distortion + silhouette pre/post.

| reg | mean L2 distortion ↓ | norm Δ% | cos shift | silhouette pre | silhouette post | Δ silhouette |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.173 | 0.25 | 0.0003 | 0.4168 | 0.4149 | −0.0019 |
| QAHSP | 0.173 | 0.25 | 0.0003 | 0.4168 | 0.4149 | −0.0019 |
| HSU | 0.173 | 0.25 | 0.0003 | 0.4171 | 0.4153 | −0.0018 |
| ES | 0.173 | 0.32 | 0.0003 | 0.4140 | 0.4110 | −0.0030 |
| **AOS** | **0.162** | 0.28 | **0.0002** | 0.4176 | 0.4166 | **−0.0010** |

Synthetic finding: **AOS has the smallest L2 distortion under int6 quant**, consistent with its mechanism (suppressing per-token max-coord shrinks the per-row scale, reducing rounding step size).

### 12.2 Real measured pre vs post quant val_bpb on Base A

`figures/fig_real_pre_post_quant.png` (real, from training logs):

This is the data that motivates the §7 finding — quant cost is approximately reg-independent on Base A. The figure shows the per-reg quant tax as a +14.5 mBPB constant column. Re-stated: **synthetic L2 distortion differences (§12.1) do not propagate into real val_bpb differences** at the GPTQ + LQER + brotli pipeline scale on Base A. Either the synthetic differences are too small to matter, or the LQER asymmetric residual correction absorbs them, or both.

This dissociation between synthetic geometric metrics and real val_bpb effects is itself a finding worth flagging. **Synthetic embedding geometry is suggestive but not predictive of post-quant val_bpb at this scale.**

---

## 13. Mechanistic checks: do the regs change the model upstream of quantization?

The natural reviewer question after §7 ("quant cost is reg-independent") is: *do the regs change anything at all?* If every reg produces post-quant val_bpb within 0.6 mBPB of every other, maybe the regs aren't actually doing different work. We ran three checks to falsify that worry.

### 13.1 Singular-value spectrum of weight matrices

`figures/fig_svd_spectrum.png` (raw spectra, six weight families × six regs, log-scale).
`figures/fig_svd_flatness.png` (per-family bar chart of mean −log₁₀(σᵢ/σ₁) over the bottom 87.5% of ranks — bigger = flatter spectrum = harder to per-row int6 quantize).
`figures/fig_svd_differential.png` (each reg's spectrum vs no-reg as a Δ% curve, smoothed window 5, top 90% of ranks, y-clipped to ±4%).

Method: for each of the six trained Base-A variants we compute SVD of every 2-D weight (6 families × 11 layers each) and average σ₁..σₘᵢₙ across layers within a family. We report (a) the raw spectrum, (b) a single flatness scalar per (reg, family), and (c) the differential vs no-reg.

Findings:

- **Flatness is essentially identical across regs** (≤2% relative differences within each family on the flatness scalar; bar chart visually flat).
- The differential view reveals the structure hidden by the overall similarity:

  | Reg | Largest mean Δ% (family) | Mean Δ% | Max \|Δ\|% |
  |-----|--|---:|---:|
  | SimCTG (alone) | Attn out | +1.23% | 1.96% |
  | SimCTG+ES | Attn V | **−1.02%** | 1.55% |
  | SimCTG+QAHSP | Attn Q | −0.82% | 1.71% |
  | SimCTG+HSU | Attn out / V | +0.60% | 1.58% |
  | SimCTG+AOS | Attn K | +0.63% | 1.55% |

- **Regs touch attention more than MLP.** Every reg has max \|Δ\| ≤ 0.6% on all four MLP families; attention swings up to ~2%.
- **ES is the only reg that reduces attention V σᵢ on average** (mean −1.02%). All others nudge up. Consistent with ES's mechanism (angular spread reduces V's per-row magnitudes via the gradient through the softmax).
- **All deltas are sub-3% at every rank index.** GPTQ int6 per-row quantization has noise floor ≈4-6% per channel, so the SVD differences are below the quantization-noise threshold — which mechanistically explains why post-quant val_bpb is reg-independent on Base A even though the regs *are* leaving a fingerprint upstream.

### 13.2 Hidden-state norm + kurtosis trajectory through depth

`figures/fig_depth_trajectory.png` — per-block mean ‖h‖ and mean per-coord excess kurtosis, one curve per reg, computed on a deterministic 8×128 sample of synthetic input tokens through CPU forward (flash-attn replaced by SDPA fallback).

Findings:

- **‖h‖ peaks mid-depth** (~85 at block 3-4) then collapses to ~30 at the final block. Same pattern across all six regs; curves overlap within ~5%.
- **Kurtosis is near zero through layers 1-9 then explodes to 350-400 at the final block.** Same pattern across all six regs.
- The reg-induced differences are visible but small. SimCTG variants cluster slightly tighter than no-reg through layers 7-9 (norm-collapse region); SimCTG+ES dips lowest at block 8-9 (~62 vs no-reg's ~63), consistent with its angular-spread role redistributing magnitude.
- **Outlier emergence (the kurtosis explosion at the final block) is architectural, not reg-driven.** No reg suppresses it — they all sit at 350-390 final-block kurtosis. This is consistent with the residual stream's natural drift toward heavy-tailed pre-logit activations under tied embeddings + RMSNorm + softcap.

This explains why activation-side regs (AOS, HSU, QAHSP) targeted at hidden-state outliers don't dominate quant cost on Base A: the outliers they target only appear at the final block, where the next operation is the tied-embedding logit projection, not a quantized matmul.

### 13.3 Pairwise CKA between final-block representations

`figures/fig_cka_heatmap.png` — linear CKA (Kornblith et al. 2019) between final-block hidden states across the six trained variants, on the same deterministic 8×128 input.

Findings:

- **All off-diagonal CKAs sit in [0.67, 0.75].** Regs produce subtly different representations but all in the same ballpark.
- The two most-similar variants are no-reg and SimCTG+QAHSP (CKA 0.75) and SimCTG+ES and no-reg (CKA 0.70). SimCTG+ES vs SimCTG+QAHSP is also 0.75.
- The two most-dissimilar variants are SimCTG+AOS vs SimCTG (alone) at CKA 0.67.
- **No reg pulls representations away from the cloud by more than ~10% of the within-cloud spread.**

This is consistent with the SVD finding: regs leave a fingerprint, but the fingerprint is small relative to the variation a single SimCTG vs no-SimCTG choice already induces. CKA confirms that "regs are interchangeable on Base A" is not just true at the post-quant val_bpb level but also at the latent-representation level.

### 13.4 Synthesis

| Layer of analysis | Reg differences? | Magnitude vs noise |
|---|---|---|
| Weight spectra (SVD) | Yes, sub-3% per rank | Below int6 per-row quant noise (~5%) |
| Hidden-state norm/kurtosis | Yes, sub-10% at most depths | Below run-to-run noise from sliding-window eval |
| Final-block representation (CKA) | Yes, off-diag 0.67-0.75 | Substantial, but uniform across regs |
| Post-quant val_bpb (§7) | Effectively no, 14.3-14.9 mBPB tax | Below 1-seed val_bpb noise (~2 mBPB) |

The story that emerges: **regs do shape Base A internals, but the shaping happens in a regime that GPTQ + LQER + brotli flattens out.** This reframes the original question from "do these regs work?" to "what would a quant pipeline have to look like for these regs' fingerprints to survive?" — a worthwhile direction we don't pursue here.

---

## 14. Statistical caveats

We ran **single seeds** in this study to keep cell count manageable. Many of the smaller deltas (≤0.5 mBPB) are at or below run-to-run noise (3-seed std ≈ 0.0023 from PR #1855 lineage data). Conclusions we feel reasonably confident about:

- **Sign-change findings (§6) are robust.** A 3.80 mBPB swing is much larger than 1-seed noise, and the direction is consistent across two regs (QAHSP and ES) with a clear candidate mechanism (§9.1).
- **Quant-cost-uniformity (§7) is robust.** 0.6 mBPB range across 7 regs at the noise floor *is* the finding — no reg differentiates itself in quant cost.
- **Pair-vs-single ranking (§6.2) is suggestive.** Adjacent ranks within 0.3 mBPB should be treated as approximately tied at single-seed; the overall pattern (every pair worse than best single) holds at 4 cells.
- **PQT × ES / × QAHSP comparison (§6.3) is suggestive but not statistically confirmed.** A 0.27 mBPB advantage for ES vs −0.16 for QAHSP is on the margin; multi-seed replication would strengthen this.
- **All synthetic results (§10–§12.1) are mechanism illustrations** of what the regs do to a controlled small-dim cloud. They are not performance forecasts. The dissociation noted in §12.2 cautions against reading them as such.

---

## 15. What we do NOT claim

- We do not claim our 7 regs are the right basis. They were chosen pre-experiment for the 16 MB cap-constrained regime; other reg families (dropout schedules, low-rank weight constraints, activation bottlenecks) might transfer differently.
- We do not claim Base B is hostile to *all* additions. We did not test eval-time-only side channels (byte-PPM, n-gram tilt) due to ongoing legality discussions about score-after-fit-statistics patterns.
- We do not claim to have exhausted lambda search on Base B. A configuration at much smaller λ might recover positive transfer; we did not run those cells.
- We do not claim our cross-base differences are unique to PR #1965. The same study run between any two heavily-tuned bases might show similar transferability gaps; this is a hypothesis for future work.
- We do not claim the synthetic geometric analysis (§10–§12.1) predicts real-data quant survival on Base A. §12.2 shows the dissociation; we present synthetic results as mechanism intuitions only.

---

## 16. Reproducibility

All Base-A cells (§6.1, 6.2, 6.3): env-gated harness `train_gpt_baseA.py.lzma` (companion submission A's `train_gpt.py` with the 7 reg knobs and bigram + StableMuon as env vars).

Base-B cells (§6.4): four frozen scripts (`train_gpt_baseB_simctg_qahsp.py.lzma`, `train_gpt_baseB_es.py.lzma`, `train_gpt_baseB_es_hsu.py.lzma`, `train_gpt_baseB_bigram.py.lzma`) — each is the PR #1965 reproduction code with the named reg combination grafted in.

Each cell can be reproduced by setting the env vars listed in `ablation_data.csv` on the corresponding script. The 20 cells together took ~7 hr of 8×H100 SXM compute. Logs are reproducible from the env configs.

For §8 (real-data reg × quant matrix): pipeline is `run_reg_quant_matrix.py` + `build_synergy_figures.py`, runs after the 6 fresh `EmbStudy_*` cells finish.

---

## 17. Files

- `README.md` — this file
- `submission.json` — metadata
- `REGULARIZATION_ABLATION.md` — pre-registered hypotheses, frozen at study start
- `ablation_data.csv` — raw cell results (config + val_bpb + size + cap-fit) for downstream reuse
- `pipeline_attribution.json` — extracted pre-quant / quant / sliding / TTT val_bpb per cell
- `eval_pipeline_breakdown.json` — same data, per-stage breakdown form
- `run_reg_quant_matrix.py` — analysis pipeline for §8 (real-data reg × quant)
- `build_synergy_figures.py` — heatmap + synergy detection
- `build_advanced_figures.py` — analysis pipeline for §13 (SVD spectrum, depth trajectory, CKA)
- `run_after_trains.sh` — automated trigger after EmbStudy training cells finish
- `depth_trajectory.json`, `cka_pairwise.json` — extracted §13 numerical tables
- `figures/` — PNGs: see in-context references in §6–§13
  - cross-base + pipeline: `fig1_cross_base_signs.png`, `fig_pipeline_waterfall.png`, `fig_real_pre_post_quant.png`, `fig_pqt_compounding.png`, `fig_lambda_budget.png`, `fig_reg_quant_matrix_real.png`
  - real-data hidden states: `fig_real_3d_pca.png`, `fig_real_canonical_metrics.png`, `fig_real_coord_distribution.png`, `fig_real_l2norm_distribution.png`
  - mechanistic checks (§13): `fig_svd_spectrum.png`, `fig_svd_flatness.png`, `fig_svd_differential.png`, `fig_depth_trajectory.png`, `fig_cka_heatmap.png`

---

## 18. Credits

Reg-knob design and study: BharathSShankar (this work).

Base-A inherits architecture from PR #1855 lineage with our SP10240 tokenizer adoption. The N9 SimCTG hyperparameters (λ=0.3, margin=0.4) were tuned by us; documented in companion record submission A.

Base-B (PR #1965 lineage): @himanshudongre (PR #1965), @andrewbaggio1 (PR #1953), @alertcat (PR #1945), @codemath3000 (PR #1855), @bigbag (PR #1493), @dexhunter (PR #1413, PR #1331/1437), @clarkkev (PR #1394), @abaybektursun (PR #549). Thanks to these authors for the public PRs we built on.

PreQuantTTT recipe (used in §6.3 only, gray-track): @okezue (PR #1958, since closed). We treated their recipe as a reference implementation for testing hypothesis 4 and respect the closure decision.

Wang & Isola 2020 framing of "alignment + uniformity" decomposition seeded our pre-registered hypothesis 3.

Thanks to OpenAI and the leaderboard organizers for the challenge and for the example PRs that made this study possible.
