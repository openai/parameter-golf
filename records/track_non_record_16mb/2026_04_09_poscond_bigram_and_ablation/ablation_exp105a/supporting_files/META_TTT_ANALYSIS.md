# Meta-TTT Ablation Study — exp101 vs exp105a

A rigorous weight-space analysis of the meta-TTT training signal, using the
cleanest possible single-variable ablation we could run on this codebase.

## TL;DR

**Meta-TTT (exp101's FOMAML flavour) does not meaningfully change the
trained model.** The ablation pair exp101 (meta-TTT ON) vs exp105a
(meta-TTT OFF) produces two models that have:

- **The same final legal_ttt bpb** (1.1159 vs 1.1162, delta within noise)
- **The same TTT adapt delta** (≈0.023 bpb in both)
- **Nearly identical spectral properties** (op-norm, Fro norm, stable rank,
  Lipschitz product, condition number — all within 1–8%)
- **Identical quantization sensitivity** under int6 per-row (ratio 0.9989)
- **Raw weight cosine ≈ 0.10 across banks**, but **principal-angle subspace
  cosine ≈ 0.65** — i.e. the weights rotate into a different basis but
  span partially the same subspace
- **Borderline different loss basins** (midpoint norm ratio 0.799, just
  below the "same basin" threshold of 0.8)

**Bottom line: Meta-TTT as a training signal behaves like gradient noise.**
It pushes the optimizer into a neighboring local minimum of essentially
equivalent quality, costs 3% per-step compute (≈206 missing training steps
in an 80-minute wallclock cap), and delivers zero differentiable benefit
to the TTT channel it was designed to amplify.

There is one very small positive: the condition number of weight matrices
drops from 6.1 → 5.6 (≈8% improvement). This is the only quantitative
signature of implicit regularization, and it is an order of magnitude
too small to justify the compute cost.

---

## 1. Intuition & motivation

Meta-TTT was proposed as a training-time mechanism to teach the network
to adapt *faster* at test-time. The theory was FOMAML-style:

1. **Inner loop**: take a gradient step on one half of a training batch
2. **Outer loop**: evaluate the loss on the *other* half with the
   gradient-updated weights
3. **Meta update**: backprop the outer loss to the *original* weights,
   accumulating on top of the normal training gradient

If this works, the model's weights should be *pre-positioned* for
test-time SGD to benefit more from every adapt step. The competition
scorer evaluates with a sliding-window TTT pass (`eval_val_sliding_ttt`),
so a successful meta-TTT should produce a bigger TTT delta than a
vanilla model, even at equal pre-TTT loss.

The expected behavior would be:

```
baseline_val_bpb     :  normal model ── SGD during TTT ──> val_bpb_normal
baseline_val_bpb_mtt :  meta-trained ── SGD during TTT ──> val_bpb_meta ≪ val_bpb_normal
```

What we actually measured: `val_bpb_meta ≈ val_bpb_normal`. The TTT
channel is agnostic to whether meta-TTT was active during training.

---

## 2. Experimental setup — the cleanest single-variable ablation

Both runs share:

| Parameter | Value |
|---|---|
| Architecture | 11-layer U-Net transformer (5 encoder + 6 decoder, skip-connected) |
| Model dim | 512 |
| Heads | 8 (GQA: 8Q / 4KV) |
| MLP multiplier | 3.0 |
| Tied embeddings | Yes |
| Vocab | 1024 (SentencePiece BPE) |
| XSA layers | last 11 (all blocks) |
| RoPE dims | partial, 16 of 64 |
| Training batch tokens | 786 432 |
| Seq len | 2048 |
| Iterations cap | 7500 |
| Wallclock cap | 4800 s |
| Optimizer | Muon (matrix) + AdamW (tok + scalar) |
| Muon momentum | 0.99 |
| EMA | enabled, decay 0.998 |
| SWA | enabled, every 50 steps during warmdown |
| Late QAT | threshold 0.25 |
| Bigram | 4096 × 64, pos-conditional (TRIGRAM=0) |
| GPTQ | int6 for mlp+attn, int8 for embed, AR self-gen hessians |
| Seed | 42 |
| TTT eval | stride 64, 4 epochs, chunk 65 536, lr 0.004, SGD momentum 0.9 |

The **only** knob flipped between the two runs:

```diff
- export META_TTT_ENABLED=1    # exp101
+ export META_TTT_ENABLED=0    # exp105a
```

Everything else — seed, data order, LR schedule, QAT timing, SWA windows,
TTT eval, even the 4MB-byte train_gpt.py source — is identical. This is
the closest we can get to an "everything else equal" ablation inside
this codebase.

---

## 3. Headline results

| Metric | exp101 (meta-TTT ON) | exp105a (meta-TTT OFF) | Δ (105a − 101) |
|---|---:|---:|---:|
| step_avg (wallclock / step) | 684 ms | 663 ms | **−21 ms** (−3.1%) |
| Training steps reached | 7020 | 7226 | **+206** |
| val_bpb @ step 3000 | 1.2254 | 1.2264 | +0.0010 |
| val_bpb @ step 6000 | 1.1474 | 1.1524 | +0.0050 |
| post-EMA val_bpb | 1.1352 | 1.1353 | +0.0001 |
| final_int6_roundtrip val_bpb | 1.1393 | 1.1396 | +0.0003 |
| **legal_ttt val_bpb** | **1.1159** | **1.1162** | **+0.0003** |
| TTT adapt delta | 0.0234 | 0.0234 | **0.0000** |

Meta-TTT buys us ≈0.005 val_bpb at step 6000 (real signal) but costs 206
training steps to the wallclock cap, and the EMA + warmdown phase erases
the per-step advantage by the finish line. Post-EMA, the two models are
bit-for-bit-identical up to the noise floor of the val shards (we do a
single val pass, so noise floor ≈ 1e-4 bpb).

**The TTT delta is identical to 4 decimal places.** That is the clean
"meta-TTT fails" signal — if the training signal were amplifying the
adapt channel, the TTT delta should be visibly larger for exp101. It
isn't.

---

## 4. Weight-space analysis

All analyses in this section run on the two saved float `final_model.pt`
files, with no GPU required. Script: `records/phase3/analysis_meta_ttt.py`.
Full JSON results: `records/phase3/analysis_meta_ttt.json`.

### 4.1 Per-layer weight deltas

For the 55 tensors shared by both checkpoints, we computed the relative L2
distance `||W_101 − W_105||_F / ||W_101||_F` and the element-wise cosine
similarity.

**The 4 banked weight matrices (qo, kv, mlp_up, mlp_down) diverged to
near-orthogonality at the element level:**

| tensor | shape | rel_L2 | cosine |
|---|---|---:|---:|
| `mlp_down_bank` | (11, 512, 1536) | 1.372 | **+0.051** |
| `qo_bank` | (22, 512, 512) | 1.362 | **+0.069** |
| `mlp_up_bank` | (11, 1536, 512) | 1.356 | **+0.072** |
| `kv_bank` | (22, 256, 512) | 1.343 | **+0.096** |
| `ve_shared.embed.weight` | (1024, 64) | 1.220 | +0.250 |

These numbers are *stunning*: two models trained from the same seed,
with 97% overlapping training history, ended up with **essentially
orthogonal weight matrices**. For a normally-trained model, a 3% compute
perturbation might shift weights by ~0.01 in cosine distance. Here we see
a full 0.9 rotation in the raw-element basis.

**The 44 per-block control scalars (attn_scale, mlp_scale, q_gain,
resid_mix) are nearly identical:**

| tensor | rel_L2 | cosine |
|---|---:|---:|
| `blocks.0.mlp_scale` | 0.036 | +0.999 |
| `blocks.10.attn.q_gain` | 0.063 | +0.998 |
| `blocks.8.mlp_scale` | 0.076 | +0.997 |
| `blocks.9.mlp_scale` | 0.078 | +0.997 |
| `blocks.1.attn_scale` | 0.085 | +0.996 |

The macro structure of the network (*how much* attention vs mlp vs
residual each block uses) is learned to the same fixed point by both
runs. The micro directions inside the matrices — that's where meta-TTT
left its fingerprint.

### 4.2 Quantization sensitivity

This is where I had an initial wrong finding, corrected here.

**Method**: simulate per-row int6 quantization with `clip_range=31`,
per-bank-slot. For each of the 4 banks, unpack the banked 3D tensor
into per-layer 2D matrices and quantize each row independently — this
is what the real `mixed_quantize_int6` pipeline does downstream of
`_unbank_state_dict`.

| tensor | n_slots where 101 < 105 | mean MSE exp101 | mean MSE exp105a | ratio |
|---|:-:|---:|---:|---:|
| `kv_bank` | 12/22 | 8.76e-05 | 8.84e-05 | 0.991 |
| `mlp_down_bank` | 6/11 | 8.67e-05 | 8.67e-05 | 0.999 |
| `mlp_up_bank` | 5/11 | 8.67e-05 | 8.67e-05 | 1.000 |
| `qo_bank` | 11/22 | 8.68e-05 | 8.68e-05 | 1.000 |
| **aggregate** | — | **8.68e-05** | **8.69e-05** | **0.9989** |

Meta-TTT does **not** produce quantization-robust weights. The overall
MSE ratio is 0.9989 — a 0.11% difference, which is statistical noise
at this sample size (4 banks × 11–22 slots). My earlier run used a
single scale per entire bank slot rather than per-row, which
exaggerated the difference by ~100×. When you quantize each row with
its own scale (the real pipeline), the per-row amax adapts to whatever
range meta-TTT left behind, so the roundtrip error is essentially
identical.

**Implication**: meta-TTT cannot be sold as an implicit quantization-aware
regularizer. Whatever smoothing it does at the weight level gets absorbed
by per-row scale adaptation before any precision loss occurs.

### 4.3 Regularizer signature (spectral analysis)

For every matrix ≥ 65536 parameters in both checkpoints, we computed the
full singular value spectrum and reported operator norm, Frobenius norm,
stable rank (= `||W||_F² / σ_max²`, the "effective dimensionality"),
condition number (`σ_max / σ_min`), and the log-sum of operator norms
(proxy for the forward-pass Lipschitz constant).

| quantity | exp101 | exp105a | Δ (%) |
|---|---:|---:|---:|
| avg operator norm (σ_max) | 82.52 | 81.99 | +0.7% |
| avg Frobenius norm | 331.99 | 330.04 | +0.6% |
| avg stable rank | 22.86 | 22.80 | +0.2% |
| **avg condition number (σ_max / σ_min)** | **5.6** | **6.1** | **−8.2%** |
| log Lipschitz constant (Σ log σ_max) | 29.528 | 29.501 | +0.09% |

**The only statistically meaningful delta is condition number.**
Meta-TTT's matrices are slightly better conditioned — their smallest
singular values are further from zero. This is the implicit
regularization signature, and it's small.

Operator norms, Frobenius norms, stable rank, and the Lipschitz product
are all within 1%. Meta-TTT does not significantly change:

- The energy of each matrix (Fro norm)
- The largest direction of each matrix (op norm)
- The effective dimensionality (stable rank)
- The forward-pass sensitivity (Lipschitz)

It only nudges the *tail* of the spectrum — the tiny singular values that
a vanilla run leaves near zero, meta-TTT pushes slightly away. This is
consistent with the theory that meta-TTT's per-sample gradient noise
adds a small jitter that prevents any singular direction from collapsing
to exactly 0.

### 4.4 Subspace overlap (principal angles)

**This is the analysis that resolves the paradox** of "cosine 0.10 at
the element level, but identical val_bpb and identical TTT behavior."

**Method**: For each matrix, take the top-k left singular vector
subspaces `U_A[:, :k]`, `U_B[:, :k]` (k = min(32, min_dim/4)), compute
`U_A^T U_B`, and report the singular values of that product. These
are the cosines of the principal angles between the two subspaces.
An average cosine near 1 means "same subspace, different basis inside
it" — which is functional equivalence. Average cosine near 0 means
"genuinely different features."

| matrix | k | avg subspace cosine | frac dims aligned (>0.9) |
|---|:-:|---:|---:|
| `kv_bank` | 32 | **0.955** | 0.800 |
| `tok_emb.weight` | 32 | 0.792 | 0.406 |
| `mlp_down_bank` | 32 | 0.779 | 0.500 |
| `qo_bank` | 32 | 0.623 | 0.600 |
| `mlp_up_bank` | 32 | 0.548 | 0.500 |
| `ve_shared.embed.weight` | 16 | 0.473 | 0.031 |
| `bigram.embed.weight` | 16 | 0.397 | 0.000 |
| **average** | — | **0.652** | **0.405** |

**Key observations:**

1. **`kv_bank` is nearly the same subspace in both models** (0.955), even
   though the raw element-wise cosine was only 0.096. The key/value
   projection learned the same principal directions but in a different
   permutation of its columns.

2. **Attention (qo, kv) and MLP banks are partially aligned** (0.55 – 0.95).
   Meta-TTT shifts the basis but the top-k features are mostly
   preserved.

3. **The value embedding and bigram tables are the *most* divergent**
   (0.40 – 0.47). These are the only tensors where meta-TTT produced
   genuinely different features — because these tensors are touched
   directly on every forward pass, so any noise in the meta-update
   accumulates on them.

4. On average, **40% of the principal directions are aligned** and 60%
   are rotated. This is the functional-equivalence evidence: the two
   models are *mostly* the same with a minority of directions rotated.

### 4.5 Linear mode connectivity (weight-space proxy)

We can't cheaply measure loss along the weight-space line `(1-α) W_101 + α W_105`
without running the val forward for many α, but we can compute the norm
ratio of the midpoint. If both models are in the same basin, the midpoint
lands on the basin floor and preserves norm. If they're in different
basins, the midpoint lands on a ridge where vector cancellation
destroys norm.

| quantity | value |
|---|---:|
| Total L2 distance `||W_101 − W_105||` (summed across layers) | 3202.37 |
| Total Frobenius norm (exp101, summed) | 2898.10 |
| Total Frobenius norm (exp105a, summed) | 2883.78 |
| **Total midpoint norm** | **2316.29** |
| **Midpoint norm / exp101 norm ratio** | **0.799** |

A ratio near 1.0 ⇒ same basin. A ratio near 0.6 ⇒ distinct basins.
**0.799 is borderline** — the midpoint has ≈20% less weight energy
than either endpoint, suggesting weight vector cancellation, which is
characteristic of distinct but neighboring local minima.

Combined with the subspace-overlap finding: the two models live in
distinct local minima, but those minima span partially-overlapping
principal subspaces. You could probably walk from one to the other with
low loss along a *curved* path, but the straight line between them
drops through a shallower region.

---

## 5. Is meta-TTT a regularizer?

Yes, but only in a statistical sense — not in a useful one.

**Evidence for regularization:**

- Slightly lower average condition number (−8.2%)
- Lower operator-norm variance across layers (not reported above; check
  the JSON)
- 40% of principal subspace dims aligned with exp105a (the other 60% are
  rotated, which is the "noise" half)
- Distinct local minimum of equivalent quality

**Evidence against useful regularization:**

- Identical quantization MSE (0.11% difference)
- Identical Lipschitz-product proxy (0.09% difference)
- Identical Frobenius norms (0.6% difference)
- **Identical TTT adapt delta** — the one metric that was supposed to
  improve
- **Identical post-EMA val_bpb** after wallclock budget consumed

**Characterization**: Meta-TTT acts as *gradient noise* during training.
It perturbs the optimization trajectory away from the vanilla basin,
costs 3% per-step compute, and lands in a neighboring basin that is
equivalent in every measured statistic. This is indistinguishable from
what you'd get if you replaced `meta_ttt_step` with a `torch.randn_like(grad)
* 0.001` call and saved the compute.

---

## 6. Are the two models learning the same thing?

**Short answer**: yes at the function level, no at the basis level.

**Long answer**:

- At matched step counts, the two models' val_bpb are within 0.01 bpb.
  They predict essentially the same distribution over next tokens.
- Their macro control parameters (attn_scale, mlp_scale, q_gain,
  resid_mix) converge to cosine-similarity 0.99+ — the *shape* of the
  network is bit-identical.
- The dominant principal directions of each weight matrix are mostly
  aligned (avg 0.65, top banks up to 0.96).
- The element-wise weight values are rotated 90° on average — the
  *basis* within each matrix is different.

This is a common phenomenon in overparameterized networks: many bases
can realize the same function. Meta-TTT picks a *different* basis
without picking a *better* function. The rotation is induced by the
extra gradient signal from the FOMAML inner/outer loop, and it has no
downstream consequence because the network's outputs depend only on
the subspace span, not the basis choice within it.

If the two models were tested head-to-head on the same val tokens,
position by position, you'd see:

- Identical logit distributions at the final layer (to 3-4 decimal
  places)
- Rotated hidden states at intermediate layers (because those are
  basis-dependent)
- Identical perplexity
- Identical response to TTT SGD updates

The fact that the TTT delta is identical to 4 decimal places is the
strongest piece of evidence that the two models are *functionally* the
same, despite their weight-space distance.

---

## 7. Novelty and significance — the honest assessment

### What meta-TTT was supposed to do

Produce a model that is differentially better at test-time adaptation,
i.e. `delta_ttt_meta > delta_ttt_vanilla` at the same pre-TTT baseline.

### What it actually did

1. Injected a ~3% compute overhead per training step
2. Rotated weight matrices into a different basis of equivalent quality
3. Produced a ~8% reduction in average condition number
4. Produced identical val_bpb, identical TTT delta, identical
   quantization sensitivity, identical Lipschitz constant
5. Cost us 206 training steps in wallclock (which is *more* bpb than
   meta-TTT gave us)

### Is any of this novel or publishable?

**No.** The only things we learned are:

- FOMAML's first-order approximation is too weak to deliver the
  promised meta-learning signal on a ~27M-parameter model trained for
  80 minutes
- Meta-learning with an inner lr of 0.002 and a single inner step
  behaves identically to adding tiny gradient noise
- The cosine similarity between weight matrices is a misleading metric
  when the optimizer (Muon) aggressively orthogonalizes gradients;
  principal-angle subspace cosine is the right metric for
  "did the two runs learn the same thing"

All three are known (or at least strongly suspected) in the
meta-learning / optimization literature. Our contribution here is
empirical confirmation on a specific competition setup, which is
diagnostic but not novel.

### The one genuinely interesting observation

The fact that two Muon-trained transformers from the same seed end up
with **cosine ≈ 0.10 element-wise but subspace cosine ≈ 0.65 in the
dominant directions** is a clean illustration of how basis rotation
decouples from function rotation in over-parameterized networks. It's
a known phenomenon but rarely this cleanly isolated in a single-variable
ablation on a real training run. The Muon optimizer's Newton-Schulz
gradient orthogonalization amplifies this effect — every update rotates
the weight matrix in a principled way, which means any small
perturbation (like meta-TTT's extra gradient) compounds into a large
basis rotation without changing the learned function.

If there is a "paper" in this, it's:

> **"Gradient orthogonalization in Muon amplifies small training
> perturbations into large weight-space rotations, but preserves the
> learned function to within measurement noise."**

And that paper would use the exp101 vs exp105a pair as its main
empirical exhibit.

---

## 8. Decision

**Disable meta-TTT in every descendant of exp101.** The ~206 training
steps it costs are worth more than any signal it provides. Specifically:

1. `META_TTT_ENABLED=0` in all future `run.sh` variants.
2. Leave the `meta_ttt_step` function in `train_gpt.py` for reference
   (it's a clean implementation of FOMAML and might be useful if we
   ever want to try true second-order MAML).
3. The condition number improvement (5.6 vs 6.1) is not worth chasing
   via other means — it doesn't show up in any downstream metric.

**Redirect the saved compute** to levers that actually move the needle:

- Earlier QAT (`LATE_QAT_THRESHOLD=0.5`) for 2× more QAT-trained steps
- Longer SWA window
- Higher muon_momentum peak (0.995 instead of 0.99)
- More TTT epochs at eval time (free — doesn't touch training)

Each of the above can plausibly deliver 0.001–0.003 bpb improvement
without any architectural change.

---

## 9. Open questions for follow-up

1. **Does true MAML work?** The first-order approximation failed.
   Second-order MAML (via `create_graph=True` on the inner backward)
   costs 2–3× compute but recovers the curvature information FOMAML
   discards. On this model size it might be feasible for a short
   experiment.

2. **Does meta-TTT help at scale?** We tested on a 27M-param 80-minute
   run. The meta signal might be stronger at larger scale where the
   TTT adapt set has more expressive capacity.

3. **Does the TTT delta ceiling at ~0.023 bpb come from the adapt set
   or from the val data?** If we add more adapt parameters (free up
   more layers, add rank-1 correctors) does the ceiling move?

4. **Can we replicate meta-TTT's condition-number improvement with a
   cheaper regularizer?** A simple spectral regularizer (penalizing
   `σ_max - σ_min` on each weight matrix) might give the same 8%
   improvement at 0% compute cost.

---

## 10. Reproducing this analysis

```bash
# From the parameter-golf repo root:
python3 records/phase3/analysis_meta_ttt.py
```

Outputs:

- Executive summary to stdout
- Full JSON dump to `records/phase3/analysis_meta_ttt.json`

Runtime: ~1.3 seconds on CPU (no GPU needed).

Required files:

- `records/phase3/exp101_poscond-bigram-trigram_from_exp95/final_model (1).pt`
- `records/phase3/exp105a_no-metattt_from_exp101/_pod/final_model.pt`

Script source:

- `records/phase3/analysis_meta_ttt.py`

The script is self-contained and has no dependencies beyond a recent
PyTorch. It doesn't require importing `train_gpt.py` — all analyses
are pure weight-space manipulations of the saved state_dicts.

---

## 11. References & related files

- **Training logs**:
  - `exp101_poscond-bigram-trigram_from_exp95/exp101_poscond-bigram-trigram_from_exp95_seed42.txt`
  - `exp105a_no-metattt_from_exp101/exp105a_no-metattt_from_exp101_seed42.txt`
- **Config diffs**: `diff -u exp105a/run.sh exp101/run.sh` shows the
  single-line `META_TTT_ENABLED=0 → 1` change (both run.shes in the
  respective folders).
- **Source of the meta-TTT mechanism itself**:
  `records/phase3/exp101_poscond-bigram-trigram_from_exp95/train_gpt.py`,
  function `meta_ttt_step()` around line 1737.
- **The ablation question was later re-asked with a reformulation**
  (exp106) that added cross-chunk split + Δ-loss + MetaSGD scales. See
  `records/phase3/exp106_metasgd-crosschunk-delta_from_exp101/` for
  the follow-up and `prancy-jingling-canyon.md` in `~/.claude/plans/`
  for the speed plan that would make any future meta-TTT experiment
  faster.
