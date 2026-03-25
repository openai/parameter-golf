# Claude Code implementation brief: start from the current best record, then add a recurrent core with error correction

Implement a **minimal-diff branch** on top of the current best 10-minute / 16 MB Parameter Golf record:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

Use that record as the starting point for architecture, optimizer, quantization/export path, and legal test-time training (TTT). Do **not** start from PR #363 as the code base. Instead, use PR #363 only as the motivating failure case for recurrent quantization instability.

## Ground truth to preserve from the current best

Mirror the current best setup as closely as possible before adding recurrence:

- 11 layers, width 512, 8 heads, 4 KV heads
- MLP expansion 3× with `LeakyReLU(0.5)^2`
- `BigramHash=1536`
- XSA in the last 4 layers
- partial RoPE on 16 of 64 dims
- LayerNorm scaling of `1 / sqrt(layer + 1)`
- VE128 in layers 9–10
- EMA + tight SWA
- `GPTQ-lite int6 + lzma`
- Parameter Banking + Parallel Muon
- legal score-first TTT with 32K-token chunks, SGD with momentum, 3 epochs per chunk, all blocks unfrozen in the current record, and total eval under the time budget

The current best README reports a **3-seed mean of 1.1194 BPB**, roughly **15.95 MB** artifacts, about **83.4 ms/step**, and about **409 s** of TTT time inside a total evaluation time around **530 s**. Preserve that spirit: recurrence must be added without breaking the training/eval budget or the export path.

## Why recurrence needs extra care

PR #363 is the reference failure mode, not the base implementation. Its summary describes a looped architecture where shared blocks were reused across recurrence cycles. The PR reports that a `4 unique blocks × 3 cycles` setup went from **2.0711 BPB pre-quant** to **2.4402 BPB post-quant**, with the writeup attributing the collapse to roughly **900× amplification** of quantization error through recurrence. It also reports that a separate “noisy QAT” experiment largely removed the recurrence quantization gap. Treat that as the problem this branch is solving.

In the recurrent setting, the report’s dynamical-systems framing is the right mental model. With shared quantized weights

$$
W_q = W + \varepsilon,
$$

a first-order perturbation grows roughly like

$$
\lVert J \rVert^k \cdot \lVert \varepsilon h_0 \rVert,
$$

where $J$ is the Jacobian of the shared update and $k$ is the number of recurrence passes. This means recurrence is **not** “free extra depth”; it is a noisy iterative system.

## Design objective

Build a recurrent variant that **inherits as much as possible from the current best record** while changing only the transformer body where necessary.

Concretely:

- keep the current best tokenizer, data path, optimizer stack, EMA/SWA, export flow, and legal TTT protocol,
- keep the current best non-recurrent baseline runnable,
- replace only a contiguous middle portion of the 11-layer stack with a shared recurrent core,
- and add explicit training-time + test-time correction for quantization error.

## Recommended architecture migration

Do **not** convert the entire 11-layer stack into one shared loop immediately.

Instead, start from the current best 11-layer stack and partition it into:

1. **stem**: early unique layers
2. **recurrent core**: a shared block or shared block-group repeated for `K` passes
3. **tail**: late unique layers

### Safe first migration

A good first version is:

- keep layers 0–2 unique
- replace layers 3–7 with a recurrent core derived from 1–2 shared layers repeated `K` times
- keep layers 8–10 unique

This preserves the current best model’s input processing and output refinement while localizing recurrence to the middle, where it is least likely to break the full recipe.

## Recurrent update equations

Let the current-best-derived stem produce

$$
h_0 = \text{Stem}(x).
$$

Let the recurrent core update be

$$
h_{k+1} = f_{W_q}(h_k + c_k), \qquad k = 0, \dots, K-1,
$$

and let the final tail and LM head produce

$$
\text{logits} = \text{LMHead}(\text{Tail}(h_K)).
$$

The correction term $c_k$ depends on the script variant.

## Research constraints to encode

### 1. Full-rollout QAT is mandatory

All recurrence passes must be present during training. Compute the LM loss only after the final recurrent pass:

$$
\mathcal{L} = \operatorname{CE}(\text{LMHead}(\text{Tail}(h_K)), y).
$$

Use STE fake quantization inside the shared recurrent core during the rollout.

### 2. Heterogeneous precision should remain available

Per the report’s cited recurrent-model quantization literature, the recurrently reused matrices are the sensitive ones. Therefore:

- prioritize fake quant and export quant support for the shared attention and MLP matrices,
- keep LayerNorm and tiny scalars in higher precision,
- and make it easy to leave embeddings and the LM head less aggressively quantized.

### 3. Error feedback is the main experimental lever

Use the report’s delta-sigma / error-feedback idea as the center of the implementation.

Approximate the quantization residual action by

$$
e_k \approx (W - W_q) h_k,
$$

then inject a compensation term into the next pass:

$$
c_k = D_k e_k,
$$

$$
h_{k+1} = f_{W_q}(h_k + c_k).
$$

Do **not** store the full dense residual matrix.

### 4. Low-rank residual approximation is the default practical path

Implement

$$
e_k \approx U(V^\top h_k),
$$

where $U, V \in \mathbb{R}^{d \times r}$ and $r \in \{1,2,4\}$.

This is the small-parameter correction branch that fits the Parameter Golf budget.

### 5. Jacobian control, clipping, and residual scaling are secondary stabilizers

Add optional flags for:

- hidden-state clipping between passes,
- per-pass residual scaling,
- a light Jacobian proxy regularizer.

Use them as ablations and safety rails.

### 6. TTT and recurrence interact, so default conservatively

The current best record relies heavily on legal TTT. In a recurrent model, updating shared weights during TTT affects **all** recurrence passes at once. Therefore the recurrent branch should support safer TTT defaults, including:

- freezing the recurrent core during TTT,
- adapting only tail layers during TTT,
- or using a smaller TTT LR for the recurrent core than for the unique layers.

Keep the original “all blocks unfrozen” TTT option available, but do not make it the only path.

## Files to create

Create these top-level scripts:

1. `train_bestbase_recurrent_qat.py`
2. `train_bestbase_recurrent_feedback_fixed.py`
3. `train_bestbase_recurrent_feedback_learned.py`

Create these shared modules:

- `model_recurrent_bestbase.py`
- `quant.py`
- `feedback.py`
- `stability.py`
- `ttt_recurrent.py`
- `train_utils_recurrent.py`

## Script 1: `train_bestbase_recurrent_qat.py`

### Goal

Take the current best architecture and replace the chosen middle stack with a shared recurrent core trained with **full-rollout QAT**, but without explicit error feedback.

### Forward pass

Run:

$$
h_0 = \text{Stem}(x),
$$

$$
h_{k+1} = f_{W_q}(h_k),
$$

$$
\text{logits} = \text{LMHead}(\text{Tail}(h_K)).
$$

### Requirements

- preserve the current best defaults outside the recurrent core,
- apply fake quant only to the shared recurrent core by default,
- compute loss only after the final pass,
- support `num_passes`, `shared_core_layers`, and `recurrent_layer_range` flags,
- preserve the current-best export path as closely as possible.

### Purpose

This script answers: how much of the recurrence problem disappears if we simply start from the current best recipe and train the real quantized recurrent rollout?

## Script 2: `train_bestbase_recurrent_feedback_fixed.py`

### Goal

Add a small fixed-form error-feedback path on top of Script 1.

### Residual approximation

Use

$$
e_k = U(V^\top h_k)
$$

with tiny rank by default.

### Correction options

Support at least:

1. identity correction

$$
c_k = e_k
$$

2. shared diagonal correction

$$
c_k = d \odot e_k
$$

where $d \in \mathbb{R}^d$ is learned or initialized at ones.

### Recurrent update

Use

$$
h_{k+1} = f_{W_q}(h_k + c_k).
$$

### Requirements

- correction is inactive on pass 0,
- full-rollout QAT remains enabled,
- keep parameter overhead tiny,
- log correction norms and per-pass activation growth.

### Purpose

This script tests whether a tiny correction path can recover recurrence while leaving the current best recipe mostly untouched.

## Script 3: `train_bestbase_recurrent_feedback_learned.py`

### Goal

Make the correction operator explicitly learnable.

### Residual approximation

Use

$$
e_k = U(V^\top h_k).
$$

### Learned correction operator

Support:

1. **shared diagonal**

$$
c_k = D e_k, \qquad D = \operatorname{diag}(d)
$$

2. **per-pass diagonal**

$$
c_k = D_k e_k
$$

3. **shared low-rank**

$$
c_k = U_D(V_D^\top e_k)
$$

4. **per-pass low-rank**

$$
c_k = U_{D,k}(V_{D,k}^\top e_k)
$$

### Recurrent update

Use

$$
h_{k+1} = f_{W_q}(h_k + c_k).
$$

### Requirements

- full-rollout QAT stays on,
- learned correction trains jointly with the recurrent core,
- optional affine junction correction is available,
- optional Jacobian proxy regularization is available,
- support warm-start phases that freeze the recurrent core while fitting correction modules.

### Purpose

This is the strongest version and the main target for experiments.

## Base-model implementation requirements

## `model_recurrent_bestbase.py`

This module should be a minimal diff on top of the current best training stack.

### Preserve from the current best

- LeakyReLU(0.5)^2 MLP
- BigramHash path
- XSA last-4-layer support where still applicable
- partial RoPE behavior
- VE128 in layers 9–10 if those layers remain unique
- Parameter Banking compatibility where practical
- EMA/SWA hooks

### Recurrent-core insertion

Allow the user to specify which contiguous layers are replaced by the recurrent core. The recurrent core can be either:

- a single shared block,
- or a small shared block group repeated `K` times.

### Correction injection API

The recurrent block/group must accept an optional correction tensor:

```python
def forward(self, x, correction=None, ...):
    if correction is not None:
        x = x + correction
    ...
    return x
```

## `quant.py`

Implement fake quantization and export helpers suitable for the shared recurrent core.

### Required features

- symmetric quantization,
- configurable bits (support at least 5, 6, 8),
- per-tensor and per-row modes,
- selective application to the recurrent core,
- export helper that matches training fake quant closely.

For a weight tensor $W$ with scale $s$:

$$
q = \operatorname{clip}\left(\operatorname{round}(W / s), q_{\min}, q_{\max}\right),
$$

$$
W_q = s q.
$$

Use STE so the forward uses $W_q$ while gradients flow through $W$.

## `feedback.py`

Implement:

### `LowRankResidual`

$$
e_k = U(V^\top h_k)
$$

### `DiagonalFeedback`

$$
c_k = d \odot e_k
$$

### `LowRankFeedback`

$$
c_k = U_D(V_D^\top e_k)
$$

### Optional `AffineJunction`

$$
c_k^{\text{aff}} = \gamma_k \odot h_k + \beta_k
$$

Keep all of these lightweight and sequence-shape aware.

## `stability.py`

Implement:

### Per-pass diagnostics

Track:

- $\lVert h_k \rVert$
- $\lVert h_{k+1} - h_k \rVert$
- $\lVert e_k \rVert$
- $\lVert c_k \rVert$

### Growth proxy

$$
\rho_k^{\text{emp}} = \frac{\lVert h_{k+1} \rVert}{\lVert h_k \rVert + \epsilon}
$$

### Optional clipping

$$
h_k \leftarrow \operatorname{clip}(h_k, -\alpha, \alpha)
$$

or norm clipping.

### Optional residual scaling

$$
h_{k+1} = h_k + \alpha_k F(h_k + c_k)
$$

### Optional Jacobian proxy penalty

Add a cheap finite-difference sensitivity penalty under a flag.

## `ttt_recurrent.py`

Implement a recurrent-aware TTT wrapper around the current best legal TTT protocol.

### Scoring phase

Preserve the current record’s score-first requirement:

- score each chunk under `torch.inference_mode()`
- do not mutate weights during scoring

### Adaptation phase

Support TTT regimes:

1. `tail_only`
2. `tail_plus_stem`
3. `all_unique_layers`
4. `all_layers`
5. `all_layers_with_recurrent_lr_scale`

Also support:

- separate LR scale for recurrent core,
- separate freeze mask for correction modules,
- momentum SGD as in the current best record.

## CLI flags to add

Support at least:

- `--recurrent-layer-range`
- `--shared-core-layers`
- `--num-passes`
- `--quant-bits`
- `--quant-mode`
- `--feedback-rank`
- `--feedback-mode`
- `--per-pass-feedback`
- `--affine-junction`
- `--clip-hidden`
- `--clip-value`
- `--residual-scale-init`
- `--jacobian-proxy-weight`
- `--ttt-regime`
- `--ttt-recurrent-lr-scale`
- `--leave-embeddings-fp16`
- `--leave-head-fp16`

## Experimental plan

Run the experiments in this order.

### Experiment A: preserve the current best, add only recurrence + QAT

- start from the current best defaults
- replace a middle layer range with a recurrent core
- use full-rollout QAT
- no error feedback
- no TTT changes yet

### Experiment B: fixed feedback

- same as A
- add low-rank residual branch
- identity or shared diagonal feedback

### Experiment C: learned feedback

- same as B
- learned diagonal or low-rank correction operator

### Experiment D: recurrent-aware TTT ablation

Using the best model from C, compare:

- `tail_only`
- `all_unique_layers`
- `all_layers_with_recurrent_lr_scale`

This directly checks whether the current best TTT recipe survives the shared-core setting.

### Experiment E: stabilizer ablation

Test:

- hidden clipping,
- residual scaling,
- affine junction correction,
- Jacobian proxy penalty.

## Logging requirements

At each debug or validation interval, log:

- train loss,
- val loss,
- val BPB,
- per-pass activation norms,
- per-pass empirical growth ratios,
- correction norms,
- gradient norm,
- step time,
- pre-TTT vs post-TTT BPB,
- pre-quant vs fake-quant gap if cheap to compute.

## Success criteria

The branch should make it easy to answer:

1. Can the current best recipe remain competitive after replacing part of the 11-layer stack with a recurrent core?
2. How much does full-rollout QAT repair the recurrent quantization failure by itself?
3. How much more does fixed feedback recover?
4. Does learned feedback beat fixed feedback at the same tiny parameter budget?
5. Which TTT regime is safest for shared recurrent weights?

## Final deliverables

Return:

1. the three training scripts,
2. the shared modules,
3. a short `README_recurrent_from_bestbase.md`,
4. and a concise note describing:
   - what was preserved from the current best record,
   - what was borrowed conceptually from PR #363,
   - and what was changed to make recurrence quantization-stable.

## Final implementation principle

Treat the current best record as the **production-grade scaffold** and PR #363 as the **failure case to solve**.

That means:

- preserve the winning optimizer / quantization / TTT / architecture defaults wherever possible,
- localize recurrence to the smallest part of the stack that can still save parameters,
- and treat quantized recurrence as a controlled dynamical system with explicit error correction.
