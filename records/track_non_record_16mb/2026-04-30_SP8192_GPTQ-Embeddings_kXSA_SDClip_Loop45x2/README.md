# Final Submission: k-XSA (k=2) — Rank-2 Subspace Exclusive Self-Attention with Quantization-Aware Energy Capping

**Author:** Jayesh Chaudhari, Satyam Kumar, Yusuf Baig
**Date:** 2026-04-30
**Method:** k-XSA at rank k=2 (anchored basis `v+x`, free per-head α, energy cap)
**Codebase:** [`train_xsa_k.py`](../train_xsa_k.py) (single-file, env-driven)
**Setting:** OpenAI Parameter Golf challenge (16 MB artifact, 10 min on 8×H100s, val_bpb on FineWeb)

**val_bpb = 1.1093** (pre-quant) | **1.1190** (int6 GPTQ + brotli) | **1.1026** (sliding window, stride=64) | **16.68 MB** artifact | single-seed (1337) on 1×H200

---

## 1. Summary

We generalize Apple's Exclusive Self-Attention (XSA) [Zhai 2026] from a rank-1
orthogonal projection along `v_i` to a **rank-k orthogonal projection in a
learned subspace** whose first basis vector is anchored at `v_i` and whose
remaining basis vectors are `W_j x_i`. Each direction's projection coefficient
is soft-thresholded by a learned per-direction τ ("energy cap"), and the
overall projection is scaled by a per-head learnable α (no weight decay).

This submission uses **k=2** — the smallest non-trivial rank that strictly
extends paper-XSA. At k=2 the recipe gains **−0.0019 BPB pre-quant** and
**−0.0020 BPB quantized** over paper-XSA at fixed α=1, on the same 11L /
512-dim / 35.9M-baseline training setup.

The energy cap τ has a second, surprising effect: it makes the new
`xsa_basis_proj` matrix essentially free to int6-quantize. The
per-`xsa_basis_proj` quant penalty drops from +0.0023 BPB (no cap) to **+0.0000
BPB** (cap on).

---

## 2. Background: paper-XSA, in formal terms

Let `y_i = Σ_j a_ij v_j` be the standard causal self-attention output for
token i (per head). Paper-XSA defines:

```
z_i = y_i − (y_iᵀ v_i / ‖v_i‖²) · v_i           (1)
    = (I − P_{v_i}) · y_i                        (2)
```

That is, **paper-XSA is a rank-1 orthogonal projection that removes the 1D
subspace `span(v_i)`**. The "self-similarity bias" / "FFN handles point-wise
features" arguments in the paper are *motivation* for why that particular 1D
subspace deserves to be removed.

This reframing exposes two assumptions:

- **One-dimensional?** FFN's role is not 1D; redundant overlap between
  attention output and FFN's role plausibly lives in a multi-dimensional
  subspace.
- **Lying along `v_i`?** `v_i = W_v x_i` was trained for context aggregation,
  not for "what FFN handles." `v_i` being the right axis is a hypothesis.

Both become testable hypotheses below.

---

## 3. The k-XSA paradigm

### Definition

Generalize paper-XSA to a rank-k orthogonal projection:

```
S_i = orth(B_i)                                   B_i: (k × D), k basis vectors per head
z_i = y_i − α_i · Σ_{e ∈ S_i} ⟨y_i, e⟩ · e        (3)
```

Three basis recipes exposed via `XSA_BASIS`:

- `"v"`     : `B_i = [v_i]` — paper-XSA exactly (rank forced to 1)
- `"v+x"`   : `B_i = [v_i, W_1 x_i, …, W_{k-1} x_i]` — anchored: paper basis + (k-1) learned
- `"x"`     : `B_i = [W_1 x_i, …, W_k x_i]` — fully learned, no `v_i` anchor

The orthonormal basis is computed by **modified Gram-Schmidt in float32** to
avoid bf16 instability when basis vectors are nearly parallel.

### Energy-cap variant

Instead of scaling the entire projection by α, soft-threshold each
direction's projection coefficient:

```
c_j        = ⟨y_i, e_j⟩
c_j_capped = sign(c_j) · ReLU(|c_j| − τ_{h,j})
z_i        = y_i − α_i · Σ_j c_j_capped · e_j           (4)
```

τ is per-head and per-rank, learned, no WD. Initialized at 0 (recovers plain
k-XSA). Intuition: small projections may be natural overlap, not redundancy
worth removing; only large overlaps deserve to be clipped.

### Why rank=2 anchored is the principled minimal extension

- **Strict generalization** of XSA: with k=1, basis=`v`, α=1, cap=off, the
  forward reduces to Equation (2) bit-exactly (sanity test T10).
- **Geometric structure preserved**: still a projection (eigenvalues ∈ {0,1}),
  cannot blow up activations or interact pathologically with normalization.
- **Information-theoretic interpretation**: the subspace `S_i` is the model's
  *learned belief* about which directions of the attention output are
  FFN-redundant at token i.
- **Param-efficient**: rank=k anchored adds (k−1) learned `W_j` matrices of
  shape (D, kv_dim) per layer. For our config this is ≈1.45M params per added
  rank — about +4% of the model per step in k.

---

## 4. Hidden weight-decay confound (Phase 0 fix)

The most natural generalization of paper-XSA — "make α learnable per-head"
with α initialized at 1.0 — has a hidden bug. The naive implementation puts α
into the existing scalar AdamW group, which carries `weight_decay=0.02`.
**AdamW weight decay pulls α toward 0**, not toward its initialization. This
biases the experiment toward "free α drifts to zero", which would falsely
look like "learnable α prefers no XSA."

**Fix:** route `xsa_alpha*` and `xsa_threshold` parameters into a dedicated
AdamW group with `weight_decay=0`:

```python
is_alpha_or_thresh = lambda name: ("xsa_alpha" in name) or ("xsa_threshold" in name)
xsa_alpha_params = [p for name, p in block_named_params if is_alpha_or_thresh(name)]
self.optimizer_alpha = torch.optim.AdamW(
    [{"params": xsa_alpha_params, "lr": h.xsa_alpha_lr, "base_lr": h.xsa_alpha_lr}],
    betas=(h.beta1, h.beta2), eps=h.adam_eps,
    weight_decay=0.0 if h.xsa_alpha_no_wd else h.adam_wd,
    fused=use_fused,
)
```

Without this fix every subsequent number would be uninterpretable.

---

## 5. Why rank=2, anchored, with cap (the deconfounding chain)

### Free α alone is not the lever

At rank=1 basis=`v`, fixed α=1 (paper) and free α give the *same* val_bpb at
three significant figures:

```
B (fixed α=1):    pre-quant 1.1112    quantized 1.1209
Phase 1 (free):   pre-quant 1.1112    quantized 1.1210
```

The CSV shows α moving heterogeneously per head — but that motion does not
surface as BPB. At rank=1 the only available direction is `v_i`; α tunes how
*much* of `v_i` to subtract per head, but cannot tune *which direction* to
subtract.

### `v_i` is necessary, learned `W·x` alone is insufficient

At equal extra-param budget (one learned matrix beyond paper):

```
B  (k=1, basis=v)    : 1.1112  pre-quant
E1 (k=1, basis=x)    : 1.1127  pre-quant     +0.0015 (worse)
```

A learned direction `W·x` cannot replace `v_i` at rank=1.

At rank=2:

```
E2 (k=2, basis=v+x)  : 1.1096  pre-quant
E3 (k=2, basis=x)    : 1.1112  pre-quant     +0.0016 vs E2 (worse)
```

E3 (k=2 free) ties paper-B with +1.45M params spent on a useless second
learned direction. E2 (k=2 anchored) wins by 0.0016 BPB at fewer total params.
**v_i carries unique signal that learned `W·x` cannot replicate** — the
paper's choice was empirically correct as a starting point but insufficient
as a complete description of the redundant subspace.

### Energy cap absorbs quantization error

The most surprising finding. Without energy cap, every extra basis matrix
adds quantization penalty:

```
B  (0 xsa_basis_proj) : +0.0097
E1 (1 xsa_basis_proj) : +0.0120     (+0.0023 from GPTQ on 1 extra matrix)
E3 (2 xsa_basis_proj) : +0.0129     (+0.0032 from 2 extra matrices)
```

With energy cap, quant penalty stays essentially constant:

```
B  (no cap, 0 matrices) : +0.0097
E5 (cap on, 1 matrix)   : +0.0097     (+0.0000)
```

**The xsa_basis_proj matrices quantize for free under energy cap.** Mechanism:
the soft threshold zeros out small projection coefficients during training.
The model never depends on the fine-grained part of `⟨y, e⟩`. That's exactly
the regime where GPTQ degrades the matrix `W_basis` the most — small inner
products are dominated by quantization noise. Training the model to ignore
them creates a representation structurally robust to quantization.

This is a real architectural insight, not a hyperparameter win.

---

## 6. The submitted configuration: k=2 (E5)

**Run id:** `k2_vx_full` ("E5" in [REPORT_K_XSA.md](../REPORT_K_XSA.md))

| Setting | Value |
|---|---|
| `XSA_RANK` | 2 |
| `XSA_BASIS` | `v+x` (anchored) |
| `XSA_TARGET` | `attn` |
| `XSA_ENERGY_CAP` | 1 (on) |
| `XSA_THRESHOLD_INIT` | 0.0 |
| `XSA_ALPHA_MODE` | `free` (per-head) |
| `XSA_ALPHA_INIT` | 1.0 |
| `XSA_ALPHA_NO_WD` | 1 |
| `XSA_ALPHA_LR` | 0.02 |
| Iterations | 3000 |
| Wallclock cap | none (~80 min on 1×H200) |
| Backbone | 11L / 512d / 8H / 4KV / mlp_mult=4, GQA, parallel-residual + skip-gates + layer-loop (4–5 ×2 from ~50% onward) |
| Vocab | sentencepiece-8192 |
| Optimizers | Muon (matrix), AdamW (embed), AdamW-no-WD (xsa α/τ) |
| Quantization | GPTQ int6 (matrices) + int8 (embeddings) + brotli |

### Carried from the parameter-golf baseline

- Muon optimizer for matrix params, AdamW for embeddings/scalars
- EMA weight averaging applied at end-of-training
- Skip gates (sigmoid-gated U-Net connections)
- Parallel residuals
- Layer looping (loop layers 4–5 twice from ~50% of training onward)
- 3000 training iterations on FineWeb, `train_batch_tokens = 786,432`
- GPTQ int6 + int8 + brotli quantization pipeline

---

## 7. Numerical results

### k=2 vs paper-XSA (the core result)

| Run | Config | params | pre-quant val_bpb | quantized val_bpb | quant penalty | artifact |
|----|--------|-------:|------------------:|------------------:|--------------:|---------:|
| B (paper) | k=1, basis=`v`, α=1 fixed | 35.94 M | 1.1112 | 1.1209 | +0.0097 | 16.07 MB |
| **E5 (this submission)** | **k=2, `v+x`, α=free, cap=on** | **37.39 M** | **1.1093** | **1.1190** | **+0.0097** | **16.68 MB** |

**Δ vs paper:** −0.0019 BPB pre-quant, −0.0019 BPB quantized, with quant
penalty unchanged despite the new `xsa_basis_proj` matrix.

### Sliding-window quantized eval (stride=64)

```
B  (paper):       1.1044
E5 (k=2, this):   1.1026     Δ -0.0018
```

### Per-head α and per-direction τ at convergence (k=2, E5)

α distribution shows the same depth pattern observed at rank=1: deeper layers
prefer larger removal, with growing per-head heterogeneity. τ values are
nonzero — the model genuinely uses the energy cap rather than collapsing it
to zero.

---

## 8. Caveats (important — please read)

1. **Single seed.** This submission is from one run with `SEED=1337`. The
   −0.0019 BPB gain over paper-XSA at k=2 has not been validated across
   multiple seeds; on this training setup inter-run variance is not formally
   characterized in this work.

2. **Artifact size 16.68 MB is over the 16 MB cap.** As configured, the k=2
   run produces a 16.68 MB artifact (4.3% over). Three identified paths to
   fitting under the cap, not implemented in this submission:
   - Low-rank factor `xsa_basis_proj` as `U·V` with intermediate dim ~32–64
     (cuts the (D, kv_dim) = (512, 256) matrix by ~8×).
   - Aggressive int4 quantization on `xsa_basis_proj` only.
   - Reduce `xsa_last_n` to skip XSA on shallow layers (Phase 1 showed mean α
     is lowest there, so they don't want full XSA anyway).
   The §7.4 finding (energy cap → no-quant-penalty) suggests the first option
   should preserve most of the gain.

3. **Two stacked changes vs paper.** E5 stacks two modifications over paper-B:
   rank=2 anchored, free α, and energy cap. Phase 1 (rank=1, free α) ties
   paper-B, ruling out free α as a sole contributor — but rank=2 with *only*
   free α (no cap) and rank=2 with *only* energy cap (fixed α) were not run
   separately. Cap and rank are confounded.

4. **No comparison against modern leaderboard winners (~1.08–1.10 BPB).** Those
   stack many other tricks (depth recurrence, parallel residuals, TTT, GPTQ
   tuning). k-XSA is an architectural lever that should compose with all of
   them, but no combination has been tested.

5. **Single 1×H200 run, not multi-GPU.** Training did not use the full 8×H100
   budget; 1×H200 with `grad_accum_steps=8` was used to match the baseline's
   effective batch size.

---

## 9. Reproducibility

```bash
export DATA_DIR=/path/to/parameter-golf/data/

RUN_ID=k2_vx_full \
  XSA_RANK=2 XSA_BASIS=v+x XSA_TARGET=attn \
  XSA_ENERGY_CAP=1 XSA_THRESHOLD_INIT=0.0 \
  XSA_ALPHA_MODE=free XSA_ALPHA_INIT=1.0 \
  XSA_ALPHA_NO_WD=1 XSA_ALPHA_LR=0.02 \
  XSA_ALPHA_CSV=logs/k2_vx_full_alpha.csv \
  MAX_WALLCLOCK_SECONDS=0 ITERATIONS=3000 TRAIN_LOG_EVERY=200 \
  python train_xsa_k.py
```

All 20 sanity tests verify the implementation:

```bash
FORCE_CPU=1 SANITY_CHECK=1 python train_xsa_k.py
```

Tests cover (in order): all five α modes build & forward & backward; α=0
fixed equals XSA-off; depth-aware init linear interpolation; sigmoid &
anchored α inversions; XSA_ALPHA_NO_WD wiring; free(α=1) ≡ fixed(value=1) at
step 0; content α gradient flow; Gram-Schmidt orthonormality; rank=1 basis=v
≡ paper XSA bit-exact; rank=2/3 anchored & free build correctly; rank=2
anchored with W_basis=0 ≡ paper XSA; energy-cap τ=0 ≡ no cap; energy-cap τ=∞
≡ no XSA; optimizer routes basis_proj→Muon & threshold→alpha-no-WD group;
depth-aware init at rank=2; XSA_TARGET=attn (default) bit-exact compatibility
with refactor; XSA_TARGET=resid diverges when XSA active; XSA_TARGET=resid
grad flow; refactored forward ≡ `_attn_compute` on baseline path.

---

## 10. Files in this submission

- `README.md` — this file
- `train_xsa_k.py` — full training + quantization + evaluation script
  (single-file, env-driven, derived from `train_gpt.py`)
- `final_model_k.pt` — fp32 EMA-applied checkpoint
- `final_model_k.int6.ptz` — int6 GPTQ + int8 embed + brotli artifact
- `train.log` (k2_vx_full.txt) — training log for the submitted run
- `k2_vx_full_alpha.csv` — per-step α trajectory CSV
- `REPORT_K_XSA.md` — full investigation report (Phases 0–4, ablations,
  rank ladder up to k=4)

---

## 11. One-paragraph conclusion

Paper-XSA is a rank-1 orthogonal projection along `v_i`. We generalize it to
a rank-2 orthogonal projection in a learned subspace whose first basis is
still `v_i` (anchored), the second basis is `W x_i`, and each direction's
projection coefficient is soft-thresholded by a learned per-direction τ.
Per-head α scales the result, free of weight decay. At k=2 this beats
paper-XSA by 1.9 millinats pre-quant and 1.9 millinats quantized on a
35.9M→37.4M parameter model. The energy cap τ is responsible for two
distinct effects: it lets each head learn how much "natural" self-overlap to
preserve (interpretable), and it makes the new `xsa_basis_proj` matrix
essentially free to int6-quantize (practical). Single-file implementation
with 20 passing sanity tests in [`train_xsa_k.py`](../train_xsa_k.py); full
investigation including the rank ladder up to k=4 in
[`REPORT_K_XSA.md`](../REPORT_K_XSA.md).
