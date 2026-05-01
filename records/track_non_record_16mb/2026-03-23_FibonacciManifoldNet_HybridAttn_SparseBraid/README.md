# FMN v11b — Hybrid Fibonacci Manifold Network + Causal Self-Attention

**Track:** non-record / 16MB (novel architecture showcase)  
**Architecture:** Hybrid FMN+XSA — Fibonacci Manifold Network with top-layer causal attention and sparse entity memory  
**Best val_bpb (pre-quant):** 1.4137 @ step 47,000  
**Submission val_bpb (post-quant roundtrip):** 1.4233  
**Post-quant penalty:** +0.0001 BPB (essentially lossless)  
**Model size (int8+zlib):** 14,716,657 bytes (14.72 MB) ✓  
**Total submission size:** 14,785,394 bytes (14.79 MB) ✓ (cap: 16,000,000 bytes)

---

## What This Is

A language model that replaces the standard transformer block entirely — no multi-head self-attention in 9 of 12 layers, and the feed-forward component (PsiNet) operates only on the ψ-eigenspace residual rather than the full hidden state. Token and learned positional embeddings feed into the Fibonacci backbone. Every layer is built around a single shared matrix **W₀** constrained to satisfy the Fibonacci characteristic equation:

```
W₀² = W₀ + I
```

This forces W₀'s eigenvalues to exactly `{φ ≈ 1.618, ψ ≈ −0.618}` — the golden ratio roots. The constraint is enforced throughout training via retraction-based Riemannian optimization (eigenvalue snap after each step), keeping `‖W₀² − W₀ − I‖_F ≈ 1.75×10⁻⁶` across 50,000 steps.

The top 3 of 12 layers additionally include 4-head causal self-attention (XSA) and a sparse entity memory register (Braid). In this hybrid, XSA provides the explicit global token-binding path, while braid provides a recurrent slot memory with local write search and learned read-back.

---

## Architecture

### Two parallel streams through all layers

```
Φ-stream:  M₀ → M₁ → ... → M_L        (manifold state, W₀ applied at each depth)
Ψ-stream:  Σ |ψ|^k · M_ψ_k, k=1..L   (structural residuals, depth-weighted decay)
```

Every depth step decomposes the hidden state into its φ-component and ψ-component via an exact closed-form projector derived from the Fibonacci recurrence.

Given `x` (the pre-W₀ state) and `M_k = x @ W₀ᵀ` (one application of the operator), the Fibonacci characteristic `W₀² = W₀ + I` implies that any vector `x` decomposes as `x = x_φ + x_ψ` where `x_φ @ W₀ᵀ = φ·x_φ` and `x_ψ @ W₀ᵀ = ψ·x_ψ`. From these two equations:

```
M_φ = (x + φ·M_k) / (1 + φ²)     ← exact eigenspace projection
M_ψ = x − M_φ                      ← complementary component
```

**Derivation:** Write `x = x_φ + x_ψ`. Then `M_k = x @ W₀ᵀ = x_φ @ W₀ᵀ + x_ψ @ W₀ᵀ = φ·x_φ + ψ·x_ψ`. So `x + φ·M_k = (1+φ²)·x_φ + (1+φψ)·x_ψ`. Since `φψ = −1` (product of roots of `λ²−λ−1=0`), the second term vanishes: `x + φ·M_k = (1+φ²)·x_φ`. Dividing gives `M_φ = x_φ` exactly.

This is equivalent to Richardson extrapolation applied to the pair `(x, W₀x/φ)` — the alternating `ψ`-component error cancels exactly between the two samples because `ψ = −φ⁻¹`.

### Lower layers (0–8) — pure FMN

```
Input M_{k-1}
  → LN → FibonacciMixer → +residual → M_seq
  → LN → x @ W₀ᵀ  →  M_k
  → φ-projector: M_φ = (x + φ·M_k) / (1 + φ²)
  → M_ψ = x − M_φ → LN → PsiNet (2-layer MLP: Linear→LeakyReLU(0.5)→square→Linear)
  → M_out = M_seq + dropout(M_φ + PsiNet(M_ψ))
  → M_ψ also emitted to Ψ-stream (weighted by |ψ|^k)
```

O(T·kernel·d) per layer. No attention matrix in layers 0–8. Causality in FibonacciMixer enforced by left-padding (`padding=kernel_size−1`, truncate to `[:T]`).

### Top layers (9–11) — FMN + XSA + Braid

```
Input M_{k-1}
  → LN → FibonacciMixer → +residual → M_seq
  → LN → TinyCausalSelfAttention (4 heads, causal) → +residual
  → LN → x @ W₀ᵀ  →  M_k
  → φ-projector: M_φ = (x + φ·M_k) / (1 + φ²)
  → M_ψ = x − M_φ → LN → PsiNet (2-layer MLP: Linear→LeakyReLU(0.5)→square→Linear)
  → M_out = M_seq + dropout(M_φ + PsiNet(M_ψ))
  → M_ψ also emitted to Ψ-stream (weighted by |ψ|^k)
  → SparseBraidRegister (shared recurrent K=7 slots, d_b=128, window=64)   ← persistent entity memory
```

### FibonacciMixer

Learned input gate controlling how much Fibonacci-weighted context vs. raw token identity:

```
output = σ(gate(x)) ⊙ proj(conv(x))  +  (1 − σ(gate(x))) ⊙ x
```

Initialized with weights `w[j] = φ^{-(kernel-j)}` — most-recent position has weight φ⁻¹, oldest has weight φ⁻²⁵⁶. At kernel=256, each layer can mix over the 256 most recent tokens with Fibonacci-weighted decay. Stacking 12 layers gives a theoretical receptive field of 3,072 tokens, though in practice the block size is 1,024 and signal attenuates exponentially through the φ-decay.

### TinyCausalSelfAttention (XSA)

4-head causal self-attention with pre-LayerNorm, used in top-3 layers only. Attention weight matrices fall under the Muon optimizer (Newton-Schulz orthogonalization). The attention-specific `O(T²·d)` path is present in 3 of 12 layers rather than all 12.

The motivation: FibonacciMixer is excellent at position-local mixing (Fibonacci-weighted lookback), but cannot resolve global co-reference or long-range token binding. XSA handles exactly that, at the layers where it matters most.

### SparseBraidRegister

A single recurrent `K=7` slot state is initialized once per sequence item and then updated by each braid-enabled top layer; each of those layers has its own braid parameters. Each slot has width `d_b=128`. The slots maintain persistent representations of recurring entities across the sequence:

- **TopK sparse write**: For each token `i` in the sequence, compute bilinear salience `x_i · (W₀ · x_j)` against tokens `j` in a local causal window of 64 preceding positions. Select the top-16 (i, j) pairs globally. Compute displacement vectors `W_V(x_i − x_j)` for the selected pairs, aggregate with softmax-weighted salience, and route the resulting relational vector `R_t` to slots via learned slot keys with per-slot bias (std=2.0 for strong symmetry breaking from step 0).
- **GRU-style gated carry**: Slots update via `s' = (1−g)·s + g·(w·R_t)` where `g = σ(W_gate([s; w·R_t]))` with gate bias initialized to −2 (≈0.12 open at init for slow-open behavior).
- **Strand mixer**: K×K self-attention across slots after the gated update — co-active entity tracks share context before read-back. Controlled by a learned per-dimension gate `λ_strand` initialized to 0 (no-op at init, grows as needed).
- **Directional render gate**: Per-slot, per-dimension, direction-dependent read gate. Context direction computed from W₀'s spectral projectors (exact, no eigendecomposition): `φ_dir = ‖P_φ · x̄‖ / (‖P_φ · x̄‖ + ‖P_ψ · x̄‖)` where `P_φ = (W₀ − ψI)/(φ−ψ)`. The render gate `tanh(render_coeffs[k] · [1, φ_dir, ψ_dir])` is zero-initialized so braid injection starts at zero and opens gradually.
- **Per-slot temperatures**: Learned scalar temperatures (via softplus for positivity) scale write-routing compatibility scores, allowing individual slots to specialize as sharp entity trackers or broad context integrators. Temperatures are *not* applied to read-back attention to avoid positive feedback collapse.
- **Aux losses**: `L_persist = MSE(s' − s)` penalizes violent slot churn and `L_div` is the mean squared off-diagonal cosine similarity penalty against slot collapse. The code supports both losses, but the v11b submission run uses `BRAID_PERSIST_W=0.0` and `BRAID_DIV_W=0.01`.
- **Read-back**: Sequence tokens attend over updated slots via `Q_read(x) · K_read(slots)`, with values gated through the directional render gate before the attention-weighted sum is added to the residual stream. `W_V_read` is zero-initialized so braid injection starts exactly at zero.

The slow-open design (zero-init render gate + zero-init V_read + gate bias −2 + low braid LR at 0.1×) is critical for stability. Without it, the braid overwhelms the Fibonacci backbone before it has learned useful slot representations (the v3 catastrophic divergence failure mode).

### FMNRiemannianAdam (W₀ optimizer)

W₀ lives on the Fibonacci manifold — the set of symmetric matrices with eigenvalues in `{φ, ψ}`. Standard gradient descent would walk W₀ off the manifold. FMNRiemannianAdam uses retraction-based Riemannian Adam:

1. Eigendecompose W₀ in float64 → get eigenvectors Q and φ/ψ mask
2. Project Euclidean gradient into the tangent space (cross-block entries in eigenbasis only)
3. Accumulate Adam momentum entirely in tangent space (momentum is re-projected to the current tangent space each step)
4. Compute Adam direction (bias-corrected, scalar variance) and re-project to tangent space
5. Retract back to manifold via `projx`: snap eigenvalues to {φ, ψ}, reconstruct W₀ = Q·diag(snapped)·Qᵀ. If eigendecomposition is ill-conditioned after the step, the step size is halved repeatedly (up to 14 times) until it succeeds
6. Transport momentum to the new tangent space at the retracted point

This is a *retraction*, not a geodesic (exponential map) — the step is taken in the tangent space and then projected back, rather than following the manifold's curvature exactly. In practice, the retraction is sufficient: `w0_err = ‖W₀² − W₀ − I‖_F ≈ 1.75×10⁻⁶` maintained across the full 50k run.

### Final output

```
LN(M_L + psi_proj(Ψ-stream)) → LM head / √d
```

The LM head shares weights with the token embedding matrix (`wte.weight` and `lm_head.weight` are tied). Logits are scaled by `1/√d` (d=512) for training stability.

The Ψ-stream aggregates all per-layer structural residuals with depth-dependent decay `|ψ|^k`, forming a summary of "how the model arrived at" the final state, separate from the manifold state itself.

---

## Model Configuration

```
n_layer       = 12
n_embd        = 512
mixer_kernel  = 256
braid_K       = 7
braid_d_b     = 128
braid_window  = 64
xsa_heads     = 4
xsa_start     = layer 9  (top 3 of 12)
braid_start   = layer 9  (top 3 of 12)
vocab_size    = 1024 (SentencePiece BPE)
block_size    = 1024

Total params  = 16.75M
  W₀ shared   =  0.262M  (one matrix, used at all 12 depths)
  Braid (3L)  =  0.973M
```

W₀ is a single 512×512 matrix shared across all 12 layers — 0.262M parameters reused at every depth. Relative to 12 independent depth-specific matrices, this guarantees a globally consistent eigenspace decomposition across depth and reduces the number of stored/quantized operator tensors from 12 to 1.

---

## Training

**Dataset:** FineWeb 10B tokens, SentencePiece 1024-vocab BPE  
**Hardware:** 2× RTX GPU (RTX 5080 + RTX 5060 Ti), torchrun DDP  
**Step time:** ~2246 ms/step (2-GPU)

| Hyperparameter | Value |
|---|---|
| TRAIN_BATCH_TOKENS | 131,072 |
| TRAIN_SEQ_LEN | 1024 |
| ITERATIONS | 50,000 |
| WARMUP_STEPS | 20 |
| embed_lr | 0.05 |
| matrix_lr (Muon) | 0.010 |
| braid_lr | 0.001 |
| W₀ optimizer | FMNRiemannianAdam |
| grad_clip_norm | 1.0 |

**Optimizer breakdown:**
- **W₀** — `FMNRiemannianAdam`: retraction-based Riemannian Adam, float64 eigendecomposition, hard manifold constraint
- **All other non-braid matrix parameters** — Muon (Newton-Schulz, 5 steps, momentum warmup 0.85→0.95), including mixer, PsiNet, XSA, and `psi_proj`
- **Braid matrices** — Muon at `braid_lr = 0.001` (`0.1 × matrix_lr`)
- **Token embeddings** — Adam at `embed_lr = 0.05`
- **Other non-matrix parameters** — Adam at `scalar_lr = 0.01`, with braid scalars at `braid_lr = 0.001`

---

## Results

### Validation trajectory (2-GPU, TRAIN_BATCH_TOKENS=131,072)

| Step | val_bpb | val_loss | braid_gate | strand | temp_std |
|------|---------|----------|------------|--------|----------|
| 1,000 | 1.6169 | 2.7300 | 0.010 | 0.041 | 0.040 |
| 5,000 | 1.4937 | 2.5221 | 0.070 | 0.039 | 0.378 |
| 10,000 | 1.4569 | 2.4598 | 0.136 | 0.042 | 0.862 |
| 15,000 | 1.4468 | 2.4428 | 0.190 | 0.056 | 1.282 |
| 20,000 | 1.4337 | 2.4207 | 0.223 | 0.092 | 1.531 |
| 25,000 | 1.4317 | 2.4173 | 0.251 | 0.176 | 1.675 |
| 30,000 | 1.4220 | 2.4009 | 0.274 | 0.262 | 1.848 |
| 35,000 | 1.4223 | 2.4015 | 0.288 | 0.307 | 1.987 |
| 40,000 | 1.4194 | 2.3966 | 0.300 | 0.323 | 2.094 |
| 41,000 | 1.4144 | 2.3881 | 0.302 | 0.326 | 2.123 |
| **47,000** | **1.4137** | **2.3870** | 0.312 | 0.336 | 2.270 |
| 50,000 | 1.4232 | 2.4030 | 0.316 | 0.340 | 2.334 |

**Best pre-quant val_bpb: 1.4137 at step 47,000** (no warmdown — final step regresses slightly without LR decay).  
**Submission checkpoint: step 50,000** — pre-quant 1.4232, post-quant roundtrip 1.4233 (+0.0001 penalty at the same checkpoint).  
Note: the +0.0001 penalty is measured at the submission checkpoint (step 50k), not between best-pre-quant (step 47k) and post-quant (step 50k).

Three observations from the trajectory:
1. **Braid gate climbs monotonically** — 0.010 → 0.316 over 50k steps, no collapse. The slow-open design (gate bias −2, zero-init render/V_read) is working as intended.
2. **Strand mixer activates slowly** — strand grows from 0.041 at step 1k to 0.340 at step 50k, tracking the gate with a lag. Cross-slot sharing increases as slots develop distinct specializations.
3. **temp_std keeps rising through the full run** — slot temperatures still diverging at step 50k (0.04 → 2.33), indicating the slots were still specializing at end of training. A longer run or warmdown would likely continue improving.

### Why the braid gate opens more slowly than previous runs

Previous runs (K=16 d_b=64) hit braid_gate=0.46 at step 5k. This run (K=7 d_b=128) is at 0.070 at step 5k and 0.313 at step 48k. With fewer slots but wider per-slot state, each slot is harder to saturate — slots are learning richer representations rather than quickly snapping to simple attractors. The slower gate opening corresponds to better-quality entity representations at each slot.

---

## Quantization Properties

FMN compresses to int8+zlib with near-zero BPB penalty. The hybrid architecture achieves the best quantization of any FMN variant tested:

| Model | Post-quant penalty |
|-------|--------------------|
| Typical transformer int8 | +0.01 – 0.03 BPB |
| Competition int6 QAT (best) | +0.02 – 0.08 BPB |
| Pure FMN (no braid/attn) | +0.0006 BPB |
| **FMN v11b hybrid** | **+0.0001 BPB** |

Likely contributors to the unusually small observed quantization loss:
 
1. **W₀** — eigenvalues snapped to {φ, ψ} throughout training. This keeps `W₀` highly structured; the tiny post-quant penalty is an empirical result for this checkpoint, not a theorem about every two-eigenvalue matrix.
 
2. **Braid displacement vectors** — writes are computed as `W_V(x_i − x_j)` from TopK-selected token pairs, i.e. learned projections of pairwise displacements. These are inherently near-zero-centered (differences between tokens in the same context window), which maps cleanly onto int8's symmetric range.
 
3. **XSA weight matrices** — in this run, the Q/K/V/O projections also quantized cleanly. A plausible explanation is that they operate on FMN-regularized representations, but that explanation is empirical rather than a formal guarantee.

The practical implication is simply that this checkpoint suffers unusually little degradation under the int8+zlib roundtrip.

---

## Why W₀ Sharing Works

W₀ is a single 512×512 matrix applied at all 12 depths. This is not just parameter efficiency — it creates structural properties that would not exist with 12 independent matrices:

**Depth has real algebraic meaning.** The φ-eigenspace is the same at layer 1 and layer 12. Information in the φ-eigenspace at depth 1 is in the *same eigenspace* at depth 12. A token that encodes a certain concept in M_φ at depth 1 will find that concept in the same eigenspace when W₀ is applied again at depth 11. Depth isn't an arbitrary ordering label — it's a count of Fibonacci operator applications.

**Hardware/runtime effects are empirical.** Reusing one `W₀` may help locality in practice, but cache behavior and step-time changes are hardware- and implementation-dependent measurements, not architectural guarantees.

**The stored operator is shared.** The checkpoint stores one quantized approximation of `W₀` rather than 12 independently quantized matrices. That reduces parameter redundancy, but it does **not** imply an exact 12× bound on downstream activation error.

---

## Running the Submission

```bash
# Multi-GPU (used for this submission)
ITERATIONS=50000 TRAIN_BATCH_TOKENS=131072 \
  torchrun --nproc_per_node=2 train_gpt.py 2>&1 | tee run.log

# Single GPU
ITERATIONS=50000 TRAIN_BATCH_TOKENS=65536 \
  python train_gpt.py 2>&1 | tee run.log
```

Key environment variables:

| Variable | Default | Notes |
|---|---|---|
| `TRAIN_BATCH_TOKENS` | 262,144 | Total tokens per step across all GPUs |
| `ITERATIONS` | 10,000 | Training steps |
| `WARMDOWN_ITERS` | 0 | Cosine LR warmdown steps at end |
| `NUM_LAYERS` | 12 | FMN depth |
| `MODEL_DIM` | 512 | Embedding dimension |
| `MIXER_KERNEL` | 256 | FibonacciMixer receptive field |
| `BRAID_K` | 7 | Number of entity slots per braid layer |
| `BRAID_D_B` | 128 | Per-slot state dimension |
| `MAX_WALLCLOCK_SECONDS` | 0 (disabled) | Wall-clock cutoff for timed tracks |

---

## Architecture Progression

This is the first FMN variant where braid produces a net BPB improvement over pure FMN. The history matters:

| Version | Change | val_bpb | Size |
|---------|--------|---------|------|
| v1a | Pure FMN, k=64, LR=0.006 | 1.5506 | 9.07 MB |
| v2c | k=256, LR=0.010 | 1.5421 | 9.99 MB |
| v3 | + full-model braid K=4 LR=0.010 | DIVERGED | — |
| v8 | + Riemannian Adam, sparse braid (no birth) | 1.6492 | — |
| v9 | + birth mechanism (gate now stable) | 1.5622 | 17.18 MB ✗ |
| v10 | d_b=64 to fit size limit | 1.5756 | 13.22 MB ✓ |
| v14 | FMNRootOptimizer, bigger batch | 1.5325 | 10.86 MB |
| **v11b** | **Top-3-layer XSA + braid, K=7 d_b=128** | **1.4233** (post-quant) | **14.10 MiB** |

Key insight at v11b: braid was losing to pure FMN because braid parameters competed with FMN parameters for the 16MB budget. The fix was making braid local-window only (window=64) and adding XSA to handle the global binding that braid was struggling to learn. With braid freed from global token routing, both mechanisms work better independently.
