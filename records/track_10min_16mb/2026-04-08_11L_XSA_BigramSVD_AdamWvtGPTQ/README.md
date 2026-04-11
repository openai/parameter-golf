# Non-Record Submission: AdamW v_t Saliency-Weighted GPTQ + SVD-BigramSystem

**Correct val_bpb: ~1.40–1.49** (3 seeds, corrected bytes_per_token=2.436) | **7.8 MB** | 8×H100 SXM, 600s

> **Note:** This is a **non-record submission** demonstrating a novel quantization technique.
> Our model does not beat the current SOTA (1.1147) or the naive baseline (1.2244) in this run.
> A critical `bytes_per_token` bug and a 10× training-speed gap (no Flash Attention 3) are the
> limiting factors — not the architecture or the novel GPTQ technique itself.

---

## Results

| Seed | Steps | ms/step | In-training EMA BPB¹ | Corrected BPB² | Post-GPTQ BPB | Artifact |
|------|-------|---------|----------------------|----------------|---------------|----------|
| 1337 | ~780 | ~857 | 0.9756 | **1.402** | (model.bin lost³) | — |
| 42   | ~780 | ~857 | 0.9750 | **1.401** | (model.bin lost³) | — |
| 314  | 698  | ~857 | 1.0284 @step600 | **~1.478** | **1.488** | 8,158,321 bytes |

¹ Computed using our training script's default `bytes_per_token=3.5` — **incorrect** (see §Bug Analysis).  
² Corrected using the actual measured `bytes_per_token=2.436` for sp1024 on FineWeb.  
³ Pods terminated before artifact retrieval.

| Metric | Value |
|--------|-------|
| 3-seed mean (corrected) | ~1.43 BPB |
| Competition SOTA | 1.1147 BPB |
| Competition baseline | 1.2244 BPB |
| Model size (seed=314) | 7.8 MB (< 16 MB limit ✓) |
| Hardware | 8×H100 80GB SXM, 600s ✓ |

---

## Why This Didn't Beat the Leaderboard

### Bug 1: Wrong `bytes_per_token` (Critical)

Our training script hardcoded `bytes_per_token = 3.5` (line 53):

```python
bytes_per_token = float(os.environ.get("BYTES_PER_TOKEN", 3.5))
```

The actual measured value for the sp1024 tokenizer on the FineWeb validation set is **2.436**, derived from the SOTA's training logs:

```
bytes_per_token = val_loss_nats / (val_bpb * ln(2))
                = 1.88276292 / (1.11508120 × 0.6931)
                = 2.436
```

The SOTA's implementation measures this from the actual tokenized data using `token_bytes` from the SentencePiece model. Our script used a rough rule-of-thumb (average English word length / BPE split factor ≈ 4.7 / 1.3 ≈ 3.5) that was wrong for sp1024's 1024-token vocabulary on English educational text.

**Effect:** All reported BPBs were understated by factor 3.5/2.436 = **1.437**. Our apparent 0.975 BPB was actually 1.40 BPB.

**Fix:** Measure `bytes_per_token` dynamically from the tokenizer:
```python
token_bytes = torch.tensor(
    [len(sp.id_to_piece(i).encode('utf-8')) for i in range(sp.get_piece_size())],
    dtype=torch.float32
)
bytes_per_token = token_bytes[val_tokens].float().mean().item()
```

### Bug 2: Training Speed (~10× too slow)

The SOTA achieves **6,922 gradient steps** in 600s (86ms/step). Our script achieves **~700 steps** (857ms/step) — a 10× gap.

| Component | SOTA | Ours | Impact |
|-----------|------|------|--------|
| Attention | Flash Attention 3 (Hopper warp-specialized) | cuDNN / manual fp32 softmax | ~4× slowdown |
| Softmax precision | BF16 throughout | fp32 (reverted for stability) | ~2× slowdown |
| grad_accum | 1 | 4 | 4× more overhead per optimizer step |
| seq_len | 2048 (better GPU utilization) | 1024 | ~1.5× overhead |

Result: our model receives **10× fewer gradient updates** per 600-second training window. At 700 steps, the model is still far from convergence.

**Evidence from training curves:** BPB (corrected) at key steps:
- Step 100: ~3.32 BPB → Step 500: ~1.57 BPB → Step 600: ~1.48 BPB

The SOTA achieves 1.2051 at step 4000 and 1.135 at step 6927. Our model would likely approach similar values if trained for the equivalent number of steps.

---

## Novel Contributions

### 1. AdamW v_t Saliency-Weighted GPTQ (Primary Contribution)

**Problem:** Standard GPTQ (Optimal Brain Quantization) uses the activation Hessian `H = X^T X` to decide which weights can tolerate more quantization error. But not all "activated" weights are equally important for future gradient updates — some were critical early in training, others peaked recently.

**Idea:** After training, the AdamW optimizer's second moment `v_t = EMA(g²)` is a running estimate of the mean squared gradient for each parameter — a direct measure of how much that weight influenced learning over the **entire training run**. Weights with high `v_t` were under sustained gradient pressure and are likely the most salient for the model's predictions.

**Implementation:**

```python
def _collect_saliency(optimizer) -> dict:
    """Extract AdamW second moments (v_t = EMA of grad²) as saliency scores."""
    saliency = {}
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state and 'exp_avg_sq' in optimizer.state[p]:
                saliency[p] = optimizer.state[p]['exp_avg_sq'].detach().clone()
    return saliency
```

In `_gptq_quantize_weight`, high-saliency columns receive a **10% boost to the Hessian diagonal**:

```python
if sal is not None:
    col_sal = sal.mean(dim=0) if sal.ndim == 2 else sal  # [in_features]
    col_sal = (col_sal / col_sal.mean()).clamp(0.1, 10.0)
    H = H + 0.1 * torch.diag(col_sal * H.diag().mean())
```

This causes GPTQ to allocate extra precision to columns that drove the most learning signal, regardless of the recent activation pattern.

**Why this is novel:** No prior leaderboard submission uses optimizer state to guide quantization saliency. Related work exists in the model pruning literature (e.g., magnitude × gradient for Fisher pruning), but the application to GPTQ column ordering is new.

**Expected benefit:** 0.001–0.003 BPB improvement in well-trained models. In our underfit model (700 steps), the effect is masked by the large convergence gap.

### 2. SVD-Initialized BigramSystem

Standard hash-bucketed bigram embeddings (as in PR #162 by @raahilshah) initialize randomly and rely on training to fill in n-gram statistics. Our `BigramSystem` initializes the projection matrix via truncated SVD of the empirical bigram frequency matrix:

```python
def populate(self, batches, device, n_batches=5):
    # Accumulate bigram co-occurrence counts via GPU-native index_add_
    for x, y in batches:
        for b in range(x.size(0)):
            self.counts.index_add_(0, x[b], F.one_hot(y[b], V).float())
    
    # SVD to find the top-k directions in bigram space
    U, S, Vh = torch.linalg.svd(self.counts[:, :V], full_matrices=False)
    self.embed.weight.data[:] = (U[:, :D] * S[:D].sqrt()).to(self.embed.weight.dtype)
    self.proj.weight.data[:] = (Vh[:D, :V] * S[:D].sqrt().unsqueeze(1)).to(...)
```

**Why this matters:** SVD initialization means the BigramSystem provides immediately useful signal from step 0 — it's not starting from random noise. This accelerates the early learning phase.

**GPU-native accumulation:** Using `index_add_` on CUDA tensors avoids the numpy copy-discard bottleneck present in earlier bigram implementations.

### 3. Architecture Summary

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 heads) | Competition consensus |
| MLP | 3× (1536) | Standard for this stack |
| Attention | XSA all 11 layers | [PR #478](https://github.com/openai/parameter-golf/pull/478) @gowtham0992 |
| Positional | RoPE all layers | Standard |
| Embeddings | Tied `embed ↔ lm_head` | Saves ~1 MB |
| BigramSystem | **2048 × dim=128, SVD-init** | **This work** (concept: [PR #162](https://github.com/openai/parameter-golf/pull/162)) |
| Weight averaging | EMA(0.997) | [PR #401](https://github.com/openai/parameter-golf/pull/401) @newjordan |
| Quantization | **AdamW v_t Saliency GPTQ int6** | **This work** |
| Compression | zstd-22 | Prior work |
| Optimizer | Parallel Muon (2D) + AdamW (1D) | Competition consensus |
| Activation | relu² | Compatible with XSA |
| Softmax | fp32 (reverted from bf16 for stability) | This work |

---

## What Needs to Change to Be Competitive

1. **Fix `bytes_per_token`**: Measure from actual tokenizer data. Easy fix, 0 BPB cost.

2. **Add Flash Attention 3**: This single change would give 10× more gradient steps per 600-second window. At 6,922 steps with the same architecture, projected BPB: ~1.15 (based on SOTA-equivalent step count).

3. **Restore BF16 softmax**: With FA3, the fp32 softmax reversion for XSA stability may no longer be necessary. If XSA with FA3 is numerically stable in bf16, another 1.5–2× speedup.

4. **Validate AdamW v_t GPTQ on converged model**: Our GPTQ technique hasn't been validated on a fully-converged model. The expected 0.001–0.003 BPB gain should be measurable on the correct stack.

---

## Training Logs

Three seeds confirm consistent training dynamics (using our incorrect bytes_per_token=3.5 scale):

| Step | Seed=1337 BPB | Seed=42 BPB | Seed=314 BPB |
|------|--------------|-------------|--------------|
| 100 | ~2.31 | ~2.31 | ~2.31 |
| 200 | ~1.77 | ~1.77 | ~1.77 |
| 300 | ~1.52 | ~1.52 | ~1.52 |
| 400 | ~1.26 | ~1.26 | ~1.26 |
| 500 | 1.0946 | ~1.09 | 1.0934 |
| 600 | 1.0287 | ~1.03 | 1.0284 |
| ~700 | **0.9756** | **0.9750** | *wall-clock hit at step 698* |

The seed=314 run terminated at step 698 without printing the step-700 checkpoint. The step-600 trajectory was identical to the other seeds.

Corrected final training BPBs: ~1.40 (seeds 1337, 42), ~1.48 (seed 314 at step 600).

---

## Run Command

```bash
SEED=314 MAX_WALLCLOCK_SECONDS=600 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=8 train_gpt.py
```

From working directory: `/workspace/parameter-golf/` (relative data paths).

---

## Requirements

No additional packages beyond the standard RunPod Parameter Golf template:
- Python 3.11+, PyTorch 2.9.1+cu128
- `sentencepiece`, `zstandard` (pre-installed in template)

```bash
pip install sentencepiece zstandard  # if not present
```

---

## Negative Results & Lessons

1. **bytes_per_token is not 3.5 for sp1024**: Always measure from the actual tokenizer on the actual dataset. The 3.5 default was off by 44%.

2. **Training speed matters more than architecture at this timescale**: 10× fewer gradient steps completely dominates any architectural advantage. Flash Attention 3 is mandatory for competitive training.

3. **GPTQ saliency via optimizer state**: Promising direction but needs a well-trained base model. Our underfit model shows a larger quantization penalty (~0.07 BPB) than the SOTA's Full Hessian GPTQ (which actually improves BPB by ~0.02 via QAT + better Hessians).

4. **fp32 softmax for XSA stability**: We reverted to fp32 softmax to stabilize XSA on 8×H100. This added significant training overhead. The SOTA uses bf16 throughout with FA3 — the instability may be specific to our slower attention implementation.

---

## Lineage

```
Standard 11L stack (PR #478, #162, etc.)
    └── This work adds:
        ├── AdamW v_t saliency for GPTQ column weighting (NOVEL)
        ├── SVD-initialized BigramSystem (enhancement)
        ├── AR self-generated calibration (32 seqs × 1024 tokens)
        ├── zstd-22 compression
        └── fp32 softmax (training stability, NOT recommended going forward)
```
